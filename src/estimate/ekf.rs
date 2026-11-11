//! STM-based orbit EKF with heterogeneous, analytically differentiated sensors.
//!
//! The state is GCRF position/velocity, optionally followed by constant C_d A/m.
//! Dynamics use the same propagation configuration as the batch solver.
//! Measurements use instantaneous partials, without chaining them with the STM.

use super::FilterHistory;
use crate::kernels::{
    Interferometer, KernelError, MeasurementKind, ParameterMode, PreparedBatch, PropagationConfig,
    StationGeometry, StationGeometrySource, V3, V6, az_el_partial_prepared, compute_sensitivities,
    propagate_with_stm, r_of, range_rate_partial, v_of, wrap_pi,
};
use numeris::{
    Matrix, Vector,
    estimate::{Ekf, EkfStep, EstimateError, rts_smooth},
};
use satkit::Instant;
use std::fmt;

#[derive(Debug)]
pub enum EkfError {
    Kernel(KernelError),
    Estimate(EstimateError),
}

impl fmt::Display for EkfError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Kernel(e) => e.fmt(f),
            Self::Estimate(e) => write!(f, "EKF estimation failed: {e}"),
        }
    }
}
impl std::error::Error for EkfError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Kernel(e) => Some(e),
            Self::Estimate(_) => None,
        }
    }
}
impl From<KernelError> for EkfError {
    fn from(e: KernelError) -> Self {
        Self::Kernel(e)
    }
}
impl From<EstimateError> for EkfError {
    fn from(e: EstimateError) -> Self {
        Self::Estimate(e)
    }
}
pub type EkfResult<T> = std::result::Result<T, EkfError>;
pub type SmoothedOrbit<const N: usize> = Vec<(Vector<f64, N>, Matrix<f64, N, N>)>;

/// A sensor supplies its prediction and instantaneous derivative in physical
/// units. Implement `innovation` for periodic or other non-Euclidean readings.
/// Closures returning `(prediction, partials)` implement this trait directly.
pub trait MeasurementModel<const M: usize> {
    fn evaluate(&self, state: &V6) -> crate::kernels::Result<(Vector<f64, M>, Matrix<f64, M, 6>)>;

    fn innovation(&self, observed: &Vector<f64, M>, predicted: &Vector<f64, M>) -> Vector<f64, M> {
        observed - predicted
    }
}

impl<const M: usize, F> MeasurementModel<M> for F
where
    F: Fn(&V6) -> crate::kernels::Result<(Vector<f64, M>, Matrix<f64, M, 6>)>,
{
    fn evaluate(&self, state: &V6) -> crate::kernels::Result<(Vector<f64, M>, Matrix<f64, M, 6>)> {
        self(state)
    }
}

/// Range-rate, azimuth/elevation, or combined sensor using prepared geometry.
/// `M = 1`: RR; `M = 2`: AZ, EL; `M = 3`: RR, AZ, EL.
#[derive(Debug)]
pub struct GroundSensor<'g, const M: usize> {
    geometry: StationGeometrySource<'g>,
}

impl<'g, const M: usize> GroundSensor<'g, M> {
    pub fn new(geometry: impl Into<StationGeometrySource<'g>>) -> crate::kernels::Result<Self> {
        if !(1..=3).contains(&M) {
            return Err(KernelError::InvalidInput(
                "ground sensor dimension must be 1, 2 or 3",
            ));
        }
        Ok(Self {
            geometry: geometry.into(),
        })
    }
    pub fn geometry(&self) -> &StationGeometry {
        &self.geometry
    }
}

impl<const M: usize> MeasurementModel<M> for GroundSensor<'_, M> {
    fn evaluate(&self, state: &V6) -> crate::kernels::Result<(Vector<f64, M>, Matrix<f64, M, 6>)> {
        let g = &self.geometry;
        let mut predicted = Vector::zeros();
        let mut h = Matrix::zeros();
        if M == 1 || M == 3 {
            let (rr, partial) =
                range_rate_partial(&r_of(state), &v_of(state), g.position(), g.velocity())?;
            predicted[0] = rr;
            h.set_block(0, 0, &partial);
        }
        if M == 2 || M == 3 {
            let (angles, partial) =
                az_el_partial_prepared(&r_of(state), g.position(), g.rotation())?;
            let offset = usize::from(M == 3);
            predicted[offset] = angles[0];
            predicted[offset + 1] = angles[1];
            h.set_block(offset, 0, &partial);
        }
        Ok((predicted, h))
    }

    fn innovation(&self, observed: &Vector<f64, M>, predicted: &Vector<f64, M>) -> Vector<f64, M> {
        let mut residual = observed - predicted;
        if M >= 2 {
            let az = usize::from(M == 3);
            residual[az] = wrap_pi(residual[az]);
        }
        residual
    }
}

/// Calibrated continuous phase sensor with a known nominal carrier/baseline.
#[derive(Debug)]
pub struct PhaseSensor<'g> {
    pub geometry: StationGeometrySource<'g>,
    pub interferometer: Interferometer,
    pub baseline_enu_m: V3,
}
impl MeasurementModel<1> for PhaseSensor<'_> {
    fn evaluate(&self, state: &V6) -> crate::kernels::Result<(Vector<f64, 1>, Matrix<f64, 1, 6>)> {
        let pred =
            self.interferometer
                .evaluate(state, &self.geometry, &self.baseline_enu_m, 0.0)?;
        let mut h = Matrix::zeros();
        for k in 0..6 {
            h[(0, k)] = pred.partial[(1, k)];
        }
        Ok((Vector::from_array([pred.phase_rad]), h))
    }
}

/// A sensor reading with its exact epoch and full noise covariance R.
/// Correlated components are updated together; independent sensors may use
/// different dimensions and covariances at the same epoch.
pub struct Observation<const M: usize, Model> {
    pub time: Instant,
    pub value: Vector<f64, M>,
    pub covariance: Matrix<f64, M, M>,
    pub model: Model,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct UpdateOptions {
    /// NIS threshold (chi-squared with M degrees of freedom). None disables gating.
    pub gate: Option<f64>,
}

#[derive(Clone, Copy, Debug)]
pub struct UpdateResult<const M: usize> {
    pub innovation: Vector<f64, M>,
    pub nis: f64,
    pub accepted: bool,
}

/// One orbit estimate accepting any measurement dimension at each update.
/// All Kalman prediction/update arithmetic is delegated to `numeris::estimate::Ekf`.
/// Six-state and seven-state filters share the batch solver's `ParameterMode`.
pub struct OrbitEkf<const N: usize> {
    config: PropagationConfig,
    epoch: Instant,
    filter: Ekf<f64, N, 1>,
    timestamps: Vec<Instant>,
    steps: FilterHistory<N>,
}

impl<const N: usize> OrbitEkf<N> {
    pub fn new(
        config: PropagationConfig,
        initial: Vector<f64, N>,
        covariance: Matrix<f64, N, N>,
    ) -> EkfResult<Self> {
        if N != config.mode().columns() {
            return Err(KernelError::InvalidInput(
                "EKF state dimension does not match parameter mode",
            )
            .into());
        }
        check_parameters(&initial)?;
        check_covariance(&covariance, true)?;
        let epoch = config.reference_epoch();
        Ok(Self {
            config,
            epoch,
            filter: Ekf::new(initial, covariance),
            timestamps: vec![epoch],
            steps: FilterHistory::new(record(initial, covariance, Matrix::eye())),
        })
    }

    pub fn state(&self) -> &Vector<f64, N> {
        self.filter.state()
    }
    pub fn covariance(&self) -> &Matrix<f64, N, N> {
        self.filter.covariance()
    }
    pub fn epoch(&self) -> Instant {
        self.epoch
    }
    pub fn config(&self) -> &PropagationConfig {
        &self.config
    }

    /// One record per distinct prediction epoch, including the initial epoch.
    /// Multiple updates at one epoch revise that record's updated state/covariance.
    pub fn steps(&self) -> &FilterHistory<N> {
        &self.steps
    }
    pub fn timestamps(&self) -> &[Instant] {
        &self.timestamps
    }

    /// Directly invokes numeris's built-in RTS smoother. Results align with
    /// `timestamps()`, including the initial prior epoch and prediction-only epochs.
    pub fn smooth(&self) -> EkfResult<SmoothedOrbit<N>> {
        Ok(rts_smooth(&self.steps.to_records())?)
    }

    /// Predict forward with the local STM, relinearized at the latest estimate.
    /// Q is the already-discretized process covariance in endpoint GCRF units.
    /// Same-epoch predictions are identity operations and require zero Q.
    /// Failure leaves the filter and its history unchanged.
    pub fn predict_to(&mut self, time: Instant, q: Option<&Matrix<f64, N, N>>) -> EkfResult<()> {
        if time < self.epoch {
            return Err(KernelError::InvalidInput("EKF epochs must be chronological").into());
        }
        if let Some(q) = q {
            check_covariance(q, false)?;
        }
        if time == self.epoch {
            if q.is_some_and(|q| q.as_slice().iter().any(|v| *v != 0.0)) {
                return Err(KernelError::InvalidInput(
                    "same-epoch prediction requires zero process noise",
                )
                .into());
            }
            return Ok(());
        }
        let state = V6::from_slice(&self.filter.x.as_slice()[..6]);
        let (propagated, transition) = match self.config.mode() {
            ParameterMode::StateOnly { fixed_drag } => {
                let (x, phi) = propagate_with_stm(
                    &state,
                    &self.epoch,
                    &time,
                    self.config.settings(),
                    fixed_drag,
                )?;
                let mut f = Matrix::eye();
                f.set_block(0, 0, &phi);
                (x, f)
            }
            ParameterMode::StateAndDrag { delta_cd } => {
                let sensitivities = compute_sensitivities(
                    &state,
                    &self.epoch,
                    &[time],
                    self.config.settings(),
                    self.filter.x[6],
                    delta_cd,
                )?;
                let mut f = Matrix::eye();
                f.set_block(0, 0, &sensitivities.augmented_transition(0)?);
                (sensitivities.states[0], f)
            }
        };
        let mut predicted = self.filter.x;
        predicted.set_block(0, 0, &propagated);
        let mut candidate = Ekf::<f64, N, 1>::new(self.filter.x, self.filter.p);
        candidate.predict(|_| predicted, |_| transition, q);
        check_parameters(&candidate.x)?;
        check_covariance(&candidate.p, true)?;
        self.steps
            .push(record(candidate.x, candidate.p, transition));
        self.timestamps.push(time);
        self.filter = candidate;
        self.epoch = time;
        Ok(())
    }

    /// Update at the current epoch using a sensor's supplied partials. A temporary
    /// numeris EKF gives each sensor its own const-generic measurement dimension.
    /// The augmented drag column is zero: geometric readings depend on drag
    /// through propagation/cross-covariance, rather than directly.
    /// Errors and rejected readings leave state/covariance and history unchanged.
    pub fn update<const M: usize>(
        &mut self,
        observed: &Vector<f64, M>,
        model: &impl MeasurementModel<M>,
        r: &Matrix<f64, M, M>,
        options: UpdateOptions,
    ) -> EkfResult<UpdateResult<M>> {
        if M == 0 || !is_finite(observed) || options.gate.is_some_and(|g| !g.is_finite() || g < 0.0)
        {
            return Err(
                KernelError::InvalidInput("measurement dimensions, values or NIS gate").into(),
            );
        }
        check_covariance(r, true)?;
        let state = V6::from_slice(&self.filter.x.as_slice()[..6]);
        let (predicted, partial) = model.evaluate(&state)?;
        let innovation = model.innovation(observed, &predicted);
        if !is_finite(&predicted) || !is_finite(&partial) || !is_finite(&innovation) {
            return Err(KernelError::InvalidInput(
                "measurement prediction, partials or innovation must be finite",
            )
            .into());
        }
        let mut h = Matrix::<f64, M, N>::zeros();
        h.set_block(0, 0, &partial);
        let mut candidate = Ekf::<f64, N, M>::new(self.filter.x, self.filter.p);
        // Feeding a zero prediction and the already computed innovation preserves
        // angular/custom residuals without subtracting large predictions twice.
        let nis = candidate.update(&innovation, |_| Vector::zeros(), |_| h, r)?;
        if !nis.is_finite() {
            return Err(KernelError::Numerical("innovation exceeds numeric range").into());
        }
        let accepted = options.gate.is_none_or(|gate| nis <= gate);
        if accepted {
            check_parameters(&candidate.x)?;
            check_covariance(&candidate.p, true)?;
            self.filter.x = candidate.x;
            self.filter.p = candidate.p;
            let last = self.steps.current_mut();
            last.x_updated = candidate.x;
            last.p_updated = candidate.p;
        }
        Ok(UpdateResult {
            innovation,
            nis,
            accepted,
        })
    }

    /// Predict and update for one reading. On error, rolls back both operations.
    /// A gated outlier retains the prediction and records a prediction-only step.
    pub fn observe<const M: usize>(
        &mut self,
        observation: &Observation<M, impl MeasurementModel<M>>,
        q: Option<&Matrix<f64, N, N>>,
        options: UpdateOptions,
    ) -> EkfResult<UpdateResult<M>> {
        let saved = (
            self.filter.x,
            self.filter.p,
            self.epoch,
            self.steps.checkpoint(),
        );
        self.predict_to(observation.time, q)?;
        match self.update(
            &observation.value,
            &observation.model,
            &observation.covariance,
            options,
        ) {
            Ok(result) => Ok(result),
            Err(error) => {
                self.filter.x = saved.0;
                self.filter.p = saved.1;
                self.epoch = saved.2;
                self.steps.restore(saved.3);
                self.timestamps.truncate(self.steps.len());
                Err(error)
            }
        }
    }

    /// Feed the current batch solver's observations into this filter, sorted by
    /// exact epoch. Ties retain supplied batch/row order. Q is evaluated once per
    /// new epoch from the latest updated state; same-epoch sensors add no Q.
    /// Successful earlier readings remain committed if a later reading fails.
    pub fn filter_batches(
        &mut self,
        batches: &[PreparedBatch],
        mut process_noise: impl FnMut(
            &Vector<f64, N>,
            Instant,
            Instant,
        ) -> crate::kernels::Result<Matrix<f64, N, N>>,
        options: UpdateOptions,
    ) -> EkfResult<Vec<BatchUpdate>> {
        let mut rows: Vec<_> = batches
            .iter()
            .enumerate()
            .flat_map(|(b, batch)| {
                batch
                    .timestamps()
                    .iter()
                    .enumerate()
                    .map(move |(row, &time)| (time, b, row))
            })
            .collect();
        rows.sort_by_key(|&(time, _, _)| time);
        if rows.first().is_some_and(|r| r.0 < self.epoch) {
            return Err(KernelError::InvalidInput("batch readings precede EKF epoch").into());
        }
        let mut results = Vec::with_capacity(rows.len());
        for (time, batch_index, row) in rows {
            let batch = &batches[batch_index];
            let q = if time == self.epoch {
                None
            } else {
                Some(process_noise(self.state(), self.epoch, time)?)
            };
            let (nis, accepted) = self.observe_prepared_batch(batch, row, q.as_ref(), options)?;
            results.push(BatchUpdate {
                time,
                batch: batch_index,
                row,
                nis,
                accepted,
            });
        }
        Ok(results)
    }

    fn observe_prepared_batch(
        &mut self,
        batch: &PreparedBatch,
        row: usize,
        q: Option<&Matrix<f64, N, N>>,
        options: UpdateOptions,
    ) -> EkfResult<(f64, bool)> {
        Ok(match batch.kind() {
            MeasurementKind::Joint => self.observe_joint_dimension(batch, row, q, options)?,
            MeasurementKind::PhaseDifference => self.observe_phase_batch(batch, row, q, options)?,
            MeasurementKind::RangeRate => self.observe_batch::<1>(batch, row, q, options)?,
            MeasurementKind::AzimuthElevation => self.observe_batch::<2>(batch, row, q, options)?,
            MeasurementKind::Combined => self.observe_batch::<3>(batch, row, q, options)?,
        })
    }
    fn observe_phase_batch(
        &mut self,
        batch: &PreparedBatch,
        row: usize,
        q: Option<&Matrix<f64, N, N>>,
        options: UpdateOptions,
    ) -> EkfResult<(f64, bool)> {
        let time = batch.timestamps()[row];
        let (column, sigma) = batch
            .observation_column(0)
            .ok_or(KernelError::InvalidInput("phase observation column"))?;
        let observation = Observation {
            time,
            value: Vector::from_array([column[row]]),
            covariance: Matrix::new([[sigma * sigma]]),
            model: PhaseSensor {
                baseline_enu_m: batch
                    .baselines_enu_m()
                    .ok_or(KernelError::InvalidInput("phase baselines"))?[row],
                geometry: batch
                    .geometry()
                    .row(row)
                    .ok_or(KernelError::InvalidInput("batch geometry row"))?
                    .into(),
                interferometer: batch
                    .interferometer()
                    .ok_or(KernelError::InvalidInput("phase interferometer"))?,
            },
        };
        let result = self.observe(&observation, q, options)?;
        Ok((result.nis, result.accepted))
    }
    fn observe_joint_dimension(
        &mut self,
        batch: &PreparedBatch,
        row: usize,
        q: Option<&Matrix<f64, N, N>>,
        options: UpdateOptions,
    ) -> EkfResult<(f64, bool)> {
        Ok(match batch.components() {
            1 => self.observe_joint_batch::<1>(batch, row, q, options)?,
            2 => self.observe_joint_batch::<2>(batch, row, q, options)?,
            3 => self.observe_joint_batch::<3>(batch, row, q, options)?,
            4 => self.observe_joint_batch::<4>(batch, row, q, options)?,
            5 => self.observe_joint_batch::<5>(batch, row, q, options)?,
            _ => return Err(KernelError::InvalidInput("joint sensor dimension").into()),
        })
    }
    fn observe_joint_batch<const M: usize>(
        &mut self,
        batch: &PreparedBatch,
        row: usize,
        q: Option<&Matrix<f64, N, N>>,
        options: UpdateOptions,
    ) -> EkfResult<(f64, bool)> {
        let observation = batch.joint_observation::<M>(row)?;
        let result = self.observe(&observation, q, options)?;
        Ok((result.nis, result.accepted))
    }

    fn observe_batch<const M: usize>(
        &mut self,
        batch: &PreparedBatch,
        row: usize,
        q: Option<&Matrix<f64, N, N>>,
        options: UpdateOptions,
    ) -> EkfResult<(f64, bool)> {
        let mut value = Vector::zeros();
        let mut covariance = Matrix::zeros();
        for i in 0..M {
            let (column, sigma) = batch
                .observation_column(i)
                .ok_or(KernelError::InvalidInput("batch sensor dimension"))?;
            value[i] = column[row];
            covariance[(i, i)] = sigma * sigma;
        }
        let observation = Observation {
            time: batch.timestamps()[row],
            value,
            covariance,
            model: GroundSensor::<M>::new(
                batch
                    .geometry()
                    .row(row)
                    .ok_or(KernelError::InvalidInput("batch geometry row"))?,
            )?,
        };
        let result = self.observe(&observation, q, options)?;
        Ok((result.nis, result.accepted))
    }
}

#[derive(Clone, Copy, Debug)]
pub struct BatchUpdate {
    pub time: Instant,
    pub batch: usize,
    pub row: usize,
    pub nis: f64,
    pub accepted: bool,
}

/// Discretized continuous white acceleration noise in GCRF, with optional
/// independent drag random walk. Acceleration spectral densities are in m²/s³;
/// drag spectral density is (m²/kg)²/s. The six-state form requires zero drag PSD.
pub fn white_acceleration_noise<const N: usize>(
    seconds: f64,
    acceleration_psd: &V3,
    drag_psd: f64,
) -> crate::kernels::Result<Matrix<f64, N, N>> {
    check_noise_inputs::<N>(seconds, acceleration_psd, drag_psd)?;
    let mut q = Matrix::zeros();
    for i in 0..3 {
        let velocity_variance = acceleration_psd[i] * seconds;
        let cross = velocity_variance * seconds / 2.0;
        q[(i, i)] = velocity_variance * seconds * seconds / 3.0;
        q[(i, i + 3)] = cross;
        q[(i + 3, i)] = cross;
        q[(i + 3, i + 3)] = velocity_variance;
    }
    if N == 7 {
        q[(6, 6)] = drag_psd * seconds;
    }
    if !is_finite(&q) {
        return Err(KernelError::Numerical(
            "process noise exceeds numeric range",
        ));
    }
    Ok(q)
}

fn check_noise_inputs<const N: usize>(
    seconds: f64,
    acceleration_psd: &V3,
    drag_psd: f64,
) -> crate::kernels::Result<()> {
    if !matches!(N, 6 | 7)
        || !seconds.is_finite()
        || seconds < 0.0
        || !is_finite(acceleration_psd)
        || acceleration_psd.as_slice().iter().any(|v| *v < 0.0)
        || !drag_psd.is_finite()
        || drag_psd < 0.0
        || (N == 6 && drag_psd != 0.0)
    {
        return Err(KernelError::InvalidInput(
            "process noise dimensions, duration or spectral densities",
        ));
    }
    Ok(())
}

fn record<const N: usize>(
    x: Vector<f64, N>,
    p: Matrix<f64, N, N>,
    f: Matrix<f64, N, N>,
) -> EkfStep<f64, N> {
    EkfStep {
        x_predicted: x,
        p_predicted: p,
        x_updated: x,
        p_updated: p,
        f_jacobian: f,
    }
}

fn is_finite<const R: usize, const C: usize>(matrix: &Matrix<f64, R, C>) -> bool {
    matrix.as_slice().iter().all(|x| x.is_finite())
}

fn check_parameters<const N: usize>(state: &Vector<f64, N>) -> crate::kernels::Result<()> {
    if N < 6 || !is_finite(state) || (N == 7 && state[6] < 0.0) {
        return Err(KernelError::InvalidInput(
            "EKF state must be finite with nonnegative drag",
        ));
    }
    let radius = V3::from_slice(&state.as_slice()[..3]).norm();
    if !radius.is_finite() || radius == 0.0 {
        return Err(KernelError::InvalidInput(
            "EKF state must have nonzero geocentric radius",
        ));
    }
    Ok(())
}

// Scale to correlation units before checking positive definiteness so mixed
// position/velocity/drag variances do not depend on an absolute pivot tolerance.
pub(super) fn check_covariance<const N: usize>(
    p: &Matrix<f64, N, N>,
    positive: bool,
) -> crate::kernels::Result<()> {
    if N == 0 || !is_finite(p) {
        return Err(invalid_covariance());
    }
    let mut scaled = scale_covariance(p, positive)?;
    if !is_finite(&scaled) {
        return Err(invalid_covariance());
    }
    if positive {
        scaled.cholesky().map_err(|_| invalid_covariance())?;
    } else {
        check_semidefinite(&mut scaled)?;
    }
    Ok(())
}
fn invalid_covariance() -> KernelError {
    KernelError::InvalidInput("covariance must be finite, symmetric and positive (semi)definite")
}
fn scale_covariance<const N: usize>(
    p: &Matrix<f64, N, N>,
    positive: bool,
) -> crate::kernels::Result<Matrix<f64, N, N>> {
    let mut scaled = Matrix::zeros();
    for i in 0..N {
        if p[(i, i)] < 0.0 || (positive && p[(i, i)] == 0.0) {
            return Err(invalid_covariance());
        }
        for j in 0..N {
            scaled[(i, j)] = scale_covariance_entry(p, i, j)?;
        }
    }
    Ok(scaled)
}
fn scale_covariance_entry<const N: usize>(
    p: &Matrix<f64, N, N>,
    i: usize,
    j: usize,
) -> crate::kernels::Result<f64> {
    if (p[(i, j)] - p[(j, i)]).abs() > 1e-12 * p[(i, j)].abs().max(p[(j, i)].abs()) {
        return Err(invalid_covariance());
    }
    if p[(i, i)] == 0.0 || p[(j, j)] == 0.0 {
        if p[(i, j)] != 0.0 {
            return Err(invalid_covariance());
        }
        return Ok(0.0);
    }
    Ok(p[(i, j)] / p[(i, i)].sqrt() / p[(j, j)].sqrt())
}
fn check_semidefinite<const N: usize>(
    scaled: &mut Matrix<f64, N, N>,
) -> crate::kernels::Result<()> {
    // Schur complements admit zero pivots for rank-deficient process noise.
    for k in 0..N {
        let pivot = scaled[(k, k)];
        if pivot < -1e-12 {
            return Err(invalid_covariance());
        }
        if pivot <= 0.0 {
            if (k + 1..N).any(|i| scaled[(i, k)].abs() > 1e-12) {
                return Err(invalid_covariance());
            }
        } else {
            for i in k + 1..N {
                for j in k + 1..N {
                    scaled[(i, j)] -= scaled[(i, k)] * scaled[(k, j)] / pivot;
                }
            }
        }
    }
    Ok(())
}
