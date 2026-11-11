//! Calibrated passive measurements and convenient batch fitting.
//!
//! Values are extracted observables, not receiver IQ samples. Joint observations
//! use Doppler Hz, phase-difference rad, differential Doppler Hz, azimuth rad,
//! elevation rad, in that order, omitting absent components. Batch residuals
//! remain grouped by component, as in the original prepared-batch API.

use crate::estimate::{MeasurementModel, Observation, OrbitEkf};
use crate::kernels::*;
use crate::optim::{
    CovarianceFit, CovarianceFitError, FitError, FitResult, ODProblem, fit_lm,
    fit_lm_with_covariance,
};
use numeris::{DynMatrix, DynVector, Matrix, Vector, dynmatrix::DynCholesky, optim::LmSettings};
use satkit::Instant;

mod rf;
pub use rf::{GaussianPrior, RfBuilder, RfPriors};

/// Six-component GCRF orbit filter with fixed drag.
pub type OrbitFilter = OrbitEkf<6>;
/// GCRF orbit filter also estimating C_d A/m.
pub type OrbitFilterWithDrag = OrbitEkf<7>;

/// A calibrated measurement model. Offsets are known constants in physical units.
/// A joint model may contain each variant once, with independent RF baselines.
/// Baselines are antenna 1 minus antenna 2 in antenna 1's ENU frame, in metres.
/// `B` is `V3` for sensors and `Vec<V3>` in measurement order for batches.
#[derive(Copy, Debug)]
pub enum Observable<B = V3> {
    Doppler {
        carrier_hz: f64,
        offset_hz: f64,
    },
    PhaseDifference {
        interferometer: Interferometer,
        baseline_enu_m: B,
        frequency_offset_hz: f64,
        phase_offset_rad: f64,
    },
    DifferentialDoppler {
        interferometer: Interferometer,
        baseline_enu_m: B,
    },
    AzimuthElevation,
}

impl<B: Copy> Clone for Observable<B> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<B> Observable<B> {
    /// Transform the RF baseline input, for example from a sensor vector to
    /// repeated per-measurement vectors using `|b| vec![b; measurements]`.
    pub fn map_baseline<C>(self, map: impl FnOnce(B) -> C) -> Observable<C> {
        match self {
            Self::Doppler {
                carrier_hz,
                offset_hz,
            } => Observable::Doppler {
                carrier_hz,
                offset_hz,
            },
            Self::PhaseDifference {
                interferometer,
                baseline_enu_m,
                frequency_offset_hz,
                phase_offset_rad,
            } => Observable::PhaseDifference {
                interferometer,
                baseline_enu_m: map(baseline_enu_m),
                frequency_offset_hz,
                phase_offset_rad,
            },
            Self::DifferentialDoppler {
                interferometer,
                baseline_enu_m,
            } => Observable::DifferentialDoppler {
                interferometer,
                baseline_enu_m: map(baseline_enu_m),
            },
            Self::AzimuthElevation => Observable::AzimuthElevation,
        }
    }
    fn order(&self) -> usize {
        match self {
            Self::Doppler { .. } => 0,
            Self::PhaseDifference { .. } => 1,
            Self::DifferentialDoppler { .. } => 2,
            Self::AzimuthElevation => 3,
        }
    }
    fn validate(&self) -> Result<()> {
        match self {
            Self::Doppler {
                carrier_hz,
                offset_hz,
            } if !carrier_hz.is_finite() || *carrier_hz <= 0.0 || !offset_hz.is_finite() => {
                Err(KernelError::InvalidInput("Doppler carrier/offset"))
            }
            Self::PhaseDifference {
                frequency_offset_hz,
                phase_offset_rad,
                ..
            } if !frequency_offset_hz.is_finite() || !phase_offset_rad.is_finite() => {
                Err(KernelError::InvalidInput("phase calibration offsets"))
            }
            _ => Ok(()),
        }
    }
    fn dimension(&self) -> usize {
        if matches!(self, Self::AzimuthElevation) {
            2
        } else {
            1
        }
    }
}

#[derive(Debug)]
pub(crate) struct Model<B>(Box<[Observable<B>]>);

pub(crate) trait Baseline {
    fn validate(&self, measurements: usize) -> Result<()>;
    fn at(&self, row: usize) -> &V3;
}
impl Baseline for V3 {
    fn validate(&self, _: usize) -> Result<()> {
        validate_baseline(self)
    }
    fn at(&self, _: usize) -> &V3 {
        self
    }
}
impl Baseline for Vec<V3> {
    fn validate(&self, measurements: usize) -> Result<()> {
        validate_baselines(self, measurements)
    }
    fn at(&self, row: usize) -> &V3 {
        &self[row]
    }
}

impl<B: Baseline> Model<B> {
    fn new(mut observables: Vec<Observable<B>>, measurements: usize) -> Result<Self> {
        if observables.is_empty() {
            return Err(KernelError::InvalidInput("empty observable selection"));
        }
        observables.sort_by_key(|o| o.order());
        if observables.windows(2).any(|w| w[0].order() == w[1].order()) {
            return Err(KernelError::InvalidInput("duplicate observable selection"));
        }
        for observable in &observables {
            observable.validate()?;
            match observable {
                Observable::PhaseDifference { baseline_enu_m, .. }
                | Observable::DifferentialDoppler { baseline_enu_m, .. } => {
                    baseline_enu_m.validate(measurements)?
                }
                _ => {}
            }
        }
        Ok(Self(observables.into()))
    }
    fn dimension(&self) -> usize {
        self.0.iter().map(|o| o.dimension()).sum()
    }
    fn azimuth_column(&self) -> Option<usize> {
        self.0
            .last()
            .filter(|o| matches!(o, Observable::AzimuthElevation))
            .map(|_| self.dimension() - 2)
    }
    fn predict(
        &self,
        state: &V6,
        geometry: &StationGeometry,
        time: Instant,
        measurement: usize,
    ) -> Result<(Vector<f64, 5>, Matrix<f64, 5, 6>)> {
        let mut value = Vector::zeros();
        let mut partial = Matrix::zeros();
        let mut row = 0;
        for observable in self.0.iter() {
            match observable {
                Observable::Doppler {
                    carrier_hz,
                    offset_hz,
                } => {
                    let (rr, h) = range_rate_partial(
                        &r_of(state),
                        &v_of(state),
                        geometry.position(),
                        geometry.velocity(),
                    )?;
                    let scale = -carrier_hz / SPEED_OF_LIGHT;
                    value[row] = scale * rr + *offset_hz;
                    partial.set_block(row, 0, &(h * scale));
                }
                Observable::PhaseDifference {
                    interferometer,
                    baseline_enu_m,
                    frequency_offset_hz,
                    phase_offset_rad,
                } => {
                    let pred = interferometer.evaluate(
                        state,
                        geometry,
                        baseline_enu_m.at(measurement),
                        *frequency_offset_hz,
                    )?;
                    value[row] = pred.phase_rad + phase_offset_rad;
                    partial.set_block(row, 0, &pred.partial.block::<1, 6>(1, 0));
                }
                Observable::DifferentialDoppler {
                    interferometer,
                    baseline_enu_m,
                } => {
                    let pred = interferometer.differential(
                        state,
                        geometry,
                        baseline_enu_m.at(measurement),
                        time,
                    )?;
                    value[row] = pred.difference_hz;
                    partial.set_block(row, 0, &pred.partial);
                }
                Observable::AzimuthElevation => {
                    let (angles, h) = az_el_partial_prepared(
                        &r_of(state),
                        geometry.position(),
                        geometry.rotation(),
                    )?;
                    value[row] = angles[0];
                    value[row + 1] = angles[1];
                    partial.set_block(row, 0, &h);
                }
            }
            row += observable.dimension();
        }
        if !value
            .as_slice()
            .iter()
            .chain(partial.as_slice())
            .all(|v| v.is_finite())
        {
            return Err(KernelError::Numerical(
                "measurement prediction exceeds numeric range",
            ));
        }
        Ok((value, partial))
    }
}

#[derive(Debug)]
enum ModelSource<'g> {
    Owned(Model<V3>),
    Borrowed(&'g Model<Vec<V3>>, usize),
}
impl ModelSource<'_> {
    fn predict(
        &self,
        state: &V6,
        geometry: &StationGeometry,
        time: Instant,
    ) -> Result<(Vector<f64, 5>, Matrix<f64, 5, 6>)> {
        match self {
            Self::Owned(model) => model.predict(state, geometry, time, 0),
            Self::Borrowed(model, row) => model.predict(state, geometry, time, *row),
        }
    }
    fn azimuth_column(&self) -> Option<usize> {
        match self {
            Self::Owned(model) => model.azimuth_column(),
            Self::Borrowed(model, _) => model.azimuth_column(),
        }
    }
}

/// Instantaneous calibrated sensor, suitable for `OrbitEkf::observe` or `update`.
/// Use the named sensor constructors for one observable, or `joint_sensor` for fusion.
#[derive(Debug)]
pub struct CalibratedSensor<'g, const M: usize> {
    model: ModelSource<'g>,
    geometry: StationGeometrySource<'g>,
    time: Instant,
}

impl<const M: usize> MeasurementModel<M> for CalibratedSensor<'_, M> {
    fn evaluate(&self, state: &V6) -> Result<(Vector<f64, M>, Matrix<f64, M, 6>)> {
        let (value, partial) = self.model.predict(state, &self.geometry, self.time)?;
        let mut v = Vector::zeros();
        let mut h = Matrix::zeros();
        for i in 0..M {
            v[i] = value[i];
            for k in 0..6 {
                h[(i, k)] = partial[(i, k)];
            }
        }
        Ok((v, h))
    }
    fn innovation(&self, observed: &Vector<f64, M>, predicted: &Vector<f64, M>) -> Vector<f64, M> {
        let mut r = observed - predicted;
        if let Some(i) = self.model.azimuth_column() {
            r[i] = wrap_pi(r[i]);
        }
        r
    }
}

/// Prepare a joint sensor. `M` must equal the selected component count (1–5).
/// Geometry must have been prepared for `time`.
pub fn joint_sensor<'g, const M: usize>(
    geometry: impl Into<StationGeometrySource<'g>>,
    time: Instant,
    observables: Vec<Observable>,
) -> Result<CalibratedSensor<'g, M>> {
    let model = Model::new(observables, 1)?;
    if M != model.dimension() {
        return Err(KernelError::InvalidInput(
            "sensor dimension does not match observables",
        ));
    }
    Ok(CalibratedSensor {
        model: ModelSource::Owned(model),
        geometry: geometry.into(),
        time,
    })
}

pub fn doppler_sensor<'g>(
    geometry: impl Into<StationGeometrySource<'g>>,
    time: Instant,
    carrier_hz: f64,
) -> Result<CalibratedSensor<'g, 1>> {
    joint_sensor(
        geometry,
        time,
        vec![Observable::Doppler {
            carrier_hz,
            offset_hz: 0.0,
        }],
    )
}
pub fn phase_difference_sensor<'g>(
    geometry: impl Into<StationGeometrySource<'g>>,
    time: Instant,
    interferometer: Interferometer,
    baseline_enu_m: V3,
) -> Result<CalibratedSensor<'g, 1>> {
    joint_sensor(
        geometry,
        time,
        vec![Observable::PhaseDifference {
            interferometer,
            baseline_enu_m,
            frequency_offset_hz: 0.0,
            phase_offset_rad: 0.0,
        }],
    )
}
pub fn differential_doppler_sensor<'g>(
    geometry: impl Into<StationGeometrySource<'g>>,
    time: Instant,
    interferometer: Interferometer,
    baseline_enu_m: V3,
) -> Result<CalibratedSensor<'g, 1>> {
    joint_sensor(
        geometry,
        time,
        vec![Observable::DifferentialDoppler {
            interferometer,
            baseline_enu_m,
        }],
    )
}
pub fn azimuth_elevation_sensor<'g>(
    geometry: impl Into<StationGeometrySource<'g>>,
    time: Instant,
) -> Result<CalibratedSensor<'g, 2>> {
    joint_sensor(geometry, time, vec![Observable::AzimuthElevation])
}

/// Prepare measurements sharing a schedule and a full covariance per epoch.
/// Each values row and covariance uses canonical component order, regardless of
/// the order of `observables`. Covariances must be symmetric positive definite.
/// Different schedules should be supplied as separate batches to `fit`/`filter_batches`.
pub fn joint<'g>(
    geometry: impl Into<GeometrySource<'g>>,
    observables: Vec<Observable<Vec<V3>>>,
    values: Vec<Vec<f64>>,
    covariances: Vec<DynMatrix<f64>>,
) -> Result<PreparedBatch<'g>> {
    let geometry = geometry.into();
    let model = Model::new(observables, geometry.len())?;
    let m = model.dimension();
    if geometry.is_empty()
        || values.len() != geometry.len()
        || covariances.len() != geometry.len()
        || values
            .iter()
            .any(|r| r.len() != m || !r.iter().all(|v| v.is_finite()))
    {
        return Err(KernelError::InvalidInput(
            "joint observation dimensions/values",
        ));
    }
    geometry
        .len()
        .checked_mul(m)
        .ok_or(KernelError::InvalidInput("output dimensions overflow"))?;
    let epochs = covariances
        .into_iter()
        .map(|covariance| {
            let factor = factor_covariance(&covariance, m)?;
            Ok(NoiseAtEpoch { covariance, factor })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(PreparedBatch::from_joint(
        geometry,
        JointBatch {
            model,
            values,
            noise: Noise::PerEpoch(epochs),
        },
    ))
}

fn factor_covariance(covariance: &DynMatrix<f64>, m: usize) -> Result<DynCholesky<f64>> {
    if covariance.nrows() != m
        || covariance.ncols() != m
        || !covariance.as_slice().iter().all(|v| v.is_finite())
    {
        return Err(KernelError::InvalidInput(
            "joint covariance dimensions/values",
        ));
    }
    for i in 0..m {
        for j in 0..i {
            let scale = covariance[(i, i)].abs().sqrt() * covariance[(j, j)].abs().sqrt();
            if (covariance[(i, j)] - covariance[(j, i)]).abs() > 1e-12 * scale {
                return Err(KernelError::InvalidInput(
                    "joint covariance must be symmetric",
                ));
            }
        }
    }
    covariance
        .cholesky()
        .map_err(|_| KernelError::InvalidInput("joint covariance must be positive definite"))
}

fn valid_noise_sigma(sigma: f64) -> bool {
    sigma.is_finite() && sigma > 0.0 && sigma.powi(2).is_finite() && sigma.powi(2) != 0.0
}

fn validate_independent_columns(
    columns: &[Vec<f64>],
    sigmas: &[f64],
    measurements: usize,
) -> Result<()> {
    if columns.iter().any(|c| c.len() != measurements)
        || sigmas.len() != columns.len()
        || sigmas.iter().any(|s| !valid_noise_sigma(*s))
    {
        return Err(KernelError::InvalidInput(
            "observation lengths or noise sigmas",
        ));
    }
    Ok(())
}

fn independent<'g>(
    geometry: impl Into<GeometrySource<'g>>,
    observables: Vec<Observable<Vec<V3>>>,
    columns: Vec<Vec<f64>>,
    sigmas: &[f64],
) -> Result<PreparedBatch<'g>> {
    let geometry = geometry.into();
    validate_independent_columns(&columns, sigmas, geometry.len())?;
    let values = (0..geometry.len())
        .map(|i| columns.iter().map(|c| c[i]).collect())
        .collect();
    let mut covariance = DynMatrix::zeros(columns.len(), columns.len());
    for (i, sigma) in sigmas.iter().enumerate() {
        covariance[(i, i)] = sigma.powi(2);
    }
    let model = Model::new(observables, geometry.len())?;
    if geometry.is_empty() || model.dimension() != columns.len() {
        return Err(KernelError::InvalidInput("joint observation dimensions"));
    }
    if columns.iter().flatten().any(|value| !value.is_finite()) {
        return Err(KernelError::InvalidInput("observations must be finite"));
    }
    geometry
        .len()
        .checked_mul(model.dimension())
        .ok_or(KernelError::InvalidInput("output dimensions overflow"))?;
    let factor = factor_covariance(&covariance, model.dimension())?;
    Ok(PreparedBatch::from_joint(
        geometry,
        JointBatch {
            model,
            values,
            noise: Noise::Shared(NoiseAtEpoch { covariance, factor }),
        },
    ))
}

/// Carrier-relative Doppler in Hz; receiver/transmitter offset already calibrated.
pub fn doppler<'g>(
    geometry: impl Into<GeometrySource<'g>>,
    carrier_hz: f64,
    values_hz: Vec<f64>,
    sigma_hz: f64,
) -> Result<PreparedBatch<'g>> {
    independent(
        geometry,
        vec![Observable::Doppler {
            carrier_hz,
            offset_hz: 0.0,
        }],
        vec![values_hz],
        &[sigma_hz],
    )
}
/// Continuous, unwrapped phase difference in radians, with calibrated offsets.
/// Supply one ENU baseline vector per measurement; repeat vectors for fixed baselines.
pub fn phase_difference<'g>(
    geometry: impl Into<GeometrySource<'g>>,
    interferometer: Interferometer,
    baselines_enu_m: Vec<V3>,
    values_rad: Vec<f64>,
    sigma_rad: f64,
) -> Result<PreparedBatch<'g>> {
    independent(
        geometry,
        vec![Observable::PhaseDifference {
            interferometer,
            baseline_enu_m: baselines_enu_m,
            frequency_offset_hz: 0.0,
            phase_offset_rad: 0.0,
        }],
        vec![values_rad],
        &[sigma_rad],
    )
}
/// Signed f1 − f2 in Hz. No ordinary Doppler or phase observations are required.
/// Supply one ENU baseline vector per measurement; repeat vectors for fixed baselines.
pub fn differential_doppler<'g>(
    geometry: impl Into<GeometrySource<'g>>,
    interferometer: Interferometer,
    baselines_enu_m: Vec<V3>,
    values_hz: Vec<f64>,
    sigma_hz: f64,
) -> Result<PreparedBatch<'g>> {
    independent(
        geometry,
        vec![Observable::DifferentialDoppler {
            interferometer,
            baseline_enu_m: baselines_enu_m,
        }],
        vec![values_hz],
        &[sigma_hz],
    )
}
/// Azimuth eastward from north and elevation in radians.
pub fn azimuth_elevation<'g>(
    geometry: impl Into<GeometrySource<'g>>,
    azimuth_rad: Vec<f64>,
    elevation_rad: Vec<f64>,
    sigma_az_rad: f64,
    sigma_el_rad: f64,
) -> Result<PreparedBatch<'g>> {
    independent(
        geometry,
        vec![Observable::AzimuthElevation],
        vec![azimuth_rad, elevation_rad],
        &[sigma_az_rad, sigma_el_rad],
    )
}

/// Covariance of `[f1, f1 − f2]` given independent antenna frequency errors.
/// The positive cross term preserves the noise shared through antenna 1.
pub fn doppler_differential_covariance(
    sigma_antenna1_hz: f64,
    sigma_antenna2_hz: f64,
) -> Result<Matrix<f64, 2, 2>> {
    for s in [sigma_antenna1_hz, sigma_antenna2_hz] {
        if !s.is_finite() || s <= 0.0 || !s.powi(2).is_finite() || s.powi(2) == 0.0 {
            return Err(KernelError::InvalidInput("antenna noise sigmas"));
        }
    }
    let a = sigma_antenna1_hz.powi(2);
    let b = sigma_antenna2_hz.powi(2);
    let r = Matrix::new([[a, a], [a, a + b]]);
    if !r.as_slice().iter().all(|v| v.is_finite()) || r.cholesky().is_err() {
        return Err(KernelError::Numerical(
            "antenna covariance exceeds numeric range",
        ));
    }
    Ok(r)
}

/// Fit prepared passive batches from an existing orbit guess using numeris defaults.
pub fn fit<'g>(
    config: PropagationConfig,
    batches: impl Into<crate::optim::BatchSource<'g>>,
    guess: &DynVector<f64>,
) -> std::result::Result<FitResult, FitError> {
    fit_with_options(config, batches, guess, None, &LmSettings::default())
}
pub fn fit_with_options<'g>(
    config: PropagationConfig,
    batches: impl Into<crate::optim::BatchSource<'g>>,
    guess: &DynVector<f64>,
    scales: Option<&[f64]>,
    settings: &LmSettings<f64>,
) -> std::result::Result<FitResult, FitError> {
    fit_lm(&ODProblem::new(config, batches)?, guess, scales, settings)
}

/// Fit calibrated prepared batches with absolute physical parameter covariance.
///
/// Inputs and LM settings match [`fit_with_options`]. Measurement covariances
/// define absolute uncertainties. A converged rank-deficient solution returns
/// an error; covariance is never regularized or scaled by residual variance.
/// See [`crate::optim::parameter_covariance`] for numerical rank semantics.
pub fn fit_with_covariance<'g>(
    config: PropagationConfig,
    batches: impl Into<crate::optim::BatchSource<'g>>,
    guess: &DynVector<f64>,
    scales: Option<&[f64]>,
    settings: &LmSettings<f64>,
) -> std::result::Result<CovarianceFit, CovarianceFitError> {
    fit_lm_with_covariance(&ODProblem::new(config, batches)?, guess, scales, settings)
}

#[derive(Debug)]
pub(crate) struct JointBatch {
    model: Model<Vec<V3>>,
    values: Vec<Vec<f64>>,
    noise: Noise,
}

#[derive(Debug)]
struct NoiseAtEpoch {
    covariance: DynMatrix<f64>,
    factor: DynCholesky<f64>,
}
#[derive(Debug)]
enum Noise {
    Shared(NoiseAtEpoch),
    PerEpoch(Vec<NoiseAtEpoch>),
}
impl Noise {
    fn row(&self, row: usize) -> &NoiseAtEpoch {
        match self {
            Self::Shared(noise) => noise,
            Self::PerEpoch(epochs) => &epochs[row],
        }
    }
}

impl JointBatch {
    pub(crate) fn dimension(&self) -> usize {
        self.model.dimension()
    }

    pub(crate) fn observation<'g, const M: usize>(
        &'g self,
        geometry: &'g PreparedGeometry,
        row: usize,
    ) -> Result<Observation<M, CalibratedSensor<'g, M>>> {
        if M != self.dimension() || row >= geometry.len() {
            return Err(KernelError::InvalidInput("joint observation row/dimension"));
        }
        let mut value = Vector::zeros();
        let mut covariance = Matrix::zeros();
        for i in 0..M {
            value[i] = self.values[row][i];
            for j in 0..M {
                covariance[(i, j)] = self.noise.row(row).covariance[(i, j)];
            }
        }
        Ok(Observation {
            time: geometry.timestamps()[row],
            value,
            covariance,
            model: CalibratedSensor {
                model: ModelSource::Borrowed(&self.model, row),
                geometry: StationGeometrySource::Borrowed(
                    geometry
                        .row(row)
                        .ok_or(KernelError::InvalidInput("joint geometry row"))?,
                ),
                time: geometry.timestamps()[row],
            },
        })
    }

    pub(crate) fn evaluate(
        &self,
        geometry: &PreparedGeometry,
        sens: &Sensitivities,
        mapping: Option<&[usize]>,
        output: &mut Evaluation,
        offset: usize,
    ) -> Result<()> {
        let p = sens.columns()?;
        for (i, station) in geometry.iter_rows().enumerate() {
            let index = mapping.map_or(i, |indices| indices[i]);
            let sample = SensitivityRow {
                time: sens.timestamps[index],
                state: &sens.states[index],
                stm: &sens.stm[index],
                state_dcd: sens.state_dcd.get(index),
            };
            let (predicted, partial) = self.model.predict(sample.state, station, sample.time, i)?;
            self.write_row(i, &predicted, &partial, &sample, output, offset, p)?;
        }
        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    fn write_row(
        &self,
        i: usize,
        predicted: &Vector<f64, 5>,
        partial: &Matrix<f64, 5, 6>,
        sample: &SensitivityRow<'_>,
        output: &mut Evaluation,
        offset: usize,
        p: usize,
    ) -> Result<()> {
        let m = self.dimension();
        let n = self.values.len();
        let lower = self.noise.row(i).factor.l();
        let chained = partial * sample.stm;
        let drag = sample
            .state_dcd
            .map_or_else(Vector::zeros, |drag| partial * drag);
        let mut residual = Vector::<f64, 5>::zeros();
        let mut jacobian = Matrix::<f64, 5, 7>::zeros();
        for j in 0..m {
            residual[j] = predicted[j] - self.values[i][j];
            if self.model.azimuth_column() == Some(j) {
                residual[j] = wrap_pi(residual[j]);
            }
            for k in 0..6 {
                jacobian[(j, k)] = chained[(j, k)];
            }
            if p == 7 {
                jacobian[(j, 6)] = drag[j];
            }
            // Forward solve Lr and LJ for this epoch's full covariance.
            for earlier in 0..j {
                let l = lower[(j, earlier)];
                residual[j] -= l * residual[earlier];
                for k in 0..p {
                    jacobian[(j, k)] -= l * jacobian[(earlier, k)];
                }
            }
            residual[j] /= lower[(j, j)];
            let row = offset + j * n + i;
            if !residual[j].is_finite() {
                return Err(KernelError::Numerical(
                    "whitened residual exceeds numeric range",
                ));
            }
            output.residual[row] = residual[j];
            for k in 0..p {
                jacobian[(j, k)] /= lower[(j, j)];
                if !jacobian[(j, k)].is_finite() {
                    return Err(KernelError::Numerical(
                        "whitened Jacobian exceeds numeric range",
                    ));
                }
                output.jacobian[(row, k)] = jacobian[(j, k)];
            }
        }
        Ok(())
    }
}
