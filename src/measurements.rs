//! Calibrated passive measurements and convenient batch fitting.
//!
//! Values are extracted observables, not receiver IQ samples. Joint observations
//! use Doppler Hz, phase-difference rad, differential Doppler Hz, azimuth rad,
//! elevation rad, in that order, omitting absent components. Batch residuals
//! remain grouped by component, as in the original prepared-batch API.

use crate::estimate::{MeasurementModel, Observation, OrbitEkf};
use crate::kernels::*;
use crate::optim::{FitError, FitResult, ODProblem, fit_lm};
use numeris::{DynMatrix, DynVector, Matrix, Vector, optim::LmSettings};
use satkit::Instant;
use std::sync::Arc;

mod rf;
pub use rf::{GaussianPrior, RfBuilder, RfPriors};

/// Six-component GCRF orbit filter with fixed drag.
pub type OrbitFilter = OrbitEkf<6>;
/// GCRF orbit filter also estimating C_d A/m.
pub type OrbitFilterWithDrag = OrbitEkf<7>;

/// A calibrated measurement model. Offsets are known constants in physical units.
/// A joint model may contain each variant once, with independent RF baselines.
#[derive(Clone, Copy, Debug)]
pub enum Observable {
    Doppler {
        carrier_hz: f64,
        offset_hz: f64,
    },
    PhaseDifference {
        interferometer: Interferometer,
        frequency_offset_hz: f64,
        phase_offset_rad: f64,
    },
    DifferentialDoppler {
        interferometer: Interferometer,
    },
    AzimuthElevation,
}

impl Observable {
    fn order(self) -> usize {
        match self {
            Self::Doppler { .. } => 0,
            Self::PhaseDifference { .. } => 1,
            Self::DifferentialDoppler { .. } => 2,
            Self::AzimuthElevation => 3,
        }
    }
    fn dimension(self) -> usize {
        if matches!(self, Self::AzimuthElevation) {
            2
        } else {
            1
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct Model(Arc<[Observable]>);

impl Model {
    fn new(mut observables: Vec<Observable>) -> Result<Self> {
        if observables.is_empty() {
            return Err(KernelError::InvalidInput("empty observable selection"));
        }
        observables.sort_by_key(|o| o.order());
        if observables.windows(2).any(|w| w[0].order() == w[1].order()) {
            return Err(KernelError::InvalidInput("duplicate observable selection"));
        }
        for observable in &observables {
            match *observable {
                Observable::Doppler {
                    carrier_hz,
                    offset_hz,
                } if !carrier_hz.is_finite() || carrier_hz <= 0.0 || !offset_hz.is_finite() => {
                    return Err(KernelError::InvalidInput("Doppler carrier/offset"));
                }
                Observable::PhaseDifference {
                    frequency_offset_hz,
                    phase_offset_rad,
                    ..
                } if !frequency_offset_hz.is_finite() || !phase_offset_rad.is_finite() => {
                    return Err(KernelError::InvalidInput("phase calibration offsets"));
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
    ) -> Result<(Vector<f64, 5>, Matrix<f64, 5, 6>)> {
        let mut value = Vector::zeros();
        let mut partial = Matrix::zeros();
        let mut row = 0;
        for observable in self.0.iter() {
            match *observable {
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
                    value[row] = scale * rr + offset_hz;
                    partial.set_block(row, 0, &(h * scale));
                }
                Observable::PhaseDifference {
                    interferometer,
                    frequency_offset_hz,
                    phase_offset_rad,
                } => {
                    let pred = interferometer.evaluate(state, geometry, frequency_offset_hz)?;
                    value[row] = pred.phase_rad + phase_offset_rad;
                    for k in 0..6 {
                        partial[(row, k)] = pred.partial[(1, k)];
                    }
                }
                Observable::DifferentialDoppler { interferometer } => {
                    let pred = interferometer.differential(state, geometry, time)?;
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
            return Err(KernelError::InvalidInput(
                "measurement prediction exceeds numeric range",
            ));
        }
        Ok((value, partial))
    }
}

/// Instantaneous calibrated sensor, suitable for `OrbitEkf::observe` or `update`.
/// Use the named sensor constructors for one observable, or `joint_sensor` for fusion.
#[derive(Clone, Debug)]
pub struct CalibratedSensor<const M: usize> {
    model: Model,
    geometry: StationGeometry,
    time: Instant,
}

impl<const M: usize> MeasurementModel<M> for CalibratedSensor<M> {
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
pub fn joint_sensor<const M: usize>(
    geometry: StationGeometry,
    time: Instant,
    observables: Vec<Observable>,
) -> Result<CalibratedSensor<M>> {
    let model = Model::new(observables)?;
    if M != model.dimension() {
        return Err(KernelError::InvalidInput(
            "sensor dimension does not match observables",
        ));
    }
    Ok(CalibratedSensor {
        model,
        geometry,
        time,
    })
}

pub fn doppler_sensor(
    geometry: StationGeometry,
    time: Instant,
    carrier_hz: f64,
) -> Result<CalibratedSensor<1>> {
    joint_sensor(
        geometry,
        time,
        vec![Observable::Doppler {
            carrier_hz,
            offset_hz: 0.0,
        }],
    )
}
pub fn phase_difference_sensor(
    geometry: StationGeometry,
    time: Instant,
    interferometer: Interferometer,
) -> Result<CalibratedSensor<1>> {
    joint_sensor(
        geometry,
        time,
        vec![Observable::PhaseDifference {
            interferometer,
            frequency_offset_hz: 0.0,
            phase_offset_rad: 0.0,
        }],
    )
}
pub fn differential_doppler_sensor(
    geometry: StationGeometry,
    time: Instant,
    interferometer: Interferometer,
) -> Result<CalibratedSensor<1>> {
    joint_sensor(
        geometry,
        time,
        vec![Observable::DifferentialDoppler { interferometer }],
    )
}
pub fn azimuth_elevation_sensor(
    geometry: StationGeometry,
    time: Instant,
) -> Result<CalibratedSensor<2>> {
    joint_sensor(geometry, time, vec![Observable::AzimuthElevation])
}

/// Prepare measurements sharing a schedule and a full covariance per epoch.
/// Each values row and covariance uses canonical component order, regardless of
/// the order of `observables`. Covariances must be symmetric positive definite.
/// Different schedules should be supplied as separate batches to `fit`/`filter_batches`.
pub fn joint(
    geometry: Arc<PreparedGeometry>,
    observables: Vec<Observable>,
    values: Vec<Vec<f64>>,
    covariances: Vec<DynMatrix<f64>>,
) -> Result<PreparedBatch> {
    let model = Model::new(observables)?;
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
    let mut lower = Vec::with_capacity(covariances.len());
    for covariance in &covariances {
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
        lower.push(
            covariance
                .cholesky()
                .map_err(|_| {
                    KernelError::InvalidInput("joint covariance must be positive definite")
                })?
                .l()
                .clone(),
        );
    }
    Ok(PreparedBatch::from_joint(
        geometry,
        JointBatch {
            model,
            values,
            covariances,
            lower,
        },
    ))
}

fn independent(
    geometry: Arc<PreparedGeometry>,
    observables: Vec<Observable>,
    columns: Vec<Vec<f64>>,
    sigmas: &[f64],
) -> Result<PreparedBatch> {
    if columns.iter().any(|c| c.len() != geometry.len())
        || sigmas.len() != columns.len()
        || sigmas
            .iter()
            .any(|s| !s.is_finite() || *s <= 0.0 || !s.powi(2).is_finite() || s.powi(2) == 0.0)
    {
        return Err(KernelError::InvalidInput(
            "observation lengths or noise sigmas",
        ));
    }
    let values = (0..geometry.len())
        .map(|i| columns.iter().map(|c| c[i]).collect())
        .collect();
    let mut covariance = DynMatrix::zeros(columns.len(), columns.len());
    for (i, sigma) in sigmas.iter().enumerate() {
        covariance[(i, i)] = sigma.powi(2);
    }
    let covariances = vec![covariance; geometry.len()];
    joint(geometry, observables, values, covariances)
}

/// Carrier-relative Doppler in Hz; receiver/transmitter offset already calibrated.
pub fn doppler(
    geometry: Arc<PreparedGeometry>,
    carrier_hz: f64,
    values_hz: Vec<f64>,
    sigma_hz: f64,
) -> Result<PreparedBatch> {
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
pub fn phase_difference(
    geometry: Arc<PreparedGeometry>,
    interferometer: Interferometer,
    values_rad: Vec<f64>,
    sigma_rad: f64,
) -> Result<PreparedBatch> {
    independent(
        geometry,
        vec![Observable::PhaseDifference {
            interferometer,
            frequency_offset_hz: 0.0,
            phase_offset_rad: 0.0,
        }],
        vec![values_rad],
        &[sigma_rad],
    )
}
/// Signed f1 − f2 in Hz. No ordinary Doppler or phase observations are required.
pub fn differential_doppler(
    geometry: Arc<PreparedGeometry>,
    interferometer: Interferometer,
    values_hz: Vec<f64>,
    sigma_hz: f64,
) -> Result<PreparedBatch> {
    independent(
        geometry,
        vec![Observable::DifferentialDoppler { interferometer }],
        vec![values_hz],
        &[sigma_hz],
    )
}
/// Azimuth eastward from north and elevation in radians.
pub fn azimuth_elevation(
    geometry: Arc<PreparedGeometry>,
    azimuth_rad: Vec<f64>,
    elevation_rad: Vec<f64>,
    sigma_az_rad: f64,
    sigma_el_rad: f64,
) -> Result<PreparedBatch> {
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
        return Err(KernelError::InvalidInput(
            "antenna covariance exceeds numeric range",
        ));
    }
    Ok(r)
}

/// Fit prepared passive batches from an existing orbit guess using numeris defaults.
pub fn fit(
    config: PropagationConfig,
    batches: Vec<PreparedBatch>,
    guess: &DynVector<f64>,
) -> std::result::Result<FitResult, FitError> {
    fit_with_options(config, batches, guess, None, &LmSettings::default())
}
pub fn fit_with_options(
    config: PropagationConfig,
    batches: Vec<PreparedBatch>,
    guess: &DynVector<f64>,
    scales: Option<&[f64]>,
    settings: &LmSettings<f64>,
) -> std::result::Result<FitResult, FitError> {
    fit_lm(ODProblem::new(config, batches)?, guess, scales, settings)
}

#[derive(Clone, Debug)]
pub(crate) struct JointBatch {
    model: Model,
    values: Vec<Vec<f64>>,
    covariances: Vec<DynMatrix<f64>>,
    lower: Vec<DynMatrix<f64>>,
}

impl JointBatch {
    pub(crate) fn dimension(&self) -> usize {
        self.model.dimension()
    }

    pub(crate) fn observation<const M: usize>(
        &self,
        geometry: &PreparedGeometry,
        row: usize,
    ) -> Result<Observation<M, CalibratedSensor<M>>> {
        if M != self.dimension() || row >= geometry.len() {
            return Err(KernelError::InvalidInput("joint observation row/dimension"));
        }
        let mut value = Vector::zeros();
        let mut covariance = Matrix::zeros();
        for i in 0..M {
            value[i] = self.values[row][i];
            for j in 0..M {
                covariance[(i, j)] = self.covariances[row][(i, j)];
            }
        }
        Ok(Observation {
            time: geometry.timestamps()[row],
            value,
            covariance,
            model: CalibratedSensor {
                model: self.model.clone(),
                geometry: geometry.row(row).unwrap().clone(),
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
        let m = self.dimension();
        let n = geometry.len();
        for i in 0..n {
            let index = mapping.map_or(i, |indices| indices[i]);
            if sens.timestamps.get(index) != Some(&geometry.timestamps()[i]) {
                return Err(KernelError::InvalidInput(
                    "sensitivity and geometry epochs do not match",
                ));
            }
            if !sens.stm[index].as_slice().iter().all(|v| v.is_finite())
                || (p == 7
                    && !sens.state_dcd[index]
                        .as_slice()
                        .iter()
                        .all(|v| v.is_finite()))
            {
                return Err(KernelError::InvalidInput("sensitivities must be finite"));
            }
            let (predicted, partial) = self.model.predict(
                &sens.states[index],
                geometry.row(i).unwrap(),
                geometry.timestamps()[i],
            )?;
            let chained = partial * sens.stm[index];
            let drag = if p == 7 {
                partial * sens.state_dcd[index]
            } else {
                Vector::zeros()
            };
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
                    let l = self.lower[i][(j, earlier)];
                    residual[j] -= l * residual[earlier];
                    for k in 0..p {
                        jacobian[(j, k)] -= l * jacobian[(earlier, k)];
                    }
                }
                residual[j] /= self.lower[i][(j, j)];
                let row = offset + j * n + i;
                if !residual[j].is_finite() {
                    return Err(KernelError::InvalidInput(
                        "whitened residual exceeds numeric range",
                    ));
                }
                output.residual[row] = residual[j];
                for k in 0..p {
                    jacobian[(j, k)] /= self.lower[i][(j, j)];
                    if !jacobian[(j, k)].is_finite() {
                        return Err(KernelError::InvalidInput(
                            "whitened Jacobian exceeds numeric range",
                        ));
                    }
                    output.jacobian[(row, k)] = jacobian[(j, k)];
                }
            }
        }
        Ok(())
    }
}
