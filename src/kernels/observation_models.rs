//! Instantaneous observables and chained batch residuals/Jacobians.
use super::*;
use numeris::{DynMatrix, DynVector};
use satkit::Instant;
use std::sync::Arc;

/// One simultaneous geometric observation with its epoch and station location.
#[derive(Clone, Copy, Debug)]
pub struct Measurement {
    pub time: Instant,
    pub station: GroundStation,
    pub range_rate: f64,
    pub azimuth: f64,
    pub elevation: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct MeasurementNoise {
    pub sigma_rr: f64,
    pub sigma_az: f64,
    pub sigma_el: f64,
}

/// Predicted range-rate (m/s) and its instantaneous 1×6 state Jacobian.
pub fn range_rate_partial(
    r_sat: &V3,
    v_sat: &V3,
    r_sta: &V3,
    v_sta: &V3,
) -> Result<(f64, Matrix<f64, 1, 6>)> {
    if !finite(r_sat) || !finite(v_sat) || !finite(r_sta) || !finite(v_sta) {
        return Err(KernelError::InvalidInput(
            "measurement state must be finite",
        ));
    }
    let rho_vec = r_sat - r_sta;
    let v_rel = v_sat - v_sta;
    let rho = rho_vec.norm();
    if rho == 0.0 {
        return Err(KernelError::ZeroRange);
    }
    if !rho.is_finite() {
        return Err(KernelError::InvalidInput("range exceeds numeric range"));
    }
    let los = rho_vec / rho;
    let rho_dot = los.dot(&v_rel);
    let d_dr = (v_rel - los * rho_dot) / rho;
    let mut h = Matrix::<f64, 1, 6>::zeros();
    for k in 0..3 {
        h[(0, k)] = d_dr[k];
        h[(0, k + 3)] = los[k];
    }
    if !rho_dot.is_finite() || !finite(&h) {
        return Err(KernelError::InvalidInput(
            "range-rate geometry exceeds numeric range",
        ));
    }
    Ok((rho_dot, h))
}

/// Azimuth/elevation (rad) and their instantaneous 2×6 state Jacobian.
/// Velocity columns are zero. At zenith/nadir (horizontal/range ≤ 1e-12),
/// azimuth is unobservable: callers must omit it or use a LOS measurement.
pub fn az_el_partial(
    r_sat: &V3,
    r_sta: &V3,
    east: &V3,
    north: &V3,
    up: &V3,
) -> Result<([f64; 2], Matrix<f64, 2, 6>)> {
    let mut rotation = M3::zeros();
    rotation.set_block(0, 0, east);
    rotation.set_block(0, 1, north);
    rotation.set_block(0, 2, up);
    az_el_partial_with_rotation(r_sat, r_sta, &rotation)
}

/// Angle prediction/Jacobian using a precomputed ENU→GCRF rotation matrix.
/// No geodetic conversion or frame/quaternion evaluation occurs here.
pub fn az_el_partial_with_rotation(
    r_sat: &V3,
    r_sta: &V3,
    enu_to_gcrf: &M3,
) -> Result<([f64; 2], Matrix<f64, 2, 6>)> {
    let rotation = ValidatedRotation::new(*enu_to_gcrf)?;
    az_el_partial_prepared(r_sat, r_sta, &rotation)
}

/// Prediction and instantaneous partial using an already validated rotation.
pub fn az_el_partial_prepared(
    r_sat: &V3,
    r_sta: &V3,
    rotation: &ValidatedRotation,
) -> Result<([f64; 2], Matrix<f64, 2, 6>)> {
    if !finite(r_sat) || !finite(r_sta) {
        return Err(KernelError::InvalidInput(
            "measurement position must be finite",
        ));
    }
    let enu_to_gcrf = rotation.matrix();
    let east = enu_to_gcrf.block::<3, 1>(0, 0);
    let north = enu_to_gcrf.block::<3, 1>(0, 1);
    let up = enu_to_gcrf.block::<3, 1>(0, 2);
    let local = enu_to_gcrf.transpose() * (r_sat - r_sta);
    let (e, n, u) = (local[0], local[1], local[2]);
    let horiz = e.hypot(n);
    let rho = horiz.hypot(u);
    if rho == 0.0 {
        return Err(KernelError::ZeroRange);
    }
    if horiz <= 1e-12 * rho {
        return Err(KernelError::UndefinedAzimuth);
    }
    let az = e.atan2(n);
    // atan2 avoids asin's rounding to ±1 close to zenith/nadir.
    let el = u.atan2(horiz);
    let d_az = (east * (n / horiz) - north * (e / horiz)) / horiz;
    let d_el = (up * (horiz / rho) - (east * (e / horiz) + north * (n / horiz)) * (u / rho)) / rho;
    let mut h = Matrix::<f64, 2, 6>::zeros();
    for k in 0..3 {
        h[(0, k)] = d_az[k];
        h[(1, k)] = d_el[k];
    }
    if !finite(&h) || !rho.is_finite() {
        return Err(KernelError::InvalidInput(
            "angle geometry exceeds numeric range",
        ));
    }
    Ok(([az, el], h))
}

#[derive(Clone, Debug)]
pub struct FullJacobian {
    pub range_rate: DynMatrix<f64>,
    pub azimuth: DynMatrix<f64>,
    pub elevation: DynMatrix<f64>,
}

fn check_geometry(n: usize, station: &GCRFTrajectory, enu: &EnuBasisTrajectory) -> Result<()> {
    if station.timestamps.len() != n
        || station.position.len() != n
        || station.velocity.len() != n
        || enu.timestamps != station.timestamps
        || station.ground_stations.len() != n
        || enu.enu_to_gcrf.len() != n
    {
        return Err(KernelError::InvalidInput(
            "station/ENU lengths or epochs do not match",
        ));
    }
    Ok(())
}

/// Unwhitened measurement derivatives with respect to the seven reference
/// parameters. The instantaneous measurement kernels are also usable directly.
pub fn compute_full_jacobian(
    sens: &Sensitivities,
    station: &GCRFTrajectory,
    enu: &EnuBasisTrajectory,
) -> Result<FullJacobian> {
    let n = sens.states.len();
    check_geometry(n, station, enu)?;
    if sens.timestamps != station.timestamps || sens.stm.len() != n || sens.state_dcd.len() != n {
        return Err(KernelError::InvalidInput(
            "sensitivity lengths or epochs do not match",
        ));
    }
    let mut jac = FullJacobian {
        range_rate: DynMatrix::zeros(n, 7),
        azimuth: DynMatrix::zeros(n, 7),
        elevation: DynMatrix::zeros(n, 7),
    };
    for i in 0..n {
        let r = r_of(&sens.states[i]);
        let v = v_of(&sens.states[i]);
        let (_, rr) = range_rate_partial(&r, &v, &station.position[i], &station.velocity[i])?;
        let (_, angles) =
            az_el_partial_with_rotation(&r, &station.position[i], &enu.enu_to_gcrf[i])?;
        if !finite(&sens.stm[i]) || !finite(&sens.state_dcd[i]) {
            return Err(KernelError::InvalidInput("sensitivities must be finite"));
        }
        let rr_state = rr * sens.stm[i];
        let rr_drag = rr * sens.state_dcd[i];
        let ang_state = angles * sens.stm[i];
        let ang_drag = angles * sens.state_dcd[i];
        for k in 0..6 {
            jac.range_rate[(i, k)] = rr_state[(0, k)];
            jac.azimuth[(i, k)] = ang_state[(0, k)];
            jac.elevation[(i, k)] = ang_state[(1, k)];
        }
        jac.range_rate[(i, 6)] = rr_drag[0];
        jac.azimuth[(i, 6)] = ang_drag[0];
        jac.elevation[(i, 6)] = ang_drag[1];
    }
    Ok(jac)
}

#[derive(Clone, Debug)]
pub struct Observations {
    pub range_rate: Vec<f64>,
    pub azimuth: Vec<f64>,
    pub elevation: Vec<f64>,
    /// Positive standard deviations: m/s, rad, rad respectively.
    pub sigma_rr: f64,
    pub sigma_az: f64,
    pub sigma_el: f64,
}

/// Owned output, also usable as scratch across evaluations of the same shape.
/// After an evaluation error its contents are unspecified.
#[derive(Clone, Debug, PartialEq)]
pub struct Evaluation {
    pub residual: DynVector<f64>,
    pub jacobian: DynMatrix<f64>,
}

impl Default for Evaluation {
    fn default() -> Self {
        Self {
            residual: DynVector::zeros(0),
            jacobian: DynMatrix::zeros(0, 0),
        }
    }
}

impl Evaluation {
    // numeris does not expose resizing dynamic matrices. Reuse storage when
    // the shape matches; replace only the buffer whose shape has changed.
    pub(crate) fn resize(&mut self, rows: usize, columns: usize) -> Result<()> {
        rows.checked_mul(columns)
            .ok_or(KernelError::InvalidInput("output dimensions overflow"))?;
        if self.residual.len() != rows {
            self.residual = DynVector::zeros(rows);
        }
        if self.jacobian.nrows() != rows || self.jacobian.ncols() != columns {
            self.jacobian = DynMatrix::zeros(rows, columns);
        }
        Ok(())
    }
    pub fn rss(&self) -> f64 {
        self.residual.dot(&self.residual)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MeasurementKind {
    RangeRate,
    PhaseDifference,
    AzimuthElevation,
    Combined,
    /// Calibrated passive observables with per-epoch full covariance.
    Joint,
}

#[derive(Clone, Debug)]
struct ObservationColumn {
    values: Vec<f64>,
    sigma: f64,
    angular: bool,
}

/// Immutable measurement-specific batch. Geometry can be shared using `Arc`.
/// Rows are grouped by observable: RR; AZ, EL; or RR, AZ, EL.
#[derive(Clone, Debug)]
pub struct PreparedBatch {
    geometry: Arc<PreparedGeometry>,
    kind: MeasurementKind,
    observations: Vec<ObservationColumn>,
    interferometer: Option<Interferometer>,
    joint: Option<Box<crate::measurements::JointBatch>>,
}

impl PreparedBatch {
    pub(crate) fn from_joint(
        geometry: Arc<PreparedGeometry>,
        joint: crate::measurements::JointBatch,
    ) -> Self {
        Self {
            geometry,
            kind: MeasurementKind::Joint,
            observations: Vec::new(),
            interferometer: None,
            joint: Some(Box::new(joint)),
        }
    }

    /// Number of physical components per epoch, including both angles.
    pub fn components(&self) -> usize {
        self.joint
            .as_ref()
            .map_or(self.observations.len(), |j| j.dimension())
    }

    /// Get a facade observation with its full physical covariance and sensor.
    /// Available for batches constructed by `tychonic::measurements`.
    pub fn joint_observation<const M: usize>(
        &self,
        row: usize,
    ) -> Result<crate::estimate::Observation<M, crate::measurements::CalibratedSensor<M>>> {
        self.joint
            .as_ref()
            .ok_or(KernelError::InvalidInput(
                "batch is not a joint passive batch",
            ))?
            .observation(&self.geometry, row)
    }

    /// Calibrated carrier-relative Doppler in Hz.
    pub fn doppler(
        geometry: Arc<PreparedGeometry>,
        carrier_hz: f64,
        values_hz: Vec<f64>,
        sigma_hz: f64,
    ) -> Result<Self> {
        crate::measurements::doppler(geometry, carrier_hz, values_hz, sigma_hz)
    }

    /// Calibrated standalone f1 − f2 in Hz.
    pub fn differential_doppler(
        geometry: Arc<PreparedGeometry>,
        interferometer: Interferometer,
        values_hz: Vec<f64>,
        sigma_hz: f64,
    ) -> Result<Self> {
        crate::measurements::differential_doppler(geometry, interferometer, values_hz, sigma_hz)
    }

    /// Calibrated continuous phase (rad). Use PassiveRfProblem to estimate
    /// receiver offsets and correlated OU drift.
    pub fn phase_difference(
        geometry: Arc<PreparedGeometry>,
        interferometer: Interferometer,
        values: Vec<f64>,
        sigma_rad: f64,
    ) -> Result<Self> {
        let mut batch = Self::new(
            geometry,
            MeasurementKind::PhaseDifference,
            vec![ObservationColumn {
                values,
                sigma: sigma_rad,
                angular: false,
            }],
        )?;
        batch.interferometer = Some(interferometer);
        Ok(batch)
    }
    pub fn interferometer(&self) -> Option<Interferometer> {
        self.interferometer
    }
    pub fn range_rate(
        geometry: Arc<PreparedGeometry>,
        values: Vec<f64>,
        sigma: f64,
    ) -> Result<Self> {
        Self::new(
            geometry,
            MeasurementKind::RangeRate,
            vec![ObservationColumn {
                values,
                sigma,
                angular: false,
            }],
        )
    }

    pub fn azimuth_elevation(
        geometry: Arc<PreparedGeometry>,
        azimuth: Vec<f64>,
        elevation: Vec<f64>,
        sigma_az: f64,
        sigma_el: f64,
    ) -> Result<Self> {
        Self::new(
            geometry,
            MeasurementKind::AzimuthElevation,
            vec![
                ObservationColumn {
                    values: azimuth,
                    sigma: sigma_az,
                    angular: true,
                },
                ObservationColumn {
                    values: elevation,
                    sigma: sigma_el,
                    angular: false,
                },
            ],
        )
    }

    pub fn combined(geometry: Arc<PreparedGeometry>, obs: Observations) -> Result<Self> {
        Self::new(
            geometry,
            MeasurementKind::Combined,
            vec![
                ObservationColumn {
                    values: obs.range_rate,
                    sigma: obs.sigma_rr,
                    angular: false,
                },
                ObservationColumn {
                    values: obs.azimuth,
                    sigma: obs.sigma_az,
                    angular: true,
                },
                ObservationColumn {
                    values: obs.elevation,
                    sigma: obs.sigma_el,
                    angular: false,
                },
            ],
        )
    }

    pub fn from_measurements(
        measurements: &[Measurement],
        noise: MeasurementNoise,
    ) -> Result<Self> {
        let times: Vec<_> = measurements.iter().map(|m| m.time).collect();
        let stations: Vec<_> = measurements.iter().map(|m| m.station).collect();
        Self::combined(
            Arc::new(PreparedGeometry::new(&times, &stations)?),
            Observations {
                range_rate: measurements.iter().map(|m| m.range_rate).collect(),
                azimuth: measurements.iter().map(|m| m.azimuth).collect(),
                elevation: measurements.iter().map(|m| m.elevation).collect(),
                sigma_rr: noise.sigma_rr,
                sigma_az: noise.sigma_az,
                sigma_el: noise.sigma_el,
            },
        )
    }

    fn new(
        geometry: Arc<PreparedGeometry>,
        kind: MeasurementKind,
        observations: Vec<ObservationColumn>,
    ) -> Result<Self> {
        if geometry.is_empty() {
            return Err(KernelError::InvalidInput("empty observation batch"));
        }
        geometry
            .len()
            .checked_mul(observations.len())
            .ok_or(KernelError::InvalidInput("output dimensions overflow"))?;
        for obs in &observations {
            if obs.values.len() != geometry.len() || !obs.values.iter().all(|x| x.is_finite()) {
                return Err(KernelError::InvalidInput("observation lengths or values"));
            }
            if !obs.sigma.is_finite() || obs.sigma <= 0.0 {
                return Err(KernelError::InvalidInput(
                    "measurement sigmas must be finite and positive",
                ));
            }
        }
        Ok(Self {
            geometry,
            kind,
            observations,
            interferometer: None,
            joint: None,
        })
    }

    pub fn geometry(&self) -> &Arc<PreparedGeometry> {
        &self.geometry
    }
    pub fn kind(&self) -> MeasurementKind {
        self.kind
    }
    pub fn timestamps(&self) -> &[Instant] {
        self.geometry.timestamps()
    }
    pub fn rows(&self) -> usize {
        self.geometry.len() * self.components()
    }

    /// Values and standard deviation for a column in the documented observable
    /// order (RR; AZ, EL; or RR, AZ, EL). Joint batches use
    /// `joint_observation` instead; they and out-of-range indices return `None`.
    pub fn observation_column(&self, index: usize) -> Option<(&[f64], f64)> {
        self.observations
            .get(index)
            .map(|column| (column.values.as_slice(), column.sigma))
    }

    /// Propagate reference state/parameters then evaluate this model.
    pub fn evaluate(
        &self,
        parameters: &DynVector<f64>,
        config: &PropagationConfig,
    ) -> Result<Evaluation> {
        let mut output = Evaluation::default();
        self.evaluate_into(parameters, config, &mut output)?;
        Ok(output)
    }

    pub fn evaluate_into(
        &self,
        parameters: &DynVector<f64>,
        config: &PropagationConfig,
        output: &mut Evaluation,
    ) -> Result<()> {
        let sens = config.sample(parameters, self.timestamps())?;
        self.evaluate_from_sensitivities_into(&sens, output)
    }

    /// Reuse externally sampled sensitivities in exactly the batch's row order.
    pub fn evaluate_from_sensitivities(&self, sens: &Sensitivities) -> Result<Evaluation> {
        let mut output = Evaluation::default();
        self.evaluate_from_sensitivities_into(sens, &mut output)?;
        Ok(output)
    }

    pub fn evaluate_from_sensitivities_into(
        &self,
        sens: &Sensitivities,
        output: &mut Evaluation,
    ) -> Result<()> {
        if sens.timestamps != self.timestamps() {
            return Err(KernelError::InvalidInput(
                "sensitivity and geometry epochs do not match",
            ));
        }
        self.evaluate_mapped_into(sens, None, output, 0, self.rows())
    }

    // Optimizer supplies a prepared epoch-union mapping and writes directly
    // into stacked output. Mapping bounds/epochs remain checked here.
    pub(crate) fn evaluate_mapped_into(
        &self,
        sens: &Sensitivities,
        mapping: Option<&[usize]>,
        output: &mut Evaluation,
        offset: usize,
        total_rows: usize,
    ) -> Result<()> {
        let columns = sens.columns()?;
        if offset
            .checked_add(self.rows())
            .is_none_or(|end| end > total_rows)
            || mapping.is_some_and(|m| m.len() != self.geometry.len())
        {
            return Err(KernelError::InvalidInput(
                "batch output or mapping dimensions",
            ));
        }
        output.resize(total_rows, columns)?;
        if let Some(joint) = &self.joint {
            return joint.evaluate(&self.geometry, sens, mapping, output, offset);
        }
        match (self.kind, columns) {
            (MeasurementKind::RangeRate | MeasurementKind::PhaseDifference, 6) => {
                self.evaluate_rows::<1, 6>(sens, mapping, output, offset)
            }
            (MeasurementKind::RangeRate | MeasurementKind::PhaseDifference, _) => {
                self.evaluate_rows::<1, 7>(sens, mapping, output, offset)
            }
            (MeasurementKind::AzimuthElevation, 6) => {
                self.evaluate_rows::<2, 6>(sens, mapping, output, offset)
            }
            (MeasurementKind::AzimuthElevation, _) => {
                self.evaluate_rows::<2, 7>(sens, mapping, output, offset)
            }
            (MeasurementKind::Combined, 6) => {
                self.evaluate_rows::<3, 6>(sens, mapping, output, offset)
            }
            (MeasurementKind::Combined, _) => {
                self.evaluate_rows::<3, 7>(sens, mapping, output, offset)
            }
            (MeasurementKind::Joint, _) => unreachable!("joint batch handled above"),
        }
    }

    fn evaluate_rows<const M: usize, const P: usize>(
        &self,
        sens: &Sensitivities,
        mapping: Option<&[usize]>,
        output: &mut Evaluation,
        offset: usize,
    ) -> Result<()> {
        let n = self.geometry.len();
        for i in 0..n {
            let index = mapping.map_or(i, |m| m[i]);
            if sens.timestamps.get(index) != Some(&self.timestamps()[i]) {
                return Err(KernelError::InvalidInput(
                    "sensitivity and geometry epochs do not match",
                ));
            }
            let geometry = self.geometry.row(i).expect("validated geometry row");
            let state = &sens.states[index];
            let r = r_of(state);
            let v = v_of(state);
            let mut predictions = [0.0; M];
            let mut partial = Matrix::<f64, M, 6>::zeros();
            if let Some(interferometer) = self.interferometer {
                let pred = interferometer.evaluate(state, geometry, 0.0)?;
                predictions[0] = pred.phase_rad;
                for k in 0..6 {
                    partial[(0, k)] = pred.partial[(1, k)];
                }
            } else if M != 2 {
                let (rr, h) = range_rate_partial(&r, &v, geometry.position(), geometry.velocity())?;
                predictions[0] = rr;
                partial.set_block(0, 0, &h);
            }
            if M != 1 {
                let (angles, h) =
                    az_el_partial_prepared(&r, geometry.position(), geometry.rotation())?;
                let start = if M == 3 { 1 } else { 0 };
                predictions[start] = angles[0];
                predictions[start + 1] = angles[1];
                partial.set_block(start, 0, &h);
            }
            if !finite(state)
                || !finite(&sens.stm[index])
                || (!sens.state_dcd.is_empty() && !finite(&sens.state_dcd[index]))
            {
                return Err(KernelError::InvalidInput(
                    "states and sensitivities must be finite",
                ));
            }
            let mut sensitivity = Matrix::<f64, 6, P>::zeros();
            sensitivity.set_block(0, 0, &sens.stm[index]);
            if P == 7 {
                sensitivity.set_block(0, 6, &sens.state_dcd[index]);
            }
            let chained = chain_rule_prevalidated(&partial, &sensitivity);
            for m in 0..M {
                let obs = &self.observations[m];
                let row = offset + m * n + i;
                let difference = predictions[m] - obs.values[i];
                let residual = (if obs.angular {
                    wrap_pi(difference)
                } else {
                    difference
                }) / obs.sigma;
                if !residual.is_finite() {
                    return Err(KernelError::InvalidInput(
                        "whitened residual exceeds numeric range",
                    ));
                }
                output.residual[row] = residual;
                for k in 0..P {
                    let derivative = chained[(m, k)] / obs.sigma;
                    if !derivative.is_finite() {
                        return Err(KernelError::InvalidInput(
                            "whitened Jacobian exceeds numeric range",
                        ));
                    }
                    output.jacobian[(row, k)] = derivative;
                }
            }
        }
        Ok(())
    }
}
