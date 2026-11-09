//! Continuous carrier phase using the far-field model of Henault & Guimond
//! (2025), equation (3). The baseline is antenna 1 minus antenna 2 in ENU.
use super::*;
use numeris::DynMatrix;
use satkit::Instant;

pub const SPEED_OF_LIGHT: f64 = 299_792_458.0;

/// Surveyed baseline (metres, east/north/up) and nominal carrier (Hz).
#[derive(Clone, Copy, Debug)]
pub struct Interferometer {
    baseline_enu: V3,
    carrier_hz: f64,
    spherical_phase: bool,
}

impl Interferometer {
    pub fn new(baseline_enu: V3, carrier_hz: f64) -> Result<Self> {
        if !finite(&baseline_enu)
            || !baseline_enu.norm().is_finite()
            || !carrier_hz.is_finite()
            || carrier_hz <= 0.0
        {
            return Err(KernelError::InvalidInput(
                "baseline/carrier must be finite; carrier positive",
            ));
        }
        Ok(Self {
            baseline_enu,
            carrier_hz,
            spherical_phase: false,
        })
    }
    pub fn baseline_enu(&self) -> &V3 {
        &self.baseline_enu
    }
    pub fn carrier_hz(&self) -> f64 {
        self.carrier_hz
    }
    pub fn evaluate(
        &self,
        state: &V6,
        geometry: &StationGeometry,
        frequency_bias_hz: f64,
    ) -> Result<RfPrediction> {
        phase_partial(
            state,
            geometry.position(),
            geometry.velocity(),
            &geometry.rotation().to_gcrf(&self.baseline_enu),
            self.carrier_hz,
            frequency_bias_hz,
            self.spherical_phase,
        )
    }
    /// Exact spherical range difference, evaluated with a rationalized formula
    /// to avoid subtracting two large ranges. Default remains the reference's
    /// far-field phase model for compatibility.
    pub fn with_spherical_phase(mut self) -> Self {
        self.spherical_phase = true;
        self
    }
    pub fn uses_spherical_phase(self) -> bool {
        self.spherical_phase
    }

    /// Signed frequency difference f1 - f2 for the two antenna reference
    /// points. Baseline is r1-r2; both sites rotate with Earth.
    pub fn differential(
        &self,
        state: &V6,
        geometry: &StationGeometry,
        time: Instant,
    ) -> Result<DifferentialDopplerPrediction> {
        let baseline = geometry.rotation().to_gcrf(&self.baseline_enu);
        let omega = satkit::frametransform::qitrf2gcrf_approx(&time)
            * v3(0.0, 0.0, satkit::consts::OMEGA_EARTH);
        let mut predicted = differential_doppler_partial(
            state,
            geometry.position(),
            geometry.velocity(),
            &baseline,
            &omega,
            self.carrier_hz,
        )?;
        predicted.baseline_partial = predicted.baseline_partial * geometry.rotation().matrix();
        Ok(predicted)
    }

    pub fn corrected(self, correction_enu_m: &V3) -> Result<Self> {
        let mut corrected = Self::new(self.baseline_enu + correction_enu_m, self.carrier_hz)?;
        corrected.spherical_phase = self.spherical_phase;
        Ok(corrected)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct DifferentialDopplerPrediction {
    pub difference_hz: f64,
    /// Instantaneous derivative with respect to satellite GCRF state.
    pub partial: Matrix<f64, 1, 6>,
    /// Derivative with respect to baseline (GCRF in the raw kernel, ENU in
    /// Interferometer::differential).
    pub baseline_partial: Matrix<f64, 1, 3>,
}

/// Exact instantaneous geometric differential Doppler, f1-f2 =
/// (fc/c)(range_rate2-range_rate1). A common additive frequency offset cancels.
/// Earth rotation omega is GCRF radians/s; baseline is r1-r2 in GCRF metres.
pub fn differential_doppler_partial(
    state: &V6,
    station_position: &V3,
    station_velocity: &V3,
    baseline: &V3,
    omega: &V3,
    carrier_hz: f64,
) -> Result<DifferentialDopplerPrediction> {
    Interferometer::new(*baseline, carrier_hz)?;
    if !finite(omega) {
        return Err(KernelError::InvalidInput("Earth rotation must be finite"));
    }
    let (rr1, h1) = range_rate_partial(
        &r_of(state),
        &v_of(state),
        station_position,
        station_velocity,
    )?;
    let (rr2, h2) = range_rate_partial(
        &r_of(state),
        &v_of(state),
        &(station_position - baseline),
        &(station_velocity - omega.cross(baseline)),
    )?;
    let scale = carrier_hz / SPEED_OF_LIGHT;
    let mut baseline_partial = Matrix::zeros();
    for k in 0..3 {
        let mut axis = V3::zeros();
        axis[k] = 1.0;
        baseline_partial[(0, k)] =
            scale * (h2[(0, k)] + v3(h2[(0, 3)], h2[(0, 4)], h2[(0, 5)]).dot(&omega.cross(&axis)));
    }
    let difference_hz = scale * (rr2 - rr1);
    let partial = (h2 - h1) * scale;
    if !difference_hz.is_finite() || !finite(&partial) || !finite(&baseline_partial) {
        return Err(KernelError::Numerical(
            "differential Doppler exceeds numeric range",
        ));
    }
    Ok(DifferentialDopplerPrediction {
        difference_hz,
        partial,
        baseline_partial,
    })
}

/// Frequency is the Doppler shift relative to the nominal carrier (Hz), avoiding
/// subtraction of two GHz-sized values. Phase is continuous radians.
#[derive(Clone, Copy, Debug)]
pub struct RfPrediction {
    pub doppler_hz: f64,
    pub phase_rad: f64,
    /// Rows: Doppler, phase; columns: instantaneous GCRF position/velocity.
    pub partial: Matrix<f64, 2, 6>,
    pub phase_frequency_bias_partial: f64,
    pub phase_baseline_partial_gcrf: Matrix<f64, 1, 3>,
}

pub fn phase_difference_partial(
    state: &V6,
    station_position: &V3,
    station_velocity: &V3,
    baseline_gcrf: &V3,
    carrier_hz: f64,
    frequency_bias_hz: f64,
) -> Result<RfPrediction> {
    phase_partial(
        state,
        station_position,
        station_velocity,
        baseline_gcrf,
        carrier_hz,
        frequency_bias_hz,
        false,
    )
}
fn phase_partial(
    state: &V6,
    station_position: &V3,
    station_velocity: &V3,
    baseline_gcrf: &V3,
    carrier_hz: f64,
    frequency_bias_hz: f64,
    spherical: bool,
) -> Result<RfPrediction> {
    Interferometer::new(*baseline_gcrf, carrier_hz)?;
    if !frequency_bias_hz.is_finite() {
        return Err(KernelError::InvalidInput("frequency bias must be finite"));
    }
    let (rr, hrr) = range_rate_partial(
        &r_of(state),
        &v_of(state),
        station_position,
        station_velocity,
    )?;
    let displacement = r_of(state) - station_position;
    let range = displacement.norm();
    let los = displacement / range;
    let (path, dpath, baseline_derivative) = if spherical {
        let other = displacement + baseline_gcrf;
        let other_range = other.norm();
        if other_range == 0.0 {
            return Err(KernelError::ZeroRange);
        }
        if !other_range.is_finite() {
            return Err(KernelError::Numerical(
                "antenna 2 range exceeds numeric range",
            ));
        }
        let other_los = other / other_range;
        (
            (displacement * 2.0 + baseline_gcrf).dot(baseline_gcrf) / (range + other_range),
            other_los - los,
            other_los,
        )
    } else {
        let path = baseline_gcrf.dot(&los);
        (path, (baseline_gcrf - los * path) / range, los)
    };
    let doppler_hz = -carrier_hz / SPEED_OF_LIGHT * rr + frequency_bias_hz;
    let received_hz = carrier_hz + doppler_hz;
    let k = 2.0 * PI / SPEED_OF_LIGHT;
    let mut partial = Matrix::zeros();
    for j in 0..6 {
        partial[(0, j)] = -carrier_hz / SPEED_OF_LIGHT * hrr[(0, j)];
        partial[(1, j)] =
            k * (path * partial[(0, j)] + if j < 3 { received_hz * dpath[j] } else { 0.0 });
    }
    let phase_rad = k * received_hz * path;
    if !doppler_hz.is_finite() || !phase_rad.is_finite() || !finite(&partial) {
        return Err(KernelError::Numerical(
            "phase geometry exceeds numeric range",
        ));
    }
    Ok(RfPrediction {
        doppler_hz,
        phase_rad,
        partial,
        phase_frequency_bias_partial: k * path,
        phase_baseline_partial_gcrf: baseline_derivative.transpose() * (k * received_hz),
    })
}

/// Zero-mean stationary OU drift around a separately estimated hardware mean.
#[derive(Clone, Copy, Debug)]
pub struct OuPhaseNoise {
    tau_seconds: f64,
    stationary_sigma_rad: f64,
}

impl OuPhaseNoise {
    pub fn new(tau_seconds: f64, stationary_sigma_rad: f64) -> Result<Self> {
        if !tau_seconds.is_finite()
            || tau_seconds <= 0.0
            || !stationary_sigma_rad.is_finite()
            || stationary_sigma_rad <= 0.0
            || !(stationary_sigma_rad * stationary_sigma_rad).is_finite()
            || stationary_sigma_rad * stationary_sigma_rad == 0.0
        {
            return Err(KernelError::InvalidInput(
                "OU time and sigma must be finite and positive",
            ));
        }
        Ok(Self {
            tau_seconds,
            stationary_sigma_rad,
        })
    }
    pub fn tau_seconds(self) -> f64 {
        self.tau_seconds
    }
    pub fn stationary_sigma_rad(self) -> f64 {
        self.stationary_sigma_rad
    }
    pub fn transition(self, seconds: f64) -> Result<(f64, f64)> {
        if !seconds.is_finite() || seconds < 0.0 {
            return Err(KernelError::InvalidInput(
                "OU duration must be finite and nonnegative",
            ));
        }
        let x = seconds / self.tau_seconds;
        Ok((
            (-x).exp(),
            self.stationary_sigma_rad.powi(2) * -(-2.0 * x).exp_m1(),
        ))
    }
    pub fn covariance(self, times: &[Instant], white_sigma_rad: f64) -> Result<DynMatrix<f64>> {
        if !white_sigma_rad.is_finite()
            || white_sigma_rad <= 0.0
            || !white_sigma_rad.powi(2).is_finite()
            || white_sigma_rad.powi(2) == 0.0
        {
            return Err(KernelError::InvalidInput(
                "white phase sigma must be finite and positive",
            ));
        }
        let n = times.len();
        let mut c = DynMatrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                c[(i, j)] = self.stationary_sigma_rad.powi(2)
                    * (-((times[i] - times[j]).as_seconds()).abs() / self.tau_seconds).exp()
                    + if i == j { white_sigma_rad.powi(2) } else { 0.0 };
            }
        }
        Ok(c)
    }
}
