//! Caller-clocked behavioral antenna and receiver kernels.
//!
//! These models are development fixtures, not calibrated hardware models.
//! The caller owns scheduling, API transport, observation identity and truth
//! isolation. Angles use radians; frequencies use Hz; time uses seconds.

use crate::{
    detection::{DetectionModel, EnuDirection, PointingGeometry},
    kernels::{
        KernelError, PreparedGeometry, Result, SPEED_OF_LIGHT, propagate_sgp4_gcrf, r_of,
        range_rate_partial, v_of, wrap_pi,
    },
    satkit::{Duration, Instant, TLE, sgp4::SGP4Source},
};

/// Finite, independent azimuth/elevation actuator limits.
#[derive(Clone, Copy, Debug)]
pub struct MountConfig {
    /// Maximum axis speed in radians/second.
    pub rate_rad_s: f64,
    /// Maximum axis acceleration in radians/second².
    pub acceleration_rad_s2: f64,
    /// Lower physical elevation limit in radians.
    pub min_elevation_rad: f64,
    /// Upper physical elevation limit in radians.
    pub max_elevation_rad: f64,
}
impl Default for MountConfig {
    fn default() -> Self {
        Self {
            rate_rad_s: 6.0_f64.to_radians(),
            acceleration_rad_s2: 10.0_f64.to_radians(),
            min_elevation_rad: 0.0,
            max_elevation_rad: std::f64::consts::FRAC_PI_2,
        }
    }
}
impl MountConfig {
    /// Validate speed, acceleration and ordered physical elevation limits.
    pub fn validate(&self) -> Result<()> {
        positive(self.rate_rad_s)?;
        positive(self.acceleration_rad_s2)?;
        if !self.min_elevation_rad.is_finite()
            || !self.max_elevation_rad.is_finite()
            || self.min_elevation_rad < 0.0
            || self.max_elevation_rad > std::f64::consts::FRAC_PI_2
            || self.min_elevation_rad >= self.max_elevation_rad
        {
            return Err(KernelError::InvalidInput("physical elevation limits"));
        }
        Ok(())
    }
}

/// Rate/acceleration-limited mount. Azimuth uses the shortest wrapped path.
#[derive(Clone, Copy, Debug)]
pub struct Mount {
    angles: [f64; 2],
    velocity: [f64; 2],
    config: MountConfig,
}
impl Mount {
    /// Initialize the measured aperture position, clamping physical elevation.
    pub fn new(angles: [f64; 2], config: MountConfig) -> Result<Self> {
        config.validate()?;
        finite(&angles)?;
        Ok(Self {
            angles: [
                angles[0].rem_euclid(std::f64::consts::TAU),
                angles[1].clamp(config.min_elevation_rad, config.max_elevation_rad),
            ],
            velocity: [0.0; 2],
            config,
        })
    }
    /// Measured aperture azimuth/elevation, never spacecraft truth.
    pub fn angles(&self) -> [f64; 2] {
        self.angles
    }
    /// Advance one fixed step towards a target. Invalid inputs preserve state.
    pub fn advance(&mut self, target: [f64; 2], seconds: f64) -> Result<()> {
        finite(&target)?;
        positive(seconds)?;
        let target = [
            self.angles[0] + wrap_pi(target[0] - self.angles[0]),
            target[1].clamp(self.config.min_elevation_rad, self.config.max_elevation_rad),
        ];
        for (axis, goal) in target.into_iter().enumerate() {
            let error = goal - self.angles[axis];
            let desired = error.signum()
                * (2.0 * self.config.acceleration_rad_s2 * error.abs())
                    .sqrt()
                    .min(self.config.rate_rad_s);
            let change = self.config.acceleration_rad_s2 * seconds;
            let velocity =
                self.velocity[axis] + (desired - self.velocity[axis]).clamp(-change, change);
            let movement = velocity * seconds;
            if movement.abs() >= error.abs() && movement * error >= 0.0 {
                self.angles[axis] = goal;
                self.velocity[axis] = 0.0;
            } else {
                self.angles[axis] += movement;
                self.velocity[axis] = velocity;
            }
        }
        self.angles[0] = self.angles[0].rem_euclid(std::f64::consts::TAU);
        self.angles[1] =
            self.angles[1].clamp(self.config.min_elevation_rad, self.config.max_elevation_rad);
        Ok(())
    }
}

/// Gaussian power beam with hard angular exclusion and receiver hysteresis.
#[derive(Clone, Copy, Debug)]
pub struct ReceiverConfig {
    /// Full width at half maximum of the power beam, radians.
    pub beam_fwhm_rad: f64,
    /// Hard maximum angular separation for any measurement, radians.
    pub cutoff_rad: f64,
    /// On-axis Eb/N0 in dB at a range of 1000 km.
    pub on_axis_ebn0_db: f64,
    /// Minimum Eb/N0 for acquisition, dB.
    pub acquire_ebn0_db: f64,
    /// Minimum Eb/N0 to retain lock, dB.
    pub hold_ebn0_db: f64,
    /// Consecutive valid simulated seconds required to acquire.
    pub acquisition_s: f64,
    /// Gaussian Doppler standard deviation in Hz.
    pub sigma_doppler_hz: f64,
    /// Additive receiver frequency bias in Hz.
    pub bias_hz: f64,
}
impl Default for ReceiverConfig {
    fn default() -> Self {
        Self {
            beam_fwhm_rad: 3.0_f64.to_radians(),
            cutoff_rad: 3.0_f64.to_radians(),
            on_axis_ebn0_db: 20.0,
            acquire_ebn0_db: 5.0,
            hold_ebn0_db: 3.0,
            acquisition_s: 0.3,
            sigma_doppler_hz: 1.0,
            bias_hz: 0.0,
        }
    }
}
impl ReceiverConfig {
    /// Validate all physical receiver parameters.
    pub fn validate(&self) -> Result<()> {
        finite(&[
            self.on_axis_ebn0_db,
            self.acquire_ebn0_db,
            self.hold_ebn0_db,
            self.bias_hz,
        ])?;
        for value in [
            self.beam_fwhm_rad,
            self.cutoff_rad,
            self.acquisition_s,
            self.sigma_doppler_hz,
        ] {
            positive(value)?;
        }
        if self.cutoff_rad > std::f64::consts::PI || self.hold_ebn0_db > self.acquire_ebn0_db {
            return Err(KernelError::InvalidInput("receiver cutoff or hysteresis"));
        }
        Ok(())
    }
    /// Beam/link power. Returns no RF measurement outside visibility/cutoff.
    pub fn ebn0_db(&self, separation_rad: f64, range_m: f64, visible: bool) -> Result<Option<f64>> {
        self.validate()?;
        finite(&[separation_rad])?;
        positive(range_m)?;
        if separation_rad < 0.0 {
            return Err(KernelError::InvalidInput("negative separation"));
        }
        if !visible || separation_rad > self.cutoff_rad + 1e-12 {
            return Ok(None);
        }
        let attenuation = 40.0 * 2.0_f64.log10() * (separation_rad / self.beam_fwhm_rad).powi(2);
        Ok(Some(
            self.on_axis_ebn0_db - 20.0 * (range_m / 1e6).log10() - attenuation,
        ))
    }
}
impl DetectionModel for ReceiverConfig {
    fn probabilities(
        &mut self,
        command: EnuDirection,
        particles: &[PointingGeometry],
        listen_s: f64,
        output: &mut [f64],
    ) -> crate::detection::Result<()> {
        if output.len() != particles.len() || !listen_s.is_finite() || listen_s <= 0.0 {
            return Err(crate::detection::AcquisitionError::InvalidInput(
                "receiver probability inputs",
            ));
        }
        for (point, result) in particles.iter().zip(output) {
            let ebn0 = self.ebn0_db(
                command.separation(point.direction_enu),
                point.range_m,
                point.visible,
            )?;
            *result = match ebn0 {
                Some(value) if value >= self.acquire_ebn0_db && listen_s >= self.acquisition_s => {
                    0.95
                }
                _ => 0.0,
            };
        }
        Ok(())
    }
}

/// One receiver sample. Unlocked samples contain no fabricated frequency.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReceiverSample {
    /// Current carrier lock.
    pub locked: bool,
    /// Valid received Doppler in Hz, including bias/noise.
    pub doppler_hz: Option<f64>,
    /// Received Eb/N0, dB; absent when outside reception geometry.
    pub ebn0_db: Option<f64>,
}

/// Stable seeded SplitMix64 stream for receiver noise and caller resampling.
#[derive(Clone, Copy, Debug)]
pub struct SeededNoise(u64);
impl SeededNoise {
    /// Create an independent reproducible stream.
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }
    /// Uniform draw in [0, 1).
    pub fn uniform(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        ((value ^ (value >> 31)) >> 11) as f64 * (1.0 / ((1_u64 << 53) as f64))
    }
    /// Standard normal draw via Box–Muller, with no zero logarithm.
    pub fn normal(&mut self) -> f64 {
        let radius = (-2.0 * self.uniform().max(f64::MIN_POSITIVE).ln()).sqrt();
        radius * (std::f64::consts::TAU * self.uniform()).cos()
    }
}

/// Hysteretic receiver. Carrier exclusion immediately clears lock and dwell.
#[derive(Clone, Copy, Debug)]
pub struct Receiver {
    config: ReceiverConfig,
    noise: SeededNoise,
    locked: bool,
    dwell_s: f64,
}
impl Receiver {
    /// Validate configuration and initialize unlocked.
    pub fn new(config: ReceiverConfig, seed: u64) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            config,
            noise: SeededNoise::new(seed),
            locked: false,
            dwell_s: 0.0,
        })
    }
    /// Advance acquisition/lock and draw one frequency sample. Invalid inputs
    /// preserve receiver state and its random stream.
    pub fn sample(
        &mut self,
        separation_rad: f64,
        range_m: f64,
        visible: bool,
        doppler_hz: f64,
        seconds: f64,
    ) -> Result<ReceiverSample> {
        positive(seconds)?;
        finite(&[doppler_hz])?;
        let ebn0_db = self.config.ebn0_db(separation_rad, range_m, visible)?;
        let threshold = if self.locked {
            self.config.hold_ebn0_db
        } else {
            self.config.acquire_ebn0_db
        };
        if ebn0_db.is_none_or(|value| value < threshold) {
            self.locked = false;
            self.dwell_s = 0.0;
        } else {
            self.dwell_s += seconds;
            self.locked |= self.dwell_s + 1e-12 >= self.config.acquisition_s;
        }
        let doppler_hz = self.locked.then(|| {
            doppler_hz + self.config.bias_hz + self.config.sigma_doppler_hz * self.noise.normal()
        });
        Ok(ReceiverSample {
            locked: self.locked,
            doppler_hz,
            ebn0_db,
        })
    }
}

/// Actual-epoch topocentric geometry and Doppler from a supplied orbit.
#[derive(Clone, Copy, Debug)]
pub struct OrbitalSample {
    /// Topocentric direction/range/visibility.
    pub pointing: PointingGeometry,
    /// Geometric Doppler, negative for recession, Hz.
    pub doppler_hz: f64,
}
/// Propagate once and evaluate both pointing and range-rate at the given UTC
/// epoch. This routine does not apply antenna controls or receiver noise.
pub fn orbital_sample(
    tle: &mut TLE,
    time: Instant,
    station: crate::kernels::GroundStation,
    carrier_hz: f64,
) -> Result<OrbitalSample> {
    positive(carrier_hz)?;
    let state = propagate_sgp4_gcrf(tle, &[time])?[0];
    let geometry = PreparedGeometry::new(&[time], &[station])?;
    let row = geometry
        .row(0)
        .ok_or(KernelError::InvalidInput("station row"))?;
    let pointing = crate::detection::pointing_geometry(&r_of(&state), row, 0.0)
        .map_err(|_| KernelError::Numerical("simulation pointing geometry"))?;
    let (range_rate, _) =
        range_rate_partial(&r_of(&state), &v_of(&state), row.position(), row.velocity())?;
    Ok(OrbitalSample {
        pointing,
        doppler_hz: -carrier_hz * range_rate / SPEED_OF_LIGHT,
    })
}

/// Copy a TLE with a changed element epoch and invalidated propagation cache.
/// Observation epochs remain unchanged. This is not an ACU time-ahead control.
pub fn shifted_epoch(prior: &TLE, seconds: f64) -> Result<TLE> {
    finite(&[seconds])?;
    // Satkit owns cached mean elements; an epoch hypothesis needs an isolated copy.
    #[allow(clippy::disallowed_methods)]
    let mut result = prior.clone();
    result.epoch = prior
        .epoch
        .checked_add(Duration::from_seconds(seconds))
        .ok_or(KernelError::InvalidInput("shifted element epoch"))?;
    *result.satrec_mut() = None;
    Ok(result)
}

fn positive(value: f64) -> Result<()> {
    if !value.is_finite() || value <= 0.0 {
        return Err(KernelError::InvalidInput(
            "positive finite simulation parameter",
        ));
    }
    Ok(())
}
fn finite(values: &[f64]) -> Result<()> {
    if !values.iter().all(|value| value.is_finite()) {
        return Err(KernelError::InvalidInput("finite simulation parameter"));
    }
    Ok(())
}
