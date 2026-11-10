use super::{AcquisitionError, GroundStation, Result, positive, probability};
use crate::kernels::{V3, v3};
use satkit::{Duration, Instant};
use std::f64::consts::{FRAC_PI_2, TAU};

/// A normalized, finite, nonzero East–North–Up pointing.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EnuDirection(V3);
impl EnuDirection {
    pub fn new(vector: V3) -> Result<Self> {
        let scale = vector.iter().map(|x| x.abs()).fold(0.0, f64::max);
        if !vector.iter().all(|x| x.is_finite()) || scale == 0.0 {
            return Err(AcquisitionError::InvalidInput(
                "direction must be finite and nonzero",
            ));
        }
        let scaled = vector / scale;
        Ok(Self(scaled / scaled.norm()))
    }
    /// Azimuth eastward from north; elevation in [-π/2, π/2].
    pub fn from_az_el(azimuth_rad: f64, elevation_rad: f64) -> Result<Self> {
        if !azimuth_rad.is_finite()
            || !elevation_rad.is_finite()
            || !(-FRAC_PI_2..=FRAC_PI_2).contains(&elevation_rad)
        {
            return Err(AcquisitionError::InvalidInput("azimuth/elevation radians"));
        }
        let (sa, ca) = azimuth_rad.sin_cos();
        let (se, ce) = elevation_rad.sin_cos();
        Self::new(v3(ce * sa, ce * ca, se))
    }
    pub fn vector(&self) -> &V3 {
        &self.0
    }
    /// Azimuth in [0, 2π), defined as zero at zenith/nadir.
    pub fn az_el(&self) -> [f64; 2] {
        let horizontal = self.0[0].hypot(self.0[1]);
        let az = if horizontal < 1e-14 {
            0.0
        } else {
            self.0[0].atan2(self.0[1]).rem_euclid(TAU)
        };
        [az, self.0[2].atan2(horizontal)]
    }
    pub fn elevation_rad(&self) -> f64 {
        self.az_el()[1]
    }
    /// Stable angular separation, including tiny angles and antipodes.
    pub fn separation(&self, other: Self) -> f64 {
        self.0
            .cross(&other.0)
            .norm()
            .atan2(self.0.dot(&other.0).clamp(-1.0, 1.0))
    }
}

/// Independent configuration; validate once before preparing a controller.
#[derive(Clone, Copy, Debug)]
pub struct MpcConfig {
    pub search_duration_s: f64,
    pub move_s: f64,
    pub settle_s: f64,
    pub listen_s: f64,
    pub horizon: usize,
    pub beam_width: usize,
    pub candidate_count: usize,
    pub candidate_extent_sigma: f64,
    pub ess_threshold: f64,
}
impl Default for MpcConfig {
    fn default() -> Self {
        Self {
            search_duration_s: 192.0,
            move_s: 4.0,
            settle_s: 0.5,
            listen_s: 1.5,
            horizon: 4,
            beam_width: 40,
            candidate_count: 17,
            candidate_extent_sigma: 2.8,
            ess_threshold: 0.5,
        }
    }
}
impl MpcConfig {
    pub fn cycle_duration_s(&self) -> f64 {
        self.move_s + self.settle_s + self.listen_s
    }
    pub fn cycle_count(&self) -> Result<usize> {
        self.validate()?;
        Ok((self.search_duration_s / self.cycle_duration_s()).round() as usize)
    }
    pub fn validate(&self) -> Result<()> {
        self.validate_timing()?;
        if self.horizon == 0
            || self.beam_width == 0
            || self.candidate_count == 0
            || self.candidate_count.is_multiple_of(2)
        {
            return Err(AcquisitionError::InvalidInput(
                "positive horizon/beam width and odd candidate count required",
            ));
        }
        if !self.candidate_extent_sigma.is_finite() || self.candidate_extent_sigma < 0.0 {
            return Err(AcquisitionError::InvalidInput(
                "candidate extent must be finite and nonnegative",
            ));
        }
        probability(self.ess_threshold)
    }
    fn validate_timing(&self) -> Result<()> {
        positive(self.search_duration_s, "search duration")?;
        positive(self.move_s, "move duration")?;
        positive(self.listen_s, "listen duration")?;
        if !self.settle_s.is_finite() || self.settle_s < 0.0 {
            return Err(AcquisitionError::InvalidInput("settle duration"));
        }
        let duration = self.cycle_duration_s();
        let count = (self.search_duration_s / duration).round();
        if !duration.is_finite()
            || count < 1.0
            || count >= usize::MAX as f64
            || (count * duration - self.search_duration_s).abs() > 1e-8
        {
            return Err(AcquisitionError::InvalidInput(
                "search must contain complete cycles",
            ));
        }
        Ok(())
    }
}

/// Search action. Commands remain fixed through settling and listening.
#[derive(Clone, Copy, Debug)]
pub struct Action {
    pub cycle: usize,
    pub direction_enu: EnuDirection,
    pub listen_start: Instant,
    pub listen_end: Instant,
    pub candidate_index: Option<usize>,
}
impl Action {
    pub fn observation_time(&self) -> Instant {
        self.listen_start
            + Duration::from_seconds((self.listen_end - self.listen_start).as_seconds() / 2.0)
    }
    pub(crate) fn at(
        start: Instant,
        cycle: usize,
        direction_enu: EnuDirection,
        config: &MpcConfig,
    ) -> Self {
        let listen_start = start
            + Duration::from_seconds(
                cycle as f64 * config.cycle_duration_s() + config.move_s + config.settle_s,
            );
        Self {
            cycle,
            direction_enu,
            listen_start,
            listen_end: listen_start + Duration::from_seconds(config.listen_s),
            candidate_index: None,
        }
    }
}
#[derive(Debug)]
pub struct ActionPlan {
    pub actions: Vec<Action>,
    pub acquisition_probability: f64,
    pub retained_acquisition_probability: f64,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Observation {
    Miss,
    Acquired,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SearchStatus {
    Ready,
    AwaitingObservation,
    Acquired,
    Exhausted,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ObservationUpdate {
    pub resampled: bool,
    pub status: SearchStatus,
}

/// Geometry supplied to a caller's communications model.
#[derive(Clone, Copy, Debug)]
pub struct PointingGeometry {
    pub direction_enu: EnuDirection,
    pub range_m: f64,
    pub elevation_rad: f64,
    pub visible: bool,
}

/// Evaluate acquisition probability per current particle into `output`.
/// Models must overwrite every output, return probabilities in [0, 1], and
/// give invisible particles probability zero. Evaluation may repeat during
/// optimization; use a deterministic model within each plan.
pub trait DetectionModel {
    fn probabilities(
        &mut self,
        command: EnuDirection,
        particles: &[PointingGeometry],
        listen_s: f64,
        output: &mut [f64],
    ) -> Result<()>;
}

/// Contact and actuator settings. The antenna is already at the initial
/// direction when this search budget begins; omitted direction means nominal.
#[derive(Clone, Copy, Debug)]
pub struct SearchSetup {
    pub station: GroundStation,
    pub start: Instant,
    pub config: MpcConfig,
    pub antenna: super::AntennaLimits,
    pub elevation_mask_rad: f64,
    pub initial_direction: Option<EnuDirection>,
}
impl SearchSetup {
    pub fn new(station: GroundStation, start: Instant) -> Self {
        Self {
            station,
            start,
            config: MpcConfig::default(),
            antenna: super::AntennaLimits::default(),
            elevation_mask_rad: 10_f64.to_radians(),
            initial_direction: None,
        }
    }
}
