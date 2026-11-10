//! Small acquisition fixture shared by the example and integration tests.
#![allow(dead_code)]

use tychonic::{
    detection::{
        AcquisitionError, Action, BcMpc, DetectionModel, EnuDirection, GroundStation, Observation,
        ParticleBelief, PointingGeometry, Result, SearchSetup, SearchStatus,
        propagate_pointing_grid,
    },
    kernels::{KernelError, PreparedGeometry, propagate_sgp4_gcrf, r_of},
    satkit::{Duration, ITRFCoord, Instant, TLE, frametransform},
};

pub const RECEIVER_SEED: u64 = 42;
pub const RESAMPLING_SEED: u64 = 43;
pub const BEAM_FWHM_RAD: f64 = 2.0 * std::f64::consts::PI / 180.0;

// SplitMix64, as used by benches/support: stable across dependency versions.
struct Rng(u64);
impl Rng {
    fn unit(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
        (z >> 11) as f64 * (1.0 / ((1_u64 << 53) as f64))
    }
}

pub fn tle() -> Result<TLE> {
    TLE::load_2line(
        "1 25544U 98067A   24356.58519896  .00014389  00000-0  25222-3 0  9992",
        "2 25544  51.6403 106.8969 0007877   6.1421 113.2479 15.50801739487617",
    )
    .map_err(|error| KernelError::Tle(error.to_string()).into())
}

pub fn station_and_start() -> Result<(GroundStation, Instant)> {
    let mut orbit = tle()?;
    let center = orbit.epoch + Duration::from_seconds(600.0);
    let states = propagate_sgp4_gcrf(&mut orbit, &[center])?;
    let local = frametransform::qitrf2gcrf(&center).conjugate() * r_of(&states[0]);
    let (lat, lon, _) = ITRFCoord::from_vector(&local).to_geodetic_rad();
    Ok((
        GroundStation {
            lat_deg: lat.to_degrees() + 2.5,
            lon_deg: lon.to_degrees(),
            alt_m: 0.0,
        },
        center - Duration::from_seconds(96.0),
    ))
}

/// Python prototype's Gaussian beam and explicit on-axis range/elevation model.
pub struct GaussianReceiver;
impl DetectionModel for GaussianReceiver {
    fn probabilities(
        &mut self,
        command: EnuDirection,
        particles: &[PointingGeometry],
        listen_s: f64,
        output: &mut [f64],
    ) -> Result<()> {
        for (point, probability) in particles.iter().zip(output) {
            let error = command.separation(point.direction_enu);
            let beam = (-4.0 * 2_f64.ln() * (error / BEAM_FWHM_RAD).powi(2)).exp();
            let p_second = if point.visible {
                0.8 * (1e6 / point.range_m).powi(2).min(1.0) * point.elevation_rad.sin() * beam
            } else {
                0.0
            };
            *probability = -(listen_s * (-p_second).ln_1p()).exp_m1();
        }
        Ok(())
    }
}

#[derive(Debug, PartialEq)]
pub struct StepRecord {
    pub cycle: usize,
    pub direction_enu: EnuDirection,
    pub listen_start: Instant,
    pub listen_end: Instant,
    pub elapsed_s: f64,
    pub pointing_error_rad: f64,
    pub true_visible: bool,
    pub detection_probability: f64,
    pub receiver_draw: f64,
    pub slew_time_s: f64,
    pub predicted_acquisition_probability: f64,
    pub retained_acquisition_probability: f64,
    pub resampled: bool,
    pub observation: Observation,
}

#[derive(Debug)]
pub struct SimulationResult {
    pub setup: SearchSetup,
    pub initial_direction: EnuDirection,
    pub initial_pointing_error_rad: f64,
    pub status: SearchStatus,
    pub controller_cycles: usize,
    pub plant_cycles: usize,
    pub plant_time: Instant,
    pub records: Vec<StepRecord>,
}

/// Ideal rest-to-rest mount and stochastic receiver. Truth never enters planning.
pub struct SimulatedPlant {
    truth: TLE,
    setup: SearchSetup,
    direction: EnuDirection,
    cycle: usize,
    time: Instant,
    rng: Rng,
}
impl SimulatedPlant {
    pub fn new(truth: TLE, setup: SearchSetup, direction: EnuDirection) -> Self {
        Self {
            truth,
            setup,
            direction,
            cycle: 0,
            time: setup.start,
            rng: Rng(RECEIVER_SEED),
        }
    }

    fn truth_at(&mut self, time: Instant) -> Result<PointingGeometry> {
        let geometry = PreparedGeometry::new(&[time], &[self.setup.station])?;
        Ok(propagate_pointing_grid(&mut self.truth, &geometry, self.setup.elevation_mask_rad)?[0])
    }

    fn validate_timing(&self, action: Action) -> Result<()> {
        if self.cycle >= self.setup.config.cycle_count()? || action.cycle != self.cycle {
            return Err(AcquisitionError::InvalidState(
                "plant requires the next cycle within its search budget",
            ));
        }
        let config = self.setup.config;
        let expected_start = self.setup.start
            + Duration::from_seconds(
                self.cycle as f64 * config.cycle_duration_s() + config.move_s + config.settle_s,
            );
        let expected_end = expected_start + Duration::from_seconds(config.listen_s);
        if action.listen_start != expected_start || action.listen_end != expected_end {
            return Err(AcquisitionError::InvalidInput("plant listen timing"));
        }
        Ok(())
    }

    fn validate_motion(&self, action: Action) -> Result<f64> {
        if action.direction_enu.elevation_rad() < self.setup.elevation_mask_rad {
            return Err(AcquisitionError::InvalidInput("plant pointing below mask"));
        }
        let slew_s = self
            .setup
            .antenna
            .slew_time_s(self.direction, action.direction_enu)?;
        if slew_s > self.setup.config.move_s + 1e-8 {
            return Err(AcquisitionError::InvalidInput("plant infeasible slew"));
        }
        Ok(slew_s)
    }

    pub fn execute(
        &mut self,
        action: Action,
        receiver: &mut impl DetectionModel,
    ) -> Result<StepRecord> {
        self.validate_timing(action)?;
        let slew_time_s = self.validate_motion(action)?;
        let truth = self.truth_at(action.observation_time())?;
        // NaN catches a model that succeeds without writing its output.
        let mut probability = [f64::NAN];
        receiver.probabilities(
            action.direction_enu,
            &[truth],
            self.setup.config.listen_s,
            &mut probability,
        )?;
        if !probability[0].is_finite() || !(0.0..=1.0).contains(&probability[0]) {
            return Err(AcquisitionError::Detection(
                "invalid plant probability".into(),
            ));
        }
        let receiver_draw = self.rng.unit();
        let observation = if receiver_draw < probability[0] {
            Observation::Acquired
        } else {
            Observation::Miss
        };
        self.direction = action.direction_enu;
        self.time = action.listen_end;
        self.cycle += 1;
        Ok(StepRecord {
            cycle: action.cycle,
            direction_enu: action.direction_enu,
            listen_start: action.listen_start,
            listen_end: action.listen_end,
            elapsed_s: (self.time - self.setup.start).as_seconds(),
            pointing_error_rad: action.direction_enu.separation(truth.direction_enu),
            true_visible: truth.visible,
            detection_probability: probability[0],
            receiver_draw,
            slew_time_s,
            predicted_acquisition_probability: 0.0,
            retained_acquisition_probability: 0.0,
            resampled: false,
            observation,
        })
    }
}

pub fn scenario() -> Result<(BcMpc, SimulatedPlant)> {
    let (station, start) = station_and_start()?;
    let setup = SearchSetup::new(station, start);
    let mut planning = tle()?;
    // Mean anomaly is in degrees; mean motion is revolutions per day.
    planning.mean_anomaly += 10.0 * planning.mean_motion / 240.0;
    let particles = [-30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0]
        .into_iter()
        .map(|phase| {
            // Fresh TLEs avoid reusing an SGP4 cache after changing elements.
            let mut orbit = tle()?;
            orbit.mean_anomaly += (10.0 + phase) * orbit.mean_motion / 240.0;
            Ok(orbit.into())
        })
        .collect::<Result<Vec<_>>>()?;
    let belief = ParticleBelief::new(particles, vec![1.0; 7])?;
    let controller = BcMpc::new(planning.into(), belief, setup)?;
    let plant = SimulatedPlant::new(tle()?, setup, controller.direction());
    Ok((controller, plant))
}

pub fn run_simulation(mut receiver: impl DetectionModel) -> Result<SimulationResult> {
    let (mut controller, mut plant) = scenario()?;
    let initial_direction = controller.direction();
    let initial_pointing_error_rad =
        initial_direction.separation(plant.truth_at(plant.setup.start)?.direction_enu);
    let mut resampling = Rng(RESAMPLING_SEED);
    let mut records = Vec::new();
    while !controller.finished() {
        let plan = controller.plan(&mut receiver)?;
        // Execute only the first action, then update and replan on feedback.
        let mut record = plant.execute(plan.actions[0], &mut receiver)?;
        record.predicted_acquisition_probability = plan.acquisition_probability;
        record.retained_acquisition_probability = plan.retained_acquisition_probability;
        record.resampled = controller
            .observe(record.observation, || resampling.unit())?
            .resampled;
        records.push(record);
    }
    Ok(SimulationResult {
        setup: plant.setup,
        initial_direction,
        initial_pointing_error_rad,
        status: controller.status(),
        controller_cycles: controller.cycle(),
        plant_cycles: plant.cycle,
        plant_time: plant.time,
        records,
    })
}
