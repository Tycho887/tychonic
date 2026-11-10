#![allow(clippy::disallowed_methods)]
#[path = "../examples/support/acquisition.rs"]
mod simulation;

use simulation::{BEAM_FWHM_RAD, GaussianReceiver, SimulationResult, run_simulation};
use tychonic::{
    detection::*,
    kernels::{self, PreparedGeometry, V3, v3},
    satkit::{
        Duration, Frame, ITRFCoord, Instant, TLE, frametransform,
        omm::OMM,
        sgp4::{SGP4Error, sgp4},
    },
};

fn close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "{actual} != {expected}"
    );
}
fn direction(az_deg: f64, el_deg: f64) -> EnuDirection {
    EnuDirection::from_az_el(az_deg.to_radians(), el_deg.to_radians()).unwrap()
}
fn tle() -> TLE {
    simulation::tle().unwrap()
}
fn belief(weights: Vec<f64>) -> ParticleBelief {
    ParticleBelief::new((0..weights.len()).map(|_| tle().into()).collect(), weights).unwrap()
}

#[test]
fn normalized_weights_miss_updates_and_ess() {
    let mut belief = belief(vec![0.2, 0.3, 0.5]);
    belief.update_miss(&[0.1, 0.5, 1.0]).unwrap();
    for (&actual, expected) in belief
        .weights()
        .iter()
        .zip([0.02 / 0.67, 0.15 / 0.67, 0.5 / 0.67])
    {
        close(actual, expected, 1e-14);
    }
    let before = belief.weights().to_vec();
    belief.update_miss(&[1.0; 3]).unwrap();
    for (&a, b) in belief.weights().iter().zip(before) {
        close(a, b, 1e-14);
    }
    let expected_ess = 1.0 / belief.weights().iter().map(|w| w * w).sum::<f64>();
    close(belief.effective_sample_size(), expected_ess, 1e-14);
    let mut extreme = self::belief(vec![1e308; 3]);
    extreme.update_miss(&[1e-320, 2e-320, 3e-320]).unwrap();
    for (&a, b) in extreme
        .weights()
        .iter()
        .zip([1.0 / 6.0, 2.0 / 6.0, 3.0 / 6.0])
    {
        close(a, b, 1e-12);
    }
}

#[test]
fn invalid_beliefs_and_updates_do_not_change_state() {
    for weights in [
        vec![],
        vec![0.0; 3],
        vec![-1.0, 1.0, 1.0],
        vec![f64::NAN; 3],
        vec![f64::INFINITY; 3],
    ] {
        assert!(
            ParticleBelief::new((0..weights.len()).map(|_| tle().into()).collect(), weights)
                .is_err()
        );
    }
    assert!(ParticleBelief::new(vec![tle().into()], vec![1.0, 1.0]).is_err());
    let mut belief = belief(vec![1.0, 0.0, 0.0]);
    for misses in [
        vec![0.0, 1.0, 1.0],
        vec![-1.0; 3],
        vec![2.0; 3],
        vec![f64::NAN; 3],
        vec![0.5],
    ] {
        assert!(belief.update_miss(&misses).is_err());
        assert_eq!(belief.weights(), &[1.0, 0.0, 0.0]);
        assert_eq!(belief.indices(), &[0, 1, 2]);
    }
}

#[test]
fn systematic_resampling_preserves_original_pool_and_handles_zero_weights() {
    assert_eq!(
        systematic_resampling_indices(&[0.0, 1.0, 0.0], 0.0).unwrap(),
        vec![1, 1, 1]
    );
    assert_eq!(
        systematic_resampling_indices(&[0.7, 0.3, 0.0], 0.2).unwrap(),
        vec![0, 0, 1]
    );
    let mut belief = belief(vec![0.7, 0.3, 0.0]);
    let pool = belief.particles().as_ptr();
    assert!(belief.resample_if_needed(1.0, || 0.2).unwrap());
    assert_eq!(belief.indices(), &[0, 0, 1]);
    assert_eq!(belief.particles().as_ptr(), pool);
    assert_eq!(belief.weights(), &[1.0 / 3.0; 3]);
    assert!(
        !belief
            .resample_if_needed(0.5, || panic!("no draw above ESS threshold"))
            .unwrap()
    );
    for draw in [-0.1, 1.0, f64::NAN] {
        assert!(systematic_resampling_indices(belief.weights(), draw).is_err());
    }
    for threshold in [-0.1, 1.1, f64::NAN] {
        assert!(belief.resample_if_needed(threshold, || 0.0).is_err());
    }
}

#[test]
fn direction_conventions_and_slew_profiles() {
    let north = direction(0.0, 0.0);
    let east = direction(90.0, 0.0);
    close(north.separation(east), std::f64::consts::FRAC_PI_2, 1e-14);
    assert_eq!(north.vector(), &v3(0.0, 1.0, 0.0));
    close(direction(-10.0, 25.0).az_el()[0].to_degrees(), 350.0, 1e-12);
    assert_eq!(direction(100.0, 90.0).az_el()[0], 0.0);
    assert!(EnuDirection::new(V3::zeros()).is_err());
    assert!(EnuDirection::new(v3(f64::NAN, 0.0, 1.0)).is_err());
    assert!(EnuDirection::from_az_el(0.0, 2.0).is_err());
    close(
        EnuDirection::new(v3(1e308, 1e308, 0.0))
            .unwrap()
            .vector()
            .norm(),
        1.0,
        1e-14,
    );
    let antenna = AntennaLimits {
        rate_rad_s: 2.0,
        acceleration_rad_s2: 4.0,
    };
    close(
        antenna
            .slew_time_s(north, direction(0.5_f64.to_degrees(), 0.0))
            .unwrap(),
        2.0 * (0.5_f64 / 4.0).sqrt(),
        1e-14,
    );
    close(
        antenna.slew_time_s(north, east).unwrap(),
        std::f64::consts::FRAC_PI_2 / 2.0 + 0.5,
        1e-14,
    );
    let paper = AntennaLimits::default();
    let partial = paper.partial_slew(north, east, 4.0).unwrap();
    close(
        north.separation(partial).to_degrees(),
        1.10 * (4.0 - 1.10),
        1e-12,
    );
    assert!(paper.reachable(north, partial, 4.0).unwrap());
    assert!(!paper.reachable(north, east, 4.0).unwrap());
    assert_eq!(paper.partial_slew(north, east, 0.0).unwrap(), north);
    let antipode = paper
        .partial_slew(north, direction(180.0, 0.0), 4.0)
        .unwrap();
    assert!(paper.reachable(north, antipode, 4.0).unwrap());
    assert!(paper.reachable(north, east, -1.0).is_err());
}

#[test]
fn candidate_spread_is_unweighted_population_spread_with_degenerate_fallback() {
    let center = direction(0.0, 45.0);
    let tangent = v3(1.0, 0.0, 0.0);
    let points: Vec<_> = [-0.1_f64, 0.0, 0.1]
        .into_iter()
        .map(|angle| {
            EnuDirection::new(center.vector() * angle.cos() + tangent * angle.sin()).unwrap()
        })
        .collect();
    let candidates = along_track_candidates(center, &points, tangent, 3, 2.8).unwrap();
    close(
        center.separation(candidates[2]),
        2.8 * (0.02_f64 / 3.0).sqrt(),
        1e-14,
    );
    assert_eq!(candidates[1], center);
    assert!(candidates[0].vector()[0] < 0.0 && candidates[2].vector()[0] > 0.0);
    let fallback = along_track_candidates(center, &points, V3::zeros(), 3, 2.8).unwrap();
    close(
        center.separation(fallback[2]),
        center.separation(candidates[2]),
        1e-14,
    );
    assert_eq!(
        along_track_candidates(center, &[center; 3], tangent, 5, 2.8).unwrap(),
        vec![center; 5]
    );
    assert_eq!(
        along_track_candidates(center, &points, tangent, 1, 2.8).unwrap(),
        vec![center]
    );
    assert!(along_track_candidates(center, &points, tangent, 2, 2.8).is_err());
    assert!(along_track_candidates(center, &points, tangent, 3, 100.0).is_err());
    let opposite = EnuDirection::new(-center.vector()).unwrap();
    assert!(along_track_candidates(center, &[opposite], tangent, 3, 2.8).is_err());
}

struct TableModel {
    candidates: Vec<EnuDirection>,
    table: Vec<Vec<Vec<f64>>>,
}
impl DetectionModel for TableModel {
    fn probabilities(
        &mut self,
        command: EnuDirection,
        particles: &[PointingGeometry],
        _: f64,
        output: &mut [f64],
    ) -> Result<()> {
        let candidate = self
            .candidates
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| command.separation(**a).total_cmp(&command.separation(**b)))
            .unwrap()
            .0;
        for (geometry, value) in particles.iter().zip(output) {
            let encoded = geometry.range_m as usize - 1;
            *value = self.table[encoded / 100][candidate][encoded % 100];
        }
        Ok(())
    }
}
fn synthetic(
    candidates: &[EnuDirection],
    count: usize,
    particles: usize,
    retained_indices: &[usize],
) -> (Vec<PlanningCycle>, Vec<Action>) {
    let start = Instant::from_unixtime(1704110400.0);
    let mut cycles = Vec::new();
    let mut actions = Vec::new();
    for (i, &index) in retained_indices.iter().enumerate().take(count) {
        let timing = Action {
            cycle: i,
            direction_enu: candidates[index],
            listen_start: start + Duration::from_seconds(i as f64 * 6.0 + 4.5),
            listen_end: start + Duration::from_seconds(i as f64 * 6.0 + 6.0),
            candidate_index: Some(index),
        };
        cycles.push(PlanningCycle {
            timing,
            candidates: candidates.to_vec(),
            particles: (0..particles)
                .map(|p| PointingGeometry {
                    direction_enu: candidates[1],
                    range_m: (i * 100 + p + 1) as f64,
                    elevation_rad: candidates[1].elevation_rad(),
                    visible: true,
                })
                .collect(),
        });
        actions.push(timing);
    }
    (cycles, actions)
}
fn settings(horizon: usize, beam_width: usize) -> PlannerSettings {
    PlannerSettings {
        horizon,
        beam_width,
        move_s: 4.0,
        listen_s: 1.5,
        elevation_mask_rad: 0.0,
        antenna: AntennaLimits {
            rate_rad_s: 100.0,
            acceleration_rad_s2: 100.0,
        },
    }
}

#[test]
fn full_contact_objective_matches_exhaustive_search_and_python_reference() {
    let candidates: Vec<_> = [-10.0, 0.0, 10.0]
        .into_iter()
        .map(|az| direction(az, 45.0))
        .collect();
    let mut table = vec![vec![vec![0.0; 3]; 3]; 3];
    table[0][0] = vec![0.9, 0.0, 0.0];
    table[0][2] = vec![0.0, 0.8, 0.8];
    table[2][1] = vec![0.95, 0.0, 0.0];
    let weights = [0.6, 0.2, 0.2];
    let mut exhaustive = 0.0_f64;
    for a in 0..3 {
        for b in 0..3 {
            let survival: f64 = (0..3)
                .map(|i| {
                    weights[i]
                        * (1.0 - table[0][a][i])
                        * (1.0 - table[1][b][i])
                        * (1.0 - table[2][1][i])
                })
                .sum();
            exhaustive = exhaustive.max(1.0 - survival);
        }
    }
    let (cycles, retained) = synthetic(&candidates, 3, 3, &[1, 1, 1]);
    let mut model = TableModel {
        candidates: candidates.to_vec(),
        table,
    };
    let plan = optimize_prefix(
        candidates[1],
        &retained,
        &cycles,
        &weights,
        &settings(2, 100),
        &mut model,
    )
    .unwrap();
    close(plan.acquisition_probability, exhaustive, 1e-14);
    close(plan.acquisition_probability, 0.89, 1e-14);
    assert_eq!(plan.actions[0].candidate_index, Some(2));
    assert_eq!(plan.actions[2].direction_enu, retained[2].direction_enu);
    assert!(plan.acquisition_probability > plan.retained_acquisition_probability);
}

#[test]
fn continuation_boundary_and_incumbent_are_preserved() {
    let candidates: Vec<_> = [0.0, 2.5, 5.0]
        .into_iter()
        .map(|az| direction(az, 45.0))
        .collect();
    let (cycles, retained) = synthetic(&candidates, 2, 1, &[1, 2]);
    let mut model = TableModel {
        candidates: candidates.to_vec(),
        table: vec![
            vec![vec![0.9], vec![0.1], vec![0.0]],
            vec![vec![0.0], vec![0.0], vec![0.8]],
        ],
    };
    let mut constrained = settings(1, 100);
    constrained.antenna = AntennaLimits::default();
    let plan = optimize_prefix(
        candidates[1],
        &retained,
        &cycles,
        &[1.0],
        &constrained,
        &mut model,
    )
    .unwrap();
    close(plan.acquisition_probability, 0.82, 1e-14);
    assert_ne!(plan.actions[0].direction_enu, candidates[0]);
    assert_eq!(plan.actions[1].direction_enu, candidates[2]);

    let candidates: Vec<_> = [-10.0, 0.0, 10.0]
        .into_iter()
        .map(|az| direction(az, 45.0))
        .collect();
    let (cycles, retained) = synthetic(&candidates, 2, 2, &[2, 0]);
    let mut model = TableModel {
        candidates: candidates.to_vec(),
        table: vec![
            vec![vec![0.95, 0.0], vec![0.0; 2], vec![0.0, 0.9]],
            vec![vec![0.9, 0.0], vec![0.0; 2], vec![0.0; 2]],
        ],
    };
    let plan = optimize_prefix(
        candidates[1],
        &retained,
        &cycles,
        &[0.5; 2],
        &settings(2, 1),
        &mut model,
    )
    .unwrap();
    close(plan.acquisition_probability, 0.9, 1e-14);
    close(
        plan.acquisition_probability,
        plan.retained_acquisition_probability,
        1e-14,
    );
    assert_eq!(
        plan.actions
            .iter()
            .map(|a| a.direction_enu)
            .collect::<Vec<_>>(),
        retained.iter().map(|a| a.direction_enu).collect::<Vec<_>>()
    );
}

#[test]
fn scoring_handles_hard_probabilities_long_schedules_and_tiny_detection() {
    assert_eq!(schedule_acquisition_probability(&[1.0], &[]).unwrap(), 0.0);
    assert_eq!(
        schedule_acquisition_probability(&[1.0], &[vec![1.0]]).unwrap(),
        1.0
    );
    assert_eq!(
        schedule_acquisition_probability(&[0.0, 1.0], &[vec![1.0, 0.0]]).unwrap(),
        0.0
    );
    close(
        schedule_acquisition_probability(&[1.0], &[vec![1e-20]]).unwrap(),
        1e-20,
        1e-35,
    );
    close(
        schedule_acquisition_probability(&[0.3, 0.7], &[vec![1e-20, 0.0]]).unwrap(),
        3e-21,
        1e-35,
    );
    close(
        schedule_acquisition_probability(&[1.0], &vec![vec![0.001]; 10000]).unwrap(),
        1.0 - 0.999_f64.powi(10000),
        1e-14,
    );
    for row in [vec![f64::NAN], vec![1.1], vec![-0.1], vec![]] {
        assert!(schedule_acquisition_probability(&[1.0], &[row]).is_err());
    }
}

struct PreferredDirectionModel(EnuDirection);
impl DetectionModel for PreferredDirectionModel {
    fn probabilities(
        &mut self,
        command: EnuDirection,
        particles: &[PointingGeometry],
        _: f64,
        output: &mut [f64],
    ) -> Result<()> {
        for (point, probability) in particles.iter().zip(output) {
            *probability = if point.visible && command.separation(self.0) < 1e-12 {
                0.9
            } else {
                0.0
            };
        }
        Ok(())
    }
}

#[test]
fn optimizer_can_hold_outside_candidates_and_rejects_below_mask_commands() {
    let initial = direction(0.0, 45.0);
    let candidates: Vec<_> = [-1.0, 0.0, 1.0]
        .into_iter()
        .map(|az| direction(az, 40.0))
        .collect();
    let (cycles, retained) = synthetic(&candidates, 1, 1, &[1]);
    let plan = optimize_prefix(
        initial,
        &retained,
        &cycles,
        &[1.0],
        &settings(1, 10),
        &mut PreferredDirectionModel(initial),
    )
    .unwrap();
    assert_eq!(plan.actions[0].direction_enu, initial);
    assert_eq!(plan.actions[0].candidate_index, None);
    close(plan.acquisition_probability, 0.9, 1e-14);

    let candidates = vec![direction(0.0, -10.0), initial, direction(10.0, 45.0)];
    let (cycles, retained) = synthetic(&candidates, 1, 1, &[1]);
    let mut model = TableModel {
        candidates: candidates.to_vec(),
        table: vec![vec![vec![1.0], vec![0.2], vec![0.7]]],
    };
    let plan = optimize_prefix(
        initial,
        &retained,
        &cycles,
        &[1.0],
        &settings(1, 10),
        &mut model,
    )
    .unwrap();
    assert_eq!(plan.actions[0].candidate_index, Some(2));
    close(plan.acquisition_probability, 0.7, 1e-14);
    assert!(
        optimize_prefix(
            candidates[0],
            &retained,
            &cycles,
            &[1.0],
            &settings(1, 10),
            &mut model
        )
        .is_err()
    );
}

fn station_and_start() -> (GroundStation, Instant) {
    simulation::station_and_start().unwrap()
}

#[test]
fn tle_omm_and_topocentric_geometry_match_independent_satkit_transforms() {
    let (site, start) = station_and_start();
    let times = [start, start + Duration::from_seconds(60.0), start];
    let geometry = PreparedGeometry::new(&times, &[site; 3]).unwrap();
    let orbit = tle();
    let omm = OMM::from_tle(&orbit);
    let mut source = Ephemeris::Tle(orbit.clone());
    let mut omm_source = Ephemeris::Omm(omm);
    let points = propagate_pointing_grid(&mut source, &geometry, 10_f64.to_radians()).unwrap();
    let omm_points =
        propagate_pointing_grid(&mut omm_source, &geometry, 10_f64.to_radians()).unwrap();
    let mut independent = orbit.clone();
    let propagated = sgp4(&mut independent, &times).unwrap();
    let coord = ITRFCoord::from_geodetic_deg(site.lat_deg, site.lon_deg, site.alt_m);
    for (i, point) in points.iter().enumerate() {
        assert_eq!(propagated.errcode[i], SGP4Error::SGP4Success);
        let r = V3::from_slice(propagated.pos.col_slice(i));
        let velocity = V3::from_slice(propagated.vel.col_slice(i));
        let (gcrf, _) =
            frametransform::transform_state(Frame::TEME, Frame::GCRF, &times[i], &r, &velocity)
                .unwrap();
        // PreparedGeometry uses the existing approximate station rotation.
        let local = coord.q_enu2itrf().conjugate()
            * (frametransform::qitrf2gcrf_approx(&times[i]).conjugate() * gcrf - coord.itrf);
        close(point.range_m, local.norm(), 1e-7);
        assert!(
            point
                .direction_enu
                .separation(EnuDirection::new(local).unwrap())
                < 1e-12
        );
        close(point.range_m, omm_points[i].range_m, 1e-6);
        assert!(point.direction_enu.separation(omm_points[i].direction_enu) < 1e-12);
        let state = kernels::initial_state_from_tle(&orbit, &times[i]).unwrap();
        let shared = kernels::propagate_sgp4_gcrf(&mut source, &[times[i]]).unwrap();
        assert_eq!(state, shared[0]);
    }
    assert!(
        propagate_pointing_grid(&mut source, &PreparedGeometry::new(&[], &[]).unwrap(), 0.0)
            .unwrap()
            .is_empty()
    );
    let station = geometry.row(0).unwrap();
    let zenith = station.position() + station.rotation().to_gcrf(&v3(0.0, 0.0, 1000.0));
    let nadir = station.position() + station.rotation().to_gcrf(&v3(0.0, 0.0, -1000.0));
    close(
        pointing_geometry(&zenith, station, 0.0)
            .unwrap()
            .elevation_rad,
        std::f64::consts::FRAC_PI_2,
        1e-12,
    );
    assert!(!pointing_geometry(&nadir, station, 0.0).unwrap().visible);
    assert!(matches!(
        pointing_geometry(station.position(), station, 0.0),
        Err(AcquisitionError::Kernel(kernels::KernelError::ZeroRange))
    ));
    assert!(kernels::propagate_sgp4_gcrf(&mut source, &[Instant::INVALID]).is_err());
    let mut invalid = tle();
    invalid.mean_motion = -1.0;
    assert!(kernels::propagate_sgp4_gcrf(&mut invalid, &times).is_err());
}

fn controller(cycles: usize, threshold: f64) -> BcMpc {
    let (site, start) = station_and_start();
    let mut setup = SearchSetup::new(site, start);
    setup.config.search_duration_s = cycles as f64 * 6.0;
    setup.config.candidate_count = 3;
    setup.config.beam_width = 10;
    setup.config.ess_threshold = threshold;
    let particles: Vec<_> = [-20.0, 0.0, 20.0]
        .into_iter()
        .enumerate()
        .map(|(i, phase)| {
            let mut orbit = tle();
            orbit.mean_anomaly += phase * orbit.mean_motion / 240.0;
            if i == 1 {
                Ephemeris::Omm(OMM::from_tle(&orbit))
            } else {
                orbit.into()
            }
        })
        .collect();
    BcMpc::new(
        tle().into(),
        ParticleBelief::new(particles, vec![1.0; 3]).unwrap(),
        setup,
    )
    .unwrap()
}
struct ConstantModel(f64);
impl DetectionModel for ConstantModel {
    fn probabilities(
        &mut self,
        _: EnuDirection,
        particles: &[PointingGeometry],
        _: f64,
        output: &mut [f64],
    ) -> Result<()> {
        for (particle, p) in particles.iter().zip(output) {
            *p = if particle.visible { self.0 } else { 0.0 };
        }
        Ok(())
    }
}
struct FaultModel;
impl DetectionModel for FaultModel {
    fn probabilities(
        &mut self,
        _: EnuDirection,
        _: &[PointingGeometry],
        _: f64,
        _: &mut [f64],
    ) -> Result<()> {
        Err(AcquisitionError::Detection("receiver unavailable".into()))
    }
}
struct MissingOutput;
impl DetectionModel for MissingOutput {
    fn probabilities(
        &mut self,
        _: EnuDirection,
        _: &[PointingGeometry],
        _: f64,
        _: &mut [f64],
    ) -> Result<()> {
        Ok(())
    }
}

#[test]
fn controller_transitions_and_errors_are_atomic() {
    let mut controller = controller(3, 0.5);
    assert_eq!(controller.status(), SearchStatus::Ready);
    assert!(controller.observe(Observation::Miss, || 0.0).is_err());
    let initial = controller.direction();
    assert!(controller.plan(&mut FaultModel).is_err());
    assert!(controller.plan(&mut MissingOutput).is_err());
    assert!(controller.plan(&mut ConstantModel(f64::NAN)).is_err());
    assert_eq!(controller.status(), SearchStatus::Ready);
    assert_eq!(controller.direction(), initial);
    let plan = controller.plan(&mut ConstantModel(0.0)).unwrap();
    assert_eq!(plan.actions.len(), 3);
    let first = plan.actions[0];
    assert_eq!(controller.status(), SearchStatus::AwaitingObservation);
    assert!(controller.plan(&mut ConstantModel(0.0)).is_err());
    let update = controller
        .observe(Observation::Miss, || panic!("no resampling required"))
        .unwrap();
    assert!(!update.resampled);
    assert_eq!(controller.cycle(), 1);
    assert_eq!(controller.direction(), first.direction_enu);
    let next = controller.plan(&mut ConstantModel(0.0)).unwrap();
    assert_eq!(next.actions.len(), 2);
    assert_eq!(next.actions[0].cycle, 1);
    let update = controller
        .observe(Observation::Acquired, || {
            panic!("confirmation never resamples")
        })
        .unwrap();
    assert_eq!(update.status, SearchStatus::Acquired);
    assert!(controller.finished());
    assert!(controller.plan(&mut ConstantModel(0.0)).is_err());
    assert!(controller.observe(Observation::Miss, || 0.0).is_err());

    let mut unsupported = self::controller(1, 0.5);
    unsupported.plan(&mut ConstantModel(1.0)).unwrap();
    let weights = unsupported.belief().weights().to_vec();
    assert!(matches!(
        unsupported.observe(Observation::Miss, || 0.0),
        Err(AcquisitionError::ImpossibleObservation)
    ));
    assert_eq!(unsupported.status(), SearchStatus::AwaitingObservation);
    assert_eq!(unsupported.cycle(), 0);
    assert_eq!(unsupported.belief().weights(), weights);
    unsupported.observe(Observation::Acquired, || 0.0).unwrap();
}

struct ResamplingModel;
impl DetectionModel for ResamplingModel {
    fn probabilities(
        &mut self,
        _: EnuDirection,
        particles: &[PointingGeometry],
        _: f64,
        output: &mut [f64],
    ) -> Result<()> {
        for (i, (geometry, p)) in particles.iter().zip(output).enumerate() {
            *p = if !geometry.visible || i == 2 {
                0.0
            } else {
                1.0
            };
        }
        Ok(())
    }
}
struct IdenticalGeometryModel;
impl DetectionModel for IdenticalGeometryModel {
    fn probabilities(
        &mut self,
        _: EnuDirection,
        particles: &[PointingGeometry],
        _: f64,
        output: &mut [f64],
    ) -> Result<()> {
        for point in particles {
            close(point.range_m, particles[0].range_m, 0.0);
            assert_eq!(point.direction_enu, particles[0].direction_enu);
        }
        output.fill(0.0);
        Ok(())
    }
}
#[test]
fn resampled_geometry_uses_ancestor_indices_and_invalid_draw_rolls_back() {
    let mut controller = controller(2, 0.5);
    controller.plan(&mut ResamplingModel).unwrap();
    let weights = controller.belief().weights().to_vec();
    assert!(controller.observe(Observation::Miss, || 1.0).is_err());
    assert_eq!(controller.belief().weights(), weights);
    assert_eq!(controller.belief().indices(), &[0, 1, 2]);
    assert_eq!(controller.cycle(), 0);
    let update = controller.observe(Observation::Miss, || 0.25).unwrap();
    assert!(update.resampled);
    assert_eq!(controller.belief().indices(), &[2, 2, 2]);
    controller.plan(&mut IdenticalGeometryModel).unwrap();
    assert_eq!(
        controller
            .observe(Observation::Miss, || 0.0)
            .unwrap()
            .status,
        SearchStatus::Exhausted
    );
}

#[test]
fn search_configuration_and_low_level_planner_inputs_are_checked() {
    let default = MpcConfig::default();
    assert_eq!(default.cycle_count().unwrap(), 32);
    for config in [
        MpcConfig {
            horizon: 0,
            ..default
        },
        MpcConfig {
            beam_width: 0,
            ..default
        },
        MpcConfig {
            candidate_count: 2,
            ..default
        },
        MpcConfig {
            search_duration_s: 193.0,
            ..default
        },
        MpcConfig {
            move_s: 0.0,
            ..default
        },
        MpcConfig {
            settle_s: -1.0,
            ..default
        },
        MpcConfig {
            listen_s: f64::NAN,
            ..default
        },
        MpcConfig {
            ess_threshold: 1.1,
            ..default
        },
    ] {
        assert!(config.validate().is_err());
    }
    let (site, start) = station_and_start();
    let mut setup = SearchSetup::new(site, start);
    setup.initial_direction = Some(direction(0.0, -10.0));
    assert!(BcMpc::new(tle().into(), belief(vec![1.0]), setup).is_err());
    setup.initial_direction = None;
    setup.start = Instant::INVALID;
    assert!(BcMpc::new(tle().into(), belief(vec![1.0]), setup).is_err());
    let candidates: Vec<_> = [-10.0, 0.0, 10.0]
        .into_iter()
        .map(|az| direction(az, 45.0))
        .collect();
    let (mut cycles, mut retained) = synthetic(&candidates, 1, 1, &[1]);
    retained[0].candidate_index = Some(5);
    assert!(
        optimize_prefix(
            candidates[1],
            &retained,
            &cycles,
            &[1.0],
            &settings(1, 1),
            &mut ConstantModel(0.0)
        )
        .is_err()
    );
    retained[0].candidate_index = Some(1);
    cycles[0].particles[0].visible = false;
    assert!(
        optimize_prefix(
            candidates[1],
            &retained,
            &cycles,
            &[1.0],
            &settings(1, 1),
            &mut ConstantModel(0.0)
        )
        .is_err()
    );
}

#[test]
fn private_bootstrap_makes_partial_slews_when_candidates_are_unreachable() {
    let (site, start) = station_and_start();
    let geometry = PreparedGeometry::new(&[start], &[site]).unwrap();
    let nominal = propagate_pointing_grid(&mut tle(), &geometry, 0.0).unwrap()[0];
    let angles = nominal.direction_enu.az_el();
    let initial = EnuDirection::from_az_el(angles[0] + std::f64::consts::PI, angles[1]).unwrap();
    let mut setup = SearchSetup::new(site, start);
    setup.initial_direction = Some(initial);
    setup.config.search_duration_s = 18.0;
    setup.config.candidate_count = 1;
    let mut controller = BcMpc::new(tle().into(), belief(vec![1.0]), setup).unwrap();
    let plan = controller.plan(&mut ConstantModel(0.0)).unwrap();
    assert_eq!(plan.actions[0].candidate_index, None);
    assert!(initial.separation(plan.actions[0].direction_enu) > 0.0);
    let mut previous = initial;
    for action in &plan.actions {
        assert!(
            setup
                .antenna
                .reachable(previous, action.direction_enu, 4.0)
                .unwrap()
        );
        assert!(action.direction_enu.elevation_rad() >= setup.elevation_mask_rad);
        previous = action.direction_enu;
    }
}

struct ReferencePlant {
    truth: TLE,
    station: GroundStation,
    previous: EnuDirection,
    next_cycle: usize,
}
impl ReferencePlant {
    fn execute(
        &mut self,
        action: Action,
        receiver: &mut GaussianReceiver,
        uniform: f64,
    ) -> Observation {
        assert_eq!(action.cycle, self.next_cycle);
        assert!(
            AntennaLimits::default()
                .reachable(self.previous, action.direction_enu, 4.0)
                .unwrap()
        );
        assert!(action.direction_enu.elevation_rad() >= 10_f64.to_radians());
        close(
            (action.listen_end - action.listen_start).as_seconds(),
            1.5,
            1e-6,
        );
        let time = action.observation_time();
        let propagated = sgp4(&mut self.truth, &[time]).unwrap();
        let (r, _) = frametransform::transform_state(
            Frame::TEME,
            Frame::GCRF,
            &time,
            &V3::from_slice(propagated.pos.col_slice(0)),
            &V3::from_slice(propagated.vel.col_slice(0)),
        )
        .unwrap();
        let geometry = PreparedGeometry::new(&[time], &[self.station]).unwrap();
        let truth = pointing_geometry(&r, geometry.row(0).unwrap(), 10_f64.to_radians()).unwrap();
        let mut probability = [0.0];
        receiver
            .probabilities(action.direction_enu, &[truth], 1.5, &mut probability)
            .unwrap();
        self.previous = action.direction_enu;
        self.next_cycle += 1;
        if uniform < probability[0] {
            Observation::Acquired
        } else {
            Observation::Miss
        }
    }
}

#[test]
fn reference_caller_loop_with_private_bootstrap_and_separate_hidden_truth() {
    let (station, start) = station_and_start();
    let setup = SearchSetup::new(station, start);
    let mut planning = tle();
    planning.mean_anomaly += 10.0 * planning.mean_motion / 240.0;
    let particles = [-30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0]
        .into_iter()
        .map(|phase| {
            let mut orbit = planning.clone();
            orbit.mean_anomaly += phase * orbit.mean_motion / 240.0;
            orbit.into()
        })
        .collect();
    let belief = ParticleBelief::new(particles, vec![1.0; 7]).unwrap();
    let mut controller = BcMpc::new(planning.into(), belief, setup).unwrap();
    let mut plant = ReferencePlant {
        truth: tle(),
        station,
        previous: controller.direction(),
        next_cycle: 0,
    };
    let mut receiver = GaussianReceiver;
    let mut probabilities = Vec::new();
    // A caller-owned loop: plan, execute one action, then report the outcome.
    while !controller.finished() {
        let plan = controller.plan(&mut receiver).unwrap();
        assert!(plan.acquisition_probability >= plan.retained_acquisition_probability);
        probabilities.push(plan.acquisition_probability);
        let action = plan.actions[0];
        // Force two valid misses, then use a deterministic receiver draw.
        let draw = if action.cycle < 2 { 1.0 } else { 0.0 };
        let observation = plant.execute(action, &mut receiver, draw);
        controller.observe(observation, || 0.375).unwrap();
    }
    assert_eq!(controller.status(), SearchStatus::Acquired);
    assert!(controller.cycle() >= 3 && controller.cycle() <= 32);
    assert!(
        probabilities
            .iter()
            .all(|p| p.is_finite() && (0.0..=1.0).contains(p))
    );
}

fn assert_simulation_contract(result: &SimulationResult) {
    let setup = result.setup;
    let config = setup.config;
    assert_eq!(result.controller_cycles, result.plant_cycles);
    assert_eq!(result.records.len(), result.plant_cycles);
    assert!(result.plant_cycles <= config.cycle_count().unwrap());
    let mut previous = result.initial_direction;
    for (cycle, step) in result.records.iter().enumerate() {
        assert_eq!(step.cycle, cycle);
        let expected_start = setup.start
            + Duration::from_seconds(
                cycle as f64 * config.cycle_duration_s() + config.move_s + config.settle_s,
            );
        assert_eq!(step.listen_start, expected_start);
        assert_eq!(
            step.listen_end,
            expected_start + Duration::from_seconds(config.listen_s)
        );
        close(
            step.elapsed_s,
            (cycle + 1) as f64 * config.cycle_duration_s(),
            1e-6,
        );
        assert!(step.elapsed_s <= config.search_duration_s);
        assert!(step.direction_enu.elevation_rad() >= setup.elevation_mask_rad);
        assert!(step.slew_time_s <= config.move_s + 1e-8);
        close(
            step.slew_time_s,
            setup
                .antenna
                .slew_time_s(previous, step.direction_enu)
                .unwrap(),
            1e-12,
        );
        for probability in [
            step.detection_probability,
            step.predicted_acquisition_probability,
            step.retained_acquisition_probability,
        ] {
            assert!(probability.is_finite() && (0.0..=1.0).contains(&probability));
        }
        assert!(step.predicted_acquisition_probability >= step.retained_acquisition_probability);
        assert!((0.0..1.0).contains(&step.receiver_draw));
        assert_eq!(
            step.observation == Observation::Acquired,
            step.receiver_draw < step.detection_probability
        );
        previous = step.direction_enu;
    }
    assert_eq!(result.plant_time, result.records.last().unwrap().listen_end);
}

#[test]
fn seeded_simulated_satellite_is_discovered_with_reproducible_feedback() {
    let first = run_simulation(GaussianReceiver).unwrap();
    let repeat = run_simulation(GaussianReceiver).unwrap();
    assert_simulation_contract(&first);
    assert_eq!(first.records, repeat.records);
    assert_eq!(first.status, SearchStatus::Acquired);
    assert_eq!(repeat.status, first.status);
    assert_eq!(
        first.initial_pointing_error_rad,
        repeat.initial_pointing_error_rad
    );
    assert!(first.initial_pointing_error_rad > BEAM_FWHM_RAD / 2.0);
    assert!(first.plant_cycles > 0);
    let confirmation = first.records.last().unwrap();
    assert_eq!(confirmation.observation, Observation::Acquired);
    assert!(confirmation.true_visible);
    // The confirmation must reflect useful pointing, not a tiny beam-tail probability.
    assert!(confirmation.pointing_error_rad < BEAM_FWHM_RAD);
    assert!(
        first.records[..first.records.len() - 1]
            .iter()
            .all(|step| step.observation == Observation::Miss)
    );
    assert!(!confirmation.resampled);
}

#[test]
fn simulated_zero_probability_receiver_exhausts_the_search_budget() {
    let result = run_simulation(ConstantModel(0.0)).unwrap();
    assert_simulation_contract(&result);
    assert_eq!(result.status, SearchStatus::Exhausted);
    assert_eq!(result.plant_cycles, 32);
    close(
        (result.plant_time - result.setup.start).as_seconds(),
        192.0,
        1e-6,
    );
    assert!(result.records.iter().all(|step| {
        step.observation == Observation::Miss && step.detection_probability == 0.0
    }));
}

#[test]
fn simulated_execution_errors_preserve_pending_action_and_receiver_randomness() {
    let (mut controller, mut plant) = simulation::scenario().unwrap();
    let (_, mut reference) = simulation::scenario().unwrap();
    let action = controller.plan(&mut GaussianReceiver).unwrap().actions[0];
    let mut wrong_cycle = action;
    wrong_cycle.cycle += 1;
    let mut wrong_timing = action;
    wrong_timing.listen_start += Duration::from_seconds(1.0);
    let mut below_mask = action;
    below_mask.direction_enu = direction(0.0, -10.0);
    let mut unreachable = action;
    unreachable.direction_enu = EnuDirection::from_az_el(
        controller.direction().az_el()[0] + std::f64::consts::PI,
        10_f64.to_radians(),
    )
    .unwrap();
    for rejected in [wrong_cycle, wrong_timing, below_mask, unreachable] {
        assert!(plant.execute(rejected, &mut GaussianReceiver).is_err());
    }
    assert!(plant.execute(action, &mut FaultModel).is_err());
    assert!(plant.execute(action, &mut MissingOutput).is_err());
    assert!(plant.execute(action, &mut ConstantModel(f64::NAN)).is_err());
    assert_eq!(controller.status(), SearchStatus::AwaitingObservation);
    assert_eq!(controller.cycle(), 0);
    let observation = plant.execute(action, &mut GaussianReceiver).unwrap();
    assert_eq!(
        observation,
        reference.execute(action, &mut GaussianReceiver).unwrap()
    );
    assert!(plant.execute(action, &mut GaussianReceiver).is_err());
    controller
        .observe(observation.observation, || 0.375)
        .unwrap();
    assert_eq!(controller.cycle(), 1);
}
