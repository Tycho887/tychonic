use std::f64::consts::PI;

use numeris::{DynVector, optim::LmSettings};
use satkit::{Duration, Instant, orbitprop::SatPropertiesSimple};
use std::sync::Arc;
use tychonic::{
    kernels::*,
    optim::{CachedProblem, ODProblem, fit_lm},
};

// Mutable test inputs are deliberately separate from prepared production types.
#[derive(Clone)]
struct BatchInput {
    reference_epoch: satkit::Instant,
    t_meas: Vec<satkit::Instant>,
    station: GCRFTrajectory,
    enu: EnuBasisTrajectory,
    obs: Observations,
    settings: satkit::orbitprop::PropSettings,
    delta_cd: f64,
}
impl BatchInput {
    fn prepare(&self) -> Result<ODProblem> {
        if self.t_meas != self.station.timestamps {
            return Err(KernelError::InvalidInput(
                "geometry and observation epochs do not match",
            ));
        }
        ODProblem::new(
            PropagationConfig::new(
                self.reference_epoch,
                self.settings.clone(),
                ParameterMode::StateAndDrag {
                    delta_cd: self.delta_cd,
                },
            )?,
            vec![PreparedBatch::combined(
                Arc::new(PreparedGeometry::from_trajectories(
                    self.station.clone(),
                    self.enu.clone(),
                )?),
                self.obs.clone(),
            )?],
        )
    }
    fn validate(&self) -> Result<()> {
        self.prepare().map(|_| ())
    }
    fn eval(&self, x: &DynVector<f64>) -> Result<(DynVector<f64>, numeris::DynMatrix<f64>)> {
        let e = self.prepare()?.evaluate(x)?;
        Ok((e.residual, e.jacobian))
    }
    fn from_measurements(
        reference_epoch: satkit::Instant,
        measurements: &[Measurement],
        noise: MeasurementNoise,
        settings: satkit::orbitprop::PropSettings,
        delta_cd: f64,
    ) -> Result<Self> {
        // Exercise the production measurement constructor as well as raw fixtures.
        ODProblem::from_measurements(
            reference_epoch,
            measurements,
            noise,
            settings.clone(),
            delta_cd,
        )?;
        let times: Vec<_> = measurements.iter().map(|m| m.time).collect();
        let stations: Vec<_> = measurements.iter().map(|m| m.station).collect();
        let (station, enu) = precompute_station_geometry(&times, &stations)?;
        Ok(Self {
            reference_epoch,
            t_meas: times,
            station,
            enu,
            settings,
            delta_cd,
            obs: Observations {
                range_rate: measurements.iter().map(|m| m.range_rate).collect(),
                azimuth: measurements.iter().map(|m| m.azimuth).collect(),
                elevation: measurements.iter().map(|m| m.elevation).collect(),
                sigma_rr: noise.sigma_rr,
                sigma_az: noise.sigma_az,
                sigma_el: noise.sigma_el,
            },
        })
    }
}

fn fixture(offsets: &[f64]) -> (BatchInput, DynVector<f64>) {
    let site = GroundStation {
        lat_deg: 40.0,
        lon_deg: -75.0,
        alt_m: 100.0,
    };
    fixture_with_stations(offsets, &vec![site; offsets.len()])
}

fn fixture_with_stations(offsets: &[f64], sites: &[GroundStation]) -> (BatchInput, DynVector<f64>) {
    let epoch = Instant::from_datetime(2024, 1, 1, 12, 0, 0.25).unwrap();
    let site = GroundStation {
        lat_deg: 40.0,
        lon_deg: -75.0,
        alt_m: 100.0,
    };
    let (centre, basis) = precompute_station_geometry(&[epoch], &[site]).unwrap();
    let state = v6_from(
        &(centre.position[0] + basis.up(0) * 300e3 + basis.east(0) * 80e3 + basis.north(0) * 40e3),
        &(centre.velocity[0] + basis.east(0) * 7100.0 + basis.north(0) * 3000.0),
    );
    let times: Vec<_> = offsets
        .iter()
        .map(|dt| epoch + Duration::from_seconds(*dt))
        .collect();
    let (station, enu) = precompute_station_geometry(&times, sites).unwrap();
    let settings = od_prop_settings();
    let mut obs = Observations {
        range_rate: Vec::new(),
        azimuth: Vec::new(),
        elevation: Vec::new(),
        sigma_rr: 0.05,
        sigma_az: 1e-4,
        sigma_el: 2e-4,
    };
    for (i, t) in times.iter().enumerate() {
        // Endpoint propagation is independent of the batch dense-output path.
        let propagated = propagate_state(
            &state,
            &epoch,
            t,
            &settings,
            &SatPropertiesSimple::new(0.02, 0.0),
        )
        .unwrap();
        let (rr, _) = range_rate_partial(
            &r_of(&propagated),
            &v_of(&propagated),
            &station.position[i],
            &station.velocity[i],
        )
        .unwrap();
        let ([az, el], _) = az_el_partial_with_rotation(
            &r_of(&propagated),
            &station.position[i],
            &enu.enu_to_gcrf[i],
        )
        .unwrap();
        obs.range_rate.push(rr);
        obs.azimuth.push(az);
        obs.elevation.push(el);
    }
    let mut truth = DynVector::zeros(7);
    for k in 0..6 {
        truth[k] = state[k];
    }
    truth[6] = 0.02;
    (
        BatchInput {
            reference_epoch: epoch,
            t_meas: times,
            station,
            enu,
            obs,
            settings,
            delta_cd: 1e-4,
        },
        truth,
    )
}

#[test]
fn whitened_batch_jacobian_matches_full_residual_differences() {
    let (problem, mut x) = fixture(&[-600.5, -120.125, 0.0, 60.25, 420.0, 600.5]);
    x[0] += 20.0;
    x[2] -= 30.0;
    x[4] += 0.02;
    x[6] += 0.001;
    assert_batch_jacobian(&problem, &x);
}

fn assert_batch_jacobian(problem: &BatchInput, x: &DynVector<f64>) {
    let (r, j) = problem.eval(x).unwrap();
    assert_eq!(r.len(), 3 * problem.t_meas.len());
    assert_eq!((j.nrows(), j.ncols()), (r.len(), 7));
    for k in 0..7 {
        let step = if k < 3 {
            10.0
        } else if k < 6 {
            0.01
        } else {
            2e-4
        };
        let mut plus = x.clone();
        let mut minus = x.clone();
        plus[k] += step;
        minus[k] -= step;
        let (rp, _) = problem.eval(&plus).unwrap();
        let (rm, _) = problem.eval(&minus).unwrap();
        for row in 0..r.len() {
            let fd = (rp[row] - rm[row]) / (2.0 * step);
            let atol = if k == 6 { 0.002 } else { 2e-5 };
            assert!(
                (j[(row, k)] - fd).abs() < atol + 3e-4 * fd.abs(),
                "J[{row},{k}] = {:.10e}, residual difference = {fd:.10e}",
                j[(row, k)]
            );
        }
    }
}

#[test]
fn mixed_measurement_records_preserve_stations_and_have_correct_batch_jacobians() {
    let sites = [
        GroundStation {
            lat_deg: 40.0,
            lon_deg: -75.0,
            alt_m: 100.0,
        },
        GroundStation {
            lat_deg: 40.3,
            lon_deg: -74.2,
            alt_m: 400.0,
        },
        GroundStation {
            lat_deg: 39.3,
            lon_deg: -75.4,
            alt_m: 1200.0,
        },
    ];
    let stations: Vec<_> = (0..6).map(|i| sites[i % sites.len()]).collect();
    let (source, mut x) =
        fixture_with_stations(&[-600.5, -120.125, 0.0, 0.0, 60.25, 600.5], &stations);
    let measurements: Vec<_> = source
        .t_meas
        .iter()
        .enumerate()
        .map(|(i, time)| Measurement {
            time: *time,
            station: stations[i],
            range_rate: source.obs.range_rate[i],
            azimuth: source.obs.azimuth[i],
            elevation: source.obs.elevation[i],
        })
        .collect();
    let problem = BatchInput::from_measurements(
        source.reference_epoch,
        &measurements,
        MeasurementNoise {
            sigma_rr: source.obs.sigma_rr,
            sigma_az: source.obs.sigma_az,
            sigma_el: source.obs.sigma_el,
        },
        source.settings,
        source.delta_cd,
    )
    .unwrap();
    assert_eq!(problem.t_meas, source.t_meas);
    assert_eq!(problem.station.ground_stations, stations);
    assert_eq!(problem.obs.range_rate, source.obs.range_rate);
    assert_eq!(problem.obs.azimuth, source.obs.azimuth);
    assert_eq!(problem.obs.elevation, source.obs.elevation);
    assert_eq!(problem.t_meas[2], problem.t_meas[3]);
    assert!((problem.station.position[2] - problem.station.position[3]).norm() > 1000.0);
    assert!(
        problem
            .eval(&x)
            .unwrap()
            .0
            .dot(&problem.eval(&x).unwrap().0)
            < 1e-6
    );
    let stored_rotations = problem.enu.enu_to_gcrf.clone();
    let stored_positions = problem.station.position.clone();
    x[0] += 20.0;
    x[2] -= 30.0;
    x[4] += 0.02;
    x[6] += 0.001;
    assert_batch_jacobian(&problem, &x);
    assert_eq!(problem.enu.enu_to_gcrf, stored_rotations);
    assert_eq!(problem.station.position, stored_positions);
}

#[test]
fn residual_sign_whitening_and_azimuth_wrapping_are_consistent() {
    let (mut problem, truth) = fixture(&[0.0]);
    let az = problem.obs.azimuth[0];
    problem.obs.range_rate[0] += 3.0 * problem.obs.sigma_rr;
    problem.obs.azimuth[0] = az + 2.0 * PI + 2.0 * problem.obs.sigma_az;
    problem.obs.elevation[0] -= 4.0 * problem.obs.sigma_el;
    let (r, j) = problem.eval(&truth).unwrap();
    for (actual, expected) in r.as_slice().iter().zip([-3.0, -2.0, 4.0]) {
        assert!((actual - expected).abs() < 1e-10);
    }
    for row in 0..3 {
        assert_eq!(j[(row, 6)], 0.0);
    }
    // Explicitly cross the atan2 branch at north-through-south azimuth.
    let epoch = problem.reference_epoch;
    problem.station.position[0] = V3::zeros();
    problem.station.velocity[0] = V3::zeros();
    problem.enu.enu_to_gcrf[0] = M3::eye();
    problem.obs.azimuth[0] = -PI + 1e-6;
    let x = DynVector::from_vec(vec![7.0, -7e6, 1e6, 0.0, 7000.0, 0.0, 0.02]);
    let (r, _) = problem.eval(&x).unwrap();
    assert!((r[1] + 0.02).abs() < 1e-8);
    assert_eq!(problem.t_meas[0], epoch);
}

#[test]
fn identity_stm_and_arbitrary_chain_rule_match_measurement_perturbations() {
    let (problem, truth) = fixture(&[0.0]);
    let state = V6::from_slice(&truth.as_slice()[..6]);
    let mut stm = M6::eye();
    stm[(0, 3)] = 120.0;
    stm[(1, 0)] = -0.2;
    stm[(2, 5)] = -60.0;
    stm[(4, 2)] = 0.001;
    let drag = numeris::vector![100.0, -70.0, 20.0, 0.3, -0.1, 0.2];
    let sens = Sensitivities {
        timestamps: problem.t_meas.clone(),
        states: vec![state],
        stm: vec![stm],
        state_dcd: vec![drag],
    };
    let jac = compute_full_jacobian(&sens, &problem.station, &problem.enu).unwrap();
    let predict = |x: &V6| {
        let (rr, _) = range_rate_partial(
            &r_of(x),
            &v_of(x),
            &problem.station.position[0],
            &problem.station.velocity[0],
        )
        .unwrap();
        let ([az, el], _) = az_el_partial(
            &r_of(x),
            &problem.station.position[0],
            &problem.enu.east(0),
            &problem.enu.north(0),
            &problem.enu.up(0),
        )
        .unwrap();
        [rr, az, el]
    };
    for k in 0..7 {
        let column = if k < 6 { stm.block::<6, 1>(0, k) } else { drag };
        let step = if k < 3 { 0.1 } else { 0.001 };
        let p = predict(&(state + column * step));
        let m = predict(&(state - column * step));
        for (row, matrix) in [&jac.range_rate, &jac.azimuth, &jac.elevation]
            .into_iter()
            .enumerate()
        {
            let diff = if row == 1 {
                wrap_pi(p[row] - m[row])
            } else {
                p[row] - m[row]
            };
            let fd = diff / (2.0 * step);
            assert!((matrix[(0, k)] - fd).abs() < 1e-8 + 1e-6 * fd.abs());
        }
    }
}

#[test]
fn variable_batches_including_1024_observations_have_consistent_shapes() {
    for n in [1, 2, 7, 1024] {
        let (problem, truth) = fixture(&vec![0.0; n]);
        let (r, j) = problem.eval(&truth).unwrap();
        assert_eq!(r.len(), 3 * n);
        assert_eq!((j.nrows(), j.ncols()), (3 * n, 7));
        assert_eq!(r.dot(&r), 0.0);
    }
}

#[test]
fn invalid_inputs_epochs_and_dimensions_are_rejected() {
    let (problem, truth) = fixture(&[0.0]);
    for sigma in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        let mut p = problem.clone();
        p.obs.sigma_rr = sigma;
        assert!(p.eval(&truth).is_err());
    }
    for len in [0, 6, 8] {
        assert!(problem.eval(&DynVector::zeros(len)).is_err());
    }
    let mut nan = truth.clone();
    nan[3] = f64::NAN;
    assert!(problem.eval(&nan).is_err());
    let mut p = problem.clone();
    p.obs.azimuth.clear();
    assert!(p.eval(&truth).is_err());
    let mut p = problem.clone();
    p.obs.elevation[0] = f64::NAN;
    assert!(p.eval(&truth).is_err());
    let mut p = problem.clone();
    p.enu.enu_to_gcrf.clear();
    assert!(p.eval(&truth).is_err());
    let mut p = problem.clone();
    p.station.timestamps[0] += Duration::from_seconds(0.001);
    assert!(p.eval(&truth).is_err());
    let mut p = problem.clone();
    p.enu.timestamps[0] += Duration::from_seconds(0.001);
    assert!(p.eval(&truth).is_err());
    let mut p = problem.clone();
    p.obs.sigma_rr = f64::MIN_POSITIVE;
    p.obs.range_rate[0] = 1e100;
    assert!(p.eval(&truth).is_err());
    let (empty, _) = fixture(&[]);
    assert!(empty.validate().is_err());
    let sens = Sensitivities {
        timestamps: problem.t_meas.clone(),
        states: vec![V6::from_slice(&truth.as_slice()[..6])],
        stm: vec![],
        state_dcd: vec![V6::zeros()],
    };
    assert!(compute_full_jacobian(&sens, &problem.station, &problem.enu).is_err());
    let mut sens = Sensitivities {
        timestamps: vec![problem.reference_epoch + Duration::from_seconds(0.001)],
        states: vec![V6::from_slice(&truth.as_slice()[..6])],
        stm: vec![M6::eye()],
        state_dcd: vec![V6::zeros()],
    };
    assert!(compute_full_jacobian(&sens, &problem.station, &problem.enu).is_err());
    sens.stm[0][(0, 0)] = f64::NAN;
    assert!(sens.augmented_transition(0).is_err());
}

#[test]
fn cached_evaluations_refresh_for_new_parameters_and_return_shared_values() {
    let (input, truth) = fixture(&[0.0]);
    let cached = CachedProblem::new(input.prepare().unwrap());
    let first = cached.evaluate(&truth).unwrap();
    let second = cached.evaluate(&truth).unwrap();
    assert!(std::rc::Rc::ptr_eq(&first, &second));
    let mut copy = (*first).clone();
    copy.residual[0] = 100.0;
    copy.jacobian[(0, 0)] = 100.0;
    assert_eq!(cached.evaluate(&truth).unwrap().residual[0], 0.0);
    let mut perturbed = truth.clone();
    perturbed[0] += 100.0;
    assert!(cached.evaluate(&perturbed).unwrap().rss() > 0.0);
    let refreshed = cached.evaluate(&truth).unwrap();
    assert_eq!(refreshed.rss(), 0.0);
    let mut invalid = truth.clone();
    invalid[6] = f64::NAN;
    assert!(cached.evaluate(&invalid).is_err());
    assert!(std::rc::Rc::ptr_eq(
        &refreshed,
        &cached.evaluate(&truth).unwrap()
    ));
}

fn fit(
    problem: BatchInput,
    truth: &DynVector<f64>,
    offset: f64,
    settings: &LmSettings<f64>,
) -> (DynVector<f64>, f64) {
    let mut guess = truth.clone();
    guess[0] += offset;
    guess[1] -= 0.75 * offset;
    guess[2] += 0.5 * offset;
    guess[3] += 0.0005 * offset;
    guess[6] += 0.001;
    let problem = problem.prepare().unwrap();
    let initial_rss = problem.evaluate(&guess).unwrap().rss();
    let result = fit_lm(problem, &guess, None, settings).unwrap();
    assert!(
        result.rss < initial_rss * 0.01,
        "fit did not substantially improve residuals"
    );
    (result.parameters, result.rss)
}

#[test]
fn synthetic_noiseless_od_recovers_all_seven_parameters_from_multiple_guesses() {
    for offset in [-100.0, 30.0, 200.0] {
        // Final-epoch batch estimation: all observations precede the state.
        let offsets: Vec<_> = (0..31).map(|i| -600.0 + i as f64 * 20.0).collect();
        let (problem, truth) = fixture(&offsets);
        let settings = LmSettings {
            grad_tol: 1e-6,
            x_tol: 1e-9,
            f_tol: 1e-10,
            ..LmSettings::default()
        };
        let (optimum, rss) = fit(problem, &truth, offset, &settings);
        let pos_error = (0..3)
            .map(|k| (optimum[k] - truth[k]).powi(2))
            .sum::<f64>()
            .sqrt();
        let vel_error = (3..6)
            .map(|k| (optimum[k] - truth[k]).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(rss < 1e-6, "RSS {rss}");
        assert!(pos_error < 0.001, "position error {pos_error}");
        assert!(vel_error < 1e-5, "velocity error {vel_error}");
        assert!(
            (optimum[6] - truth[6]).abs() < 5e-6,
            "drag error {}",
            optimum[6] - truth[6]
        );
    }
}

#[test]
fn simultaneous_two_station_od_recovers_the_reference_state_and_drag() {
    let sites = [
        GroundStation {
            lat_deg: 40.0,
            lon_deg: -75.0,
            alt_m: 100.0,
        },
        GroundStation {
            lat_deg: 40.3,
            lon_deg: -74.2,
            alt_m: 400.0,
        },
    ];
    let offsets: Vec<_> = (0..21)
        .flat_map(|i| [-600.0 + 30.0 * i as f64; 2])
        .collect();
    let stations: Vec<_> = (0..offsets.len()).map(|i| sites[i % sites.len()]).collect();
    let (problem, truth) = fixture_with_stations(&offsets, &stations);
    let settings = LmSettings {
        grad_tol: 1e-6,
        x_tol: 1e-9,
        f_tol: 1e-10,
        ..LmSettings::default()
    };
    let (optimum, rss) = fit(problem, &truth, 100.0, &settings);
    let pos_error = (0..3)
        .map(|k| (optimum[k] - truth[k]).powi(2))
        .sum::<f64>()
        .sqrt();
    let vel_error = (3..6)
        .map(|k| (optimum[k] - truth[k]).powi(2))
        .sum::<f64>()
        .sqrt();
    assert!(rss < 1e-6, "RSS {rss}");
    assert!(pos_error < 0.001, "position error {pos_error}");
    assert!(vel_error < 1e-5, "velocity error {vel_error}");
    assert!(
        (optimum[6] - truth[6]).abs() < 5e-6,
        "drag error {}",
        optimum[6] - truth[6]
    );
}

#[test]
fn deterministic_noisy_od_has_expected_residual_scale_and_state_accuracy() {
    let offsets: Vec<_> = (0..41).map(|i| -800.0 + i as f64 * 20.0).collect();
    let (mut problem, truth) = fixture(&offsets);
    // High-precision synthetic sensors let this short arc constrain drag.
    problem.obs.sigma_rr = 0.001;
    problem.obs.sigma_az = 1e-6;
    problem.obs.sigma_el = 2e-6;
    let mut seed = 123456789_u64;
    let mut uniform = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((seed >> 11) as f64 + 0.5) / ((1_u64 << 53) as f64)
    };
    for (values, sigma) in [
        (&mut problem.obs.range_rate, problem.obs.sigma_rr),
        (&mut problem.obs.azimuth, problem.obs.sigma_az),
        (&mut problem.obs.elevation, problem.obs.sigma_el),
    ] {
        for value in values {
            let noise = (-2.0 * uniform().ln()).sqrt() * (2.0 * PI * uniform()).cos();
            *value += sigma * noise;
        }
    }
    // Match stopping tolerances to a nonzero noise residual minimum and the
    // approximate drag STM rather than noiseless numerical precision.
    let settings = LmSettings {
        grad_tol: 1e-2,
        x_tol: 1e-5,
        f_tol: 1e-5,
        ..LmSettings::default()
    };
    let (optimum, rss) = fit(problem, &truth, 100.0, &settings);
    let reduced_chi2 = rss / (3 * offsets.len() - 7) as f64;
    assert!(
        (0.6..1.5).contains(&reduced_chi2),
        "reduced chi² {reduced_chi2}"
    );
    let pos_error = (0..3)
        .map(|k| (optimum[k] - truth[k]).powi(2))
        .sum::<f64>()
        .sqrt();
    let vel_error = (3..6)
        .map(|k| (optimum[k] - truth[k]).powi(2))
        .sum::<f64>()
        .sqrt();
    assert!(pos_error < 10.0, "position error {pos_error}");
    assert!(vel_error < 0.02, "velocity error {vel_error}");
    assert!(
        (optimum[6] - truth[6]).abs() < 0.003,
        "drag error {}",
        optimum[6] - truth[6]
    );
}
