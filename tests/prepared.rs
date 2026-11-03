#[path = "../benches/support/mod.rs"]
#[allow(dead_code)]
mod support;

use numeris::{DynVector, Matrix};
use satkit::{Duration, Instant};
use std::sync::Arc;
use tychonic::{
    kernels::*,
    optim::{ODProblem, fit_lm},
};

fn batches(f: &support::Fixture) -> [PreparedBatch; 3] {
    let geometry =
        Arc::new(PreparedGeometry::from_trajectories(f.station.clone(), f.enu.clone()).unwrap());
    let n = geometry.len();
    [
        PreparedBatch::range_rate(geometry.clone(), vec![0.0; n], 0.05).unwrap(),
        PreparedBatch::azimuth_elevation(geometry.clone(), vec![0.0; n], vec![0.0; n], 1e-4, 2e-4)
            .unwrap(),
        PreparedBatch::combined(
            geometry,
            Observations {
                range_rate: vec![0.0; n],
                azimuth: vec![0.0; n],
                elevation: vec![0.0; n],
                sigma_rr: 0.05,
                sigma_az: 1e-4,
                sigma_el: 2e-4,
            },
        )
        .unwrap(),
    ]
}

fn leo_state(geometry: &StationGeometry) -> V6 {
    let east = geometry.rotation().to_gcrf(&v3(1.0, 0.0, 0.0));
    let north = geometry.rotation().to_gcrf(&v3(0.0, 1.0, 0.0));
    let up = geometry.rotation().to_gcrf(&v3(0.0, 0.0, 1.0));
    v6_from(
        &(geometry.position() + up * 300e3 + east * 80e3 + north * 40e3),
        &(geometry.velocity() + east * 7100.0 + north * 3000.0),
    )
}

#[test]
fn each_prepared_model_has_correct_six_and_seven_column_derivatives() {
    let f = support::Fixture::new(64).unwrap();
    let models = batches(&f);
    let reference = f.reference_evaluation().unwrap();
    let prepared = models[2]
        .evaluate_from_sensitivities(&f.sensitivities)
        .unwrap();
    assert_eq!(reference.residual, prepared.residual);
    for (a, b) in reference
        .jacobian
        .as_slice()
        .iter()
        .zip(prepared.jacobian.as_slice())
    {
        assert!((a - b).abs() <= 1e-12 * (1.0 + a.abs()));
    }
    for columns in [6, 7] {
        let mut sens = f.sensitivities.clone();
        if columns == 6 {
            sens.state_dcd.clear();
        }
        for model in &models {
            let original = model.evaluate_from_sensitivities(&sens).unwrap();
            assert_eq!(original.jacobian.ncols(), columns);
            for k in 0..columns {
                let step = if k < 3 { 0.1 } else { 0.001 };
                let mut plus = sens.clone();
                let mut minus = sens.clone();
                for i in 0..sens.states.len() {
                    let direction = if k < 6 {
                        sens.stm[i].block::<6, 1>(0, k)
                    } else {
                        sens.state_dcd[i]
                    };
                    plus.states[i] += direction * step;
                    minus.states[i] -= direction * step;
                }
                let p = model.evaluate_from_sensitivities(&plus).unwrap();
                let m = model.evaluate_from_sensitivities(&minus).unwrap();
                for row in 0..model.rows() {
                    let expected = (p.residual[row] - m.residual[row]) / (2.0 * step);
                    let actual = original.jacobian[(row, k)];
                    let observable = row / sens.states.len();
                    let rr = model.kind() != MeasurementKind::AzimuthElevation && observable == 0;
                    let atol = if rr { 1e-6 / 0.05 } else { 1e-11 / 1e-4 };
                    assert!(
                        (actual - expected).abs() <= atol + 1e-5 * actual.abs().max(expected.abs()),
                        "{:?}, columns {columns}, row {row}, column {k}: {actual} != {expected}",
                        model.kind()
                    );
                }
            }
        }
    }
}

#[test]
fn output_reuse_matches_owned_and_recovers_after_errors_and_shape_changes() {
    let f = support::Fixture::new(16).unwrap();
    let models = batches(&f);
    let mut output = Evaluation::default();
    for model in &models {
        let owned = model.evaluate_from_sensitivities(&f.sensitivities).unwrap();
        model
            .evaluate_from_sensitivities_into(&f.sensitivities, &mut output)
            .unwrap();
        let rp = output.residual.as_slice().as_ptr();
        let jp = output.jacobian.as_slice().as_ptr();
        model
            .evaluate_from_sensitivities_into(&f.sensitivities, &mut output)
            .unwrap();
        assert_eq!(output, owned);
        assert_eq!(rp, output.residual.as_slice().as_ptr());
        assert_eq!(jp, output.jacobian.as_slice().as_ptr());
        let mut bad = f.sensitivities.clone();
        bad.stm[8][(0, 0)] = f64::NAN;
        assert!(
            model
                .evaluate_from_sensitivities_into(&bad, &mut output)
                .is_err()
        );
        model
            .evaluate_from_sensitivities_into(&f.sensitivities, &mut output)
            .unwrap();
        assert_eq!(output, owned);
        bad = f.sensitivities.clone();
        bad.timestamps[0] += Duration::from_seconds(0.000001);
        assert!(model.evaluate_from_sensitivities(&bad).is_err());
        bad = f.sensitivities.clone();
        bad.state_dcd.pop();
        assert!(model.evaluate_from_sensitivities(&bad).is_err());
        let mut state_only = f.sensitivities.clone();
        state_only.state_dcd.clear();
        model
            .evaluate_from_sensitivities_into(&state_only, &mut output)
            .unwrap();
        assert_eq!(output.jacobian.ncols(), 6);
    }
}

#[test]
fn range_rate_only_works_at_zenith_and_checked_rotation_imports_reject_bad_bases() {
    let t = Instant::from_unixtime(1704110400.25);
    let site = GroundStation {
        lat_deg: 40.0,
        lon_deg: -75.0,
        alt_m: 100.0,
    };
    let geometry = Arc::new(PreparedGeometry::new(&[t], &[site]).unwrap());
    let g = geometry.row(0).unwrap();
    let up = g.rotation().to_gcrf(&v3(0.0, 0.0, 1.0));
    let sens = Sensitivities {
        timestamps: vec![t],
        states: vec![v6_from(&(g.position() + up * 1e6), g.velocity())],
        stm: vec![M6::eye()],
        state_dcd: Vec::new(),
    };
    let rr = PreparedBatch::range_rate(geometry.clone(), vec![0.0], 1.0).unwrap();
    assert_eq!(rr.evaluate_from_sensitivities(&sens).unwrap().rss(), 0.0);
    let angles =
        PreparedBatch::azimuth_elevation(geometry, vec![0.0], vec![0.0], 1.0, 1.0).unwrap();
    assert!(matches!(
        angles.evaluate_from_sensitivities(&sens),
        Err(KernelError::UndefinedAzimuth)
    ));
    for matrix in [M3::eye() * 2.0, M3::zeros(), M3::eye() * f64::NAN] {
        assert!(ValidatedRotation::new(matrix).is_err());
        assert!(
            az_el_partial_with_rotation(&r_of(&sens.states[0]), &V3::zeros(), &matrix).is_err()
        );
    }
    assert!(
        chain_rule(
            &Matrix::<f64, 1, 6>::from_fn(|_, _| f64::INFINITY),
            &M6::eye()
        )
        .is_err()
    );
}

#[test]
fn geometry_interns_exact_station_epoch_pairs_and_preserves_order() {
    let t = Instant::from_unixtime(1704110400.25);
    let site = GroundStation {
        lat_deg: 40.0,
        lon_deg: -75.0,
        alt_m: 100.0,
    };
    let high = GroundStation {
        alt_m: 2100.0,
        ..site
    };
    let next = t + Duration::from_seconds(0.000001);
    let times = [next, t, t, next, t];
    let sites = [site, site, high, site, site];
    let geometry = PreparedGeometry::new(&times, &sites).unwrap();
    assert_eq!(geometry.timestamps(), times);
    assert_eq!(geometry.stations(), sites);
    assert_eq!(geometry.unique_entries(), 3);
    assert!(std::ptr::eq(
        geometry.row(0).unwrap(),
        geometry.row(3).unwrap()
    ));
    assert!(std::ptr::eq(
        geometry.row(1).unwrap(),
        geometry.row(4).unwrap()
    ));
    assert!(geometry.row(5).is_none());
    let (station, enu) = precompute_station_geometry(&times, &sites).unwrap();
    for i in 0..times.len() {
        let row = geometry.row(i).unwrap();
        assert_eq!(*row.position(), station.position[i]);
        assert_eq!(*row.velocity(), station.velocity[i]);
        assert_eq!(*row.rotation().matrix(), enu.enu_to_gcrf[i]);
    }
    let upper = geometry.row(2).unwrap();
    let lower = geometry.row(1).unwrap();
    let up = lower.rotation().to_gcrf(&v3(0.0, 0.0, 1.0));
    assert!((upper.position() - lower.position() - up * 2000.0).norm() < 1e-8);
    let shared = Arc::new(geometry);
    let rr = PreparedBatch::range_rate(shared.clone(), vec![0.0; 5], 1.0).unwrap();
    let angles =
        PreparedBatch::azimuth_elevation(shared.clone(), vec![0.0; 5], vec![0.0; 5], 1.0, 1.0)
            .unwrap();
    assert!(Arc::ptr_eq(rr.geometry(), angles.geometry()));
    assert!(PreparedBatch::range_rate(shared.clone(), vec![0.0], 1.0).is_err());
    for sigma in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(PreparedBatch::range_rate(shared.clone(), vec![0.0; 5], sigma).is_err());
    }
}

#[test]
fn state_only_sampling_and_stacked_batches_share_exact_epochs() {
    let f = support::Fixture::new(1).unwrap();
    let epoch = f.sensitivities.timestamps[0];
    let state = leo_state(f.prepared_geometry.row(0).unwrap());
    let x6 = DynVector::from_slice(state.as_slice());
    let mut parameters = state.as_slice().to_vec();
    parameters.push(0.02);
    let x7 = DynVector::from_vec(parameters);
    let times = [
        epoch + Duration::from_seconds(120.125),
        epoch,
        epoch - Duration::from_seconds(120.25),
        epoch,
    ];
    let site = f.station.ground_stations[0];
    let geometry = Arc::new(PreparedGeometry::new(&times, &[site; 4]).unwrap());
    let rr = PreparedBatch::range_rate(geometry.clone(), vec![0.0; 4], 0.05).unwrap();
    let angles =
        PreparedBatch::azimuth_elevation(geometry, vec![0.0; 4], vec![0.0; 4], 1e-4, 2e-4).unwrap();
    // State-only has no difference step to validate or drag perturbation arcs.
    let c6 = PropagationConfig::new(
        epoch,
        od_prop_settings(),
        ParameterMode::StateOnly { fixed_drag: 0.02 },
    )
    .unwrap();
    let c7 = PropagationConfig::new(
        epoch,
        od_prop_settings(),
        ParameterMode::StateAndDrag { delta_cd: 1e-4 },
    )
    .unwrap();
    let s6 = c6.sample(&x6, &times).unwrap();
    let s7 = c7.sample(&x7, &times).unwrap();
    assert!(s6.state_dcd.is_empty());
    assert_eq!(s6.states, s7.states);
    assert_eq!(s6.stm, s7.stm);
    assert_eq!(s6.stm[1], M6::eye());
    assert!(s6.augmented_transition(1).is_err());
    for (config, x) in [(c6, x6), (c7, x7)] {
        let r = rr.evaluate(&x, &config).unwrap();
        let a = angles.evaluate(&x, &config).unwrap();
        // Add a third batch with a different schedule; order stays explicit.
        let short = PreparedBatch::range_rate(
            Arc::new(PreparedGeometry::new(&[epoch], &[site]).unwrap()),
            vec![0.0],
            0.05,
        )
        .unwrap();
        let single = short.evaluate(&x, &config).unwrap();
        let problem = ODProblem::new(config, vec![angles.clone(), short, rr.clone()]).unwrap();
        assert_eq!(problem.timestamps().len(), 3);
        let stacked = problem.evaluate(&x).unwrap();
        let expected: Vec<_> = a
            .residual
            .as_slice()
            .iter()
            .chain(single.residual.as_slice())
            .chain(r.residual.as_slice())
            .copied()
            .collect();
        for (actual, expected) in stacked.residual.as_slice().iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-8);
        }
        let mut offset = 0;
        for e in [&a, &single, &r] {
            for i in 0..e.residual.len() {
                for k in 0..x.len() {
                    assert!((stacked.jacobian[(offset + i, k)] - e.jacobian[(i, k)]).abs() < 1e-8);
                }
            }
            offset += e.residual.len();
        }
        let mut output = stacked.clone();
        let ptr = output.jacobian.as_slice().as_ptr();
        problem.evaluate_into(&x, &mut output).unwrap();
        assert_eq!(output, stacked);
        assert_eq!(ptr, output.jacobian.as_slice().as_ptr());
        assert!(fit_lm(problem, &x, Some(&[0.0]), &Default::default()).is_err());
    }
}

#[test]
fn thin_lm_adapter_recovers_six_state_parameters_from_stacked_models() {
    let f = support::Fixture::new(1).unwrap();
    let epoch = f.sensitivities.timestamps[0];
    let state = leo_state(f.prepared_geometry.row(0).unwrap());
    let truth = DynVector::from_slice(state.as_slice());
    let times: Vec<_> = (0..13)
        .map(|i| epoch + Duration::from_seconds(-300.0 + 50.0 * i as f64))
        .collect();
    let geometry = Arc::new(
        PreparedGeometry::new(&times, &vec![f.station.ground_stations[0]; times.len()]).unwrap(),
    );
    let config = PropagationConfig::new(
        epoch,
        od_prop_settings(),
        ParameterMode::StateOnly { fixed_drag: 0.02 },
    )
    .unwrap();
    let sens = config.sample(&truth, &times).unwrap();
    let mut rr = Vec::new();
    let mut az = Vec::new();
    let mut el = Vec::new();
    for (i, state) in sens.states.iter().enumerate() {
        let g = geometry.row(i).unwrap();
        rr.push(
            range_rate_partial(&r_of(state), &v_of(state), g.position(), g.velocity())
                .unwrap()
                .0,
        );
        let (angles, _) = az_el_partial_prepared(&r_of(state), g.position(), g.rotation()).unwrap();
        az.push(angles[0]);
        el.push(angles[1]);
    }
    let problem = ODProblem::new(
        config,
        vec![
            PreparedBatch::range_rate(geometry.clone(), rr, 0.05).unwrap(),
            PreparedBatch::azimuth_elevation(geometry, az, el, 1e-4, 2e-4).unwrap(),
        ],
    )
    .unwrap();
    let mut guess = truth.clone();
    guess[0] += 30.0;
    guess[1] -= 25.0;
    guess[2] += 15.0;
    guess[3] += 0.02;
    let settings = numeris::optim::LmSettings {
        grad_tol: 1e-6,
        x_tol: 1e-9,
        f_tol: 1e-10,
        ..Default::default()
    };
    let fit = fit_lm(problem.clone(), &guess, None, &settings).unwrap();
    assert!(fit.rss < 1e-6);
    assert!(
        (0..6)
            .map(|k| (fit.parameters[k] - truth[k]).powi(2))
            .sum::<f64>()
            .sqrt()
            < 0.01
    );
    guess[0] = f64::NAN;
    assert!(matches!(
        fit_lm(problem, &guess, None, &settings),
        Err(tychonic::optim::FitError::Kernel(_))
    ));
}
