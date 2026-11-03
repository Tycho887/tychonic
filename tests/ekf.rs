use numeris::{DynVector, Matrix, Vector, estimate::rts_smooth, optim::LmSettings};
use satkit::{Duration, Instant, orbitprop::SatPropertiesSimple};
use std::{f64::consts::PI, sync::Arc};
use tychonic::{
    estimate::*,
    kernels::*,
    optim::{ODProblem, fit_lm},
};

fn epoch() -> Instant {
    Instant::from_datetime(2024, 1, 1, 12, 0, 0.25).unwrap()
}
fn leo() -> V6 {
    v6_from(&v3(6678e3, 0.0, 0.0), &v3(0.0, 7100.0, 3000.0))
}
fn config(mode: ParameterMode) -> PropagationConfig {
    PropagationConfig::new(epoch(), od_prop_settings(), mode).unwrap()
}
fn covariance<const N: usize>() -> Matrix<f64, N, N> {
    let mut p = Matrix::zeros();
    for i in 0..N {
        p[(i, i)] = if i < 3 {
            1e4
        } else if i < 6 {
            0.01
        } else {
            1e-4
        };
    }
    p
}
fn assert_close<const R: usize, const C: usize>(
    a: &Matrix<f64, R, C>,
    b: &Matrix<f64, R, C>,
    atol: f64,
    rtol: f64,
) {
    for i in 0..R {
        for j in 0..C {
            assert!(
                (a[(i, j)] - b[(i, j)]).abs() <= atol + rtol * b[(i, j)].abs(),
                "[{i},{j}]: {} != {}",
                a[(i, j)],
                b[(i, j)]
            );
        }
    }
}

#[test]
fn heterogeneous_correlated_linear_sensors_match_information_solution() {
    let initial = leo();
    let p0 = covariance::<6>();
    let mut filter = OrbitEkf::new(
        config(ParameterMode::StateOnly { fixed_drag: 0.0 }),
        initial,
        p0,
    )
    .unwrap();
    let h2 = Matrix::new([
        [1.0, 0.2, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.1, 0.0, 0.0, 0.0],
    ]);
    let r2 = Matrix::new([[25.0, 8.0], [8.0, 16.0]]);
    let z2 = h2 * initial + Vector::from_array([20.0, -15.0]);
    let first = filter
        .update(
            &z2,
            &|x: &V6| Ok((h2 * x, h2)),
            &r2,
            UpdateOptions::default(),
        )
        .unwrap();
    assert!(first.accepted && first.nis > 0.0);
    let h1 = Matrix::new([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]);
    let r1 = Matrix::new([[9.0]]);
    let z1 = h1 * initial + Vector::from_array([7.0]);
    filter
        .update(
            &z1,
            &|x: &V6| Ok((h1 * x, h1)),
            &r1,
            UpdateOptions::default(),
        )
        .unwrap();

    // Independent information-form posterior in correction coordinates avoids
    // subtracting large orbital position terms in the reference calculation.
    let w2 = r2.cholesky().unwrap().inverse();
    let w1 = r1.cholesky().unwrap().inverse();
    let information =
        p0.cholesky().unwrap().inverse() + h2.transpose() * w2 * h2 + h1.transpose() * w1 * h1;
    let expected_p = information.cholesky().unwrap().inverse();
    let expected_x = initial
        + expected_p
            * (h2.transpose() * w2 * (z2 - h2 * initial)
                + h1.transpose() * w1 * (z1 - h1 * initial));
    assert_close(filter.state(), &expected_x, 1e-9, 1e-13);
    assert_close(filter.covariance(), &expected_p, 1e-9, 1e-12);
    assert_eq!(filter.steps().len(), 1);
    assert_eq!(filter.steps()[0].x_predicted, initial);
    assert_eq!(filter.steps()[0].x_updated, *filter.state());
}

#[test]
fn prediction_uses_local_stm_and_augmented_drag_transition() {
    let t = epoch();
    let end = t + Duration::from_seconds(60.125);
    let initial = leo();
    let p0 = covariance::<6>();
    let q = white_acceleration_noise::<6>(60.125, &v3(1e-8, 0.0, 2e-8), 0.0).unwrap();
    let mut filter = OrbitEkf::new(
        config(ParameterMode::StateOnly { fixed_drag: 0.02 }),
        initial,
        p0,
    )
    .unwrap();
    let (expected_x, phi) =
        propagate_with_stm(&initial, &t, &end, &od_prop_settings(), 0.02).unwrap();
    filter.predict_to(end, Some(&q)).unwrap();
    assert_close(filter.state(), &expected_x, 0.0, 0.0);
    assert_close(
        filter.covariance(),
        &(phi * p0 * phi.transpose() + q),
        1e-9,
        1e-13,
    );
    assert_eq!(filter.steps()[1].f_jacobian, phi);

    // The next prediction must start from the updated orbit, not the original
    // batch reference state, and its history must retain that local Jacobian.
    let h = Matrix::new([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]);
    filter
        .update(
            &Vector::from_array([expected_x[0] + 10.0]),
            &|x: &V6| Ok((h * x, h)),
            &Matrix::new([[1.0]]),
            UpdateOptions::default(),
        )
        .unwrap();
    let next = end + Duration::from_seconds(10.0);
    let (next_x, next_phi) =
        propagate_with_stm(filter.state(), &end, &next, &od_prop_settings(), 0.02).unwrap();
    filter.predict_to(next, None).unwrap();
    assert_eq!(filter.state(), &next_x);
    assert_eq!(filter.steps()[2].f_jacobian, next_phi);

    let mut x7 = V7::zeros();
    x7.set_block(0, 0, &initial);
    x7[6] = 0.02;
    let p7 = covariance::<7>();
    let mut augmented = OrbitEkf::new(
        config(ParameterMode::StateAndDrag { delta_cd: 1e-4 }),
        x7,
        p7,
    )
    .unwrap();
    let sens =
        compute_sensitivities(&initial, &t, &[end], &od_prop_settings(), 0.02, 1e-4).unwrap();
    let f7 = sens.augmented_transition(0).unwrap();
    augmented.predict_to(end, None).unwrap();
    assert_eq!(augmented.state()[6], 0.02);
    assert_eq!(augmented.steps()[1].f_jacobian, f7);
    assert_close(
        augmented.covariance(),
        &(f7 * p7 * f7.transpose()),
        1e-9,
        1e-13,
    );
    // A geometric reading estimates drag through the propagated cross covariance.
    let before_drag = augmented.state()[6];
    augmented
        .update(
            &Vector::from_array([sens.states[0][0] + 0.1]),
            &|x: &V6| Ok((h * x, h)),
            &Matrix::new([[1.0]]),
            UpdateOptions::default(),
        )
        .unwrap();
    assert_ne!(augmented.state()[6], before_drag);
    assert!(augmented.smooth().is_ok());
}

fn station() -> GroundStation {
    GroundStation {
        lat_deg: 40.0,
        lon_deg: -75.0,
        alt_m: 100.0,
    }
}

#[test]
fn ground_sensors_reuse_partials_and_wrap_azimuth_across_branch_cut() {
    // External geometry provides an exact ENU basis and a branch-cut geometry.
    let t = epoch();
    let prepared = PreparedGeometry::from_trajectories(
        GCRFTrajectory {
            timestamps: vec![t],
            ground_stations: vec![station()],
            position: vec![V3::zeros()],
            velocity: vec![v3(1.0, 2.0, 3.0)],
        },
        EnuBasisTrajectory {
            timestamps: vec![t],
            enu_to_gcrf: vec![M3::eye()],
        },
    )
    .unwrap();
    let g = prepared.row(0).unwrap().clone();
    let initial = v6_from(&v3(1e-3, -6678e3, 200e3), &v3(0.0, 7100.0, 3000.0));
    let rr_model = GroundSensor::<1>::new(g.clone()).unwrap();
    let az_model = GroundSensor::<2>::new(g.clone()).unwrap();
    let combined = GroundSensor::<3>::new(g).unwrap();
    let (rr, hr) = rr_model.evaluate(&initial).unwrap();
    let (angles, ha) = az_model.evaluate(&initial).unwrap();
    let (z, h) = combined.evaluate(&initial).unwrap();
    assert_eq!(z[0], rr[0]);
    assert_eq!(h.block::<1, 6>(0, 0), hr);
    assert_eq!(h.block::<2, 6>(1, 0), ha);
    assert!((angles[0] - PI).abs() < 1e-8);
    let mut shifted = angles;
    shifted[0] -= 2.0 * PI;
    shifted[0] += 1e-8;
    let mut filter = OrbitEkf::new(
        config(ParameterMode::StateOnly { fixed_drag: 0.0 }),
        initial,
        covariance::<6>(),
    )
    .unwrap();
    let result = filter
        .update(
            &shifted,
            &az_model,
            &(Matrix::eye() * 1e-8),
            UpdateOptions { gate: Some(9.21) },
        )
        .unwrap();
    assert!(result.accepted);
    assert!((result.innovation[0] - 1e-8).abs() < 1e-15);
    assert!(result.nis < 1e-6);
}

#[test]
fn gating_and_errors_preserve_filter_and_history() {
    let mut filter = OrbitEkf::new(
        config(ParameterMode::StateOnly { fixed_drag: 0.0 }),
        leo(),
        covariance::<6>(),
    )
    .unwrap();
    let h = Matrix::new([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]);
    let model = |x: &V6| Ok((h * x, h));
    let r = Matrix::new([[1.0]]);
    let x = *filter.state();
    let p = *filter.covariance();
    let outlier = filter
        .update(
            &Vector::from_array([x[0] + 1e6]),
            &model,
            &r,
            UpdateOptions { gate: Some(6.63) },
        )
        .unwrap();
    assert!(!outlier.accepted && outlier.nis > 6.63);
    assert_eq!(*filter.state(), x);
    assert_eq!(*filter.covariance(), p);
    assert_eq!(filter.steps()[0].x_updated, x);

    let bad = Observation {
        time: epoch() + Duration::from_seconds(10.0),
        value: Vector::from_array([f64::NAN]),
        covariance: r,
        model,
    };
    assert!(
        filter
            .observe(&bad, None, UpdateOptions::default())
            .is_err()
    );
    assert_eq!(filter.epoch(), epoch());
    assert_eq!(*filter.state(), x);
    assert_eq!(*filter.covariance(), p);
    assert_eq!(filter.steps().len(), 1);
    assert_eq!(filter.timestamps().len(), 1);
    assert!(
        filter
            .predict_to(epoch() - Duration::from_seconds(1.0), None)
            .is_err()
    );
    assert!(filter.predict_to(epoch(), Some(&M6::eye())).is_err());
    filter.predict_to(epoch(), Some(&M6::zeros())).unwrap();
    assert_eq!(filter.steps().len(), 1);

    let observation = Observation {
        time: bad.time,
        value: Vector::from_array([1e9]),
        covariance: r,
        model,
    };
    let rejected = filter
        .observe(&observation, None, UpdateOptions { gate: Some(6.63) })
        .unwrap();
    assert!(!rejected.accepted);
    assert_eq!(filter.steps().len(), 2);
    assert_eq!(filter.steps()[1].x_predicted, filter.steps()[1].x_updated);
}

#[test]
fn invalid_covariances_models_and_process_noise_return_errors() {
    let c = config(ParameterMode::StateOnly { fixed_drag: 0.0 });
    assert!(OrbitEkf::new(c.clone(), V7::zeros(), covariance::<7>()).is_err());
    assert!(OrbitEkf::new(c.clone(), leo(), M6::zeros()).is_err());
    let mut bad = M6::eye();
    bad[(0, 1)] = 2.0;
    assert!(OrbitEkf::new(c.clone(), leo(), bad).is_err());
    bad[(1, 0)] = 2.0;
    assert!(OrbitEkf::new(c.clone(), leo(), bad).is_err());
    let mut filter = OrbitEkf::new(c, leo(), covariance::<6>()).unwrap();
    let end = epoch() + Duration::from_seconds(1.0);
    assert!(filter.predict_to(end, Some(&bad)).is_err());
    assert!(filter.predict_to(end, Some(&(M6::eye() * -1.0))).is_err());
    let mut invalid_model = Matrix::<f64, 1, 6>::zeros();
    invalid_model[(0, 0)] = f64::NAN;
    assert!(
        filter
            .update(
                &Vector::zeros(),
                &|_: &V6| Ok((Vector::zeros(), invalid_model)),
                &Matrix::eye(),
                UpdateOptions::default()
            )
            .is_err()
    );
    let model = |x: &V6| {
        Ok((
            Vector::from_array([x[0]]),
            Matrix::new([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
        ))
    };
    for gate in [Some(-1.0), Some(f64::NAN)] {
        assert!(
            filter
                .update(
                    &Vector::zeros(),
                    &model,
                    &Matrix::eye(),
                    UpdateOptions { gate }
                )
                .is_err()
        );
    }
    assert!(
        filter
            .update(
                &Vector::zeros(),
                &model,
                &Matrix::zeros(),
                UpdateOptions::default()
            )
            .is_err()
    );
    assert!(white_acceleration_noise::<6>(-1.0, &V3::zeros(), 0.0).is_err());
    assert!(white_acceleration_noise::<6>(1.0, &V3::zeros(), 1e-8).is_err());
    assert!(white_acceleration_noise::<7>(1.0, &v3(-1.0, 0.0, 0.0), 0.0).is_err());
    let q = white_acceleration_noise::<7>(2.0, &v3(3.0, 0.0, 0.0), 5.0).unwrap();
    assert_eq!(q[(0, 0)], 8.0);
    assert_eq!(q[(0, 3)], 6.0);
    assert_eq!(q[(3, 3)], 6.0);
    assert_eq!(q[(6, 6)], 10.0);
    // A valid, rank-one process covariance is allowed.
    let v = numeris::vector![1.0, 2.0, 0.0, 0.1, 0.0, 0.0];
    filter.predict_to(end, Some(&(v * v.transpose()))).unwrap();
    assert!(filter.covariance().cholesky().is_ok());
}

#[test]
fn mixed_station_batches_sort_exact_epochs_and_rts_agrees_with_batch_orbit_fit() {
    let reference = epoch();
    let truth = leo();
    let mut times: Vec<_> = (0..21)
        .map(|i| reference + Duration::from_seconds(i as f64 * 15.125))
        .collect();
    // Caller order and duplicate epochs differ from the filter's timeline.
    times.reverse();
    times.push(times[5]);
    let sites: Vec<_> = (0..times.len())
        .map(|i| {
            if i % 2 == 0 {
                station()
            } else {
                GroundStation {
                    lat_deg: 35.0,
                    lon_deg: -105.0,
                    alt_m: 2100.0,
                }
            }
        })
        .collect();
    let geometry = Arc::new(PreparedGeometry::new(&times, &sites).unwrap());
    let mut rr = Vec::new();
    let mut az = Vec::new();
    let mut el = Vec::new();
    for (i, time) in times.iter().enumerate() {
        // State-only endpoints are independent of the STM prediction path.
        let state = propagate_state(
            &truth,
            &reference,
            time,
            &od_prop_settings(),
            &SatPropertiesSimple::new(0.0, 0.0),
        )
        .unwrap();
        let (z, _) = GroundSensor::<3>::new(geometry.row(i).unwrap().clone())
            .unwrap()
            .evaluate(&state)
            .unwrap();
        rr.push(z[0]);
        az.push(z[1]);
        el.push(z[2]);
    }
    let batches = vec![
        PreparedBatch::range_rate(geometry.clone(), rr, 0.01).unwrap(),
        PreparedBatch::azimuth_elevation(geometry, az, el, 1e-6, 1e-6).unwrap(),
    ];
    let cfg = config(ParameterMode::StateOnly { fixed_drag: 0.0 });
    let problem = ODProblem::new(cfg.clone(), batches.clone()).unwrap();
    let initial = truth + numeris::vector![10.0, -8.0, 6.0, 0.01, -0.01, 0.005];
    let batch_fit = fit_lm(
        problem,
        &DynVector::from_slice(initial.as_slice()),
        None,
        &LmSettings {
            max_iter: 30,
            grad_tol: 1e-3,
            ..LmSettings::default()
        },
    )
    .unwrap();
    let batch_state = V6::from_slice(batch_fit.parameters.as_slice());
    let mut filter = OrbitEkf::new(cfg, initial, covariance::<6>()).unwrap();
    let mut noise_calls = 0;
    let updates = filter
        .filter_batches(
            &batches,
            |_, from, to| {
                assert!(to > from);
                noise_calls += 1;
                Ok(M6::zeros())
            },
            UpdateOptions::default(),
        )
        .unwrap();
    assert_eq!(updates.len(), times.len() * 2);
    assert!(updates.windows(2).all(|w| w[0].time <= w[1].time));
    assert!(updates.iter().all(|u| u.accepted && u.nis.is_finite()));
    assert_eq!(noise_calls, 20);
    assert_eq!(filter.steps().len(), 21);
    // Stable ties preserve batch ordering and duplicate rows within each batch.
    for pair in updates.windows(2).filter(|w| w[0].time == w[1].time) {
        assert!((pair[0].batch, pair[0].row) < (pair[1].batch, pair[1].row));
    }
    let smoothed = filter.smooth().unwrap();
    let direct = rts_smooth(filter.steps()).unwrap();
    assert_eq!(smoothed, direct);
    assert_eq!(smoothed.last().unwrap().0, *filter.state());
    assert_eq!(smoothed.last().unwrap().1, *filter.covariance());
    for (i, (_, p)) in smoothed.iter().enumerate() {
        assert!(p.as_slice().iter().all(|v| v.is_finite()));
        assert!(p.cholesky().is_ok());
        for j in 0..6 {
            assert!(p[(j, j)] <= filter.steps()[i].p_updated[(j, j)] + 1e-8);
        }
    }
    let initial_smoothed = smoothed[0].0;
    assert!((r_of(&initial_smoothed) - r_of(&truth)).norm() < 0.1);
    assert!((v_of(&initial_smoothed) - v_of(&truth)).norm() < 0.002);
    assert!((r_of(&initial_smoothed) - r_of(&batch_state)).norm() < 0.1);
    assert!((v_of(&initial_smoothed) - v_of(&batch_state)).norm() < 0.002);
}

#[test]
fn combined_measurement_records_fuse_noisy_orbit_with_estimated_drag() {
    let truth = leo();
    let noise = MeasurementNoise {
        sigma_rr: 0.01,
        sigma_az: 1e-6,
        sigma_el: 1e-6,
    };
    let times: Vec<_> = (0..25)
        .map(|i| epoch() + Duration::from_seconds(i as f64 * 15.0))
        .collect();
    let sites: Vec<_> = (0..times.len())
        .map(|i| {
            if i % 2 == 0 {
                station()
            } else {
                GroundStation {
                    lat_deg: 35.0,
                    lon_deg: -105.0,
                    alt_m: 2100.0,
                }
            }
        })
        .collect();
    let geometry = PreparedGeometry::new(&times, &sites).unwrap();
    let readings: Vec<_> = times
        .iter()
        .enumerate()
        .map(|(i, time)| {
            let state = propagate_state(
                &truth,
                &epoch(),
                time,
                &od_prop_settings(),
                &SatPropertiesSimple::new(0.02, 0.0),
            )
            .unwrap();
            let (predicted, _) = GroundSensor::<3>::new(geometry.row(i).unwrap().clone())
                .unwrap()
                .evaluate(&state)
                .unwrap();
            Measurement {
                time: *time,
                station: sites[i],
                range_rate: predicted[0] + noise.sigma_rr * (i as f64 * 1.7).sin(),
                azimuth: predicted[1] + noise.sigma_az * (i as f64 * 2.3).cos(),
                elevation: predicted[2] + noise.sigma_el * (i as f64 * 0.7).sin(),
            }
        })
        .collect();
    let batch = PreparedBatch::from_measurements(&readings, noise).unwrap();
    let initial = truth + numeris::vector![50.0, -40.0, 30.0, 0.05, -0.03, 0.01];
    let mut initial7 = V7::zeros();
    initial7.set_block(0, 0, &initial);
    initial7[6] = 0.02;
    let mut filter = OrbitEkf::new(
        config(ParameterMode::StateAndDrag { delta_cd: 1e-4 }),
        initial7,
        covariance::<7>(),
    )
    .unwrap();
    let updates = filter
        .filter_batches(
            &[batch],
            |_, from, to| {
                white_acceleration_noise((to - from).as_seconds(), &v3(1e-10, 1e-10, 1e-10), 1e-12)
            },
            UpdateOptions::default(),
        )
        .unwrap();
    assert_eq!(updates.len(), times.len());
    assert!(updates.iter().all(|u| u.accepted && u.nis.is_finite()));
    let smoothed = filter.smooth().unwrap();
    let smoothed_initial = smoothed[0].0.block::<6, 1>(0, 0);
    assert!((r_of(&smoothed_initial) - r_of(&truth)).norm() < 10.0);
    assert!((v_of(&smoothed_initial) - v_of(&truth)).norm() < 0.02);
    assert!((smoothed[0].0[6] - 0.02).abs() < 0.005);
    assert_eq!(filter.timestamps(), &times);
    for (i, (x, p)) in smoothed.iter().enumerate() {
        assert!(x.as_slice().iter().all(|v| v.is_finite()));
        assert!(p.cholesky().is_ok());
        for j in 0..7 {
            assert!(p[(j, j)] <= filter.steps()[i].p_updated[(j, j)] + 1e-8);
        }
    }
}
