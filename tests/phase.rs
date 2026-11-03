use numeris::{DynVector, Matrix, Vector, estimate::Ekf, optim::LmSettings};
use satkit::{Duration, Instant};
use std::sync::Arc;
use tychonic::{estimate::*, kernels::*, optim::*};

fn epoch() -> Instant {
    Instant::from_datetime(2024, 1, 1, 12, 0, 0.25).unwrap()
}
fn config(mode: ParameterMode) -> PropagationConfig {
    PropagationConfig::new(epoch(), od_prop_settings(), mode).unwrap()
}
fn geometry(times: &[Instant]) -> Arc<PreparedGeometry> {
    Arc::new(
        PreparedGeometry::new(
            times,
            &vec![
                GroundStation {
                    lat_deg: 40.0,
                    lon_deg: -75.0,
                    alt_m: 100.0
                };
                times.len()
            ],
        )
        .unwrap(),
    )
}
fn interferometer() -> Interferometer {
    Interferometer::new(v3(59.0, 3.0, 0.2), 2.23e9).unwrap()
}
fn ou() -> OuPhaseNoise {
    OuPhaseNoise::new(21600.0, 5.0_f64.to_radians()).unwrap()
}
fn truth(g: &StationGeometry) -> V6 {
    v6_from(
        &(g.position() + g.rotation().to_gcrf(&v3(80e3, 40e3, 400e3))),
        &(g.velocity() + g.rotation().to_gcrf(&v3(7100.0, 3000.0, 0.0))),
    )
}

#[test]
fn analytic_phase_derivatives_include_received_frequency_and_offset() {
    let station = v3(6378e3, 0.0, 0.0);
    let velocity = v3(0.0, 465.0, 0.0);
    for baseline in [v3(0.0, 59.0, 0.0), v3(4.0, -30.0, 20.0)] {
        for offset in [
            v3(400e3, 80e3, 40e3),
            v3(400e3, 0.0, 0.0),
            v3(1e7, -8e6, 4e6),
        ] {
            let x = v6_from(&(station + offset), &v3(20.0, 7300.0, 1000.0));
            let prediction =
                phase_difference_partial(&x, &station, &velocity, &baseline, 2.23e9, 12.0).unwrap();
            for k in 0..6 {
                let step = if k < 3 { 1.0 } else { 0.1 };
                let mut xp = x;
                let mut xm = x;
                xp[k] += step;
                xm[k] -= step;
                let a = phase_difference_partial(&xp, &station, &velocity, &baseline, 2.23e9, 12.0)
                    .unwrap();
                let b = phase_difference_partial(&xm, &station, &velocity, &baseline, 2.23e9, 12.0)
                    .unwrap();
                for (row, difference) in [
                    (0, a.doppler_hz - b.doppler_hz),
                    (1, a.phase_rad - b.phase_rad),
                ] {
                    let fd = difference / (2.0 * step);
                    let h = prediction.partial[(row, k)];
                    assert!(
                        (fd - h).abs() < 2e-10 + 2e-7 * h.abs(),
                        "row {row} column {k}: {fd} vs {h}"
                    );
                }
            }
            // Large bias perturbation resolves the very small direct coupling.
            let a =
                phase_difference_partial(&x, &station, &velocity, &baseline, 2.23e9, 1e6 + 12.0)
                    .unwrap();
            assert!(
                ((a.phase_rad - prediction.phase_rad) / 1e6
                    - prediction.phase_frequency_bias_partial)
                    .abs()
                    < 1e-15
            );
            let reversed =
                phase_difference_partial(&x, &station, &velocity, &(-baseline), 2.23e9, 12.0)
                    .unwrap();
            assert_eq!(prediction.phase_rad, -reversed.phase_rad);
        }
    }
}

#[test]
fn phase_validation_zero_baseline_and_zenith() {
    let g = geometry(&[epoch()]);
    let g = g.row(0).unwrap();
    let zenith = v6_from(
        &(g.position() + g.rotation().to_gcrf(&v3(0.0, 0.0, 400e3))),
        g.velocity(),
    );
    assert!(interferometer().evaluate(&zenith, g, 0.0).is_ok());
    let zero = Interferometer::new(V3::zeros(), 2.23e9)
        .unwrap()
        .evaluate(&zenith, g, 0.0)
        .unwrap();
    assert_eq!(zero.phase_rad, 0.0);
    for k in 0..6 {
        assert_eq!(zero.partial[(1, k)], 0.0);
    }
    assert!(Interferometer::new(v3(f64::NAN, 0.0, 0.0), 2.23e9).is_err());
    assert!(Interferometer::new(V3::zeros(), 0.0).is_err());
    assert!(
        interferometer()
            .evaluate(&v6_from(g.position(), g.velocity()), g, 0.0)
            .is_err()
    );
    assert!(
        interferometer()
            .evaluate(&zenith, g, f64::INFINITY)
            .is_err()
    );
    assert!(OuPhaseNoise::new(0.0, 1.0).is_err());
    assert!(ou().transition(-1.0).is_err());
    assert!(ou().covariance(&[epoch()], 0.0).is_err());
}

#[test]
fn exact_ou_transition_and_covariance_handle_duplicate_unsorted_times() {
    let process = ou();
    let variance = process.stationary_sigma_rad().powi(2);
    assert_eq!(process.transition(0.0).unwrap(), (1.0, 0.0));
    for dt in [1e-8, 10.0, 900.0, 21600.0, 1e9] {
        let (a, q) = process.transition(dt).unwrap();
        assert!((a * a * variance + q - variance).abs() < 1e-16);
    }
    let (a, q) = process.transition(900.0).unwrap();
    let rms_change = (variance * (1.0 - a).powi(2) + q).sqrt().to_degrees();
    assert!((rms_change - 1.428468).abs() < 1e-5);
    let times = [epoch() + Duration::from_seconds(10.0), epoch(), epoch()];
    let c = process.covariance(&times, 0.01).unwrap();
    assert_eq!(c[(1, 2)], variance);
    assert!((c[(0, 1)] - variance * (-10.0 / 21600.0_f64).exp()).abs() < 1e-16);
    assert!(c.cholesky().is_ok());
}

fn fixture(mode: ParameterMode, offsets: &[f64]) -> (PassiveRfProblem, DynVector<f64>) {
    let times: Vec<_> = offsets
        .iter()
        .map(|dt| epoch() + Duration::from_seconds(*dt))
        .collect();
    let g = geometry(&times);
    let x = truth(geometry(&[epoch()]).row(0).unwrap());
    let mut parameters = x.as_slice().to_vec();
    if mode.columns() == 7 {
        parameters.push(0.02);
    }
    parameters.extend([2.0, 0.7]);
    let parameters = DynVector::from_slice(&parameters);
    let cfg = config(mode);
    let p = mode.columns();
    let s = cfg
        .sample(&DynVector::from_slice(&parameters.as_slice()[..p]), &times)
        .unwrap();
    let predictions: Vec<_> = (0..times.len())
        .map(|i| {
            interferometer()
                .evaluate(&s.states[i], g.row(i).unwrap(), 2.0)
                .unwrap()
        })
        .collect();
    let mut sigma = vec![1000.0, 1000.0, 1000.0, 1.0, 1.0, 1.0];
    if p == 7 {
        sigma.push(0.01);
    }
    sigma.extend([10.0, 3.0]);
    let data = PassiveRfData {
        geometry: g,
        interferometer: interferometer(),
        doppler_hz: predictions.iter().map(|v| v.doppler_hz).collect(),
        phase_rad: Some(predictions.iter().map(|v| v.phase_rad + 0.7).collect()),
        sigma_doppler_hz: 0.3,
        sigma_phase_rad: 0.01,
    };
    let prior = RfPrior {
        mean: parameters.clone(),
        sigma: DynVector::from_slice(&sigma),
    };
    (
        PassiveRfProblem::new(cfg, data, prior, ou()).unwrap(),
        parameters,
    )
}

#[test]
fn rf_batch_whitening_chaining_and_nuisance_columns_match_finite_differences() {
    for mode in [
        ParameterMode::StateOnly { fixed_drag: 0.02 },
        ParameterMode::StateAndDrag { delta_cd: 1e-4 },
    ] {
        let (problem, x) = fixture(mode, &[60.25, -20.0, 0.0, 60.25]);
        let e = problem.evaluate(&x).unwrap();
        assert!(e.rss() < 1e-15);
        for k in 0..x.len() {
            let step = if k < 3 {
                0.5
            } else if k < 6 {
                0.01
            } else if mode.columns() == 7 && k == 6 {
                1e-4
            } else {
                0.1
            };
            let mut xp = x.clone();
            let mut xm = x.clone();
            xp[k] += step;
            xm[k] -= step;
            let ep = problem.evaluate(&xp).unwrap();
            let em = problem.evaluate(&xm).unwrap();
            for row in 0..e.residual.len() {
                let fd = (ep.residual[row] - em.residual[row]) / (2.0 * step);
                let h = e.jacobian[(row, k)];
                assert!(
                    (fd - h).abs() < 1e-5 + 1e-4 * h.abs(),
                    "row {row} col {k}: {fd} vs {h}"
                );
            }
        }
        assert!(
            problem
                .conditional_drift(&x)
                .unwrap()
                .iter()
                .all(|v| v.abs() < 1e-10)
        );
    }
}

#[test]
fn phase_prepared_batch_and_sensor_share_kernel_and_filter_schedule() {
    let (problem, x) = fixture(
        ParameterMode::StateOnly { fixed_drag: 0.02 },
        &[20.0, 0.0, 20.0],
    );
    let data = problem.data();
    let s = problem
        .config()
        .sample(
            &DynVector::from_slice(&x.as_slice()[..6]),
            data.geometry.timestamps(),
        )
        .unwrap();
    let phases: Vec<_> = (0..3)
        .map(|i| {
            interferometer()
                .evaluate(&s.states[i], data.geometry.row(i).unwrap(), 0.0)
                .unwrap()
                .phase_rad
        })
        .collect();
    let batch =
        PreparedBatch::phase_difference(data.geometry.clone(), interferometer(), phases, 0.01)
            .unwrap();
    let e = batch.evaluate_from_sensitivities(&s).unwrap();
    assert!(e.rss() < 1e-15);
    for i in 0..3 {
        let sensor = PhaseSensor {
            geometry: data.geometry.row(i).unwrap().clone(),
            interferometer: interferometer(),
        };
        let (_, h) = sensor.evaluate(&s.states[i]).unwrap();
        let chained = h * s.stm[i];
        for k in 0..6 {
            assert!((e.jacobian[(i, k)] - chained[(0, k)] / 0.01).abs() < 1e-12);
        }
    }
    let mut filter = OrbitEkf::<6>::new(
        problem.config().clone(),
        V6::from_slice(&x.as_slice()[..6]),
        M6::eye(),
    )
    .unwrap();
    filter
        .filter_batches(
            &[batch],
            |_, _, _| Ok(M6::zeros()),
            UpdateOptions::default(),
        )
        .unwrap();
    assert_eq!(filter.timestamps().len(), 2);
}

#[test]
fn augmented_filter_same_epoch_matches_batch_information_solution() {
    let (problem, x) = fixture(
        ParameterMode::StateOnly { fixed_drag: 0.02 },
        &[0.0, 0.0, 0.0],
    );
    let prior = problem.prior();
    let fit = fit_passive_rf(&problem, &x, &LmSettings::default()).unwrap();
    let mut filter = PassiveRfEkf::<9>::new(problem.config().clone(), ou(), prior).unwrap();
    filter
        .filter_data(problem.data(), UpdateOptions::default())
        .unwrap();
    assert_eq!(filter.timestamps().len(), 1);
    // Marginalized drift and augmented-state covariance must agree.
    for i in 0..8 {
        for j in 0..8 {
            let a = fit.covariance[(i, j)];
            let b = filter.covariance()[(i, j)];
            assert!(
                (a - b).abs() < 2e-6 + 2e-8 * a.abs(),
                "[{i},{j}] {a} vs {b}"
            );
        }
    }
    let before = *filter.state();
    let p = *filter.covariance();
    assert!(filter.predict_to(epoch(), Some(&M6::eye())).is_err());
    assert_eq!(*filter.state(), before);
    assert_eq!(*filter.covariance(), p);
    let mut bad = problem.data().clone();
    bad.doppler_hz[0] = f64::NAN;
    assert!(filter.filter_data(&bad, UpdateOptions::default()).is_err());
    assert_eq!(*filter.state(), before);
    let gated = filter
        .update(
            &Vector::from_array([1e9]),
            problem.data().geometry.row(0).unwrap(),
            interferometer(),
            &Matrix::new([[1.0]]),
            UpdateOptions { gate: Some(1.0) },
        )
        .unwrap();
    assert!(!gated.accepted);
    assert_eq!(*filter.state(), before);
}

#[test]
fn correlated_ou_batch_matches_sequential_linear_gaussian_solution() {
    // Independent information solution for mean + OU drift, with irregular epochs.
    let offsets = [0.0, 10.0, 100.0, 100.0, 900.0];
    let times: Vec<_> = offsets
        .iter()
        .map(|dt| epoch() + Duration::from_seconds(*dt))
        .collect();
    let values = [0.1, 0.12, 0.08, 0.11, 0.09];
    let sigma = 0.02;
    let c = ou().covariance(&times, sigma).unwrap();
    let inv = c.cholesky().unwrap().inverse();
    let posterior_variance = 1.0 / (1.0 + inv.as_slice().iter().sum::<f64>());
    let posterior_mean = posterior_variance
        * (0..values.len())
            .map(|i| {
                (0..values.len())
                    .map(|j| inv[(i, j)] * values[j])
                    .sum::<f64>()
            })
            .sum::<f64>();
    let mut f = Ekf::<f64, 2, 1>::new(
        Vector::from_array([0.0, 0.0]),
        Matrix::new([[1.0, 0.0], [0.0, ou().stationary_sigma_rad().powi(2)]]),
    );
    for i in 0..offsets.len() {
        let (a, q) = ou()
            .transition(if i == 0 {
                0.0
            } else {
                offsets[i] - offsets[i - 1]
            })
            .unwrap();
        f.predict(
            |x| Vector::from_array([x[0], a * x[1]]),
            |_| Matrix::new([[1.0, 0.0], [0.0, a]]),
            Some(&Matrix::new([[0.0, 0.0], [0.0, q]])),
        );
        f.update(
            &Vector::from_array([values[i]]),
            |x| Vector::from_array([x[0] + x[1]]),
            |_| Matrix::new([[1.0, 1.0]]),
            &Matrix::new([[sigma * sigma]]),
        )
        .unwrap();
    }
    assert!((f.state()[0] - posterior_mean).abs() < 1e-12);
    assert!((f.covariance()[(0, 0)] - posterior_variance).abs() < 1e-12);
}

#[test]
fn noisy_pass_recovers_offsets_and_supports_augmented_drag() {
    let mode = ParameterMode::StateAndDrag { delta_cd: 1e-4 };
    let (problem, x) = fixture(mode, &[0.0, 20.0, 40.0, 60.0, 80.0]);
    let mut guess = x.clone();
    guess[0] += 10.0;
    guess[3] += 0.01;
    let fit = fit_passive_rf(
        &problem,
        &guess,
        &LmSettings {
            grad_tol: 1e-4,
            ..LmSettings::default()
        },
    )
    .unwrap();
    assert!(fit.rss < 1e-7);
    let mut filter =
        PassiveRfEkf::<10>::new(problem.config().clone(), ou(), problem.prior()).unwrap();
    filter
        .filter_data(problem.data(), UpdateOptions::default())
        .unwrap();
    assert!((filter.state()[6] - x[6]).abs() < 1e-10);
    let smooth = filter.smooth().unwrap();
    assert_eq!(smooth.last().unwrap().0, *filter.state());
    assert!(PassiveRfEkf::<9>::new(problem.config().clone(), ou(), problem.prior()).is_err());
    let mut data = problem.data().clone();
    for i in 0..data.doppler_hz.len() {
        data.doppler_hz[i] += 0.1 * (i as f64).sin();
        data.phase_rad.as_mut().unwrap()[i] += 0.001 * (i as f64).cos();
    }
    let noisy = PassiveRfProblem::new(
        problem.config().clone(),
        data,
        problem.prior().clone(),
        ou(),
    )
    .unwrap();
    let a = fit_passive_rf(
        &noisy,
        &x,
        &LmSettings {
            grad_tol: 1e-3,
            ..LmSettings::default()
        },
    )
    .unwrap();
    let b = fit_passive_rf(
        &noisy,
        &x,
        &LmSettings {
            grad_tol: 1e-3,
            ..LmSettings::default()
        },
    )
    .unwrap();
    assert_eq!(a.parameters, b.parameters);
    assert!(a.rss > 0.0 && a.rss < 1.0);
    assert!(a.covariance.cholesky().is_ok());
}

#[test]
fn iteration_limit_requires_verified_local_convergence() {
    let (problem, x) = fixture(
        ParameterMode::StateOnly { fixed_drag: 0.02 },
        &[0.0, 20.0, 40.0],
    );
    let settings = LmSettings {
        max_iter: 0,
        ..LmSettings::default()
    };
    let stationary = fit_passive_rf(&problem, &x, &settings).unwrap();
    assert!(stationary.linearized_tolerance);
    let mut bad = x.clone();
    bad[0] += 1000.0;
    bad[3] += 1.0;
    assert!(fit_passive_rf(&problem, &bad, &settings).is_err());
}
