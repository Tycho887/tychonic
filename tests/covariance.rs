#![allow(clippy::disallowed_methods)]
//! Absolute calibrated covariance is independent of residual noise realization.
use tychonic::{
    kernels::*,
    measurements::{self as m, Observable},
    numeris::{DynMatrix, DynVector, optim::LmSettings},
    optim::{
        CovarianceError, CovarianceMethod, ODProblem, fit_lm_with_covariance, parameter_covariance,
    },
    satkit::{Duration, Instant},
};

fn close(a: f64, b: f64, atol: f64, rtol: f64) {
    assert!((a - b).abs() <= atol + rtol * b.abs(), "{a} != {b}");
}

#[test]
fn absolute_linear_reference_does_not_require_residual_degrees_of_freedom() {
    let j = DynMatrix::from_rows(2, 2, &[2.0, 1.0, 0.0, 3.0]);
    let cov = parameter_covariance(&j).unwrap();
    assert_eq!(cov.rank, 2);
    assert_eq!(cov.method, CovarianceMethod::AbsoluteWhitenedInformation);
    for (i, row) in [[5.0 / 18.0, -1.0 / 18.0], [-1.0 / 18.0, 1.0 / 9.0]]
        .iter()
        .enumerate()
    {
        for (k, expected) in row.iter().enumerate() {
            close(cov.matrix[(i, k)], *expected, 1e-14, 1e-13);
        }
    }
}

#[test]
fn correlated_noise_and_column_units_preserve_information() {
    // H = diag(1e-6, 1e3), R = [[4, 2], [2, 5]], L = [[2, 0], [1, 2]].
    // J = L^-1 H; covariance = H^-1 R H^-T.
    let j = DynMatrix::from_rows(2, 2, &[0.5e-6, 0.0, -0.25e-6, 500.0]);
    let cov = parameter_covariance(&j).unwrap();
    for (i, row) in [[4e12, 2000.0], [2000.0, 5e-6]].iter().enumerate() {
        for (k, expected) in row.iter().enumerate() {
            close(cov.matrix[(i, k)], *expected, 1e-12, 1e-12);
        }
    }
}

#[test]
fn singular_underdetermined_and_nonfinite_jacobians_return_structured_errors() {
    for j in [
        DynMatrix::from_rows(3, 2, &[1.0, 2.0, 2.0, 4.0, 3.0, 6.0]),
        DynMatrix::from_rows(1, 2, &[1.0, 0.0]),
    ] {
        assert!(matches!(
            parameter_covariance(&j),
            Err(CovarianceError::RankDeficient {
                rank: 1,
                parameters: 2
            })
        ));
    }
    assert!(matches!(
        parameter_covariance(&DynMatrix::zeros(0, 2)),
        Err(CovarianceError::InvalidInput(_))
    ));
    for bad in [f64::NAN, f64::INFINITY] {
        let j = DynMatrix::from_rows(2, 2, &[1.0, 0.0, 0.0, bad]);
        assert!(matches!(
            parameter_covariance(&j),
            Err(CovarianceError::InvalidInput(_))
        ));
    }
}

fn config(drag: bool) -> PropagationConfig {
    PropagationConfig::new(
        Instant::from_unixtime(1704110400.25),
        od_prop_settings(),
        if drag {
            ParameterMode::StateAndDrag { delta_cd: 1e-4 }
        } else {
            ParameterMode::StateOnly { fixed_drag: 0.02 }
        },
    )
    .unwrap()
}

fn fixture(
    drag: bool,
    noise_multiplier: f64,
) -> (Vec<PreparedBatch<'static>>, DynVector<f64>, Vec<Vec<f64>>) {
    let reference = config(drag).reference_epoch();
    let first = GroundStation {
        lat_deg: 40.0,
        lon_deg: -75.0,
        alt_m: 100.0,
    };
    let second = GroundStation {
        lat_deg: 40.3,
        lon_deg: -74.8,
        alt_m: 120.0,
    };
    let base = PreparedGeometry::new(&[reference], &[first]).unwrap();
    let g = base.row(0).unwrap();
    let state = v6_from(
        &(g.position() + g.rotation().to_gcrf(&v3(80e3, 40e3, 400e3))),
        &(g.velocity() + g.rotation().to_gcrf(&v3(7100.0, 3000.0, 0.0))),
    );
    let mut x = state.as_slice().to_vec();
    if drag {
        x.push(0.02);
    }
    let x = DynVector::from_slice(&x);
    let seconds: Vec<_> = (0..42).map(|i| f64::from(i) * 8.0).collect();
    let times: Vec<_> = seconds
        .iter()
        .map(|s| reference + Duration::from_seconds(*s))
        .collect();
    let stations: Vec<_> = (0..times.len())
        .map(|i| if i < 21 { first } else { second })
        .collect();
    let geometry = PreparedGeometry::new(&times, &stations).unwrap();
    let sens = config(drag).sample(&x, &times).unwrap();
    let interferometer = Interferometer::new(v3(59.0, 3.0, 0.2), 2.23e9).unwrap();
    let models = vec![
        Observable::Doppler {
            carrier_hz: 2.23e9,
            offset_hz: 0.0,
        },
        Observable::PhaseDifference {
            interferometer,
            frequency_offset_hz: 0.0,
            phase_offset_rad: 0.0,
        },
        Observable::DifferentialDoppler { interferometer },
        Observable::AzimuthElevation,
    ];
    let sigmas = [0.5, 0.01, 0.7, 1e-4, 2e-4];
    let noise = DynMatrix::from_fn(5, 5, |i, j| {
        sigmas[i] * sigmas[j] * if i == j { 1.0 } else { 0.15 }
    });
    let mut all = vec![];
    let mut batches = vec![];
    for i in 0..times.len() {
        let g = geometry.row(i).unwrap();
        let state = &sens.states()[i];
        let mut z = vec![
            -2.23e9 / SPEED_OF_LIGHT
                * range_rate_partial(&r_of(state), &v_of(state), g.position(), g.velocity())
                    .unwrap()
                    .0,
            interferometer.evaluate(state, g, 0.0).unwrap().phase_rad,
            interferometer
                .differential(state, g, times[i])
                .unwrap()
                .difference_hz,
        ];
        z.extend(
            az_el_partial_prepared(&r_of(state), g.position(), g.rotation())
                .unwrap()
                .0
                .as_slice(),
        );
        for (component, value) in z.iter_mut().enumerate() {
            *value += noise_multiplier
                * sigmas[component]
                * (i as f64 * 1.731 + component as f64 * 0.827).sin();
        }
        all.push(z.clone());
        batches.push(
            m::joint(
                PreparedGeometry::new(&[times[i]], &[stations[i]]).unwrap(),
                models.clone(),
                vec![z],
                vec![noise.clone()],
            )
            .unwrap(),
        );
    }
    (batches, x, all)
}

fn settings() -> LmSettings<f64> {
    LmSettings {
        max_iter: 400,
        grad_tol: 1e-4,
        x_tol: 1e-8,
        f_tol: 1e-10,
        ..LmSettings::default()
    }
}

#[test]
fn mixed_fit_native_covariance_matches_final_information_and_scale_changes() {
    let (batches, x, _) = fixture(false, 0.3);
    let mut guess = DynVector::from_slice(x.as_slice());
    guess[0] += 30.0;
    guess[4] += 0.02;
    let fitted =
        m::fit_with_covariance(config(false), &batches, &guess, None, &settings()).unwrap();
    assert_eq!(fitted.rank, 6);
    assert_eq!(
        fitted.covariance_method,
        CovarianceMethod::AbsoluteWhitenedInformation
    );
    assert!(
        (0..6)
            .map(|k| (fitted.parameters[k] - x[k]).powi(2))
            .sum::<f64>()
            .sqrt()
            < 100.0
    );
    let problem = ODProblem::new(config(false), &batches).unwrap();
    let final_evaluation = problem.evaluate(&fitted.parameters).unwrap();
    let expected = parameter_covariance(&final_evaluation.jacobian).unwrap();
    let scaled = fit_lm_with_covariance(
        &problem,
        &guess,
        Some(&[1.0, 2.0, 3.0, 0.01, 0.02, 0.03]),
        &settings(),
    )
    .unwrap();
    for i in 0..6 {
        assert!(fitted.covariance[(i, i)] > 0.0);
        for j in 0..6 {
            close(
                fitted.covariance[(i, j)],
                expected.matrix[(i, j)],
                1e-8,
                1e-8,
            );
            close(
                fitted.covariance[(i, j)],
                fitted.covariance[(j, i)],
                0.0,
                0.0,
            );
            close(
                scaled.covariance[(i, j)],
                fitted.covariance[(i, j)],
                1e-5,
                1e-5,
            );
        }
    }
}

#[test]
fn seven_parameter_covariance_uses_physical_drag_units() {
    let (batches, x, _) = fixture(true, 0.0);
    let fit = m::fit_with_covariance(config(true), batches, &x, None, &settings()).unwrap();
    assert_eq!(fit.rank, 7);
    assert_eq!(fit.covariance.nrows(), 7);
    assert!((0..7).all(|i| fit.covariance[(i, i)].is_finite() && fit.covariance[(i, i)] > 0.0));
}

#[test]
fn exact_fit_covariance_stays_positive_and_old_api_agrees() {
    let (batches, x, _) = fixture(false, 0.0);
    let old = m::fit_with_options(config(false), &batches, &x, None, &settings()).unwrap();
    let new = m::fit_with_covariance(config(false), &batches, &x, None, &settings()).unwrap();
    assert_eq!(old.parameters, new.parameters);
    assert_eq!(old.iterations, new.iterations);
    assert_eq!(old.rss, new.rss);
    assert!((0..6).all(|i| new.covariance[(i, i)] > 0.0));
}

#[test]
fn large_batches_use_compact_factorization() {
    let j = DynMatrix::from_fn(
        20_000,
        2,
        |row, column| {
            if row % 2 == column { 2.0 } else { 0.0 }
        },
    );
    let covariance = parameter_covariance(&j).unwrap();
    close(covariance.matrix[(0, 0)], 1.0 / 40_000.0, 1e-16, 1e-12);
    close(covariance.matrix[(1, 1)], 1.0 / 40_000.0, 1e-16, 1e-12);
}

#[test]
fn physical_covariance_overflow_is_an_error_instead_of_an_invalid_matrix() {
    let j = DynMatrix::from_rows(2, 2, &[1e-200, 0.0, 0.0, 1e-200]);
    assert!(matches!(
        parameter_covariance(&j),
        Err(CovarianceError::Numerical(_))
    ));
}

#[test]
fn converged_but_unobservable_fit_returns_covariance_error() {
    let (batches, x, _) = fixture(false, 0.0);
    // One epoch has five mixed components for six unknown state coordinates.
    let batches = batches.into_iter().take(1).collect::<Vec<_>>();
    let error = m::fit_with_covariance(config(false), batches, &x, None, &settings()).unwrap_err();
    assert!(matches!(
        error,
        tychonic::optim::CovarianceFitError::Covariance(CovarianceError::RankDeficient {
            parameters: 6,
            ..
        })
    ));
}

#[test]
fn generation_two_fixture_has_later_held_out_observations() {
    let (_, x, values) = fixture(false, 0.3);
    let mut prior = x.as_slice().to_vec();
    prior[0] += 30.0;
    prior[4] += 0.02;
    let sigmas = [0.5, 0.01, 0.7, 1e-4, 2e-4];
    let covariance = (0..5)
        .map(|i| {
            (0..5)
                .map(|j| sigmas[i] * sigmas[j] * if i == j { 1.0 } else { 0.15 })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let rows = values.iter().enumerate().map(|(i, z)| {
        let site = if i < 21 { (40.0, -75.0, 100.0, 0) } else { (40.3, -74.8, 120.0, 1) };
        format!("{{\"observation_id\":\"mixed-{i}\",\"contact_id\":\"00000000-0000-0000-0000-00000000000{}\",\"antenna_id\":\"antenna-{}\",\"epoch_unix_s\":{},\"station\":{{\"lat_deg\":{},\"lon_deg\":{},\"alt_m\":{}}},\"doppler_hz\":{},\"phase_difference_rad\":{},\"differential_doppler_hz\":{},\"azimuth_rad\":{},\"elevation_rad\":{},\"baseline_enu_m\":[59.0,3.0,0.2],\"measurement_covariance\":{covariance:?}}}", site.3 + 1, site.3, 1704110400.25 + i as f64 * 8.0, site.0, site.1, site.2, z[0], z[1], z[2], z[3], z[4])
    }).collect::<Vec<_>>();
    assert_eq!(rows.len(), 42);
    let fixture = format!(
        "{{\"estimator\":\"batch_lm\",\"spacecraft_id\":\"fixture-spacecraft\",\"reference_epoch_unix_s\":1704110400.25,\"initial_state_gcrf\":{prior:?},\"observations\":[{}],\"carrier_hz\":2230000000.0,\"sigma_doppler_hz\":0.5,\"sigma_angle_rad\":0.0001,\"fixed_drag_m2_kg\":0.02,\"max_iterations\":400,\"gradient_tolerance\":0.0001,\"step_tolerance\":0.00000001,\"cost_tolerance\":0.0000000001}}\n",
        rows.join(",")
    );
    if let Ok(path) = std::env::var("DART_FIXTURE_OUTPUT") {
        std::fs::write(path, fixture).unwrap();
    }
}
