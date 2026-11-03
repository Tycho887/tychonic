use satkit::{
    Duration, Instant,
    orbitprop::{Integrator, SatPropertiesSimple, TideModel},
};
use tychonic::kernels::*;

fn epoch() -> Instant {
    Instant::from_datetime(2024, 1, 1, 12, 0, 0.25).unwrap()
}

fn leo() -> V6 {
    v6_from(&v3(6678e3, 0.0, 0.0), &v3(0.0, 7100.0, 3000.0))
}

fn close(actual: f64, expected: f64, atol: f64, rtol: f64, context: &str) {
    assert!(
        (actual - expected).abs() <= atol + rtol * expected.abs(),
        "{context}: got {actual:.12e}, expected {expected:.12e}, error {:.3e}",
        (actual - expected).abs()
    );
}

#[test]
fn stm_matches_state_differences_for_leo_eccentric_and_geo_in_both_directions() {
    let settings = od_prop_settings();
    let cases = [
        (leo(), 0.0),
        (leo(), 0.02),
        (
            v6_from(&v3(7000e3, 0.0, 0.0), &v3(0.0, 9500.0, 3000.0)),
            0.0,
        ),
        (
            v6_from(&v3(42164e3, 0.0, 0.0), &v3(0.0, 3074.66, 0.0)),
            0.02,
        ),
    ];
    let t = epoch();
    for (case, (x, cd)) in cases.into_iter().enumerate() {
        let props = SatPropertiesSimple::new(cd, 0.0);
        for dt in [-600.0, -60.0, 60.0, 600.0] {
            let end = t + Duration::from_seconds(dt);
            let (state, phi) = propagate_with_stm(&x, &t, &end, &settings, cd).unwrap();
            let simple = propagate_state(&x, &t, &end, &settings, &props).unwrap();
            let pos_difference = (r_of(&state) - r_of(&simple)).norm();
            let vel_difference = (v_of(&state) - v_of(&simple)).norm();
            assert!(
                pos_difference < 0.0001,
                "case {case}, dt {dt}: state/STM position difference {pos_difference} m"
            );
            assert!(
                vel_difference < 1e-6,
                "case {case}, dt {dt}: state/STM velocity difference {vel_difference} m/s"
            );
            for k in 0..6 {
                let step = if k < 3 { 10.0 } else { 0.01 };
                let mut plus = x;
                let mut minus = x;
                plus[k] += step;
                minus[k] -= step;
                let p = propagate_state(&plus, &t, &end, &settings, &props).unwrap();
                let m = propagate_state(&minus, &t, &end, &settings, &props).unwrap();
                let fd = (p - m) / (2.0 * step);
                for row in 0..6 {
                    let context = format!("case {case}, dt {dt}, Φ[{row},{k}]");
                    // Satkit's drag density gradient is altitude-only, using
                    // a 100 m difference; it is an approximate variational model.
                    close(
                        phi[(row, k)],
                        fd[row],
                        if row < 3 { 3e-5 } else { 2e-7 },
                        3e-5,
                        &context,
                    );
                }
            }
        }
    }
}

#[test]
fn one_hour_stm_linearization_predicts_small_state_perturbations() {
    let t = epoch();
    let settings = od_prop_settings();
    let cases = [
        (leo(), 0.02),
        (
            v6_from(&v3(7000e3, 0.0, 0.0), &v3(0.0, 9500.0, 3000.0)),
            0.0,
        ),
        (
            v6_from(&v3(42164e3, 0.0, 0.0), &v3(0.0, 3074.66, 0.0)),
            0.02,
        ),
    ];
    for (case, (x, cd)) in cases.into_iter().enumerate() {
        for dt in [-3600.0, 3600.0] {
            let end = t + Duration::from_seconds(dt);
            let props = SatPropertiesSimple::new(cd, 0.0);
            let (state, stm) = propagate_with_stm(&x, &t, &end, &settings, cd).unwrap();
            let simple = propagate_state(&x, &t, &end, &settings, &props).unwrap();
            assert!((r_of(&state) - r_of(&simple)).norm() < 0.0001);
            assert!((v_of(&state) - v_of(&simple)).norm() < 1e-6);
            for k in 0..6 {
                let step = if k < 3 { 10.0 } else { 0.01 };
                let mut plus = x;
                let mut minus = x;
                plus[k] += step;
                minus[k] -= step;
                let p = propagate_state(&plus, &t, &end, &settings, &props).unwrap();
                let m = propagate_state(&minus, &t, &end, &settings, &props).unwrap();
                let actual_response = (p - m) * 0.5;
                let predicted_response = stm.block::<6, 1>(0, k) * step;
                // Compare in physical units: individual STM components can
                // nearly cancel at an orbital return, making relative-only
                // derivative bounds inappropriate for the approximate drag STM.
                let error = predicted_response - actual_response;
                let pos_error = r_of(&error).norm();
                let vel_error = v_of(&error).norm();
                assert!(
                    pos_error < 0.002,
                    "case {case}, dt {dt}, column {k}: linearized position error {pos_error} m"
                );
                assert!(
                    vel_error < 5e-6,
                    "case {case}, dt {dt}, column {k}: linearized velocity error {vel_error} m/s"
                );
            }
        }
    }
}

#[test]
fn zero_duration_is_exact_identity_even_for_nonzero_drag() {
    let x = leo();
    let t = epoch();
    let settings = od_prop_settings();
    let (state, phi) = propagate_with_stm(&x, &t, &t, &settings, 0.02).unwrap();
    assert_eq!(state, x);
    assert_eq!(phi, M6::eye());
    let sens = compute_sensitivities(&x, &t, &[t, t], &settings, 0.02, 1e-4).unwrap();
    for i in 0..2 {
        assert_eq!(sens.states[i], x);
        assert_eq!(sens.stm[i], M6::eye());
        assert_eq!(sens.state_dcd[i], V6::zeros());
        assert_eq!(sens.augmented_transition(i).unwrap(), M7::eye());
    }
    assert!(sens.augmented_transition(2).is_err());
}

#[test]
fn dense_arcs_preserve_unordered_and_duplicate_epochs() {
    let x = leo();
    let t = epoch();
    let settings = od_prop_settings();
    let times = [
        t + Duration::from_seconds(121.25),
        t - Duration::from_seconds(600.5),
        t,
        t - Duration::from_seconds(60.125),
        t + Duration::from_seconds(600.5),
        t - Duration::from_seconds(60.125),
    ];
    let sens = compute_sensitivities(&x, &t, &times, &settings, 0.02, 1e-4).unwrap();
    for (i, end) in times.iter().enumerate() {
        let (state, phi) = propagate_with_stm(&x, &t, end, &settings, 0.02).unwrap();
        assert!((r_of(&sens.states[i]) - r_of(&state)).norm() < 0.0001);
        assert!((v_of(&sens.states[i]) - v_of(&state)).norm() < 1e-6);
        for row in 0..6 {
            for k in 0..6 {
                close(
                    sens.stm[i][(row, k)],
                    phi[(row, k)],
                    1e-7,
                    1e-8,
                    "dense STM",
                );
            }
        }
    }
    assert_eq!(sens.states[3], sens.states[5]);
    assert_eq!(sens.stm[3], sens.stm[5]);
    assert_eq!(sens.state_dcd[3], sens.state_dcd[5]);
}

#[test]
fn drag_derivatives_match_independent_endpoint_differences_and_smaller_steps() {
    let x = leo();
    let t = epoch();
    let settings = od_prop_settings();
    let times = [
        t - Duration::from_seconds(600.0),
        t + Duration::from_seconds(600.0),
    ];
    for cd in [0.0, 5e-5, 0.02] {
        let sens = compute_sensitivities(&x, &t, &times, &settings, cd, 1e-4).unwrap();
        let smaller = compute_sensitivities(&x, &t, &times, &settings, cd, 5e-5).unwrap();
        for (i, end) in times.iter().enumerate() {
            // Different step and endpoint propagation, independent of dense output.
            let step = 2e-4;
            let plus = propagate_state(
                &x,
                &t,
                end,
                &settings,
                &SatPropertiesSimple::new(cd + step, 0.0),
            )
            .unwrap();
            let fd = if cd >= step {
                let minus = propagate_state(
                    &x,
                    &t,
                    end,
                    &settings,
                    &SatPropertiesSimple::new(cd - step, 0.0),
                )
                .unwrap();
                (plus - minus) / (2.0 * step)
            } else {
                let base =
                    propagate_state(&x, &t, end, &settings, &SatPropertiesSimple::new(cd, 0.0))
                        .unwrap();
                let twice = propagate_state(
                    &x,
                    &t,
                    end,
                    &settings,
                    &SatPropertiesSimple::new(cd + 2.0 * step, 0.0),
                )
                .unwrap();
                ((plus - base) * 4.0 - (twice - base)) / (2.0 * step)
            };
            assert!(
                r_of(&sens.state_dcd[i]).norm() > 1.0,
                "drag column must be observable at this altitude"
            );
            for row in 0..6 {
                let atol = if row < 3 { 0.005 } else { 1e-5 };
                close(
                    sens.state_dcd[i][row],
                    fd[row],
                    atol,
                    2e-4,
                    "endpoint drag difference",
                );
                close(
                    sens.state_dcd[i][row],
                    smaller.state_dcd[i][row],
                    atol,
                    2e-4,
                    "drag step stability",
                );
            }
        }
    }
}

#[test]
fn geo_has_zero_drag_sensitivity() {
    let x = v6_from(&v3(42164e3, 0.0, 0.0), &v3(0.0, 3074.66, 0.0));
    let t = epoch();
    let sens = compute_sensitivities(
        &x,
        &t,
        &[t + Duration::from_seconds(3600.0)],
        &od_prop_settings(),
        0.02,
        1e-4,
    )
    .unwrap();
    assert_eq!(sens.state_dcd[0], V6::zeros());
}

#[test]
fn forward_backward_roundtrip_and_augmented_transition_composition() {
    let x = leo();
    let t = epoch();
    let mid = t + Duration::from_seconds(300.0);
    let end = t + Duration::from_seconds(600.0);
    let settings = od_prop_settings();
    let first = compute_sensitivities(&x, &t, &[mid, end], &settings, 0.02, 1e-4).unwrap();
    let second =
        compute_sensitivities(&first.states[0], &mid, &[end], &settings, 0.02, 1e-4).unwrap();
    let composed = second.augmented_transition(0).unwrap() * first.augmented_transition(0).unwrap();
    let direct = first.augmented_transition(1).unwrap();
    for row in 0..7 {
        for k in 0..7 {
            close(
                composed[(row, k)],
                direct[(row, k)],
                if k == 6 { 0.003 } else { 2e-6 },
                3e-5,
                "augmented composition",
            );
        }
    }
    let (returned, inverse) =
        propagate_with_stm(&first.states[1], &end, &t, &settings, 0.02).unwrap();
    assert!((r_of(&returned) - r_of(&x)).norm() < 0.0001);
    assert!((v_of(&returned) - v_of(&x)).norm() < 1e-6);
    let identity = inverse * first.stm[1];
    for row in 0..6 {
        for k in 0..6 {
            close(
                identity[(row, k)],
                if row == k { 1.0 } else { 0.0 },
                2e-6,
                0.0,
                "inverse STM",
            );
        }
    }
}

#[test]
fn bad_parameters_and_unsupported_force_models_fail_cleanly() {
    let x = leo();
    let t = epoch();
    let settings = od_prop_settings();
    for (cd, step) in [
        (-0.01, 1e-4),
        (f64::NAN, 1e-4),
        (0.02, 0.0),
        (0.02, -1e-4),
        (0.02, f64::INFINITY),
        (1e100, 1e-10),
    ] {
        assert!(compute_sensitivities(&x, &t, &[t], &settings, cd, step).is_err());
    }
    for integrator in [Integrator::RODAS4, Integrator::GaussJackson8] {
        let mut invalid = settings.clone();
        invalid.integrator = integrator;
        assert!(propagate_with_stm(&x, &t, &t, &invalid, 0.02).is_err());
    }
    let mut invalid = settings.clone();
    invalid.use_relativistic_correction = true;
    assert!(propagate_with_stm(&x, &t, &t, &invalid, 0.02).is_err());
    invalid.use_relativistic_correction = false;
    invalid.tide_model = TideModel::SolidStep1;
    assert!(propagate_with_stm(&x, &t, &t, &invalid, 0.02).is_err());
    assert!(compute_sensitivities(&V6::zeros(), &t, &[t], &settings, 0.02, 1e-4).is_err());
    let empty = compute_sensitivities(&x, &t, &[], &settings, 0.02, 1e-4).unwrap();
    assert!(empty.states.is_empty());
}
