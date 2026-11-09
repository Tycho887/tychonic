#![allow(clippy::disallowed_methods)]
#[path = "../benches/support/mod.rs"]
#[allow(dead_code)]
mod support;

use std::f64::consts::{FRAC_PI_2, PI};
use support::*;
use tychonic::kernels::*;

fn close(actual: f64, expected: f64, atol: f64, rtol: f64) {
    assert!(actual.is_finite() && expected.is_finite());
    assert!(
        (actual - expected).abs() <= atol + rtol * expected.abs(),
        "actual={actual:e}, expected={expected:e}, atol={atol:e}, rtol={rtol:e}"
    );
}

#[test]
fn seeded_measurement_and_chained_jacobians_match_central_differences() {
    let f = Fixture::new(512).unwrap();
    let chained = compute_full_jacobian(&f.sensitivities, &f.station, &f.enu).unwrap();
    for i in 0..512 {
        let state = f.sensitivities.states()[i];
        let range = (r_of(&state) - f.station.position[i]).norm();
        let predict = |x: &V6| {
            let (rr, _) = range_rate_partial(
                &r_of(x),
                &v_of(x),
                &f.station.position[i],
                &f.station.velocity[i],
            )
            .unwrap();
            let ([az, el], _) = az_el_partial_with_rotation(
                &r_of(x),
                &f.station.position[i],
                &f.enu.enu_to_gcrf[i],
            )
            .unwrap();
            [rr, az, el]
        };
        let (_, rr) = range_rate_partial(
            &r_of(&state),
            &v_of(&state),
            &f.station.position[i],
            &f.station.velocity[i],
        )
        .unwrap();
        let (_, angles) = az_el_partial_with_rotation(
            &r_of(&state),
            &f.station.position[i],
            &f.enu.enu_to_gcrf[i],
        )
        .unwrap();
        for k in 0..6 {
            let step = if k < 3 {
                (range * 1e-8).max(0.01)
            } else {
                0.001
            };
            let mut p = state;
            let mut m = state;
            p[k] += step;
            m[k] -= step;
            let p = predict(&p);
            let m = predict(&m);
            close(rr[(0, k)], (p[0] - m[0]) / (2.0 * step), 5e-7, 3e-6);
            close(
                angles[(0, k)],
                wrap_pi(p[1] - m[1]) / (2.0 * step),
                3e-12,
                3e-6,
            );
            close(angles[(1, k)], (p[2] - m[2]) / (2.0 * step), 3e-12, 3e-6);
        }
        for k in 0..7 {
            let column = if k < 6 {
                f.sensitivities.stm()[i].block::<6, 1>(0, k)
            } else {
                f.sensitivities.state_dcd()[i]
            };
            let step = if k < 3 { 0.1 } else { 0.001 };
            let p = predict(&(state + column * step));
            let m = predict(&(state - column * step));
            close(
                chained.range_rate[(i, k)],
                (p[0] - m[0]) / (2.0 * step),
                1e-6,
                1e-5,
            );
            close(
                chained.azimuth[(i, k)],
                wrap_pi(p[1] - m[1]) / (2.0 * step),
                1e-11,
                1e-5,
            );
            close(
                chained.elevation[(i, k)],
                (p[2] - m[2]) / (2.0 * step),
                1e-11,
                1e-5,
            );
        }
    }
}

#[test]
fn seeded_rotations_and_los_roundtrip_across_ranges_and_branch_crossings() {
    let f = Fixture::new(256).unwrap();
    for i in 0..256 {
        let rotation = f.enu.enu_to_gcrf[i];
        let q = f.quaternions[i];
        assert!((rotation * f.local[i] - q * f.local[i]).norm() < 1e-14);
        assert!((rotation.transpose() * (rotation * f.local[i]) - f.local[i]).norm() < 1e-14);
        for [az, el] in [
            f.angles[i],
            [PI - 1e-9, 0.7],
            [-PI + 1e-9, -0.7],
            [0.0, FRAC_PI_2 - 1e-6],
            [1.0, -FRAC_PI_2 + 1e-6],
        ] {
            let direct =
                convert_az_el_to_gcrf(az, el, &f.coords[i], &f.station.timestamps[i]).unwrap();
            let stored = convert_az_el_to_gcrf_with_rotation(az, el, &rotation).unwrap();
            assert!((direct - stored).norm() < 1e-14);
            // Use local zero-origin geometry to avoid cancellation of tiny horizontal
            // components against Earth-scale station coordinates near the poles.
            for range in [1e3, 4e5, 4e7] {
                let ([a, e], h) =
                    az_el_partial_with_rotation(&(stored * range), &V3::zeros(), &rotation)
                        .unwrap();
                let angular_tol = if el.abs() > 1.5 { 2e-9 } else { 1e-13 };
                close(wrap_pi(a - az), 0.0, angular_tol, 0.0);
                close(e, el, 1e-13, 0.0);
                assert!(h.as_slice().iter().all(|v| v.is_finite()));
            }
        }
    }
}

#[test]
fn azimuth_threshold_and_error_paths_are_separate_from_valid_fixtures() {
    for sign in [-1.0, 1.0] {
        for factor in [0.0, 0.999, 1.0] {
            assert!(matches!(
                az_el_partial_with_rotation(
                    &v3(factor * 1e-12, 0.0, sign),
                    &V3::zeros(),
                    &M3::eye()
                ),
                Err(KernelError::UndefinedAzimuth)
            ));
        }
        for factor in [1.001, 2.0, 1e3] {
            let (_, h) = az_el_partial_with_rotation(
                &v3(factor * 1e-12, 0.0, sign),
                &V3::zeros(),
                &M3::eye(),
            )
            .unwrap();
            assert!(h.as_slice().iter().all(|v| v.is_finite()));
        }
    }
    assert!(matches!(
        range_rate_partial(&V3::zeros(), &V3::zeros(), &V3::zeros(), &V3::zeros()),
        Err(KernelError::ZeroRange)
    ));
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(az_el_partial_with_rotation(&v3(bad, 1.0, 1.0), &V3::zeros(), &M3::eye()).is_err());
        assert!(convert_az_el_to_gcrf_with_rotation(bad, 0.5, &M3::eye()).is_err());
    }
    let mut invalid = M3::eye();
    invalid[(0, 0)] = 2.0;
    assert!(az_el_partial_with_rotation(&v3(1.0, 2.0, 3.0), &V3::zeros(), &invalid).is_err());
}

#[test]
fn large_chained_batches_have_finite_outputs_and_preserve_selected_rows() {
    let f = Fixture::new(65536).unwrap();
    let jac = compute_full_jacobian(&f.sensitivities, &f.station, &f.enu).unwrap();
    let prepared = f
        .batch
        .evaluate_from_sensitivities(&f.sensitivities)
        .unwrap();
    assert_eq!(
        (prepared.jacobian.nrows(), prepared.jacobian.ncols()),
        (3 * 65536, 7)
    );
    assert!(
        prepared
            .residual
            .as_slice()
            .iter()
            .chain(prepared.jacobian.as_slice())
            .all(|v| v.is_finite())
    );
    for matrix in [&jac.range_rate, &jac.azimuth, &jac.elevation] {
        assert_eq!((matrix.nrows(), matrix.ncols()), (65536, 7));
        assert!(matrix.as_slice().iter().all(|v| v.is_finite()));
    }
    for i in [0, 15, 255, 1023, 16383, 65535] {
        let x = &f.sensitivities.states()[i];
        let (_, h) = range_rate_partial(
            &r_of(x),
            &v_of(x),
            &f.station.position[i],
            &f.station.velocity[i],
        )
        .unwrap();
        let expected = h * f.sensitivities.stm()[i];
        for k in 0..6 {
            assert_eq!(jac.range_rate[(i, k)], expected[(0, k)]);
        }
        for (observable, matrix, sigma) in [
            (0, &jac.range_rate, 0.05),
            (1, &jac.azimuth, 1e-4),
            (2, &jac.elevation, 2e-4),
        ] {
            for k in 0..7 {
                let expected = matrix[(i, k)] / sigma;
                let actual = prepared.jacobian[(observable * 65536 + i, k)];
                assert!((actual - expected).abs() <= 1e-12 * (1.0 + expected.abs()));
            }
        }
    }
}
