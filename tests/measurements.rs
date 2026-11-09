#![allow(clippy::disallowed_methods)]
use std::f64::consts::PI;

use numeris::Quaternion;
use satkit::{Duration, Frame, ITRFCoord, Instant, TLE, frametransform};
use tychonic::kernels::*;

fn close(actual: f64, expected: f64, atol: f64, rtol: f64) {
    assert!(
        (actual - expected).abs() <= atol + rtol * expected.abs(),
        "got {actual:.12e}, expected {expected:.12e}"
    );
}

#[test]
fn range_rate_sign_and_velocity_conventions() {
    let r = v3(3.0, 4.0, 0.0);
    let sta = v3(100.0, -20.0, 30.0);
    let vsta = v3(-80.0, 40.0, 0.0);
    for speed in [-100.0, 0.0, 250.0] {
        let (rr, h) =
            range_rate_partial(&(sta + r), &(vsta + r * (speed / 5.0)), &sta, &vsta).unwrap();
        close(rr, speed, 1e-12, 0.0);
        for k in 0..3 {
            close(h[(0, k)], 0.0, 1e-12, 0.0);
            close(h[(0, 3 + k)], r[k] / 5.0, 1e-14, 0.0);
        }
    }
    let (rr, _) =
        range_rate_partial(&(sta + r), &(vsta + v3(-4.0, 3.0, 0.0)), &sta, &vsta).unwrap();
    close(rr, 0.0, 1e-13, 0.0);
}

#[test]
fn measurement_jacobians_match_central_differences_across_geometry() {
    let rotations = [
        Quaternion::identity(),
        Quaternion::rotx(0.7) * Quaternion::rotz(-1.1),
    ];
    let angles: [(f64, f64); 7] = [
        (0.0, 0.0),
        (PI / 2.0, 0.4),
        (-PI / 2.0, -0.3),
        (PI - 1e-8, 0.8),
        (-PI + 1e-8, -0.8),
        (2.3, 1.5),
        (-1.2, -1.5),
    ];
    let station = v3(-4.1e6, 3.2e6, 2.5e6);
    let vstation = v3(-250.0, -320.0, 1.0);
    for q in rotations {
        let east = q * v3(1.0, 0.0, 0.0);
        let north = q * v3(0.0, 1.0, 0.0);
        let up = q * v3(0.0, 0.0, 1.0);
        for (az, el) in angles {
            for range in [1e3, 4e5, 4e7] {
                let offset =
                    (east * (az.sin() * el.cos()) + north * (az.cos() * el.cos()) + up * el.sin())
                        * range;
                let x = v6_from(&(station + offset), &v3(5400.0, -3600.0, 2400.0));
                let ([az_pred, el_pred], ha) =
                    az_el_partial(&r_of(&x), &station, &east, &north, &up).unwrap();
                close(wrap_pi(az_pred - az), 0.0, 1e-11, 0.0);
                close(el_pred, el, 1e-11, 0.0);
                let (_, hr) =
                    range_rate_partial(&r_of(&x), &v_of(&x), &station, &vstation).unwrap();
                for k in 0..6 {
                    let step = if k < 3 { 0.01 } else { 0.001 };
                    let mut plus = x;
                    let mut minus = x;
                    plus[k] += step;
                    minus[k] -= step;
                    let (rp, _) =
                        range_rate_partial(&r_of(&plus), &v_of(&plus), &station, &vstation)
                            .unwrap();
                    let (rm, _) =
                        range_rate_partial(&r_of(&minus), &v_of(&minus), &station, &vstation)
                            .unwrap();
                    close(hr[(0, k)], (rp - rm) / (2.0 * step), 2e-7, 4e-7);
                    let (ap, _) =
                        az_el_partial(&r_of(&plus), &station, &east, &north, &up).unwrap();
                    let (am, _) =
                        az_el_partial(&r_of(&minus), &station, &east, &north, &up).unwrap();
                    close(
                        ha[(0, k)],
                        wrap_pi(ap[0] - am[0]) / (2.0 * step),
                        2e-12,
                        4e-7,
                    );
                    close(ha[(1, k)], (ap[1] - am[1]) / (2.0 * step), 2e-12, 4e-7);
                }
            }
        }
    }
}

#[test]
fn angles_are_stable_near_zenith_and_nadir() {
    let e = v3(1.0, 0.0, 0.0);
    let n = v3(0.0, 1.0, 0.0);
    let u = v3(0.0, 0.0, 1.0);
    for sign in [-1.0, 1.0] {
        let r = v3(0.001, 0.002, sign * 1e6);
        let ([az, el], h) = az_el_partial(&r, &V3::zeros(), &e, &n, &u).unwrap();
        assert!(az.is_finite() && el.is_finite());
        for k in 0..6 {
            assert!(h[(0, k)].is_finite() && h[(1, k)].is_finite());
        }
        close(az, 0.5_f64.atan(), 1e-15, 0.0);
        close(h[(1, 0)], -sign / (5.0_f64.sqrt() * 1e6), 1e-20, 1e-12);
    }
}

#[test]
fn singular_and_invalid_inputs_return_errors() {
    let z = V3::zeros();
    let e = v3(1.0, 0.0, 0.0);
    let n = v3(0.0, 1.0, 0.0);
    let u = v3(0.0, 0.0, 1.0);
    assert!(matches!(
        range_rate_partial(&z, &z, &z, &z),
        Err(KernelError::ZeroRange)
    ));
    assert!(matches!(
        az_el_partial(&z, &z, &e, &n, &u),
        Err(KernelError::ZeroRange)
    ));
    for sign in [-1.0, 1.0] {
        assert!(matches!(
            az_el_partial(&(u * sign), &z, &e, &n, &u),
            Err(KernelError::UndefinedAzimuth)
        ));
    }
    assert!(az_el_partial(&v3(1.0, 1.0, 1.0), &z, &(e * 2.0), &n, &u).is_err());
    assert!(az_el_partial(&v3(1.0, 1.0, 1.0), &z, &e, &n, &(-u)).is_err());
    assert!(range_rate_partial(&e, &v3(f64::NAN, 0.0, 0.0), &z, &z).is_err());
    assert!(range_rate_partial(&v3(1e300, 0.0, 0.0), &e, &z, &z).is_err());
    assert!(az_el_partial(&v3(f64::INFINITY, 0.0, 0.0), &z, &e, &n, &u).is_err());
}

#[test]
fn angle_wrapping_handles_multiple_turns_and_nonfinite_inputs() {
    for turns in -100..=100 {
        close(wrap_pi(0.3 + 2.0 * PI * turns as f64), 0.3, 2e-13, 0.0);
    }
    close(wrap_pi(PI), -PI, 0.0, 0.0);
    close(wrap_pi(-PI), -PI, 0.0, 0.0);
    close(wrap_pi(-2e-8), -2e-8, 1e-15, 0.0);
    assert_eq!(wrap_pi(-1e-20), -1e-20);
    assert_eq!(wrap_pi(1e-20), 1e-20);
    assert!(wrap_pi(f64::NAN).is_nan());
    assert!(wrap_pi(f64::INFINITY).is_nan());
    assert!(wrap_pi(-f64::INFINITY).is_nan());
    assert!(wrap_pi(1e300).is_finite());
}

#[test]
fn station_enu_los_roundtrip_and_subsecond_motion() {
    let t = Instant::from_datetime(2024, 1, 1, 12, 0, 0.25).unwrap();
    let times = [
        t - Duration::from_seconds(0.125),
        t,
        t + Duration::from_seconds(0.125),
    ];
    for (lat, lon, alt) in [
        (0.0, 0.0, 0.0),
        (40.0, -75.0, 100.0),
        (-65.0, 170.0, 2000.0),
        (90.0, 30.0, 0.0),
        (-90.0, -120.0, 0.0),
    ] {
        let coord = ITRFCoord::from_geodetic_deg(lat, lon, alt);
        let station = compute_station_trajectory(&times, lat, lon, alt).unwrap();
        let enu = compute_enu_basis(&times, lat, lon, alt).unwrap();
        assert_eq!(station.timestamps, times);
        assert_eq!(enu.timestamps, times);
        let numerical_velocity = (station.position[2] - station.position[0]) / 0.25;
        assert!((numerical_velocity - station.velocity[1]).norm() < 0.003);
        if lat.abs() < 80.0 {
            assert!((station.position[1] - station.position[0]).norm() > 10.0);
        }
        close(enu.east(1).dot(&enu.north(1)), 0.0, 1e-14, 0.0);
        assert!((enu.east(1).cross(&enu.north(1)) - enu.up(1)).norm() < 1e-14);
        for (az, el) in [(0.0, 0.0), (PI / 2.0, 0.5), (-2.7, -0.8), (PI - 1e-7, 1.2)] {
            let los = convert_az_el_to_gcrf(az, el, &coord, &t).unwrap();
            close(los.norm(), 1.0, 1e-14, 0.0);
            let ([az2, el2], _) = az_el_partial(
                &(station.position[1] + los * 4e5),
                &station.position[1],
                &enu.east(1),
                &enu.north(1),
                &enu.up(1),
            )
            .unwrap();
            close(wrap_pi(az2 - az), 0.0, 1e-14, 0.0);
            close(el2, el, 1e-14, 0.0);
        }
    }
    // Unix epochs prior to 1970 and fractional seconds remain representable.
    let old = [Instant::from_unixtime(-0.25)];
    assert_eq!(
        compute_station_trajectory(&old, 0.0, 0.0, 0.0)
            .unwrap()
            .timestamps,
        old
    );
    assert!(compute_station_trajectory(&times, 91.0, 0.0, 0.0).is_err());
    assert!(compute_enu_basis(&times, 0.0, f64::NAN, 0.0).is_err());
}

#[test]
fn tle_conversion_matches_vallado_reference_in_metres() {
    let tle = TLE::load_2line(
        "1 00005U 58002B   00179.78495062  .00000023  00000-0  28098-4 0  4753",
        "2 00005  34.2682 348.7242 1859667 331.7664  19.3264 10.82419157413667",
    )
    .unwrap();
    let gcrf = initial_state_from_tle(&tle, &tle.epoch).unwrap();
    let (r, v) = frametransform::transform_state(
        Frame::GCRF,
        Frame::TEME,
        &tle.epoch,
        &r_of(&gcrf),
        &v_of(&gcrf),
    )
    .unwrap();
    let reference_r = v3(7022.46529266e3, -1400.08296755e3, 0.03995155e3);
    let reference_v = v3(1.893841015e3, 6.405893759e3, 4.534807250e3);
    assert!((r - reference_r).norm() < 0.02);
    assert!((v - reference_v).norm() < 2e-5);
}
