#![allow(clippy::disallowed_methods)]
use satkit::{Duration, ITRFCoord, Instant, frametransform};
use tychonic::{kernels::*, optim::ODProblem};

#[test]
fn mixed_station_geometry_matches_direct_frame_and_los_transforms() {
    let t = Instant::from_unixtime(1704110400.25);
    let sites = [
        GroundStation {
            lat_deg: 0.0,
            lon_deg: 0.0,
            alt_m: 0.0,
        },
        GroundStation {
            lat_deg: 40.0,
            lon_deg: -75.0,
            alt_m: 100.0,
        },
        GroundStation {
            lat_deg: -65.0,
            lon_deg: 170.0,
            alt_m: 2000.0,
        },
        GroundStation {
            lat_deg: 90.0,
            lon_deg: 30.0,
            alt_m: 0.0,
        },
        GroundStation {
            lat_deg: -90.0,
            lon_deg: -120.0,
            alt_m: 1000.0,
        },
    ];
    let times = [
        t,
        t,
        t + Duration::from_seconds(60.125),
        t,
        t - Duration::from_seconds(60.25),
    ];
    let (trajectory, enu) = precompute_station_geometry(&times, &sites).unwrap();
    assert_eq!(trajectory.ground_stations, sites);
    for (i, site) in sites.iter().enumerate() {
        let coord = ITRFCoord::from_geodetic_deg(site.lat_deg, site.lon_deg, site.alt_m);
        let (r, v) =
            frametransform::itrf_to_gcrf_state_approx(&coord.itrf, &V3::zeros(), &times[i]);
        assert!((trajectory.position[i] - r).norm() < 1e-8);
        assert!((trajectory.velocity[i] - v).norm() < 1e-10);
        let rotation = enu.enu_to_gcrf[i];
        assert!((rotation.transpose() * rotation - M3::eye()).norm_inf() < 1e-14);
        let local = v3(0.3, -0.5, 0.8);
        let direct = frametransform::qitrf2gcrf_approx(&times[i]) * (coord.q_enu2itrf() * local);
        assert!((rotation * local - direct).norm() < 1e-14);
        assert!((rotation.transpose() * direct - local).norm() < 1e-14);
        for (az, el) in [(0.0, 0.0), (1.2, 0.5), (-2.7, -0.8)] {
            let expected = convert_az_el_to_gcrf(az, el, &coord, &times[i]).unwrap();
            let los = convert_az_el_to_gcrf_with_rotation(az, el, &rotation).unwrap();
            assert!((los - expected).norm() < 1e-14);
            let ([az2, el2], h) = az_el_partial_with_rotation(
                &(trajectory.position[i] + 4e5 * los),
                &trajectory.position[i],
                &rotation,
            )
            .unwrap();
            assert!(wrap_pi(az2 - az).abs() < 1e-13);
            assert!((el2 - el).abs() < 1e-13);
            let (_, direct_h) = az_el_partial(
                &(trajectory.position[i] + 4e5 * los),
                &trajectory.position[i],
                &enu.east(i),
                &enu.north(i),
                &enu.up(i),
            )
            .unwrap();
            assert_eq!(h, direct_h);
        }
    }
}

#[test]
fn station_altitude_changes_position_along_up_without_changing_rotation() {
    let t = Instant::from_unixtime(1704110400.25);
    let sites = [
        GroundStation {
            lat_deg: 40.0,
            lon_deg: -75.0,
            alt_m: 100.0,
        },
        GroundStation {
            lat_deg: 40.0,
            lon_deg: -75.0,
            alt_m: 2100.0,
        },
    ];
    let (trajectory, enu) = precompute_station_geometry(&[t, t], &sites).unwrap();
    assert!((enu.enu_to_gcrf[0] - enu.enu_to_gcrf[1]).norm_inf() < 1e-14);
    assert!((trajectory.position[1] - trajectory.position[0] - 2000.0 * enu.up(0)).norm() < 1e-8);
    // Station velocity must change with its larger Earth-rotation radius.
    assert!((trajectory.velocity[1] - trajectory.velocity[0]).norm() > 0.1);
}

#[test]
fn malformed_geometry_and_measurement_records_are_rejected() {
    let t = Instant::from_unixtime(1704110400.25);
    let site = GroundStation {
        lat_deg: 40.0,
        lon_deg: -75.0,
        alt_m: 100.0,
    };
    assert!(precompute_station_geometry(&[t], &[]).is_err());
    assert!(precompute_station_geometry(&[Instant::INVALID], &[site]).is_err());
    assert!(PreparedGeometry::new(&[Instant::INVALID], &[site]).is_err());
    let coord = ITRFCoord::from_geodetic_deg(site.lat_deg, site.lon_deg, site.alt_m);
    assert!(convert_az_el_to_gcrf(0.0, 0.0, &coord, &Instant::INVALID).is_err());
    for invalid in [
        GroundStation {
            lat_deg: 91.0,
            ..site
        },
        GroundStation {
            lon_deg: f64::NAN,
            ..site
        },
        GroundStation {
            alt_m: f64::INFINITY,
            ..site
        },
    ] {
        assert!(precompute_station_geometry(&[t], &[invalid]).is_err());
    }
    let noise = MeasurementNoise {
        sigma_rr: 0.05,
        sigma_az: 1e-4,
        sigma_el: 1e-4,
    };
    let m = Measurement {
        time: t,
        station: site,
        range_rate: 0.0,
        azimuth: 1.0,
        elevation: 0.5,
    };
    assert!(ODProblem::from_measurements(t, &[], noise, od_prop_settings(), 1e-4).is_err());
    assert!(
        ODProblem::from_measurements(
            t,
            &[Measurement {
                range_rate: f64::NAN,
                ..m
            }],
            noise,
            od_prop_settings(),
            1e-4
        )
        .is_err()
    );
    assert!(
        ODProblem::from_measurements(
            t,
            &[m],
            MeasurementNoise {
                sigma_az: 0.0,
                ..noise
            },
            od_prop_settings(),
            1e-4
        )
        .is_err()
    );
    assert!(
        az_el_partial_with_rotation(&v3(1.0, 1.0, 1.0), &V3::zeros(), &(M3::eye() * 2.0)).is_err()
    );
    assert!(convert_az_el_to_gcrf_with_rotation(0.0, 2.0, &M3::eye()).is_err());
}
