#![allow(clippy::disallowed_methods)]
use numeris::{Matrix, Vector};
use satkit::{
    Duration, Frame, ITRFCoord, Instant, TLE, frametransform,
    sgp4::{SGP4Source, sgp4},
};
use tychonic::{estimate::*, kernels::*};

fn prior() -> TLE {
    TLE::load_2line(
        "1 25544U 98067A   24356.58519896  .00014389  00000-0  25222-3 0  9992",
        "2 25544  51.6403 106.8969 0007877   6.1421 113.2479 15.50801739487617",
    )
    .unwrap()
}

fn corrected(prior: &TLE, seconds: f64) -> TLE {
    let mut truth = prior.clone();
    truth.epoch += Duration::from_seconds(seconds);
    *truth.satrec_mut() = None;
    truth
}

fn truth_state(tle: &TLE, time: Instant) -> (V3, V3) {
    let mut tle = tle.clone();
    let propagated = sgp4(&mut tle, &[time]).unwrap();
    assert_eq!(propagated.errcode[0], satkit::sgp4::SGP4Error::SGP4Success);
    let r = V3::from_slice(propagated.pos.col_slice(0));
    let v = V3::from_slice(propagated.vel.col_slice(0));
    frametransform::transform_state(Frame::TEME, Frame::GCRF, &time, &r, &v).unwrap()
}

fn station_near(tle: &TLE, time: Instant, latitude_offset: f64) -> GroundStation {
    let (r, _) = truth_state(tle, time);
    let local = frametransform::qitrf2gcrf_approx(&time).conjugate() * r;
    let (lat, lon, _) = ITRFCoord::from_vector(&local).to_geodetic_rad();
    GroundStation {
        lat_deg: lat.to_degrees() + latitude_offset,
        lon_deg: lon.to_degrees(),
        alt_m: 0.0,
    }
}

// The truth path explicitly shifts a fresh TLE and evaluates the observables
// directly, independently of the UKF's measurement and epoch-shift helpers.
fn measurement(tle: &TLE, time: Instant, station: GroundStation, bias: f64) -> Measurement {
    let geometry = PreparedGeometry::new(&[time], &[station]).unwrap();
    let g = geometry.row(0).unwrap();
    let (r, v) = truth_state(tle, time);
    let los = r - g.position();
    let enu = g.rotation().to_enu(&los);
    Measurement {
        time,
        station,
        range_rate: los.dot(&(v - g.velocity())) / los.norm() + bias,
        azimuth: enu[0].atan2(enu[1]),
        elevation: enu[2].atan2(enu[0].hypot(enu[1])),
    }
}

fn filter(prior: TLE, time: Instant) -> TcaUkf {
    TcaUkf::new(
        prior,
        time,
        Vector::zeros(),
        Matrix::new([[150.0_f64.powi(2), 0.0], [0.0, 200.0_f64.powi(2)]]),
    )
    .unwrap()
}

fn snapshot(filter: &TcaUkf) -> (TcaState, TcaCovariance, Instant) {
    (*filter.state(), *filter.covariance(), filter.epoch())
}

fn assert_covariance(p: &TcaCovariance) {
    assert!(p.as_slice().iter().all(|v| v.is_finite()));
    assert_eq!(p[(0, 1)], p[(1, 0)]);
    assert!(p.cholesky().is_ok(), "{p:?}");
}

#[test]
fn recovers_signed_epoch_errors_and_constant_pass_biases() {
    for (offset, bias) in [
        (-120.0, -45.0),
        (-30.0, 25.0),
        (-5.0, -8.0),
        (5.0, 8.0),
        (30.0, -25.0),
        (120.0, 45.0),
    ] {
        let prior = prior();
        let truth = corrected(&prior, offset);
        let center = prior.epoch + Duration::from_seconds(600.0);
        let station = station_near(&truth, center, 2.5);
        let start = center - Duration::from_seconds(240.0);
        let mut filter = filter(prior.clone(), start);
        let noise = tca_measurement_noise(0.5).unwrap();
        for i in 0..=48 {
            let time = start + Duration::from_seconds(i as f64 * 10.0);
            let reading = measurement(&truth, time, station, bias);
            assert!(reading.elevation > 0.0, "synthetic pass should be visible");
            let result = filter
                .observe(&reading, &noise, None, UpdateOptions::default())
                .unwrap();
            assert!(result.accepted && result.nis.is_finite());
            assert_covariance(filter.covariance());
        }
        assert!(
            (filter.state()[0] - offset).abs() < 0.1,
            "offset {offset}, recovered {:?}",
            filter.state()
        );
        assert!(
            (filter.state()[1] - bias).abs() < 0.5,
            "bias {bias}, recovered {:?}",
            filter.state()
        );
        assert!(filter.covariance()[(0, 0)] < 1.0);
        assert_eq!(filter.prior_tle(), &prior);
        let mut final_tle = filter.corrected_tle().unwrap();
        assert!((final_tle.epoch - truth.epoch).as_seconds().abs() < 0.1);
        final_tle.epoch = prior.epoch;
        assert_eq!(final_tle, prior, "all other TLE fields must be preserved");
    }
}

#[test]
fn epoch_correction_uses_actual_station_and_frame_epoch() {
    let prior = prior();
    let time = prior.epoch + Duration::from_seconds(600.125);
    for offset in [-120.0, 5.0, 120.0] {
        let truth = corrected(&prior, offset);
        let station = station_near(&truth, time, 2.5);
        let reading = measurement(&truth, time, station, 17.0);
        let mut filter = TcaUkf::new(
            prior.clone(),
            time,
            Vector::from_array([offset, 17.0]),
            Matrix::new([[1e-8, 0.0], [0.0, 1e-8]]),
        )
        .unwrap();
        let result = filter
            .observe(
                &reading,
                &tca_measurement_noise(0.5).unwrap(),
                None,
                UpdateOptions::default(),
            )
            .unwrap();
        assert!(result.innovation[0].abs() < 1e-6, "{:?}", result.innovation);
        assert!(result.innovation[1].abs() < 1e-10);
        assert!(result.innovation[2].abs() < 1e-10);
        assert!(result.nis < 1e-10);
    }
}

#[test]
fn corrected_tle_clears_preexisting_sgp4_cache_and_preserves_fields() {
    let fresh = prior();
    let time = fresh.epoch + Duration::from_seconds(600.0);
    let mut cached = fresh.clone();
    sgp4(&mut cached, &[time]).unwrap();
    assert!(cached.satrec_mut().is_some());
    let initial = Vector::from_array([120.123456, 17.0]);
    let p = Matrix::new([[900.0, 3.0], [3.0, 100.0]]);
    let mut a = TcaUkf::new(fresh.clone(), time, initial, p).unwrap();
    let mut b = TcaUkf::new(cached.clone(), time, initial, p).unwrap();
    let truth = corrected(&fresh, initial[0]);
    let mut output = b.corrected_tle().unwrap();
    assert_eq!(output, truth);
    assert!(output.satrec_mut().is_none());
    assert_eq!(truth_state(&output, time), truth_state(&truth, time));
    let station = station_near(&truth, time, 2.5);
    let reading = measurement(&truth, time, station, initial[1]);
    let noise = tca_measurement_noise(0.5).unwrap();
    let ra = a
        .observe(&reading, &noise, None, UpdateOptions::default())
        .unwrap();
    let rb = b
        .observe(&reading, &noise, None, UpdateOptions::default())
        .unwrap();
    assert_eq!(snapshot(&a), snapshot(&b));
    assert_eq!(ra.innovation, rb.innovation);
    assert_eq!(ra.nis, rb.nis);
    assert!(
        cached.satrec_mut().is_some(),
        "caller-owned prior is untouched"
    );
}

#[test]
fn constant_prediction_adds_q_once_and_validates_time_and_covariance() {
    let prior = prior();
    let time = prior.epoch;
    let mut filter = filter(prior, time);
    let original = snapshot(&filter);
    let q = Matrix::new([[0.0, 0.0], [0.0, 0.25]]);
    filter.predict_to(time, None).unwrap();
    filter.predict_to(time, Some(&Matrix::zeros())).unwrap();
    assert_eq!(snapshot(&filter), original);
    assert!(filter.predict_to(time, Some(&q)).is_err());
    assert_eq!(snapshot(&filter), original);
    let later = time + Duration::from_seconds(10.0);
    filter.predict_to(later, Some(&q)).unwrap();
    assert_eq!(*filter.state(), original.0);
    assert_eq!(*filter.covariance(), original.1 + q);
    let predicted = snapshot(&filter);
    assert!(filter.predict_to(time, None).is_err());
    assert!(filter.predict_to(Instant::INVALID, None).is_err());
    assert!(
        filter
            .predict_to(
                later + Duration::from_seconds(1.0),
                Some(&Matrix::new([[1.0, 2.0], [2.0, 1.0]]))
            )
            .is_err()
    );
    assert_eq!(snapshot(&filter), predicted);
}

#[test]
fn gates_joint_reading_but_retains_prediction_and_accepts_same_epoch_updates() {
    let prior = prior();
    let time = prior.epoch + Duration::from_seconds(600.0);
    let station = station_near(&prior, time, 2.5);
    let mut filter = filter(prior.clone(), time - Duration::from_seconds(1.0));
    let saved = snapshot(&filter);
    let q = Matrix::new([[0.25, 0.0], [0.0, 0.01]]);
    let mut reading = measurement(&prior, time, station, 0.0);
    reading.range_rate += 1e6;
    let noise = tca_measurement_noise(0.5).unwrap();
    let result = filter
        .observe(
            &reading,
            &noise,
            Some(&q),
            UpdateOptions { gate: Some(10.0) },
        )
        .unwrap();
    assert!(!result.accepted && result.nis > 10.0);
    assert_eq!(snapshot(&filter), (saved.0, saved.1 + q, time));
    reading.range_rate -= 1e6;
    assert!(
        filter
            .observe(&reading, &noise, None, UpdateOptions::default())
            .unwrap()
            .accepted
    );
    let p = *filter.covariance();
    assert!(
        filter
            .observe(&reading, &noise, None, UpdateOptions::default())
            .unwrap()
            .accepted
    );
    assert!(filter.covariance()[(1, 1)] < p[(1, 1)]);
    assert_covariance(filter.covariance());
}

#[test]
fn weaker_pointing_noise_reduces_angular_influence() {
    let prior = prior();
    let time = prior.epoch + Duration::from_seconds(600.0);
    let truth = corrected(&prior, 30.0);
    let station = station_near(&truth, time, 2.5);
    let reading = measurement(&truth, time, station, 0.0);
    let mut strong = filter(prior, time);
    let mut weak = TcaUkf::new(
        strong.prior_tle().clone(),
        strong.epoch(),
        *strong.state(),
        *strong.covariance(),
    )
    .unwrap();
    let mut noise = tca_measurement_noise(1e6).unwrap();
    strong
        .observe(&reading, &noise, None, UpdateOptions::default())
        .unwrap();
    noise.sigma_az *= 100.0;
    noise.sigma_el *= 100.0;
    weak.observe(&reading, &noise, None, UpdateOptions::default())
        .unwrap();
    assert!(strong.state()[0].abs() > weak.state()[0].abs() * 2.0);
    assert!(strong.covariance()[(0, 0)] < weak.covariance()[(0, 0)]);
}

#[test]
fn observation_errors_roll_back_prediction_and_update() {
    let prior = prior();
    let time = prior.epoch + Duration::from_seconds(600.0);
    let station = station_near(&prior, time, 2.5);
    let mut reading = measurement(&prior, time, station, 0.0);
    let mut filter = filter(prior, time - Duration::from_seconds(1.0));
    let saved = snapshot(&filter);
    let noise = tca_measurement_noise(0.5).unwrap();
    let q = Matrix::new([[1.0, 0.0], [0.0, 1.0]]);
    reading.station.lat_deg = 91.0;
    assert!(
        filter
            .observe(&reading, &noise, Some(&q), UpdateOptions::default())
            .is_err()
    );
    assert_eq!(snapshot(&filter), saved);
    reading.station = station;
    reading.elevation = std::f64::consts::PI;
    assert!(
        filter
            .observe(&reading, &noise, Some(&q), UpdateOptions::default())
            .is_err()
    );
    assert_eq!(snapshot(&filter), saved);
    reading.elevation = 0.5;
    for gate in [f64::NAN, f64::INFINITY, -1.0] {
        assert!(
            filter
                .observe(
                    &reading,
                    &noise,
                    Some(&q),
                    UpdateOptions { gate: Some(gate) }
                )
                .is_err()
        );
        assert_eq!(snapshot(&filter), saved);
    }
    for invalid in [0.0, -1.0, f64::NAN, f64::INFINITY, 1e200] {
        let mut bad = noise;
        bad.sigma_el = invalid;
        assert!(
            filter
                .observe(&reading, &bad, Some(&q), UpdateOptions::default())
                .is_err()
        );
        assert_eq!(snapshot(&filter), saved);
    }
    for component in 0..3 {
        let mut bad = reading;
        match component {
            0 => bad.range_rate = f64::NAN,
            1 => bad.azimuth = f64::INFINITY,
            _ => bad.elevation = f64::NAN,
        }
        assert!(
            filter
                .observe(&bad, &noise, Some(&q), UpdateOptions::default())
                .is_err()
        );
        assert_eq!(snapshot(&filter), saved);
    }
    reading.time = saved.2 - Duration::from_seconds(1.0);
    assert!(
        filter
            .observe(&reading, &noise, None, UpdateOptions::default())
            .is_err()
    );
    assert_eq!(snapshot(&filter), saved);
}

#[test]
fn invalid_sigma_epoch_and_undefined_azimuth_roll_back_prediction() {
    let prior = prior();
    let time = prior.epoch + Duration::from_seconds(600.0);
    let station = station_near(&prior, time, 2.5);
    let reading = measurement(&prior, time, station, 0.0);
    let noise = tca_measurement_noise(0.5).unwrap();
    let mut wide = TcaUkf::new(
        prior.clone(),
        time - Duration::from_seconds(1.0),
        Vector::zeros(),
        Matrix::new([[1e26, 0.0], [0.0, 1.0]]),
    )
    .unwrap();
    let saved = snapshot(&wide);
    assert!(
        wide.observe(&reading, &noise, None, UpdateOptions::default())
            .is_err()
    );
    assert_eq!(snapshot(&wide), saved);
    let overhead = station_near(&prior, time, 0.0);
    let mut filter = filter(prior, time - Duration::from_seconds(1.0));
    let saved = snapshot(&filter);
    let reading = Measurement {
        station: overhead,
        ..reading
    };
    assert!(matches!(
        filter.observe(&reading, &noise, None, UpdateOptions::default()),
        Err(KernelError::UndefinedAzimuth)
    ));
    assert_eq!(snapshot(&filter), saved);
}

#[test]
fn validates_prior_state_covariance_and_default_noise() {
    let prior = prior();
    let time = prior.epoch;
    let noise = tca_measurement_noise(0.25).unwrap();
    assert_eq!(noise.sigma_rr, 0.25);
    assert_eq!(noise.sigma_az, 2.5_f64.to_radians());
    assert_eq!(noise.sigma_el, 2.5_f64.to_radians());
    for sigma in [0.0, -1.0, f64::NAN, f64::INFINITY, 1e200, 1e-200] {
        assert!(tca_measurement_noise(sigma).is_err());
    }
    for initial in [
        Vector::from_array([f64::NAN, 0.0]),
        Vector::from_array([0.0, f64::INFINITY]),
        Vector::from_array([1e20, 0.0]),
    ] {
        assert!(TcaUkf::new(prior.clone(), time, initial, Matrix::eye()).is_err());
    }
    for p in [
        Matrix::zeros(),
        Matrix::new([[1.0, 2.0], [2.0, 1.0]]),
        Matrix::new([[1.0, 0.2], [0.0, 1.0]]),
        Matrix::new([[f64::NAN, 0.0], [0.0, 1.0]]),
    ] {
        assert!(TcaUkf::new(prior.clone(), time, Vector::zeros(), p).is_err());
    }
    assert!(
        TcaUkf::new(
            prior.clone(),
            Instant::INVALID,
            Vector::zeros(),
            Matrix::eye()
        )
        .is_err()
    );
    let mut bad = prior.clone();
    bad.ephem_type = 4;
    assert!(matches!(
        TcaUkf::new(bad, time, Vector::zeros(), Matrix::eye()),
        Err(KernelError::Tle(_))
    ));
    let mut bad = prior;
    bad.epoch = Instant { raw: i64::MAX - 1 };
    assert!(TcaUkf::new(bad, time, Vector::from_array([1.0, 0.0]), Matrix::eye()).is_err());
}

#[test]
fn failed_sigma_point_sgp4_propagation_rolls_back_prediction() {
    let mut prior = prior();
    prior.bstar = 1.0;
    let epoch = prior.epoch;
    let time = epoch + Duration::from_seconds(1.0);
    let station = station_near(&prior, time, 2.5);
    let reading = measurement(&prior, time, station, 0.0);
    let mut filter = TcaUkf::new(
        prior,
        epoch,
        Vector::zeros(),
        Matrix::new([[1e12, 0.0], [0.0, 1.0]]),
    )
    .unwrap();
    let saved = snapshot(&filter);
    let result = filter.observe(
        &reading,
        &tca_measurement_noise(0.5).unwrap(),
        Some(&Matrix::eye()),
        UpdateOptions::default(),
    );
    assert!(matches!(result, Err(KernelError::Tle(_))), "{result:?}");
    assert_eq!(snapshot(&filter), saved);
}
