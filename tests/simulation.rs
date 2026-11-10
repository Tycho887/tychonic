//! Independent behavioral plant checks; no controller has access to truth.
use tychonic::{
    detection::EnuDirection,
    simulation::{Mount, MountConfig, Receiver, ReceiverConfig},
};

#[test]
fn cutoff_is_inclusive_and_loss_is_immediate() {
    let config = ReceiverConfig {
        acquisition_s: 0.2,
        ..ReceiverConfig::default()
    };
    let mut receiver = Receiver::new(config, 42).unwrap();
    let edge = 3.0_f64.to_radians();
    assert!(!receiver.sample(edge, 1e6, true, 123.0, 0.1).unwrap().locked);
    let sample = receiver.sample(edge, 1e6, true, 123.0, 0.1).unwrap();
    assert!(sample.locked);
    assert!(sample.doppler_hz.is_some());
    let sample = receiver
        .sample(edge + 1e-10, 1e6, true, 123.0, 0.1)
        .unwrap();
    assert!(!sample.locked);
    assert!(sample.doppler_hz.is_none());
}

#[test]
fn mount_moves_with_limits_and_does_not_teleport() {
    let config = MountConfig::default();
    let mut mount = Mount::new([0.0, 0.5], config).unwrap();
    mount.advance([0.3, 0.7], 0.1).unwrap();
    let angles = mount.angles();
    assert!(angles[0] > 0.0 && angles[0] < 0.3);
    for _ in 0..500 {
        mount.advance([0.3, 2.0], 0.1).unwrap();
    }
    assert!((mount.angles()[1] - config.max_elevation_rad).abs() < 1e-10);
    let direction = EnuDirection::from_az_el(mount.angles()[0], mount.angles()[1]).unwrap();
    assert!(direction.elevation_rad() <= config.max_elevation_rad + 1e-12);
}

#[test]
fn noise_is_seeded_and_unlocked_samples_are_absent() {
    let mut a = Receiver::new(ReceiverConfig::default(), 7).unwrap();
    let mut b = Receiver::new(ReceiverConfig::default(), 7).unwrap();
    for _ in 0..20 {
        assert_eq!(
            a.sample(0.0, 1e6, true, 100.0, 0.1).unwrap(),
            b.sample(0.0, 1e6, true, 100.0, 0.1).unwrap()
        );
    }
    assert!(
        a.sample(0.0, 1e6, false, 100.0, 0.1)
            .unwrap()
            .doppler_hz
            .is_none()
    );
}

#[test]
fn invalid_steps_preserve_mount_and_receiver() {
    let mut mount = Mount::new([0.0, 0.5], MountConfig::default()).unwrap();
    assert!(mount.advance([f64::NAN, 0.0], 0.1).is_err());
    assert_eq!(mount.angles(), [0.0, 0.5]);
    let mut receiver = Receiver::new(ReceiverConfig::default(), 1).unwrap();
    assert!(receiver.sample(0.0, 1e6, true, 1.0, -1.0).is_err());
    assert!(!receiver.sample(0.0, 1e6, true, 1.0, 0.1).unwrap().locked);
}

#[test]
fn doppler_matches_independent_range_difference_and_epoch_changes_clear_cache() {
    use tychonic::{
        kernels::{GroundStation, SPEED_OF_LIGHT},
        satkit::{Duration, TLE},
        simulation::{orbital_sample, shifted_epoch},
    };
    let mut tle = TLE::load_2line(
        "1 25544U 98067A   24356.58519896  .00014389  00000-0  25222-3 0  9992",
        "2 25544  51.6403 106.8969 0007877   6.1421 113.2479 15.50801739487617",
    )
    .unwrap();
    let station = GroundStation {
        lat_deg: 19.43636457546619,
        lon_deg: -30.817318939448704,
        alt_m: 0.0,
    };
    let time = tle.epoch + Duration::from_seconds(550.0);
    let before = orbital_sample(
        &mut tle,
        time - Duration::from_seconds(0.01),
        station,
        2.2e9,
    )
    .unwrap();
    let after = orbital_sample(
        &mut tle,
        time + Duration::from_seconds(0.01),
        station,
        2.2e9,
    )
    .unwrap();
    let sample = orbital_sample(&mut tle, time, station, 2.2e9).unwrap();
    let frequency =
        -(after.pointing.range_m - before.pointing.range_m) / 0.02 * 2.2e9 / SPEED_OF_LIGHT;
    assert!((sample.doppler_hz - frequency).abs() < 0.5);
    let mut shifted = shifted_epoch(&tle, 20.0).unwrap();
    let shifted_sample = orbital_sample(&mut shifted, time, station, 2.2e9).unwrap();
    assert!(
        sample
            .pointing
            .direction_enu
            .separation(shifted_sample.pointing.direction_enu)
            > 0.01
    );
    assert!(shifted_epoch(&tle, f64::NAN).is_err());
}
