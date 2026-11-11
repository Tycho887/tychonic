#![allow(clippy::disallowed_methods)]
#[path = "support/mod.rs"]
mod copies;

use std::f64::consts::TAU;
use tychonic::{
    OrbitFilter,
    estimate::*,
    kernels::*,
    measurements::{self as m, *},
    numeris::{DynMatrix, DynVector, Matrix, optim::LmSettings},
    optim::{ODProblem, PassiveRfData, PassiveRfProblem, RfPrior, fit_lm},
    satkit::{Duration, Instant},
};

fn epoch() -> Instant {
    Instant::from_unixtime(1704110400.25)
}
fn config(drag: bool) -> PropagationConfig {
    PropagationConfig::new(
        epoch(),
        od_prop_settings(),
        if drag {
            ParameterMode::StateAndDrag { delta_cd: 1e-4 }
        } else {
            ParameterMode::StateOnly { fixed_drag: 0.02 }
        },
    )
    .unwrap()
}
fn geometry(seconds: &[f64]) -> PreparedGeometry {
    let times: Vec<_> = seconds
        .iter()
        .map(|s| epoch() + Duration::from_seconds(*s))
        .collect();

    PreparedGeometry::new(
        &times,
        &vec![
            GroundStation {
                lat_deg: 40.0,
                lon_deg: -75.0,
                alt_m: 100.0,
            };
            times.len()
        ],
    )
    .unwrap()
}
fn initial() -> V6 {
    let g = geometry(&[0.0]);
    let g = g.row(0).unwrap();
    v6_from(
        &(g.position() + g.rotation().to_gcrf(&v3(80e3, 40e3, 400e3))),
        &(g.velocity() + g.rotation().to_gcrf(&v3(7100.0, 3000.0, 0.0))),
    )
}
fn interferometer() -> Interferometer {
    Interferometer::new(2.23e9).unwrap()
}
fn selection(mask: u8) -> Vec<Observable> {
    let all = [
        Observable::Doppler {
            carrier_hz: 2.23e9,
            offset_hz: 0.0,
        },
        Observable::PhaseDifference {
            interferometer: interferometer(),
            baseline_enu_m: baseline(),
            frequency_offset_hz: 0.0,
            phase_offset_rad: 0.0,
        },
        Observable::DifferentialDoppler {
            interferometer: interferometer(),
            baseline_enu_m: baseline(),
        },
        Observable::AzimuthElevation,
    ];
    all.into_iter()
        .enumerate()
        .filter_map(|(i, o)| (mask & (1 << i) != 0).then_some(o))
        .collect()
}
fn batch_selection(mask: u8, measurements: usize) -> Vec<Observable<Vec<V3>>> {
    selection(mask)
        .into_iter()
        .map(|o| o.map_baseline(|b| vec![b; measurements]))
        .collect()
}
fn values(mask: u8, state: &V6, g: &StationGeometry, time: Instant) -> Vec<f64> {
    let mut z = Vec::new();
    if mask & 1 != 0 {
        z.push(
            -2.23e9 / SPEED_OF_LIGHT
                * range_rate_partial(&r_of(state), &v_of(state), g.position(), g.velocity())
                    .unwrap()
                    .0,
        );
    }
    if mask & 2 != 0 {
        z.push(
            interferometer()
                .evaluate(state, g, &baseline(), 0.0)
                .unwrap()
                .phase_rad,
        );
    }
    if mask & 4 != 0 {
        z.push(
            interferometer()
                .differential(state, g, &baseline(), time)
                .unwrap()
                .difference_hz,
        );
    }
    if mask & 8 != 0 {
        z.extend(
            az_el_partial_prepared(&r_of(state), g.position(), g.rotation())
                .unwrap()
                .0
                .as_slice(),
        );
    }
    z
}
fn noise(mask: u8) -> DynMatrix<f64> {
    let sigmas: Vec<_> = [(1, 0.5), (2, 0.01), (4, 0.7), (8, 1e-4), (8, 2e-4)]
        .into_iter()
        .filter_map(|(bit, s)| (mask & bit != 0).then_some(s))
        .collect();
    // A positive-definite correlated covariance, including cross-type terms.
    DynMatrix::from_fn(sigmas.len(), sigmas.len(), |i, j| {
        sigmas[i] * sigmas[j] * if i == j { 1.0 } else { 0.15 }
    })
}
fn batch(mask: u8, seconds: &[f64], drag: bool) -> (PreparedBatch<'static>, DynVector<f64>) {
    let g = geometry(seconds);
    let mut x = initial().as_slice().to_vec();
    if drag {
        x.push(0.02);
    }
    let x = DynVector::from_slice(&x);
    let sens = config(drag).sample(&x, g.timestamps()).unwrap();
    let z = (0..g.len())
        .map(|i| {
            values(
                mask,
                &sens.states()[i],
                g.row(i).unwrap(),
                g.timestamps()[i],
            )
        })
        .collect();
    let covariances = vec![noise(mask); g.len()];
    // Selection order is deliberately reversed; values retain canonical order.
    let mut observables = batch_selection(mask, g.len());
    observables.reverse();
    (joint(g, observables, z, covariances).unwrap(), x)
}
fn close(a: f64, b: f64, atol: f64, rtol: f64) {
    assert!((a - b).abs() <= atol + rtol * b.abs(), "{a} != {b}");
}

fn check_information<const M: usize>(batch: &PreparedBatch) {
    let observation = batch.joint_observation::<M>(0).unwrap();
    let (_, h) = observation.model.evaluate(&initial()).unwrap();
    let p = Matrix::<f64, 6, 6>::from_fn(|i, j| {
        if i == j {
            if i < 3 { 1e4 } else { 0.01 }
        } else {
            0.0
        }
    });
    let expected = (p.cholesky().unwrap().inverse()
        + h.transpose() * observation.covariance.cholesky().unwrap().inverse() * h)
        .cholesky()
        .unwrap()
        .inverse();
    let mut filter = OrbitFilter::new(config(false), initial(), p).unwrap();
    let result = filter
        .observe(&observation, None, UpdateOptions::default())
        .unwrap();
    assert!(result.accepted);
    for i in 0..6 {
        for j in 0..6 {
            close(filter.covariance()[(i, j)], expected[(i, j)], 2e-8, 2e-10);
        }
    }
}

#[test]
fn all_fifteen_selections_match_kernels_and_information_updates() {
    for mask in 1..16 {
        let (b, x) = batch(mask, &[0.0], false);
        let e = b.evaluate(&x, &config(false)).unwrap();
        assert!(e.rss() < 1e-18);
        match b.components() {
            1 => check_information::<1>(&b),
            2 => check_information::<2>(&b),
            3 => check_information::<3>(&b),
            4 => check_information::<4>(&b),
            5 => check_information::<5>(&b),
            _ => unreachable!(),
        }
        let mut filter = OrbitFilter::new(config(false), initial(), M6::eye()).unwrap();
        assert!(
            filter
                .filter_batches(&[b], |_, _, _| Ok(M6::zeros()), UpdateOptions::default())
                .unwrap()[0]
                .accepted
        );
    }
}

#[test]
fn every_selection_has_correct_chained_whitened_jacobians() {
    for drag in [false, true] {
        for mask in 1..16 {
            let (b, x) = batch(mask, &[30.0, 0.0, 30.0], drag);
            let c = config(drag);
            let e = b.evaluate(&x, &c).unwrap();
            for k in 0..x.len() {
                let step = if k < 3 {
                    0.5
                } else if k < 6 {
                    0.01
                } else {
                    1e-4
                };
                let mut plus = x.clone();
                let mut minus = x.clone();
                plus[k] += step;
                minus[k] -= step;
                let a = b.evaluate(&plus, &c).unwrap();
                let z = b.evaluate(&minus, &c).unwrap();
                for row in 0..b.rows() {
                    close(
                        (a.residual[row] - z.residual[row]) / (2.0 * step),
                        e.jacobian[(row, k)],
                        2e-5,
                        3e-4,
                    );
                }
            }
        }
    }
}

#[test]
fn named_batches_and_sensors_support_independent_observables() {
    let g = geometry(&[0.0]);
    let z = values(15, &initial(), g.row(0).unwrap(), epoch());
    let batches = [
        doppler(&g, 2.23e9, vec![z[0]], 0.5).unwrap(),
        phase_difference(
            &g,
            interferometer(),
            vec![baseline(); g.len()],
            vec![z[1]],
            0.01,
        )
        .unwrap(),
        differential_doppler(
            &g,
            interferometer(),
            vec![baseline(); g.len()],
            vec![z[2]],
            0.7,
        )
        .unwrap(),
        azimuth_elevation(&g, vec![z[3]], vec![z[4]], 1e-4, 2e-4).unwrap(),
    ];
    for b in &batches {
        assert_eq!(
            b.evaluate(&DynVector::from_slice(initial().as_slice()), &config(false))
                .unwrap()
                .rss(),
            0.0
        );
    }
    let sensors = [
        doppler_sensor(g.row(0).unwrap(), epoch(), 2.23e9).unwrap(),
        phase_difference_sensor(g.row(0).unwrap(), epoch(), interferometer(), baseline()).unwrap(),
        differential_doppler_sensor(g.row(0).unwrap(), epoch(), interferometer(), baseline())
            .unwrap(),
    ];
    for (i, s) in sensors.iter().enumerate() {
        close(s.evaluate(&initial()).unwrap().0[0], z[i], 0.0, 0.0);
    }
    let angles = azimuth_elevation_sensor(g.row(0).unwrap(), epoch())
        .unwrap()
        .evaluate(&initial())
        .unwrap()
        .0;
    assert_eq!(angles.as_slice(), &z[3..]);
    assert!(PreparedBatch::doppler(&g, 2.23e9, vec![z[0]], 0.5).is_ok());
    assert!(
        PreparedBatch::differential_doppler(
            &g,
            interferometer(),
            vec![baseline(); g.len()],
            vec![z[2]],
            0.7
        )
        .is_ok()
    );
}

#[test]
fn shared_antenna_noise_matches_two_independent_frequency_measurements() {
    let r = doppler_differential_covariance(0.5, 0.8).unwrap();
    let transform = Matrix::new([[1.0, 0.0], [1.0, -1.0]]);
    let independent = Matrix::new([[0.25, 0.0], [0.0, 0.64]]);
    let expected = transform * independent * transform.transpose();
    for i in 0..2 {
        for j in 0..2 {
            close(r[(i, j)], expected[(i, j)], 2e-16, 0.0);
        }
    }
    let (b, _) = batch(5, &[0.0], false);
    let observation = b.joint_observation::<2>(0).unwrap();
    let (_, h) = observation.model.evaluate(&initial()).unwrap();
    let h_antennas = transform * h;
    let joint_information = h.transpose() * r.cholesky().unwrap().inverse() * h;
    let antenna_information =
        h_antennas.transpose() * independent.cholesky().unwrap().inverse() * h_antennas;
    for i in 0..6 {
        for j in 0..6 {
            close(
                joint_information[(i, j)],
                antenna_information[(i, j)],
                1e-12,
                1e-12,
            );
        }
    }
    assert!(doppler_differential_covariance(0.0, 1.0).is_err());
    assert!(doppler_differential_covariance(1e308, 1.0).is_err());
}

#[test]
fn scheduling_gating_and_failed_joint_updates_preserve_history() {
    let (b, _) = batch(15, &[20.0, 0.0, 20.0], false);
    let mut filter = OrbitFilter::new(config(false), initial(), M6::eye()).unwrap();
    let mut q_calls = 0;
    let results = filter
        .filter_batches(
            std::slice::from_ref(&b),
            |_, _, _| {
                q_calls += 1;
                Ok(M6::zeros())
            },
            UpdateOptions::default(),
        )
        .unwrap();
    assert_eq!(q_calls, 1);
    assert_eq!(results.iter().map(|r| r.row).collect::<Vec<_>>(), [1, 0, 2]);
    assert_eq!(filter.timestamps().len(), 2);
    let mut observation = b.joint_observation::<5>(0).unwrap();
    observation.time += Duration::from_seconds(1.0);
    observation.covariance[(0, 0)] = -1.0;
    let saved = (
        *filter.state(),
        *filter.covariance(),
        filter.epoch(),
        filter.steps().len(),
    );
    assert!(
        filter
            .observe(&observation, None, UpdateOptions::default())
            .is_err()
    );
    assert_eq!(
        saved,
        (
            *filter.state(),
            *filter.covariance(),
            filter.epoch(),
            filter.steps().len()
        )
    );
    observation = b.joint_observation::<5>(0).unwrap();
    observation.value[0] += 1000.0;
    let result = filter
        .observe(&observation, None, UpdateOptions { gate: Some(1.0) })
        .unwrap();
    assert!(!result.accepted);
    assert_eq!(
        saved,
        (
            *filter.state(),
            *filter.covariance(),
            filter.epoch(),
            filter.steps().len()
        )
    );
}

#[test]
fn angular_wrapping_validation_and_offset_calibration() {
    let (b, x) = batch(15, &[0.0], false);
    let mut o = b.joint_observation::<5>(0).unwrap();
    let pred = o.model.evaluate(&initial()).unwrap().0;
    o.value[3] += TAU;
    close(o.model.innovation(&o.value, &pred)[3], 0.0, 1e-15, 0.0);
    o.value[1] += TAU;
    close(o.model.innovation(&o.value, &pred)[1], TAU, 5e-14, 0.0);
    let g = b.geometry();
    let mut z = values(15, &initial(), g.row(0).unwrap(), epoch());
    z[3] += TAU;
    let wrapped = joint(g, batch_selection(15, g.len()), vec![z], vec![noise(15)]).unwrap();
    assert!(wrapped.evaluate(&x, &config(false)).unwrap().rss() < 1e-20);
    assert!(joint(g, vec![], vec![vec![]], vec![DynMatrix::zeros(0, 0)]).is_err());
    assert!(joint_sensor::<2>(g.row(0).unwrap(), epoch(), selection(1)).is_err());
    assert!(
        joint(
            g,
            vec![selection(1)[0]; 2]
                .into_iter()
                .map(|o| o.map_baseline(|b| vec![b; g.len()]))
                .collect(),
            vec![vec![0.0; 2]],
            vec![noise(5)]
        )
        .is_err()
    );
    for r in [
        DynMatrix::zeros(1, 1),
        DynMatrix::from_fn(1, 1, |_, _| f64::NAN),
        DynMatrix::zeros(2, 2),
    ] {
        assert!(joint(g, batch_selection(1, g.len()), vec![vec![0.0]], vec![r]).is_err());
    }
    let mut r = noise(5);
    r[(0, 1)] += 0.1;
    assert!(joint(g, batch_selection(5, g.len()), vec![vec![0.0; 2]], vec![r]).is_err());
    assert!(doppler(g, -1.0, vec![0.0], 1.0).is_err());
    assert!(doppler(g, 1.0, vec![0.0], 1e-300).is_err());
    assert!(
        differential_doppler(g, interferometer(), vec![baseline(); g.len()], vec![], 1.0).is_err()
    );
    assert!(b.joint_observation::<5>(1).is_err());
    let offsets = vec![
        Observable::Doppler {
            carrier_hz: 2.23e9,
            offset_hz: 12.0,
        },
        Observable::PhaseDifference {
            interferometer: interferometer(),
            baseline_enu_m: baseline(),
            frequency_offset_hz: 12.0,
            phase_offset_rad: 0.4,
        },
    ];
    let sensor = joint_sensor::<2>(g.row(0).unwrap(), epoch(), offsets).unwrap();
    let z = sensor.evaluate(&initial()).unwrap().0;
    close(
        z[0],
        values(1, &initial(), g.row(0).unwrap(), epoch())[0] + 12.0,
        1e-10,
        0.0,
    );
    close(
        z[1],
        interferometer()
            .evaluate(&initial(), g.row(0).unwrap(), &baseline(), 12.0)
            .unwrap()
            .phase_rad
            + 0.4,
        0.0,
        0.0,
    );
}

#[test]
fn convenience_fits_and_epoch_union_match_advanced_solver() {
    let (first, x) = batch(15, &[0.0, 20.0, 40.0, 60.0, 80.0, 100.0, 120.0], false);
    let (second, _) = batch(4, &[10.0, 40.0, 90.0], false);
    let batches = vec![first, second];
    let mut guess = x.clone();
    guess[0] += 30.0;
    guess[4] += 0.02;
    let settings = LmSettings {
        grad_tol: 1e-2,
        x_tol: 1e-5,
        f_tol: 1e-5,
        ..LmSettings::default()
    };
    let advanced = fit_lm(
        &ODProblem::new(config(false), &batches).unwrap(),
        &guess,
        None,
        &settings,
    )
    .unwrap();
    let facade = m::fit_with_options(config(false), &batches, &guess, None, &settings).unwrap();
    assert_eq!(facade.parameters, advanced.parameters);
    assert_eq!(facade.rss, advanced.rss);
    assert!(facade.rss < 1e-8);
    let defaults = m::fit(config(false), batches, &x).unwrap();
    assert!(defaults.rss < 1e-18);
}

#[test]
fn named_rf_priors_match_advanced_batch_and_filter_setup() {
    let g = geometry(&[0.0, 20.0]);
    let x = DynVector::from_slice(initial().as_slice());
    let s = config(false).sample(&x, g.timestamps()).unwrap();
    let data = PassiveRfData::new(
        &g,
        interferometer(),
        vec![baseline(); g.len()],
        (0..g.len())
            .map(|i| {
                interferometer()
                    .evaluate(&s.states()[i], g.row(i).unwrap(), &baseline(), 3.0)
                    .unwrap()
                    .doppler_hz
            })
            .collect(),
        Some(
            (0..g.len())
                .map(|i| {
                    interferometer()
                        .evaluate(&s.states()[i], g.row(i).unwrap(), &baseline(), 3.0)
                        .unwrap()
                        .phase_rad
                        + 0.2
                })
                .collect(),
        ),
        0.5,
        0.01,
    )
    .unwrap();
    let priors = RfPriors {
        orbit_mean: x.clone(),
        orbit_sigma: DynVector::from_slice(&[100.0, 100.0, 100.0, 0.1, 0.1, 0.1]),
        frequency_offset_hz: GaussianPrior {
            mean: 3.0,
            sigma: 1.0,
        },
        phase_offset_rad: GaussianPrior {
            mean: 0.2,
            sigma: 0.5,
        },
    };
    let drift = OuPhaseNoise::new(21600.0, 0.1).unwrap();
    let builder = RfBuilder::new(config(false), &data, priors, drift);
    let advanced = PassiveRfProblem::new(
        config(false),
        &data,
        RfPrior::new(
            DynVector::from_slice(&[x[0], x[1], x[2], x[3], x[4], x[5], 3.0, 0.2]),
            DynVector::from_slice(&[100.0, 100.0, 100.0, 0.1, 0.1, 0.1, 1.0, 0.5]),
        )
        .unwrap(),
        drift,
    )
    .unwrap();
    let facade = builder.build().unwrap();
    assert_eq!(
        facade.evaluate(&facade.initial_parameters()).unwrap(),
        advanced.evaluate(&advanced.initial_parameters()).unwrap()
    );
    let mut filter = builder.filter::<9>().unwrap();
    let mut expected = PassiveRfEkf::<9>::new(config(false), drift, advanced.prior()).unwrap();
    assert_eq!(filter.state(), expected.state());
    builder
        .filter_data(&mut filter, UpdateOptions::default())
        .unwrap();
    expected
        .filter_data(advanced.data(), UpdateOptions::default())
        .unwrap();
    assert_eq!(filter.state(), expected.state());
    assert_eq!(filter.covariance(), expected.covariance());
    assert!(builder.filter::<10>().is_err());
    let fit = builder.fit().unwrap();
    assert!(fit.rss < 1e-15);
    assert!(
        builder
            .baseline_calibration(v3(0.01, 0.01, 0.01))
            .filter::<12>()
            .is_ok()
    );
}

fn baseline() -> V3 {
    v3(59.0, 3.0, 0.2)
}
