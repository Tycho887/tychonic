#![allow(clippy::disallowed_methods)]
#[path = "support/mod.rs"]
mod copies;
use copies::*;

use numeris::{DynVector, Matrix, Vector, optim::LmSettings};
use satkit::{Duration, Instant};
use tychonic::{estimate::*, kernels::*, optim::*};

fn epoch() -> Instant {
    Instant::from_datetime(2024, 1, 1, 12, 0, 0.25).unwrap()
}
fn setup() -> (
    PropagationConfig,
    PreparedGeometry,
    V6,
    Interferometer,
    RfPrior,
) {
    let times: Vec<_> = [0.0, 10.0, 30.0]
        .iter()
        .map(|dt| epoch() + Duration::from_seconds(*dt))
        .collect();
    let g = PreparedGeometry::new(
        &times,
        &vec![
            GroundStation {
                lat_deg: 40.0,
                lon_deg: -75.0,
                alt_m: 100.0
            };
            times.len()
        ],
    )
    .unwrap();
    let row = g.row(0).unwrap();
    let x = v6_from(
        &(row.position() + row.rotation().to_gcrf(&v3(50e3, 80e3, 400e3))),
        &(row.velocity() + row.rotation().to_gcrf(&v3(7100.0, 3000.0, 0.0))),
    );
    let mut mean = x.as_slice().to_vec();
    mean.extend([2.0, 0.7]);
    let prior = RfPrior::new(
        DynVector::from_slice(&mean),
        DynVector::from_slice(&[1000.0, 1000.0, 1000.0, 1.0, 1.0, 1.0, 10.0, 3.0]),
    )
    .unwrap();
    (
        PropagationConfig::new(
            epoch(),
            od_prop_settings(),
            ParameterMode::StateOnly { fixed_drag: 0.02 },
        )
        .unwrap(),
        g,
        x,
        Interferometer::new(2.23e9).unwrap(),
        prior,
    )
}

#[test]
fn exact_differential_state_and_baseline_jacobians_match_central_differences() {
    let (cfg, g, x, interferometer, _) = setup();
    let s = cfg
        .sample(&DynVector::from_slice(x.as_slice()), g.timestamps())
        .unwrap();
    for i in 0..g.len() {
        let row = g.row(i).unwrap();
        let t = g.timestamps()[i];
        let x = s.states()[i];
        let predicted = interferometer
            .differential(&x, row, &baseline(), t)
            .unwrap();
        for k in 0..6 {
            let step = if k < 3 { 5.0 } else { 1.0 };
            let mut xp = x;
            let mut xm = x;
            xp[k] += step;
            xm[k] -= step;
            let fd = (interferometer
                .differential(&xp, row, &baseline(), t)
                .unwrap()
                .difference_hz
                - interferometer
                    .differential(&xm, row, &baseline(), t)
                    .unwrap()
                    .difference_hz)
                / (2.0 * step);
            assert!((fd - predicted.partial[(0, k)]).abs() < 2e-9 + 2e-5 * fd.abs());
        }
        for k in 0..3 {
            let mut delta = V3::zeros();
            delta[k] = 0.01;
            let a = interferometer
                .differential(&x, row, &(baseline() + delta), t)
                .unwrap();
            let b = interferometer
                .differential(&x, row, &(baseline() + (-delta)), t)
                .unwrap();
            let fd = (a.difference_hz - b.difference_hz) / 0.02;
            assert!((fd - predicted.baseline_partial[(0, k)]).abs() < 1e-7 + 1e-5 * fd.abs());
        }
        let baseline = row.rotation().to_gcrf(&baseline());
        let omega = satkit::frametransform::qitrf2gcrf_approx(&t)
            * v3(0.0, 0.0, satkit::consts::OMEGA_EARTH);
        let rho = r_of(&x) - row.position();
        let los = rho / rho.norm();
        let w = v_of(&x) - row.velocity();
        let losdot = (w - los * los.dot(&w)) / rho.norm();
        let first_order = interferometer.carrier_hz() / SPEED_OF_LIGHT
            * (baseline.dot(&losdot) + omega.cross(&baseline).dot(&los));
        assert!((first_order - predicted.difference_hz).abs() < 0.002);
    }
}

#[test]
fn spherical_phase_has_stable_paths_and_analytic_state_baseline_partials() {
    let (_, g, x, int, _) = setup();
    let row = g.row(0).unwrap();
    let int = int.with_spherical_phase();
    let pred = int.evaluate(&x, row, &baseline(), 2.0).unwrap();
    assert!(int.uses_spherical_phase());
    for k in 0..6 {
        let step = if k < 3 { 0.5 } else { 0.1 };
        let mut xp = x;
        let mut xm = x;
        xp[k] += step;
        xm[k] -= step;
        let fd = (int.evaluate(&xp, row, &baseline(), 2.0).unwrap().phase_rad
            - int.evaluate(&xm, row, &baseline(), 2.0).unwrap().phase_rad)
            / (2.0 * step);
        assert!((fd - pred.partial[(1, k)]).abs() < 2e-9 + 1e-5 * fd.abs());
    }
    let h = pred.phase_baseline_partial_gcrf * row.rotation().matrix();
    for k in 0..3 {
        let mut delta = V3::zeros();
        delta[k] = 0.001;
        let fd = (int
            .evaluate(&x, row, &(baseline() + delta), 2.0)
            .unwrap()
            .phase_rad
            - int
                .evaluate(&x, row, &(baseline() + (-delta)), 2.0)
                .unwrap()
                .phase_rad)
            / 0.002;
        assert!((fd - h[(0, k)]).abs() < 1e-7);
    }
    let rho = r_of(&x) - row.position();
    let b = row.rotation().to_gcrf(&baseline());
    let path = (rho + b).norm() - rho.norm();
    let frequency = int.carrier_hz() + pred.doppler_hz;
    assert!(
        (pred.phase_rad - 2.0 * std::f64::consts::PI / SPEED_OF_LIGHT * frequency * path).abs()
            < 1e-7
    );
}

#[test]
fn zero_baseline_cancellation_and_shared_noise_covariance() {
    let (_, g, x, _, _) = setup();
    let zero = Interferometer::new(2.23e9)
        .unwrap()
        .differential(&x, g.row(0).unwrap(), &V3::zeros(), epoch())
        .unwrap();
    assert_eq!(zero.difference_hz, 0.0);
    assert_eq!(zero.partial, Matrix::<f64, 1, 6>::zeros());
    let data = PassiveRfData::new(
        g,
        Interferometer::new(2.23e9).unwrap(),
        vec![V3::zeros(); 3],
        vec![0.0; 3],
        Some(vec![0.0; 3]),
        0.3,
        0.1,
    )
    .unwrap();
    let dd = DifferentialDopplerData {
        difference_hz: vec![0.0; 3],
        sigma_antenna2_hz: 0.4,
    };
    let r = dd.covariance(&data);
    assert_eq!(r[(0, 2)], 0.09);
    assert_eq!(r[(2, 2)], 0.25);
    // Independent f1/f2 likelihood is identical under the invertible transform.
    let errors = Vector::from_array([0.2, 0.05, 0.2 - (-0.1)]);
    let rss = errors.dot(&r.cholesky().unwrap().solve(&errors));
    assert!(
        (rss - (0.2_f64 / 0.3).powi(2) - (0.05_f64 / 0.1).powi(2) - (-0.1_f64 / 0.4).powi(2)).abs()
            < 1e-12
    );
    let mut invalid = DifferentialDopplerData {
        difference_hz: dd.difference_hz.to_vec(),
        sigma_antenna2_hz: dd.sigma_antenna2_hz,
    };
    invalid.sigma_antenna2_hz = 0.0;
    assert!(invalid.validate(&data).is_err());
    let no_phase = PassiveRfData::new(
        data.geometry(),
        data.interferometer(),
        data.baselines_enu_m().to_vec(),
        data.doppler_hz().to_vec(),
        None,
        data.sigma_doppler_hz(),
        data.sigma_phase_rad(),
    )
    .unwrap();
    assert!(dd.validate(&no_phase).is_err());
    assert!(
        differential_doppler_partial(
            &x,
            data.geometry().row(0).unwrap().position(),
            data.geometry().row(0).unwrap().velocity(),
            &V3::zeros(),
            &v3(f64::NAN, 0.0, 0.0),
            2.23e9
        )
        .is_err()
    );
}

#[test]
fn estimated_drag_and_baseline_have_explicit_augmented_layouts() {
    let (original, _) = problem(false);
    let mut mean = original.prior().mean().as_slice()[..6].to_vec();
    mean.extend([0.02, 2.0, 0.7]);
    let prior = RfPrior::new(
        DynVector::from_slice(&mean),
        DynVector::from_slice(&[1000.0, 1000.0, 1000.0, 1.0, 1.0, 1.0, 0.01, 10.0, 3.0]),
    )
    .unwrap();
    let cfg = PropagationConfig::new(
        epoch(),
        od_prop_settings(),
        ParameterMode::StateAndDrag { delta_cd: 1e-4 },
    )
    .unwrap();
    let ou = OuPhaseNoise::new(21600.0, 0.087).unwrap();
    let problem = PassiveRfProblem::new_extended(
        copy_config(&cfg),
        original.data(),
        &prior,
        ou,
        original.extensions(),
    )
    .unwrap();
    assert_eq!(problem.columns(), 12);
    let mut filter =
        PassiveRfEkf::<13>::new_extended(cfg, ou, &prior, Some(v3(0.005, 0.005, 0.005))).unwrap();
    filter
        .filter_data_extended(
            problem.data(),
            problem.extensions().differential_doppler.as_ref(),
            UpdateOptions::default(),
        )
        .unwrap();
    assert!(filter.state()[6] >= 0.0);
    assert!(filter.smooth().is_ok());
    let e = problem.evaluate(&problem.initial_parameters()).unwrap();
    assert!(e.rss() < 1e-12);
    let (first, second) = polar_sites();
    let remote_baseline = first.baseline_enu_to(&second).unwrap();
    let remote = Interferometer::new(2.23e9).unwrap();
    let times = problem.data().geometry().timestamps();
    let states = problem
        .config()
        .sample(&DynVector::from_slice(&prior.mean().as_slice()[..7]), times)
        .unwrap()
        .states()
        .to_vec();
    let mut ext = copy_extensions(problem.extensions());
    ext.differential_interferometer = Some(remote);
    ext.differential_baselines_enu_m = Some(vec![remote_baseline; times.len()]);
    ext.differential_baseline_sigma_m = Some(v3(0.005, 0.005, 0.005));
    ext.differential_doppler.as_mut().unwrap().difference_hz = states
        .iter()
        .enumerate()
        .map(|(i, x)| {
            remote
                .differential(
                    x,
                    problem.data().geometry().row(i).unwrap(),
                    &remote_baseline,
                    times[i],
                )
                .unwrap()
                .difference_hz
        })
        .collect();
    let combined = PassiveRfProblem::new_extended(
        copy_config(problem.config()),
        problem.data(),
        &prior,
        ou,
        &ext,
    )
    .unwrap();
    assert_eq!(combined.columns(), 15);
    assert!(
        combined
            .evaluate(&combined.initial_parameters())
            .unwrap()
            .rss()
            < 1e-12
    );
    let mut f = PassiveRfEkf::<16>::new_with_calibration(
        copy_config(combined.config()),
        ou,
        &prior,
        ext.baseline_sigma_m,
        ext.differential_baseline_sigma_m,
    )
    .unwrap();
    f.filter_data_with_extensions(combined.data(), &ext, UpdateOptions::default())
        .unwrap();
    assert!(f.smooth().is_ok());
}

fn problem(same_epoch: bool) -> (PassiveRfProblem<'static>, DynVector<f64>) {
    let (cfg, mut g, x, int, prior) = setup();
    if same_epoch {
        g = PreparedGeometry::new(&[epoch(); 3], &[g.stations()[0]; 3]).unwrap();
    }
    let s = cfg
        .sample(&DynVector::from_slice(x.as_slice()), g.timestamps())
        .unwrap();
    let predictions: Vec<_> = (0..g.len())
        .map(|i| {
            int.evaluate(&s.states()[i], g.row(i).unwrap(), &baseline(), 2.0)
                .unwrap()
        })
        .collect();
    let dd = DifferentialDopplerData {
        difference_hz: (0..g.len())
            .map(|i| {
                int.differential(
                    &s.states()[i],
                    g.row(i).unwrap(),
                    &baseline(),
                    g.timestamps()[i],
                )
                .unwrap()
                .difference_hz
            })
            .collect(),
        sigma_antenna2_hz: 0.3,
    };
    let data = PassiveRfData::new(
        g,
        int,
        vec![baseline(); s.timestamps().len()],
        predictions.iter().map(|p| p.doppler_hz).collect(),
        Some(predictions.iter().map(|p| p.phase_rad + 0.7).collect()),
        0.3,
        0.01,
    )
    .unwrap();
    let problem = PassiveRfProblem::new_extended(
        cfg,
        data,
        prior,
        OuPhaseNoise::new(21600.0, 0.087).unwrap(),
        RfExtensions {
            baseline_sigma_m: Some(v3(0.005, 0.005, 0.005)),
            differential_doppler: Some(dd),
            ..RfExtensions::default()
        },
    )
    .unwrap();
    let x = problem.initial_parameters();
    (problem, x)
}

#[test]
fn full_whitened_batch_has_correct_orbit_and_baseline_columns() {
    let (problem, x) = problem(false);
    let e = problem.evaluate(&x).unwrap();
    assert!(e.rss() < 1e-15);
    assert_eq!(e.jacobian.ncols(), 11);
    for k in 0..11 {
        let step = if k < 3 {
            1.0
        } else if k < 6 {
            0.05
        } else if k < 8 {
            0.1
        } else {
            0.001
        };
        let mut xp = x.clone();
        let mut xm = x.clone();
        xp[k] += step;
        xm[k] -= step;
        let a = problem.evaluate(&xp).unwrap();
        let b = problem.evaluate(&xm).unwrap();
        for row in 0..e.residual.len() {
            let fd = (a.residual[row] - b.residual[row]) / (2.0 * step);
            let h = e.jacobian[(row, k)];
            assert!(
                (fd - h).abs() < 1e-5 + 1e-4 * h.abs(),
                "[{row},{k}] {fd} vs {h}"
            );
        }
    }
    let fit = fit_passive_rf(&problem, &x, &LmSettings::default()).unwrap();
    assert_eq!(fit.parameters.len(), 11);
}

#[test]
fn augmented_baseline_filter_matches_same_epoch_batch_covariance() {
    let (problem, x) = problem(true);
    let fit = fit_passive_rf(&problem, &x, &LmSettings::default()).unwrap();
    let mut filter = PassiveRfEkf::<12>::new_extended(
        copy_config(problem.config()),
        OuPhaseNoise::new(21600.0, 0.087).unwrap(),
        problem.prior(),
        Some(v3(0.005, 0.005, 0.005)),
    )
    .unwrap();
    filter
        .filter_data_extended(
            problem.data(),
            problem.extensions().differential_doppler.as_ref(),
            UpdateOptions::default(),
        )
        .unwrap();
    for i in 0..11 {
        for j in 0..11 {
            let fi = if i >= 8 { i + 1 } else { i };
            let fj = if j >= 8 { j + 1 } else { j };
            let a = fit.covariance[(i, j)];
            let b = filter.covariance()[(fi, fj)];
            assert!((a - b).abs() < 2e-6 + 2e-7 * a.abs());
        }
    }
    assert_eq!(filter.timestamps().len(), 1);
    let smooth = filter.smooth().unwrap();
    assert_eq!(smooth.last().unwrap().0, *filter.state());
    let t = epoch() + Duration::from_seconds(10.0);
    filter.predict_to(t, None).unwrap();
    for k in 9..12 {
        assert_eq!(filter.steps().get(1).unwrap().f_jacobian[(k, k)], 1.0);
    }
    assert!(filter.smooth().is_ok());
}

fn polar_sites() -> (GroundStation, GroundStation) {
    (
        GroundStation {
            lat_deg: 69.662,
            lon_deg: 18.940,
            alt_m: 100.0,
        },
        GroundStation {
            lat_deg: 78.230,
            lon_deg: 15.410,
            alt_m: 450.0,
        },
    )
}

#[test]
fn continental_chord_matches_independent_station_geometry_and_exact_range_rates() {
    let (first, second) = polar_sites();
    let times = [epoch(), epoch() + Duration::from_seconds(1000.0)];
    let g1 = PreparedGeometry::new(&times, &[first; 2]).unwrap();
    let g2 = PreparedGeometry::new(&times, &[second; 2]).unwrap();
    let b = first.baseline_enu_to(&second).unwrap();
    assert!((900e3..1100e3).contains(&b.norm()));
    // Curvature produces a large up component: a surface distance in a flat
    // local plane is not the vector joining the two antenna reference points.
    assert!(b[2] > 70e3);
    let int = Interferometer::new(2.23e9).unwrap();
    for (i, &time) in times.iter().enumerate() {
        let a = g1.row(i).unwrap();
        let z = g2.row(i).unwrap();
        let bg = a.rotation().to_gcrf(&b);
        let omega = satkit::frametransform::qitrf2gcrf_approx(&time)
            * v3(0.0, 0.0, satkit::consts::OMEGA_EARTH);
        assert!((a.position() - bg - z.position()).norm() < 3e-9);
        assert!((a.velocity() - omega.cross(&bg) - z.velocity()).norm() < 2e-12);
        let x = v6_from(
            &(a.position() + a.rotation().to_gcrf(&v3(0.0, 450e3, 600e3))),
            &(a.velocity() + a.rotation().to_gcrf(&v3(-3000.0, 6500.0, 0.0))),
        );
        let pred = int.differential(&x, a, &b, time).unwrap();
        let (rr1, h1) =
            range_rate_partial(&r_of(&x), &v_of(&x), a.position(), a.velocity()).unwrap();
        let (rr2, h2) =
            range_rate_partial(&r_of(&x), &v_of(&x), z.position(), z.velocity()).unwrap();
        assert!((pred.difference_hz - 2.23e9 / SPEED_OF_LIGHT * (rr2 - rr1)).abs() < 1e-9);
        assert!((pred.partial - (h2 - h1) * (2.23e9 / SPEED_OF_LIGHT)).norm_inf() < 1e-12);
        for k in 0..6 {
            let step = if k < 3 { 1.0 } else { 0.01 };
            let mut xp = x;
            let mut xm = x;
            xp[k] += step;
            xm[k] -= step;
            let fd = (int.differential(&xp, a, &b, time).unwrap().difference_hz
                - int.differential(&xm, a, &b, time).unwrap().difference_hz)
                / (2.0 * step);
            assert!((fd - pred.partial[(0, k)]).abs() < 1e-7);
        }
        for k in 0..3 {
            let mut d = V3::zeros();
            d[k] = 0.01;
            let fd = (int
                .differential(&x, a, &(b + d), time)
                .unwrap()
                .difference_hz
                - int
                    .differential(&x, a, &(b + (-d)), time)
                    .unwrap()
                    .difference_hz)
                / 0.02;
            assert!((fd - pred.baseline_partial[(0, k)]).abs() < 1e-7);
        }
    }
    assert!(
        GroundStation {
            lat_deg: 91.0,
            ..first
        }
        .baseline_enu_to(&second)
        .is_err()
    );
}

fn separate_problem(same_epoch: bool, local_calibration: bool) -> PassiveRfProblem<'static> {
    let (original, _) = problem(same_epoch);
    let (first, second) = polar_sites();
    let remote_baseline = first.baseline_enu_to(&second).unwrap();
    let int = Interferometer::new(2.23e9).unwrap();
    let states = original
        .config()
        .sample(
            &DynVector::from_slice(&original.prior().mean().as_slice()[..6]),
            original.data().geometry().timestamps(),
        )
        .unwrap()
        .states()
        .to_vec();
    let dd = DifferentialDopplerData {
        difference_hz: states
            .iter()
            .enumerate()
            .map(|(i, x)| {
                int.differential(
                    x,
                    original.data().geometry().row(i).unwrap(),
                    &remote_baseline,
                    original.data().geometry().timestamps()[i],
                )
                .unwrap()
                .difference_hz
            })
            .collect(),
        sigma_antenna2_hz: 0.3,
    };
    // The chord is expressed in antenna-1 ENU; using it at the test station
    // still defines a separate Earth-fixed receiver and exercises the layouts.
    PassiveRfProblem::new_extended(
        copy_config(original.config()),
        copy_data_owned(original.data()),
        copy_prior(original.prior()),
        OuPhaseNoise::new(21600.0, 0.087).unwrap(),
        RfExtensions {
            baseline_sigma_m: local_calibration.then_some(v3(0.005, 0.005, 0.005)),
            differential_doppler: Some(dd),
            differential_interferometer: Some(int),
            differential_baselines_enu_m: Some(vec![remote_baseline; states.len()]),
            differential_baseline_sigma_m: Some(v3(0.005, 0.005, 0.005)),
        },
    )
    .unwrap()
}

#[test]
fn separate_phase_and_doppler_baseline_columns_match_finite_differences() {
    for local in [false, true] {
        let problem = separate_problem(false, local);
        let x = problem.initial_parameters();
        let e = problem.evaluate(&x).unwrap();
        assert!(e.rss() < 1e-12);
        for k in 0..problem.columns() {
            let step = if k < 3 {
                1.0
            } else if k < 6 {
                0.05
            } else if k < 8 {
                0.1
            } else {
                0.001
            };
            let mut xp = x.clone();
            let mut xm = x.clone();
            xp[k] += step;
            xm[k] -= step;
            let a = problem.evaluate(&xp).unwrap();
            let b = problem.evaluate(&xm).unwrap();
            for row in 0..e.residual.len() {
                let fd = (a.residual[row] - b.residual[row]) / (2.0 * step);
                let h = e.jacobian[(row, k)];
                assert!(
                    (fd - h).abs() < 1e-5 + 1e-4 * h.abs(),
                    "[{row},{k}] {fd} vs {h}"
                );
            }
        }
        let n = problem.data().geometry().len();
        let remote = problem.columns() - 3;
        for i in 0..n {
            for k in remote..remote + 3 {
                assert_eq!(e.jacobian[(n + i, k)], 0.0);
            }
            if local {
                for k in 8..11 {
                    assert_eq!(e.jacobian[(2 * n + i, k)], 0.0);
                }
            }
        }
    }
}

fn compare_separate_covariance<const N: usize>(local: bool) {
    let problem = separate_problem(true, local);
    let fit = fit_passive_rf(
        &problem,
        &problem.initial_parameters(),
        &LmSettings::default(),
    )
    .unwrap();
    let ext = problem.extensions();
    let mut f = PassiveRfEkf::<N>::new_with_calibration(
        copy_config(problem.config()),
        OuPhaseNoise::new(21600.0, 0.087).unwrap(),
        problem.prior(),
        ext.baseline_sigma_m,
        ext.differential_baseline_sigma_m,
    )
    .unwrap();
    f.filter_data_with_extensions(problem.data(), ext, UpdateOptions::default())
        .unwrap();
    for i in 0..problem.columns() {
        for j in 0..problem.columns() {
            let fi = if i >= 8 { i + 1 } else { i };
            let fj = if j >= 8 { j + 1 } else { j };
            let a = fit.covariance[(i, j)];
            let b = f.covariance()[(fi, fj)];
            assert!(
                (a - b).abs() < 1e-5 + 1e-6 * a.abs(),
                "[{i},{j}]: {a} vs {b}"
            );
        }
    }
    assert!(f.smooth().is_ok());
    let mut bad = copy_extensions(ext);
    bad.differential_interferometer = None;
    assert!(
        PassiveRfProblem::new_extended(
            copy_config(problem.config()),
            problem.data(),
            problem.prior(),
            OuPhaseNoise::new(21600.0, 0.087).unwrap(),
            bad
        )
        .is_err()
    );
}

#[test]
fn independent_remote_calibration_filter_matches_batch_covariance() {
    compare_separate_covariance::<12>(false);
    compare_separate_covariance::<15>(true);
}

fn baseline() -> V3 {
    v3(59.0, 3.0, 0.2)
}
