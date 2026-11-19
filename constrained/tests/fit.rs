mod support;
use support::*;
use tychonic::{kernels::V7, optim::ContactId};
use tychonic_constrained::*;

#[test]
fn projected_objective_and_cone_derivatives() {
    let mut fixture = Fixture::new();
    fixture.angles.reverse();
    fixture.angles.push(fixture.angles[0]);
    let model = fixture.model(PointingSelection::default(), 0.005);
    assert_eq!(model.selected_pointings().len(), 9);
    assert_eq!(model.epochs().len(), fixture.rates.len());
    let truth = model.evaluate(&fixture.truth).unwrap();
    assert!(truth.rss < 1e-12);
    for (i, b) in fixture.biases.iter().enumerate() {
        assert!((truth.biases[&ContactId::new(i as u128)] - b).abs() < 1e-8);
    }
    let x = fixture.seed;
    let v = model.evaluate(&x).unwrap();
    for (k, scale) in DEFAULT_SCALES.iter().enumerate() {
        let h = scale * 0.015;
        let mut dx = V7::zeros();
        dx[k] = h;
        let plus = model.evaluate(&(x + dx)).unwrap();
        let minus = model.evaluate(&(x - dx)).unwrap();
        let fd = (plus.rss - minus.rss) / (4. * h) * scale;
        let g = v.gradient[k] * scale;
        assert!(
            (g - fd).abs() < 0.04 + 0.004 * fd.abs(),
            "objective column {k}: {g} vs {fd}"
        );
        for (j, row) in v.cone_jacobian.iter().enumerate() {
            let fd = (plus.cone_values[j] - minus.cone_values[j]) / (2. * h) * scale;
            assert!(
                (row[k] * scale - fd).abs() < 2e-4 + 0.004 * fd.abs(),
                "cone {j} column {k}"
            );
        }
    }
}

#[test]
fn rust_solver_recovers_infeasible_seed() {
    let fixture = Fixture::new();
    let model = fixture.model(PointingSelection::default(), 0.0003);
    assert!(model.evaluate(&fixture.seed).unwrap().worst_violation_rad > 0.);
    let result = fit_constrained_range_rate(&model, &FitOptions::default()).unwrap();
    assert!(result.worst_violation_rad <= 1e-8);
    assert!(result.rss < 0.05, "RSS {}", result.rss);
    assert!((result.parameters[6] - fixture.truth[6]).abs() < 0.002);
}

#[test]
#[ignore = "release accuracy and timing sweep"]
fn budget_sweep() {
    let mut fixture = Fixture::new();
    // Deterministic receiver noise and a pass-edge pointing just outside the
    // truth cone make the held-out feasibility check meaningful.
    for (i, (row, sigma)) in fixture.rates.iter_mut().zip(&fixture.sigmas).enumerate() {
        row.range_rate += 0.5 * sigma * (i as f64 * 2.3).sin();
    }
    fixture.angles[0].elevation += 0.00016;
    let full = fixture.model(PointingSelection::All, 0.0003);
    for selection in [
        PointingSelection::NearTca(1),
        PointingSelection::default(),
        PointingSelection::NearTca(10),
        PointingSelection::All,
    ] {
        let start = std::time::Instant::now();
        let model = fixture.model(selection, 0.0003);
        let fit = fit_constrained_range_rate(&model, &FitOptions::default()).unwrap();
        let elapsed = start.elapsed().as_secs_f64();
        let err = fit.parameters - fixture.truth;
        let max_bias = fixture
            .biases
            .iter()
            .enumerate()
            .map(|(i, b)| (fit.biases[&ContactId::new(i as u128)] - b).abs())
            .fold(0., f64::max);
        let full_eval = full.evaluate(&fit.parameters).unwrap();
        let selected: std::collections::BTreeSet<_> =
            fit.pointings.iter().map(|p| (p.contact, p.time)).collect();
        let worst_omitted = full
            .selected_pointings()
            .zip(full_eval.cone_values)
            .filter(|(p, _)| !selected.contains(&(p.contact, p.time)))
            .map(|(_, g)| 2. * (g.sqrt() * (0.0003_f64 / 4.).sin()).min(1.).asin() - 0.00015)
            .fold(0., f64::max);
        println!(
            "{selection:?}: cones={} samples={} time={elapsed:.3}s position={:.4}m velocity={:.6}m/s drag={:.8} bias={max_bias:.6}m/s omitted={worst_omitted:.3e}rad rss={:.7}",
            fit.pointings.len(),
            fit.sensitivity_evaluations,
            err.block::<3, 1>(0, 0).norm(),
            err.block::<3, 1>(3, 0).norm(),
            err[6].abs(),
            fit.rss
        );
        assert!(fit.rss < 100. && err.block::<3, 1>(0, 0).norm() < 20.);
        assert!(fit.worst_violation_rad <= 1e-8);
        if matches!(selection, PointingSelection::NearTca(3)) {
            assert!(worst_omitted > 0.);
        }
    }
}

#[test]
fn conflicting_cones_and_iteration_limit_are_reported() {
    let mut fixture = Fixture::new();
    let model = fixture.model(PointingSelection::default(), 0.0003);
    let error = fit_constrained_range_rate(
        &model,
        &FitOptions {
            max_iterations: 0,
            ..Default::default()
        },
    )
    .unwrap_err();
    assert!(
        matches!(
            error,
            FitError::Solver {
                status: SolveStatus::MaxIterations,
                ..
            }
        ),
        "{error}"
    );
    let first = fixture.angles[8];
    let extra = ContactId::new(9);
    let other = ContactPointing {
        contact: extra,
        elevation: -first.elevation,
        azimuth: first.azimuth + std::f64::consts::PI,
        ..first
    };
    fixture.angles = vec![first, other];
    fixture.rates.push(tychonic::optim::ContactRangeRate {
        contact: extra,
        ..fixture.rates[8]
    });
    fixture.sigmas.push(1.);
    let model = fixture.model(PointingSelection::All, 0.01);
    let error = fit_constrained_range_rate(&model, &FitOptions::default()).unwrap_err();
    assert!(
        matches!(error, FitError::ConflictingPointings { .. }),
        "{error}"
    );
}

#[test]
fn active_cone_changes_rate_optimum() {
    let mut f = Fixture::new();
    let mut angle = f.angles[25];
    let half_angle = 0.001;
    angle.elevation += half_angle + 0.00002;
    f.angles = vec![angle];
    let model = f.model(PointingSelection::All, 2. * half_angle);
    assert!(model.evaluate(&f.truth).unwrap().cone_values[0] > 1.);
    let fit = fit_constrained_range_rate(&model, &FitOptions::default()).unwrap();
    let at_fit = model.evaluate(&fit.parameters).unwrap();
    assert!(
        (at_fit.cone_values[0] - 1.).abs() < 2e-4,
        "{:?}",
        at_fit.cone_values
    );
    assert!(fit.worst_violation_rad < 1e-8 && fit.rss > 1e-5);
}

#[test]
fn drag_gradient_at_zero_uses_the_physical_side() {
    let f = Fixture::new();
    let model = f.model(PointingSelection::default(), 0.002);
    let mut p = f.seed;
    p[6] = 0.;
    let base = model.evaluate(&p).unwrap();
    let h = 2e-5;
    p[6] = h;
    let one = model.evaluate(&p).unwrap();
    p[6] = 2. * h;
    let two = model.evaluate(&p).unwrap();
    let fd = (-3. * base.rss + 4. * one.rss - two.rss) / (4. * h);
    assert!((fd - base.gradient[6]).abs() < 10. + 0.01 * fd.abs());
}

#[test]
fn rates_only_fit_matches_explicit_orbit_and_bias_solve() {
    let mut fixture = Fixture::new();
    fixture.angles.clear();
    let model = fixture.model(PointingSelection::default(), 0.1);
    let fit = fit_constrained_range_rate(&model, &FitOptions::default()).unwrap();
    assert!(fit.pointings.is_empty() && fit.rss < 1e-5);
    // Independent 10-variable fit with a physical bias column for each pass.
    use tychonic::kernels::{PreparedBatch, PreparedGeometry};
    use tychonic::numeris::{DynMatrix, DynVector};
    use tychonic::optim::ODProblem;
    let times: Vec<_> = fixture.rates.iter().map(|r| r.time).collect();
    let sites: Vec<_> = fixture.rates.iter().map(|r| r.station).collect();
    let values = fixture.rates.iter().map(|r| r.range_rate).collect();
    let geometry = PreparedGeometry::new(&times, &sites).unwrap();
    let batch = PreparedBatch::range_rate(geometry, values, 1.).unwrap();
    let raw = ODProblem::new(config_with_span(fixture.epoch, 5100.), vec![batch]).unwrap();
    let mut orbit = DynVector::from_slice(fixture.seed.as_slice());
    let mut biases = [0.; 3];
    for _ in 0..5 {
        let e = raw.evaluate(&orbit).unwrap();
        let mut j = DynMatrix::zeros(times.len(), 10);
        let mut rhs = DynVector::zeros(times.len());
        for (i, rate) in fixture.rates.iter().enumerate() {
            let contact = rate.contact.as_u128() as usize;
            let sigma = fixture.sigmas[i];
            rhs[i] = -(e.residual[i] + biases[contact]) / sigma;
            j[(i, 7 + contact)] = 1. / sigma;
            for k in 0..7 {
                j[(i, k)] = e.jacobian[(i, k)] * DEFAULT_SCALES[k] / sigma;
            }
        }
        let step = j.qr().unwrap().solve(&rhs);
        for k in 0..7 {
            orbit[k] += step[k] * DEFAULT_SCALES[k];
        }
        for i in 0..3 {
            biases[i] += step[7 + i];
        }
    }
    assert!(
        (V7::from_slice(orbit.as_slice()) - fit.parameters)
            .block::<3, 1>(0, 0)
            .norm()
            < 0.02
    );
    for (i, b) in biases.iter().enumerate() {
        assert!((fit.biases[&ContactId::new(i as u128)] - b).abs() < 1e-4);
    }
}

#[test]
#[ignore = "24 hour constrained state and drag qualification"]
fn constrained_24_hour_recovery() {
    let f = Fixture::with_centers([-43080., 0., 43080.]);
    let model = f.model(PointingSelection::default(), 0.001);
    // Daily propagation and drag differencing have a higher gradient noise floor.
    let settings = FitOptions {
        tolerance: 1e-3,
        ..Default::default()
    };
    let fit = fit_constrained_range_rate(&model, &settings).unwrap();
    let error = fit.parameters - f.truth;
    assert!(error.block::<3, 1>(0, 0).norm() < 10. && error.block::<3, 1>(3, 0).norm() < 0.1);
    assert!(error[6].abs() < 1e-4 && fit.worst_violation_rad <= 1e-8);
    println!(
        "24 h: position {:.5} m, velocity {:.7} m/s, drag {:.8} m²/kg, cone excess {:.3e} rad, {} sensitivity evaluations",
        error.block::<3, 1>(0, 0).norm(),
        error.block::<3, 1>(3, 0).norm(),
        error[6].abs(),
        fit.worst_violation_rad,
        fit.sensitivity_evaluations
    );
}

#[test]
fn preparation_preserves_rate_rows_and_distinct_constraint_epochs() {
    use tychonic::satkit::Duration;
    let mut f = Fixture::new();
    let baseline = f
        .model(PointingSelection::All, 0.1)
        .evaluate(&f.seed)
        .unwrap();
    f.rates.reverse();
    f.sigmas.reverse();
    let p = f.angles[1];
    f.angles = vec![
        p,
        ContactPointing {
            time: p.time + Duration::from_seconds(0.25),
            ..p
        },
        p,
    ];
    let model = f.model(PointingSelection::All, 0.1);
    assert_eq!(model.selected_pointings().len(), 2);
    assert_eq!(model.epochs().len(), f.rates.len() + 1);
    let e = model.evaluate(&f.seed).unwrap();
    assert!((e.rss - baseline.rss).abs() < 0.01);
    for (id, bias) in &e.biases {
        assert!((bias - baseline.biases[id]).abs() < 1e-5);
    }
    f.angles[0].station = f.rates[0].station;
    assert!(
        ConstrainedRangeRateProblem::new(
            config_with_span(f.epoch, 5100.),
            f.seed,
            &f.rates,
            &f.sigmas,
            &f.angles,
            0.1,
            Default::default()
        )
        .is_err()
    );
}
