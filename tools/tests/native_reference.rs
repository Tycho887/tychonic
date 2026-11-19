//! Compare independent NLP solvers on the same prepared orbit and cone constraints.
#![cfg(feature = "native-reference")]
#[path = "../../constrained/tests/support/mod.rs"]
mod support;
use ipopt::{BasicProblem, ConstrainedProblem, Index, Ipopt};
use tychonic::kernels::{M7, V7};
use tychonic_constrained::*;

struct Reference<'a> {
    model: &'a ConstrainedRangeRateProblem,
    basis: M7,
    #[allow(clippy::disallowed_types)]
    cache: std::cell::RefCell<Option<(V7, RangeRateEvaluation)>>,
}
impl Reference<'_> {
    fn physical(&self, x: &[f64]) -> V7 {
        self.model.seed() + self.basis * V7::from_slice(x)
    }
    fn read(&self, x: &[f64], use_value: impl FnOnce(&RangeRateEvaluation)) -> bool {
        let key = self.physical(x);
        let mut cache = self.cache.borrow_mut();
        if cache.as_ref().map(|p| &p.0) != Some(&key) {
            *cache = self.model.evaluate(&key).ok().map(|e| (key, e));
        }
        cache.as_ref().map(|p| use_value(&p.1)).is_some()
    }
}
impl BasicProblem for Reference<'_> {
    fn num_variables(&self) -> usize {
        7
    }
    fn bounds(&self, l: &mut [f64], u: &mut [f64]) -> bool {
        l.fill(-1e20);
        u.fill(1e20);
        l[6] = -self.model.seed()[6] / self.basis[(6, 6)];
        true
    }
    fn initial_point(&self, x: &mut [f64]) -> bool {
        x.fill(0.);
        true
    }
    fn objective(&self, x: &[f64], _: bool, f: &mut f64) -> bool {
        self.read(x, |e| *f = e.rss * 0.5)
    }
    fn objective_grad(&self, x: &[f64], _: bool, g: &mut [f64]) -> bool {
        self.read(x, |e| {
            g.copy_from_slice((self.basis.transpose() * V7::from_slice(&e.gradient)).as_slice());
        })
    }
}
impl ConstrainedProblem for Reference<'_> {
    fn num_constraints(&self) -> usize {
        self.model.selected_pointings().len()
    }
    fn num_constraint_jacobian_non_zeros(&self) -> usize {
        7 * self.num_constraints()
    }
    fn constraint_bounds(&self, l: &mut [f64], u: &mut [f64]) -> bool {
        l.fill(-1e20);
        u.fill(1.);
        true
    }
    fn constraint(&self, x: &[f64], _: bool, g: &mut [f64]) -> bool {
        self.read(x, |e| g.copy_from_slice(&e.cone_values))
    }
    fn constraint_jacobian_indices(&self, r: &mut [Index], c: &mut [Index]) -> bool {
        for i in 0..r.len() {
            r[i] = (i / 7) as Index;
            c[i] = (i % 7) as Index;
        }
        true
    }
    fn constraint_jacobian_values(&self, x: &[f64], _: bool, g: &mut [f64]) -> bool {
        self.read(x, |e| {
            for (row, j) in g.chunks_exact_mut(7).zip(&e.cone_jacobian) {
                row.copy_from_slice((self.basis.transpose() * V7::from_slice(j)).as_slice());
            }
        })
    }
    fn num_hessian_non_zeros(&self) -> usize {
        0
    }
    fn hessian_indices(&self, _: &mut [Index], _: &mut [Index]) -> bool {
        true
    }
    fn hessian_values(&self, _: &[f64], _: bool, _: f64, _: &[f64], _: &mut [f64]) -> bool {
        true
    }
}

#[test]
fn reference_feasible_and_infeasible_start() {
    let fixture = support::Fixture::new();
    for fov in [0.10, 0.0003] {
        let model = fixture.model(PointingSelection::default(), fov);
        assert_eq!(
            model.evaluate(&fixture.seed).unwrap().worst_violation_rad > 0.,
            fov == 0.0003
        );
        let rust = fit_constrained_range_rate(&model, &FitOptions::default()).unwrap();
        let x = native_solution(&model);
        let v = model.evaluate(&x).unwrap();
        println!(
            "FOV {fov}: native RSS {:.9}, Rust RSS {:.9}, native excess {:.3e} rad, Rust excess {:.3e} rad",
            v.rss, rust.rss, v.worst_violation_rad, rust.worst_violation_rad
        );
        assert!(v.worst_violation_rad <= 1e-8);
        assert!((v.rss - rust.rss).abs() < 1e-4);
        assert!((x - fixture.truth).block::<3, 1>(0, 0).norm() < 10.);
        for (i, b) in fixture.biases.iter().enumerate() {
            let key = tychonic::optim::ContactId::new(i as u128);
            assert!((rust.biases[&key] - v.biases[&key]).abs() < 0.01);
            assert!((b - v.biases[&key]).abs() < 0.01);
        }
    }
}

fn native_solution(model: &ConstrainedRangeRateProblem) -> V7 {
    let mut solver = Ipopt::new(Reference {
        model,
        basis: model.linearize(model.seed(), &DEFAULT_SCALES).unwrap().1,
        cache: Default::default(),
    })
    .unwrap();
    solver
        .set_option("hessian_approximation", "limited-memory")
        .unwrap();
    solver.set_option("bound_relax_factor", 0.).unwrap();
    solver.set_option("tol", 1e-5).unwrap();
    solver.set_option("print_level", 0).unwrap();
    let native = solver.solve();
    assert!(
        matches!(
            native.status,
            ipopt::SolveStatus::SolveSucceeded | ipopt::SolveStatus::SolvedToAcceptableLevel
        ),
        "{:?}",
        native.status
    );
    native
        .solver_data
        .problem
        .physical(native.solver_data.solution.primal_variables)
}

#[test]
fn compare_at_an_active_boundary() {
    let mut f = support::Fixture::new();
    let mut p = f.angles[25];
    p.elevation += 0.00102;
    f.angles = vec![p];
    let model = f.model(PointingSelection::All, 0.002);
    let rust = fit_constrained_range_rate(&model, &FitOptions::default()).unwrap();
    let x = native_solution(&model);
    let native = model.evaluate(&x).unwrap();
    assert!(native.worst_violation_rad <= 1e-8);
    assert!((native.cone_values[0] - 1.).abs() < 2e-4);
    assert!(
        (native.rss - rust.rss).abs() < 1e-4,
        "native {}, Rust {}",
        native.rss,
        rust.rss
    );
}
