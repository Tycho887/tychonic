//! Sequential LM with one fallible residual/Jacobian evaluation per point.
use super::{FitError, LmOutcome};
use crate::kernels::{Evaluation, KernelError, Result};
use numeris::{
    DynMatrix, DynVector,
    optim::{LmSettings, OptimError},
};

pub(super) fn run_lm(
    columns: usize,
    guess: &DynVector<f64>,
    scales: Option<&[f64]>,
    settings: &LmSettings<f64>,
    evaluate: impl FnMut(&DynVector<f64>, &mut Evaluation) -> Result<()>,
) -> std::result::Result<LmOutcome, FitError> {
    run_lm_with_drag(
        columns,
        guess,
        scales,
        settings,
        (columns == 7).then_some(6),
        evaluate,
    )
}

pub(super) fn run_lm_with_drag(
    columns: usize,
    guess: &DynVector<f64>,
    scales: Option<&[f64]>,
    settings: &LmSettings<f64>,
    drag_index: Option<usize>,
    mut evaluate: impl FnMut(&DynVector<f64>, &mut Evaluation) -> Result<()>,
) -> std::result::Result<LmOutcome, FitError> {
    let defaults = [100.0, 100.0, 100.0, 0.1, 0.1, 0.1, 0.01];
    let scales = scales.unwrap_or_else(|| defaults.get(..columns).unwrap_or(&[]));
    let coordinates = Coordinates::new(guess, scales, columns, drag_index)?;
    validate_settings(settings)?;
    let mut current = Point::new(columns);
    current
        .parameters
        .as_mut_slice()
        .copy_from_slice(guess.as_slice());
    evaluate(&current.parameters, &mut current.evaluation)?;
    let rows = current.evaluation.residual.len();
    current.cost = evaluation_cost(&current.evaluation, rows, columns)?;
    let mut trial = Point::new(columns);
    let mut scaled = DynMatrix::zeros(rows, columns);
    let mut mu = settings.mu_init;
    for iteration in 0..settings.max_iter {
        scale_jacobian(&current.evaluation, scales, &mut scaled);
        let gradient = gradient(&scaled, &current.evaluation.residual)?;
        if gradient.norm() < settings.grad_tol {
            return Ok(current.finish(iteration));
        }
        let step = damped_step(&scaled, &gradient, mu)?;
        coordinates.trial(&current.correction, &step, &mut trial)?;
        if !coordinates.admissible(&trial.parameters) {
            mu = increase_damping(mu, settings);
            continue;
        }
        evaluate(&trial.parameters, &mut trial.evaluation)?;
        trial.cost = evaluation_cost(&trial.evaluation, rows, columns)?;
        let predicted = predicted_reduction(&step, &gradient, mu)?;
        if !improves(current.cost, trial.cost, predicted) {
            mu = increase_damping(mu, settings);
            continue;
        }
        let actual = current.cost - trial.cost;
        std::mem::swap(&mut current, &mut trial);
        mu = (mu * settings.mu_decrease).max(settings.mu_min);
        if converged(&step, &current, actual, settings) {
            return Ok(current.finish(iteration + 1));
        }
    }
    Err(FitError::Solver(OptimError::MaxIterations))
}

struct Point {
    correction: DynVector<f64>,
    parameters: DynVector<f64>,
    evaluation: Evaluation,
    cost: f64,
}

impl Point {
    fn new(columns: usize) -> Self {
        Self {
            correction: DynVector::zeros(columns),
            parameters: DynVector::zeros(columns),
            evaluation: Evaluation::default(),
            cost: 0.0,
        }
    }
    fn finish(self, iterations: usize) -> LmOutcome {
        LmOutcome {
            parameters: self.parameters,
            evaluation: self.evaluation,
            iterations,
        }
    }
}

struct Coordinates<'a> {
    guess: &'a DynVector<f64>,
    scales: &'a [f64],
    drag_index: Option<usize>,
}

impl<'a> Coordinates<'a> {
    fn new(
        guess: &'a DynVector<f64>,
        scales: &'a [f64],
        columns: usize,
        drag_index: Option<usize>,
    ) -> Result<Self> {
        if columns == 0
            || guess.len() != columns
            || scales.len() != columns
            || drag_index.is_some_and(|i| i >= columns)
        {
            return Err(KernelError::InvalidInput("LM parameter dimensions"));
        }
        if !guess.as_slice().iter().all(|x| x.is_finite())
            || !scales.iter().all(|s| s.is_finite() && *s > 0.0)
        {
            return Err(KernelError::InvalidInput(
                "LM guess must be finite and scales finite and positive",
            ));
        }
        let result = Self {
            guess,
            scales,
            drag_index,
        };
        if !result.admissible(guess) {
            return Err(KernelError::InvalidInput(
                "initial drag must be nonnegative",
            ));
        }
        Ok(result)
    }
    fn trial(
        &self,
        current: &DynVector<f64>,
        step: &DynVector<f64>,
        trial: &mut Point,
    ) -> Result<()> {
        for k in 0..self.scales.len() {
            trial.correction[k] = current[k] + step[k];
            trial.parameters[k] = self.guess[k] + self.scales[k] * trial.correction[k];
        }
        require_finite(trial.parameters.as_slice())?;
        require_finite(trial.correction.as_slice())?;
        require_finite(&[trial.correction.norm()])
    }
    fn admissible(&self, parameters: &DynVector<f64>) -> bool {
        self.drag_index.is_none_or(|i| parameters[i] >= 0.0)
    }
}

fn validate_settings(s: &LmSettings<f64>) -> Result<()> {
    if ![s.grad_tol, s.x_tol, s.f_tol]
        .iter()
        .all(|v| v.is_finite() && *v >= 0.0)
    {
        return Err(KernelError::InvalidInput(
            "LM tolerances must be finite and nonnegative",
        ));
    }
    if ![s.mu_init, s.mu_min, s.mu_max, s.mu_increase, s.mu_decrease]
        .iter()
        .all(|v| v.is_finite() && *v > 0.0)
    {
        return Err(KernelError::InvalidInput(
            "LM damping settings must be finite and positive",
        ));
    }
    if s.mu_min > s.mu_init || s.mu_init > s.mu_max || s.mu_increase <= 1.0 || s.mu_decrease >= 1.0
    {
        return Err(KernelError::InvalidInput("LM damping bounds or factors"));
    }
    Ok(())
}

fn require_finite(values: &[f64]) -> Result<()> {
    if !values.iter().all(|v| v.is_finite()) {
        return Err(KernelError::Numerical(
            "LM arithmetic exceeds numeric range",
        ));
    }
    Ok(())
}

fn evaluation_cost(e: &Evaluation, rows: usize, columns: usize) -> Result<f64> {
    if rows == 0
        || e.residual.len() != rows
        || e.jacobian.nrows() != rows
        || e.jacobian.ncols() != columns
    {
        return Err(KernelError::InvalidInput(
            "LM evaluation dimensions changed or are empty",
        ));
    }
    require_finite(e.residual.as_slice())?;
    require_finite(e.jacobian.as_slice())?;
    let cost = 0.5 * e.rss();
    require_finite(&[cost])?;
    Ok(cost)
}

pub(super) fn scale_jacobian(e: &Evaluation, scales: &[f64], output: &mut DynMatrix<f64>) {
    for row in 0..output.nrows() {
        for (column, scale) in scales.iter().enumerate() {
            output[(row, column)] = e.jacobian[(row, column)] * scale;
        }
    }
}

fn gradient(j: &DynMatrix<f64>, residual: &DynVector<f64>) -> Result<DynVector<f64>> {
    let mut result = DynVector::zeros(j.ncols());
    for k in 0..j.ncols() {
        result[k] = (0..j.nrows()).map(|row| j[(row, k)] * residual[row]).sum();
    }
    require_finite(result.as_slice())?;
    require_finite(&[result.norm()])?;
    Ok(result)
}

fn damped_step(j: &DynMatrix<f64>, gradient: &DynVector<f64>, mu: f64) -> Result<DynVector<f64>> {
    let mut normal = j.transpose() * j;
    let mut negative = DynVector::zeros(gradient.len());
    for k in 0..gradient.len() {
        normal[(k, k)] += mu;
        negative[k] = -gradient[k];
    }
    require_finite(normal.as_slice())?;
    let step = match normal.cholesky() {
        Ok(factor) => factor.solve(&negative),
        Err(_) => normal
            .lu()
            .map_err(|_| KernelError::Singular)?
            .solve(&negative),
    };
    require_finite(step.as_slice())?;
    require_finite(&[step.norm()])?;
    Ok(step)
}

fn predicted_reduction(step: &DynVector<f64>, gradient: &DynVector<f64>, mu: f64) -> Result<f64> {
    let predicted = (0..step.len())
        .map(|k| step[k] * (mu * step[k] - gradient[k]))
        .sum();
    require_finite(&[predicted])?;
    Ok(predicted)
}

fn improves(cost: f64, trial_cost: f64, predicted: f64) -> bool {
    predicted > 0.0
        && predicted.abs() >= f64::EPSILON * cost.abs()
        && (cost - trial_cost) / predicted > 0.0
}

fn increase_damping(mu: f64, settings: &LmSettings<f64>) -> f64 {
    (mu * settings.mu_increase).min(settings.mu_max)
}

fn converged(
    step: &DynVector<f64>,
    point: &Point,
    actual: f64,
    settings: &LmSettings<f64>,
) -> bool {
    step.norm() < settings.x_tol * (1.0 + point.correction.norm())
        || actual.abs() < settings.f_tol * (1.0 + point.cost.abs())
}

#[cfg(test)]
#[allow(clippy::disallowed_methods)]
mod tests {
    use super::*;
    use numeris::optim::least_squares_lm_dyn;

    fn rosenbrock(x: &DynVector<f64>, output: &mut Evaluation) -> Result<()> {
        output.resize(2, 2)?;
        output.residual[0] = 10.0 * (x[1] - x[0] * x[0]);
        output.residual[1] = 1.0 - x[0];
        output.jacobian[(0, 0)] = -20.0 * x[0];
        output.jacobian[(0, 1)] = 10.0;
        output.jacobian[(1, 0)] = -1.0;
        output.jacobian[(1, 1)] = 0.0;
        Ok(())
    }

    #[test]
    fn combined_driver_matches_numeris_on_a_nonlinear_problem() {
        let guess = DynVector::from_slice(&[-1.2, 1.0]);
        let settings = LmSettings {
            max_iter: 500,
            ..LmSettings::default()
        };
        let expected = least_squares_lm_dyn(
            |x| {
                let mut e = Evaluation::default();
                rosenbrock(x, &mut e).unwrap();
                e.residual
            },
            |x| {
                let mut e = Evaluation::default();
                rosenbrock(x, &mut e).unwrap();
                e.jacobian
            },
            &guess,
            &settings,
        )
        .unwrap();
        let mut shifts = Vec::new();
        let actual = run_lm(2, &guess, Some(&[1.0, 1.0]), &settings, |x, output| {
            shifts.push([x[0], x[1]]);
            rosenbrock(x, output)
        })
        .unwrap();
        // The local driver solves corrections around guess. Compare with the
        // same coordinates in Numeris rather than changing its step tolerance.
        let expected_corrections = least_squares_lm_dyn(
            |z| {
                let x = DynVector::from_slice(&[guess[0] + z[0], guess[1] + z[1]]);
                let mut e = Evaluation::default();
                rosenbrock(&x, &mut e).unwrap();
                e.residual
            },
            |z| {
                let x = DynVector::from_slice(&[guess[0] + z[0], guess[1] + z[1]]);
                let mut e = Evaluation::default();
                rosenbrock(&x, &mut e).unwrap();
                e.jacobian
            },
            &DynVector::zeros(2),
            &settings,
        )
        .unwrap();
        for k in 0..2 {
            assert!((actual.parameters[k] - expected.x[k]).abs() < 1e-7);
            assert!((actual.parameters[k] - guess[k] - expected_corrections.x[k]).abs() < 1e-12);
        }
        assert_eq!(actual.iterations, expected_corrections.iterations);
        assert!(shifts.windows(2).any(|w| w[1][0] < w[0][0]));
        assert!(actual.evaluation.rss() < 1e-12);
    }

    #[test]
    fn malformed_or_nonfinite_evaluations_return_errors() {
        let guess = DynVector::zeros(1);
        let settings = LmSettings::default();
        let mut calls = 0;
        let error = run_lm(1, &guess, Some(&[1.0]), &settings, |_, output| {
            calls += 1;
            output.resize(if calls == 1 { 1 } else { 2 }, 1)?;
            output.residual.as_mut_slice().fill(1.0);
            output.jacobian.as_mut_slice().fill(1.0);
            Ok(())
        });
        assert!(matches!(
            error,
            Err(FitError::Kernel(KernelError::InvalidInput(_)))
        ));
        let error = run_lm(1, &guess, Some(&[1.0]), &settings, |_, output| {
            output.resize(1, 1)?;
            output.residual[0] = f64::MAX;
            output.jacobian[(0, 0)] = 1.0;
            Ok(())
        });
        assert!(matches!(
            error,
            Err(FitError::Kernel(KernelError::Numerical(_)))
        ));
    }

    #[test]
    fn negative_drag_trials_never_reach_the_evaluator() {
        let mut seen = Vec::new();
        let mut guess = DynVector::zeros(7);
        guess[6] = 0.01;
        let settings = LmSettings {
            max_iter: 20,
            grad_tol: 0.0,
            x_tol: 0.0,
            f_tol: 0.0,
            ..LmSettings::default()
        };
        let result = run_lm(7, &guess, None, &settings, |x, output| {
            seen.push(x[6]);
            output.resize(1, 7)?;
            output.residual[0] = x[6] + 1.0;
            output.jacobian.as_mut_slice().fill(0.0);
            output.jacobian[(0, 6)] = 1.0;
            Ok(())
        });
        assert!(matches!(
            result,
            Err(FitError::Solver(OptimError::MaxIterations))
        ));
        assert!(seen.iter().all(|drag| *drag >= 0.0));
        assert!(seen.len() < settings.max_iter + 1);
    }

    #[test]
    fn solver_settings_are_validated_before_evaluation_and_kernel_errors_survive() {
        let guess = DynVector::zeros(1);
        for mu_init in [f64::NAN, 0.0, f64::INFINITY] {
            let settings = LmSettings {
                mu_init,
                ..LmSettings::default()
            };
            let mut called = false;
            assert!(
                run_lm(1, &guess, Some(&[1.0]), &settings, |_, _| {
                    called = true;
                    Ok(())
                })
                .is_err()
            );
            assert!(!called);
        }
        let result = run_lm(1, &guess, Some(&[1.0]), &LmSettings::default(), |_, _| {
            Err(KernelError::Tle("test propagation failure".to_owned()))
        });
        assert!(
            matches!(result, Err(FitError::Kernel(KernelError::Tle(message))) if message == "test propagation failure")
        );
    }
}
