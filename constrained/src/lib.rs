//! Rust interior-point batch fitting for seven-dimensional state and drag.
#![forbid(unsafe_code)]
mod initialization;
pub use ripopt::SolveStatus;
use ripopt::{NlpProblem, SolverOptions};
use std::{collections::BTreeMap, fmt};
pub use tychonic::optim::constrained::*;
use tychonic::{
    kernels::{KernelError, M7, V7},
    optim::ContactId,
};

/// Tolerances apply to the scaled NLP and the final physical cone check respectively.
#[derive(Debug)]
pub struct FitOptions {
    pub scales: [f64; 7],
    pub max_iterations: usize,
    pub tolerance: f64,
    pub angular_tolerance_rad: f64,
}
impl Default for FitOptions {
    fn default() -> Self {
        Self {
            scales: DEFAULT_SCALES,
            max_iterations: 200,
            tolerance: 1e-5,
            angular_tolerance_rad: 1e-8,
        }
    }
}

#[derive(Debug)]
pub struct ConstrainedFit {
    pub parameters: V7,
    pub biases: BTreeMap<ContactId, f64>,
    pub rss: f64,
    pub pointings: Vec<ContactPointing>,
    pub status: SolveStatus,
    pub iterations: usize,
    pub sensitivity_evaluations: usize,
    pub worst_violation_rad: f64,
}

#[derive(Debug)]
pub enum FitError {
    ConflictingPointings {
        first: ContactId,
        second: ContactId,
        time: tychonic::satkit::Instant,
    },
    Evaluation(KernelError),
    Solver {
        status: SolveStatus,
        iterations: usize,
    },
    InfeasibleResult {
        drag: f64,
        worst_violation_rad: f64,
    },
}
impl fmt::Display for FitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Evaluation(e) => e.fmt(f),
            Self::ConflictingPointings {
                first,
                second,
                time,
            } => write!(
                f,
                "disjoint pointing cones for contacts {first} and {second} at {time:?}"
            ),
            Self::Solver { status, iterations } => write!(
                f,
                "ripopt stopped with {status:?} after {iterations} iterations"
            ),
            Self::InfeasibleResult {
                drag,
                worst_violation_rad,
            } => write!(
                f,
                "returned orbit violates physical bounds (drag {drag}, angular excess {worst_violation_rad} rad)"
            ),
        }
    }
}
impl std::error::Error for FitError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Evaluation(e) => Some(e),
            _ => None,
        }
    }
}
impl From<KernelError> for FitError {
    fn from(e: KernelError) -> Self {
        Self::Evaluation(e)
    }
}

#[derive(Default)]
struct Cache {
    point: Option<([f64; 7], RangeRateEvaluation)>,
    error: Option<KernelError>,
    evaluations: usize,
}

struct Adapter<'a> {
    model: &'a ConstrainedRangeRateProblem,
    origin: V7,
    basis: M7,
    initial: [f64; 7],
    // ripopt requires shared callback receivers. Each sequential solve owns this cache.
    #[allow(clippy::disallowed_types)]
    cache: std::cell::RefCell<Cache>,
}

impl<'a> Adapter<'a> {
    fn new(model: &'a ConstrainedRangeRateProblem, scales: &[f64; 7]) -> Result<Self, KernelError> {
        let start = initialization::prepare(model, scales)?;
        let initialization::Start {
            origin,
            basis,
            initial,
            value,
            evaluations,
        } = start;
        let cache = Cache {
            point: Some((initial, value)),
            evaluations,
            error: None,
        };
        Ok(Self {
            model,
            origin,
            basis,
            initial,
            cache: cache.into(),
        })
    }

    fn scaled_gradient(&self, physical: &[f64; 7], out: &mut [f64]) {
        let scaled = self.basis.transpose() * V7::from_slice(physical);
        out.copy_from_slice(scaled.as_slice());
    }

    fn physical(&self, x: &[f64; 7]) -> V7 {
        let seed = &self.origin;
        let mut result = seed + self.basis * V7::from_slice(x);
        if x[6] == -seed[6] / self.basis[(6, 6)] {
            result[6] = 0.;
        }
        result
    }

    fn read(&self, x: &[f64], write: impl FnOnce(&RangeRateEvaluation)) -> bool {
        let Ok(key) = <&[f64; 7]>::try_from(x) else {
            return false;
        };
        let mut cache = self.cache.borrow_mut();
        if cache.point.as_ref().map(|p| &p.0) != Some(key) {
            cache.point = None;
            cache.evaluations += 1;
            match self.model.evaluate(&self.physical(key)) {
                Ok(value) => {
                    cache.point = Some((*key, value));
                    cache.error = None;
                }
                Err(e) => {
                    cache.error = Some(e);
                    return false;
                }
            }
        }
        if let Some((_, value)) = &cache.point {
            write(value);
            return true;
        }
        false
    }
}
impl NlpProblem for Adapter<'_> {
    fn num_variables(&self) -> usize {
        7
    }
    fn num_constraints(&self) -> usize {
        self.model.selected_pointings().len()
    }
    fn bounds(&self, lower: &mut [f64], upper: &mut [f64]) {
        lower.fill(f64::NEG_INFINITY);
        upper.fill(f64::INFINITY);
        lower[6] = -self.origin[6] / self.basis[(6, 6)];
    }
    fn constraint_bounds(&self, lower: &mut [f64], upper: &mut [f64]) {
        lower.fill(f64::NEG_INFINITY);
        upper.fill(1.);
    }
    fn initial_point(&self, x: &mut [f64]) {
        x.copy_from_slice(&self.initial);
    }
    fn objective(&self, x: &[f64], _: bool, out: &mut f64) -> bool {
        self.read(x, |v| *out = v.rss * 0.5)
    }
    fn gradient(&self, x: &[f64], _: bool, out: &mut [f64]) -> bool {
        self.read(x, |v| self.scaled_gradient(&v.gradient, out))
            && out.iter().all(|v| v.is_finite())
    }
    fn constraints(&self, x: &[f64], _: bool, out: &mut [f64]) -> bool {
        self.read(x, |v| out.copy_from_slice(&v.cone_values))
    }
    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (0..self.num_constraints())
            .flat_map(|row| (0..7).map(move |k| (row, k)))
            .unzip()
    }
    fn jacobian_values(&self, x: &[f64], _: bool, out: &mut [f64]) -> bool {
        self.read(x, |v| {
            for (row, jac) in out.as_chunks_mut::<7>().0.iter_mut().zip(&v.cone_jacobian) {
                self.scaled_gradient(jac, row);
            }
        }) && out.iter().all(|v| v.is_finite())
    }
    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![], vec![])
    }
    fn hessian_values(&self, _: &[f64], _: bool, _: f64, _: &[f64], out: &mut [f64]) -> bool {
        out.is_empty()
    }
}

/// Minimize projected range-rate RSS with hard circular FOV constraints.
pub fn fit_constrained_range_rate(
    model: &ConstrainedRangeRateProblem,
    options: &FitOptions,
) -> Result<ConstrainedFit, FitError> {
    validate_options(options)?;
    if let Some((a, b)) = model.conflicting_pointings() {
        return Err(FitError::ConflictingPointings {
            first: a.contact,
            second: b.contact,
            time: a.time,
        });
    }
    let adapter = Adapter::new(model, &options.scales)?;
    let mut settings = SolverOptions {
        hessian_approximation_lbfgs: true,
        enable_preprocessing: false,
        detect_linear_constraints: false,
        bound_relax_factor: 0.,
        print_level: 0,
        tol: options.tolerance,
        constr_viol_tol: options.tolerance,
        max_iter: options.max_iterations,
        ..Default::default()
    };
    let result = solve_with_barrier_retry(&adapter, &mut settings);
    let failure = || FitError::Solver {
        status: result.status,
        iterations: result.iterations,
    };
    if !matches!(
        result.status,
        SolveStatus::Optimal | SolveStatus::Acceptable
    ) {
        if let Some(error) = adapter.cache.borrow_mut().error.take() {
            return Err(error.into());
        }
        return Err(failure());
    }
    let key = result.x.as_slice().try_into().map_err(|_| failure())?;
    let parameters = adapter.physical(key);
    adapter.read(key, |_| {});
    let cache = adapter.cache.into_inner();
    let (_, evaluation) = cache
        .point
        .ok_or_else(|| cache.error.map(FitError::from).unwrap_or_else(failure))?;
    if parameters[6] < 0. || evaluation.worst_violation_rad > options.angular_tolerance_rad {
        return Err(FitError::InfeasibleResult {
            drag: parameters[6],
            worst_violation_rad: evaluation.worst_violation_rad,
        });
    }
    Ok(ConstrainedFit {
        parameters,
        biases: evaluation.biases,
        rss: evaluation.rss,
        pointings: model.selected_pointings().copied().collect(),
        status: result.status,
        iterations: result.iterations,
        sensitivity_evaluations: cache.evaluations,
        worst_violation_rad: evaluation.worst_violation_rad,
    })
}

// A stalled line search can discard a feasible iterate before satisfying
// stationarity. Retry with an adaptive barrier from the validated start, sharing
// the caller's iteration budget. Only a solver-confirmed termination is accepted.
fn solve_with_barrier_retry(
    adapter: &Adapter<'_>,
    settings: &mut SolverOptions,
) -> ripopt::SolveResult {
    let first = ripopt::solve(adapter, settings);
    if first.status != SolveStatus::NumericalError
        || first.iterations >= settings.max_iter
        || adapter.cache.borrow().error.is_some()
    {
        return first;
    }
    settings.max_iter -= first.iterations;
    settings.mu_strategy_adaptive = true;
    let mut retry = ripopt::solve(adapter, settings);
    retry.iterations += first.iterations;
    retry
}

fn validate_options(options: &FitOptions) -> Result<(), FitError> {
    if !(options.scales.iter().all(|s| s.is_finite() && *s > 0.)
        && options.tolerance.is_finite()
        && options.tolerance > 0.
        && options.angular_tolerance_rad.is_finite()
        && options.angular_tolerance_rad >= 0.)
    {
        return Err(KernelError::InvalidInput("invalid NLP scales or tolerances").into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tychonic::{kernels::*, optim::ContactRangeRate, satkit::Instant};

    #[test]
    fn callbacks_share_exact_key_and_invalidate_failed_refresh() {
        let epoch = Instant::from_date(2024, 1, 1).unwrap();
        let site = GroundStation {
            lat_deg: 0.,
            lon_deg: 0.,
            alt_m: 0.,
        };
        let rate = ContactRangeRate {
            contact: ContactId::new(1),
            station: site,
            time: epoch,
            range_rate: 432.,
        };
        let config = PropagationConfig::new(
            epoch,
            od_prop_settings(),
            ParameterMode::StateAndDrag { delta_cd: 1e-4 },
        )
        .unwrap();
        let model = ConstrainedRangeRateProblem::new(
            config,
            V7::from_slice(&[6.7e6, 0., 0., 0., 7600., 400., 0.]),
            &[rate],
            &[1.],
            &[],
            1.,
            Default::default(),
        )
        .unwrap();
        let adapter = Adapter::new(&model, &DEFAULT_SCALES).unwrap();
        let mut x = [0.; 7];
        let mut f = -1.;
        assert!(adapter.objective(&x, true, &mut f));
        assert_eq!(f, 0.); // one rate sample is completely absorbed by its pass bias
        assert!(adapter.constraints(&x, true, &mut []));
        assert!(adapter.gradient(&x, false, &mut [0.; 7]));
        assert!(adapter.jacobian_values(&x, true, &mut []));
        assert_eq!(adapter.cache.borrow().evaluations, 1);
        x[6] = -0.1;
        assert!(!adapter.objective(&x, false, &mut f));
        assert!(adapter.cache.borrow().point.is_none());
        x[6] = 0.;
        assert!(adapter.gradient(&x, false, &mut [0.; 7]));
        assert_eq!(adapter.cache.borrow().evaluations, 3);
    }
}
