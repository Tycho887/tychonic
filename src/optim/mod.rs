//! Thin stacking and LM integration over verified observation kernels.
pub mod constrained;
pub mod passive_rf;
pub mod varpro;
pub use passive_rf::*;
pub use varpro::*;

use crate::kernels::{
    Evaluation, KernelError, Measurement, MeasurementNoise, ParameterMode, PreparedBatch,
    PropagationConfig, Result,
};
use numeris::{
    DynVector,
    optim::{LmSettings, OptimError},
};
mod input;
pub use input::InputSource;
mod covariance;
pub use covariance::{
    CovarianceError, CovarianceFitError, CovarianceMethod, ParameterCovariance,
    parameter_covariance,
};
mod lm;
use lm::{run_lm, run_lm_with_drag};
use satkit::{Instant, orbitprop::PropSettings};
use std::{collections::BTreeMap, fmt};

/// Physical reference parameters are GCRF m, m/s, and constant Cd A/m in m²/kg.
pub const DEFAULT_SCALES: [f64; 7] = [100., 100., 100., 0.1, 0.1, 0.1, 0.01];

/// Immutable collection of batches with a prepared union of exact epochs.
#[derive(Debug)]
pub enum BatchSource<'g> {
    Owned(Vec<PreparedBatch<'g>>),
    Borrowed(&'g [PreparedBatch<'g>]),
}
impl<'g> From<Vec<PreparedBatch<'g>>> for BatchSource<'g> {
    fn from(value: Vec<PreparedBatch<'g>>) -> Self {
        Self::Owned(value)
    }
}
impl<'g> From<&'g [PreparedBatch<'g>]> for BatchSource<'g> {
    fn from(value: &'g [PreparedBatch<'g>]) -> Self {
        Self::Borrowed(value)
    }
}
impl<'g> From<&'g Vec<PreparedBatch<'g>>> for BatchSource<'g> {
    fn from(value: &'g Vec<PreparedBatch<'g>>) -> Self {
        Self::Borrowed(value)
    }
}
impl<'g> std::ops::Deref for BatchSource<'g> {
    type Target = [PreparedBatch<'g>];
    fn deref(&self) -> &Self::Target {
        match self {
            Self::Owned(value) => value,
            Self::Borrowed(value) => value,
        }
    }
}

#[derive(Debug)]
pub struct ODProblem<'g> {
    config: PropagationConfig,
    batches: BatchSource<'g>,
    timestamps: Vec<Instant>,
    mappings: Vec<Vec<usize>>,
    rows: usize,
    prior: Option<(DynVector<f64>, numeris::DynMatrix<f64>)>,
}

impl<'g> ODProblem<'g> {
    pub fn new(config: PropagationConfig, batches: impl Into<BatchSource<'g>>) -> Result<Self> {
        let batches = batches.into();
        if batches.is_empty() {
            return Err(KernelError::InvalidInput("empty observation problem"));
        }
        let mut epochs = BTreeMap::new();
        for batch in batches.iter() {
            for &time in batch.timestamps() {
                epochs.insert(time, 0);
            }
        }
        let timestamps: Vec<_> = epochs.keys().copied().collect();
        for (index, value) in epochs.values_mut().enumerate() {
            *value = index;
        }
        let mappings = batches
            .iter()
            .map(|b| b.timestamps().iter().map(|t| epochs[t]).collect())
            .collect();
        let rows = batches
            .iter()
            .try_fold(0usize, |n, batch| n.checked_add(batch.rows()))
            .ok_or(KernelError::InvalidInput("stacked dimensions overflow"))?;
        rows.checked_mul(config.mode().columns())
            .ok_or(KernelError::InvalidInput("stacked dimensions overflow"))?;
        Ok(Self {
            config,
            batches,
            timestamps,
            mappings,
            prior: None,
            rows,
        })
    }

    /// Include one independent Gaussian prior at the reference epoch.
    /// The rows appended to the objective are L^-1 (x - mean), where LL^T=P.
    pub fn with_prior(
        mut self,
        mean: DynVector<f64>,
        covariance: &numeris::DynMatrix<f64>,
    ) -> Result<Self> {
        let n = self.config.mode().columns();
        if self.prior.is_some() || mean.len() != n || !mean.as_slice().iter().all(|v| v.is_finite())
        {
            return Err(KernelError::InvalidInput("invalid or repeated orbit prior"));
        }
        let factor = crate::measurements::factor_covariance(covariance, n)?;
        let lower = factor.l();
        let mut whitening = numeris::DynMatrix::zeros(n, n);
        for i in 0..n {
            for j in 0..=i {
                let sum: f64 = (j..i).map(|k| lower[(i, k)] * whitening[(k, j)]).sum();
                whitening[(i, j)] = (f64::from(i == j) - sum) / lower[(i, i)];
            }
        }
        self.rows = self
            .rows
            .checked_add(n)
            .filter(|rows| rows.checked_mul(n).is_some())
            .ok_or(KernelError::InvalidInput("prior dimensions overflow"))?;
        self.prior = Some((mean, whitening));
        Ok(self)
    }

    pub fn from_measurements(
        reference_epoch: Instant,
        measurements: &[Measurement],
        noise: MeasurementNoise,
        settings: PropSettings,
        delta_cd: f64,
    ) -> Result<Self> {
        Self::new(
            PropagationConfig::new(
                reference_epoch,
                settings,
                ParameterMode::StateAndDrag { delta_cd },
            )?,
            vec![PreparedBatch::from_measurements(measurements, noise)?],
        )
    }

    pub fn config(&self) -> &PropagationConfig {
        &self.config
    }
    pub fn batches(&self) -> &[PreparedBatch<'g>] {
        &self.batches
    }
    pub fn timestamps(&self) -> &[Instant] {
        &self.timestamps
    }
    /// True when the objective includes independent prior information.
    pub fn uses_weighted_prior(&self) -> bool {
        self.prior.is_some()
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn evaluate(&self, parameters: &DynVector<f64>) -> Result<Evaluation> {
        let mut output = Evaluation::default();
        self.evaluate_into(parameters, &mut output)?;
        Ok(output)
    }

    pub fn evaluate_into(
        &self,
        parameters: &DynVector<f64>,
        output: &mut Evaluation,
    ) -> Result<()> {
        let sens = self.config.sample(parameters, &self.timestamps)?;
        let mut offset = 0;
        for (batch, mapping) in self.batches.iter().zip(&self.mappings) {
            batch.evaluate_mapped_into(&sens, Some(mapping), output, offset, self.rows)?;
            offset += batch.rows();
        }
        if let Some((mean, whitening)) = &self.prior {
            for i in 0..mean.len() {
                output.residual[offset + i] = (0..=i)
                    .map(|j| whitening[(i, j)] * (parameters[j] - mean[j]))
                    .sum();
                for j in 0..mean.len() {
                    output.jacobian[(offset + i, j)] = whitening[(i, j)];
                }
            }
        }
        Ok(())
    }
}

/// Last-parameter cache with exclusive access and reusable owned output.
///
/// A returned evaluation keeps the cache immutably borrowed until its last use:
/// ```compile_fail
/// use tychonic::{optim::CachedProblem, numeris::DynVector};
/// fn overlapping_access(cache: &mut CachedProblem<'_>, x: &DynVector<f64>) {
///     let previous = cache.evaluate(x).unwrap();
///     cache.evaluate(x).unwrap();
///     println!("{}", previous.rss());
/// }
/// ```
pub struct CachedProblem<'g> {
    inner: ODProblem<'g>,
    key: Vec<f64>,
    output: Evaluation,
    valid: bool,
}

impl<'g> CachedProblem<'g> {
    pub fn new(problem: ODProblem<'g>) -> Self {
        Self {
            inner: problem,
            key: Vec::new(),
            output: Evaluation::default(),
            valid: false,
        }
    }
    pub fn inner(&self) -> &ODProblem<'g> {
        &self.inner
    }
    pub fn evaluate(&mut self, parameters: &DynVector<f64>) -> Result<&Evaluation> {
        if !self.valid || self.key != parameters.as_slice() {
            self.valid = false;
            self.inner.evaluate_into(parameters, &mut self.output)?;
            self.key.clear();
            self.key.extend_from_slice(parameters.as_slice());
            self.valid = true;
        }
        Ok(&self.output)
    }
}

#[derive(Debug)]
pub enum FitError {
    Kernel(KernelError),
    Solver(OptimError),
}
impl fmt::Display for FitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Kernel(e) => e.fmt(f),
            Self::Solver(e) => write!(f, "LM solver failed: {e}"),
        }
    }
}
impl std::error::Error for FitError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Kernel(e) => Some(e),
            Self::Solver(_) => None,
        }
    }
}
impl From<KernelError> for FitError {
    fn from(e: KernelError) -> Self {
        Self::Kernel(e)
    }
}

#[derive(Debug)]
pub struct FitResult {
    pub parameters: DynVector<f64>,
    pub iterations: usize,
    pub rss: f64,
}

struct LmOutcome {
    parameters: DynVector<f64>,
    iterations: usize,
    evaluation: Evaluation,
}

/// Solve dimensionless corrections around `guess`. Defaults preserve the demo's
/// metre, m/s and drag scales. Evaluations are fallible and sequential.
pub fn fit_lm(
    problem: &ODProblem<'_>,
    guess: &DynVector<f64>,
    scales: Option<&[f64]>,
    settings: &LmSettings<f64>,
) -> std::result::Result<FitResult, FitError> {
    let columns = problem.config.mode().columns();
    let outcome = run_lm(columns, guess, scales, settings, |x, output| {
        problem.evaluate_into(x, output)
    })?;
    Ok(FitResult {
        parameters: outcome.parameters,
        iterations: outcome.iterations,
        rss: outcome.evaluation.rss(),
    })
}

/// Converged parameters with identifiable absolute measurement uncertainty.
#[derive(Debug)]
pub struct CovarianceFit {
    /// Fitted parameters in physical state and optional drag units.
    pub parameters: DynVector<f64>,
    /// Number of LM iterations before convergence.
    pub iterations: usize,
    /// Sum of squares of the final calibrated, whitened residuals.
    pub rss: f64,
    /// Physical parameter covariance, without RSS scaling or LM damping.
    pub covariance: numeris::DynMatrix<f64>,
    /// Numerical rank of the final whitened information.
    pub rank: usize,
    /// Statistical interpretation of the covariance matrix.
    pub covariance_method: CovarianceMethod,
}

/// Fit a prepared problem and obtain absolute physical parameter covariance.
///
/// Uses the same solver and defaults as [`fit_lm`]. Covariance is calculated from
/// its retained converged evaluation; no extra propagation pass is required.
/// See [`parameter_covariance`] for rank tolerance, units, and failure policy.
/// A converged but unobservable fit returns [`CovarianceFitError::Covariance`].
pub fn fit_lm_with_covariance(
    problem: &ODProblem<'_>,
    guess: &DynVector<f64>,
    scales: Option<&[f64]>,
    settings: &LmSettings<f64>,
) -> std::result::Result<CovarianceFit, CovarianceFitError> {
    let columns = problem.config.mode().columns();
    let outcome = run_lm(columns, guess, scales, settings, |x, output| {
        problem.evaluate_into(x, output)
    })?;
    let uncertainty = parameter_covariance(&outcome.evaluation.jacobian)?;
    Ok(CovarianceFit {
        parameters: outcome.parameters,
        iterations: outcome.iterations,
        rss: outcome.evaluation.rss(),
        covariance: uncertainty.matrix,
        rank: uncertainty.rank,
        covariance_method: uncertainty.method,
    })
}
