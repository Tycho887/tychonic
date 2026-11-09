//! Thin stacking and LM integration over verified observation kernels.
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
mod lm;
use lm::{run_lm, run_lm_with_drag};
use satkit::{Instant, orbitprop::PropSettings};
use std::{collections::BTreeMap, fmt};

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
            rows,
        })
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
