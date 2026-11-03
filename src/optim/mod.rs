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
    DynMatrix, DynVector,
    optim::{LmSettings, least_squares_lm_dyn},
};
use satkit::{Instant, orbitprop::PropSettings};
use std::{cell::RefCell, collections::BTreeMap, fmt, rc::Rc};

/// Immutable collection of batches with a prepared union of exact epochs.
#[derive(Clone, Debug)]
pub struct ODProblem {
    config: PropagationConfig,
    batches: Vec<PreparedBatch>,
    timestamps: Vec<Instant>,
    mappings: Vec<Vec<usize>>,
    rows: usize,
}

impl ODProblem {
    pub fn new(config: PropagationConfig, batches: Vec<PreparedBatch>) -> Result<Self> {
        if batches.is_empty() {
            return Err(KernelError::InvalidInput("empty observation problem"));
        }
        let mut epochs = BTreeMap::new();
        for batch in &batches {
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
    pub fn batches(&self) -> &[PreparedBatch] {
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

type CacheEntry = (Vec<f64>, Rc<Evaluation>);

/// Last-parameter cache; shared results avoid cloning the entire evaluation.
pub struct CachedProblem {
    inner: ODProblem,
    cache: RefCell<Option<CacheEntry>>,
}

impl CachedProblem {
    pub fn new(problem: ODProblem) -> Self {
        Self {
            inner: problem,
            cache: RefCell::new(None),
        }
    }
    pub fn inner(&self) -> &ODProblem {
        &self.inner
    }

    pub fn evaluate(&self, parameters: &DynVector<f64>) -> Result<Rc<Evaluation>> {
        if let Some((key, result)) = self.cache.borrow().as_ref()
            && key == parameters.as_slice()
        {
            return Ok(Rc::clone(result));
        }
        let result = Rc::new(self.inner.evaluate(parameters)?);
        *self.cache.borrow_mut() = Some((parameters.as_slice().to_vec(), Rc::clone(&result)));
        Ok(result)
    }
}

#[derive(Debug)]
pub enum FitError {
    Kernel(KernelError),
    Solver(String),
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
    evaluation: Rc<Evaluation>,
}

/// Shared LM driver: validates scales, maps dimensionless corrections to
/// parameters, rejects negative drag, and caches evaluations between callbacks.
fn run_lm(
    columns: usize,
    guess: &DynVector<f64>,
    scales: Option<&[f64]>,
    settings: &LmSettings<f64>,
    evaluate: impl Fn(&DynVector<f64>) -> Result<Rc<Evaluation>>,
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

fn run_lm_with_drag(
    columns: usize,
    guess: &DynVector<f64>,
    scales: Option<&[f64]>,
    settings: &LmSettings<f64>,
    drag_index: Option<usize>,
    evaluate: impl Fn(&DynVector<f64>) -> Result<Rc<Evaluation>>,
) -> std::result::Result<LmOutcome, FitError> {
    let defaults = [100.0, 100.0, 100.0, 0.1, 0.1, 0.1, 0.01];
    let scales = scales.unwrap_or_else(|| defaults.get(..columns).unwrap_or(&[]));
    if scales.len() != columns || !scales.iter().all(|s| s.is_finite() && *s > 0.0) {
        return Err(KernelError::InvalidInput(
            "parameter scales must be finite, positive and dimensionally correct",
        )
        .into());
    }
    let initial = evaluate(guess)?;
    let rows = initial.residual.len();
    let parameters = |z: &DynVector<f64>| {
        let mut x = guess.clone();
        for k in 0..columns {
            x[k] += scales[k] * z[k];
        }
        x
    };
    let failure = RefCell::new(None);
    let result = least_squares_lm_dyn(
        |z| {
            let x = parameters(z);
            if drag_index.is_some_and(|index| x[index] < 0.0) {
                return DynVector::fill(rows, f64::INFINITY);
            }
            match evaluate(&x) {
                Ok(e) => e.residual.clone(),
                Err(e) => {
                    *failure.borrow_mut() = Some(e);
                    DynVector::fill(rows, f64::INFINITY)
                }
            }
        },
        |z| match evaluate(&parameters(z)) {
            Ok(e) => {
                let mut j = e.jacobian.clone();
                for row in 0..rows {
                    for k in 0..columns {
                        j[(row, k)] *= scales[k];
                    }
                }
                j
            }
            Err(e) => {
                *failure.borrow_mut() = Some(e);
                DynMatrix::zeros(rows, columns)
            }
        },
        &DynVector::zeros(columns),
        settings,
    );
    if let Some(e) = failure.into_inner() {
        return Err(e.into());
    }
    let result = result.map_err(|e| FitError::Solver(format!("{e:?}")))?;
    let parameters = parameters(&result.x);
    let evaluation = evaluate(&parameters)?;
    Ok(LmOutcome {
        parameters,
        iterations: result.iterations,
        evaluation,
    })
}

/// Solve dimensionless corrections around `guess`. Defaults preserve the demo's
/// metre, m/s and drag scales. Kernel errors survive numeris's infallible callbacks.
pub fn fit_lm(
    problem: ODProblem,
    guess: &DynVector<f64>,
    scales: Option<&[f64]>,
    settings: &LmSettings<f64>,
) -> std::result::Result<FitResult, FitError> {
    let columns = problem.config.mode().columns();
    let cached = CachedProblem::new(problem);
    let outcome = run_lm(columns, guess, scales, settings, |x| cached.evaluate(x))?;
    Ok(FitResult {
        parameters: outcome.parameters,
        iterations: outcome.iterations,
        rss: outcome.evaluation.rss(),
    })
}
