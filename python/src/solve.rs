//! Calibrated batch fitting and deterministic ephemeris export.
use crate::{common::*, config::NativePropagation, measurement::NativeBatch};
use pyo3::{exceptions::PyValueError, prelude::*};
use tychonic::{
    kernels::initial_state_from_tle,
    numeris::{DynMatrix, DynVector, optim::LmSettings},
    optim::{self, CovarianceFitError, ODProblem},
    satkit::{Instant, TLE, omm::OMM},
};

pub(crate) type Rows = Vec<Vec<f64>>;
pub(crate) fn rows(m: &DynMatrix<f64>) -> Rows {
    (0..m.nrows())
        .map(|i| (0..m.ncols()).map(|j| m[(i, j)]).collect())
        .collect()
}

fn covariance_error(error: CovarianceFitError) -> PyErr {
    match error {
        CovarianceFitError::Fit(e) => fit_error(e),
        CovarianceFitError::Covariance(optim::CovarianceError::RankDeficient {
            rank,
            parameters,
        }) => RankFailure::new_err((rank, parameters)),
        CovarianceFitError::Covariance(e) => CovarianceFailure::new_err(e.to_string()),
    }
}

type Fit = (
    Vec<f64>,
    Option<Rows>,
    f64,
    usize,
    Option<usize>,
    Option<String>,
    String,
);
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature=(batches, config, guess, covariance, options, prior=None))]
pub fn calibrated(
    py: Python<'_>,
    batches: Vec<PyRef<'_, NativeBatch>>,
    config: &NativePropagation,
    guess: Vec<f64>,
    covariance: bool,
    options: (Option<usize>, Option<f64>, Option<f64>, Option<f64>),
    prior: Option<(Vec<f64>, Rows)>,
) -> PyResult<Fit> {
    let config = config.build()?;
    let batches: Vec<_> = batches.iter().map(|b| &**b).collect();
    py.detach(move || {
        let batches = batches
            .into_iter()
            .map(NativeBatch::borrow_batch)
            .collect::<PyResult<Vec<_>>>()?;
        let mut problem = ODProblem::new(config, batches).map_err(kernel_error)?;
        if let Some((mean, p)) = prior {
            let n = mean.len();
            if p.len() != n || p.iter().any(|r| r.len() != n) {
                return Err(PyValueError::new_err("prior covariance dimensions"));
            }
            problem = problem
                .with_prior(
                    DynVector::from_vec(mean),
                    &DynMatrix::from_fn(n, n, |i, j| p[i][j]),
                )
                .map_err(kernel_error)?;
        }
        let settings = LmSettings::default();
        let settings = LmSettings {
            max_iter: options.0.unwrap_or(settings.max_iter),
            grad_tol: options.1.unwrap_or(settings.grad_tol),
            x_tol: options.2.unwrap_or(settings.x_tol),
            f_tol: options.3.unwrap_or(settings.f_tol),
            ..settings
        };
        let guess = DynVector::from_vec(guess);
        let epoch = problem
            .timestamps()
            .last()
            .ok_or_else(|| PyValueError::new_err("empty observations"))?
            .as_rfc3339();
        if covariance {
            let fit = optim::fit_lm_with_covariance(&problem, &guess, None, &settings)
                .map_err(covariance_error)?;
            Ok((
                fit.parameters.as_slice().to_vec(),
                Some(rows(&fit.covariance)),
                fit.rss,
                fit.iterations,
                Some(fit.rank),
                Some(if problem.uses_weighted_prior() {
                    "gaussian_posterior".to_owned()
                } else {
                    fit.covariance_method.to_string()
                }),
                epoch,
            ))
        } else {
            let fit = optim::fit_lm(&problem, &guess, None, &settings).map_err(fit_error)?;
            Ok((
                fit.parameters.as_slice().to_vec(),
                None,
                fit.rss,
                fit.iterations,
                None,
                None,
                epoch,
            ))
        }
    })
}

#[pyfunction]
pub fn tle_seed(py: Python<'_>, source: &str, at: &str) -> PyResult<(Vec<f64>, f64)> {
    let source = tle(source)?;
    let at = instant(at)?;
    py.detach(|| {
        Ok((
            initial_state_from_tle(&source, &at)
                .map_err(kernel_error)?
                .as_slice()
                .to_vec(),
            (at - source.epoch).as_days(),
        ))
    })
}

#[pyfunction]
#[pyo3(signature=(config,x,at,covariance=None))]
pub fn transport(
    py: Python<'_>,
    config: &NativePropagation,
    x: Vec<f64>,
    at: &str,
    covariance: Option<Rows>,
) -> PyResult<(Vec<f64>, Option<Rows>)> {
    let time = instant(at)?;
    let n = x.len();
    let p = covariance
        .map(|p| {
            if p.len() != n || p.iter().any(|r| r.len() != n) {
                return Err(PyValueError::new_err("transport covariance dimensions"));
            }
            Ok(DynMatrix::from_fn(n, n, |i, j| p[i][j]))
        })
        .transpose()?;
    py.detach(|| {
        let (state, covariance) = config
            .inner
            .transport(&DynVector::from_vec(x), p.as_ref(), time)
            .map_err(kernel_error)?;
        Ok((state.as_slice().to_vec(), covariance.as_ref().map(rows)))
    })
}

fn times(labels: Vec<String>) -> PyResult<Vec<Instant>> {
    labels.iter().map(|s| instant(s)).collect()
}
#[pyfunction]
pub fn arc(
    py: Python<'_>,
    config: &NativePropagation,
    x: Vec<f64>,
    epochs: Vec<String>,
) -> PyResult<Vec<[f64; 6]>> {
    let epochs = times(epochs)?;
    py.detach(|| {
        let sample = config
            .inner
            .sample(&DynVector::from_vec(x), &epochs)
            .map_err(kernel_error)?;
        Ok(sample
            .states()
            .iter()
            .map(|v| std::array::from_fn(|i| v[i]))
            .collect())
    })
}
pub(crate) fn elements(tle: &TLE) -> PyResult<([String; 2], String)> {
    let lines = tle
        .to_2line()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let omm = serde_json::to_string(&OMM::from_tle(tle))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((lines, omm))
}
#[pyfunction]
pub fn mean_elements(
    py: Python<'_>,
    config: &NativePropagation,
    x: Vec<f64>,
    epochs: Vec<String>,
) -> PyResult<([String; 2], String)> {
    let epochs = times(epochs)?;
    py.detach(|| {
        let sample = config
            .inner
            .sample(&DynVector::from_vec(x), &epochs)
            .map_err(kernel_error)?;
        let states: Vec<_> = sample
            .states()
            .iter()
            .map(|v| std::array::from_fn(|i| v[i]))
            .collect();
        let (tle, result) = TLE::fit_from_states(&states, &epochs, config.inner.reference_epoch())
            .map_err(|e| SolverError::new_err(e.to_string()))?;
        if !result.best_norm.is_finite()
            || matches!(
                result.status,
                tychonic::satkit::tle::TleFitStatus::MaxIterations
                    | tychonic::satkit::tle::TleFitStatus::DampingSaturated
            )
        {
            return Err(SolverError::new_err(format!(
                "mean-element fit failed: {} after {} iterations (residual norm {:.3e} m)",
                result.status, result.n_iter, result.best_norm
            )));
        }
        elements(&tle)
    })
}
