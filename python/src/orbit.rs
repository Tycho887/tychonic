use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::{exceptions::PyValueError, prelude::*};
use serde::Deserialize;
use tychonic::{
    estimate::{OrbitEkf, UpdateOptions},
    kernels::{KernelError, PreparedBatch},
    numeris::{DynVector, Matrix, Vector, optim::LmSettings},
    satkit::Instant,
};

use crate::{
    common::{
        callable, ekf_error, fit_error, instant, json, kernel_error, matrix, matrix_rows, vector,
    },
    config::NativePropagation,
    measurement::NativeBatch,
};

type Snapshot = (Vec<f64>, Vec<Vec<f64>>, String);
type History = (Vec<String>, Vec<Vec<f64>>, Vec<Vec<Vec<f64>>>);
type ReadingUpdate = (Vec<f64>, f64, bool);
type BatchUpdate = (String, usize, usize, f64, bool);

#[derive(Deserialize)]
struct LmInput {
    max_iter: usize,
    grad_tol: f64,
    x_tol: f64,
    f_tol: f64,
    mu_init: f64,
    mu_min: f64,
    mu_max: f64,
    mu_increase: f64,
    mu_decrease: f64,
    parameter_scales: Option<Vec<f64>>,
}

#[pyfunction]
pub fn fit_orbit(
    py: Python<'_>,
    batches: Vec<PyRef<'_, NativeBatch>>,
    x: Vec<f64>,
    propagation: &NativePropagation,
    options: &str,
) -> PyResult<(Vec<f64>, f64, usize)> {
    let config = propagation.build()?;
    let options: LmInput = json(options)?;
    let settings = LmSettings {
        max_iter: options.max_iter,
        grad_tol: options.grad_tol,
        x_tol: options.x_tol,
        f_tol: options.f_tol,
        mu_init: options.mu_init,
        mu_min: options.mu_min,
        mu_max: options.mu_max,
        mu_increase: options.mu_increase,
        mu_decrease: options.mu_decrease,
    };
    let references: Vec<&NativeBatch> = batches.iter().map(|batch| &**batch).collect();
    py.detach(move || {
        let batches = references
            .into_iter()
            .map(NativeBatch::borrow_batch)
            .collect::<PyResult<Vec<_>>>()?;
        let fit = tychonic::fit_with_options(
            config,
            &batches,
            &DynVector::from_slice(&x),
            options.parameter_scales.as_deref(),
            &settings,
        )
        .map_err(fit_error)?;
        Ok((fit.parameters.as_slice().to_vec(), fit.rss, fit.iterations))
    })
}

enum Filter {
    State(Box<OrbitEkf<6>>),
    Drag(Box<OrbitEkf<7>>),
}

macro_rules! with_filter {
    ($value:expr, $filter:ident, $body:expr) => {
        match $value {
            Filter::State($filter) => $body,
            Filter::Drag($filter) => $body,
        }
    };
}

#[pyclass]
pub struct NativeOrbitEKF {
    inner: Filter,
}

#[pymethods]
impl NativeOrbitEKF {
    #[new]
    #[pyo3(signature=(x,covariance,propagation,bias_hz=false))]
    fn new(
        py: Python<'_>,
        x: PyReadonlyArray1<'_, f64>,
        covariance: PyReadonlyArray2<'_, f64>,
        propagation: &NativePropagation,
        bias_hz: bool,
    ) -> PyResult<Self> {
        let config = propagation.build()?;
        let inner = if bias_hz {
            let (x, p) = (vector::<7>(&x)?, matrix::<7>(&covariance)?);
            Filter::Drag(Box::new(
                py.detach(|| OrbitEkf::with_doppler_bias(config, x, p))
                    .map_err(ekf_error)?,
            ))
        } else {
            match config.mode().columns() {
                6 => {
                    let (x, p) = (vector::<6>(&x)?, matrix::<6>(&covariance)?);
                    Filter::State(Box::new(
                        py.detach(|| OrbitEkf::new(config, x, p))
                            .map_err(ekf_error)?,
                    ))
                }
                7 => {
                    let (x, p) = (vector::<7>(&x)?, matrix::<7>(&covariance)?);
                    Filter::Drag(Box::new(
                        py.detach(|| OrbitEkf::new(config, x, p))
                            .map_err(ekf_error)?,
                    ))
                }
                _ => return Err(PyValueError::new_err("unsupported orbit state dimension")),
            }
        };
        Ok(Self { inner })
    }

    fn snapshot(&self) -> Snapshot {
        with_filter!(
            &self.inner,
            filter,
            (
                filter.state().as_slice().to_vec(),
                matrix_rows(filter.covariance()),
                filter.epoch().as_rfc3339()
            )
        )
    }

    fn discard_history(&mut self) {
        with_filter!(&mut self.inner, filter, filter.discard_history());
    }

    fn update_bias(
        &mut self,
        py: Python<'_>,
        batches: Vec<PyRef<'_, NativeBatch>>,
        acceleration: f64,
        bias: f64,
        retain_history: bool,
        gate: Option<f64>,
    ) -> PyResult<crate::measurement::Receipts> {
        let Filter::Drag(filter) = &mut self.inner else {
            return Err(PyValueError::new_err("requires augmented state"));
        };
        let batches: Vec<_> = batches.iter().map(|b| &**b).collect();
        Ok(py.detach(|| {
            crate::measurement::ordered_updates(&batches, |b, row| {
                let dt = (b.inner.timestamps()[row] - filter.epoch()).as_seconds();
                let q = tychonic::estimate::white_acceleration_noise::<7>(
                    dt,
                    &Vector::from_array([acceleration; 3]),
                    bias,
                )
                .map_err(kernel_error)?;
                let (_, nis, accepted) = observe_row(filter, &b.inner, row, &q, gate)?;
                if !retain_history {
                    filter.discard_history();
                }
                Ok((nis, accepted))
            })
        }))
    }

    fn saved_steps(&self) -> Vec<SavedStep> {
        with_filter!(
            &self.inner,
            filter,
            filter
                .timestamps()
                .iter()
                .zip(filter.steps().iter())
                .map(|(time, step)| (
                    time.as_rfc3339(),
                    step.x_predicted.as_slice().to_vec(),
                    step.p_predicted.as_slice().to_vec(),
                    step.x_updated.as_slice().to_vec(),
                    step.p_updated.as_slice().to_vec(),
                    step.f_jacobian.as_slice().to_vec()
                ))
                .collect()
        )
    }

    fn restore_steps(&mut self, steps: Vec<SavedStep>) -> PyResult<()> {
        with_filter!(&mut self.inner, filter, load_steps(filter, steps))
    }

    fn history(&self) -> History {
        with_filter!(
            &self.inner,
            filter,
            (
                filter
                    .timestamps()
                    .iter()
                    .map(Instant::as_rfc3339)
                    .collect(),
                filter
                    .steps()
                    .iter()
                    .map(|step| step.x_updated.as_slice().to_vec())
                    .collect(),
                filter
                    .steps()
                    .iter()
                    .map(|step| matrix_rows(&step.p_updated))
                    .collect()
            )
        )
    }

    fn smooth(&self, py: Python<'_>) -> PyResult<History> {
        py.detach(|| with_filter!(&self.inner, filter, smoothed(filter)))
    }

    fn predict_to(
        &mut self,
        py: Python<'_>,
        time: &str,
        q: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<()> {
        let time = instant(time)?;
        match &mut self.inner {
            Filter::State(filter) => {
                let q = matrix(&q)?;
                py.detach(|| filter.predict_to(time, Some(&q)))
                    .map_err(ekf_error)
            }
            Filter::Drag(filter) => {
                let q = matrix(&q)?;
                py.detach(|| filter.predict_to(time, Some(&q)))
                    .map_err(ekf_error)
            }
        }
    }

    fn observe(
        &mut self,
        py: Python<'_>,
        batch: &NativeBatch,
        row: usize,
        q: PyReadonlyArray2<'_, f64>,
        gate: Option<f64>,
    ) -> PyResult<ReadingUpdate> {
        match &mut self.inner {
            Filter::State(filter) => {
                let q = matrix(&q)?;
                py.detach(|| observe_row(filter, &batch.inner, row, &q, gate))
            }
            Filter::Drag(filter) => {
                let q = matrix(&q)?;
                py.detach(|| observe_row(filter, &batch.inner, row, &q, gate))
            }
        }
    }

    fn filter_batches(
        &mut self,
        py: Python<'_>,
        batches: Vec<PyRef<'_, NativeBatch>>,
        process_noise: Py<PyAny>,
        gate: Option<f64>,
    ) -> PyResult<Vec<BatchUpdate>> {
        callable(process_noise.bind(py))?;
        let references: Vec<&NativeBatch> = batches.iter().map(|batch| &**batch).collect();
        py.detach(|| {
            let batches = references
                .into_iter()
                .map(NativeBatch::borrow_batch)
                .collect::<PyResult<Vec<_>>>()?;
            with_filter!(
                &mut self.inner,
                filter,
                filter_batches(filter, &batches, &process_noise, gate)
            )
        })
    }
}

fn smoothed<const N: usize>(filter: &OrbitEkf<N>) -> PyResult<History> {
    let records = filter.smooth().map_err(ekf_error)?;
    Ok((
        filter
            .timestamps()
            .iter()
            .map(Instant::as_rfc3339)
            .collect(),
        records.iter().map(|(x, _)| x.as_slice().to_vec()).collect(),
        records.iter().map(|(_, p)| matrix_rows(p)).collect(),
    ))
}

fn observe_row<const N: usize>(
    filter: &mut OrbitEkf<N>,
    batch: &PreparedBatch<'_>,
    row: usize,
    q: &Matrix<f64, N, N>,
    gate: Option<f64>,
) -> PyResult<ReadingUpdate> {
    match batch.components() {
        1 => observe::<N, 1>(filter, batch, row, q, gate),
        2 => observe::<N, 2>(filter, batch, row, q, gate),
        3 => observe::<N, 3>(filter, batch, row, q, gate),
        4 => observe::<N, 4>(filter, batch, row, q, gate),
        5 => observe::<N, 5>(filter, batch, row, q, gate),
        _ => Err(PyValueError::new_err("unsupported measurement dimension")),
    }
}

fn observe<const N: usize, const M: usize>(
    filter: &mut OrbitEkf<N>,
    batch: &PreparedBatch<'_>,
    row: usize,
    q: &Matrix<f64, N, N>,
    gate: Option<f64>,
) -> PyResult<ReadingUpdate> {
    let observation = batch.joint_observation::<M>(row).map_err(kernel_error)?;
    let result = filter
        .observe(&observation, Some(q), UpdateOptions { gate })
        .map_err(ekf_error)?;
    Ok((
        result.innovation.as_slice().to_vec(),
        result.nis,
        result.accepted,
    ))
}

fn noise_callback<const N: usize>(
    callback: &Py<PyAny>,
    x: &Vector<f64, N>,
    previous: Instant,
    next: Instant,
) -> PyResult<Matrix<f64, N, N>> {
    Python::attach(|py| {
        let state = x.as_slice().to_vec().into_pyarray(py);
        let result = callback.call1(py, (state, previous.as_rfc3339(), next.as_rfc3339()))?;
        let array = result.bind(py).extract::<PyReadonlyArray2<'_, f64>>()?;
        matrix(&array)
    })
}

fn filter_batches<const N: usize>(
    filter: &mut OrbitEkf<N>,
    batches: &[PreparedBatch<'_>],
    callback: &Py<PyAny>,
    gate: Option<f64>,
) -> PyResult<Vec<BatchUpdate>> {
    let mut callback_error = None;
    let result = filter.filter_batches(
        batches,
        |x, previous, next| {
            noise_callback(callback, x, previous, next).map_err(|error| {
                callback_error = Some(error);
                KernelError::InvalidInput("Python process-noise callback failed")
            })
        },
        UpdateOptions { gate },
    );
    if let Some(error) = callback_error {
        return Err(error);
    }
    Ok(result
        .map_err(ekf_error)?
        .into_iter()
        .map(|update| {
            (
                update.time.as_rfc3339(),
                update.batch,
                update.row,
                update.nis,
                update.accepted,
            )
        })
        .collect())
}

#[pyfunction]
pub fn white_acceleration_noise(
    seconds: f64,
    acceleration_psd: [f64; 3],
    drag_psd: f64,
    estimate_drag: bool,
) -> PyResult<Vec<Vec<f64>>> {
    let acceleration = Vector::from_array(acceleration_psd);
    if estimate_drag {
        tychonic::estimate::white_acceleration_noise::<7>(seconds, &acceleration, drag_psd)
            .map(|q| matrix_rows(&q))
            .map_err(kernel_error)
    } else {
        tychonic::estimate::white_acceleration_noise::<6>(seconds, &acceleration, drag_psd)
            .map(|q| matrix_rows(&q))
            .map_err(kernel_error)
    }
}

type SavedStep = (String, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>);
fn load_steps<const N: usize>(filter: &mut OrbitEkf<N>, steps: Vec<SavedStep>) -> PyResult<()> {
    let records = steps
        .into_iter()
        .map(|(time, xp, pp, x, p, f)| {
            if xp.len() != N || x.len() != N || [pp.len(), p.len(), f.len()] != [N * N; 3] {
                return Err(PyValueError::new_err("checkpoint dimensions"));
            }
            let mat = |v: &[f64]| Matrix::from_fn(|i, j| v[j * N + i]);
            Ok((
                instant(&time)?,
                tychonic::numeris::estimate::EkfStep {
                    x_predicted: Vector::from_slice(&xp),
                    p_predicted: mat(&pp),
                    x_updated: Vector::from_slice(&x),
                    p_updated: mat(&p),
                    f_jacobian: mat(&f),
                },
            ))
        })
        .collect::<PyResult<Vec<_>>>()?;
    filter.restore_history(records).map_err(ekf_error)
}
