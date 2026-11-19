use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::{
    create_exception,
    exceptions::{PyRuntimeError, PyTypeError, PyValueError},
    prelude::*,
};
use tychonic::{
    detection::AcquisitionError,
    estimate::EkfError,
    kernels::KernelError,
    numeris::{Matrix, Vector},
    optim::FitError,
    satkit::{Instant, TLE, omm::OMM},
};

create_exception!(_native, TychonicError, PyRuntimeError);
create_exception!(_native, NumericalError, TychonicError);
create_exception!(_native, SolverError, TychonicError);
create_exception!(_native, DataError, TychonicError);
create_exception!(_native, AcquisitionStateError, TychonicError);

create_exception!(_native, RankFailure, NumericalError);
create_exception!(_native, CovarianceFailure, NumericalError);

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = module.py();
    module.add("RankFailure", py.get_type::<RankFailure>())?;
    module.add("CovarianceFailure", py.get_type::<CovarianceFailure>())?;
    module.add("TychonicError", py.get_type::<TychonicError>())?;
    module.add("NumericalError", py.get_type::<NumericalError>())?;
    module.add("SolverError", py.get_type::<SolverError>())?;
    module.add("DataError", py.get_type::<DataError>())?;
    module.add(
        "AcquisitionStateError",
        py.get_type::<AcquisitionStateError>(),
    )?;
    Ok(())
}

pub fn kernel_error(error: KernelError) -> PyErr {
    match error {
        KernelError::InvalidInput(message) => PyValueError::new_err(message),
        KernelError::Propagation(error) => propagation_error(error),
        other => {
            let message = other.to_string();
            if is_data_error(&message) {
                DataError::new_err(format!(
                    "{message}. Provision Satkit data and set SATKIT_DATA; see docs/python.md"
                ))
            } else {
                NumericalError::new_err(message)
            }
        }
    }
}

fn data_error(message: impl std::fmt::Display) -> PyErr {
    DataError::new_err(format!(
        "{message}. Provision Satkit data and set SATKIT_DATA; see docs/python.md"
    ))
}

fn propagation_error(error: tychonic::satkit::orbitprop::Error) -> PyErr {
    use tychonic::satkit::orbitprop::Error;
    match error {
        Error::InvalidGravityDegree { .. }
        | Error::InvalidGravityOrder { .. }
        | Error::InvalidStateColumns { .. }
        | Error::InvalidPrecomputeStep { .. } => PyValueError::new_err(error.to_string()),
        Error::Jplephem(_)
        | Error::EopCoverage { .. }
        | Error::FrameTransform(_)
        | Error::Gravity(_) => data_error(error),
        other => NumericalError::new_err(other.to_string()),
    }
}

fn is_data_error(message: &str) -> bool {
    let lower = message.to_lowercase();
    [
        "datafile",
        "data file",
        "not found",
        "no such file",
        "eop coverage",
        "missing data",
    ]
    .iter()
    .any(|needle| lower.contains(needle))
}

pub fn fit_error(error: FitError) -> PyErr {
    match error {
        FitError::Kernel(error) => kernel_error(error),
        FitError::Solver(error) => SolverError::new_err(error.to_string()),
    }
}

pub fn ekf_error(error: EkfError) -> PyErr {
    match error {
        EkfError::Kernel(error) => kernel_error(error),
        EkfError::Estimate(error) => NumericalError::new_err(error.to_string()),
    }
}

pub fn acquisition_error(error: AcquisitionError) -> PyErr {
    match error {
        AcquisitionError::InvalidInput(message) => PyValueError::new_err(message),
        AcquisitionError::InvalidState(message) => AcquisitionStateError::new_err(message),
        AcquisitionError::Kernel(error) => kernel_error(error),
        other => NumericalError::new_err(other.to_string()),
    }
}

pub fn instant(value: &str) -> PyResult<Instant> {
    Instant::from_rfc3339(value).map_err(|error| PyValueError::new_err(error.to_string()))
}

#[pyfunction]
pub fn roundtrip_time(value: &str) -> PyResult<String> {
    Ok(instant(value)?.as_rfc3339())
}

pub fn vector<const N: usize>(value: &PyReadonlyArray1<'_, f64>) -> PyResult<Vector<f64, N>> {
    if value.len() != N {
        return Err(PyValueError::new_err(format!(
            "expected state shape ({N},)"
        )));
    }
    let owned: Vec<_> = value.as_array().iter().copied().collect();
    Ok(Vector::from_slice(&owned))
}

pub fn matrix<const N: usize>(value: &PyReadonlyArray2<'_, f64>) -> PyResult<Matrix<f64, N, N>> {
    if value.shape() != [N, N] {
        return Err(PyValueError::new_err(format!(
            "expected covariance shape ({N}, {N})"
        )));
    }
    let view = value.as_array();
    Ok(Matrix::from_fn(|i, j| view[[i, j]]))
}

pub fn matrix_rows<const N: usize>(matrix: &Matrix<f64, N, N>) -> Vec<Vec<f64>> {
    (0..N)
        .map(|i| (0..N).map(|j| matrix[(i, j)]).collect())
        .collect()
}

pub fn json<T: serde::de::DeserializeOwned>(value: &str) -> PyResult<T> {
    serde_json::from_str(value).map_err(|error| PyValueError::new_err(error.to_string()))
}

pub fn tle(value: &str) -> PyResult<TLE> {
    let lines: Vec<_> = value
        .lines()
        .filter(|line| !line.trim().is_empty())
        .collect();
    let lines = match lines.as_slice() {
        [first, second] => (*first, *second),
        [_, first, second] => (*first, *second),
        _ => {
            return Err(PyValueError::new_err(
                "expected two TLE lines, optionally preceded by a name",
            ));
        }
    };
    TLE::load_2line(lines.0, lines.1).map_err(|error| PyValueError::new_err(error.to_string()))
}

pub fn omm(value: &str) -> PyResult<OMM> {
    let mut records =
        OMM::from_json_string(value).map_err(|error| PyValueError::new_err(error.to_string()))?;
    if records.len() != 1 {
        return Err(PyValueError::new_err("expected one OMM record"));
    }
    records
        .pop()
        .ok_or_else(|| PyValueError::new_err("empty OMM record"))
}

pub fn callable(value: &Bound<'_, PyAny>) -> PyResult<()> {
    if !value.is_callable() {
        return Err(PyTypeError::new_err("expected a callable"));
    }
    Ok(())
}

#[pyfunction]
pub fn elapsed(previous: &str, next: &str) -> PyResult<f64> {
    Ok((instant(next)? - instant(previous)?).as_seconds())
}
