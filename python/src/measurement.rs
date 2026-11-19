use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, PyUntypedArrayMethods};
use pyo3::{exceptions::PyValueError, prelude::*};
use serde::Deserialize;
use tychonic::{
    estimate::MeasurementModel,
    kernels::{GroundStation, Interferometer, PreparedBatch, PreparedGeometry, V3, V6},
    measurements::{self, Observable},
    numeris::{DynMatrix, DynVector},
};

use crate::{
    common::{instant, json, kernel_error},
    config::NativePropagation,
};

#[derive(Deserialize)]
struct ModelInput {
    kind: String,
    carrier_hz: Option<f64>,
    #[serde(default)]
    offset_hz: f64,
    baseline_enu_m: Option<[f64; 3]>,
    #[serde(default)]
    spherical_phase: bool,
    #[serde(default)]
    frequency_offset_hz: f64,
    #[serde(default)]
    phase_offset_rad: f64,
}

impl ModelInput {
    fn interferometer(&self) -> PyResult<Interferometer> {
        let carrier = self
            .carrier_hz
            .ok_or_else(|| PyValueError::new_err("carrier_hz is required"))?;
        let value = Interferometer::new(carrier).map_err(kernel_error)?;
        Ok(if self.spherical_phase {
            value.with_spherical_phase()
        } else {
            value
        })
    }

    fn baselines(&self, n: usize, column: Option<Vec<V3>>) -> PyResult<Vec<V3>> {
        if let Some(values) = column {
            return Ok(values);
        }
        let b = self
            .baseline_enu_m
            .ok_or_else(|| PyValueError::new_err("missing baseline"))?;
        Ok(vec![V3::from_array(b); n])
    }

    fn observable(&self, n: usize, column: Option<Vec<V3>>) -> PyResult<Observable<Vec<V3>>> {
        match self.kind.as_str() {
            "doppler" => Ok(Observable::Doppler {
                carrier_hz: self
                    .carrier_hz
                    .ok_or_else(|| PyValueError::new_err("carrier_hz is required"))?,
                offset_hz: self.offset_hz,
            }),
            "phase_difference" => Ok(Observable::PhaseDifference {
                interferometer: self.interferometer()?,
                baseline_enu_m: self.baselines(n, column)?,
                frequency_offset_hz: self.frequency_offset_hz,
                phase_offset_rad: self.phase_offset_rad,
            }),
            "differential_doppler" => Ok(Observable::DifferentialDoppler {
                interferometer: self.interferometer()?,
                baseline_enu_m: self.baselines(n, column)?,
            }),
            "azimuth_elevation" => Ok(Observable::AzimuthElevation),
            _ => Err(PyValueError::new_err("unknown observable")),
        }
    }
}

/// Own validated observations and geometry independently of the source arrays.
#[pyclass(frozen)]
pub struct NativeBatch {
    pub inner: PreparedBatch<'static>,
    tca_carrier: Option<f64>,
}

impl NativeBatch {
    pub fn borrow_batch(&self) -> PyResult<PreparedBatch<'_>> {
        Ok(self.inner.as_borrowed())
    }
}

#[pymethods]
impl NativeBatch {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature=(times,stations,models,values,covariance,phase_baselines=None,differential_baselines=None))]
    fn new(
        py: Python<'_>,
        times: &Bound<'_, PyAny>,
        stations: Vec<[f64; 3]>,
        models: &str,
        values: PyReadonlyArray2<'_, f64>,
        covariance: PyReadonlyArray3<'_, f64>,
        phase_baselines: Option<PyReadonlyArray2<'_, f64>>,
        differential_baselines: Option<PyReadonlyArray2<'_, f64>>,
    ) -> PyResult<Self> {
        let times = if let Ok(values) = times.extract::<PyReadonlyArray1<'_, f64>>() {
            values
                .as_array()
                .iter()
                .map(|s| {
                    if !s.is_finite() || s.abs() > 2.5e11 {
                        return Err(PyValueError::new_err("Unix epoch out of range"));
                    }
                    Ok(tychonic::satkit::Instant::from_unixtime_microseconds(
                        (s * 1e6).round() as i64,
                    ))
                })
                .collect::<PyResult<Vec<_>>>()?
        } else {
            times
                .extract::<Vec<String>>()?
                .iter()
                .map(|time| instant(time))
                .collect::<PyResult<Vec<_>>>()?
        };
        let mut stations: Vec<_> = stations
            .into_iter()
            .map(|s| GroundStation {
                lat_deg: s[0],
                lon_deg: s[1],
                alt_m: s[2],
            })
            .collect();
        if stations.len() == 1 {
            stations.resize(times.len(), stations[0]);
        }
        let inputs: Vec<ModelInput> = json(models)?;
        let mut phase = phase_baselines.map(baseline_column).transpose()?;
        let mut differential = differential_baselines.map(baseline_column).transpose()?;
        let models = inputs
            .iter()
            .map(|model| {
                model.observable(
                    times.len(),
                    match model.kind.as_str() {
                        "phase_difference" => phase.take(),
                        "differential_doppler" => differential.take(),
                        _ => None,
                    },
                )
            })
            .collect::<PyResult<Vec<_>>>()?;
        let values: Vec<Vec<f64>> = values
            .as_array()
            .rows()
            .into_iter()
            .map(|row| row.to_vec())
            .collect();
        let shape = covariance.shape();
        if shape[0] != times.len() || shape[1] != shape[2] {
            return Err(PyValueError::new_err("invalid covariance dimensions"));
        }
        let view = covariance.as_array();
        let covariance: Vec<_> = (0..shape[0])
            .map(|k| DynMatrix::from_fn(shape[1], shape[2], |i, j| view[[k, i, j]]))
            .collect();
        py.detach(move || {
            let geometry = PreparedGeometry::new(&times, &stations).map_err(kernel_error)?;
            let tca_carrier = match models.as_slice() {
                [
                    Observable::Doppler { carrier_hz, .. },
                    Observable::AzimuthElevation,
                ] => Some(*carrier_hz),
                _ => None,
            };
            let inner =
                measurements::joint(geometry, models, values, covariance).map_err(kernel_error)?;
            Ok(Self { inner, tca_carrier })
        })
    }

    fn predict(
        &self,
        py: Python<'_>,
        propagation: &NativePropagation,
        x: Vec<f64>,
    ) -> PyResult<Vec<Vec<f64>>> {
        let config = propagation.build()?;
        py.detach(|| {
            let samples = config
                .sample(&DynVector::from_slice(&x), self.inner.timestamps())
                .map_err(kernel_error)?;
            samples
                .states()
                .iter()
                .enumerate()
                .map(|(row, state)| predicted_row(&self.inner, row, state))
                .collect()
        })
    }
}

fn baseline_column(data: PyReadonlyArray2<'_, f64>) -> PyResult<Vec<V3>> {
    if data.shape()[1] != 3 {
        return Err(PyValueError::new_err(
            "baseline columns require three ENU components",
        ));
    }
    Ok(data
        .as_array()
        .rows()
        .into_iter()
        .map(|r| V3::from_array([r[0], r[1], r[2]]))
        .collect())
}

fn predicted_row(batch: &PreparedBatch<'_>, row: usize, state: &V6) -> PyResult<Vec<f64>> {
    match batch.components() {
        1 => predict::<1>(batch, row, state),
        2 => predict::<2>(batch, row, state),
        3 => predict::<3>(batch, row, state),
        4 => predict::<4>(batch, row, state),
        5 => predict::<5>(batch, row, state),
        _ => Err(PyValueError::new_err("unsupported measurement dimension")),
    }
}

fn predict<const M: usize>(
    batch: &PreparedBatch<'_>,
    row: usize,
    state: &V6,
) -> PyResult<Vec<f64>> {
    let observation = batch.joint_observation::<M>(row).map_err(kernel_error)?;
    let (value, _) = observation.model.evaluate(state).map_err(kernel_error)?;
    Ok(value.as_slice().to_vec())
}

pub(crate) type Receipt = (usize, usize, String, f64, bool);
pub(crate) type Receipts = (Vec<Receipt>, Option<PyErr>);

/// Commit in stable time order and return receipts even when a later row fails.
pub(crate) fn ordered_updates(
    batches: &[&NativeBatch],
    mut observe: impl FnMut(&NativeBatch, usize) -> PyResult<(f64, bool)>,
) -> Receipts {
    let mut schedule: Vec<_> = batches
        .iter()
        .enumerate()
        .flat_map(|(index, b)| {
            b.inner
                .timestamps()
                .iter()
                .enumerate()
                .map(move |(row, time)| (*time, index, row))
        })
        .collect();
    schedule.sort_unstable();
    let mut updates = Vec::with_capacity(schedule.len());
    for (time, index, row) in schedule {
        match observe(batches[index], row) {
            Ok((nis, accepted)) => updates.push((index, row, time.as_rfc3339(), nis, accepted)),
            Err(error) => return (updates, Some(error)),
        }
    }
    (updates, None)
}
impl NativeBatch {
    pub(crate) fn tca_carrier(&self) -> PyResult<f64> {
        self.tca_carrier.ok_or_else(|| {
            PyValueError::new_err("TCA updates require simultaneous Doppler, azimuth and elevation")
        })
    }
}
