use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::{exceptions::PyValueError, prelude::*};
use tychonic::{
    estimate::{TcaUkf, UpdateOptions},
    kernels::{GroundStation, Measurement, MeasurementNoise},
    satkit::omm::OMM,
};

use crate::common::{instant, kernel_error, matrix, matrix_rows, tle, vector};

#[pyclass]
pub struct NativeTcaUKF {
    inner: TcaUkf,
}

#[pymethods]
impl NativeTcaUKF {
    #[new]
    #[pyo3(signature=(prior,epoch,x,covariance,doppler_hz=false))]
    fn new(
        py: Python<'_>,
        prior: &str,
        epoch: &str,
        x: PyReadonlyArray1<'_, f64>,
        covariance: PyReadonlyArray2<'_, f64>,
        doppler_hz: bool,
    ) -> PyResult<Self> {
        let (tle, epoch, x, covariance) = (
            tle(prior)?,
            instant(epoch)?,
            vector(&x)?,
            matrix(&covariance)?,
        );
        let make = if doppler_hz {
            TcaUkf::new_doppler
        } else {
            TcaUkf::new
        };
        py.detach(|| make(tle, epoch, x, covariance))
            .map(|inner| Self { inner })
            .map_err(kernel_error)
    }

    fn snapshot(&self) -> (Vec<f64>, Vec<Vec<f64>>, String) {
        (
            self.inner.state().as_slice().to_vec(),
            matrix_rows(self.inner.covariance()),
            self.inner.epoch().as_rfc3339(),
        )
    }

    fn corrected_elements(&self) -> PyResult<([String; 2], String)> {
        crate::solve::elements(&self.inner.corrected_tle().map_err(kernel_error)?)
    }
    fn update_hz(
        &mut self,
        py: Python<'_>,
        batches: Vec<PyRef<'_, crate::measurement::NativeBatch>>,
        noise: [f64; 4],
        gate: Option<f64>,
    ) -> crate::measurement::Receipts {
        let batches: Vec<_> = batches.iter().map(|b| &**b).collect();
        py.detach(|| {
            crate::measurement::ordered_updates(&batches, |batch, row| {
                let carrier = batch.tca_carrier()?;
                let obs = batch
                    .inner
                    .joint_observation::<3>(row)
                    .map_err(kernel_error)?;
                let station = batch
                    .inner
                    .geometry()
                    .row(row)
                    .ok_or_else(|| PyValueError::new_err("invalid row"))?;
                let q = bounded_noise(&self.inner, obs.time, noise)?;
                let u = self
                    .inner
                    .observe_doppler(
                        obs.time,
                        station,
                        obs.value,
                        carrier,
                        &obs.covariance,
                        Some(&q),
                        UpdateOptions { gate },
                    )
                    .map_err(kernel_error)?;
                Ok((u.nis, u.accepted))
            })
        })
    }

    fn predict_bounded(&mut self, time: &str, noise: [f64; 4]) -> PyResult<()> {
        let time = instant(time)?;
        let q = bounded_noise(&self.inner, time, noise)?;
        self.inner.predict_to(time, Some(&q)).map_err(kernel_error)
    }

    fn corrected_omm(&self) -> PyResult<String> {
        let corrected = self.inner.corrected_tle().map_err(kernel_error)?;
        serde_json::to_string(&OMM::from_tle(&corrected))
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    fn predict_to(
        &mut self,
        py: Python<'_>,
        time: &str,
        q: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<()> {
        let (time, q) = (instant(time)?, matrix(&q)?);
        py.detach(|| self.inner.predict_to(time, Some(&q)))
            .map_err(kernel_error)
    }

    #[allow(clippy::too_many_arguments)] // Mirrors one combined reading plus update settings.
    fn observe(
        &mut self,
        py: Python<'_>,
        time: &str,
        station: [f64; 3],
        values: [f64; 3],
        sigmas: [f64; 3],
        q: PyReadonlyArray2<'_, f64>,
        gate: Option<f64>,
    ) -> PyResult<(Vec<f64>, f64, bool)> {
        let measurement = Measurement {
            time: instant(time)?,
            station: GroundStation {
                lat_deg: station[0],
                lon_deg: station[1],
                alt_m: station[2],
            },
            range_rate: values[0],
            azimuth: values[1],
            elevation: values[2],
        };
        let noise = MeasurementNoise {
            sigma_rr: sigmas[0],
            sigma_az: sigmas[1],
            sigma_el: sigmas[2],
        };
        let q = matrix(&q)?;
        let result = py
            .detach(|| {
                self.inner
                    .observe(&measurement, &noise, Some(&q), UpdateOptions { gate })
            })
            .map_err(kernel_error)?;
        Ok((
            result.innovation.as_slice().to_vec(),
            result.nis,
            result.accepted,
        ))
    }
}

fn bounded_noise(
    filter: &TcaUkf,
    time: tychonic::satkit::Instant,
    noise: [f64; 4],
) -> PyResult<tychonic::estimate::TcaCovariance> {
    let dt = (time - filter.epoch()).as_seconds();
    if dt < 0.0 || noise.iter().any(|v| !v.is_finite() || *v < 0.0) || noise[2..].contains(&0.0) {
        return Err(PyValueError::new_err(
            "invalid TCA prediction interval or process noise",
        ));
    }
    Ok(tychonic::estimate::TcaCovariance::from_fn(|i, j| {
        if i == j {
            (dt * noise[i]).min((noise[i + 2].powi(2) - filter.covariance()[(i, i)]).max(0.0))
        } else {
            0.0
        }
    }))
}
