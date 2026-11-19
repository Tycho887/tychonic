use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1};
use pyo3::{exceptions::PyValueError, prelude::*};
use serde::Deserialize;
use tychonic::{
    detection::{
        self, BcMpc, DetectionModel, EnuDirection, Ephemeris, ParticleBelief, PointingGeometry,
        SearchStatus,
    },
    kernels::{GroundStation, V3},
};

use crate::common::{acquisition_error, callable, instant, json, omm, tle};

#[derive(Deserialize)]
struct MpcInput {
    station: [f64; 3],
    start: String,
    config: ConfigInput,
    antenna: AntennaInput,
    elevation_mask_rad: f64,
    initial_direction: Option<[f64; 3]>,
}

#[derive(Deserialize)]
struct ConfigInput {
    search_duration_s: f64,
    move_s: f64,
    settle_s: f64,
    listen_s: f64,
    horizon: usize,
    beam_width: usize,
    candidate_count: usize,
    candidate_extent_sigma: f64,
    ess_threshold: f64,
}

#[derive(Deserialize)]
struct AntennaInput {
    rate_rad_s: f64,
    acceleration_rad_s2: f64,
}

impl MpcInput {
    fn setup(self) -> PyResult<detection::SearchSetup> {
        let c = self.config;
        Ok(detection::SearchSetup {
            station: GroundStation {
                lat_deg: self.station[0],
                lon_deg: self.station[1],
                alt_m: self.station[2],
            },
            start: instant(&self.start)?,
            config: detection::MpcConfig {
                search_duration_s: c.search_duration_s,
                move_s: c.move_s,
                settle_s: c.settle_s,
                listen_s: c.listen_s,
                horizon: c.horizon,
                beam_width: c.beam_width,
                candidate_count: c.candidate_count,
                candidate_extent_sigma: c.candidate_extent_sigma,
                ess_threshold: c.ess_threshold,
            },
            antenna: detection::AntennaLimits {
                rate_rad_s: self.antenna.rate_rad_s,
                acceleration_rad_s2: self.antenna.acceleration_rad_s2,
            },
            elevation_mask_rad: self.elevation_mask_rad,
            initial_direction: self
                .initial_direction
                .map(|v| EnuDirection::new(V3::from_array(v)))
                .transpose()
                .map_err(acquisition_error)?,
        })
    }
}

type Action = (usize, Vec<f64>, String, String, Option<usize>);
type Plan = (Vec<Action>, f64, f64);

#[pyclass]
pub struct NativeBCMPC {
    inner: BcMpc,
}

fn ephemeris(value: &str) -> PyResult<Ephemeris> {
    if value.trim_start().starts_with('{') {
        Ok(omm(value)?.into())
    } else {
        Ok(tle(value)?.into())
    }
}

fn status(value: SearchStatus) -> &'static str {
    match value {
        SearchStatus::Ready => "ready",
        SearchStatus::AwaitingObservation => "awaiting_observation",
        SearchStatus::Acquired => "acquired",
        SearchStatus::Exhausted => "exhausted",
    }
}

#[pymethods]
impl NativeBCMPC {
    #[new]
    fn new(
        py: Python<'_>,
        nominal: &str,
        particles: Vec<String>,
        weights: Vec<f64>,
        setup: &str,
    ) -> PyResult<Self> {
        let nominal = ephemeris(nominal)?;
        let particles = particles
            .iter()
            .map(|value| ephemeris(value))
            .collect::<PyResult<Vec<_>>>()?;
        let setup = json::<MpcInput>(setup)?.setup()?;
        let belief = ParticleBelief::new(particles, weights).map_err(acquisition_error)?;
        py.detach(|| BcMpc::new(nominal, belief, setup))
            .map(|inner| Self { inner })
            .map_err(acquisition_error)
    }

    fn snapshot(&self) -> (&'static str, usize, Vec<f64>, Vec<usize>, Vec<f64>, f64) {
        (
            status(self.inner.status()),
            self.inner.cycle(),
            self.inner.belief().weights().to_vec(),
            self.inner.belief().indices().to_vec(),
            self.inner.direction().vector().as_slice().to_vec(),
            self.inner.belief().effective_sample_size(),
        )
    }

    fn plan(&mut self, py: Python<'_>, callback: Py<PyAny>) -> PyResult<Plan> {
        callable(callback.bind(py))?;
        py.detach(|| {
            let mut model = PythonDetection {
                callback,
                error: None,
            };
            let result = self.inner.plan(&mut model);
            if let Some(error) = model.error {
                return Err(error);
            }
            let plan = result.map_err(acquisition_error)?;
            Ok(plan_rows(plan))
        })
    }

    fn plan_receiver(&mut self, py: Python<'_>, settings: [f64; 5]) -> PyResult<Plan> {
        let mut model = tychonic::simulation::ReceiverConfig {
            beam_fwhm_rad: settings[0],
            cutoff_rad: settings[1],
            on_axis_ebn0_db: settings[2],
            acquire_ebn0_db: settings[3],
            hold_ebn0_db: settings[3],
            acquisition_s: settings[4],
            ..Default::default()
        };
        model.validate().map_err(crate::common::kernel_error)?;
        py.detach(|| {
            self.inner
                .plan(&mut model)
                .map(plan_rows)
                .map_err(acquisition_error)
        })
    }

    fn observe(
        &mut self,
        py: Python<'_>,
        outcome: &str,
        draw: Py<PyAny>,
    ) -> PyResult<(bool, &'static str)> {
        callable(draw.bind(py))?;
        let outcome = match outcome {
            "miss" => detection::Observation::Miss,
            "acquired" => detection::Observation::Acquired,
            _ => {
                return Err(PyValueError::new_err(
                    "outcome must be 'miss' or 'acquired'",
                ));
            }
        };
        py.detach(|| {
            let mut callback_error = None;
            let result = self.inner.observe(outcome, || {
                let result = Python::attach(|py| draw.call0(py)?.extract::<f64>(py));
                match result {
                    Ok(draw) => draw,
                    Err(error) => {
                        callback_error = Some(error);
                        f64::NAN
                    }
                }
            });
            if let Some(error) = callback_error {
                return Err(error);
            }
            let result = result.map_err(acquisition_error)?;
            Ok((result.resampled, status(result.status)))
        })
    }
}

struct PythonDetection {
    callback: Py<PyAny>,
    error: Option<PyErr>,
}

impl PythonDetection {
    fn evaluate(
        &self,
        command: EnuDirection,
        particles: &[PointingGeometry],
        listen_s: f64,
    ) -> PyResult<Vec<f64>> {
        Python::attach(|py| {
            let command = command.vector().as_slice().to_vec().into_pyarray(py);
            let directions: Vec<_> = particles
                .iter()
                .map(|p| p.direction_enu.vector().as_slice().to_vec())
                .collect();
            let directions = PyArray2::from_vec2(py, &directions)
                .map_err(|error| PyValueError::new_err(error.to_string()))?;
            let ranges: Vec<_> = particles.iter().map(|p| p.range_m).collect();
            let elevations: Vec<_> = particles.iter().map(|p| p.elevation_rad).collect();
            let visible: Vec<_> = particles.iter().map(|p| p.visible).collect();
            let output = self.callback.call1(
                py,
                (
                    command,
                    directions,
                    ranges.into_pyarray(py),
                    elevations.into_pyarray(py),
                    visible.into_pyarray(py),
                    listen_s,
                ),
            )?;
            let array = output.bind(py).extract::<PyReadonlyArray1<'_, f64>>()?;
            Ok(array.as_array().iter().copied().collect())
        })
    }
}

impl DetectionModel for PythonDetection {
    fn probabilities(
        &mut self,
        command: EnuDirection,
        particles: &[PointingGeometry],
        listen_s: f64,
        output: &mut [f64],
    ) -> detection::Result<()> {
        let result = self
            .evaluate(command, particles, listen_s)
            .and_then(|values| {
                if values.len() != output.len() {
                    return Err(PyValueError::new_err(
                        "receiver must return one probability per particle",
                    ));
                }
                output.copy_from_slice(&values);
                Ok(())
            });
        result.map_err(|error| {
            self.error = Some(error);
            detection::AcquisitionError::Detection("Python receiver callback failed".into())
        })
    }
}

fn plan_rows(plan: &detection::ActionPlan) -> Plan {
    let actions = plan
        .actions
        .iter()
        .map(|action| {
            (
                action.cycle,
                action.direction_enu.vector().as_slice().to_vec(),
                action.listen_start.as_rfc3339(),
                action.listen_end.as_rfc3339(),
                action.candidate_index,
            )
        })
        .collect();
    (
        actions,
        plan.acquisition_probability,
        plan.retained_acquisition_probability,
    )
}

#[pyfunction]
pub fn particles(source: &str, offsets: Vec<[f64; 4]>) -> PyResult<Vec<String>> {
    let original = tle(source)?;
    offsets
        .into_iter()
        .map(|[phase, inc, raan, motion]| {
            let mut value = tychonic::satkit::omm::OMM::from_tle(&original);
            value.mean_anomaly =
                (value.mean_anomaly + phase * value.mean_motion / 240.).rem_euclid(360.);
            value.inclination += inc;
            value.raan = (value.raan + raan).rem_euclid(360.);
            value.mean_motion += motion;
            serde_json::to_string(&value).map_err(|e| PyValueError::new_err(e.to_string()))
        })
        .collect()
}
