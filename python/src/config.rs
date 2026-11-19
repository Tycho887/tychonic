use pyo3::{exceptions::PyValueError, prelude::*};
use tychonic::kernels::{ParameterMode, PropagationConfig, od_prop_settings};
use tychonic::satkit::orbitprop::PropSettings;

use crate::common::{instant, kernel_error};

#[pyclass(frozen)]
pub struct NativePropagation {
    pub inner: PropagationConfig,
}

impl NativePropagation {
    /// Each fit owns its ephemeris cache for the full reference/observation interval.
    #[allow(clippy::disallowed_methods)]
    pub fn for_span(
        &self,
        first: tychonic::satkit::Instant,
        end: tychonic::satkit::Instant,
    ) -> PyResult<PropagationConfig> {
        let mut settings = self.inner.settings().clone();
        settings
            .precompute_terms(&first, &end)
            .map_err(|e| kernel_error(tychonic::kernels::KernelError::Propagation(e)))?;
        PropagationConfig::new(self.inner.reference_epoch(), settings, self.inner.mode())
            .map_err(kernel_error)
    }
    /// Satkit owns propagation settings; copy the immutable validated settings
    /// when creating an independently owned solver/filter configuration.
    #[allow(clippy::disallowed_methods)]
    pub fn build(&self) -> PyResult<PropagationConfig> {
        PropagationConfig::new(
            self.inner.reference_epoch(),
            self.inner.settings().clone(),
            self.inner.mode(),
        )
        .map_err(kernel_error)
    }
}

#[pymethods]
impl NativePropagation {
    #[new]
    fn new(
        epoch: &str,
        settings: &str,
        estimate_drag: bool,
        fixed_drag: f64,
        delta_cd: f64,
    ) -> PyResult<Self> {
        let settings = propagation_settings(settings)?;
        let mode = if estimate_drag {
            ParameterMode::StateAndDrag { delta_cd }
        } else {
            ParameterMode::StateOnly { fixed_drag }
        };
        Ok(Self {
            inner: PropagationConfig::new(instant(epoch)?, settings, mode).map_err(kernel_error)?,
        })
    }
}

fn propagation_settings(settings: &str) -> PyResult<PropSettings> {
    let mut defaults = serde_json::to_value(od_prop_settings())
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let overrides: serde_json::Map<String, serde_json::Value> = crate::common::json(settings)?;
    let object = defaults
        .as_object_mut()
        .ok_or_else(|| PyValueError::new_err("invalid propagation settings"))?;
    for (key, value) in overrides {
        if !object.contains_key(&key) || key == "precomputed" {
            return Err(PyValueError::new_err(format!(
                "unsupported propagation setting: {key}"
            )));
        }
        object.insert(key, value);
    }
    serde_json::from_value(defaults).map_err(|error| PyValueError::new_err(error.to_string()))
}
