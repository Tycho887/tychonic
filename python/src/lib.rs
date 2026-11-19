//! Python adapters; numerical algorithms remain in the independent Rust core.
#![forbid(unsafe_code)]

mod common;
mod config;
mod measurement;
mod mpc;
mod orbit;
mod profiles;
mod solve;
mod tca;

use pyo3::prelude::*;

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("core_version", tychonic::VERSION)?;
    common::register(module)?;
    register_types(module)?;
    register_functions(module)
}

fn register_types(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<config::NativePropagation>()?;
    module.add_class::<measurement::NativeBatch>()?;
    module.add_class::<orbit::NativeOrbitEKF>()?;
    module.add_class::<tca::NativeTcaUKF>()?;
    module.add_class::<mpc::NativeBCMPC>()?;
    Ok(())
}

fn register_functions(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(mpc::particles, module)?)?;
    module.add_function(wrap_pyfunction!(profiles::contacts, module)?)?;
    module.add_function(wrap_pyfunction!(profiles::passive, module)?)?;
    module.add_function(wrap_pyfunction!(solve::calibrated, module)?)?;
    module.add_function(wrap_pyfunction!(solve::transport, module)?)?;
    module.add_function(wrap_pyfunction!(solve::tle_seed, module)?)?;
    module.add_function(wrap_pyfunction!(solve::arc, module)?)?;
    module.add_function(wrap_pyfunction!(solve::mean_elements, module)?)?;
    module.add_function(wrap_pyfunction!(orbit::fit_orbit, module)?)?;
    module.add_function(wrap_pyfunction!(orbit::white_acceleration_noise, module)?)?;
    module.add_function(wrap_pyfunction!(common::elapsed, module)?)?;
    module.add_function(wrap_pyfunction!(common::roundtrip_time, module)?)?;
    Ok(())
}
