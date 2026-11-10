//! Bayesian Continuation MPC acquisition kernels.
//!
//! Supply a nominal SGP4 orbit, explicit weighted hypotheses, and a receiver
//! model. [`BcMpc`] owns the feasible continuation and particle belief; the
//! caller executes the first action from [`BcMpc::plan`] and reports only a
//! successfully completed listen interval to [`BcMpc::observe`]. ATUS is a
//! private initialization detail. Kernels run on the caller's thread.
//!
//! Distances are metres, durations seconds, and angles radians (except the
//! existing [`GroundStation`] geodetic fields, which remain degrees).
//! Directions are absolute East–North–Up pointings. Geometry is held at each
//! listening midpoint, with conditionally independent detection attempts.

mod belief;
mod geometry;
mod mpc;
mod planner;
mod types;

pub use crate::kernels::GroundStation;
pub use belief::*;
pub use geometry::*;
pub use mpc::*;
pub use planner::*;
pub use types::*;

use crate::kernels::KernelError;
use std::{error::Error, fmt};

/// Checked acquisition failures. No unsupported miss silently resets a belief.
#[derive(Debug)]
pub enum AcquisitionError {
    InvalidInput(&'static str),
    InvalidState(&'static str),
    ImpossibleObservation,
    Kernel(KernelError),
    /// A caller's receiver model failed to evaluate a command.
    Detection(String),
}

impl fmt::Display for AcquisitionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(s) => write!(f, "invalid acquisition input: {s}"),
            Self::InvalidState(s) => write!(f, "invalid acquisition state: {s}"),
            Self::ImpossibleObservation => write!(f, "miss has zero probability under the belief"),
            Self::Kernel(e) => e.fmt(f),
            Self::Detection(s) => write!(f, "detection model failed: {s}"),
        }
    }
}
impl Error for AcquisitionError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Kernel(e) => Some(e),
            _ => None,
        }
    }
}
impl From<KernelError> for AcquisitionError {
    fn from(value: KernelError) -> Self {
        Self::Kernel(value)
    }
}
pub type Result<T> = std::result::Result<T, AcquisitionError>;

fn positive(value: f64, name: &'static str) -> Result<()> {
    if !value.is_finite() || value <= 0.0 {
        return Err(AcquisitionError::InvalidInput(name));
    }
    Ok(())
}
fn probability(value: f64) -> Result<()> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(AcquisitionError::InvalidInput(
            "probability must be finite in [0, 1]",
        ));
    }
    Ok(())
}
