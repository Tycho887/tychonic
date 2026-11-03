//! Instantaneous measurement kernels for an EKF and chained batch Jacobians.
//!
//! States are GCRF `[x, y, z, vx, vy, vz]` in metres and metres/second.
//! The seventh parameter is **C_d A / m**, in m²/kg, held constant over an arc.
//! Angles are radians; azimuth is measured eastward from north. Range-rate
//! is positive for a receding target. Measurements are simultaneous geometric
//! observables (no light-time or refraction). RF kernels support receiver offsets.

use numeris::{Matrix, Vector};
use satkit::{frametransform, orbitprop};
use std::{error::Error, f64::consts::PI, fmt};

pub mod observation_models;
pub mod phase;
pub mod rotation;
pub mod stm;

pub use observation_models::*;
pub use phase::*;
pub use rotation::*;
pub use stm::*;

pub type V3 = Vector<f64, 3>;
pub type V6 = Vector<f64, 6>;
pub type V7 = Vector<f64, 7>;
pub type M3 = Matrix<f64, 3, 3>;
pub type M6 = Matrix<f64, 6, 6>;
pub type M7 = Matrix<f64, 7, 7>;

#[derive(Debug)]
pub enum KernelError {
    InvalidInput(&'static str),
    ZeroRange,
    UndefinedAzimuth,
    Singular,
    Propagation(orbitprop::Error),
    Tle(String),
    Frame(frametransform::Error),
}

impl fmt::Display for KernelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(s) => write!(f, "invalid kernel input: {s}"),
            Self::ZeroRange => write!(f, "measurement is undefined at zero range"),
            Self::UndefinedAzimuth => write!(f, "azimuth/Jacobian is undefined at zenith or nadir"),
            Self::Singular => write!(f, "information matrix is singular"),
            Self::Propagation(e) => write!(f, "propagation failed: {e}"),
            Self::Tle(e) => write!(f, "TLE propagation failed: {e}"),
            Self::Frame(e) => write!(f, "frame transform failed: {e}"),
        }
    }
}

impl Error for KernelError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Propagation(e) => Some(e),
            Self::Frame(e) => Some(e),
            _ => None,
        }
    }
}

impl From<orbitprop::Error> for KernelError {
    fn from(e: orbitprop::Error) -> Self {
        Self::Propagation(e)
    }
}

pub type Result<T> = std::result::Result<T, KernelError>;

pub fn v3(x: f64, y: f64, z: f64) -> V3 {
    numeris::vector![x, y, z]
}

pub fn v6_from(r: &V3, v: &V3) -> V6 {
    numeris::vector![r[0], r[1], r[2], v[0], v[1], v[2]]
}

pub fn r_of(s: &V6) -> V3 {
    s.block::<3, 1>(0, 0)
}

pub fn v_of(s: &V6) -> V3 {
    s.block::<3, 1>(3, 0)
}

/// Canonical angular difference in [-π, π). Non-finite inputs return NaN.
pub fn wrap_pi(a: f64) -> f64 {
    // Preserve tiny angular differences without losing them when adding a turn.
    if (-PI..PI).contains(&a) {
        return a;
    }
    let x = a.rem_euclid(2.0 * PI);
    if x >= PI { x - 2.0 * PI } else { x }
}

fn finite<const R: usize, const C: usize>(x: &Matrix<f64, R, C>) -> bool {
    (0..R).all(|i| (0..C).all(|j| x[(i, j)].is_finite()))
}

fn check_state(x: &V6) -> Result<()> {
    let radius = r_of(x).norm();
    if !finite(x) || !radius.is_finite() || radius <= 0.0 {
        return Err(KernelError::InvalidInput(
            "state must be finite with nonzero geocentric radius",
        ));
    }
    Ok(())
}

/// Compose instantaneous measurement partials with state/parameter
/// sensitivities. Suitable for any fixed-size geometric observation model.
pub fn chain_rule<const M: usize, const P: usize>(
    partial: &Matrix<f64, M, 6>,
    sensitivity: &Matrix<f64, 6, P>,
) -> Result<Matrix<f64, M, P>> {
    if !finite(partial) || !finite(sensitivity) {
        return Err(KernelError::InvalidInput(
            "partials and sensitivities must be finite",
        ));
    }
    let result = chain_rule_prevalidated(partial, sensitivity);
    if !finite(&result) {
        return Err(KernelError::InvalidInput(
            "chained Jacobian exceeds numeric range",
        ));
    }
    Ok(result)
}

// Prepared callers validate input invariants before composing and check every
// output during whitening. Keep the numerical operation shared with raw APIs.
pub(crate) fn chain_rule_prevalidated<const M: usize, const P: usize>(
    partial: &Matrix<f64, M, 6>,
    sensitivity: &Matrix<f64, 6, P>,
) -> Matrix<f64, M, P> {
    partial * sensitivity
}
