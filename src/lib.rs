//! Orbit-determination measurement and propagation kernels.
//!
//! See [`kernels`] for geometric observations and analytic Jacobians,
//! [`optim`] for prior-inclusive batch estimation, and [`estimate`] for EKF
//! updates and RTS smoothing. States use GCRF metres and metres per second;
//! carrier frequencies are Hz and phases are continuous radians.
//!
//! [`numeris`] and [`satkit`] are re-exported so downstream projects can use
//! exactly the matrix, time and propagation types accepted by this library.
#![doc = include_str!("../README.md")]
pub use numeris;
pub use satkit;
pub mod estimate;
pub mod kernels;
pub mod measurements;
pub mod optim;
pub use measurements::{OrbitFilter, OrbitFilterWithDrag, fit, fit_with_options};
