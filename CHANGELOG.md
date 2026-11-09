# Changelog

## 0.9.2

- Replace reference counting and interior mutability in Tychonic-owned code
  with owned or borrowed geometry, observation data and problems, plus reusable
  mutable evaluation buffers. Kernels execute sequentially on the caller's thread.
- Add a local fallible LM driver for ordinary, RF and variable-projection fits,
  preserving damping and convergence equations while evaluating residuals and
  Jacobians together and rejecting negative-drag trials before propagation.
- Validate sensitivity arrays, RF data, priors, epochs and integration controls
  before use. Return numerical and solver errors without panic extraction.
- Store filter history with an always-present current record; restore small
  checkpoints on failed observations and materialize RTS records explicitly.
- Retain owned covariance factors and share one factor across independent joint
  rows. Remove whole-estimator and whole-evaluation cloning.
- Forbid unsafe library code and enforce ownership/error policies with Clippy,
  a source check and CI. Pin and document the Numeris/Satkit trust boundary and
  narrowly scoped external settings/TLE copies.
- Intentionally break ownership, validated-input and history APIs; update
  examples, tests and tools. See [migration](docs/migration_0_9_2.md),
  [safety contracts](docs/safety.md) and [qualification](docs/qualification_0_9_2.md).

- Add a two-state TCA UKF for TLE epoch correction and a constant pass
  range-rate bias, using combined range-rate and soft azimuth/elevation readings.
- Default TCA pointing noise to independent 2.5-degree standard deviations;
  retain observation-time station geometry and invalidate shifted SGP4 caches.

## 0.9.0

- Add a calibrated facade for standalone Doppler Hz, continuous phase difference,
  exact differential Doppler, and azimuth/elevation.
- Fuse any selection with full per-epoch covariance in batch fits and EKF
  updates, retaining shared ordinary/differential Doppler noise.
- Add fit convenience functions, named orbit filter aliases, and an RF builder
  with named priors for existing offset, drift and baseline-calibration models.
- Preserve analytic Jacobians, prepared geometry, propagation, variable
  projection, augmented RF estimators and RTS smoothing.
- Preserve existing public functions; extend `MeasurementKind` with `Joint`.
  Downstream exhaustive matches must handle the new variant.
- Move research/profiling executables and plotting scripts to the development
  tools crate. Replace the default executable with the `batch_fit` example.
- Keep generated experiments and reference PDFs locally, outside Git tracking
  and release packages. Add MIT/Apache-2.0 licensing and release metadata.
- Clarify that fitting requires an initial orbit guess; this release does not
  add orbit initialization without a guess or receiver signal extraction.

Earlier experimental implementations are not assigned release versions here.
