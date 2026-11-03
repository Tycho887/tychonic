# Changelog

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
