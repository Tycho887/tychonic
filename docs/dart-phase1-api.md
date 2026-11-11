# DART Phase 1 covariance requirements

Status: implemented and independently verified for 0.9.4.
Consumer: DART registry dependency.

## Additive interface

Keep existing `fit_lm` and `measurements::fit_with_options` behavior unchanged.
Add `fit_lm_with_covariance` and `measurements::fit_with_covariance` with the
same problem/configuration, guess, scales, and LM settings inputs.
Return parameters, iterations, RSS, parameter covariance, numerical rank, and
an explicit absolute-whitened-information covariance method.

Use the converged evaluation already retained by LM. The covariance is the
inverse of the final whitened Jacobian information matrix in physical parameter
units. Supplied measurement covariance defines absolute uncertainty: do not
multiply by RSS or reduced chi-square and do not include optimizer damping.
Compute covariance using a stable rank-revealing numeris factorization. Detect
rank deficiency explicitly and return a structured error; do not regularize,
truncate, or return a pseudo-prior for unobservable parameters. Validate finite
entries and symmetry. Document rank tolerance and scaling semantics.

Support six-state and existing state-plus-drag parameter modes, correlated
mixed observables, and canonical observable ordering. Optimizer scales must
not change the physical covariance. Leave contact-bias Varpro's residual-scaled
Schur covariance unchanged.

## Verification

Write failing tests before implementation. Verify an analytic linear reference,
correlated-noise whitening, optimizer-scale invariance, singular/underdetermined
geometry, non-finite input, and noisy mixed Doppler/phase/differential/angle orbit
recovery. Check all retained APIs and tests, Rustdoc, formatting, and Clippy.
Include a deterministic mixed-observation JSON fixture suitable for DART's
generation-2 normalized contract, with held-out later observations for EKF.

## Release and repository boundary

Develop and verify upstream changes in the canonical local tychonic checkout,
`/home/michaeljohansen/tychonic`. Preserve unrelated local work. This source must
never become a DART path or Git dependency. Prepare a patch release and release
notes after verification; DART integration must use the published registry
release. The repository owner performs publication.
