# Continuous phase-difference observations

The implementation follows the far-field observation model in Henault and
Guimond, *Orbit determination of a LEO satellite with passive RF*, Advances in
Space Research 75 (2025), equations (1)–(3). The second supplied reference,
*One-way Doppler and Interferometry: Space Domain Awareness* (2022), gives the
corresponding phase/path-length relation. Supplied PDFs remain locally in ignored `references/`; see the
[model citations](../references/README.md) for publisher links.

Define the surveyed baseline as **antenna 1 minus antenna 2**, expressed in
antenna 1's east/north/up frame in metres. `Interferometer::new(carrier_hz)`
sets the nominal carrier frequency. Measurements supply their baseline explicitly;
batches require one vector per measurement, including repeated vectors for a
fixed baseline. The prepared station rotation
transforms the baseline to GCRF at each observation epoch. Reversing the baseline
reverses the geometric phase and its derivatives. Swapping the observed channel
order therefore requires reversing the baseline and offset convention together.

For the GCRF satellite position/velocity, let

\[
\rho=\|r-r_1\|,\qquad \ell=(r-r_1)/\rho,\qquad D=B^T\ell.
\]

The frequency observation is represented as a shift relative to the nominal
carrier, avoiding subtraction of GHz-sized values:

\[
z_f=-f_c\dot\rho/c+b_f+\epsilon_f,
\qquad f=f_c+z_f-\epsilon_f.
\]

The continuous phase model is

\[
z_\phi={2\pi\over c}fD+b_\phi+d(t)+\epsilon_\phi.
\]

The received-frequency term uses the predicted Doppler and frequency offset,
never a noisy frequency observation. White frequency and phase measurement
errors are assumed independent in the study. The streaming RF filter accepts a
full positive-definite measurement covariance for correlated readings.

All phase values, offsets and standard deviations are **radians** inside the
library. This model uses continuous, unwrapped phase and ordinary subtraction
for its residuals. A constant integer-cycle ambiguity is absorbed in the unknown
hardware mean, subject to its chosen prior; time-varying cycle slips require
upstream detection and handling. The model is intended for short baselines and
local orbit refinement, without light-time or atmospheric corrections.

## Analytic Jacobian

`phase_difference_partial` returns carrier-relative Doppler and phase predictions,
a two-row instantaneous state Jacobian, and the direct frequency-offset
derivative of phase. `Interferometer::evaluate` prepares the supplied ENU baseline in GCRF using
the supplied station geometry.

Using the existing range-rate partials,

\[
H_f=-{f_c\over c}H_{\dot\rho},\qquad
H_D=\left[{B^T(I-\ell\ell^T)\over\rho},\;0\right],\qquad
H_\phi={2\pi\over c}(fH_D+DH_f).
\]

The small velocity sensitivity through Doppler is retained. Direct offset
columns are

\[
\partial z_f/\partial b_f=1,\quad
\partial z_\phi/\partial b_f=2\pi D/c,\quad
\partial z_\phi/\partial b_\phi=
\partial z_\phi/\partial d=1.
\]

The phase kernel is defined at zenith. It does not compute azimuth, which is
singular there. Batch state derivatives multiply these instantaneous partials by
the STM and optional drag sensitivity; EKF updates use the instantaneous
partials directly.

## Receiver offset and estimation

`OuPhaseNoise` describes stationary zero-mean drift around a separately
estimated constant hardware mean:

\[
a=e^{-\Delta t/\tau},\qquad
d_{k+1}=ad_k+w_k,\qquad
\operatorname{Var}(w_k)=\sigma_d^2(1-a^2).
\]

The default study uses \(\tau=21600\) s and \(\sigma_d=5°\). RMS offset change
is about 1.43° over 900 s and 0.958° over the retained 400 s pass. This variability
is separate from the four white phase measurement standard deviations.

For calibrated phase, use `PreparedBatch::phase_difference` in `ODProblem` or
`PhaseSensor` in `OrbitEkf`. `OrbitEkf::filter_batches` schedules phase batches
alongside its existing sensor types. These interfaces assume zero receiver
offset after calibration.

For unknown drifting offsets, use `PassiveRfData`, `RfPrior`,
`PassiveRfProblem`, `fit_passive_rf`, and `PassiveRfEkf`:

- Batch parameters are orbital state (optionally drag), frequency offset in Hz,
  and phase hardware mean in radians. Batch residuals include a diagonal
  Gaussian prior and correlated phase whitening using
  \(C_{ij}=\sigma_\phi^2\delta_{ij}+\sigma_d^2e^{-|t_i-t_j|/\tau}\).
  `PassiveRfFit::drift_rad` is the conditional posterior drift mean at each input
  epoch. It does not contain drift covariance or parameter uncertainty.
- `PassiveRfEkf<9>` uses six orbital states followed by frequency offset, phase
  mean and OU drift. `<10>` inserts drag at index 6. The initial OU state has its
  stationary variance, independent of the orbital/offset prior. Prediction uses
  the orbital STM and the exact OU transition, adding process noise once per new
  epoch. `filter_data` sorts input epochs stably; `predict_to` and `update` support
  streaming use. Individual reading failures roll back their prediction, while
  earlier successful readings remain committed. Rejected readings retain their
  prediction. RTS smoothing uses the same augmented transitions.
- The batch covariance is the inverse prior-inclusive information matrix, in
  physical units, without RSS rescaling. Its orbital marginal includes receiver
  offset uncertainty. Propagate that marginal with the STM to compare against
  the final EKF covariance. The final RTS result equals the final EKF result.

The RF batch adapter uses numeris LM. Numeris 0.6.0 checks step/cost tolerances
only after accepted steps. At an iteration limit, the adapter independently
solves an undamped QR linearization at its best evaluated candidate. It accepts
that candidate only when the supplied scaled-step or predicted-cost tolerance
is met, records `linearized_tolerance = true`, and otherwise preserves the
iteration-limit error. No kernel error is converted to convergence.

## Reproducing the study

```sh
SATKIT_OFFLINE=1 cargo run --manifest-path tools/Cargo.toml --locked --release --bin phase_pass -- \
  --trials 100 --seed 20261103 --output artifacts/phase_pass
python3 tools/scripts/plot_phase_pass.py artifacts/phase_pass
```

The plotting script requires numpy and matplotlib and writes PNG, SVG and PDF
figures, `summary.csv`, and `report.md`. Run the executable with `--help` for
geometry, cadence, visibility, Doppler noise and OU settings. `--start-trial`
replays a specific trial range without changing its seeds.

To reproduce the paired drift comparison in the saved report:

```sh
SATKIT_OFFLINE=1 cargo run --manifest-path tools/Cargo.toml --locked --release --bin phase_pass -- \
  --trials 100 --drift-deg 0.001 --output artifacts/phase_pass_negligible_drift
python3 tools/scripts/plot_phase_pass.py artifacts/phase_pass \
  --compare-negligible-drift artifacts/phase_pass_negligible_drift
```

The default simulation uses the existing ISS TLE as a seed for numerical truth
propagation, a station beneath its midpoint position, a 900 s candidate window,
and a 10° elevation mask. The actual observed interval is 400 s with 41 samples.
It assumes fixed true drag, matched deterministic orbital dynamics, 0.05 m/s
equivalent Doppler noise, and prior standard deviations of 1 km per position
component, 1 m/s per velocity component, 10 Hz frequency offset, and 180° phase
mean. Each trial independently draws initial errors, hardware offsets and a
stationary OU trajectory. Standardized frequency/phase noise and those draws
are shared across the four phase noise levels and Doppler-only baseline.

Artifacts retain the configuration, solver settings, seeds, priors, truth,
observations, endpoint estimates/covariances, per-trial errors, NEES and NIS, and
the first trial's batch/EKF/RTS histories. Error statistics use successful trials
and explicitly report failures and verified iteration-limit convergence.
The RMS/percentile results describe this synthetic pass; they do not establish
real-data accuracy or performance across different baseline/pass geometries.

Validation is in `tests/phase.rs`: analytic central differences, received-frequency
coupling, baseline reversal, zenith, invalid inputs, OU covariance and irregular
epochs, batch whitening/STM/drag columns, calibrated phase scheduling, augmented
filter covariance, linear Gaussian batch/sequential equivalence, offset recovery,
and independently verified convergence.
