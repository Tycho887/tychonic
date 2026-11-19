# Constrained range-rate batch fitting

Tychonic 0.10.0 provides the prepared orbit model in
`tychonic::optim::constrained`. The companion crate `tychonic-constrained`
pins the Rust interior-point solver ripopt 0.8.2. Applications can depend on
the two crates through these workspace paths when developing against this checkout.

```toml
[dependencies]
tychonic = { path = "/path/to/tychonic" }
tychonic-constrained = { path = "/path/to/tychonic/constrained" }
```

```rust
use tychonic::kernels::{ParameterMode, PropagationConfig, od_prop_settings};
use tychonic_constrained::{ConstrainedRangeRateProblem, FitOptions, fit_constrained_range_rate};

let config = PropagationConfig::new(
    reference_epoch, od_prop_settings(), ParameterMode::StateAndDrag { delta_cd: 1e-4 },
)?;
let problem = ConstrainedRangeRateProblem::new(
    config, seed, &range_rates, &[0.05], &pointings, 0.5_f64.to_radians(), Default::default(),
)?;
let fit = fit_constrained_range_rate(&problem, &FitOptions::default())?;
```

Supply the GCRF position in metres, velocity in m/s and constant drag
susceptibility Cd A/m in m²/kg as a `V7` seed. The seventh parameter has a
zero lower bound. The verified force model includes atmospheric drag. For
repeated evaluations over longer arcs, Satkit's `PropSettings::precompute_terms`
can prepare the force terms over the entire propagation interval before
constructing the config.

A contact ID identifies one pass at a single fixed station. Use a separate ID
for each pass, including repeat visits to the same site. Every range-rate row
enters the objective. Supply a common standard deviation in m/s or a slice of
positive sigmas aligned with those rows. At every trial orbit, the optimal
constant bias added to the rate prediction is the weighted mean of observed
minus predicted range rate within the pass, using weights 1/σ². A contact
with one rate has zero weight on the orbit after projection. Estimating drag
requires sufficient time separation and rate geometry across the arc.

Supply each boresight as a `ContactPointing` with its contact, station, epoch,
azimuth clockwise from north, and elevation, both in radians. FOV is the **full
angular diameter** of a circular field, in radians. A 0.5° FOV permits a 0.25°
offset from the supplied boresight. The constraint uses squared chord length
between the observed and predicted unit directions, divided by
`4 sin²(FOV/4)`, with upper bound one. This form works at zenith. An antenna's
gain width is appropriate only if that width describes a defensible hard
limit on the source direction.

`PointingSelection::NearTca(k)` accepts 1–10 and defaults to three. Preparation
finds the smallest sampled slant range from the seed orbit. It selects that
epoch and spreads the other samples over the central part of the observed
pass. Each side extends halfway from closest approach to its last observed
epoch. Missing slots on one side transfer to the other. The earlier sample
wins a time or slant-range tie. A duplicate contact/epoch uses its first input
boresight. The selected epochs stay fixed throughout the solve. Contacts
without pointing contribute their rates. `PointingSelection::All` fits every
distinct pointing epoch and provides a reference for choosing a budget.

The initializer factors the projected range-rate Jacobian in scaled orbit
coordinates at the seed. Its QR factorization includes a small diagonal block
to bound steps in weak directions. When the proposed correction exceeds five
units in the scaled Euclidean norm, the initializer damps the step and can
relinearize at up to three successively improved orbits. It also checks a
final Gauss–Newton trial. Every accepted trial reduces projected RSS and
preserves or improves angular feasibility. The station geometry and selected
pointing epochs stay fixed throughout initialization.
The solver uses limited-memory Hessians. After a numerical failure it can retry
once with an adaptive barrier within the original iteration budget. The default
scaled solver tolerance is `1e-5`; the independent final angular tolerance is
`1e-8` rad. Choose a solver tolerance consistent with the precision of the
propagated derivatives over the arc.

The result includes the physical state, biases keyed by contact, whitened
range-rate RSS, selected pointings, solver status and iteration count,
sensitivity evaluation count, and maximum positive angular excess in radians.
Only the selected pointings define feasibility. Re-evaluate an `All` model at
the fitted parameters to check the omitted directions. The companion accepts
optimal and acceptable solver terminations after checking angular feasibility.
A failed evaluation, solver termination, or a returned orbit outside the
angular tolerance produces a typed error. The fitter returns a point estimate.

## Sampling results

The release fixture contains 51 range rates and 51 candidate pointings across
three station passes over 2.7 hours. The seed has a 135 m position error and a
0.003 m²/kg drag error. Each pass has a separate bias. The rate noise is
deterministic, and one pass-edge pointing is displaced by 160 µrad from truth
with a 150 µrad half-angle. All runs use the same rates and propagation settings.

| Pointings per pass | Cones | Sensitivity calls | Seconds | Position error (m) | Velocity error (m/s) | Drag error (m²/kg) | Max bias error (m/s) | Worst omitted excess (µrad) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3 | 24 | 2.07 | 4.18 | 0.01678 | 0.00001775 | 0.00312 | 5.46 |
| 3 | 9 | 18 | 1.52 | 4.18 | 0.01678 | 0.00001775 | 0.00312 | 5.46 |
| 10 | 30 | 30 | 2.79 | 4.18 | 0.01677 | 0.00001775 | 0.00312 | 5.46 |
| All 17 | 51 | 164 | 16.42 | 14.84 | 0.04852 | 0.00004948 | 0.00377 | — |

These are single-run release-build timings, including problem preparation and
fit. The all-pointing fit triggered the adaptive-barrier retry. All four fits
met the selected cones. The three sparse fits omitted the displaced pass-edge
reading. Including it raised the projected RSS from 6.273 to 6.373.

The separate noiseless reference uses native COIN-OR IPOPT 3.11.9 on the same
prepared model. Both solvers satisfied the cones from feasible and infeasible
seed orbits. An additional displaced-pointing case checks objective agreement
at an active boundary, and a ten-variable joint fit checks the recovered orbit
and three biases independently of the projection.

On a noiseless 24-hour fixture, the constrained fitter recovered the reference
position to 0.36 mm with ten sensitivity evaluations and zero angular excess.
That regression uses `FitOptions { tolerance: 1e-3, ..Default::default() }`.
Tighter stationarity thresholds can prolong a daily fit after the predicted
observations and fitted orbit have converged to the integration precision.

## Reproduce the qualification

Provision Satkit data and set `SATKIT_DATA` to that directory. Run these checks
from the repository root.

```sh
cargo test --locked --release -p tychonic-constrained
cargo test --locked --release -p tychonic-constrained --test fit -- --ignored --nocapture --test-threads=1
cargo test --locked --release --test bootstrap bootstrap_24_hour_fit_recovers_state_and_drag_from_weak_pointing -- --ignored --exact
```

The optional COIN-OR reference test uses the same prepared evaluator with
independent callbacks. It requires the native IPOPT development package,
CMake, libclang and a C++ compiler. On Debian/Ubuntu use the shared library
discovery path in ipopt-sys 0.6.0.

```sh
IPOPT_NO_PKG_CONFIG=1 cargo test --locked --release --manifest-path tools/Cargo.toml --features native-reference --test native_reference -- --nocapture --test-threads=1
```
