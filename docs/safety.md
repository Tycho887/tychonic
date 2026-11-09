# Safety and numerical contracts in 0.9.2

Tychonic forbids unsafe Rust in its library and uses exclusive ownership for
mutable computational state. Geometry, observation data, and estimator inputs
can be owned or borrowed. The core contains no `Arc`, `Rc`, `RefCell`, locks,
worker creation, or panic-based error extraction. Each call executes sequentially
on its caller's thread; independent callers can use separate workspaces concurrently.

This guarantee covers Tychonic's source. Its pinned numerical dependencies
contain unsafe internals and are a separate trust boundary. The audit below
identifies their relevant behavior; it is not a formal proof of their soundness.

## Ownership and structural proofs

`PreparedGeometry` owns its entries and keeps its row mapping private. Both
constructors insert an entry before storing its index. Borrowed row iteration
therefore cannot refer to a missing entry. Invalid external row requests return
`None`; geometry imports reject misaligned arrays, invalid epochs, nonfinite
station states, and invalid rotations.

A prepared batch's variant contains the observation columns and sensor it needs.
The number of columns follows the variant; a phase batch always contains an
interferometer. Joint observable selections are nonempty, unique, and sorted;
there are at most five physical components. Construction checks every row's
length, covariance, and size arithmetic. The private evaluation dispatch chooses
the corresponding fixed measurement dimension.

`Sensitivities::new` consumes aligned arrays and validates epochs and finite
values. Samplers establish the same invariant while checking propagated outputs.
A mapped evaluation validates all selected epochs before writing any rows. Epoch
lookup succeeds only for an in-range sample, and private array alignment then
proves that its state and derivatives exist. Drag columns are either absent or
present for the entire schedule. Empty externally supplied schedules use the
state-only layout.

Dimension multiplication and stacked offsets are checked before allocating or
writing output matrices. Fixed matrix blocks use the validated six/seven orbit
columns and selected measurement dimension. Bounds checking remains enabled:
proofs do not replace indexing with unchecked accesses.

`FilterHistory` stores a current record as a field and completed records in an
archive. An update always has a current record. Predictions move that record
into the archive, and checkpoints restore its fixed-size values after a failed
observation. RTS smoothing explicitly materializes contiguous fixed-size records
for Numeris's slice API. TCA observations calculate a candidate numeric estimate
and commit it only after successful prediction and update; gated readings retain
prediction as documented.

## Numerical obligations

Structural validity does not establish that every numerical calculation succeeds.
The following conditions remain checked where they depend on the evaluated state:

| Operation | Mathematical requirement | Runtime treatment |
| --- | --- | --- |
| Range rate | Nonzero satellite-to-station range | `ZeroRange`; finite-output checks |
| Azimuth Jacobian | Observable horizontal direction | Existing horizontal/range threshold of `1e-12`; `UndefinedAzimuth` |
| Rotation | Orthonormal, right-handed basis | Checked once when preparing geometry |
| Whitening | Symmetric positive-definite covariance | Check inputs and retain fallible Cholesky |
| Information/covariance recovery | Identifiable parameter directions | Retain fallible decomposition and `Singular` |
| Propagation | Supported force model and available external data | Validate setup and return Satkit errors |
| Floating-point composition | Representable intermediate and final values | Return `Numerical` on detected overflow/nonfinite output |

Propagation and geometry entry points reject `Instant::INVALID` before calling
frame or orbit routines. Integration setup checks gravity limits, a positive
step budget, and finite nonnegative adaptive tolerances with at least one
positive tolerance. Step hints must be finite and nonzero; either sign is
supported because Satkit takes their magnitude. Gauss-Jackson validates its
fixed step and ignores adaptive controls. TLE conversion checks the external
one-epoch output shape and the resulting state before exposing it.

For independent positive noise, `D` is positive definite and
`xᵀ(D + K)x > 0` for nonzero `x` whenever `K` is positive semidefinite. This
explains why adding positive white noise to OU covariance removes singularity
in exact arithmetic, even with repeated epochs. The `[f1, f1-f2]` covariance is
an invertible linear transformation of independent positive antenna noise.
Neither argument guarantees a successful floating-point factorization for
extreme scales. No failed factorization is unwrapped, silently regularized, or
classified as impossible.

The local LM driver retains Numeris 0.6.0's scaled normal equations, damping
updates, and accepted-step convergence criteria. It evaluates residuals and
Jacobians together into owned current/trial buffers, validates their shape and
finiteness, and returns errors directly. Negative-drag trial points are rejected
before evaluation. Iteration-limit RF recovery recomputes the best parameters'
evaluation and requires the existing independent QR convergence check.

`Evaluation` is caller-owned scratch. After an evaluation error its contents are
unspecified; consume it only after another successful evaluation. A failed cache
refresh invalidates the cache until a subsequent evaluation succeeds.

## Dependency audit and allowed copies

Audited versions are **Numeris 0.6.0** and **Satkit 0.24.1**, both pinned.
The enabled graph has Numeris `std`, `alloc`, `optim`, `estimate`, `ode`, and
`serde`; it does not enable Rayon. The matrix, factorization, EKF/RTS, and orbit
propagation paths used here are sequential. Tychonic does not dispatch parallel
numerical routines. SIMD is vector execution within a thread.

Numeris uses unsafe SIMD, scalar reinterpretation, and a linear-algebra helper
that obtains separate mutable column slices. Satkit uses unsafe gravity SIMD
and ephemeris byte decoding. Safe Tychonic callers still rely on those
implementations. Dependency assertions also remain outside the source-level
panic prohibition; Tychonic validates the dimensions and domains it supplies.
Dependency updates require renewed audit and regression qualification.

Satkit retains process-wide data caches and synchronization. Explicit
`update_datafiles` provisioning can spawn download workers. Provision data
before computation; `SATKIT_OFFLINE=1` uses cached files during tests and
qualification. The separate `tools` provisioning and allocation-profiling
executables are outside the numerical kernel contract and release package.
Its `PropSettings::precomputed` table also uses dependency-owned reference
counting; the allowed settings copy shares that table. This is included in the
Satkit trust boundary, alongside its process-wide caches.

The only source-level clone exceptions are external API boundaries:

| Function | Reason |
| --- | --- |
| STM `sample_arc` | Own Satkit settings to enable dense output without changing caller settings |
| `initial_state_from_tle` | Satkit SGP4 mutates its owned TLE cache |
| TCA `shifted_tle` | Produce an owned corrected TLE and clear its derived cache |
| RF builder `build` / `filter` | Each constructed estimator owns Satkit propagation settings |

Small scalar and fixed-size matrix values remain `Copy`. No prepared schedule,
evaluation, or estimator implements heap-copying `Clone`.

## Enforcement

The library uses `forbid(unsafe_code)` and Clippy prohibitions for panic extraction,
reference-counted/interior-mutable state, worker creation, and undocumented clones.
`python3 scripts/check_core_policy.py` additionally checks production source and
limits clone sites to the functions listed above. Test fixtures make explicit
copies outside the production contract. Compile-fail documentation tests check
borrow lifetimes and exclusive cache access; numerical regression tests check
Jacobians, whitening, fitting, covariance, gating, and rollback behavior.
See the [0.9.2 qualification report](qualification_0_9_2.md) for complexity
counts, performance measurements and release checks.
