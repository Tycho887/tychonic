# Migrating to 0.9.7

Baselines now belong to phase-difference and differential-Doppler measurements.
`Interferometer` retains the nominal carrier and phase-model settings. All
baselines use **antenna 1 minus antenna 2**, expressed in antenna 1's ENU frame
in metres.

## Kernels and individual measurements

Replace `Interferometer::new(baseline, carrier_hz)` with
`Interferometer::new(carrier_hz)`. Supply `&baseline` after station geometry in
`evaluate` and `differential`:

```rust,ignore
let interferometer = Interferometer::new(2.23e9)?;
let baseline = v3(59.0, 3.0, 0.2);
let phase = interferometer.evaluate(&state, &station_geometry, &baseline, 0.0)?;
let differential = interferometer.differential(
    &state, &station_geometry, &baseline, time,
)?;
```

Named `phase_difference_sensor` and `differential_doppler_sensor` constructors
accept a baseline vector after the interferometer. `PhaseSensor` literals gain
`baseline_enu_m: V3`. `PassiveRfEkf::update` takes `&baseline` after the
interferometer; `update_with_differential_baseline` also accepts an optional
`(Interferometer, V3)` pair for the separate differential measurement.

The raw GCRF phase and differential kernels retain their existing signatures.
`baseline_enu()` and `corrected()` are removed; keep your surveyed vector and
apply any known correction to that vector directly.

## Batch measurements

Named phase and differential batch functions, corresponding `PreparedBatch`
constructors, and `PassiveRfData::new` require `Vec<V3>` after the interferometer.
The vector must contain exactly one baseline per measurement, in the original
geometry and observation order. Fixed baselines use repeated entries:

```rust,ignore
let baselines = vec![baseline; geometry.len()];
let batch = measurements::phase_difference(
    &geometry, interferometer, baselines, phase_values_rad, sigma_rad,
)?;
```

Supply varying vectors in the same way. Sorting observations for filtering or
sharing propagated epochs across batches preserves each measurement's baseline,
including when timestamps repeat. Zero baselines remain valid; nonfinite vectors,
unrepresentable norms, and mismatched lengths return input errors.

## Joint measurements and RF calibration

`Observable<B = V3>` gains `baseline_enu_m: B` on both RF variants. Individual
sensors use `Observable<V3>`; `measurements::joint` uses
`Observable<Vec<V3>>`. Each RF observable supplies its own sequence. Existing
scalar configurations can become fixed-baseline batch configurations with:

```rust,ignore
let batch_models = sensor_models.into_iter()
    .map(|model| model.map_baseline(|b| vec![b; geometry.len()]))
    .collect();
```

`PassiveRfData` stores the explicit phase baseline sequence, also used by the
existing shared phase/differential RF combination. A separate
`RfExtensions::differential_interferometer` requires a matching
`differential_baselines_enu_m: Some(Vec<V3>)`. Both fields must be supplied or
both omitted. `RfBuilder::differential_doppler` accepts the separate model and
sequence as `Some((interferometer, baselines))`; `None` shares the phase sequence.

Baseline calibration continues to estimate one constant ENU correction per
stream, added to every supplied vector. Independently calibrated differential
streams retain separate correction states. This calibration behavior is
provisional; independently estimated corrections per measurement are not added.

Baselines are known instantaneous geometry. Additional antenna motion and
baseline-rate inputs are outside this change; differential Doppler retains its
existing Earth-rotation velocity model. Varying a baseline does not reset or
unwrap phase observations: callers still provide continuous, calibrated phase.
