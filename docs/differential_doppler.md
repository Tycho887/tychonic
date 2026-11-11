# Differential Doppler with local or remote receivers

The hypothetical measurement uses simultaneous, independently estimated carrier
frequencies from two antennas. The sign is **f1−f2**, consistent with the existing
baseline convention **B=r1−r2**. A common additive transmitter/receiver frequency
offset cancels. Separate unknown receiver offsets must be calibrated or modelled
before using this assumption with hardware.

Let \(\rho_i=r-r_i\), \(\ell_i=\rho_i/\|\rho_i\|\), and
\(\dot\rho_i=\ell_i^T(v-v_i)\). The instantaneous model is

\[
\Delta f=f_1-f_2={f_c\over c}(\dot\rho_2-\dot\rho_1).
\]

`differential_doppler_partial` evaluates the two geometric range-rate models and
returns their analytic difference. Its GCRF state derivative is

\[
H_{\Delta f}={f_c\over c}(H_{\dot\rho_2}-H_{\dot\rho_1}).
\]

For a rotating baseline, antenna 2 has \(r_2=r_1-B\) and
\(v_2=v_1-\omega\times B\). The baseline derivative, before conversion to ENU, is

\[
{\partial\Delta f\over\partial B}
={f_c\over c}\left(H_{\dot\rho_2,r}
+H_{\dot\rho_2,v}[\omega]_\times\right).
\]

`Interferometer::differential` rotates the surveyed ENU baseline and Earth
rotation vector into GCRF at the observation epoch and returns the baseline
derivative in ENU. Earth rotation is included in the velocity of antenna 2;
using the same velocity for both antennas would introduce a modelling error.
Measurements supply their ENU baseline explicitly. Batch inputs contain one
vector per measurement. RF calibration adds its constant ENU correction to each
supplied vector; it no longer creates a corrected interferometer.

For a short baseline, the approximate signal is

\[
\Delta f\simeq {f_c\over c}(B^T\dot\ell+\dot B^T\ell).
\]

This gives the intuition behind the phase-rate relationship. The implementation
uses the exact two-site geometric range-rate difference. The original
`Interferometer::new` keeps the reference's far-field phase model;
`with_spherical_phase()` opts into an exact geometric range difference. The new
experiment uses that option for both truth and estimation:

\[
D=\|\rho_1+B\|-\|\rho_1\|
={ (2\rho_1+B)^T B\over \|\rho_1\|+\|\rho_1+B\|},\quad
H_{D,r}=\ell_2^T-\ell_1^T,\quad H_{D,B}=\ell_2^T.
\]

The rationalized path avoids subtracting large ranges. The geometric identity
\(\dot D=\dot\rho_2-\dot\rho_1\) is exact. Phase retains the reference's
received-frequency factor; differentiating it also produces the small frequency
rate term. Differential Doppler is independently estimated from antenna
frequencies, not differentiated from the noisy phase samples.

For this 59 m baseline, the spherical-path correction can reach about 4 mm
(12° of carrier phase), significant relative to the smaller requested phase
uncertainties. The original phase-pass experiment retains its far-field model;
the combined differential experiment consistently generates and fits spherical
phase observations while reusing the original random errors.

## Measurement and calibration covariance

Assume independent antenna frequency errors \(e_1,e_2\) with variances
\(\sigma_1^2,\sigma_2^2\), and independently estimated white phase error. For
the ordered observations \([f_1,\phi,f_1-f_2]\), the covariance is

\[
R=\begin{bmatrix}
\sigma_1^2&0&\sigma_1^2\\
0&\sigma_\phi^2&0\\
\sigma_1^2&0&\sigma_1^2+\sigma_2^2
\end{bmatrix}.
\]

The ordinary and differential observations share antenna-1 noise. Treating them
as independent would count information incorrectly. This covariance makes the
frequency likelihood equivalent to observing both independent antenna
frequencies directly. In particular, even a zero baseline can improve the
ordinary frequency estimate through noise averaging, while adding no spatial
information.

The default experiment interprets **5 mm as independent 1σ errors on all three
ENU baseline-vector components**. The actual baseline error is drawn once per
trial and remains fixed. All cases estimate three constant calibration parameters
with that prior, retaining their correlation with the orbit. It is not added as
independent noise at every epoch. A baseline-length-only uncertainty would be a
different calibration prior; the present experiment includes orientation error.

At 2.23 GHz, 5 mm isotropic baseline uncertainty creates approximately 13.4°
instantaneous phase uncertainty before estimation. Its effect on differential
Doppler in this pass is only of order 0.001 Hz. Unknown constant phase mean and
baseline calibration are separately estimated; a constant phase offset cannot
absorb all the time dependence from a baseline-vector error.

The existing OU model remains a phase nuisance with a 6-hour correlation time
and 5° stationary amplitude. Its phase change over a 10 s gate corresponds to
roughly 0.00004 Hz. The experiment assumes independently estimated frequency
noise and does not differentiate the OU process. If frequency is derived from
the same phase samples, its covariance, gate averaging and offset dynamics need
to be modelled jointly; that derived measurement is not independent information.

## APIs and state layouts

- `DifferentialDopplerData` contains f1−f2 observations and antenna-2 frequency
  noise. Antenna-1 noise remains `PassiveRfData::sigma_doppler_hz`.
- `RfExtensions` optionally enables differential observations and a baseline
  calibration prior. `PassiveRfProblem::new_extended` performs full joint
  whitening, including shared Doppler noise and cross-epoch OU phase covariance.
  `initial_parameters()` builds the orbital/offset prior mean with zero baseline
  corrections. Batch calibration states follow frequency offset and phase mean:
  `[orbit, frequency offset, phase mean, delta B_E, delta B_N, delta B_U]`.
- `PassiveRfEkf::new_extended` adds the three constant ENU calibration states
  after the OU state. Use `<12>` for fixed drag and `<13>` for estimated drag.
  Without calibration states the existing `<9>/<10>` layouts remain available.
  `filter_data_extended` accepts optional differential observations. Streaming
  `update` uses dimensions 1 (Doppler), 2 (Doppler/phase), or 3
  (Doppler/phase/differential) with a full covariance supplied by the caller.
- The old RF constructors and phase-only APIs keep their previous behavior.
  Baseline uncertainty is included in the reported orbital marginal covariance.

## Reproduction and interpretation

Create the original paired phase-pass input if it is not already available, then:

```sh
SATKIT_OFFLINE=1 cargo run --manifest-path tools/Cargo.toml --locked --release --bin differential_pass -- \
  --input artifacts/phase_pass --trials 100 --baseline-sigma-mm 5 \
  --output artifacts/differential_pass
python3 tools/scripts/plot_differential_pass.py artifacts/differential_pass
```

The executable reuses the source truth, orbital priors, phase errors, phase
offsets, OU trajectories and antenna-1 frequency errors. It draws independent
antenna-2 errors and baseline calibration errors from a separate seeded stream.
It compares phase plus antenna-1 Doppler, phase plus differential Doppler and
ordinary Doppler, and an ideal co-located second-receiver averaging control.
The control isolates noise averaging without spatial information; it is not a
raw average of the two separated antennas' frequencies.

The default per-antenna noise remains 0.05 m/s equivalent, or about 0.372 Hz at
2.23 GHz. Differential noise is about 0.526 Hz. The representative 59 m-baseline
pass has a differential signal between 0.138 and 7.469 Hz. A measurable signal
does not by itself imply a precise orbital constraint: its response to the small
remaining orbit errors can be much less than its measurement noise.

Use `--noise-scale 0.1` or `0.01` for paired hypothetical per-antenna precisions
of 0.005 or 0.0005 m/s, or `--antenna2-sigma-rr` for unequal antenna noise. All
case comparisons use the same noise realizations and calibration errors.
Outputs include input/configuration provenance, all observations, reference/final
state estimates and covariance matrices, per-trial metrics, and summary plots.
Batch endpoints are at the reference epoch; EKF endpoints are at the final epoch.
Final-epoch error metrics propagate the batch orbital marginal with the STM.

These are internally consistent geometric simulations, with shared/calibrated
frequency reference, known carrier, fixed drag, no cycle slips, and no propagation
or atmospheric model mismatch. They test potential information content rather
than demonstrating a practical receiver's achievable precision.

The saved 100-trial experiment found 239–321 m batch position RMSE with phase
and antenna-1 Doppler, versus 223–311 m after adding differential Doppler. The
co-located averaging control achieved essentially the same result: the spatial
measurement added no meaningful gain in the tested geometry. Paired sweeps at
0.005 and 0.0005 m/s per antenna gave the same qualitative outcome. These
numbers include the newly introduced baseline-vector uncertainty and are not
directly comparable to the earlier experiment with an exactly known baseline.

To regenerate the precision-sweep report, run `differential_pass` with
`--noise-scale 0.1 --output artifacts/differential_pass_precision_005` and
`--noise-scale 0.01 --output artifacts/differential_pass_precision_0005`, then:

```sh
python3 tools/scripts/plot_differential_pass.py artifacts/differential_pass \
  --precision-study artifacts/differential_pass_precision_005 \
  --precision-study artifacts/differential_pass_precision_0005
```

## Tromsø–Svalbard baseline

The `--scenario tromso-svalbard` option generates a new hypothetical polar
pass, retains the local 59 m east–west phase baseline at Tromsø, and adds
frequency difference to a third antenna near Longyearbyen, Svalbard. The
illustrative coordinates are 69.662° N / 18.940° E / 100 m and
78.230° N / 15.410° E / 450 m ellipsoid height. These are nominal sites,
not surveyed antenna phase centers. Their Earth-fixed chord is 961.1 km.

The initial orbit is circular at approximately 600 km altitude, inclined
97.6°, and passes over the station-pair midpoint on 2024-01-01 around noon UTC.
The existing low-inclination source orbit cannot provide this shared northern
pass. Both stations must exceed 10° elevation: 37 samples remain at 10 s
cadence, spanning 360 s. All comparison cases use that same shared schedule.

`GroundStation::baseline_enu_to` computes the **exact chord r1−r2** in
antenna-1 ENU, including Earth curvature and both station heights. It has a
large up component and must not be replaced by a flat northward surface
distance. The existing exact differential kernel supports this chord without
a short-baseline approximation. Independent station-geometry tests check
the reconstructed remote position, Earth-rotation velocity, range rates and
analytic derivatives at multiple epochs.

`RfExtensions::differential_interferometer` selects a separate Doppler model,
with its required `differential_baselines_enu_m` sequence,
anchored at the same antenna 1 as local phase. If omitted, the previous shared
phase/Doppler baseline behavior remains. `differential_baseline_sigma_m` adds
three independent constant calibration parameters after the local correction,
with Gaussian priors. The scenario draws 5 mm 1σ on each ENU component of each
baseline, independently, with Tromsø antenna 1 fixed. Both sets of calibration
states are retained in every comparison case. Frequency covariance is unchanged:
ordinary Tromsø Doppler and its difference with Svalbard still share Tromsø noise.

For EKF use `PassiveRfEkf::new_with_calibration` and
`filter_data_with_extensions`; the dimensions are `<15>` with fixed drag or
`<16>` with estimated drag when both baselines are calibrated. State order is
`[orbit, frequency offset, phase mean, OU drift, local delta B_ENU, remote delta B_ENU]`.
The batch order omits OU drift, which remains analytically marginalized.
Streaming `update_with_differential_baseline` accepts the separate Doppler
interferometer paired with its ENU baseline vector and the full observation
covariance. Remote calibration can also be enabled with a known local phase baseline, using `<12>/<13>`.

Reproduce the paired study with:

```sh
SATKIT_OFFLINE=1 cargo run --manifest-path tools/Cargo.toml --locked --release --bin differential_pass -- \
  --scenario tromso-svalbard --trials 100 \
  --output artifacts/differential_tromso_svalbard
SATKIT_OFFLINE=1 cargo run --manifest-path tools/Cargo.toml --locked --release --bin differential_pass -- \
  --input artifacts/differential_tromso_svalbard/source_inputs --trials 100 \
  --output artifacts/differential_tromso_short
python3 tools/scripts/plot_differential_pass.py artifacts/differential_tromso_short
python3 tools/scripts/plot_differential_pass.py artifacts/differential_tromso_svalbard \
  --baseline-study artifacts/differential_tromso_short
```

The scenario generates compatible input truth, priors and noise under
`source_inputs`. Remote calibration uses a separate random stream, so the
comparison reuses identical orbital priors, local calibration errors, antenna-1
and antenna-2 frequency errors, white phase errors and OU trajectories.
Without an explicit output option the regional scenario writes to
`artifacts/differential_tromso_svalbard`. The generated scenario and `--input`
are mutually exclusive. Earlier source inputs and experiment artifacts remain
available. Receiver-noise scaling and baseline uncertainty options also apply.

The saved 100-trial study uses 0.05 m/s independent per-receiver noise,
equivalent to 0.526 Hz differential 1σ. The differential signal magnitude
spans about 6.9–64.1 kHz. Final position RMSE is:

| Phase 1σ | Short-baseline batch (m) | Tromsø–Svalbard batch (m) | Tromsø–Svalbard EKF (m) |
|---|---:|---:|---:|
| 0.1° | 156.83 | 54.78 | 55.64 |
| 1° | 164.97 | 63.68 | 63.87 |
| 5° | 173.65 | 70.95 | 71.18 |
| 10° | 180.65 | 78.59 | 78.90 |

The remote receiver improves batch position RMSE by 56–65% beyond the
co-located averaging control, showing additional geometric information in
this pass. The short-baseline results here use the same polar pass and cannot
be directly compared with the earlier low-inclination short-baseline geometry.
No solver or RTS failures occurred. The report includes velocity errors,
covariance consistency metrics, signal and shared-visibility plots, and
the paired baseline comparison.

This remains an instantaneous geometric feasibility study. Long-baseline
carrier phase is not used; only the local phase must stay continuously
unwrapped. Receiver frequency offsets are assumed calibrated, and observations
have simultaneous reception tags. Light time, different transmitter emission
epochs, transmitter frequency variation, receiver clock offsets and atmospheric
delays are omitted. Those effects need modelling for an actual continental
link, especially at the assumed sub-Hz precision. The simulation demonstrates
potential geometric information under these assumptions.
