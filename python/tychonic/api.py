"""High-level Tychonic orbit estimation and acquisition API.

Public epoch labels are UTC RFC 3339 strings. Numeric columns can enter as
POSIX seconds via ContactSamples.from_unix_seconds. Doppler is the shift in Hz
(negative for a receding spacecraft). Pointing angles are radians, east of north
for azimuth. Numerical fits, propagation and filter updates use the native extension.

Fit historical contacts, then maintain one EKF per spacecraft. Here history
and next_contact are JobData instances. The TLE covariance model is recorded
on the seed. For replay, use the seed only as the optimizer's starting guess.

    forces = ForceModel()
    seed = StateEstimate.from_tle(tle, at=first_observation, uncertainty=model)
    problem = ODProblem(history, initial_guess=seed, force_model=forces,
                        profile=CalibratedLM(covariance=True))
    fit = problem.solve(output_epoch=publication_epoch)
    orbit = fit.estimate.orbital_marginal()
    state_km, covariance_km = orbit.state_si / 1e3, orbit.covariance_si / 1e6
    catalogue = EnsembleFilter.ekf(force_model=forces, retain_history=True)
    catalogue.add_spacecraft(history.spacecraft_id, fit.estimate,
                             last_contributing_epoch=fit.last_contributing_epoch,
                             bias_prior_hz=GaussianPrior(mean=0.0, sigma=5.0))
    saved = catalogue.checkpoint(history.spacecraft_id)
    resumed = EnsembleFilter.ekf(force_model=forces, retain_history=True)
    resumed.restore(saved)
    updates = resumed.update(next_contact)
    smoothed = resumed.smooth(history.spacecraft_id)

The first integration case uses the two-contact fixture in tests/covariance.rs.
Use its Cartesian seed, fixed drag and LM settings. Fit readings 0–38 with
their full joint covariance and publish at reading 38.
Feed readings 39–41 to both catalogues. Compare full states, bias cross terms,
reading cursors, contributing epochs and the epoch-aligned smoother outputs.

Track two contacts against the same TLE. Each job supplies joint Doppler,
azimuth and elevation readings. The caller supplies the TLEs and handoff epoch.

    ukf = EnsembleFilter.tle_epoch_ukf(
        noise=TleProcessNoise(offset_psd_s2_s=0.01, bias_psd_hz2_s=0.04,
                              max_offset_sigma_s=60.0, max_bias_sigma_hz=20.0))
    prior = TleCorrection(tle, start, epoch_offset_s=0.0, doppler_bias_hz=0.0,
                          covariance=np.diag([30.0**2, 5.0**2]))
    ukf.add_spacecraft(first_contact.spacecraft_id, prior)
    ukf.update(first_contact)
    ukf.update(second_contact)
    ukf.replace_tle(first_contact.spacecraft_id, new_tle, epoch=handoff,
                    offset_prior_s=GaussianPrior(mean=0.0, sigma=15.0))

Plan one acquisition cycle from a single TLE. The caller executes the first
command and reports a miss only after completing its listening interval.

    finder = SatFinder.from_tle(tle, setup=SearchSetup(station, start),
                                n_particles=160, rng=np.random.default_rng(42))
    action = finder.plan().actions[0]
    acquired = execute_and_listen(action)
    finder.observe(acquired=acquired)
    belief = finder.read_state()
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import KW_ONLY, dataclass, field, replace
from datetime import UTC, datetime
from math import radians
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING, Generic, Literal, TypeVar
from uuid import UUID

import numpy as np
from numpy.typing import NDArray

from . import _catalogue, _exports, _native, _profiles, _search

if TYPE_CHECKING:
    import polars as pl

Epoch = str
TLE = tuple[str, str]
OMM = Mapping[str, str | float | int]
FloatArray = NDArray[np.float64]


def _check_shape(name: str, value: object, *allowed: tuple[int, ...]) -> None:
    shape = np.shape(value)
    if shape not in allowed:
        raise ValueError(f"{name} has shape {shape}. Expected one of {allowed}.")


class _UnixEpochs(Sequence[str]):
    """Keep columnar POSIX times numeric until a public UTC label is requested."""

    def __init__(self, seconds):
        self.seconds = np.array(seconds, dtype=np.float64, copy=True)
        self.seconds.flags.writeable = False

    def __len__(self):
        return len(self.seconds)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return _UnixEpochs(self.seconds[index])
        return (
            datetime.fromtimestamp(float(self.seconds[index]), UTC)
            .isoformat(timespec="microseconds")
            .replace("+00:00", "Z")
        )


@dataclass(frozen=True)
class GroundStation:
    """WGS84 geodetic coordinates and antenna identifier."""

    antenna_id: str
    latitude_deg: float
    longitude_deg: float
    altitude_m: float


@dataclass(frozen=True)
class ContactSamples:
    """One antenna and contact. RF channels require a carrier in Hz.

    A streamed row has a source-stable observation_id within its contact and
    antenna. Reuse the ID on replay, including for readings sharing an epoch.
    """

    contact_id: UUID
    station: GroundStation
    epochs: Sequence[Epoch]
    carrier_hz: float | None = None
    lock_state: NDArray[np.bool_] | None = None
    ebn0_db: FloatArray | None = None
    observation_ids: Sequence[str] | None = None

    @classmethod
    def from_unix_seconds(
        cls,
        contact_id: UUID,
        station: GroundStation,
        epoch_unix_s: FloatArray,
        *,
        carrier_hz: float | None = None,
        observation_ids: Sequence[str] | None = None,
        lock_state: NDArray[np.bool_] | None = None,
        ebn0_db: FloatArray | None = None,
    ) -> ContactSamples:
        """Ingest a numeric POSIX-seconds column at native microsecond resolution.

        Group columns by contact, antenna, carrier and available channels. POSIX
        time omits leap seconds. Use explicit UTC epoch labels for those readings.
        """
        seconds = np.asarray(epoch_unix_s)
        if (
            seconds.ndim != 1
            or seconds.dtype.kind not in "iuf"
            or not np.isfinite(seconds).all()
        ):
            raise ValueError("epoch_unix_s must be a finite numeric vector in seconds")
        epochs = _UnixEpochs(seconds)
        return cls(
            contact_id,
            station,
            epochs,
            carrier_hz=carrier_hz,
            observation_ids=observation_ids,
            lock_state=lock_state,
            ebn0_db=ebn0_db,
        )

    def __post_init__(self) -> None:
        for name in ("lock_state", "ebn0_db", "observation_ids"):
            values = getattr(self, name)
            if values is not None:
                _check_shape(name, values, (len(self.epochs),))
        if self.carrier_hz is not None and not (
            np.isfinite(self.carrier_hz) and self.carrier_hz > 0
        ):
            raise ValueError("carrier_hz must be finite and positive")


@dataclass(frozen=True, kw_only=True)
class ObservationBatch:
    """Each supplied observable has one value per epoch in samples.

    Covariance follows the field order from doppler_hz through elevation_rad,
    skipping absent columns. Supply (M,M) or (N,M,M) for N epochs and M
    components. Supply azimuth and elevation together. Each interferometric
    channel takes its own antenna-1-minus-antenna-2 ENU baseline in metres,
    either a shared (3,) vector or (N,3) in sample order. Rotate GCRF baseline
    vectors into antenna 1's local ENU at each observation epoch before ingestion.
    Absolute range (ranging) has no observation channel in this API.
    """

    samples: ContactSamples
    covariance: FloatArray
    doppler_hz: FloatArray | None = None
    phase_difference_rad: FloatArray | None = None
    differential_doppler_hz: FloatArray | None = None
    azimuth_rad: FloatArray | None = None
    elevation_rad: FloatArray | None = None
    phase_baseline_enu_m: FloatArray | None = None
    differential_baseline_enu_m: FloatArray | None = None

    def __post_init__(self) -> None:
        rows = len(self.samples.epochs)
        components = (
            self.doppler_hz,
            self.phase_difference_rad,
            self.differential_doppler_hz,
            self.azimuth_rad,
            self.elevation_rad,
        )
        columns = [values for values in components if values is not None]
        if self.samples.carrier_hz is None and any(
            v is not None for v in components[:3]
        ):
            raise ValueError("RF observations require a carrier_hz for this contact")
        if not rows or not columns:
            raise ValueError("a batch needs at least one epoch and one observable")
        if (self.azimuth_rad is None) != (self.elevation_rad is None):
            raise ValueError("supply azimuth and elevation together")
        for values in columns:
            _check_shape("observation column", values, (rows,))
        size = len(columns)
        _check_shape("covariance", self.covariance, (size, size), (rows, size, size))
        self._check_baseline(self.phase_difference_rad, self.phase_baseline_enu_m)
        self._check_baseline(
            self.differential_doppler_hz, self.differential_baseline_enu_m
        )

    def _check_baseline(
        self, channel: FloatArray | None, baseline: FloatArray | None
    ) -> None:
        if (channel is None) != (baseline is None):
            raise ValueError(
                "supply each interferometric channel with its ENU baseline"
            )
        if baseline is not None:
            _check_shape(
                "baseline_enu_m", baseline, (3,), (len(self.samples.epochs), 3)
            )


def _variance(sigma: float | FloatArray) -> FloatArray:
    """Shared or per-epoch covariance for a scalar observable."""
    values = np.asarray(sigma, dtype=np.float64)
    if values.ndim > 1 or not np.all(np.isfinite(values) & (values > 0)):
        raise ValueError("supply a positive finite sigma or a vector of sigmas")
    return np.square(values)[..., None, None]


@dataclass
class JobData:
    """One spacecraft's raw observations. Each batch has its own schedule.

    For example, append simultaneous Doppler and pointing with their joint noise.

        samples = ContactSamples.from_unix_seconds(contact_id, station, epoch_unix_s,
                                                   carrier_hz=2.2e9,
                                                   observation_ids=observation_ids)
        job = JobData(spacecraft_id)
        job.add_joint(ObservationBatch(samples=samples, doppler_hz=doppler,
                      azimuth_rad=az, elevation_rad=el, covariance=covariance))
    """

    spacecraft_id: UUID
    spacecraft_name: str | None = None
    batches: list[ObservationBatch] = field(default_factory=list, init=False)

    def add_joint(self, batch: ObservationBatch) -> None:
        """Append simultaneous channels, preserving their order and covariance."""
        self.batches.append(batch)

    def add_doppler(
        self,
        samples: ContactSamples,
        doppler_hz: FloatArray,
        *,
        sigma_hz: float | FloatArray,
    ) -> None:
        """Add absolute Doppler. Use add_joint for simultaneous soft az/el."""
        self.add_joint(
            ObservationBatch(
                samples=samples,
                doppler_hz=doppler_hz,
                covariance=_variance(sigma_hz),
            )
        )

    def add_phase_difference(
        self,
        samples: ContactSamples,
        phase_rad: FloatArray,
        *,
        baseline_enu_m: FloatArray,
        sigma_rad: float | FloatArray,
    ) -> None:
        """Add continuous, unwrapped interferometric phase difference."""
        self.add_joint(
            ObservationBatch(
                samples=samples,
                phase_difference_rad=phase_rad,
                phase_baseline_enu_m=baseline_enu_m,
                covariance=_variance(sigma_rad),
            )
        )

    def add_differential_doppler(
        self,
        samples: ContactSamples,
        differential_hz: FloatArray,
        *,
        baseline_enu_m: FloatArray,
        sigma_hz: float | FloatArray,
    ) -> None:
        """Add frequency at antenna 1 minus frequency at antenna 2."""
        self.add_joint(
            ObservationBatch(
                samples=samples,
                differential_doppler_hz=differential_hz,
                differential_baseline_enu_m=baseline_enu_m,
                covariance=_variance(sigma_hz),
            )
        )

    def to_polars(self) -> pl.DataFrame:
        """One row per batch and epoch, with IDs, telemetry and measurement columns.

        Persist with job.to_polars().write_parquet(path).
        """
        return _exports.to_polars(self)

    def export_to_ccsds_tdm(
        self,
        directory: Path,
        *,
        originator: str,
        creation_epoch: Epoch,
        spacecraft_participant: str,
        antenna_participants: Mapping[str, str],
    ) -> tuple[Path, ...]:
        """Write Doppler TDM 2.0 KVN files, grouped by contact and antenna.

        Map antenna_id to its CCSDS participant. Use each batch's carrier to
        convert signed Hz to DOPPLER_INSTANTANEOUS in km/s. Return written paths.
        Values use the single-pass downlink range-rate convention.
        """
        return _exports.write_tdm(
            self,
            directory,
            originator,
            creation_epoch,
            spacecraft_participant,
            antenna_participants,
        )


def _batch(batch):
    samples = batch.samples
    models, columns = [], []
    baseline_columns = {}
    for kind, values, baseline in (
        ("doppler", batch.doppler_hz, None),
        ("phase_difference", batch.phase_difference_rad, batch.phase_baseline_enu_m),
        (
            "differential_doppler",
            batch.differential_doppler_hz,
            batch.differential_baseline_enu_m,
        ),
    ):
        if values is not None:
            model = {"kind": kind, "carrier_hz": samples.carrier_hz}
            if baseline is not None:
                b = np.asarray(baseline)
                if b.ndim == 1:
                    model["baseline_enu_m"] = b.tolist()
                else:
                    baseline_columns[kind] = np.asarray(b, dtype=float)
            models.append(model)
            columns.append(values)
    if batch.azimuth_rad is not None:
        models.append({"kind": "azimuth_elevation"})
        columns.extend([batch.azimuth_rad, batch.elevation_rad])
    values = np.column_stack(columns).astype(np.float64, copy=False)
    n, m = values.shape
    covariance = np.broadcast_to(np.asarray(batch.covariance, dtype=float), (n, m, m))
    site = samples.station
    epochs = samples.epochs
    return _native.NativeBatch(
        epochs.seconds if isinstance(epochs, _UnixEpochs) else epochs,
        [(site.latitude_deg, site.longitude_deg, site.altitude_m)],
        json.dumps(models),
        values,
        covariance,
        baseline_columns.get("phase_difference"),
        baseline_columns.get("differential_doppler"),
    )


@dataclass(frozen=True, kw_only=True)
class ForceModel:
    """Satkit STM-compatible propagation settings. Subclasses can supply presets."""

    gravity_model: str = "EGM2008"
    gravity_degree: int = 4
    gravity_order: int = 4
    sun_gravity: bool = True
    moon_gravity: bool = True
    use_space_weather: bool = True
    fixed_drag_m2_kg: float = 0.0  # Cd A/m
    integrator: str | None = None  # None selects the native OD integrator.
    abs_tolerance: float = 1e-10
    rel_tolerance: float = 1e-13

    def _config(self, epoch, drag=False):
        settings = dict(
            gravity_model=self.gravity_model,
            gravity_degree=self.gravity_degree,
            gravity_order=self.gravity_order,
            use_sun_gravity=self.sun_gravity,
            use_moon_gravity=self.moon_gravity,
            use_spaceweather=self.use_space_weather,
            abs_error=self.abs_tolerance,
            rel_error=self.rel_tolerance,
        )
        if self.integrator is not None:
            settings["integrator"] = self.integrator
        return _native.NativePropagation(
            epoch, json.dumps(settings), drag, self.fixed_drag_m2_kg, 1e-4
        )


@dataclass(frozen=True)
class GaussianPrior:
    """Mean and one standard deviation in the units of the receiving field."""

    mean: float
    sigma: float


@dataclass(frozen=True, kw_only=True)
class TleAgeUncertainty:
    """Empirical RTN uncertainty recipe, with caller-supplied calibration.

    Each vector contains six entries: radial, along-track and cross-track
    position (m), then velocity (m/s) on the same axes. Each sigma equals
    sigma_at_epoch + abs(age_days) * sigma_growth_per_day. At the requested
    epoch resolve the independent errors into GCRF using the RTN basis for
    both position and velocity. Store the coefficients with the model version.
    """

    sigma_at_epoch: tuple[float, ...]
    sigma_growth_per_day: tuple[float, ...]
    name: Literal["rtn_linear_sigma"] = field(init=False, default="rtn_linear_sigma")
    version: Literal[1] = field(init=False, default=1)

    def __post_init__(self) -> None:
        for name in ("sigma_at_epoch", "sigma_growth_per_day"):
            values = np.asarray(getattr(self, name))
            _check_shape(name, values, (6,))
            if not np.all(np.isfinite(values) & (values >= 0)):
                raise ValueError(f"{name} must contain finite nonnegative values")
        if np.any(np.asarray(self.sigma_at_epoch) == 0):
            raise ValueError("sigma_at_epoch must be positive")


@dataclass(frozen=True)
class TleInitialization:
    reference_tle: TLE
    uncertainty: TleAgeUncertainty


class FitError(RuntimeError):
    """The native solver could not produce the requested fit."""


class CovarianceError(FitError):
    """The requested parameter covariance is unavailable."""


class RankDeficientError(CovarianceError):
    """Rank of the whitened system and number of parameters at the solution."""

    def __init__(self, rank: int, parameters: int):
        super().__init__(f"rank {rank} for {parameters} fitted parameters")
        self.rank = rank
        self.parameters = parameters


@dataclass(frozen=True)
class OrbitalMarginal:
    """Publishable GCRF [x,y,z,vx,vy,vz] in m and m/s, with 6x6 covariance."""

    epoch: Epoch
    state_si: FloatArray
    covariance_si: FloatArray

    def __post_init__(self) -> None:
        _check_shape("state_si", self.state_si, (6,))
        _check_shape("covariance_si", self.covariance_si, (6, 6))


@dataclass(frozen=True)
class StateEstimate:
    """GCRF state at epoch. A point fit may omit covariance.

    Covariance axes are x, y, z (m), vx, vy, vz (m/s), then drag_m2_kg and
    doppler_bias_hz when present, in that order. The EKF snapshot includes
    its bias. Its fixed drag is a propagation property.
    """

    epoch: Epoch
    position_m: FloatArray
    velocity_m_s: FloatArray
    _: KW_ONLY
    covariance: FloatArray | None = None
    drag_m2_kg: float | None = None
    doppler_bias_hz: float | None = None
    tle_initialization: TleInitialization | None = None

    @classmethod
    def from_tle(
        cls, tle: TLE, *, at: Epoch, uncertainty: TleAgeUncertainty
    ) -> StateEstimate:
        """Initialize via native SGP4/GCRF at 'at' with a 6x6 model covariance.

        Record the exact input TLE and uncertainty recipe in tle_initialization.
        """
        state, days = _native.tle_seed("\n".join(tle), at)
        r, v = np.array(state).reshape(2, 3)
        radial = r / np.linalg.norm(r)
        normal = np.cross(r, v)
        normal /= np.linalg.norm(normal)
        basis = np.column_stack([radial, np.cross(normal, radial), normal])
        sigma = np.array(uncertainty.sigma_at_epoch) + abs(days) * np.array(
            uncertainty.sigma_growth_per_day
        )
        rotation = np.kron(np.eye(2), basis)
        return cls(
            at,
            r,
            v,
            covariance=(rotation * sigma**2) @ rotation.T,
            tle_initialization=TleInitialization(tle, uncertainty),
        )

    @property
    def covariance_order(self) -> tuple[str, ...]:
        names: tuple[str, ...] = ("x_m", "y_m", "z_m", "vx_m_s", "vy_m_s", "vz_m_s")
        if self.drag_m2_kg is not None:
            names += ("drag_m2_kg",)
        if self.doppler_bias_hz is not None:
            names += ("doppler_bias_hz",)
        return names

    def __post_init__(self) -> None:
        _check_shape("position_m", self.position_m, (3,))
        _check_shape("velocity_m_s", self.velocity_m_s, (3,))
        if self.covariance is not None:
            size = len(self.covariance_order)
            _check_shape("covariance", self.covariance, (size, size))

    def orbital_marginal(self) -> OrbitalMarginal:
        """Copy the Cartesian state and its marginal covariance at the same epoch."""
        if self.covariance is None:
            raise CovarianceError("publication requires a covariance-bearing estimate")
        return OrbitalMarginal(
            self.epoch,
            np.concatenate((self.position_m, self.velocity_m_s)),
            self.covariance[:6, :6].copy(),
        )

    def _parameters(self, drag=False):
        orbit = np.concatenate((self.position_m, self.velocity_m_s))
        if drag:
            if self.drag_m2_kg is None:
                raise ValueError("a drag fit needs an initial drag value")
            return np.append(orbit, self.drag_m2_kg)
        return orbit

    def _export_args(self, epochs, forces):
        if self.drag_m2_kg is not None:
            forces = replace(forces, fixed_drag_m2_kg=self.drag_m2_kg)
        return (forces._config(self.epoch), self._parameters(), epochs)

    def to_tle(self, epochs: Sequence[Epoch], *, force_model: ForceModel) -> TLE:
        """Fit SGP4 mean elements to the Cartesian arc sampled at epochs."""
        return tuple(_native.mean_elements(*self._export_args(epochs, force_model))[0])

    def to_omm(self, epochs: Sequence[Epoch], *, force_model: ForceModel) -> OMM:
        """Return fitted SGP4 mean elements as an OMM mapping for this arc."""
        return json.loads(
            _native.mean_elements(*self._export_args(epochs, force_model))[1]
        )

    def to_oem(
        self,
        epochs: Sequence[Epoch],
        *,
        force_model: ForceModel,
        object_name: str,
        object_id: str,
        originator: str,
    ) -> str:
        """Return CCSDS OEM text using GCRF Cartesian samples and UTC epochs."""
        return _exports.oem(
            self, epochs, force_model, object_name, object_id, originator
        )


@dataclass(frozen=True)
class TleCorrection:
    """Estimate at the observation epoch. Covariance axes are offset (s), bias (Hz)."""

    reference_tle: TLE
    epoch: Epoch
    _: KW_ONLY
    epoch_offset_s: float
    doppler_bias_hz: float
    covariance: FloatArray

    def __post_init__(self) -> None:
        _check_shape("covariance", self.covariance, (2, 2))

    def _filter(self):
        return _native.NativeTcaUKF(
            "\n".join(self.reference_tle),
            self.epoch,
            np.array([self.epoch_offset_s, self.doppler_bias_hz], dtype=float),
            np.asarray(self.covariance, dtype=float),
            True,
        )

    def to_tle(self) -> TLE:
        """Shift the reference TLE epoch by epoch_offset_s, retaining mean elements."""
        return tuple(self._filter().corrected_elements()[0])

    def to_omm(self) -> OMM:
        """Return corrected mean elements at the precision of the internal epoch."""
        return json.loads(self._filter().corrected_elements()[1])


@dataclass(frozen=True, kw_only=True)
class OptimizerProfile:
    """Base for reusable solver profiles. None selects the solver's iteration limit."""

    max_iterations: int | None = None


@dataclass(frozen=True, kw_only=True)
class CalibratedLM(OptimizerProfile):
    """covariance=True guarantees full state covariance on success.

    The measurement-only fit uses absolute whitened information. Supplying a
    weighted_prior requests a Gaussian MAP fit and posterior covariance.
    """

    fit_drag: bool = False
    covariance: bool = True
    gradient_tolerance: float | None = None
    step_tolerance: float | None = None
    cost_tolerance: float | None = None


@dataclass(frozen=True, kw_only=True)
class ContactBiasFit(OptimizerProfile):
    """Project out one Doppler bias per contact, optionally using soft pointing."""

    fit_drag: bool = False


@dataclass(frozen=True, kw_only=True)
class PassiveRFFit(OptimizerProfile):
    """Fit RF offsets and OU phase drift using an explicit diagonal orbit prior."""

    frequency_offset_hz: GaussianPrior
    phase_offset_rad: GaussianPrior
    drift_correlation_s: float
    drift_sigma_rad: float
    fit_drag: bool = False
    phase_baseline_sigma_m: FloatArray | None = None
    differential_baseline_sigma_m: FloatArray | None = None


@dataclass(frozen=True, kw_only=True)
class ConstrainedFit(OptimizerProfile):
    """Fit orbit, drag and contact biases within hard circular pointing bounds."""

    full_fov_rad: float
    pointings_per_contact: int | Literal["all"] = 3


@dataclass(frozen=True, kw_only=True)
class FitResult:
    """Orbit estimate and diagnostics produced by the selected solver.

    For passive RF, rf_parameters contains the full fit in covariance-axis order.
    last_contributing_epoch is the newest measurement used in the objective or
    retained constraints, excluding held-out, filtered and unused pointing rows.
    Contact biases are keyed by contact UUID. A constrained fit is point-only.
    """

    estimate: StateEstimate
    last_contributing_epoch: Epoch
    used_weighted_prior: bool
    whitened_rss: float
    iterations: int
    covariance_method: str | None = None
    rank: int | None = None
    contact_bias_hz: Mapping[UUID, float] = field(default_factory=dict)
    rf_parameters: Mapping[str, float] = field(default_factory=dict)
    rf_covariance: FloatArray | None = None
    phase_drift_rad: FloatArray | None = None
    max_pointing_violation_rad: float | None = None

    @property
    def output_epoch(self) -> Epoch:
        return self.estimate.epoch


@dataclass(frozen=True)
class ODProblem:
    """initial_guess provides optimizer coordinates. Its covariance has zero weight.

    weighted_prior explicitly adds its independent Gaussian information once.
    For replay of observations already incorporated in an estimate, pass that
    estimate as initial_guess only. Supply a weighted prior at the guess epoch
    over exactly the orbit parameters being fitted. Calibrated LM supports a
    full covariance through a Cholesky-whitened prior residual.
    """

    data: JobData
    _: KW_ONLY
    initial_guess: StateEstimate
    force_model: ForceModel
    profile: OptimizerProfile
    weighted_prior: StateEstimate | None = None

    def __post_init__(self) -> None:
        prior = self.weighted_prior
        if prior is None:
            if isinstance(self.profile, PassiveRFFit):
                raise ValueError("PassiveRFFit requires an explicit weighted_prior")
            return
        if not isinstance(self.profile, (CalibratedLM, PassiveRFFit)):
            raise ValueError("this profile does not support a weighted orbit prior")
        if prior.covariance is None:
            raise ValueError("weighted_prior requires a covariance")
        if _utc_epoch_key(prior.epoch) != _utc_epoch_key(self.initial_guess.epoch):
            raise ValueError("weighted_prior and initial_guess must share an epoch")
        if prior.doppler_bias_hz is not None or (
            (prior.drag_m2_kg is not None) != self.profile.fit_drag
        ):
            raise ValueError("weighted_prior must cover exactly the fitted orbit state")
        _check_prior_covariance(
            prior.covariance, diagonal=isinstance(self.profile, PassiveRFFit)
        )

    def solve(self, *, output_epoch: Epoch | None = None) -> FitResult:
        """Return state and covariance at output_epoch (default: initial_guess.epoch).

        Transport the full fitted covariance with the variational transition,
        including drag sensitivities, before taking the orbital marginal.
        Deterministic transport adds zero process noise. RankDeficientError
        reports rank and parameter count when calibrated information is singular.
        """
        return _profiles.solve(self, output_epoch)


def _utc_epoch_key(label: Epoch) -> tuple[str, str]:
    """Compare exact UTC labels including leap-second and fractional-second fields."""
    utc = label.upper().removesuffix("Z").removesuffix("+00:00")
    second, _, fraction = utc.partition(".")
    return second, fraction.rstrip("0")


def _check_prior_covariance(covariance: FloatArray, *, diagonal: bool) -> None:
    if not np.isfinite(covariance).all() or not np.allclose(
        covariance, covariance.T, rtol=1e-12, atol=0.0
    ):
        raise ValueError("weighted_prior covariance must be finite and symmetric")
    if diagonal and np.any(covariance != np.diag(np.diag(covariance))):
        raise ValueError("PassiveRFFit supports a diagonal weighted orbit prior")
    try:
        np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as exc:
        raise ValueError("weighted_prior covariance must be positive definite") from exc


@dataclass(frozen=True, kw_only=True)
class OrbitProcessNoise:
    """White acceleration per axis and a random walk on additive Doppler bias."""

    acceleration_psd_m2_s3: float = 0.0
    bias_psd_hz2_s: float = 0.0


@dataclass(frozen=True, kw_only=True)
class TleProcessNoise:
    """Random-walk rates and caps on marginal standard deviations across contacts."""

    offset_psd_s2_s: float
    bias_psd_hz2_s: float
    max_offset_sigma_s: float
    max_bias_sigma_hz: float


@dataclass(frozen=True)
class ReadingRef:
    """Durable source identity. Equal epochs can refer to different readings."""

    contact_id: UUID
    antenna_id: str
    observation_id: str
    epoch: Epoch


@dataclass(frozen=True)
class FilterUpdate:
    """One source reading's zero-based indices and innovation test result."""

    batch_index: int
    row_index: int
    reading: ReadingRef
    accepted: bool
    normalized_innovation_squared: float


Estimate = TypeVar("Estimate", StateEstimate, TleCorrection)


@dataclass(frozen=True, kw_only=True)
class FilterCheckpoint:
    """Portable resume record for one object. Persist envelope and payload together.

    Payload includes the complete augmented state and covariance, effective fixed
    drag or reference TLE, model/noise/gate configuration and retained RTS history.
    last_processed_reading includes gated readings. last_accepted_observation
    advances only on correction. Both start empty for a batch seed.
    last_contributing_epoch starts at the seed's evidence epoch and advances
    with accepted data. Prediction alone advances only estimate_epoch.
    """

    version: int
    kind: Literal["orbit_ekf", "tle_epoch_ukf"]
    spacecraft_id: UUID
    estimate_epoch: Epoch
    last_contributing_epoch: Epoch | None
    last_processed_reading: ReadingRef | None
    last_accepted_observation: ReadingRef | None
    payload: bytes


@dataclass(frozen=True)
class SmoothedStep:
    """One RTS epoch, linkable by spacecraft and estimate.epoch.

    readings contains the original updates at this epoch, including gate outcomes.
    An epoch produced solely by predict_to has an empty readings tuple.
    """

    estimate: StateEstimate
    readings: tuple[FilterUpdate, ...]


class EnsembleFilter(ABC, Generic[Estimate]):
    """Independent spacecraft filters keyed by the same UUID used by JobData."""

    @staticmethod
    def ekf(
        *,
        force_model: ForceModel,
        noise: OrbitProcessNoise = OrbitProcessNoise(),
        gate: float | None = None,
        retain_history: bool = False,
    ) -> OrbitEKFCatalogue:
        """Configure a catalogue of Cartesian orbit and additive Doppler-bias EKFs."""
        return OrbitEKFCatalogue(
            force_model=force_model,
            noise=noise,
            gate=gate,
            retain_history=retain_history,
        )

    @staticmethod
    def tle_epoch_ukf(
        *, noise: TleProcessNoise, gate: float | None = None
    ) -> TleEpochUKFCatalogue:
        """Configure SGP4 offset/bias tracking with a bounded covariance per object."""
        return TleEpochUKFCatalogue(noise=noise, gate=gate)

    @abstractmethod
    def add_spacecraft(self, spacecraft_id: UUID, initial: Estimate) -> None:
        """Seed one catalogue entry with its initial estimate."""

    def drop_spacecraft(self, spacecraft_id: UUID) -> None:
        """Remove this object's estimate from the catalogue."""
        _catalogue.drop(self, spacecraft_id)

    def update(self, data: JobData) -> tuple[FilterUpdate, ...]:
        """Update only data.spacecraft_id, stably ordered by observation epoch.

        Gate is a normalized innovation squared threshold. A gated row retains
        the prediction. Supply observation_ids for durable reading references.
        A failure reports the last committed source row.
        """
        return _catalogue.update(self, data)

    def predict_to(self, epoch: Epoch, *, ids: Sequence[UUID] | None = None) -> None:
        """Advance selected objects through a gap. Omitting ids advances all objects."""
        _catalogue.predict(self, epoch, ids)

    def get_state(self, spacecraft_id: UUID) -> Estimate:
        """Return an independent snapshot at this object's latest processed epoch."""
        return _catalogue.get(self, spacecraft_id)

    def checkpoint(self, spacecraft_id: UUID) -> FilterCheckpoint:
        """Capture the entire entry and its reading cursors in checkpoint version 1."""
        return _catalogue.checkpoint(self, spacecraft_id)

    def restore(self, checkpoint: FilterCheckpoint) -> None:
        """Restore one entry using its saved settings, cursors and optional history.

        Reject unknown versions, incompatible kind/settings or an occupied ID
        before changing the catalogue.
        """
        _catalogue.restore(self, checkpoint)


@dataclass(frozen=True, kw_only=True)
class OrbitEKFCatalogue(EnsembleFilter[StateEstimate]):
    _entries: dict = field(default_factory=dict, init=False, repr=False, compare=False)
    _lock: RLock = field(default_factory=RLock, init=False, repr=False, compare=False)
    force_model: ForceModel
    noise: OrbitProcessNoise
    gate: float | None = None
    retain_history: bool = False

    def add_spacecraft(
        self,
        spacecraft_id: UUID,
        initial: StateEstimate,
        *,
        bias_prior_hz: GaussianPrior | None = None,
        last_contributing_epoch: Epoch | None = None,
    ) -> None:
        """Use the fitted drag as this object's fixed drag, else use the force model.

        Supply a covariance-bearing initial estimate. Retain the Cartesian block
        and any existing orbit/bias cross terms. If the estimate omits the bias,
        supply bias_prior_hz to initialize an independent seventh coordinate.
        """
        _catalogue.add_orbit(
            self, spacecraft_id, initial, bias_prior_hz, last_contributing_epoch
        )

    def smooth(self, spacecraft_id: UUID) -> tuple[SmoothedStep, ...]:
        """Run native RTS across retained epochs, including the initial prior.

        Requires retain_history=True. Return chronological augmented states and
        covariances paired with their forward-pass epochs and source readings.
        """
        return _catalogue.smooth(self, spacecraft_id)


@dataclass(frozen=True, kw_only=True)
class TleEpochUKFCatalogue(EnsembleFilter[TleCorrection]):
    _entries: dict = field(default_factory=dict, init=False, repr=False, compare=False)
    _lock: RLock = field(default_factory=RLock, init=False, repr=False, compare=False)
    noise: TleProcessNoise
    gate: float | None = None

    def add_spacecraft(self, spacecraft_id: UUID, initial: TleCorrection) -> None:
        """Start from the reference TLE and supplied covariance of offset and bias."""
        _catalogue.add_tle(self, spacecraft_id, initial)

    def replace_tle(
        self,
        spacecraft_id: UUID,
        tle: TLE,
        *,
        epoch: Epoch,
        offset_prior_s: GaussianPrior,
    ) -> None:
        """Reset the offset and cross term, retaining predicted bias and variance."""
        _catalogue.replace_tle(self, spacecraft_id, tle, epoch, offset_prior_s)


@dataclass(frozen=True, kw_only=True)
class OrbitUncertainty:
    """Independent Gaussian element perturbations. Phase shifts mean anomaly."""

    phase_sigma_s: float = 20.0
    inclination_sigma_deg: float = 0.025
    raan_sigma_deg: float = 0.035
    mean_motion_sigma_rev_day: float = 2.5e-4


# Custom sampler returns (count, 4) offsets in the OrbitUncertainty field order.
ParticleSampler = Callable[[np.random.Generator, int], FloatArray]


@dataclass(frozen=True, kw_only=True)
class AcquisitionReceiver:
    """Beam and lock parameters used by the native MPC detection model."""

    beam_fwhm_rad: float = radians(3.0)
    cutoff_rad: float = radians(3.0)
    on_axis_ebn0_db_at_1000_km: float = 20.0
    acquire_ebn0_db: float = 5.0
    acquisition_s: float = 0.3


@dataclass(frozen=True)
class SearchSetup:
    """Station, search clock, antenna limits and BC-MPC planning settings."""

    station: GroundStation
    start: Epoch
    _: KW_ONLY
    receiver: AcquisitionReceiver = AcquisitionReceiver()
    search_duration_s: float = 192.0
    move_s: float = 4.0
    settle_s: float = 0.5
    listen_s: float = 1.5
    rate_rad_s: float = radians(1.10)
    acceleration_rad_s2: float = radians(1.0)
    elevation_mask_rad: float = radians(10.0)
    initial_az_el_rad: tuple[float, float] | None = None
    horizon: int = 4
    beam_width: int = 40
    candidate_count: int = 17
    candidate_extent_sigma: float = 2.8
    resample_ess_fraction: float = 0.5


@dataclass(frozen=True)
class SearchAction:
    azimuth_rad: float
    elevation_rad: float
    listen_start: Epoch
    listen_end: Epoch


@dataclass(frozen=True)
class SearchPlan:
    actions: tuple[SearchAction, ...]
    acquisition_probability: float


@dataclass(frozen=True)
class SearchState:
    """Weight i belongs to the original hypothesis indexed by ancestor_indices[i]."""

    status: Literal["ready", "awaiting_observation", "acquired", "exhausted"]
    cycle: int
    weights: FloatArray
    ancestor_indices: NDArray[np.intp]


@dataclass(frozen=True, kw_only=True)
class SatFinder:
    """Caller-clocked BC-MPC search. Factories describe the two supported priors."""

    nominal: TLE | OMM
    setup: SearchSetup
    rng: np.random.Generator
    n_particles: int
    distribution: OrbitUncertainty | ParticleSampler | None = None
    hypotheses: tuple[TLE | OMM, ...] = ()
    weights: FloatArray | None = None

    _controller: object = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        object.__setattr__(self, "_controller", _search.controller(self))

    @classmethod
    def from_tle(
        cls,
        tle: TLE,
        *,
        setup: SearchSetup,
        n_particles: int = 160,
        distribution: OrbitUncertainty | ParticleSampler = OrbitUncertainty(),
        rng: np.random.Generator | None = None,
    ) -> SatFinder:
        """Configure equal-weight sampling at the original TLE epoch."""
        if n_particles < 1:
            raise ValueError("n_particles must be positive")
        return cls(
            nominal=tle,
            setup=setup,
            n_particles=n_particles,
            distribution=distribution,
            rng=np.random.default_rng(rng),
        )

    @classmethod
    def from_hypotheses(
        cls,
        nominal: TLE | OMM,
        hypotheses: Sequence[TLE | OMM],
        weights: FloatArray,
        *,
        setup: SearchSetup,
        rng: np.random.Generator | None = None,
    ) -> SatFinder:
        """Configure a supplied weighted particle pool and RNG for resampling."""
        if not len(hypotheses):
            raise ValueError("supply at least one hypothesis")
        _check_shape("weights", weights, (len(hypotheses),))
        if not np.all(np.isfinite(weights) & (weights >= 0)) or not np.any(weights > 0):
            raise ValueError(
                "weights must be finite and nonnegative with positive mass"
            )
        return cls(
            nominal=nominal,
            setup=setup,
            n_particles=len(hypotheses),
            hypotheses=tuple(hypotheses),
            weights=weights,
            rng=np.random.default_rng(rng),
        )

    def plan(self) -> SearchPlan:
        """Return timed pointings. Execute the first action and report its outcome."""
        return _search.plan(self)

    def observe(self, *, acquired: bool) -> None:
        """Resolve the pending action. An execution failure leaves feedback pending."""
        self._controller.observe("acquired" if acquired else "miss", self.rng.random)

    def read_state(self) -> SearchState:
        """Return status and particle belief for monitoring."""
        return _search.state(self)


__all__ = [
    "GroundStation",
    "ContactSamples",
    "ObservationBatch",
    "JobData",
    "ForceModel",
    "GaussianPrior",
    "TleAgeUncertainty",
    "TleInitialization",
    "FitError",
    "CovarianceError",
    "RankDeficientError",
    "OrbitalMarginal",
    "StateEstimate",
    "TleCorrection",
    "OptimizerProfile",
    "CalibratedLM",
    "ContactBiasFit",
    "PassiveRFFit",
    "ConstrainedFit",
    "FitResult",
    "ODProblem",
    "OrbitProcessNoise",
    "TleProcessNoise",
    "ReadingRef",
    "FilterUpdate",
    "FilterCheckpoint",
    "SmoothedStep",
    "EnsembleFilter",
    "OrbitEKFCatalogue",
    "TleEpochUKFCatalogue",
    "OrbitUncertainty",
    "AcquisitionReceiver",
    "SearchSetup",
    "SearchAction",
    "SearchPlan",
    "SearchState",
    "SatFinder",
    "Epoch",
    "TLE",
    "OMM",
    "ParticleSampler",
]
