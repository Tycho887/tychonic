"""Batch orbit estimation and sequential filters."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Callable, Sequence

import numpy as np
import satkit
from numpy.typing import ArrayLike, NDArray

from . import _native
from ._utils import (
    Stateful,
    array,
    dumps,
    process_covariance,
    time_object,
    time_string,
)
from .config import FilterOptions, LMOptions, PropagationConfig
from .measurements import GroundStation, MeasurementBatch

ProcessNoise = Callable[[NDArray[np.float64], satkit.time, satkit.time], ArrayLike]


@dataclass(frozen=True)
class FitResult:
    """Reference parameters, weighted residual sum of squares, and iterations."""

    x: NDArray[np.float64]
    rss: float
    nit: int

    @property
    def cost(self) -> float:
        return self.rss / 2


@dataclass(frozen=True)
class UpdateResult:
    innovation: NDArray[np.float64]
    nis: float
    accepted: bool


@dataclass(frozen=True)
class BatchUpdate:
    epoch: satkit.time
    batch: int
    row: int
    nis: float
    accepted: bool


@dataclass(frozen=True)
class FilterHistory:
    """One snapshot per distinct epoch, including the initial prior."""

    times: tuple[satkit.time, ...]
    states: NDArray[np.float64]
    covariances: NDArray[np.float64]


def _history(value) -> FilterHistory:
    times, states, covariances = value
    return FilterHistory(
        tuple(map(time_object, times)), np.asarray(states), np.asarray(covariances)
    )


def fit_orbit(
    batches: Sequence[MeasurementBatch],
    x0: ArrayLike,
    *,
    propagation: PropagationConfig,
    options: LMOptions | None = None,
) -> FitResult:
    """Fit calibrated observations from an initial GCRF orbit guess.

    Full measurement covariance weights residuals. The Rust LM solves scaled
    corrections around x0. Errors, including maximum iterations, raise SolverError
    or a more specific Tychonic exception. The core does not estimate fit covariance.
    """
    options = options or LMOptions()
    settings = asdict(options)
    if options.parameter_scales is not None:
        settings["parameter_scales"] = array(options.parameter_scales, ndim=1).tolist()
    x, rss, nit = _native.fit_orbit(
        [batch._native_batch for batch in batches],
        array(x0, ndim=1).tolist(),
        propagation._native_config,
        dumps(settings),
    )
    return FitResult(np.asarray(x), rss, nit)


class OrbitEKF(Stateful):
    """Six/seven-state orbit EKF fusing any calibrated measurement combination.

    Process-noise callbacks receive (state, previous_epoch, next_epoch) and return
    already-discretized GCRF Q. A supplied per-call q overrides that callback.
    Callbacks must use their supplied state; accessing this filter during native
    mutation is disallowed. NIS rejection retains prediction; errors roll it back.
    """

    def __init__(
        self,
        x0: ArrayLike,
        covariance: ArrayLike,
        *,
        propagation: PropagationConfig,
        process_noise: ProcessNoise | None = None,
        options: FilterOptions | None = None,
    ):
        if process_noise is not None and not callable(process_noise):
            raise TypeError("process_noise must be callable")
        self._filter = _native.NativeOrbitEKF(
            array(x0, ndim=1),
            array(covariance, ndim=2),
            propagation._native_config,
        )
        self._dimension = propagation.dimension
        self._process_noise = process_noise
        self.options = options or FilterOptions()

    @property
    def state(self) -> NDArray[np.float64]:
        return np.asarray(self._filter.snapshot()[0])

    @property
    def covariance(self) -> NDArray[np.float64]:
        return np.asarray(self._filter.snapshot()[1])

    @property
    def epoch(self) -> satkit.time:
        return time_object(self._filter.snapshot()[2])

    @property
    def history(self) -> FilterHistory:
        return _history(self._filter.history())

    def _noise(self, state, previous, next_epoch):
        if self._process_noise is None:
            return process_covariance(None, self._dimension)
        return process_covariance(
            self._process_noise(state, previous, next_epoch), self._dimension
        )

    def _q(self, time: satkit.time, q: ArrayLike | None):
        if q is not None or time <= self.epoch:
            return process_covariance(q, self._dimension)
        return self._noise(self.state, self.epoch, time)

    def predict_to(self, time: satkit.time, q: ArrayLike | None = None) -> None:
        """Predict chronologically; same-epoch prediction requires zero Q."""
        with self._mutation():
            self._filter.predict_to(time_string(time), self._q(time, q))

    def observe(
        self,
        batch: MeasurementBatch,
        row: int = 0,
        q: ArrayLike | None = None,
    ) -> UpdateResult:
        """Predict and jointly update one batch row, returning innovation and NIS."""
        if not 0 <= row < len(batch):
            raise ValueError("measurement row is out of range")
        with self._mutation():
            time = time_object(batch._times[row])
            value = self._filter.observe(
                batch._native_batch,
                row,
                self._q(time, q),
                self.options.gate,
            )
        return UpdateResult(np.asarray(value[0]), value[1], value[2])

    def filter_batches(self, batches: Sequence[MeasurementBatch]) -> list[BatchUpdate]:
        """Sort epochs stably; apply Q once per new epoch.

        Successful earlier updates remain committed if a later reading fails.
        Same-time rows retain their supplied batch/row order and do not add Q.
        """

        def noise(state, previous, next_epoch):
            return self._noise(state, time_object(previous), time_object(next_epoch))

        with self._mutation():
            updates = self._filter.filter_batches(
                [batch._native_batch for batch in batches],
                noise,
                self.options.gate,
            )
        return [
            BatchUpdate(time_object(time), batch, row, nis, accepted)
            for time, batch, row, nis, accepted in updates
        ]

    def smooth(self) -> FilterHistory:
        """RTS-smoothed snapshots aligned with history, including the prior epoch."""
        return _history(self._filter.smooth())


@dataclass(frozen=True)
class TcaMeasurementNoise:
    """Independent 1σ values: range rate m/s and azimuth/elevation radians."""

    sigma_rr: float = 0.5
    sigma_az: float = np.deg2rad(2.5)
    sigma_el: float = np.deg2rad(2.5)


@dataclass(frozen=True)
class TcaCorrection:
    epoch_offset_seconds: float
    range_rate_bias_m_per_s: float
    covariance: NDArray[np.float64]
    corrected_omm: dict


class TcaUKF(Stateful):
    """Pass-level UKF for [TLE epoch offset seconds, range-rate bias m/s].

    Positive epoch offset moves the TLE epoch later with other elements fixed.
    It uses five sigma points with fixed alpha=1, beta=2, kappa=0 and a circular
    azimuth mean. It estimates a single roughly Gaussian solution for one pass.
    """

    def __init__(
        self,
        tle: str,
        epoch: satkit.time,
        x0: ArrayLike,
        covariance: ArrayLike,
        *,
        options: FilterOptions | None = None,
    ):
        self._filter = _native.NativeTcaUKF(
            tle,
            time_string(epoch),
            array(x0, ndim=1),
            array(covariance, ndim=2),
        )
        self.options = options or FilterOptions()

    @property
    def state(self) -> NDArray[np.float64]:
        return np.asarray(self._filter.snapshot()[0])

    @property
    def covariance(self) -> NDArray[np.float64]:
        return np.asarray(self._filter.snapshot()[1])

    @property
    def epoch(self) -> satkit.time:
        return time_object(self._filter.snapshot()[2])

    @property
    def corrected_omm(self) -> dict:
        """Corrected orbital elements; sensor bias remains separate."""
        return json.loads(self._filter.corrected_omm())

    @property
    def correction(self) -> TcaCorrection:
        state = self.state
        return TcaCorrection(state[0], state[1], self.covariance, self.corrected_omm)

    def predict_to(self, time: satkit.time, q: ArrayLike | None = None) -> None:
        with self._mutation():
            self._filter.predict_to(time_string(time), process_covariance(q, 2))

    def observe(
        self,
        time: satkit.time,
        station: GroundStation,
        values: ArrayLike,
        *,
        noise: TcaMeasurementNoise | None = None,
        q: ArrayLike | None = None,
    ) -> UpdateResult:
        """Joint [range_rate, azimuth, elevation] update, in m/s and radians.

        Noise components are independent; arbitrary correlated measurement noise
        is not supported by the existing TCA UKF. q is already discretized (2,2).
        """
        noise = noise or TcaMeasurementNoise()
        with self._mutation():
            result = self._filter.observe(
                time_string(time),
                station._coordinates(),
                array(values, ndim=1).tolist(),
                [noise.sigma_rr, noise.sigma_az, noise.sigma_el],
                process_covariance(q, 2),
                self.options.gate,
            )
        return UpdateResult(np.asarray(result[0]), result[1], result[2])
