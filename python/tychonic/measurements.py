"""Calibrated observables with full measurement covariance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import satkit
from numpy.typing import ArrayLike, NDArray

from . import _native
from ._utils import array, dumps, time_object, time_string
from .config import PropagationConfig


@dataclass(frozen=True)
class GroundStation:
    """Geodetic latitude/longitude in degrees and altitude in metres."""

    lat_deg: float
    lon_deg: float
    alt_m: float = 0.0

    def _coordinates(self) -> tuple[float, float, float]:
        return self.lat_deg, self.lon_deg, self.alt_m


@dataclass(frozen=True)
class Interferometer:
    """Antenna 1 minus antenna 2 in antenna 1's ENU frame (metres).

    carrier_hz is the nominal RF carrier. spherical_phase selects exact range
    difference instead of the default far-field phase model.
    """

    baseline_enu_m: tuple[float, float, float]
    carrier_hz: float
    spherical_phase: bool = False


@dataclass(frozen=True)
class Observable:
    """Known calibration constants for one observable; construct with named methods."""

    kind: str
    carrier_hz: float | None = None
    offset_hz: float = 0.0
    baseline_enu_m: tuple[float, float, float] | None = None
    spherical_phase: bool = False
    frequency_offset_hz: float = 0.0
    phase_offset_rad: float = 0.0

    @classmethod
    def doppler(cls, carrier_hz: float, *, offset_hz: float = 0.0) -> Observable:
        """Carrier-relative Doppler Hz; geometric sign is −fc * range_rate / c."""
        return cls("doppler", carrier_hz=carrier_hz, offset_hz=offset_hz)

    @classmethod
    def phase_difference(
        cls,
        interferometer: Interferometer,
        *,
        frequency_offset_hz: float = 0.0,
        phase_offset_rad: float = 0.0,
    ) -> Observable:
        """Continuous phase radians; phase unwrapping belongs upstream."""
        return cls(
            "phase_difference",
            interferometer.carrier_hz,
            baseline_enu_m=tuple(interferometer.baseline_enu_m),
            spherical_phase=interferometer.spherical_phase,
            frequency_offset_hz=frequency_offset_hz,
            phase_offset_rad=phase_offset_rad,
        )

    @classmethod
    def differential_doppler(cls, interferometer: Interferometer) -> Observable:
        """Antenna 1 minus antenna 2 Doppler in Hz."""
        return cls(
            "differential_doppler",
            interferometer.carrier_hz,
            baseline_enu_m=tuple(interferometer.baseline_enu_m),
        )

    @classmethod
    def azimuth_elevation(cls) -> Observable:
        """Radians; azimuth eastward from north with a wrapped residual."""
        return cls("azimuth_elevation")


class MeasurementBatch:
    """Owned, validated measurements, constructed through joint or named helpers.

    Joint components always use Doppler, phase difference, differential Doppler,
    azimuth, elevation order, omitting absent components, regardless of model order.
    Inputs are copied. Separate independent schedules use separate batches.
    """

    def __init__(
        self,
        times: Sequence[satkit.time],
        stations: GroundStation | Sequence[GroundStation],
        observables: Sequence[Observable],
        values: ArrayLike,
        covariance: ArrayLike,
    ):
        from dataclasses import asdict

        self._times = tuple(time_string(time) for time in times)
        if isinstance(stations, GroundStation):
            stations = [stations] * len(self._times)
        values = array(values, ndim=2)
        covariance = array(covariance)
        if covariance.ndim == 2:
            covariance = np.broadcast_to(
                covariance, (len(self._times), *covariance.shape)
            ).copy()
        if covariance.ndim != 3:
            raise ValueError("covariance must have shape (M,M) or (N,M,M)")
        self._native_batch = _native.NativeBatch(
            list(self._times),
            [station._coordinates() for station in stations],
            dumps([asdict(model) for model in observables]),
            values,
            covariance,
        )
        self._components = values.shape[1]

    @property
    def times(self) -> tuple[satkit.time, ...]:
        return tuple(time_object(time) for time in self._times)

    @property
    def components(self) -> int:
        return self._components

    def __len__(self) -> int:
        return len(self._times)

    def predict(
        self, x: ArrayLike, *, propagation: PropagationConfig
    ) -> NDArray[np.float64]:
        """Predict physical observables at this schedule for a reference GCRF state."""
        return np.asarray(
            self._native_batch.predict(
                propagation._native_config,
                array(x, ndim=1).tolist(),
            )
        )


def joint(
    times: Sequence[satkit.time],
    stations: GroundStation | Sequence[GroundStation],
    observables: Sequence[Observable],
    values: ArrayLike,
    covariance: ArrayLike,
) -> MeasurementBatch:
    """Prepare canonical-order values (N,M) and SPD covariance (M,M) or (N,M,M)."""
    return MeasurementBatch(times, stations, observables, values, covariance)


def _single(
    times: Sequence[satkit.time],
    stations: GroundStation | Sequence[GroundStation],
    observable: Observable,
    values: ArrayLike,
    sigma: float,
) -> MeasurementBatch:
    sigma = float(sigma)
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("measurement sigma must be finite and positive")
    return joint(
        times, stations, [observable], array(values, ndim=1)[:, None], [[sigma**2]]
    )


def doppler(
    times: Sequence[satkit.time],
    stations: GroundStation | Sequence[GroundStation],
    carrier_hz: float,
    values: ArrayLike,
    sigma_hz: float,
) -> MeasurementBatch:
    """Prepare calibrated carrier-relative Doppler Hz with an independent sigma."""
    return _single(times, stations, Observable.doppler(carrier_hz), values, sigma_hz)


def phase_difference(
    times: Sequence[satkit.time],
    stations: GroundStation | Sequence[GroundStation],
    interferometer: Interferometer,
    values: ArrayLike,
    sigma_rad: float,
) -> MeasurementBatch:
    """Prepare calibrated continuous phase differences in radians."""
    return _single(
        times, stations, Observable.phase_difference(interferometer), values, sigma_rad
    )


def differential_doppler(
    times: Sequence[satkit.time],
    stations: GroundStation | Sequence[GroundStation],
    interferometer: Interferometer,
    values: ArrayLike,
    sigma_hz: float,
) -> MeasurementBatch:
    """Prepare f1 − f2 Doppler differences in Hz."""
    return _single(
        times,
        stations,
        Observable.differential_doppler(interferometer),
        values,
        sigma_hz,
    )


def azimuth_elevation(
    times: Sequence[satkit.time],
    stations: GroundStation | Sequence[GroundStation],
    azimuth: ArrayLike,
    elevation: ArrayLike,
    sigma_az: float,
    sigma_el: float,
) -> MeasurementBatch:
    """Prepare azimuth/elevation and independent standard deviations in radians."""
    sigmas = array([sigma_az, sigma_el], ndim=1)
    if (sigmas <= 0).any():
        raise ValueError("angle sigmas must be positive")
    values = np.column_stack((array(azimuth, ndim=1), array(elevation, ndim=1)))
    return joint(
        times, stations, [Observable.azimuth_elevation()], values, np.diag(sigmas**2)
    )


def doppler_differential_covariance(
    sigma1: float, sigma2: float
) -> NDArray[np.float64]:
    """Noise block for [antenna-1 Doppler, f1 − f2] from independent antennas."""
    sigmas = array([sigma1, sigma2], ndim=1)
    if (sigmas <= 0).any():
        raise ValueError("antenna sigmas must be positive")
    a, b = sigmas**2
    return np.array([[a, a], [a, a + b]])
