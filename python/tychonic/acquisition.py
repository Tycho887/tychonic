"""Bayesian Continuation MPC with caller-owned receiver and antenna execution."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Callable, Mapping, Sequence

import numpy as np
import satkit
from numpy.typing import ArrayLike, NDArray

from . import _native
from ._utils import Stateful, array, dumps, time_object, time_string
from .measurements import GroundStation


@dataclass(frozen=True)
class MPCConfig:
    search_duration_s: float = 192.0
    move_s: float = 4.0
    settle_s: float = 0.5
    listen_s: float = 1.5
    horizon: int = 4
    beam_width: int = 40
    candidate_count: int = 17
    candidate_extent_sigma: float = 2.8
    ess_threshold: float = 0.5


@dataclass(frozen=True)
class AntennaLimits:
    """Great-circle rest-to-rest slew limits in radians/s and radians/s²."""

    rate_rad_s: float = np.deg2rad(1.10)
    acceleration_rad_s2: float = np.deg2rad(1.00)


@dataclass(frozen=True)
class SearchSetup:
    station: GroundStation
    start: satkit.time
    config: MPCConfig = field(default_factory=MPCConfig)
    antenna: AntennaLimits = field(default_factory=AntennaLimits)
    elevation_mask_rad: float = np.deg2rad(10.0)
    initial_direction: tuple[float, float, float] | None = None


class SearchStatus(str, Enum):
    READY = "ready"
    AWAITING_OBSERVATION = "awaiting_observation"
    ACQUIRED = "acquired"
    EXHAUSTED = "exhausted"


class Observation(str, Enum):
    MISS = "miss"
    ACQUIRED = "acquired"


@dataclass(frozen=True)
class ParticleGeometry:
    """Vectors in absolute ENU, ranges in metres, elevations in radians."""

    directions_enu: NDArray[np.float64]
    ranges_m: NDArray[np.float64]
    elevations_rad: NDArray[np.float64]
    visible: NDArray[np.bool_]


DetectionProbabilities = Callable[
    [NDArray[np.float64], ParticleGeometry, float], ArrayLike
]


@dataclass(frozen=True)
class Action:
    cycle: int
    direction_enu: NDArray[np.float64]
    listen_start: satkit.time
    listen_end: satkit.time
    candidate_index: int | None

    @property
    def azimuth_elevation(self) -> NDArray[np.float64]:
        east, north, up = self.direction_enu
        return np.array(
            [
                np.arctan2(east, north) % (2 * np.pi),
                np.arctan2(up, np.hypot(east, north)),
            ]
        )


@dataclass(frozen=True)
class ActionPlan:
    actions: tuple[Action, ...]
    acquisition_probability: float
    retained_acquisition_probability: float


@dataclass(frozen=True)
class ObservationUpdate:
    resampled: bool
    status: SearchStatus


def _ephemeris(value: str | Mapping) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return dumps(dict(value))
    raise TypeError("expected TLE text or an OMM mapping")


class BCMPC(Stateful):
    """Finite-contact search controller; execute only the first action of each plan.

    The receiver callback returns acquisition probability for every active particle,
    with zero for invisible particles, deterministically within one plan. Geometry
    is evaluated at listening midpoints. Actuator/API failures leave actions pending:
    report a miss only after successfully completing the entire listening interval.
    """

    def __init__(
        self,
        nominal: str | Mapping,
        particles: Sequence[str | Mapping],
        weights: ArrayLike,
        *,
        setup: SearchSetup,
    ):
        settings = {
            "station": setup.station._coordinates(),
            "start": time_string(setup.start),
            "config": asdict(setup.config),
            "antenna": asdict(setup.antenna),
            "elevation_mask_rad": setup.elevation_mask_rad,
            "initial_direction": setup.initial_direction,
        }
        self._controller = _native.NativeBCMPC(
            _ephemeris(nominal),
            [_ephemeris(particle) for particle in particles],
            array(weights, ndim=1).tolist(),
            dumps(settings),
        )
        self._rng = np.random.default_rng()

    @property
    def status(self) -> SearchStatus:
        return SearchStatus(self._controller.snapshot()[0])

    @property
    def finished(self) -> bool:
        return self.status in (SearchStatus.ACQUIRED, SearchStatus.EXHAUSTED)

    @property
    def cycle(self) -> int:
        return self._controller.snapshot()[1]

    @property
    def weights(self) -> NDArray[np.float64]:
        return np.asarray(self._controller.snapshot()[2])

    @property
    def particle_indices(self) -> NDArray[np.int64]:
        return np.asarray(self._controller.snapshot()[3], dtype=np.int64)

    @property
    def direction_enu(self) -> NDArray[np.float64]:
        return np.asarray(self._controller.snapshot()[4])

    @property
    def effective_sample_size(self) -> float:
        return self._controller.snapshot()[5]

    def plan(self, probabilities: DetectionProbabilities) -> ActionPlan:
        if not callable(probabilities):
            raise TypeError("probabilities must be callable")

        def callback(command, directions, ranges, elevations, visible, listen_s):
            geometry = ParticleGeometry(directions, ranges, elevations, visible)
            result = array(probabilities(command, geometry, listen_s), ndim=1)
            if result.shape != ranges.shape:
                raise ValueError("receiver must return one probability per particle")
            if ((result < 0) | (result > 1)).any():
                raise ValueError("probabilities must be in [0,1]")
            if (result[~visible] != 0).any():
                raise ValueError("invisible particles must have zero probability")
            return result

        with self._mutation():
            actions, probability, retained = self._controller.plan(callback)
        return ActionPlan(
            tuple(
                Action(
                    cycle,
                    np.asarray(direction),
                    time_object(start),
                    time_object(end),
                    candidate,
                )
                for cycle, direction, start, end, candidate in actions
            ),
            probability,
            retained,
        )

    def observe(
        self,
        outcome: Observation | str,
        rng: np.random.Generator | None = None,
    ) -> ObservationUpdate:
        """Consume a completed listen; draw RNG only if miss resampling is needed."""
        outcome = Observation(outcome)
        rng = self._rng if rng is None else rng
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator")
        with self._mutation():
            resampled, status = self._controller.observe(outcome.value, rng.random)
        return ObservationUpdate(resampled, SearchStatus(status))
