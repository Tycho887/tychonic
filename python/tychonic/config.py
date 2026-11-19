"""Named settings using the numerical defaults of the Rust core."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import satkit
from numpy.typing import NDArray

from . import _native
from ._utils import dumps, time_string


@dataclass(frozen=True)
class PropagationSettings:
    """STM-compatible forces and integrator controls; errors use GCRF SI units.

    Gravity models are EGM2008, EGM96, JGM2, JGM3, or ITUGrace16. Integrators are
    RKV98, RKV98NoInterp, RKV87, RKV65, or RKTS54. Tide and relativity
    corrections are unavailable because Satkit lacks their STM partials.
    """

    gravity_degree: int = 4
    gravity_order: int = 4
    gravity_model: str = "EGM2008"
    abs_error: float = 1e-10
    rel_error: float = 1e-13
    use_spaceweather: bool = True
    use_sun_gravity: bool = True
    use_moon_gravity: bool = True
    enable_interp: bool = True
    integrator: str = "RKV98"
    max_steps: int = 1_000_000
    require_eop_coverage: bool = False
    initial_step_secs: float | None = None


@dataclass(frozen=True)
class PropagationConfig:
    """Reference epoch and six/seven-state orbit parameterization.

    States are [x, y, z, vx, vy, vz] in GCRF metres and m/s. When
    estimate_drag is true, append C_d A/m in m²/kg. delta_cd is the
    finite-difference step for drag sensitivity, not a prior uncertainty.
    """

    epoch: satkit.time
    settings: PropagationSettings = field(default_factory=PropagationSettings)
    estimate_drag: bool = False
    fixed_drag: float = 0.0
    delta_cd: float = 1e-4
    _native_config: _native.NativePropagation = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self):
        native = _native.NativePropagation(
            time_string(self.epoch),
            dumps(asdict(self.settings)),
            self.estimate_drag,
            self.fixed_drag,
            self.delta_cd,
        )
        object.__setattr__(self, "_native_config", native)

    @property
    def dimension(self) -> int:
        return 7 if self.estimate_drag else 6


@dataclass(frozen=True)
class LMOptions:
    """Levenberg–Marquardt settings for dimensionless corrections around x0.

    Default parameter scales are 100 m for position, 0.1 m/s for velocity,
    and 0.01 m²/kg for estimated drag. Tolerances apply in these coordinates.
    """

    max_iter: int = 100
    grad_tol: float = 1e-8
    x_tol: float = 1e-12
    f_tol: float = 1e-12
    mu_init: float = 1e-3
    mu_min: float = 1e-10
    mu_max: float = 1e10
    mu_increase: float = 10.0
    mu_decrease: float = 0.1
    parameter_scales: tuple[float, ...] | None = None


@dataclass(frozen=True)
class FilterOptions:
    """NIS threshold (chi-squared with measurement dimension); None disables gating."""

    gate: float | None = None

    def __post_init__(self):
        if self.gate is not None and (not np.isfinite(self.gate) or self.gate < 0):
            raise ValueError("gate must be finite and nonnegative")


@dataclass(frozen=True)
class WhiteAccelerationNoise:
    """Continuous GCRF acceleration PSD (m²/s³), optional drag random-walk PSD.

    A scalar acceleration_psd applies to each axis; a triple specifies each
    axis separately. drag_psd has units (m²/kg)²/s and requires seven states.
    """

    acceleration_psd: float | tuple[float, float, float] = 0.0
    drag_psd: float = 0.0

    def __call__(
        self, state: NDArray[np.float64], previous: satkit.time, next_epoch: satkit.time
    ) -> NDArray[np.float64]:
        seconds = (next_epoch - previous).seconds
        psd = np.broadcast_to(self.acceleration_psd, (3,)).tolist()
        return np.asarray(
            _native.white_acceleration_noise(
                seconds,
                psd,
                self.drag_psd,
                state.size == 7,
            )
        )
