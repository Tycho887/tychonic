"""Orbit determination, sensor fusion, and acquisition with NumPy and Satkit time."""

from . import measurements
from ._native import (
    AcquisitionStateError,
    DataError,
    NumericalError,
    SolverError,
    TychonicError,
)
from ._native import (
    core_version as __core_version__,
)
from .acquisition import (
    BCMPC,
    Action,
    ActionPlan,
    AntennaLimits,
    MPCConfig,
    Observation,
    ObservationUpdate,
    ParticleGeometry,
    SearchSetup,
    SearchStatus,
)
from .config import (
    FilterOptions,
    LMOptions,
    PropagationConfig,
    PropagationSettings,
    WhiteAccelerationNoise,
)
from .estimation import (
    BatchUpdate,
    FilterHistory,
    FitResult,
    OrbitEKF,
    TcaCorrection,
    TcaMeasurementNoise,
    TcaUKF,
    UpdateResult,
    fit_orbit,
)
from .measurements import GroundStation, Interferometer, MeasurementBatch, Observable

__version__ = "0.2.0"

__all__ = [
    "__version__",
    "__core_version__",
    "AcquisitionStateError",
    "Action",
    "ActionPlan",
    "AntennaLimits",
    "BCMPC",
    "BatchUpdate",
    "DataError",
    "FilterHistory",
    "FilterOptions",
    "FitResult",
    "GroundStation",
    "Interferometer",
    "LMOptions",
    "MPCConfig",
    "MeasurementBatch",
    "NumericalError",
    "Observable",
    "Observation",
    "ObservationUpdate",
    "OrbitEKF",
    "ParticleGeometry",
    "PropagationConfig",
    "PropagationSettings",
    "SearchSetup",
    "SearchStatus",
    "SolverError",
    "TcaCorrection",
    "TcaMeasurementNoise",
    "TcaUKF",
    "TychonicError",
    "UpdateResult",
    "WhiteAccelerationNoise",
    "fit_orbit",
    "measurements",
]
