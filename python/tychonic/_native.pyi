"""Internal extension interface; users should prefer the public Python facade."""

from collections.abc import Sequence
from typing import Callable

import numpy as np
from numpy.typing import NDArray

core_version: str

class TychonicError(RuntimeError): ...
class NumericalError(TychonicError): ...
class SolverError(TychonicError): ...
class DataError(TychonicError): ...
class AcquisitionStateError(TychonicError): ...

class NativePropagation:
    def __init__(
        self,
        epoch: str,
        settings: str,
        estimate_drag: bool,
        fixed_drag: float,
        delta_cd: float,
    ) -> None: ...

class NativeBatch:
    def __init__(
        self,
        times: Sequence[str] | NDArray[np.float64],
        stations: list[tuple[float, float, float]],
        models: str,
        values: NDArray[np.float64],
        covariance: NDArray[np.float64],
        phase_baselines: NDArray[np.float64] | None = ...,
        differential_baselines: NDArray[np.float64] | None = ...,
    ) -> None: ...
    def predict(
        self, propagation: NativePropagation, x: list[float]
    ) -> list[list[float]]: ...

class NativeOrbitEKF:
    def __init__(
        self,
        x: NDArray[np.float64],
        covariance: NDArray[np.float64],
        propagation: NativePropagation,
        bias_hz: bool = ...,
    ) -> None: ...
    def snapshot(self) -> tuple[list[float], list[list[float]], str]: ...
    def history(
        self,
    ) -> tuple[list[str], list[list[float]], list[list[list[float]]]]: ...
    def smooth(
        self,
    ) -> tuple[list[str], list[list[float]], list[list[list[float]]]]: ...
    def predict_to(self, time: str, q: NDArray[np.float64]) -> None: ...
    def observe(
        self, batch: NativeBatch, row: int, q: NDArray[np.float64], gate: float | None
    ) -> tuple[list[float], float, bool]: ...
    def filter_batches(
        self, batches: list[NativeBatch], process_noise: Callable, gate: float | None
    ) -> list[tuple[str, int, int, float, bool]]: ...
    def update_bias(
        self,
        batches: list[NativeBatch],
        acceleration: float,
        bias: float,
        retain_history: bool,
        gate: float | None,
    ) -> Receipts: ...
    def discard_history(self) -> None: ...
    def saved_steps(self) -> list[SavedStep]: ...
    def restore_steps(self, steps: Sequence[SavedStep]) -> None: ...

class NativeTcaUKF:
    def __init__(
        self,
        prior: str,
        epoch: str,
        x: NDArray[np.float64],
        covariance: NDArray[np.float64],
        doppler_hz: bool = ...,
    ) -> None: ...
    def snapshot(self) -> tuple[list[float], list[list[float]], str]: ...
    def corrected_elements(self) -> tuple[list[str], str]: ...
    def predict_bounded(self, time: str, noise: Sequence[float]) -> None: ...
    def update_hz(
        self, batches: list[NativeBatch], noise: Sequence[float], gate: float | None
    ) -> Receipts: ...
    def corrected_omm(self) -> str: ...
    def predict_to(self, time: str, q: NDArray[np.float64]) -> None: ...
    def observe(
        self,
        time: str,
        station: tuple[float, float, float],
        values: list[float],
        sigmas: list[float],
        q: NDArray[np.float64],
        gate: float | None,
    ) -> tuple[list[float], float, bool]: ...

class NativeBCMPC:
    def __init__(
        self, nominal: str, particles: list[str], weights: list[float], setup: str
    ) -> None: ...
    def snapshot(
        self,
    ) -> tuple[str, int, list[float], list[int], list[float], float]: ...
    def plan(
        self, callback: Callable
    ) -> tuple[list[tuple[int, list[float], str, str, int | None]], float, float]: ...
    def plan_receiver(
        self, settings: Sequence[float]
    ) -> tuple[list[tuple[int, list[float], str, str, int | None]], float, float]: ...
    def observe(self, outcome: str, draw: Callable[[], float]) -> tuple[bool, str]: ...

def fit_orbit(
    batches: list[NativeBatch],
    x: list[float],
    propagation: NativePropagation,
    options: str,
) -> tuple[list[float], float, int]: ...
def white_acceleration_noise(
    seconds: float, acceleration_psd: list[float], drag_psd: float, estimate_drag: bool
) -> list[list[float]]: ...
def roundtrip_time(value: str) -> str: ...

Rows = list[list[float]]
Receipts = tuple[list[tuple[int, int, str, float, bool]], Exception | None]
SavedStep = tuple[str, list[float], list[float], list[float], list[float], list[float]]

class RankFailure(NumericalError): ...
class CovarianceFailure(NumericalError): ...

def elapsed(previous: str, next: str) -> float: ...
def tle_seed(source: str, at: str) -> tuple[list[float], float]: ...
def particles(source: str, offsets: NDArray[np.float64]) -> list[str]: ...
def calibrated(
    batches: Sequence[NativeBatch],
    config: NativePropagation,
    guess: Sequence[float],
    covariance: bool,
    options: tuple[int | None, float | None, float | None, float | None],
    prior: tuple[Sequence[float], NDArray[np.float64]] | None = None,
) -> tuple[list[float], Rows | None, float, int, int | None, str | None, str]: ...
def transport(
    config: NativePropagation,
    x: Sequence[float],
    at: str,
    covariance: NDArray[np.float64] | None = None,
) -> tuple[list[float], Rows | None]: ...
def arc(
    config: NativePropagation, x: Sequence[float], epochs: Sequence[str]
) -> Rows: ...
def mean_elements(
    config: NativePropagation, x: Sequence[float], epochs: Sequence[str]
) -> tuple[list[str], str]: ...
def contacts(
    config: NativePropagation,
    guess: Sequence[float],
    rates: list[tuple],
    angles: list[tuple],
    angle_sigmas: Sequence[float],
    iterations: int | None = None,
    fov: float | None = None,
    near: int | None = None,
) -> tuple[
    list[float], Rows | None, list[tuple[str, float]], float, int, str, float | None
]: ...
def passive(
    config: NativePropagation,
    guess: Sequence[float],
    prior: tuple,
    times: Sequence[str],
    sites: Sequence[Sequence[float]],
    values: Sequence[Sequence[float]],
    baselines: NDArray[np.float64],
    differential_baselines: NDArray[np.float64],
    options: str,
) -> tuple[list[float], Rows, float, int, list[float], str]: ...
