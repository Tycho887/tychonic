"""Conversions shared by the public Python facade."""

from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Any

import numpy as np
import satkit
from numpy.typing import ArrayLike, NDArray


def time_string(value: satkit.time) -> str:
    if not isinstance(value, satkit.time):
        raise TypeError("expected satkit.time; convert other time types explicitly")
    return value.to_rfc3339()


def time_object(value: str) -> satkit.time:
    return satkit.time.from_rfc3339(value)


def dumps(value: Any) -> str:
    return json.dumps(value, allow_nan=False)


def array(value: ArrayLike, *, ndim: int | None = None) -> NDArray[np.float64]:
    result = np.array(value, dtype=np.float64, copy=True)
    if ndim is not None and result.ndim != ndim:
        raise ValueError(f"expected a {ndim}-dimensional array")
    if not np.isfinite(result).all():
        raise ValueError("numeric inputs must be finite")
    return result


def process_covariance(value: ArrayLike | None, dimension: int) -> NDArray[np.float64]:
    if value is None:
        return np.zeros((dimension, dimension))
    result = array(value, ndim=2)
    if result.shape != (dimension, dimension):
        raise ValueError(
            f"expected process covariance shape ({dimension}, {dimension})"
        )
    return result


class Stateful:
    """Reject reentrant/concurrent mutations, including calls from callbacks."""

    _busy = False

    @contextmanager
    def _mutation(self):
        if self._busy:
            raise RuntimeError("a mutation of this object is already in progress")
        self._busy = True
        try:
            yield
        finally:
            self._busy = False
