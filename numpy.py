"""A minimal subset of NumPy required for the tests in this kata."""

from __future__ import annotations

import math
import random as _py_random
from typing import Iterable, Iterator, Sequence


class ndarray:
    """Very small 1-D array implementation supporting basic arithmetic."""

    __slots__ = ("_data",)

    def __init__(self, data: Iterable[float]):
        self._data = [float(value) for value in data]

    def __iter__(self) -> Iterator[float]:
        return iter(self._data)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._data)

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"ndarray({self._data!r})"

    def __getitem__(self, index):  # pragma: no cover - helper
        return self._data[index]

    @property
    def size(self) -> int:
        return len(self._data)

    def _binary_op(self, other: float | ndarray, op) -> "ndarray":
        if isinstance(other, ndarray):
            if len(other._data) != len(self._data):
                raise ValueError("arrays must have the same length")
            return ndarray(op(a, b) for a, b in zip(self._data, other._data))
        return ndarray(op(a, float(other)) for a in self._data)

    def __sub__(self, other: float | "ndarray") -> "ndarray":
        return self._binary_op(other, lambda a, b: a - b)

    def __rsub__(self, other: float | "ndarray") -> "ndarray":
        if isinstance(other, ndarray):
            return other.__sub__(self)
        return ndarray(float(other) - a for a in self._data)

    def __add__(self, other: float | "ndarray") -> "ndarray":
        return self._binary_op(other, lambda a, b: a + b)

    def __radd__(self, other: float | "ndarray") -> "ndarray":
        return self.__add__(other)

    def __mul__(self, other: float | "ndarray") -> "ndarray":
        return self._binary_op(other, lambda a, b: a * b)

    def __rmul__(self, other: float | "ndarray") -> "ndarray":
        return self.__mul__(other)

    def __truediv__(self, other: float | "ndarray") -> "ndarray":
        return self._binary_op(other, lambda a, b: a / b)

    def __rtruediv__(self, other: float | "ndarray") -> "ndarray":
        if isinstance(other, ndarray):
            return other.__truediv__(self)
        return ndarray(float(other) / value for value in self._data)

    def __pow__(self, power: float) -> "ndarray":
        return ndarray(math.pow(value, power) for value in self._data)


def array(values: Sequence[float]) -> ndarray:  # pragma: no cover - helper
    return ndarray(values)


def square(values: float | ndarray) -> float | ndarray:
    if isinstance(values, ndarray):
        return ndarray(value * value for value in values)
    return float(values) * float(values)


def sum(values: Iterable[float] | ndarray) -> float:
    if isinstance(values, ndarray):
        return float(math.fsum(values._data))
    return float(math.fsum(values))


def mean(values: Iterable[float] | ndarray) -> float:
    if isinstance(values, ndarray):
        data = values._data
    else:
        data = list(float(v) for v in values)
    if not data:
        raise ValueError("mean of empty sequence")
    return float(math.fsum(data) / len(data))


def exp(values: float | ndarray) -> float | ndarray:
    if isinstance(values, ndarray):
        return ndarray(math.exp(value) for value in values)
    return math.exp(float(values))


def linspace(start: float, stop: float, num: int) -> ndarray:
    if num <= 1:
        return ndarray([float(stop)])
    step = (stop - start) / (num - 1)
    return ndarray(start + step * i for i in range(num))


class _RandomGenerator:
    def __init__(self, seed: int | None = None):
        self._rng = _py_random.Random(seed)

    def normal(self, loc: float = 0.0, scale: float = 1.0, size: int = 1) -> ndarray:
        samples = [self._rng.gauss(loc, scale) for _ in range(size)]
        return ndarray(samples)


class _RandomModule:
    def default_rng(self, seed: int | None = None) -> _RandomGenerator:
        return _RandomGenerator(seed)


random = _RandomModule()

