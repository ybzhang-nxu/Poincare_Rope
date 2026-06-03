"""Backend-neutral scalar kernels for the PJ-RoPE Phase A smoke path."""

from __future__ import annotations

import math
from typing import Iterable, Literal

RhoType = Literal["linear", "lightcone"]


def eta_l(d: float, length: float) -> float:
    """Light-cone rapidity coordinate asinh(d / L)."""

    _check_length(length)
    return math.asinh(d / length)


def phi_l(d: float, length: float) -> float:
    """RoPE-matched rapidity phase coordinate L * asinh(d / L)."""

    return length * eta_l(d, length)


def beta_l(d: float, length: float) -> float:
    """Bounded light-cone velocity coordinate d / sqrt(d^2 + L^2)."""

    _check_length(length)
    return d / math.sqrt(d * d + length * length)


def rho(d: float, length: float, rho_type: RhoType = "linear") -> float:
    """Affine recency coordinate."""

    if rho_type == "linear":
        _check_length(length)
        return d / length
    if rho_type == "lightcone":
        return eta_l(d, length)
    raise ValueError(f"unknown rho_type: {rho_type}")


def fj_component(
    d: float,
    omega: float,
    jet_order: int,
    length: float,
    *,
    damping: float = 0.0,
    amplitude: complex = 1.0 + 0.0j,
) -> float:
    """One real Fourier-jet scalar component."""

    _check_order(jet_order)
    _check_length(length)
    x = (d / length) ** jet_order / math.factorial(jet_order)
    env = math.exp(-damping * d / length)
    phase = _cis(omega * d)
    return (amplitude * x * env * phase).real


def lc_component(
    d: float,
    omega: float,
    jet_order: int,
    length: float,
    *,
    amplitude: complex = 1.0 + 0.0j,
) -> float:
    """One real light-cone compactified phase/jet component."""

    _check_order(jet_order)
    x = beta_l(d, length) ** jet_order
    phase = _cis(omega * phi_l(d, length))
    return (amplitude * x * phase).real


def affine_component(
    d: float,
    length: float,
    *,
    slope: float = 1.0,
    rho_type: RhoType = "linear",
) -> float:
    """Affine / recency scalar component."""

    return -slope * rho(d, length, rho_type)


def distance_lags(seq_len: int, *, causal: bool = True) -> list[list[int | None]]:
    """Return d = i - j lag matrix; masked future entries are None."""

    if seq_len < 1:
        raise ValueError("seq_len must be positive")
    rows: list[list[int | None]] = []
    for i in range(seq_len):
        row: list[int | None] = []
        for j in range(seq_len):
            if causal and j > i:
                row.append(None)
            else:
                row.append(i - j)
        rows.append(row)
    return rows


def values_over_lags(fn, lags: Iterable[int]) -> list[float]:
    """Evaluate a scalar lag function over integer lags."""

    return [float(fn(d)) for d in lags]


def _cis(theta: float) -> complex:
    return complex(math.cos(theta), math.sin(theta))


def _check_length(length: float) -> None:
    if length <= 0:
        raise ValueError("length must be positive")


def _check_order(jet_order: int) -> None:
    if jet_order < 0:
        raise ValueError("jet_order must be non-negative")

