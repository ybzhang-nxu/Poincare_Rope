"""Diagnostics for PJ-RoPE scalar components."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def l2_norm(values: Sequence[float]) -> float:
    return math.sqrt(sum(float(v) * float(v) for v in values))


def mse(pred: Sequence[float], target: Sequence[float]) -> float:
    _check_same_length(pred, target)
    if not pred:
        raise ValueError("mse requires at least one value")
    return sum((float(a) - float(b)) ** 2 for a, b in zip(pred, target)) / len(pred)


def effective_mass(
    gate: float,
    alpha: Mapping[int, float],
    zeta_abs: Mapping[int, float],
) -> dict[int, float]:
    """Parameter-side effective mass: gate * alpha_r * |zeta_r|."""

    orders = set(alpha) | set(zeta_abs)
    return {r: float(gate) * float(alpha.get(r, 0.0)) * float(zeta_abs.get(r, 0.0)) for r in orders}


def normalized_mass(mass: Mapping[int, float], eps: float = 1e-12) -> dict[int, float]:
    total = sum(max(0.0, float(v)) for v in mass.values())
    return {r: max(0.0, float(v)) / (total + eps) for r, v in mass.items()}


def add_component_maps(components: Mapping[int, Sequence[float]]) -> list[float]:
    """Sum component vectors indexed by jet order."""

    if not components:
        return []
    width = len(next(iter(components.values())))
    out = [0.0] * width
    for values in components.values():
        if len(values) != width:
            raise ValueError("all component vectors must have the same length")
        for idx, value in enumerate(values):
            out[idx] += float(value)
    return out


def functional_energy(
    components: Mapping[int, Sequence[float]],
    eps: float = 1e-12,
) -> dict[int, float]:
    """Function-side energy E_r = ||C_r|| / ||sum_j C_j||."""

    total = add_component_maps(components)
    denom = l2_norm(total) + eps
    return {r: l2_norm(values) / denom for r, values in components.items()}


def leave_one_order_out_mse(
    target: Sequence[float],
    components: Mapping[int, Sequence[float]],
) -> dict[int, float]:
    """MSE increase after removing each order from the full component sum."""

    full = add_component_maps(components)
    full_loss = mse(full, target)
    out: dict[int, float] = {}
    for order, values in components.items():
        without = [a - float(b) for a, b in zip(full, values)]
        out[order] = mse(without, target) - full_loss
    return out


def _check_same_length(a: Sequence[float], b: Sequence[float]) -> None:
    if len(a) != len(b):
        raise ValueError("sequences must have the same length")

