"""Tiny dense-matrix helpers for exact PJ-rotary smoke checks."""

from __future__ import annotations

import cmath
from collections.abc import Sequence

Matrix = list[list[complex]]


def matmul(a: Matrix, b: Matrix) -> Matrix:
    if not a or not b:
        raise ValueError("matmul requires non-empty matrices")
    if len(a[0]) != len(b):
        raise ValueError("inner dimensions do not match")
    rows, cols, inner = len(a), len(b[0]), len(b)
    return [
        [sum(a[i][k] * b[k][j] for k in range(inner)) for j in range(cols)]
        for i in range(rows)
    ]


def max_abs_diff(a: Matrix, b: Matrix) -> float:
    if len(a) != len(b) or len(a[0]) != len(b[0]):
        raise ValueError("matrix shapes do not match")
    return max(abs(a[i][j] - b[i][j]) for i in range(len(a)) for j in range(len(a[0])))


def jordan2_complex(d: float, omega: float, *, eta: float = 1.0, gamma: float = 0.0) -> Matrix:
    """Complex 2x2 Jordan action exp(d*((-gamma+i*omega)I + eta*N))."""

    scale = cmath.exp((-gamma + 1j * omega) * d)
    return [[scale, scale * eta * d], [0.0 + 0.0j, scale]]


def apply_matrix(a: Matrix, x: Sequence[complex]) -> list[complex]:
    if len(a[0]) != len(x):
        raise ValueError("matrix/vector dimensions do not match")
    return [sum(row[j] * x[j] for j in range(len(x))) for row in a]

