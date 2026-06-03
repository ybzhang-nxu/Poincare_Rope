"""GRAPE exact special-case controls for appendix comparisons.

These helpers intentionally implement only three controlled special cases of
the GRAPE framework:

- GRAPE-M special case / RoPE: canonical Fourier phase features.
- GRAPE-A special case / ALiBi: affine recency features.
- GRAPE-M+A special case / RoPE+ALiBi: direct sum of the two.

They are not implementations of full learned GRAPE-M, GRAPE-A, or GRAPE-AP.
The appendix uses them as primitive-basis controls so that the comparison with
PJ-RoPE stays scoped to exact special cases and fixed function spaces.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from . import torch_backend as tb


@dataclass(frozen=True)
class GrapeMCanonicalRoPE:
    """GRAPE-M special case / RoPE scalar character basis."""

    length: float
    omega: float

    def features(self, lags: torch.Tensor) -> torch.Tensor:
        phase = self.omega * lags
        return torch.stack([torch.cos(phase), torch.sin(phase)], dim=-1)


@dataclass(frozen=True)
class GrapeAALiBi:
    """GRAPE-A special case / ALiBi affine recency basis."""

    length: float

    def features(self, lags: torch.Tensor) -> torch.Tensor:
        return (lags / float(self.length))[:, None]


@dataclass(frozen=True)
class GrapeMA:
    """GRAPE-M+A special case / RoPE+ALiBi direct-sum basis."""

    length: float
    omega: float

    def features(self, lags: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [
                GrapeMCanonicalRoPE(self.length, self.omega).features(lags),
                GrapeAALiBi(self.length).features(lags),
            ],
            dim=-1,
        )


def pj_fj_basis(
    lags: torch.Tensor,
    *,
    length: float,
    omega: float,
    max_order: int,
) -> torch.Tensor:
    """Fourier--jet basis at one frequency up to ``max_order``."""

    if max_order < 0:
        raise ValueError("max_order must be non-negative")
    x = lags / float(length)
    features = []
    phase = omega * lags
    for order in range(max_order + 1):
        scale = x.pow(order)
        features.append(scale * torch.cos(phase))
        features.append(scale * torch.sin(phase))
    return torch.stack(features, dim=-1)


def pj_lc_basis(
    lags: torch.Tensor,
    *,
    length: float,
    omega: float,
    max_order: int,
    include_affine: bool = False,
) -> torch.Tensor:
    """LC compactified Fourier--jet basis at one frequency."""

    if max_order < 0:
        raise ValueError("max_order must be non-negative")
    phi = tb.phi_l(lags, float(length))
    beta = tb.beta_l(lags, float(length))
    phase = omega * phi
    features = []
    for order in range(max_order + 1):
        scale = beta.pow(order)
        features.append(scale * torch.cos(phase))
        features.append(scale * torch.sin(phase))
    if include_affine:
        features.append(tb.eta_l(lags, float(length)))
    return torch.stack(features, dim=-1)


def target_kernel(
    name: str,
    lags: torch.Tensor,
    *,
    length: float,
    omega: float,
) -> torch.Tensor:
    """Appendix fixed-projection target kernels."""

    x = lags / float(length)
    if name == "phase":
        return torch.cos(omega * lags)
    if name == "affine":
        return -x
    if name == "first_jet":
        return x * torch.cos(omega * lags)
    if name == "second_jet":
        return x.square() * torch.cos(omega * lags)
    if name == "lc_core":
        phi = tb.phi_l(lags, float(length))
        beta = tb.beta_l(lags, float(length))
        return torch.cos(omega * phi) * (1.0 + 0.4 * beta + 0.2 * beta.square())
    raise ValueError(f"unknown target: {name}")


def solve_projection(
    features: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve a least-squares projection with an intercept column."""

    design = append_intercept(features)
    weights = torch.linalg.lstsq(design, target[:, None]).solution
    prediction = design @ weights
    return weights, prediction.squeeze(-1)


def append_intercept(features: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        [
            features,
            torch.ones(features.shape[0], 1, device=features.device, dtype=features.dtype),
        ],
        dim=-1,
    )


def mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (prediction - target).square().mean()


def r2_score(prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    residual = (prediction - target).square().sum()
    total = (target - target.mean()).square().sum()
    return 1.0 - residual / (total + eps)

