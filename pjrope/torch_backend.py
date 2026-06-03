"""PyTorch backend for PJ-RoPE scalar-bias experiments.

This module is intentionally optional: importing ``pjrope`` does not require
PyTorch, but importing this module does.
"""

from __future__ import annotations

import math
from typing import Literal

import torch

RhoType = Literal["linear", "lightcone"]


def causal_lag_matrix(
    seq_len: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``d = i - j`` and a causal mask of shape ``[seq, seq]``."""

    if seq_len < 1:
        raise ValueError("seq_len must be positive")
    idx = torch.arange(seq_len, device=device)
    lags = idx[:, None] - idx[None, :]
    mask = lags >= 0
    return lags.to(dtype=dtype), mask


def eta_l(lags: torch.Tensor, length: float | torch.Tensor) -> torch.Tensor:
    return torch.asinh(lags / _as_tensor(length, lags))


def phi_l(lags: torch.Tensor, length: float | torch.Tensor) -> torch.Tensor:
    length_t = _as_tensor(length, lags)
    return length_t * eta_l(lags, length_t)


def beta_l(lags: torch.Tensor, length: float | torch.Tensor) -> torch.Tensor:
    length_t = _as_tensor(length, lags)
    return lags / torch.sqrt(lags.square() + length_t.square())


def rho(
    lags: torch.Tensor,
    length: float | torch.Tensor,
    rho_type: RhoType = "linear",
) -> torch.Tensor:
    if rho_type == "linear":
        return lags / _as_tensor(length, lags)
    if rho_type == "lightcone":
        return eta_l(lags, length)
    raise ValueError(f"unknown rho_type: {rho_type}")


def fourier_jet_components(
    lags: torch.Tensor,
    omega: torch.Tensor,
    alpha: torch.Tensor,
    zeta_re: torch.Tensor,
    zeta_im: torch.Tensor,
    *,
    length: float,
    damping: torch.Tensor | float = 0.0,
) -> torch.Tensor:
    """Return FJ components with shape ``[heads, orders, seq, seq]``.

    Args:
        lags: Tensor of shape ``[seq, seq]``.
        omega: Per-head frequencies, shape ``[heads]``.
        alpha: Conditional order spectrum, shape ``[heads, orders]``.
        zeta_re: Real signed amplitudes, shape ``[heads, orders]``.
        zeta_im: Imaginary signed amplitudes, shape ``[heads, orders]``.
        length: Normalization length ``L``.
        damping: Per-head damping scalar, shape ``[heads]`` or scalar.
    """

    _check_order_tensors(alpha, zeta_re, zeta_im)
    heads, orders = alpha.shape
    omega = _head_vector(omega, heads, lags, "omega")
    damping = _head_vector(damping, heads, lags, "damping")
    order = torch.arange(orders, device=lags.device, dtype=lags.dtype)
    factorial = torch.tensor(
        [math.factorial(int(r)) for r in range(orders)],
        device=lags.device,
        dtype=lags.dtype,
    )

    x = (lags.clamp_min(0)[None, None, :, :] / float(length)).pow(order[None, :, None, None])
    x = x / factorial[None, :, None, None]
    env = torch.exp(-damping[:, None, None, None] * lags.clamp_min(0)[None, None, :, :] / float(length))
    phase = omega[:, None, None, None] * lags.clamp_min(0)[None, None, :, :]
    cos = torch.cos(phase)
    sin = torch.sin(phase)

    amp = alpha * zeta_re
    imp = alpha * zeta_im
    return x * env * (amp[:, :, None, None] * cos - imp[:, :, None, None] * sin)


def lightcone_components(
    lags: torch.Tensor,
    omega: torch.Tensor,
    alpha: torch.Tensor,
    zeta_re: torch.Tensor,
    zeta_im: torch.Tensor,
    *,
    length: float,
) -> torch.Tensor:
    """Return LC core components with shape ``[heads, orders, seq, seq]``."""

    _check_order_tensors(alpha, zeta_re, zeta_im)
    heads, orders = alpha.shape
    omega = _head_vector(omega, heads, lags, "omega")
    order = torch.arange(orders, device=lags.device, dtype=lags.dtype)
    clipped = lags.clamp_min(0)
    beta = beta_l(clipped, length)
    x = beta[None, None, :, :].pow(order[None, :, None, None])
    phase = omega[:, None, None, None] * phi_l(clipped, length)[None, None, :, :]
    cos = torch.cos(phase)
    sin = torch.sin(phase)

    amp = alpha * zeta_re
    imp = alpha * zeta_im
    return x * (amp[:, :, None, None] * cos - imp[:, :, None, None] * sin)


def affine_bias(
    lags: torch.Tensor,
    slope: torch.Tensor | float,
    *,
    length: float,
    rho_type: RhoType = "linear",
) -> torch.Tensor:
    """Return affine recency bias with shape ``[heads, seq, seq]``."""

    if torch.as_tensor(slope).ndim == 0:
        slope_t = torch.as_tensor([slope], device=lags.device, dtype=lags.dtype)
    else:
        slope_t = torch.as_tensor(slope, device=lags.device, dtype=lags.dtype)
    coord = rho(lags.clamp_min(0), length, rho_type)
    return -slope_t[:, None, None] * coord[None, :, :]


def apply_exact_pj_rotary(
    query: torch.Tensor,
    key: torch.Tensor,
    *,
    base: float = 10000.0,
    train_length: int = 1024,
    eta: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply first-order exact PJ-rotary transforms to Q/K tensors.

    Each 4 real dimensions form two complex channels. Queries receive the
    complex Jordan action ``G(i)`` and keys receive the dual action
    ``G(-j)^H``. Their real dot product therefore implements an exact relative
    first-jet group action, analogous to how ordinary RoPE implements relative
    rotations in the unitary special case.
    """

    if query.shape != key.shape:
        raise ValueError("query and key must have the same shape")
    if query.ndim < 2:
        raise ValueError("query/key must have shape [..., seq, head_dim]")
    head_dim = query.shape[-1]
    if head_dim % 4 != 0:
        raise ValueError("exact PJ-rotary requires head_dim divisible by 4")
    if train_length < 1:
        raise ValueError("train_length must be positive")
    block_count = head_dim // 4
    cos, sin, coord = exact_pj_rotary_factors(
        query.shape[-2],
        block_count,
        device=query.device,
        dtype=query.dtype,
        base=base,
        train_length=train_length,
        eta=eta,
        rank=query.ndim,
    )
    return (
        _apply_exact_pj_query(query, cos, sin, coord),
        _apply_exact_pj_key(key, cos, sin, coord),
    )


def exact_pj_rotary_factors(
    seq_len: int,
    block_count: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
    base: float = 10000.0,
    train_length: int = 1024,
    eta: float = 1.0,
    rank: int = 4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return broadcastable cos/sin/Jordan-coordinate factors."""

    if seq_len < 1:
        raise ValueError("seq_len must be positive")
    if block_count < 1:
        raise ValueError("block_count must be positive")
    if train_length < 1:
        raise ValueError("train_length must be positive")
    positions = torch.arange(seq_len, device=device, dtype=dtype)
    inv_freq = float(base) ** (
        -torch.arange(block_count, device=device, dtype=dtype) / float(block_count)
    )
    angles = positions[:, None] * inv_freq[None, :]
    coord = (float(eta) * positions[:, None] / float(train_length)).expand(seq_len, block_count)
    shape = (1,) * max(rank - 2, 0) + (seq_len, block_count)
    return torch.cos(angles).reshape(shape), torch.sin(angles).reshape(shape), coord.reshape(shape)


def combine_pj_bias(
    gates: torch.Tensor,
    *,
    fj_components: torch.Tensor | None = None,
    affine: torch.Tensor | None = None,
    lc_components: torch.Tensor | None = None,
    causal_mask: torch.Tensor | None = None,
    future_value: float = 0.0,
) -> torch.Tensor:
    """Combine FJ/A/LC sectors into ``[heads, seq, seq]`` scalar bias."""

    if gates.ndim != 2 or gates.shape[1] != 3:
        raise ValueError("gates must have shape [heads, 3]")
    heads = gates.shape[0]
    bias = None

    if fj_components is not None:
        _check_heads(fj_components, heads, "fj_components")
        bias = _add_or_init(bias, gates[:, 0, None, None] * fj_components.sum(dim=1))
    if affine is not None:
        _check_heads(affine, heads, "affine")
        bias = _add_or_init(bias, gates[:, 1, None, None] * affine)
    if lc_components is not None:
        _check_heads(lc_components, heads, "lc_components")
        bias = _add_or_init(bias, gates[:, 2, None, None] * lc_components.sum(dim=1))

    if bias is None:
        raise ValueError("at least one component must be provided")
    if causal_mask is not None:
        bias = bias.masked_fill(~causal_mask[None, :, :], future_value)
    return bias


def parameter_effective_mass(
    gate: torch.Tensor,
    alpha: torch.Tensor,
    zeta_re: torch.Tensor,
    zeta_im: torch.Tensor,
) -> torch.Tensor:
    """Parameter-side mass ``gate * alpha_r * |zeta_r|``."""

    _check_order_tensors(alpha, zeta_re, zeta_im)
    gate = _head_vector(gate, alpha.shape[0], alpha, "gate")
    zeta_abs = torch.sqrt(zeta_re.square() + zeta_im.square())
    return gate[:, None] * alpha * zeta_abs


def functional_energy(
    components: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Function-side energy for ``[heads, orders, ...]`` components."""

    if components.ndim < 3:
        raise ValueError("components must have shape [heads, orders, ...]")
    numerator = components.flatten(start_dim=2).norm(dim=-1)
    denominator = components.sum(dim=1).flatten(start_dim=1).norm(dim=-1)
    return numerator / (denominator[:, None] + eps)


def leave_one_order_out_mse(
    target: torch.Tensor,
    components: torch.Tensor,
) -> torch.Tensor:
    """MSE increase after removing each order from the component sum."""

    if components.ndim < 3:
        raise ValueError("components must have shape [heads, orders, ...]")
    full = components.sum(dim=1)
    target = target.to(device=components.device, dtype=components.dtype)
    if target.ndim == full.ndim - 1:
        target = target.unsqueeze(0).expand_as(full)
    if target.shape != full.shape:
        raise ValueError("target must broadcast to [heads, ...]")
    full_loss = (full - target).square().flatten(start_dim=1).mean(dim=-1)
    losses = []
    for order in range(components.shape[1]):
        without = full - components[:, order]
        loss = (without - target).square().flatten(start_dim=1).mean(dim=-1)
        losses.append(loss - full_loss)
    return torch.stack(losses, dim=1)


def apply_causal_mask(
    bias: torch.Tensor,
    mask: torch.Tensor,
    *,
    future_value: float = 0.0,
) -> torch.Tensor:
    return bias.masked_fill(~mask[None, :, :], future_value)


def _apply_exact_pj_query(
    tensor: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    coord: torch.Tensor,
) -> torch.Tensor:
    parts = tensor.reshape(*tensor.shape[:-1], tensor.shape[-1] // 4, 4)
    z0_re, z0_im, z1_re, z1_im = parts.unbind(dim=-1)
    y0_re = z0_re + coord * z1_re
    y0_im = z0_im + coord * z1_im
    y1_re = z1_re
    y1_im = z1_im
    return _pack_complex_pairs(
        _rotate_re(y0_re, y0_im, cos, sin),
        _rotate_im(y0_re, y0_im, cos, sin),
        _rotate_re(y1_re, y1_im, cos, sin),
        _rotate_im(y1_re, y1_im, cos, sin),
        tensor.shape,
    )


def _apply_exact_pj_key(
    tensor: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    coord: torch.Tensor,
) -> torch.Tensor:
    parts = tensor.reshape(*tensor.shape[:-1], tensor.shape[-1] // 4, 4)
    z0_re, z0_im, z1_re, z1_im = parts.unbind(dim=-1)
    y0_re = z0_re
    y0_im = z0_im
    y1_re = z1_re - coord * z0_re
    y1_im = z1_im - coord * z0_im
    return _pack_complex_pairs(
        _rotate_re(y0_re, y0_im, cos, sin),
        _rotate_im(y0_re, y0_im, cos, sin),
        _rotate_re(y1_re, y1_im, cos, sin),
        _rotate_im(y1_re, y1_im, cos, sin),
        tensor.shape,
    )


def _rotate_re(real: torch.Tensor, imag: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return real * cos - imag * sin


def _rotate_im(real: torch.Tensor, imag: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return real * sin + imag * cos


def _pack_complex_pairs(
    z0_re: torch.Tensor,
    z0_im: torch.Tensor,
    z1_re: torch.Tensor,
    z1_im: torch.Tensor,
    out_shape: torch.Size,
) -> torch.Tensor:
    out = torch.empty(*z0_re.shape, 4, device=z0_re.device, dtype=z0_re.dtype)
    out[..., 0] = z0_re
    out[..., 1] = z0_im
    out[..., 2] = z1_re
    out[..., 3] = z1_im
    return out.reshape(out_shape)


def _as_tensor(value: float | torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(value, device=like.device, dtype=like.dtype)


def _head_vector(value, heads: int, like: torch.Tensor, name: str) -> torch.Tensor:
    out = torch.as_tensor(value, device=like.device, dtype=like.dtype)
    if out.ndim == 0:
        out = out.expand(heads)
    if out.shape != (heads,):
        raise ValueError(f"{name} must have shape [{heads}] or be scalar")
    return out


def _check_order_tensors(alpha: torch.Tensor, zeta_re: torch.Tensor, zeta_im: torch.Tensor) -> None:
    if alpha.ndim != 2:
        raise ValueError("alpha must have shape [heads, orders]")
    if zeta_re.shape != alpha.shape or zeta_im.shape != alpha.shape:
        raise ValueError("zeta_re and zeta_im must match alpha shape")


def _check_heads(tensor: torch.Tensor, heads: int, name: str) -> None:
    if tensor.shape[0] != heads:
        raise ValueError(f"{name} must have {heads} heads")


def _add_or_init(current: torch.Tensor | None, value: torch.Tensor) -> torch.Tensor:
    if current is None:
        return value
    return current + value
