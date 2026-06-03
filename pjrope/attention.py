"""Minimal PyTorch attention layers with PJ-RoPE scalar bias."""

from __future__ import annotations

import math
from typing import Literal

import torch

from . import torch_backend as tb

RhoType = Literal["linear", "lightcone"]


def causal_attention_mask(
    seq_len: int,
    *,
    device: torch.device | str | None = None,
    include_self: bool = True,
) -> torch.Tensor:
    """Return a causal attention mask of shape ``[seq, seq]``."""

    if seq_len < 1:
        raise ValueError("seq_len must be positive")
    diagonal = 0 if include_self else -1
    return torch.ones(seq_len, seq_len, device=device, dtype=torch.bool).tril(diagonal=diagonal)


class PJBias(torch.nn.Module):
    """Learnable scalar PJ bias for causal attention logits.

    The layer returns a tensor of shape ``[heads, seq, seq]`` that can be added
    directly to attention logits before the causal mask is applied.
    """

    def __init__(
        self,
        *,
        num_heads: int,
        max_order: int = 2,
        train_length: int = 1024,
        rho_type: RhoType = "linear",
        use_fj: bool = True,
        use_affine: bool = True,
        use_lc: bool = False,
        init_omega_cycles: float = 17.0,
    ) -> None:
        super().__init__()
        if num_heads < 1:
            raise ValueError("num_heads must be positive")
        if max_order < 0:
            raise ValueError("max_order must be non-negative")
        if train_length < 1:
            raise ValueError("train_length must be positive")
        if not (use_fj or use_affine or use_lc):
            raise ValueError("at least one PJ sector must be enabled")

        self.num_heads = num_heads
        self.max_order = max_order
        self.train_length = float(train_length)
        self.rho_type = rho_type
        self.use_fj = use_fj
        self.use_affine = use_affine
        self.use_lc = use_lc

        order_count = max_order + 1
        cycles = _head_cycles(num_heads, init_omega_cycles)
        omega = 2.0 * math.pi * cycles / float(train_length)
        self.omega = torch.nn.Parameter(omega)
        self.gate_logits = torch.nn.Parameter(torch.zeros(num_heads, 3))
        self.fj_alpha_logits = torch.nn.Parameter(torch.zeros(num_heads, order_count))
        self.lc_alpha_logits = torch.nn.Parameter(torch.zeros(num_heads, order_count))
        self.fj_zeta_re = torch.nn.Parameter(0.02 * torch.randn(num_heads, order_count))
        self.fj_zeta_im = torch.nn.Parameter(0.02 * torch.randn(num_heads, order_count))
        self.lc_zeta_re = torch.nn.Parameter(0.02 * torch.randn(num_heads, order_count))
        self.lc_zeta_im = torch.nn.Parameter(0.02 * torch.randn(num_heads, order_count))
        self.raw_fj_damping = torch.nn.Parameter(torch.full((num_heads,), -2.0))
        self.raw_affine_slope = torch.nn.Parameter(torch.zeros(num_heads))

        sector_mask = torch.tensor([use_fj, use_affine, use_lc], dtype=torch.bool)
        self.register_buffer("sector_mask", sector_mask, persistent=False)

    def sector_gates(self) -> torch.Tensor:
        logits = self.gate_logits.masked_fill(~self.sector_mask[None, :], torch.finfo(self.gate_logits.dtype).min)
        return torch.softmax(logits, dim=-1)

    def affine_slopes(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.raw_affine_slope)

    def order_masses(self) -> dict[str, torch.Tensor]:
        gates = self.sector_gates()
        masses = {}
        if self.use_fj:
            masses["fj"] = tb.parameter_effective_mass(
                gates[:, 0],
                torch.softmax(self.fj_alpha_logits, dim=-1),
                self.fj_zeta_re,
                self.fj_zeta_im,
            )
        if self.use_lc:
            masses["lc"] = tb.parameter_effective_mass(
                gates[:, 2],
                torch.softmax(self.lc_alpha_logits, dim=-1),
                self.lc_zeta_re,
                self.lc_zeta_im,
            )
        return masses

    def forward(
        self,
        seq_len: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        return_components: bool = False,
    ):
        if device is None:
            device = self.gate_logits.device
        if dtype is None:
            dtype = self.gate_logits.dtype

        lags, mask = tb.causal_lag_matrix(seq_len, device=device, dtype=dtype)
        return self.forward_lags(
            lags,
            mask,
            device=device,
            dtype=dtype,
            return_components=return_components,
        )

    def forward_lags(
        self,
        lags: torch.Tensor,
        mask: torch.Tensor,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        return_components: bool = False,
    ):
        if device is None:
            device = self.gate_logits.device
        if dtype is None:
            dtype = self.gate_logits.dtype
        lags = lags.to(device=device, dtype=dtype)
        mask = mask.to(device=device)
        gates = self.sector_gates().to(device=device, dtype=dtype)
        omega = self.omega.to(device=device, dtype=dtype)

        fj_components = None
        affine = None
        lc_components = None

        if self.use_fj:
            fj_components = tb.fourier_jet_components(
                lags,
                omega,
                torch.softmax(self.fj_alpha_logits, dim=-1).to(device=device, dtype=dtype),
                self.fj_zeta_re.to(device=device, dtype=dtype),
                self.fj_zeta_im.to(device=device, dtype=dtype),
                length=self.train_length,
                damping=torch.nn.functional.softplus(self.raw_fj_damping).to(device=device, dtype=dtype),
            )
        if self.use_affine:
            affine = tb.affine_bias(
                lags,
                self.affine_slopes().to(device=device, dtype=dtype),
                length=self.train_length,
                rho_type=self.rho_type,
            )
        if self.use_lc:
            lc_components = tb.lightcone_components(
                lags,
                omega,
                torch.softmax(self.lc_alpha_logits, dim=-1).to(device=device, dtype=dtype),
                self.lc_zeta_re.to(device=device, dtype=dtype),
                self.lc_zeta_im.to(device=device, dtype=dtype),
                length=self.train_length,
            )

        bias = tb.combine_pj_bias(
            gates,
            fj_components=fj_components,
            affine=affine,
            lc_components=lc_components,
            causal_mask=mask,
        )
        if not return_components:
            return bias
        return bias, {
            "gates": gates,
            "fj_components": fj_components,
            "affine": affine,
            "lc_components": lc_components,
        }


class PJCausalSelfAttention(torch.nn.Module):
    """Small multi-head causal attention module with optional PJ bias."""

    def __init__(
        self,
        *,
        embed_dim: int,
        num_heads: int,
        pj_bias: PJBias | None = None,
        dropout: float = 0.0,
        qkv_bias: bool = True,
        include_self: bool = True,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.include_self = include_self
        self.pj_bias = pj_bias
        self.qkv = torch.nn.Linear(embed_dim, 3 * embed_dim, bias=qkv_bias)
        self.out_proj = torch.nn.Linear(embed_dim, embed_dim)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, *, need_weights: bool = False):
        batch, seq_len, embed_dim = x.shape
        if embed_dim != self.embed_dim:
            raise ValueError(f"expected embed_dim={self.embed_dim}, got {embed_dim}")

        qkv = self.qkv(x).view(batch, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]

        scores = (query @ key.transpose(-2, -1)) * self.scale
        if self.pj_bias is not None:
            scores = scores + self.pj_bias(seq_len, device=x.device, dtype=x.dtype)[None, :, :, :]

        mask = causal_attention_mask(seq_len, device=x.device, include_self=self.include_self)
        attn = _masked_softmax(scores, mask[None, None, :, :])
        attn = self.dropout(attn)
        out = attn @ value
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, embed_dim)
        out = self.out_proj(out)
        if need_weights:
            return out, attn
        return out


def _head_cycles(num_heads: int, max_cycle: float) -> torch.Tensor:
    if num_heads == 1:
        return torch.tensor([float(max_cycle)])
    return torch.linspace(1.0, float(max_cycle), num_heads)


def _masked_softmax(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    min_value = torch.finfo(scores.dtype).min
    attn = torch.softmax(scores.masked_fill(~mask, min_value), dim=-1)
    attn = attn.masked_fill(~mask, 0.0)
    denom = attn.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(attn.dtype).tiny)
    return attn / denom
