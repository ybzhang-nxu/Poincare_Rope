"""Configuration objects for PJ-RoPE experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


RhoType = Literal["linear", "lightcone"]


@dataclass(frozen=True)
class PJRopeConfig:
    """Small, backend-neutral config for the first implementation pass."""

    num_heads: int = 1
    block_size: int = 3
    train_length: int = 1024
    rho_type: RhoType = "linear"
    use_fj: bool = True
    use_affine: bool = True
    use_lc: bool = False
    report_effective_mass: bool = True
    report_functional_energy: bool = True

    @property
    def max_jet_order(self) -> int:
        return self.block_size - 1

    def validate(self) -> None:
        if self.num_heads < 1:
            raise ValueError("num_heads must be positive")
        if self.block_size < 1:
            raise ValueError("block_size must be positive")
        if self.train_length < 1:
            raise ValueError("train_length must be positive")
        if self.rho_type not in ("linear", "lightcone"):
            raise ValueError(f"unknown rho_type: {self.rho_type}")

