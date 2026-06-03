"""Phase F feature-code KV-cache quantization proxy.

This is a no-training diagnostic that turns positional feature codes into a
random transformed-key cache.  It is not a full exact PJ-rotary implementation;
instead it probes the practical quantities requested by the Phase F plan:
transformed key norm growth, final-query logit scale, and cache quantization
error under the same raw/scaled/LC coordinate ablations.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import torch
except Exception as exc:  # pragma: no cover - command-line guard
    raise SystemExit(
        "This experiment requires PyTorch. Try:\n"
        "/home/riven/JordanKac/.venv/bin/python experiments/phase_f_cache_quantization.py"
    ) from exc

from experiments.phase_f_retrieval_resolution import (
    parse_float_list,
    parse_int_list,
    parse_str_list,
    position_features,
    validate_variant,
)


@dataclass(frozen=True)
class CacheQuantRow:
    variant: str
    eval_len: int
    train_len: int
    max_order: int
    cache_dim: int
    feature_dim: int
    omega_cycles: str
    seed: int
    key_norm_max: float
    key_norm_final: float
    key_norm_mean: float
    key_norm_far_mean: float
    cache_abs_max: float
    logit_abs_max: float
    logit_std: float
    int8_cache_error_tensor: float
    int8_cache_error_channel: float
    int8_cache_error_far_tensor: float
    int8_logit_rel_error_tensor: float
    int8_logit_rel_error_channel: float
    int4_cache_error_tensor: float
    int4_cache_error_channel: float
    int4_cache_error_far_tensor: float
    int4_logit_rel_error_tensor: float
    int4_logit_rel_error_channel: float
    nonfinite_count: int


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dtype = torch.float64 if args.float64 else torch.float32
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    eval_lens = parse_int_list(args.eval_lens)
    variants = parse_str_list(args.variants)
    omega_cycles = parse_float_list(args.omega_cycles)
    omegas = torch.tensor(
        [2.0 * math.pi * cycles / args.train_len for cycles in omega_cycles],
        device=device,
        dtype=dtype,
    )
    feature_dim = len(omega_cycles) * (args.max_order + 1) * 2
    key_mix, query_mix = make_random_mixes(feature_dim, args.cache_dim, device=device, dtype=dtype, seed=args.seed)

    rows: list[CacheQuantRow] = []
    for variant in variants:
        validate_variant(variant)
        for eval_len in eval_lens:
            rows.append(
                run_variant_length(
                    variant,
                    eval_len=eval_len,
                    train_len=args.train_len,
                    max_order=args.max_order,
                    cache_dim=args.cache_dim,
                    omegas=omegas,
                    omega_cycles=args.omega_cycles,
                    damping=args.damping,
                    seed=args.seed,
                    key_mix=key_mix,
                    query_mix=query_mix,
                    dtype=dtype,
                    device=device,
                )
            )

    write_outputs(rows, args.out_dir, args)
    print(f"Wrote Phase F cache quantization diagnostics to: {args.out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-len", type=int, default=1024)
    parser.add_argument("--eval-lens", default="1024,2048,4096,8192,16384,32768")
    parser.add_argument("--max-order", type=int, default=3)
    parser.add_argument("--omega-cycles", default="5,17,61")
    parser.add_argument("--damping", type=float, default=0.1)
    parser.add_argument(
        "--variants",
        default="raw,scaled,clipped,log,lc_amp_only,lc_phase_only,lc,lc_wrong_scale",
    )
    parser.add_argument("--cache-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--float64", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=Path("runs/phase_f_cache_quantization"))
    return parser.parse_args()


def make_random_mixes(
    feature_dim: int,
    cache_dim: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    scale = 1.0 / math.sqrt(feature_dim)
    key_mix = torch.randn(feature_dim, cache_dim, generator=generator, device=device, dtype=dtype) * scale
    query_mix = torch.randn(feature_dim, cache_dim, generator=generator, device=device, dtype=dtype) * scale
    return key_mix, query_mix


def run_variant_length(
    variant: str,
    *,
    eval_len: int,
    train_len: int,
    max_order: int,
    cache_dim: int,
    omegas: torch.Tensor,
    omega_cycles: str,
    damping: float,
    seed: int,
    key_mix: torch.Tensor,
    query_mix: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
) -> CacheQuantRow:
    positions = torch.arange(eval_len, device=device, dtype=dtype)
    features = position_features(
        variant,
        positions,
        train_len=train_len,
        max_order=max_order,
        omegas=omegas,
        damping=damping,
    )
    keys = features @ key_mix
    final_query = features[-1] @ query_mix
    scores = logits_from_cache(keys, final_query)
    key_norms = keys.norm(dim=1)

    int8_tensor = quantize_cache(keys, bits=8, per_channel=False)
    int8_channel = quantize_cache(keys, bits=8, per_channel=True)
    int4_tensor = quantize_cache(keys, bits=4, per_channel=False)
    int4_channel = quantize_cache(keys, bits=4, per_channel=True)

    far = slice(eval_len // 2, eval_len)
    nonfinite_count = sum(
        int((~torch.isfinite(tensor)).sum().detach().cpu())
        for tensor in (
            features,
            keys,
            final_query,
            scores,
            int8_tensor.quantized,
            int8_channel.quantized,
            int4_tensor.quantized,
            int4_channel.quantized,
        )
    )

    return CacheQuantRow(
        variant=variant,
        eval_len=eval_len,
        train_len=train_len,
        max_order=max_order,
        cache_dim=cache_dim,
        feature_dim=int(features.shape[1]),
        omega_cycles=omega_cycles,
        seed=seed,
        key_norm_max=float(key_norms.detach().max().cpu()),
        key_norm_final=float(key_norms[-1].detach().cpu()),
        key_norm_mean=float(key_norms.detach().mean().cpu()),
        key_norm_far_mean=float(key_norms[far].detach().mean().cpu()),
        cache_abs_max=float(keys.detach().abs().max().cpu()),
        logit_abs_max=float(scores.detach().abs().max().cpu()),
        logit_std=float(scores.detach().std().cpu()),
        int8_cache_error_tensor=float(int8_tensor.relative_error.detach().cpu()),
        int8_cache_error_channel=float(int8_channel.relative_error.detach().cpu()),
        int8_cache_error_far_tensor=float(relative_error(int8_tensor.quantized[far], keys[far]).detach().cpu()),
        int8_logit_rel_error_tensor=float(logit_relative_error(int8_tensor.quantized, keys, final_query).detach().cpu()),
        int8_logit_rel_error_channel=float(logit_relative_error(int8_channel.quantized, keys, final_query).detach().cpu()),
        int4_cache_error_tensor=float(int4_tensor.relative_error.detach().cpu()),
        int4_cache_error_channel=float(int4_channel.relative_error.detach().cpu()),
        int4_cache_error_far_tensor=float(relative_error(int4_tensor.quantized[far], keys[far]).detach().cpu()),
        int4_logit_rel_error_tensor=float(logit_relative_error(int4_tensor.quantized, keys, final_query).detach().cpu()),
        int4_logit_rel_error_channel=float(logit_relative_error(int4_channel.quantized, keys, final_query).detach().cpu()),
        nonfinite_count=nonfinite_count,
    )


@dataclass(frozen=True)
class QuantizedCache:
    quantized: torch.Tensor
    relative_error: torch.Tensor


def quantize_cache(keys: torch.Tensor, *, bits: int, per_channel: bool, eps: float = 1e-12) -> QuantizedCache:
    qmax = float(2 ** (bits - 1) - 1)
    if per_channel:
        scale = keys.detach().abs().amax(dim=0, keepdim=True).clamp_min(eps) / qmax
    else:
        scale = keys.detach().abs().max().clamp_min(eps) / qmax
    quantized = (keys / scale).round().clamp(-qmax, qmax) * scale
    return QuantizedCache(quantized=quantized, relative_error=relative_error(quantized, keys))


def logits_from_cache(keys: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
    values = keys.to(torch.float64) @ query.to(torch.float64)
    return (values / math.sqrt(keys.shape[1])).to(torch.float64)


def logit_relative_error(quantized_keys: torch.Tensor, keys: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
    scores = logits_from_cache(keys, query)
    quantized_scores = logits_from_cache(quantized_keys, query)
    return relative_error(quantized_scores, scores)


def relative_error(estimate: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    estimate64 = estimate.to(torch.float64)
    target64 = target.to(torch.float64)
    return (estimate64 - target64).norm() / target64.norm().clamp_min(eps)


def write_outputs(rows: list[CacheQuantRow], out_dir: Path, args: argparse.Namespace) -> None:
    rows_as_dicts = [asdict(row) for row in rows]
    with (out_dir / "results.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows_as_dicts[0].keys()))
        writer.writeheader()
        writer.writerows(rows_as_dicts)

    final_len = max(row.eval_len for row in rows)
    snapshot = [
        asdict(row)
        for row in rows
        if row.eval_len == final_len
    ]
    summary = {
        "config": {
            "train_len": args.train_len,
            "eval_lens": parse_int_list(args.eval_lens),
            "max_order": args.max_order,
            "omega_cycles": parse_float_list(args.omega_cycles),
            "damping": args.damping,
            "variants": parse_str_list(args.variants),
            "cache_dim": args.cache_dim,
            "seed": args.seed,
        },
        "final_eval_len": final_len,
        "snapshot": snapshot,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_readme(rows, out_dir, final_len)


def write_readme(rows: list[CacheQuantRow], out_dir: Path, final_len: int) -> None:
    lines = [
        "# Phase F Cache Quantization",
        "",
        "Feature-code KV-cache proxy for Phase F coordinate variants.",
        "",
        "## Files",
        "",
        "- `results.csv`: one row per variant and evaluation length.",
        "- `summary.json`: configuration and final-length snapshot.",
        "",
        "## Final-Length Snapshot",
        "",
        "| Variant | KeyNormFinal | KeyNormMax | LogitStd | Int8 cache | Int8 logit | Int4 cache | Int4 logit |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted((row for row in rows if row.eval_len == final_len), key=lambda item: item.variant):
        lines.append(
            "| `{variant}` | `{norm_final:.3e}` | `{norm_max:.3e}` | `{logit_std:.3e}` | "
            "`{i8_cache:.3e}` | `{i8_logit:.3e}` | `{i4_cache:.3e}` | `{i4_logit:.3e}` |".format(
                variant=row.variant,
                norm_final=row.key_norm_final,
                norm_max=row.key_norm_max,
                logit_std=row.logit_std,
                i8_cache=row.int8_cache_error_tensor,
                i8_logit=row.int8_logit_rel_error_tensor,
                i4_cache=row.int4_cache_error_tensor,
                i4_logit=row.int4_logit_rel_error_tensor,
            )
        )
    lines.extend(
        [
            "",
            "Notes:",
            "",
            "- This is a feature-code proxy, not a full exact PJ-rotary transform.",
            "- Key norm/logit scale are the main instability indicators.",
            "- Cache quantization is measured on the transformed key tensor, with per-tensor and per-channel variants in `results.csv`.",
        ]
    )
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
