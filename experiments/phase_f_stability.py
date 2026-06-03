"""Phase F light-cone stability and phase-resolution diagnostics.

This script is intentionally kernel/coordinate-level.  It measures the
long-context behavior that is expensive to isolate inside a trained LM:
high-order coordinate growth, scalar-bias softmax collapse, quantization
sensitivity, and far-window phase resolution.
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
        "/home/riven/JordanKac/.venv/bin/python experiments/phase_f_stability.py"
    ) from exc

from pjrope import torch_backend as tb


FJ_STYLE_VARIANTS = {"raw", "scaled", "clipped", "log", "lc_phase_only"}
LC_AMP_VARIANTS = {"lc_amp_only", "lc", "lc_wrong_scale"}


@dataclass(frozen=True)
class StabilityRow:
    variant: str
    eval_len: int
    train_len: int
    max_order: int
    omega_cycles: float
    damping: float
    bias_abs_max: float
    bias_rms: float
    weighted_logit_std: float
    final_attention_entropy: float
    final_effective_support: float
    final_average_distance: float
    qk_norm_proxy_max: float
    qk_norm_proxy_final: float
    qk_norm_proxy_mean: float
    feature_abs_max: float
    int8_quant_error_tensor: float
    int8_quant_error_per_order: float
    int4_quant_error_tensor: float
    int4_quant_error_per_order: float
    local_phase_ratio_final: float
    local_phase_ratio_far_mean: float
    far_phase_span_radians: float
    far_local_collision_rate: float
    wrapped_phase_collision_rate: float
    wrapped_phase_nearest_median: float
    far_gram_condition_number: float
    nonfinite_count: int


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dtype = torch.float64 if args.float64 else torch.float32
    eval_lens = parse_int_list(args.eval_lens)
    variants = parse_str_list(args.variants)
    omega = 2.0 * math.pi * args.omega_cycles / args.train_len

    rows: list[StabilityRow] = []
    for variant in variants:
        validate_variant(variant)
        for eval_len in eval_lens:
            rows.append(
                run_variant_length(
                    variant,
                    eval_len=eval_len,
                    train_len=args.train_len,
                    max_order=args.max_order,
                    omega=omega,
                    omega_cycles=args.omega_cycles,
                    damping=args.damping,
                    bias_scale=args.bias_scale,
                    phase_collision_threshold=args.phase_collision_threshold,
                    far_samples=args.far_samples,
                    dtype=dtype,
                    device=torch.device(args.device),
                )
            )

    write_outputs(rows, args.out_dir, args)
    print(f"Wrote Phase F stability diagnostics to: {args.out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-len", type=int, default=1024)
    parser.add_argument("--eval-lens", default="1024,2048,4096,8192,16384,32768")
    parser.add_argument("--max-order", type=int, default=3)
    parser.add_argument("--omega-cycles", type=float, default=17.0)
    parser.add_argument("--damping", type=float, default=0.1)
    parser.add_argument("--bias-scale", type=float, default=1.0)
    parser.add_argument(
        "--variants",
        default="raw,scaled,clipped,log,lc_amp_only,lc_phase_only,lc,lc_wrong_scale",
    )
    parser.add_argument("--far-samples", type=int, default=4096)
    parser.add_argument("--phase-collision-threshold", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--float64", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=Path("runs/phase_f_stability"))
    return parser.parse_args()


def run_variant_length(
    variant: str,
    *,
    eval_len: int,
    train_len: int,
    max_order: int,
    omega: float,
    omega_cycles: float,
    damping: float,
    bias_scale: float,
    phase_collision_threshold: float,
    far_samples: int,
    dtype: torch.dtype,
    device: torch.device,
) -> StabilityRow:
    if eval_len < 2:
        raise ValueError("eval_len must be at least 2")
    d = torch.arange(eval_len, device=device, dtype=dtype)
    phase = phase_coordinate(variant, d, train_len)
    components = jet_feature_components(variant, d, train_len, max_order, damping)
    norm_proxy = components.norm(dim=1)
    kernel = scalar_kernel(variant, d, train_len, max_order, omega, damping)
    logits = bias_scale * kernel

    weighted_std = weighted_logit_std(logits)
    entropy, support, avg_distance = final_attention_stats(logits, d)
    int8_tensor = symmetric_quant_error(components, bits=8, per_order=False)
    int8_order = symmetric_quant_error(components, bits=8, per_order=True)
    int4_tensor = symmetric_quant_error(components, bits=4, per_order=False)
    int4_order = symmetric_quant_error(components, bits=4, per_order=True)
    phase_metrics = phase_resolution_metrics(
        variant,
        eval_len=eval_len,
        train_len=train_len,
        max_order=max_order,
        omega=omega,
        damping=damping,
        threshold=phase_collision_threshold,
        far_samples=far_samples,
        dtype=dtype,
        device=device,
    )
    nonfinite = sum(
        int((~torch.isfinite(tensor)).sum().detach().cpu())
        for tensor in (components, kernel, logits, phase)
    )

    return StabilityRow(
        variant=variant,
        eval_len=eval_len,
        train_len=train_len,
        max_order=max_order,
        omega_cycles=omega_cycles,
        damping=damping,
        bias_abs_max=float(kernel.detach().abs().max().cpu()),
        bias_rms=float(kernel.detach().square().mean().sqrt().cpu()),
        weighted_logit_std=float(weighted_std.detach().cpu()),
        final_attention_entropy=float(entropy.detach().cpu()),
        final_effective_support=float(support.detach().cpu()),
        final_average_distance=float(avg_distance.detach().cpu()),
        qk_norm_proxy_max=float(norm_proxy.detach().max().cpu()),
        qk_norm_proxy_final=float(norm_proxy[-1].detach().cpu()),
        qk_norm_proxy_mean=float(norm_proxy.detach().mean().cpu()),
        feature_abs_max=float(components.detach().abs().max().cpu()),
        int8_quant_error_tensor=float(int8_tensor.detach().cpu()),
        int8_quant_error_per_order=float(int8_order.detach().cpu()),
        int4_quant_error_tensor=float(int4_tensor.detach().cpu()),
        int4_quant_error_per_order=float(int4_order.detach().cpu()),
        local_phase_ratio_final=phase_metrics["local_phase_ratio_final"],
        local_phase_ratio_far_mean=phase_metrics["local_phase_ratio_far_mean"],
        far_phase_span_radians=phase_metrics["far_phase_span_radians"],
        far_local_collision_rate=phase_metrics["far_local_collision_rate"],
        wrapped_phase_collision_rate=phase_metrics["wrapped_phase_collision_rate"],
        wrapped_phase_nearest_median=phase_metrics["wrapped_phase_nearest_median"],
        far_gram_condition_number=phase_metrics["far_gram_condition_number"],
        nonfinite_count=nonfinite,
    )


def phase_coordinate(variant: str, d: torch.Tensor, train_len: int) -> torch.Tensor:
    if variant in {"lc", "lc_phase_only"}:
        return tb.phi_l(d, float(train_len))
    if variant == "lc_wrong_scale":
        return tb.eta_l(d, float(train_len))
    return d


def amplitude_coordinate(variant: str, d: torch.Tensor, train_len: int) -> torch.Tensor:
    x = d / float(train_len)
    if variant == "raw":
        return d
    if variant == "scaled":
        return x
    if variant == "clipped":
        return x.clamp_max(1.0)
    if variant == "log":
        return torch.log1p(x)
    if variant == "lc_phase_only":
        return x
    if variant in LC_AMP_VARIANTS:
        return tb.beta_l(d, float(train_len))
    raise ValueError(f"unknown variant: {variant}")


def envelope(variant: str, d: torch.Tensor, train_len: int, damping: float) -> torch.Tensor:
    if variant == "scaled":
        return torch.exp(-float(damping) * d / float(train_len))
    return torch.ones_like(d)


def jet_feature_components(
    variant: str,
    d: torch.Tensor,
    train_len: int,
    max_order: int,
    damping: float,
) -> torch.Tensor:
    coord = amplitude_coordinate(variant, d, train_len)
    env = envelope(variant, d, train_len, damping)
    pieces = []
    for order in range(max_order + 1):
        component = coord.pow(order)
        if variant in FJ_STYLE_VARIANTS:
            component = component / math.factorial(order)
        pieces.append(env * component)
    return torch.stack(pieces, dim=1)


def scalar_kernel(
    variant: str,
    d: torch.Tensor,
    train_len: int,
    max_order: int,
    omega: float,
    damping: float,
) -> torch.Tensor:
    coord = amplitude_coordinate(variant, d, train_len)
    amp = coord.pow(max_order)
    if variant in FJ_STYLE_VARIANTS:
        amp = amp / math.factorial(max_order)
    amp = amp * envelope(variant, d, train_len, damping)
    phase = phase_coordinate(variant, d, train_len)
    return amp * torch.cos(float(omega) * phase)


def weighted_logit_std(logits: torch.Tensor) -> torch.Tensor:
    # A causal lag d appears T-d times in a dense T x T causal logit table.
    t = logits.numel()
    counts = torch.arange(t, 0, -1, device=logits.device, dtype=logits.dtype)
    total = counts.sum()
    mean = (counts * logits).sum() / total
    var = (counts * (logits - mean).square()).sum() / total
    return var.sqrt()


def final_attention_stats(logits: torch.Tensor, d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    probs = torch.softmax(logits, dim=0)
    tiny = torch.finfo(probs.dtype).tiny
    entropy = -(probs * probs.clamp_min(tiny).log()).sum()
    support = entropy.exp()
    avg_distance = (probs * d).sum()
    return entropy, support, avg_distance


def symmetric_quant_error(x: torch.Tensor, *, bits: int, per_order: bool, eps: float = 1e-12) -> torch.Tensor:
    qmax = float(2 ** (bits - 1) - 1)
    if per_order:
        scale = x.detach().abs().amax(dim=0, keepdim=True).clamp_min(eps) / qmax
    else:
        scale = x.detach().abs().max().clamp_min(eps) / qmax
    quantized = (x / scale).round().clamp(-qmax, qmax) * scale
    return (quantized - x).norm() / x.norm().clamp_min(eps)


def phase_resolution_metrics(
    variant: str,
    *,
    eval_len: int,
    train_len: int,
    max_order: int,
    omega: float,
    damping: float,
    threshold: float,
    far_samples: int,
    dtype: torch.dtype,
    device: torch.device,
) -> dict[str, float]:
    d = torch.arange(eval_len, device=device, dtype=dtype)
    phase = phase_coordinate(variant, d, train_len)
    increments = phase[1:] - phase[:-1]
    far_start = eval_len // 2
    far_increments = increments[far_start:].abs()
    local_final = float(increments[-1].detach().cpu())
    local_far_mean = float(far_increments.mean().detach().cpu())
    far_span = float((omega * (phase[-1] - phase[far_start])).abs().detach().cpu())
    far_local_collision = float(((omega * far_increments) < threshold).to(dtype).mean().detach().cpu())

    sampled_d = sampled_far_lags(eval_len, far_samples, device=device, dtype=dtype)
    sampled_phase = phase_coordinate(variant, sampled_d, train_len)
    theta = torch.remainder(omega * sampled_phase, 2.0 * math.pi)
    nearest = wrapped_nearest_distances(theta)
    wrapped_collision = float((nearest < threshold).to(dtype).mean().detach().cpu())
    wrapped_median = float(nearest.median().detach().cpu())
    condition = far_gram_condition_number(
        variant,
        sampled_d,
        train_len=train_len,
        max_order=max_order,
        omega=omega,
        damping=damping,
    )
    return {
        "local_phase_ratio_final": local_final,
        "local_phase_ratio_far_mean": local_far_mean,
        "far_phase_span_radians": far_span,
        "far_local_collision_rate": far_local_collision,
        "wrapped_phase_collision_rate": wrapped_collision,
        "wrapped_phase_nearest_median": wrapped_median,
        "far_gram_condition_number": condition,
    }


def sampled_far_lags(eval_len: int, samples: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    far_start = eval_len // 2
    count = min(max(2, samples), eval_len - far_start)
    idx = torch.linspace(far_start, eval_len - 1, count, device=device, dtype=dtype)
    return idx.round().unique()


def wrapped_nearest_distances(theta: torch.Tensor) -> torch.Tensor:
    if theta.numel() < 2:
        return torch.zeros_like(theta)
    sorted_theta = theta.sort().values
    gaps = sorted_theta[1:] - sorted_theta[:-1]
    wrap_gap = (sorted_theta[0] + 2.0 * math.pi - sorted_theta[-1]).unsqueeze(0)
    all_gaps = torch.cat([gaps, wrap_gap])
    prev_gap = torch.roll(all_gaps, shifts=1)
    return torch.minimum(all_gaps, prev_gap)


def far_gram_condition_number(
    variant: str,
    d: torch.Tensor,
    *,
    train_len: int,
    max_order: int,
    omega: float,
    damping: float,
    eps: float = 1e-12,
) -> float:
    coord = amplitude_coordinate(variant, d, train_len)
    phase = phase_coordinate(variant, d, train_len)
    env = envelope(variant, d, train_len, damping=damping)
    cols = []
    for order in range(max_order + 1):
        amp = coord.pow(order)
        if variant in FJ_STYLE_VARIANTS:
            amp = amp / math.factorial(order)
        amp = amp * env
        cols.append(amp * torch.cos(omega * phase))
        cols.append(amp * torch.sin(omega * phase))
    design = torch.stack(cols, dim=1)
    design = (design - design.mean(dim=0, keepdim=True)) / design.std(dim=0, keepdim=True).clamp_min(eps)
    singular = torch.linalg.svdvals(design.to(torch.float64))
    return float((singular.max() / singular.min().clamp_min(eps)).detach().cpu())


def write_outputs(rows: list[StabilityRow], out_dir: Path, args: argparse.Namespace) -> None:
    rows_as_dicts = [asdict(row) for row in rows]
    with (out_dir / "results.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows_as_dicts[0].keys()))
        writer.writeheader()
        writer.writerows(rows_as_dicts)

    final_len = max(row.eval_len for row in rows)
    snapshot = {
        row.variant: asdict(row)
        for row in rows
        if row.eval_len == final_len
    }
    summary = {
        "config": {
            "train_len": args.train_len,
            "eval_lens": parse_int_list(args.eval_lens),
            "max_order": args.max_order,
            "omega_cycles": args.omega_cycles,
            "damping": args.damping,
            "bias_scale": args.bias_scale,
            "phase_collision_threshold": args.phase_collision_threshold,
            "far_samples": args.far_samples,
            "variants": parse_str_list(args.variants),
        },
        "final_eval_len": final_len,
        "snapshot": snapshot,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_readme(rows, out_dir, final_len)


def write_readme(rows: list[StabilityRow], out_dir: Path, final_len: int) -> None:
    lines = [
        "# Phase F Stability Diagnostics",
        "",
        "Coordinate-level diagnostics for high-order PJ variants.",
        "",
        "## Files",
        "",
        "- `results.csv`: one row per variant and evaluation length.",
        "- `summary.json`: configuration and final-length snapshot.",
        "",
        "## Final-Length Snapshot",
        "",
        "| Variant | QK norm final | Bias max | Entropy | Support | AvgDist | Int8 err | Phase ratio final | Far span | Local collision | Wrapped collision | Gram cond |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted((row for row in rows if row.eval_len == final_len), key=lambda item: item.variant):
        lines.append(
            "| `{variant}` | `{qk:.3e}` | `{bias:.3e}` | `{entropy:.3f}` | `{support:.2f}` | "
            "`{dist:.2f}` | `{qerr:.3e}` | `{ratio:.3e}` | `{span:.2f}` | `{local:.2f}` | "
            "`{wrapped:.2f}` | `{cond:.3e}` |".format(
                variant=row.variant,
                qk=row.qk_norm_proxy_final,
                bias=row.bias_abs_max,
                entropy=row.final_attention_entropy,
                support=row.final_effective_support,
                dist=row.final_average_distance,
                qerr=row.int8_quant_error_tensor,
                ratio=row.local_phase_ratio_final,
                span=row.far_phase_span_radians,
                local=row.far_local_collision_rate,
                wrapped=row.wrapped_phase_collision_rate,
                cond=row.far_gram_condition_number,
            )
        )
    lines.extend(
        [
            "",
            "Notes:",
            "",
            "- `raw` uses an unnormalized `d^r / r!` FJ coordinate and is expected to explode.",
            "- `scaled` uses `(d/L)^r exp(-c d/L) / r!` with RoPE phase.",
            "- `clipped` and `log` are engineering stabilizers with RoPE phase.",
            "- `lc_amp_only` uses bounded `beta_L(d)^r` amplitude but ordinary RoPE phase.",
            "- `lc_phase_only` uses LC phase with scaled FJ amplitude.",
            "- `lc` uses both LC phase `L asinh(d/L)` and bounded `beta_L(d)^r` amplitude.",
            "- `lc_wrong_scale` is the negative control with phase `asinh(d/L)` instead of `L asinh(d/L)`.",
        ]
    )
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_variant(variant: str) -> None:
    allowed = FJ_STYLE_VARIANTS | LC_AMP_VARIANTS
    if variant not in allowed:
        raise ValueError(f"unknown variant: {variant}; allowed={sorted(allowed)}")


def parse_int_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in str(raw).split(",") if item.strip()]


def parse_str_list(raw: str) -> list[str]:
    return [item.strip() for item in str(raw).split(",") if item.strip()]


if __name__ == "__main__":
    main()
