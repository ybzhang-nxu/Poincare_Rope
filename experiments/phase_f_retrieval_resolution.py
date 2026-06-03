"""Phase F retrieval-style resolution probe for LC ablations.

The probe uses the same raw/scaled/LC coordinate variants as
``phase_f_stability.py``.  It builds positional feature codes for all lags in a
context window and asks whether a target lag can be recovered from the full
candidate set by nearest-neighbor similarity, optionally after int8/int4
quantization.

This is a no-training diagnostic: it measures whether phase compression and
coordinate stabilization leave enough distinguishability for retrieval-like
far-distance use.
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
    import torch.nn.functional as F
except Exception as exc:  # pragma: no cover - command-line guard
    raise SystemExit(
        "This experiment requires PyTorch. Try:\n"
        "/home/riven/JordanKac/.venv/bin/python experiments/phase_f_retrieval_resolution.py"
    ) from exc

from experiments.phase_f_stability import (
    FJ_STYLE_VARIANTS,
    LC_AMP_VARIANTS,
    amplitude_coordinate,
    envelope,
    phase_coordinate,
    validate_variant,
)


@dataclass(frozen=True)
class RetrievalRow:
    variant: str
    precision: str
    eval_len: int
    train_len: int
    bucket: str
    min_lag: int
    max_lag: int
    n_queries: int
    candidate_count: int
    feature_dim: int
    top1_exact: float
    within_1: float
    within_8: float
    within_64: float
    bucket_accuracy: float
    mean_abs_error: float
    median_abs_error: float
    mean_target_score: float
    mean_best_nonself_score: float
    mean_margin: float
    min_margin: float
    mean_tie_count: float


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dtype = torch.float64 if args.float64 else torch.float32
    device = torch.device(args.device)
    eval_lens = parse_int_list(args.eval_lens)
    variants = parse_str_list(args.variants)
    precisions = parse_str_list(args.precisions)
    omega_cycles = parse_float_list(args.omega_cycles)
    omegas = torch.tensor(
        [2.0 * math.pi * cycles / args.train_len for cycles in omega_cycles],
        device=device,
        dtype=dtype,
    )

    rows: list[RetrievalRow] = []
    for variant in variants:
        validate_variant(variant)
        for eval_len in eval_lens:
            candidate_lags = torch.arange(1, eval_len, device=device, dtype=dtype)
            features = position_features(
                variant,
                candidate_lags,
                train_len=args.train_len,
                max_order=args.max_order,
                omegas=omegas,
                damping=args.damping,
            )
            for precision in precisions:
                candidate_features = quantize_features(features, precision)
                candidate_features = F.normalize(candidate_features, dim=1, eps=1e-12)
                for bucket_name, start, end in lag_buckets(eval_len, args.train_len):
                    query_lags = sample_bucket_lags(
                        start,
                        end,
                        args.queries_per_bucket,
                        device=device,
                        dtype=dtype,
                    )
                    if query_lags.numel() == 0:
                        continue
                    rows.append(
                        evaluate_bucket(
                            variant,
                            precision,
                            eval_len=eval_len,
                            train_len=args.train_len,
                            bucket=bucket_name,
                            min_lag=start,
                            max_lag=end,
                            query_lags=query_lags,
                            candidate_lags=candidate_lags,
                            candidate_features=candidate_features,
                            query_batch_size=args.query_batch_size,
                            tie_eps=args.tie_eps,
                        )
                    )

    write_outputs(rows, args.out_dir, args)
    print(f"Wrote Phase F retrieval-resolution diagnostics to: {args.out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-len", type=int, default=1024)
    parser.add_argument("--eval-lens", default="4096,8192,16384,32768")
    parser.add_argument("--max-order", type=int, default=3)
    parser.add_argument("--omega-cycles", default="5,17,61")
    parser.add_argument("--damping", type=float, default=0.1)
    parser.add_argument(
        "--variants",
        default="raw,scaled,clipped,log,lc_amp_only,lc_phase_only,lc,lc_wrong_scale",
    )
    parser.add_argument("--precisions", default="fp32,int8,int4")
    parser.add_argument("--queries-per-bucket", type=int, default=128)
    parser.add_argument("--query-batch-size", type=int, default=128)
    parser.add_argument("--tie-eps", type=float, default=1e-6)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--float64", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=Path("runs/phase_f_retrieval_resolution"))
    return parser.parse_args()


def position_features(
    variant: str,
    d: torch.Tensor,
    *,
    train_len: int,
    max_order: int,
    omegas: torch.Tensor,
    damping: float,
) -> torch.Tensor:
    coord = amplitude_coordinate(variant, d, train_len)
    phase = phase_coordinate(variant, d, train_len)
    env = envelope(variant, d, train_len, damping)
    columns = []
    for omega in omegas:
        theta = omega * phase
        cos = torch.cos(theta)
        sin = torch.sin(theta)
        for order in range(max_order + 1):
            amp = coord.pow(order)
            if variant in FJ_STYLE_VARIANTS:
                amp = amp / math.factorial(order)
            amp = amp * env
            columns.append(amp * cos)
            columns.append(amp * sin)
    return torch.stack(columns, dim=1)


def quantize_features(features: torch.Tensor, precision: str, eps: float = 1e-12) -> torch.Tensor:
    if precision in {"fp32", "none"}:
        return features.to(torch.float32)
    if precision == "int8":
        bits = 8
    elif precision == "int4":
        bits = 4
    else:
        raise ValueError(f"unknown precision: {precision}")
    qmax = float(2 ** (bits - 1) - 1)
    scale = features.detach().abs().amax(dim=0, keepdim=True).clamp_min(eps) / qmax
    return ((features / scale).round().clamp(-qmax, qmax) * scale).to(torch.float32)


def lag_buckets(eval_len: int, train_len: int) -> list[tuple[str, int, int]]:
    raw = [
        ("train", 1, train_len - 1),
        ("x1_4", train_len, 4 * train_len - 1),
        ("x4_16", 4 * train_len, 16 * train_len - 1),
        ("x16_32", 16 * train_len, 32 * train_len - 1),
    ]
    out = []
    max_lag = eval_len - 1
    for name, start, end in raw:
        clipped_end = min(end, max_lag)
        if start <= clipped_end:
            out.append((name, start, clipped_end))
    return out


def sample_bucket_lags(
    start: int,
    end: int,
    queries: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    count = min(max(1, queries), end - start + 1)
    return torch.linspace(start, end, count, device=device, dtype=dtype).round().unique()


def evaluate_bucket(
    variant: str,
    precision: str,
    *,
    eval_len: int,
    train_len: int,
    bucket: str,
    min_lag: int,
    max_lag: int,
    query_lags: torch.Tensor,
    candidate_lags: torch.Tensor,
    candidate_features: torch.Tensor,
    query_batch_size: int,
    tie_eps: float,
) -> RetrievalRow:
    target_indices = (query_lags.to(torch.long) - 1).clamp_min(0)
    top_lags = []
    target_scores = []
    best_nonself_scores = []
    margins = []
    tie_counts = []
    for start in range(0, target_indices.numel(), query_batch_size):
        idx = target_indices[start : start + query_batch_size]
        query_features = candidate_features[idx]
        scores = query_features @ candidate_features.T
        local_target = torch.arange(idx.numel(), device=scores.device)
        target = scores[local_target, idx]
        masked = scores.clone()
        masked[local_target, idx] = -float("inf")
        best_nonself = masked.max(dim=1).values
        best_idx = scores.argmax(dim=1)
        top_lags.append(candidate_lags[best_idx])
        target_scores.append(target)
        best_nonself_scores.append(best_nonself)
        margins.append(target - best_nonself)
        tie_counts.append((scores >= target[:, None] - tie_eps).to(torch.float32).sum(dim=1))

    predicted = torch.cat(top_lags).to(torch.long)
    target_lags = query_lags.to(torch.long)
    abs_error = (predicted - target_lags).abs().to(torch.float32)
    target_score = torch.cat(target_scores)
    best_nonself_score = torch.cat(best_nonself_scores)
    margin = torch.cat(margins)
    tie_count = torch.cat(tie_counts)

    in_bucket = (predicted >= min_lag) & (predicted <= max_lag)
    return RetrievalRow(
        variant=variant,
        precision=precision,
        eval_len=eval_len,
        train_len=train_len,
        bucket=bucket,
        min_lag=min_lag,
        max_lag=max_lag,
        n_queries=int(query_lags.numel()),
        candidate_count=int(candidate_lags.numel()),
        feature_dim=int(candidate_features.shape[1]),
        top1_exact=mean_float(predicted == target_lags),
        within_1=mean_float(abs_error <= 1),
        within_8=mean_float(abs_error <= 8),
        within_64=mean_float(abs_error <= 64),
        bucket_accuracy=mean_float(in_bucket),
        mean_abs_error=float(abs_error.mean().detach().cpu()),
        median_abs_error=float(abs_error.median().detach().cpu()),
        mean_target_score=float(target_score.mean().detach().cpu()),
        mean_best_nonself_score=float(best_nonself_score.mean().detach().cpu()),
        mean_margin=float(margin.mean().detach().cpu()),
        min_margin=float(margin.min().detach().cpu()),
        mean_tie_count=float(tie_count.mean().detach().cpu()),
    )


def mean_float(mask: torch.Tensor) -> float:
    return float(mask.to(torch.float32).mean().detach().cpu())


def write_outputs(rows: list[RetrievalRow], out_dir: Path, args: argparse.Namespace) -> None:
    rows_as_dicts = [asdict(row) for row in rows]
    with (out_dir / "results.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows_as_dicts[0].keys()))
        writer.writeheader()
        writer.writerows(rows_as_dicts)

    final_len = max(row.eval_len for row in rows)
    hard_bucket = "x16_32"
    snapshot = [
        asdict(row)
        for row in rows
        if row.eval_len == final_len and row.bucket == hard_bucket
    ]
    summary = {
        "config": {
            "train_len": args.train_len,
            "eval_lens": parse_int_list(args.eval_lens),
            "max_order": args.max_order,
            "omega_cycles": parse_float_list(args.omega_cycles),
            "damping": args.damping,
            "variants": parse_str_list(args.variants),
            "precisions": parse_str_list(args.precisions),
            "queries_per_bucket": args.queries_per_bucket,
            "tie_eps": args.tie_eps,
        },
        "final_eval_len": final_len,
        "hard_bucket": hard_bucket,
        "snapshot": snapshot,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_readme(rows, out_dir, final_len, hard_bucket)


def write_readme(rows: list[RetrievalRow], out_dir: Path, final_len: int, hard_bucket: str) -> None:
    lines = [
        "# Phase F Retrieval Resolution",
        "",
        "No-training nearest-neighbor lag retrieval probe for Phase F coordinate variants.",
        "",
        "## Files",
        "",
        "- `results.csv`: one row per variant, precision, eval length, and lag bucket.",
        "- `summary.json`: configuration and final hard-bucket snapshot.",
        "",
        f"## Final Hard Bucket: `{hard_bucket}` at `{final_len}`",
        "",
        "| Precision | Variant | Top1 | Within64 | BucketAcc | MeanAbsErr | Margin | Ties |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    final_rows = [
        row
        for row in rows
        if row.eval_len == final_len and row.bucket == hard_bucket
    ]
    for row in sorted(final_rows, key=lambda item: (item.precision, item.variant)):
        lines.append(
            "| `{precision}` | `{variant}` | `{top1:.3f}` | `{within64:.3f}` | `{bucket:.3f}` | "
            "`{err:.1f}` | `{margin:.3e}` | `{ties:.1f}` |".format(
                precision=row.precision,
                variant=row.variant,
                top1=row.top1_exact,
                within64=row.within_64,
                bucket=row.bucket_accuracy,
                err=row.mean_abs_error,
                margin=row.mean_margin,
                ties=row.mean_tie_count,
            )
        )
    lines.extend(
        [
            "",
            "Interpretation notes:",
            "",
            "- Full-precision exact retrieval can be easy even when margins are tiny; the margin and tie count are the main resolution diagnostics.",
            "- Int4 rows expose finite-precision phase/code collisions more clearly than full precision.",
            "- `lc_wrong_scale` should perform poorly or show large tie counts because its phase span is too small.",
        ]
    )
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_int_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in str(raw).split(",") if item.strip()]


def parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in str(raw).split(",") if item.strip()]


def parse_str_list(raw: str) -> list[str]:
    return [item.strip() for item in str(raw).split(",") if item.strip()]


if __name__ == "__main__":
    main()
