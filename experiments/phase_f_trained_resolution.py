"""Phase F trained pairwise lag-resolution probe.

This lightweight task trains a small classifier on train-window lag feature
differences and evaluates whether it can distinguish a target lag from random
and nearest-confuser negatives in far buckets.  It is intentionally smaller
than a full LM retrieval benchmark, but it connects the Phase F coordinate
diagnostics to a trained decision rule.
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
        "/home/riven/JordanKac/.venv/bin/python experiments/phase_f_trained_resolution.py"
    ) from exc

from experiments.phase_f_retrieval_resolution import (
    lag_buckets,
    parse_float_list,
    parse_int_list,
    parse_str_list,
    position_features,
    quantize_features,
    sample_bucket_lags,
    validate_variant,
)


@dataclass(frozen=True)
class TrainedResolutionRow:
    variant: str
    precision: str
    eval_len: int
    train_len: int
    bucket: str
    min_lag: int
    max_lag: int
    n_queries: int
    feature_dim: int
    train_loss: float
    train_pair_accuracy: float
    positive_accept_rate: float
    random_negative_accept_rate: float
    hard_negative_accept_rate: float
    random_pair_accuracy: float
    hard_pair_accuracy: float
    hard_negative_mean_lag_gap: float
    hard_negative_median_lag_gap: float
    hard_negative_mean_feature_distance: float


class PairwiseClassifier(torch.nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, diff: torch.Tensor) -> torch.Tensor:
        return self.net(diff).squeeze(-1)


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

    rows: list[TrainedResolutionRow] = []
    for variant_index, variant in enumerate(variants):
        validate_variant(variant)
        for precision_index, precision in enumerate(precisions):
            seed = args.seed + 1009 * variant_index + 9176 * precision_index
            torch.manual_seed(seed)
            train_features, stats = build_context_features(
                variant,
                precision,
                eval_len=args.train_len,
                train_len=args.train_len,
                max_order=args.max_order,
                omegas=omegas,
                damping=args.damping,
                dtype=dtype,
                device=device,
                stats=None,
            )
            model = PairwiseClassifier(train_features.shape[1], args.hidden_dim).to(device=device)
            train_loss, train_acc = train_classifier(
                model,
                train_features,
                steps=args.steps,
                batch_size=args.batch_size,
                lr=args.lr,
                seed=seed,
            )
            for eval_len in eval_lens:
                eval_features, _ = build_context_features(
                    variant,
                    precision,
                    eval_len=eval_len,
                    train_len=args.train_len,
                    max_order=args.max_order,
                    omegas=omegas,
                    damping=args.damping,
                    dtype=dtype,
                    device=device,
                    stats=stats,
                )
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
                            model,
                            eval_features,
                            variant=variant,
                            precision=precision,
                            eval_len=eval_len,
                            train_len=args.train_len,
                            bucket=bucket_name,
                            min_lag=start,
                            max_lag=end,
                            query_lags=query_lags,
                            train_loss=train_loss,
                            train_acc=train_acc,
                            hard_batch_size=args.hard_batch_size,
                        )
                    )

    write_outputs(rows, args.out_dir, args)
    print(f"Wrote Phase F trained-resolution diagnostics to: {args.out_dir}")


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
    parser.add_argument("--steps", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--queries-per-bucket", type=int, default=96)
    parser.add_argument("--hard-batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--float64", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=Path("runs/phase_f_trained_resolution"))
    return parser.parse_args()


def build_context_features(
    variant: str,
    precision: str,
    *,
    eval_len: int,
    train_len: int,
    max_order: int,
    omegas: torch.Tensor,
    damping: float,
    dtype: torch.dtype,
    device: torch.device,
    stats: tuple[torch.Tensor, torch.Tensor] | None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    lags = torch.arange(1, eval_len, device=device, dtype=dtype)
    features = position_features(
        variant,
        lags,
        train_len=train_len,
        max_order=max_order,
        omegas=omegas,
        damping=damping,
    )
    features = quantize_features(features, precision)
    if stats is None:
        train_rows = min(train_len - 1, features.shape[0])
        mean = features[:train_rows].mean(dim=0, keepdim=True)
        std = features[:train_rows].std(dim=0, keepdim=True).clamp_min(1e-6)
        stats = (mean, std)
    mean, std = stats
    return ((features - mean) / std).to(torch.float32), stats


def train_classifier(
    model: PairwiseClassifier,
    features: torch.Tensor,
    *,
    steps: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> tuple[float, float]:
    generator = torch.Generator(device=features.device)
    generator.manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    final_loss = torch.tensor(float("nan"), device=features.device)
    final_acc = torch.tensor(0.0, device=features.device)
    for _ in range(steps):
        diff, labels = sample_train_pairs(features, batch_size, generator)
        logits = model(diff)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        final_loss = loss.detach()
        final_acc = ((logits >= 0) == labels.bool()).to(torch.float32).mean().detach()
    return float(final_loss.cpu()), float(final_acc.cpu())


def sample_train_pairs(
    features: torch.Tensor,
    batch_size: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    n = features.shape[0]
    half = batch_size // 2
    pos = torch.randint(0, n, (half,), generator=generator, device=features.device)
    neg_a = torch.randint(0, n, (batch_size - half,), generator=generator, device=features.device)
    neg_b = torch.randint(0, n, (batch_size - half,), generator=generator, device=features.device)
    neg_b = torch.where(neg_b == neg_a, (neg_b + 1) % n, neg_b)
    pos_diff = torch.zeros(half, features.shape[1], device=features.device, dtype=features.dtype)
    neg_diff = (features[neg_a] - features[neg_b]).abs()
    diff = torch.cat([pos_diff, neg_diff], dim=0)
    labels = torch.cat(
        [
            torch.ones(half, device=features.device, dtype=features.dtype),
            torch.zeros(batch_size - half, device=features.device, dtype=features.dtype),
        ]
    )
    perm = torch.randperm(batch_size, generator=generator, device=features.device)
    return diff[perm], labels[perm]


@torch.no_grad()
def evaluate_bucket(
    model: PairwiseClassifier,
    features: torch.Tensor,
    *,
    variant: str,
    precision: str,
    eval_len: int,
    train_len: int,
    bucket: str,
    min_lag: int,
    max_lag: int,
    query_lags: torch.Tensor,
    train_loss: float,
    train_acc: float,
    hard_batch_size: int,
) -> TrainedResolutionRow:
    model.eval()
    target_idx = (query_lags.to(torch.long) - 1).clamp(min=0, max=features.shape[0] - 1)
    pos_diff = torch.zeros(target_idx.numel(), features.shape[1], device=features.device)
    pos_accept = accept_rate(model(pos_diff))

    random_idx = random_negative_indices(target_idx, min_lag, max_lag, features.shape[0])
    random_diff = (features[target_idx] - features[random_idx]).abs()
    random_accept = accept_rate(model(random_diff))

    hard_idx, hard_dist = nearest_confusers(features, target_idx, min_lag, max_lag, batch_size=hard_batch_size)
    hard_diff = (features[target_idx] - features[hard_idx]).abs()
    hard_accept = accept_rate(model(hard_diff))
    lag_gap = (hard_idx - target_idx).abs().to(torch.float32)

    model.train()
    return TrainedResolutionRow(
        variant=variant,
        precision=precision,
        eval_len=eval_len,
        train_len=train_len,
        bucket=bucket,
        min_lag=min_lag,
        max_lag=max_lag,
        n_queries=int(target_idx.numel()),
        feature_dim=int(features.shape[1]),
        train_loss=train_loss,
        train_pair_accuracy=train_acc,
        positive_accept_rate=pos_accept,
        random_negative_accept_rate=random_accept,
        hard_negative_accept_rate=hard_accept,
        random_pair_accuracy=0.5 * (pos_accept + (1.0 - random_accept)),
        hard_pair_accuracy=0.5 * (pos_accept + (1.0 - hard_accept)),
        hard_negative_mean_lag_gap=float(lag_gap.mean().cpu()),
        hard_negative_median_lag_gap=float(lag_gap.median().cpu()),
        hard_negative_mean_feature_distance=float(hard_dist.mean().cpu()),
    )


def random_negative_indices(
    target_idx: torch.Tensor,
    min_lag: int,
    max_lag: int,
    feature_count: int,
) -> torch.Tensor:
    low = min_lag - 1
    high = min(max_lag, feature_count - 1)
    span = high - low + 1
    # Deterministic offset keeps this evaluation reproducible without adding a
    # separate RNG stream.
    offset = (torch.arange(target_idx.numel(), device=target_idx.device) * 37 + 1) % max(1, span)
    out = low + ((target_idx - low + offset) % span)
    return torch.where(out == target_idx, low + ((out - low + 1) % span), out)


def nearest_confusers(
    features: torch.Tensor,
    target_idx: torch.Tensor,
    min_lag: int,
    max_lag: int,
    *,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    start = min_lag - 1
    stop = min(max_lag, features.shape[0] - 1) + 1
    candidate = features[start:stop]
    hard_indices = []
    hard_distances = []
    for offset in range(0, target_idx.numel(), batch_size):
        idx = target_idx[offset : offset + batch_size]
        dist = torch.cdist(features[idx], candidate)
        local_target = (idx - start).clamp(min=0, max=candidate.shape[0] - 1)
        rows = torch.arange(idx.numel(), device=features.device)
        dist[rows, local_target] = float("inf")
        values, local = dist.min(dim=1)
        hard_indices.append(local + start)
        hard_distances.append(values)
    return torch.cat(hard_indices), torch.cat(hard_distances)


def accept_rate(logits: torch.Tensor) -> float:
    return float((logits >= 0).to(torch.float32).mean().cpu())


def write_outputs(rows: list[TrainedResolutionRow], out_dir: Path, args: argparse.Namespace) -> None:
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
            "steps": args.steps,
            "hidden_dim": args.hidden_dim,
            "queries_per_bucket": args.queries_per_bucket,
            "seed": args.seed,
        },
        "final_eval_len": final_len,
        "hard_bucket": hard_bucket,
        "snapshot": snapshot,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_readme(rows, out_dir, final_len, hard_bucket)


def write_readme(rows: list[TrainedResolutionRow], out_dir: Path, final_len: int, hard_bucket: str) -> None:
    lines = [
        "# Phase F Trained Resolution",
        "",
        "Small trained pairwise classifier for lag-code resolution.",
        "",
        "## Files",
        "",
        "- `results.csv`: one row per variant, precision, eval length, and bucket.",
        "- `summary.json`: configuration and final hard-bucket snapshot.",
        "",
        f"## Final Hard Bucket: `{hard_bucket}` at `{final_len}`",
        "",
        "| Precision | Variant | TrainAcc | PosAccept | RandNegAccept | HardNegAccept | HardPairAcc | HardGap | HardDist |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    final_rows = [
        row
        for row in rows
        if row.eval_len == final_len and row.bucket == hard_bucket
    ]
    for row in sorted(final_rows, key=lambda item: (item.precision, item.variant)):
        lines.append(
            "| `{precision}` | `{variant}` | `{train:.3f}` | `{pos:.3f}` | `{rand:.3f}` | "
            "`{hard:.3f}` | `{acc:.3f}` | `{gap:.1f}` | `{dist:.3e}` |".format(
                precision=row.precision,
                variant=row.variant,
                train=row.train_pair_accuracy,
                pos=row.positive_accept_rate,
                rand=row.random_negative_accept_rate,
                hard=row.hard_negative_accept_rate,
                acc=row.hard_pair_accuracy,
                gap=row.hard_negative_mean_lag_gap,
                dist=row.hard_negative_mean_feature_distance,
            )
        )
    lines.extend(
        [
            "",
            "Interpretation notes:",
            "",
            "- `HardNegAccept` is the false-positive rate on the nearest non-target lag in the bucket.",
            "- `HardPairAcc = 0.5 * (positive accept + hard negative reject)`.",
            "- This is a trained resolution probe, not a full sequence LM.",
        ]
    )
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
