"""Phase F trained attention value-retrieval probe.

This is a lightweight model-level appendix for the Phase F LC ablations.  A
query specifies a target lag via the positional feature code, the context holds
random value tokens at every lag, and a single trainable attention readout must
retrieve the value at the target lag.

The model is trained only on lags within ``train_len`` and evaluated on longer
contexts and far buckets.
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
        "/home/riven/JordanKac/.venv/bin/python experiments/phase_f_attention_retrieval.py"
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
class AttentionRetrievalRow:
    variant: str
    precision: str
    eval_len: int
    train_len: int
    bucket: str
    min_lag: int
    max_lag: int
    train_loss: float
    train_value_accuracy: float
    value_accuracy: float
    top_position_accuracy: float
    target_attention_mass: float
    attention_entropy: float
    effective_support: float
    mean_target_lag: float
    score_abs_max: float
    score_std: float


class LagAttentionRetriever(torch.nn.Module):
    def __init__(
        self,
        feature_dim: int,
        value_vocab: int,
        attn_dim: int,
        value_dim: int,
        *,
        score_mode: str,
        init_temperature: float,
    ) -> None:
        super().__init__()
        self.score_mode = score_mode
        if score_mode == "projected":
            self.key_proj = torch.nn.Linear(feature_dim, attn_dim, bias=False)
            self.query_proj = torch.nn.Linear(feature_dim, attn_dim, bias=False)
        elif score_mode == "fixed":
            self.key_proj = None
            self.query_proj = None
        else:
            raise ValueError(f"unknown score_mode: {score_mode}")
        self.log_temperature = torch.nn.Parameter(torch.tensor(math.log(init_temperature), dtype=torch.float32))
        self.value_embedding = torch.nn.Embedding(value_vocab, value_dim)
        self.head = torch.nn.Linear(value_dim, value_vocab)
        self.attn_dim = attn_dim

    def forward(
        self,
        candidate_features: torch.Tensor,
        query_features: torch.Tensor,
        values: torch.Tensor,
        *,
        need_weights: bool = False,
    ):
        if self.score_mode == "projected":
            keys = self.key_proj(candidate_features)
            query = self.query_proj(query_features)
            scores = (query @ keys.T) / math.sqrt(self.attn_dim)
        else:
            keys = F.normalize(candidate_features, dim=1, eps=1e-12)
            query = F.normalize(query_features, dim=1, eps=1e-12)
            scores = self.log_temperature.exp().clamp_max(1_000.0) * (query @ keys.T)
        weights = torch.softmax(scores, dim=-1)
        value_vectors = self.value_embedding(values)
        context = torch.einsum("bn,bnd->bd", weights, value_vectors)
        logits = self.head(context)
        if need_weights:
            return logits, weights, scores
        return logits


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

    rows: list[AttentionRetrievalRow] = []
    for variant_index, variant in enumerate(variants):
        validate_variant(variant)
        for precision_index, precision in enumerate(precisions):
            seed = args.seed + 1009 * variant_index + 9176 * precision_index
            torch.manual_seed(seed)
            train_features, stats = build_features(
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
            model = LagAttentionRetriever(
                feature_dim=train_features.shape[1],
                value_vocab=args.value_vocab,
                attn_dim=args.attn_dim,
                value_dim=args.value_dim,
                score_mode=args.score_mode,
                init_temperature=args.init_temperature,
            ).to(device=device)
            train_loss, train_acc = train_model(
                model,
                train_features,
                value_vocab=args.value_vocab,
                steps=args.steps,
                batch_size=args.batch_size,
                lr=args.lr,
                attention_loss_weight=args.attention_loss_weight,
                seed=seed,
            )
            for eval_len in eval_lens:
                eval_features, _ = build_features(
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
                    row = evaluate_bucket(
                        model,
                        eval_features,
                        variant=variant,
                        precision=precision,
                        eval_len=eval_len,
                        train_len=args.train_len,
                        bucket=bucket_name,
                        min_lag=start,
                        max_lag=end,
                        value_vocab=args.value_vocab,
                        batch_size=args.eval_batch_size,
                        eval_batches=args.eval_batches,
                        train_loss=train_loss,
                        train_acc=train_acc,
                    )
                    rows.append(row)

    write_outputs(rows, args.out_dir, args)
    print(f"Wrote Phase F attention-retrieval diagnostics to: {args.out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-len", type=int, default=1024)
    parser.add_argument("--eval-lens", default="4096,8192,16384,32768")
    parser.add_argument("--max-order", type=int, default=3)
    parser.add_argument("--omega-cycles", default="5,17,61")
    parser.add_argument("--damping", type=float, default=0.1)
    parser.add_argument("--variants", default="raw,scaled,lc_phase_only,lc,lc_wrong_scale")
    parser.add_argument("--precisions", default="fp32,int8,int4")
    parser.add_argument("--value-vocab", type=int, default=64)
    parser.add_argument("--attn-dim", type=int, default=64)
    parser.add_argument("--value-dim", type=int, default=64)
    parser.add_argument("--score-mode", default="fixed", choices=["fixed", "projected"])
    parser.add_argument("--init-temperature", type=float, default=32.0)
    parser.add_argument("--steps", type=int, default=360)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--eval-batch-size", type=int, default=48)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--attention-loss-weight", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--float64", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=Path("runs/phase_f_attention_retrieval"))
    return parser.parse_args()


def build_features(
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
        mean = features.mean(dim=0, keepdim=True)
        std = features.std(dim=0, keepdim=True).clamp_min(1e-6)
        stats = (mean, std)
    mean, std = stats
    return ((features - mean) / std).to(torch.float32), stats


def train_model(
    model: LagAttentionRetriever,
    train_features: torch.Tensor,
    *,
    value_vocab: int,
    steps: int,
    batch_size: int,
    lr: float,
    attention_loss_weight: float,
    seed: int,
) -> tuple[float, float]:
    generator = torch.Generator(device=train_features.device)
    generator.manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    final_loss = torch.tensor(float("nan"), device=train_features.device)
    final_acc = torch.tensor(0.0, device=train_features.device)
    for _ in range(steps):
        values, target_idx, labels = sample_retrieval_batch(
            train_features.shape[0],
            value_vocab,
            batch_size,
            low=0,
            high=train_features.shape[0] - 1,
            generator=generator,
            device=train_features.device,
        )
        logits, _, scores = model(train_features, train_features[target_idx], values, need_weights=True)
        value_loss = F.cross_entropy(logits, labels)
        attention_loss = F.cross_entropy(scores, target_idx)
        loss = value_loss + attention_loss_weight * attention_loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        final_loss = loss.detach()
        final_acc = (logits.argmax(dim=-1) == labels).to(torch.float32).mean().detach()
    return float(final_loss.cpu()), float(final_acc.cpu())


@torch.no_grad()
def evaluate_bucket(
    model: LagAttentionRetriever,
    features: torch.Tensor,
    *,
    variant: str,
    precision: str,
    eval_len: int,
    train_len: int,
    bucket: str,
    min_lag: int,
    max_lag: int,
    value_vocab: int,
    batch_size: int,
    eval_batches: int,
    train_loss: float,
    train_acc: float,
) -> AttentionRetrievalRow:
    model.eval()
    low = min_lag - 1
    high = min(max_lag, features.shape[0])
    losses = []
    value_hits = 0
    top_hits = 0
    total = 0
    target_masses = []
    entropies = []
    supports = []
    target_lags = []
    score_maxes = []
    score_stds = []
    generator = torch.Generator(device=features.device)
    generator.manual_seed(eval_len * 1000003 + low * 9176 + high)
    for _ in range(eval_batches):
        values, target_idx, labels = sample_retrieval_batch(
            features.shape[0],
            value_vocab,
            batch_size,
            low=low,
            high=high,
            generator=generator,
            device=features.device,
        )
        logits, weights, scores = model(features, features[target_idx], values, need_weights=True)
        losses.append(F.cross_entropy(logits, labels))
        pred = logits.argmax(dim=-1)
        value_hits += int((pred == labels).sum().detach().cpu())
        top_hits += int((weights.argmax(dim=-1) == target_idx).sum().detach().cpu())
        total += labels.numel()
        target_masses.append(weights.gather(1, target_idx[:, None]).mean())
        entropy = -(weights * weights.clamp_min(torch.finfo(weights.dtype).tiny).log()).sum(dim=-1)
        entropies.append(entropy.mean())
        supports.append(entropy.exp().mean())
        target_lags.append((target_idx.to(torch.float32) + 1.0).mean())
        score_maxes.append(scores.detach().abs().max())
        score_stds.append(scores.detach().std())
    model.train()
    return AttentionRetrievalRow(
        variant=variant,
        precision=precision,
        eval_len=eval_len,
        train_len=train_len,
        bucket=bucket,
        min_lag=min_lag,
        max_lag=max_lag,
        train_loss=train_loss,
        train_value_accuracy=train_acc,
        value_accuracy=value_hits / total,
        top_position_accuracy=top_hits / total,
        target_attention_mass=float(torch.stack(target_masses).mean().detach().cpu()),
        attention_entropy=float(torch.stack(entropies).mean().detach().cpu()),
        effective_support=float(torch.stack(supports).mean().detach().cpu()),
        mean_target_lag=float(torch.stack(target_lags).mean().detach().cpu()),
        score_abs_max=float(torch.stack(score_maxes).mean().detach().cpu()),
        score_std=float(torch.stack(score_stds).mean().detach().cpu()),
    )


def sample_retrieval_batch(
    candidate_count: int,
    value_vocab: int,
    batch_size: int,
    *,
    low: int,
    high: int,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    high = min(high, candidate_count - 1)
    if low < 0 or low > high:
        raise ValueError("invalid target index range")
    values = torch.randint(value_vocab, (batch_size, candidate_count), generator=generator, device=device)
    target_idx = torch.randint(low, high + 1, (batch_size,), generator=generator, device=device)
    labels = values[torch.arange(batch_size, device=device), target_idx]
    return values, target_idx, labels


def write_outputs(rows: list[AttentionRetrievalRow], out_dir: Path, args: argparse.Namespace) -> None:
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
            "value_vocab": args.value_vocab,
            "steps": args.steps,
            "attention_loss_weight": args.attention_loss_weight,
            "attn_dim": args.attn_dim,
            "value_dim": args.value_dim,
            "score_mode": args.score_mode,
            "init_temperature": args.init_temperature,
            "seed": args.seed,
        },
        "final_eval_len": final_len,
        "hard_bucket": hard_bucket,
        "snapshot": snapshot,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_readme(rows, out_dir, final_len, hard_bucket)


def write_readme(rows: list[AttentionRetrievalRow], out_dir: Path, final_len: int, hard_bucket: str) -> None:
    lines = [
        "# Phase F Attention Retrieval",
        "",
        "Trained single-attention value-retrieval probe for Phase F variants.",
        "",
        "## Files",
        "",
        "- `results.csv`: one row per variant, precision, eval length, and bucket.",
        "- `summary.json`: configuration and final hard-bucket snapshot.",
        "",
        f"## Final Hard Bucket: `{hard_bucket}` at `{final_len}`",
        "",
        "| Precision | Variant | TrainAcc | ValueAcc | TopPos | TargetMass | Support | ScoreStd |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    final_rows = [
        row
        for row in rows
        if row.eval_len == final_len and row.bucket == hard_bucket
    ]
    for row in sorted(final_rows, key=lambda item: (item.precision, item.variant)):
        lines.append(
            "| `{precision}` | `{variant}` | `{train:.3f}` | `{value:.3f}` | `{top:.3f}` | "
            "`{mass:.3f}` | `{support:.2f}` | `{score:.3e}` |".format(
                precision=row.precision,
                variant=row.variant,
                train=row.train_value_accuracy,
                value=row.value_accuracy,
                top=row.top_position_accuracy,
                mass=row.target_attention_mass,
                support=row.effective_support,
                score=row.score_std,
            )
        )
    lines.extend(
        [
            "",
            "Interpretation notes:",
            "",
            "- `TopPos` measures whether learned attention peaks at the requested lag.",
            "- `ValueAcc` is value-token retrieval accuracy from random context values.",
            "- This is still a synthetic appendix probe, not a language model.",
        ]
    )
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
