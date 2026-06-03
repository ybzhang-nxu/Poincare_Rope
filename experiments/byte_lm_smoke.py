"""Small byte-level LM smoke for Phase D long-context experiments.

This is intentionally lightweight: it can run on the built-in fallback corpus
or on a user-provided text file. The goal is to exercise the Phase D training
and evaluation path before moving to WikiText/TinyStories-scale runs.
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
        "/home/riven/JordanKac/.venv/bin/python experiments/byte_lm_smoke.py"
    ) from exc

from pjrope.attention import PJBias, causal_attention_mask
from pjrope.torch_backend import apply_exact_pj_rotary


VOCAB_SIZE = 256
METHODS = (
    "none",
    "rope",
    "rope_pi",
    "rope_ntk",
    "rope_yarn",
    "rope_ntk_affine",
    "rope_yarn_affine",
    "pj_rotary",
    "affine",
    "fj_affine",
    "fj_lc",
    "rope_affine",
    "grape_m_rope",
    "grape_a_alibi",
    "grape_ma_rope_alibi",
    "full",
    "lc_affine",
)
FALLBACK_TEXT = (
    "Poincare jet rotary position spaces mix phase, recency, and light-cone "
    "coordinates. A small byte language model can expose whether a position "
    "bias learns mostly recency with weak oscillatory corrections. "
    "The quick brown fox jumps over the lazy dog; the same motif returns at "
    "different distances, sometimes nearby and sometimes far away. "
    "Mathematics likes repetition with variation, and language likes context. "
)


@dataclass(frozen=True)
class EvalMetrics:
    loss: float
    ppl: float
    attention_entropy: float
    attention_effective_support: float
    average_attention_distance: float
    logit_std: float
    logit_abs_max: float


class ByteLM(torch.nn.Module):
    def __init__(
        self,
        *,
        method: str,
        vocab_size: int,
        train_len: int,
        embed_dim: int,
        num_heads: int,
        layers: int,
        max_order: int,
        mlp_ratio: int,
        dropout: float,
        init_affine_slope: float,
        gate_init: str,
        rope_base: float,
        attention_block_size: int,
    ) -> None:
        super().__init__()
        self.method = method
        self.embedding = torch.nn.Embedding(vocab_size, embed_dim)
        self.blocks = torch.nn.ModuleList(
            [
                ByteLMBlock(
                    method=method,
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    train_len=train_len,
                    max_order=max_order,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    init_affine_slope=init_affine_slope,
                    gate_init=gate_init,
                    rope_base=rope_base,
                    attention_block_size=attention_block_size,
                )
                for _ in range(layers)
            ]
        )
        self.norm = torch.nn.LayerNorm(embed_dim)
        self.head = torch.nn.Linear(embed_dim, vocab_size)

    def forward(self, tokens: torch.Tensor, *, need_weights: bool = False):
        hidden = self.embedding(tokens)
        last_weights = None
        for index, block in enumerate(self.blocks):
            if need_weights and index == len(self.blocks) - 1:
                hidden, last_weights = block(hidden, need_weights=True)
            else:
                hidden = block(hidden)
        logits = self.head(self.norm(hidden))
        if need_weights:
            return logits, last_weights
        return logits


class ByteLMBlock(torch.nn.Module):
    def __init__(
        self,
        *,
        method: str,
        embed_dim: int,
        num_heads: int,
        train_len: int,
        max_order: int,
        mlp_ratio: int,
        dropout: float,
        init_affine_slope: float,
        gate_init: str,
        rope_base: float,
        attention_block_size: int,
    ) -> None:
        super().__init__()
        self.norm1 = torch.nn.LayerNorm(embed_dim)
        self.norm2 = torch.nn.LayerNorm(embed_dim)
        self.attn = ByteCausalSelfAttention(
            method=method,
            embed_dim=embed_dim,
            num_heads=num_heads,
            train_len=train_len,
            max_order=max_order,
            dropout=dropout,
            init_affine_slope=init_affine_slope,
            gate_init=gate_init,
            rope_base=rope_base,
            attention_block_size=attention_block_size,
        )
        hidden_dim = mlp_ratio * embed_dim
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, embed_dim),
            torch.nn.Dropout(dropout),
        )

    def forward(self, hidden: torch.Tensor, *, need_weights: bool = False):
        if need_weights:
            out, weights = self.attn(self.norm1(hidden), need_weights=True)
            hidden = hidden + out
            hidden = hidden + self.mlp(self.norm2(hidden))
            return hidden, weights
        hidden = hidden + self.attn(self.norm1(hidden))
        hidden = hidden + self.mlp(self.norm2(hidden))
        return hidden


class ByteCausalSelfAttention(torch.nn.Module):
    def __init__(
        self,
        *,
        method: str,
        embed_dim: int,
        num_heads: int,
        train_len: int,
        max_order: int,
        dropout: float,
        init_affine_slope: float,
        gate_init: str,
        rope_base: float,
        attention_block_size: int,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        if method not in METHODS:
            raise ValueError(f"unknown method: {method}")
        self.method = method
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        if self.uses_rope and self.head_dim % 2 != 0:
            raise ValueError("RoPE methods require an even head_dim")
        if self.uses_pj_rotary and self.head_dim % 4 != 0:
            raise ValueError("exact PJ-rotary methods require head_dim divisible by 4")
        self.scale = self.head_dim**-0.5
        self.qkv = torch.nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = torch.nn.Linear(embed_dim, embed_dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.train_len = train_len
        self.rope_base = float(rope_base)
        self.attention_block_size = int(attention_block_size)
        self.pj_bias = make_pj_bias(
            method=method,
            num_heads=num_heads,
            max_order=max_order,
            train_len=train_len,
            init_affine_slope=init_affine_slope,
            gate_init=gate_init,
        )

    @property
    def uses_rope(self) -> bool:
        return self.method in (
            "rope",
            "rope_pi",
            "rope_ntk",
            "rope_yarn",
            "rope_affine",
            "rope_ntk_affine",
            "rope_yarn_affine",
            "grape_m_rope",
            "grape_ma_rope_alibi",
        )

    @property
    def uses_pj_rotary(self) -> bool:
        return self.method == "pj_rotary"

    def forward(self, x: torch.Tensor, *, need_weights: bool = False):
        batch, seq_len, embed_dim = x.shape
        qkv = self.qkv(x).view(batch, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        if self.uses_rope:
            query, key = apply_rope(
                query,
                key,
                base=self.rope_base,
                train_len=self.train_len,
                interpolate=self.method == "rope_pi",
                dynamic_ntk=self.method in ("rope_ntk", "rope_ntk_affine"),
                yarn=self.method in ("rope_yarn", "rope_yarn_affine"),
            )
        elif self.uses_pj_rotary:
            query, key = apply_exact_pj_rotary(
                query,
                key,
                base=self.rope_base,
                train_length=self.train_len,
            )

        if self.attention_block_size > 0 and seq_len > self.attention_block_size and not need_weights:
            out = self.blockwise_attention(query, key, value, seq_len)
            out = out.transpose(1, 2).contiguous().view(batch, seq_len, embed_dim)
            return self.out_proj(out)

        scores = (query @ key.transpose(-2, -1)) * self.scale
        if self.pj_bias is not None:
            scores = scores + self.pj_bias(seq_len, device=x.device, dtype=x.dtype)[None, :, :, :]
        mask = causal_attention_mask(seq_len, device=x.device, include_self=True)
        weights = masked_softmax(scores, mask[None, None, :, :])
        weights = self.dropout(weights)
        out = weights @ value
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, embed_dim)
        out = self.out_proj(out)
        if need_weights:
            return out, weights
        return out

    def blockwise_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        blocks = []
        key_t = key.transpose(-2, -1)
        for start in range(0, seq_len, self.attention_block_size):
            end = min(start + self.attention_block_size, seq_len)
            scores = (query[:, :, start:end, :] @ key_t) * self.scale
            if self.pj_bias is not None:
                scores = scores + self.pj_bias_block(
                    start,
                    end,
                    seq_len,
                    device=query.device,
                    dtype=query.dtype,
                )[None, :, :, :]
            mask = block_causal_attention_mask(start, end, seq_len, device=query.device)
            weights = masked_softmax(scores, mask[None, None, :, :])
            blocks.append(weights @ value)
        return torch.cat(blocks, dim=-2)

    def pj_bias_block(
        self,
        start: int,
        end: int,
        seq_len: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        query_idx = torch.arange(start, end, device=device)
        key_idx = torch.arange(seq_len, device=device)
        lags = (query_idx[:, None] - key_idx[None, :]).to(dtype=dtype)
        mask = lags >= 0
        return self.pj_bias.forward_lags(lags, mask, device=device, dtype=dtype)


def main() -> None:
    args = parse_args()
    validate_args(args)
    device = torch.device(args.device)
    dtype = torch.float64 if args.float64 else torch.float32
    data = load_corpus(args)
    rows = []
    seeds = parse_int_list(args.seeds) if args.seeds else [args.seed]
    args.seed_values = ",".join(str(seed) for seed in seeds)
    for seed in seeds:
        args.seed = seed
        for method in parse_str_list(args.methods):
            rows.extend(run_method(args, method, data, device=device, dtype=dtype))

    write_outputs(rows, args)
    print_summary(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-file", type=Path, default=None)
    parser.add_argument("--repeat-fallback", type=int, default=256)
    parser.add_argument("--train-len", type=int, default=128)
    parser.add_argument("--eval-lens", default="128,256")
    parser.add_argument("--methods", default="none,affine,full")
    parser.add_argument("--embed-dim", type=int, default=96)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--max-order", type=int, default=3)
    parser.add_argument("--mlp-ratio", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--init-affine-slope", type=float, default=2.0)
    parser.add_argument("--gate-init", choices=["uniform", "affine", "fj", "lc"], default="affine")
    parser.add_argument("--rope-base", type=float, default=10000.0)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument(
        "--attention-block-size",
        type=int,
        default=0,
        help="Use query-blocked attention for loss-only forward passes when seq_len exceeds this size.",
    )
    parser.add_argument(
        "--skip-attention-stats",
        action="store_true",
        help="Evaluate loss/ppl without materializing full attention weights or attention diagnostics.",
    )
    parser.add_argument(
        "--attention-stats-samples",
        type=int,
        default=0,
        help="Estimate attention diagnostics from this many evenly spaced query rows when using --skip-attention-stats.",
    )
    parser.add_argument("--log-every", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--float64", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=Path("runs/byte_lm_smoke"))
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Directory for per-method/seed model checkpoints.",
    )
    parser.add_argument(
        "--save-checkpoints",
        action="store_true",
        help="Save trained model checkpoints after each method/seed.",
    )
    parser.add_argument(
        "--eval-from-checkpoints",
        action="store_true",
        help="Load per-method/seed checkpoints and evaluate without training.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.train_len < 2:
        raise SystemExit("train-len must be at least 2")
    if args.batch_size < 1:
        raise SystemExit("batch-size must be at least 1")
    if args.eval_batch_size is not None and args.eval_batch_size < 1:
        raise SystemExit("eval-batch-size must be at least 1")
    if args.eval_batches < 1:
        raise SystemExit("eval-batches must be at least 1")
    if args.attention_block_size < 0:
        raise SystemExit("attention-block-size must be non-negative")
    if args.attention_block_size > 0 and not args.skip_attention_stats:
        raise SystemExit("attention-block-size requires --skip-attention-stats")
    if args.attention_stats_samples < 0:
        raise SystemExit("attention-stats-samples must be non-negative")
    if args.attention_stats_samples > 0 and not args.skip_attention_stats:
        raise SystemExit("attention-stats-samples requires --skip-attention-stats")
    if args.steps < 0:
        raise SystemExit("steps must be non-negative")
    if args.eval_from_checkpoints and args.checkpoint_dir is None:
        raise SystemExit("eval-from-checkpoints requires --checkpoint-dir")
    if args.eval_from_checkpoints and args.save_checkpoints:
        raise SystemExit("use either --eval-from-checkpoints or --save-checkpoints, not both")
    if args.save_checkpoints and args.checkpoint_dir is None:
        args.checkpoint_dir = args.out_dir / "checkpoints"
    for eval_len in parse_int_list(args.eval_lens):
        if eval_len < 2:
            raise SystemExit("all eval-lens must be at least 2")
    if args.embed_dim % args.num_heads != 0:
        raise SystemExit("embed-dim must be divisible by num-heads")
    head_dim = args.embed_dim // args.num_heads
    methods = parse_str_list(args.methods)
    rope_like = {
        "rope",
        "rope_pi",
        "rope_ntk",
        "rope_yarn",
        "rope_affine",
        "rope_ntk_affine",
        "rope_yarn_affine",
        "grape_m_rope",
        "grape_ma_rope_alibi",
    }
    if any(method in rope_like for method in methods) and head_dim % 2 != 0:
        raise SystemExit("RoPE methods require an even head_dim")
    if "pj_rotary" in methods and head_dim % 4 != 0:
        raise SystemExit("exact PJ-rotary requires per-head dim divisible by 4")
    for method in methods:
        if method not in METHODS:
            raise SystemExit(f"unknown method: {method}")


def run_method(
    args: argparse.Namespace,
    method: str,
    data: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> list[dict[str, float | int | str]]:
    torch.manual_seed(args.seed)
    train_data, eval_data = split_data(data)
    train_data = train_data.to(device=device)
    eval_data = eval_data.to(device=device)
    model = ByteLM(
        method=method,
        vocab_size=VOCAB_SIZE,
        train_len=args.train_len,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        layers=args.layers,
        max_order=args.max_order,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        init_affine_slope=args.init_affine_slope,
        gate_init=args.gate_init,
        rope_base=args.rope_base,
        attention_block_size=args.attention_block_size,
    ).to(device=device, dtype=dtype)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    checkpoint_path = checkpoint_file(args, method)
    if args.eval_from_checkpoints:
        load_checkpoint(model, args, method, checkpoint_path, device)

    initial = evaluate(model, eval_data, args.train_len, args, device)
    if not args.eval_from_checkpoints:
        for step in range(1, args.steps + 1):
            x, y = sample_batch(train_data, args.batch_size, args.train_len, device)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if args.log_every and step % args.log_every == 0:
                metrics = evaluate(model, eval_data, args.train_len, args, device, eval_batches=2)
                print(f"method={method:<12} step={step:04d} loss={float(loss.detach().cpu()):.4f} eval={metrics.loss:.4f}")
        if args.save_checkpoints:
            save_checkpoint(model, args, method, checkpoint_path)

    gates, slopes, masses = pj_summary(model)
    eval_lens = parse_int_list(args.eval_lens)
    metrics_by_len = {
        eval_len: evaluate(model, eval_data, eval_len, args, device)
        for eval_len in sorted(set(eval_lens + [args.train_len]))
    }
    train_eval_loss = metrics_by_len[args.train_len].loss
    rows = []
    for eval_len in eval_lens:
        metrics = metrics_by_len[eval_len]
        row: dict[str, float | int | str] = {
            "method": method,
            "seed": args.seed,
            "train_len": args.train_len,
            "eval_len": eval_len,
            "max_order": args.max_order,
            "eval_from_checkpoint": int(args.eval_from_checkpoints),
            "initial_loss": initial.loss,
            "train_eval_loss": train_eval_loss,
            "loss": metrics.loss,
            "ppl": metrics.ppl,
            "delta_loss": metrics.loss - train_eval_loss,
            "attention_entropy": metrics.attention_entropy,
            "attention_effective_support": metrics.attention_effective_support,
            "average_attention_distance": metrics.average_attention_distance,
            "logit_std": metrics.logit_std,
            "logit_abs_max": metrics.logit_abs_max,
        }
        if gates is not None:
            row["gate_fj"] = gates[0]
            row["gate_affine"] = gates[1]
            row["gate_lc"] = gates[2]
        if slopes:
            row["affine_slope_mean"] = sum(slopes) / len(slopes)
        if checkpoint_path is not None:
            row["checkpoint_path"] = str(checkpoint_path)
        for name, values in masses.items():
            for order, value in enumerate(values):
                row[f"{name}_mass_r{order}"] = value
        rows.append(row)
    return rows


def checkpoint_file(args: argparse.Namespace, method: str) -> Path | None:
    if args.checkpoint_dir is None:
        return None
    name = (
        f"{method}_seed{args.seed}_train{args.train_len}_"
        f"d{args.embed_dim}_h{args.num_heads}_l{args.layers}_o{args.max_order}.pt"
    )
    return args.checkpoint_dir / name


def checkpoint_config(args: argparse.Namespace, method: str) -> dict[str, int | float | str]:
    return {
        "method": method,
        "vocab_size": VOCAB_SIZE,
        "train_len": args.train_len,
        "embed_dim": args.embed_dim,
        "num_heads": args.num_heads,
        "layers": args.layers,
        "max_order": args.max_order,
        "mlp_ratio": args.mlp_ratio,
        "dropout": args.dropout,
        "init_affine_slope": args.init_affine_slope,
        "gate_init": args.gate_init,
        "rope_base": args.rope_base,
    }


def save_checkpoint(model: ByteLM, args: argparse.Namespace, method: str, path: Path | None) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "method": method,
            "seed": args.seed,
            "model_config": checkpoint_config(args, method),
        },
        path,
    )


def load_checkpoint(
    model: ByteLM,
    args: argparse.Namespace,
    method: str,
    path: Path | None,
    device: torch.device,
) -> None:
    if path is None:
        raise SystemExit("missing checkpoint path")
    if not path.exists():
        raise SystemExit(f"checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device)
    expected = checkpoint_config(args, method)
    found = checkpoint.get("model_config", {})
    mismatches = [
        key
        for key, value in expected.items()
        if key in found and found[key] != value
    ]
    if mismatches:
        detail = ", ".join(f"{key}: expected {expected[key]!r}, found {found.get(key)!r}" for key in mismatches)
        raise SystemExit(f"checkpoint config mismatch for {path}: {detail}")
    model.load_state_dict(checkpoint["model_state"])


@torch.no_grad()
def evaluate(
    model: ByteLM,
    data: torch.Tensor,
    seq_len: int,
    args: argparse.Namespace,
    device: torch.device,
    *,
    eval_batches: int | None = None,
) -> EvalMetrics:
    model.eval()
    batches = args.eval_batches if eval_batches is None else eval_batches
    batch_size = args.batch_size if args.eval_batch_size is None else args.eval_batch_size
    losses = []
    entropies = []
    supports = []
    distances = []
    logit_stds = []
    logit_maxes = []
    for _ in range(batches):
        x, y = sample_batch(data, batch_size, seq_len, device)
        if args.skip_attention_stats:
            logits = model(x, need_weights=False)
            if args.attention_stats_samples > 0:
                entropy, support, distance = sampled_attention_stats(model, x, args.attention_stats_samples)
            else:
                entropy = torch.tensor(float("nan"), device=device)
                support = torch.tensor(float("nan"), device=device)
                distance = torch.tensor(float("nan"), device=device)
        else:
            logits, weights = model(x, need_weights=True)
            entropy, support, distance = attention_stats(weights)
        losses.append(F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1)))
        entropies.append(entropy)
        supports.append(support)
        distances.append(distance)
        logit_stds.append(logits.std())
        logit_maxes.append(logits.detach().abs().max())
    model.train()
    loss = torch.stack(losses).mean()
    return EvalMetrics(
        loss=float(loss.detach().cpu()),
        ppl=float(torch.exp(loss.clamp_max(20.0)).detach().cpu()),
        attention_entropy=float(torch.stack(entropies).mean().detach().cpu()),
        attention_effective_support=float(torch.stack(supports).mean().detach().cpu()),
        average_attention_distance=float(torch.stack(distances).mean().detach().cpu()),
        logit_std=float(torch.stack(logit_stds).mean().detach().cpu()),
        logit_abs_max=float(torch.stack(logit_maxes).mean().detach().cpu()),
    )


def attention_stats(weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    seq_len = weights.shape[-1]
    probs = weights.clamp_min(torch.finfo(weights.dtype).tiny)
    entropy = -(weights * probs.log()).sum(dim=-1)
    idx = torch.arange(seq_len, device=weights.device)
    distance = (idx[:, None] - idx[None, :]).clamp_min(0).to(dtype=weights.dtype)
    avg_distance = (weights * distance[None, None, :, :]).sum(dim=-1)
    return entropy.mean(), entropy.exp().mean(), avg_distance.mean()


def sampled_attention_stats(
    model: ByteLM,
    tokens: torch.Tensor,
    samples: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hidden = model.embedding(tokens)
    for block in model.blocks[:-1]:
        hidden = block(hidden)

    block = model.blocks[-1]
    attn = block.attn
    x = block.norm1(hidden)
    batch, seq_len, _ = x.shape
    sample_count = min(max(1, int(samples)), seq_len)
    query_idx = torch.linspace(0, seq_len - 1, sample_count, device=x.device).round().long().unique()

    qkv = attn.qkv(x).view(batch, seq_len, 3, attn.num_heads, attn.head_dim)
    qkv = qkv.permute(2, 0, 3, 1, 4)
    query, key = qkv[0], qkv[1]
    if attn.uses_rope:
        query, key = apply_rope(
            query,
            key,
            base=attn.rope_base,
            train_len=attn.train_len,
            interpolate=attn.method == "rope_pi",
            dynamic_ntk=attn.method in ("rope_ntk", "rope_ntk_affine"),
            yarn=attn.method in ("rope_yarn", "rope_yarn_affine"),
        )
    elif attn.uses_pj_rotary:
        query, key = apply_exact_pj_rotary(
            query,
            key,
            base=attn.rope_base,
            train_length=attn.train_len,
        )

    scores = (query[:, :, query_idx, :] @ key.transpose(-2, -1)) * attn.scale
    key_idx = torch.arange(seq_len, device=x.device)
    lags = (query_idx[:, None] - key_idx[None, :]).to(dtype=x.dtype)
    mask = lags >= 0
    if attn.pj_bias is not None:
        scores = scores + attn.pj_bias.forward_lags(lags, mask, device=x.device, dtype=x.dtype)[None, :, :, :]
    weights = masked_softmax(scores, mask[None, None, :, :])
    return sampled_attention_stats_from_weights(weights, query_idx)


def sampled_attention_stats_from_weights(
    weights: torch.Tensor,
    query_idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    seq_len = weights.shape[-1]
    probs = weights.clamp_min(torch.finfo(weights.dtype).tiny)
    entropy = -(weights * probs.log()).sum(dim=-1)
    key_idx = torch.arange(seq_len, device=weights.device)
    distance = (query_idx[:, None] - key_idx[None, :]).clamp_min(0).to(dtype=weights.dtype)
    avg_distance = (weights * distance[None, None, :, :]).sum(dim=-1)
    return entropy.mean(), entropy.exp().mean(), avg_distance.mean()


def sample_batch(data: torch.Tensor, batch_size: int, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    if data.numel() <= seq_len + 1:
        raise ValueError("data is too short for requested seq_len")
    starts = torch.randint(0, data.numel() - seq_len - 1, (batch_size,), device=device)
    offsets = torch.arange(seq_len, device=device)
    x = data[starts[:, None] + offsets[None, :]]
    y = data[starts[:, None] + offsets[None, :] + 1]
    return x, y


def make_pj_bias(
    *,
    method: str,
    num_heads: int,
    max_order: int,
    train_len: int,
    init_affine_slope: float,
    gate_init: str,
) -> PJBias | None:
    if method in ("none", "rope", "rope_pi", "rope_ntk", "rope_yarn", "pj_rotary", "grape_m_rope"):
        return None
    use_fj = method in ("full", "fj_affine", "fj_lc")
    use_affine = method in (
        "affine",
        "fj_affine",
        "rope_affine",
        "rope_ntk_affine",
        "rope_yarn_affine",
        "grape_a_alibi",
        "grape_ma_rope_alibi",
        "full",
        "lc_affine",
    )
    use_lc = method in ("full", "fj_lc", "lc_affine")
    pj_bias = PJBias(
        num_heads=num_heads,
        max_order=max_order,
        train_length=train_len,
        use_fj=use_fj,
        use_affine=use_affine,
        use_lc=use_lc,
    )
    with torch.no_grad():
        pj_bias.raw_affine_slope.fill_(inverse_softplus(init_affine_slope))
        init_gate_logits(pj_bias, gate_init)
    return pj_bias


def init_gate_logits(pj_bias: PJBias, gate_init: str) -> None:
    index = {"fj": 0, "affine": 1, "lc": 2}.get(gate_init)
    if index is None or not bool(pj_bias.sector_mask[index]):
        return
    pj_bias.gate_logits.fill_(-2.0)
    pj_bias.gate_logits[:, index].fill_(2.0)


def apply_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    *,
    base: float,
    train_len: int,
    interpolate: bool,
    dynamic_ntk: bool,
    yarn: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    seq_len = query.shape[-2]
    head_dim = query.shape[-1]
    half = head_dim // 2
    positions = torch.arange(seq_len, device=query.device, dtype=query.dtype)
    if interpolate and seq_len > train_len and seq_len > 1:
        positions = positions * float(train_len - 1) / float(seq_len - 1)
    effective_base = base
    if dynamic_ntk and seq_len > train_len:
        exponent = float(head_dim) / max(float(head_dim - 2), 1.0)
        effective_base = base * (float(seq_len) / float(train_len)) ** exponent
    inv_freq = effective_base ** (-torch.arange(0, half, device=query.device, dtype=query.dtype) / float(half))
    attention_scale = 1.0
    if yarn and seq_len > train_len:
        factor = float(seq_len) / float(train_len)
        inv_freq = yarn_inv_freq(inv_freq, train_len=train_len, factor=factor)
        attention_scale = 1.0 + 0.1 * math.log(factor)
    angles = positions[:, None] * inv_freq[None, :]
    cos = torch.cos(angles)[None, None, :, :] * attention_scale
    sin = torch.sin(angles)[None, None, :, :] * attention_scale
    return rotate(query, cos, sin), rotate(key, cos, sin)


def yarn_inv_freq(
    inv_freq: torch.Tensor,
    *,
    train_len: int,
    factor: float,
    beta_slow: float = 1.0,
    beta_fast: float = 32.0,
) -> torch.Tensor:
    """YaRN-style low-frequency extrapolation and high-frequency interpolation."""

    rotations = float(train_len) * inv_freq / (2.0 * math.pi)
    extrapolate_weight = ((rotations - beta_slow) / (beta_fast - beta_slow)).clamp(0.0, 1.0)
    interpolated = inv_freq / factor
    return interpolated * (1.0 - extrapolate_weight) + inv_freq * extrapolate_weight


def rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    first, second = x[..., ::2], x[..., 1::2]
    rotated_first = first * cos - second * sin
    rotated_second = first * sin + second * cos
    out = torch.empty_like(x)
    out[..., ::2] = rotated_first
    out[..., 1::2] = rotated_second
    return out


def masked_softmax(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    min_value = torch.finfo(scores.dtype).min
    weights = torch.softmax(scores.masked_fill(~mask, min_value), dim=-1)
    weights = weights.masked_fill(~mask, 0.0)
    denom = weights.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(weights.dtype).tiny)
    return weights / denom


def block_causal_attention_mask(
    start: int,
    end: int,
    seq_len: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    query_idx = torch.arange(start, end, device=device)
    key_idx = torch.arange(seq_len, device=device)
    return key_idx[None, :] <= query_idx[:, None]


def pj_summary(model: ByteLM) -> tuple[list[float] | None, list[float], dict[str, list[float]]]:
    gates = []
    slopes = []
    fj_masses = []
    lc_masses = []
    for block in model.blocks:
        pj_bias = block.attn.pj_bias
        if pj_bias is None:
            continue
        gates.append(pj_bias.sector_gates().detach().mean(dim=0).cpu())
        slopes.extend(float(value) for value in pj_bias.affine_slopes().detach().cpu().tolist())
        masses = pj_bias.order_masses()
        if "fj" in masses:
            fj_masses.append(masses["fj"].detach().mean(dim=0).cpu())
        if "lc" in masses:
            lc_masses.append(masses["lc"].detach().mean(dim=0).cpu())
    if not gates:
        return None, [], {}
    mass_summary = {}
    if fj_masses:
        mass_summary["fj"] = [float(value) for value in torch.stack(fj_masses).mean(dim=0).tolist()]
    if lc_masses:
        mass_summary["lc"] = [float(value) for value in torch.stack(lc_masses).mean(dim=0).tolist()]
    return [float(value) for value in torch.stack(gates).mean(dim=0).tolist()], slopes, mass_summary


def load_corpus(args: argparse.Namespace) -> torch.Tensor:
    if args.text_file is not None:
        raw = args.text_file.read_bytes()
    else:
        raw = (FALLBACK_TEXT * max(1, args.repeat_fallback)).encode("utf-8")
    if len(raw) < max(parse_int_list(args.eval_lens) + [args.train_len]) + 2:
        raise SystemExit("corpus is too short for requested train/eval lengths")
    return torch.tensor(list(raw), dtype=torch.long)


def split_data(data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    split = max(1, int(0.9 * data.numel()))
    if data.numel() - split < 512:
        split = max(1, data.numel() // 2)
    return data[:split], data[split:]


def write_outputs(rows: list[dict[str, float | int | str]], args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "results.csv"
    json_path = args.out_dir / "summary.json"
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    config = vars(args).copy()
    config["out_dir"] = str(args.out_dir)
    if config.get("text_file") is not None:
        config["text_file"] = str(config["text_file"])
    if config.get("checkpoint_dir") is not None:
        config["checkpoint_dir"] = str(config["checkpoint_dir"])
    with json_path.open("w", encoding="utf-8") as file:
        json.dump({"config": config, "rows": rows, "summary": summarize(rows)}, file, indent=2)
    print(f"Wrote CSV: {csv_path}")
    print(f"Wrote summary: {json_path}")


def summarize(rows: list[dict[str, float | int | str]]) -> dict[str, dict[str, float | int]]:
    out = {}
    for method, eval_len in sorted(set((str(row["method"]), int(row["eval_len"])) for row in rows)):
        subset = [row for row in rows if row["method"] == method and int(row["eval_len"]) == eval_len]
        name = f"{method}@{eval_len}"
        out[name] = {
            "n": len(subset),
            "loss_mean": mean_row(subset, "loss"),
            "ppl_mean": mean_row(subset, "ppl"),
            "attention_entropy_mean": mean_row(subset, "attention_entropy"),
            "average_attention_distance_mean": mean_row(subset, "average_attention_distance"),
            "gate_fj_mean": mean_row(subset, "gate_fj"),
            "gate_affine_mean": mean_row(subset, "gate_affine"),
            "gate_lc_mean": mean_row(subset, "gate_lc"),
        }
    return out


def print_summary(rows: list[dict[str, float | int | str]]) -> None:
    print("Byte LM smoke summary")
    print("method       eval_len  loss    ppl      entropy  avg_dist  gates(fj/a/lc)")
    print("-" * 86)
    for method, eval_len in sorted(set((str(row["method"]), int(row["eval_len"])) for row in rows)):
        subset = [row for row in rows if row["method"] == method and int(row["eval_len"]) == eval_len]
        print(
            f"{method:<12} {eval_len:<8d} {mean_row(subset, 'loss'):.4f} "
            f"{mean_row(subset, 'ppl'):.2f}  {mean_row(subset, 'attention_entropy'):.3f}   "
            f"{mean_row(subset, 'average_attention_distance'):.2f}    {gate_string(subset)}"
        )


def mean_row(rows: list[dict[str, float | int | str]], key: str) -> float:
    saw_key = False
    values = []
    for row in rows:
        if key not in row or row[key] == "":
            continue
        saw_key = True
        value = float(row[key])
        if not math.isnan(value):
            values.append(value)
    if values:
        return sum(values) / len(values)
    return float("nan") if saw_key else 0.0


def gate_string(rows: list[dict[str, float | int | str]]) -> str:
    if not rows or "gate_fj" not in rows[0]:
        return "n/a"
    return "{:.2f}/{:.2f}/{:.2f}".format(
        mean_row(rows, "gate_fj"),
        mean_row(rows, "gate_affine"),
        mean_row(rows, "gate_lc"),
    )


def parse_int_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in str(raw).split(",") if item.strip()]


def parse_str_list(raw: str) -> list[str]:
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def inverse_softplus(value: float) -> float:
    if value > 20.0:
        return value
    return math.log(math.expm1(value))


if __name__ == "__main__":
    main()
