"""Synthetic query-LM bridge from teacher kernels to trainable attention.

The sequence contains random bit tokens followed by a query token. The final
query must classify a teacher-weighted functional of the previous bits. This is
the Phase C bridge between kernel recovery diagnostics and a trainable causal
attention model.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
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
        "/home/riven/JordanKac/.venv/bin/python experiments/synthetic_query_lm.py"
    ) from exc

from pjrope import torch_backend as tb
from pjrope.attention import PJBias, PJCausalSelfAttention


TARGETS = (
    "phase",
    "first_jet",
    "second_jet",
    "third_jet",
    "affine",
    "recency_weak_jet",
    "lc_core",
    "lc_affine",
)
SECTORS = (
    "none",
    "affine",
    "fj",
    "fj_affine",
    "lc_affine",
    "full",
    "grape_m_rope",
    "grape_a_alibi",
    "grape_ma_rope_alibi",
)


@dataclass(frozen=True)
class QueryBatch:
    tokens: torch.Tensor
    bits: torch.Tensor
    labels: torch.Tensor
    scores: torch.Tensor


@dataclass(frozen=True)
class EvalMetrics:
    loss: float
    accuracy: float
    label_balance: float
    mean_confidence: float
    attention_entropy: float
    attention_effective_support: float


class TransformerBlock(torch.nn.Module):
    def __init__(
        self,
        *,
        embed_dim: int,
        num_heads: int,
        pj_bias: PJBias | None,
        mlp_ratio: int,
        dropout: float,
        freeze_qk: bool,
    ) -> None:
        super().__init__()
        self.attn = PJCausalSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            pj_bias=pj_bias,
            dropout=dropout,
            include_self=False,
        )
        self.norm1 = torch.nn.LayerNorm(embed_dim)
        self.norm2 = torch.nn.LayerNorm(embed_dim)
        hidden_dim = mlp_ratio * embed_dim
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, embed_dim),
            torch.nn.Dropout(dropout),
        )
        if freeze_qk:
            freeze_query_key(self.attn)

    def forward(self, hidden: torch.Tensor, *, need_weights: bool = False):
        if need_weights:
            attn_out, weights = self.attn(self.norm1(hidden), need_weights=True)
            hidden = hidden + attn_out
            hidden = hidden + self.mlp(self.norm2(hidden))
            return hidden, weights
        hidden = hidden + self.attn(self.norm1(hidden))
        hidden = hidden + self.mlp(self.norm2(hidden))
        return hidden


class QueryLMModel(torch.nn.Module):
    def __init__(
        self,
        *,
        seq_len: int,
        embed_dim: int,
        num_heads: int,
        layers: int,
        max_order: int,
        sector: str,
        target: str,
        teacher: str,
        omega_cycles: float,
        init_affine_slope: float,
        gate_init: str,
        order_init: str,
        mlp_ratio: int,
        dropout: float,
        freeze_qk: bool,
    ) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(3, embed_dim)
        self.blocks = torch.nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    pj_bias=make_pj_bias(
                        sector=sector,
                        target=target,
                        teacher=teacher,
                        num_heads=num_heads,
                        max_order=max_order,
                        seq_len=seq_len,
                        omega_cycles=omega_cycles,
                        init_affine_slope=init_affine_slope,
                        gate_init=gate_init,
                        order_init=order_init,
                    ),
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    freeze_qk=freeze_qk,
                )
                for _ in range(layers)
            ]
        )
        self.norm = torch.nn.LayerNorm(embed_dim)
        self.head = torch.nn.Linear(embed_dim, 2)

    def forward(self, tokens: torch.Tensor, *, need_weights: bool = False):
        hidden = self.embedding(tokens)
        last_weights = None
        for index, block in enumerate(self.blocks):
            if need_weights and index == len(self.blocks) - 1:
                hidden, last_weights = block(hidden, need_weights=True)
            else:
                hidden = block(hidden)
        logits = self.head(self.norm(hidden[:, -1]))
        if need_weights:
            return logits, last_weights
        return logits


def main() -> None:
    args = parse_args()
    validate_args(args)
    device = torch.device(args.device)
    dtype = torch.float64 if args.float64 else torch.float32

    rows = []
    for seed in parse_int_list(args.seeds):
        for sector in parse_str_list(args.sectors):
            rows.extend(run_one(args, seed=seed, sector=sector, device=device, dtype=dtype))

    write_outputs(rows, args)
    print_summary(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", choices=["signed", "attention"], default="attention")
    parser.add_argument("--target", choices=TARGETS, default="recency_weak_jet")
    parser.add_argument("--train-len", type=int, default=96)
    parser.add_argument("--train-seq-lens", default="")
    parser.add_argument("--train-length-sampling", choices=["cycle", "random"], default="cycle")
    parser.add_argument("--eval-lens", default="96,192")
    parser.add_argument("--omega-cycles", type=float, default=17.0)
    parser.add_argument("--attention-lambda", type=float, default=4.0)
    parser.add_argument("--sectors", default="full")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--embed-dim", type=int, default=96)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--max-order", type=int, default=3)
    parser.add_argument("--mlp-ratio", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--steps", type=int, default=350)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--init-affine-slope", type=float, default=0.4)
    parser.add_argument("--gate-init", choices=["uniform", "auto"], default="uniform")
    parser.add_argument("--order-init", choices=["uniform", "auto"], default="uniform")
    parser.add_argument("--freeze-qk", action="store_true")
    parser.add_argument("--median-threshold", action="store_true")
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--float64", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=Path("runs/synthetic_query_lm"))
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.train_len < 3:
        raise SystemExit("train-len must be at least 3")
    for train_seq_len in training_lengths(args):
        if train_seq_len < 3:
            raise SystemExit("all train-seq-lens must be at least 3")
    if args.num_heads < 1 or args.embed_dim % args.num_heads != 0:
        raise SystemExit("embed-dim must be divisible by num-heads")
    if args.layers < 1:
        raise SystemExit("layers must be positive")
    for sector in parse_str_list(args.sectors):
        if sector not in SECTORS:
            raise SystemExit(f"unknown sector: {sector}")
    for eval_len in parse_int_list(args.eval_lens):
        if eval_len < 3:
            raise SystemExit("all eval-lens must be at least 3")


def run_one(
    args: argparse.Namespace,
    *,
    seed: int,
    sector: str,
    device: torch.device,
    dtype: torch.dtype,
) -> list[dict[str, float | int | str]]:
    torch.manual_seed(seed)
    model = QueryLMModel(
        seq_len=args.train_len,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        layers=args.layers,
        max_order=args.max_order,
        sector=sector,
        target=args.target,
        teacher=args.teacher,
        omega_cycles=args.omega_cycles,
        init_affine_slope=args.init_affine_slope,
        gate_init=args.gate_init,
        order_init=args.order_init,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        freeze_qk=args.freeze_qk,
    ).to(device=device, dtype=dtype)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_seq_lens = training_lengths(args)

    initial = evaluate(model, args, args.train_len, device, eval_batches=max(1, args.eval_batches // 2))
    for step in range(1, args.steps + 1):
        seq_len = choose_training_length(train_seq_lens, step=step, sampling=args.train_length_sampling, device=device)
        batch = sample_batch(args, seq_len, device=device, dtype=dtype)
        logits = model(batch.tokens)
        loss = F.cross_entropy(logits, batch.labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if args.log_every and step % args.log_every == 0:
            metrics = evaluate(model, args, args.train_len, device, eval_batches=2)
            print(
                f"sector={sector:<9} seed={seed} step={step:04d} seq_len={seq_len:<4d} "
                f"loss={float(loss.detach().cpu()):.4f} acc={metrics.accuracy:.4f}"
            )

    rows = []
    gates, slopes, masses = pj_summary(model)
    teacher_stats = teacher_distribution_stats(args, args.train_len, device, dtype)
    for eval_len in parse_int_list(args.eval_lens):
        final = evaluate(model, args, eval_len, device, eval_batches=args.eval_batches)
        row: dict[str, float | int | str] = {
            "seed": seed,
            "sector": sector,
            "teacher": args.teacher,
            "target": args.target,
            "train_len": args.train_len,
            "train_seq_lens": ",".join(str(value) for value in train_seq_lens),
            "eval_len": eval_len,
            "omega_cycles": args.omega_cycles,
            "attention_lambda": args.attention_lambda,
            "initial_loss": initial.loss,
            "initial_accuracy": initial.accuracy,
            "loss": final.loss,
            "accuracy": final.accuracy,
            "label_balance": final.label_balance,
            "mean_confidence": final.mean_confidence,
            "attention_entropy": final.attention_entropy,
            "attention_effective_support": final.attention_effective_support,
            "teacher_entropy": teacher_stats["entropy"],
            "teacher_effective_support": teacher_stats["effective_support"],
            "teacher_kernel_norm": teacher_stats["kernel_norm"],
        }
        if gates is not None:
            row["gate_fj"] = gates[0]
            row["gate_affine"] = gates[1]
            row["gate_lc"] = gates[2]
        if slopes:
            row["affine_slope_mean"] = statistics.fmean(slopes)
        for name, values in masses.items():
            for order, value in enumerate(values):
                row[f"{name}_mass_r{order}"] = value
        rows.append(row)
    return rows


def sample_batch(
    args: argparse.Namespace,
    seq_len: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> QueryBatch:
    bits = torch.randint(0, 2, (args.batch_size, seq_len - 1), device=device, dtype=torch.long)
    bits = bits.to(dtype=dtype).mul_(2.0).sub_(1.0)
    tokens = (bits > 0).to(torch.long)
    query = torch.full((args.batch_size, 1), 2, device=device, dtype=torch.long)
    tokens = torch.cat([tokens, query], dim=1)

    coeff = teacher_coefficients(args, seq_len, device=device, dtype=dtype)
    scores = (bits * coeff[None, :]).sum(dim=-1)
    threshold = scores.median() if args.median_threshold else torch.zeros((), device=device, dtype=dtype)
    labels = (scores > threshold).to(torch.long)
    return QueryBatch(tokens=tokens, bits=bits, labels=labels, scores=scores)


def training_lengths(args: argparse.Namespace) -> list[int]:
    if str(args.train_seq_lens).strip():
        return parse_int_list(args.train_seq_lens)
    return [int(args.train_len)]


def choose_training_length(
    train_seq_lens: list[int],
    *,
    step: int,
    sampling: str,
    device: torch.device,
) -> int:
    if len(train_seq_lens) == 1:
        return train_seq_lens[0]
    if sampling == "cycle":
        return train_seq_lens[(step - 1) % len(train_seq_lens)]
    if sampling == "random":
        index = int(torch.randint(0, len(train_seq_lens), (), device=device).detach().cpu())
        return train_seq_lens[index]
    raise ValueError(f"unknown train-length-sampling: {sampling}")


def teacher_coefficients(
    args: argparse.Namespace,
    seq_len: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    kernel = teacher_kernel_by_position(
        target=args.target,
        seq_len=seq_len,
        train_len=args.train_len,
        omega_cycles=args.omega_cycles,
        device=device,
        dtype=dtype,
    )
    if args.teacher == "signed":
        return kernel / kernel.square().sum().sqrt().clamp_min(torch.finfo(dtype).eps)
    if args.teacher == "attention":
        return torch.softmax(args.attention_lambda * kernel, dim=0)
    raise ValueError(f"unknown teacher: {args.teacher}")


def teacher_kernel_by_position(
    *,
    target: str,
    seq_len: int,
    train_len: int,
    omega_cycles: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    lags = torch.arange(seq_len - 1, 0, -1, device=device, dtype=dtype)
    length = float(train_len)
    omega = 2.0 * math.pi * float(omega_cycles) / length
    x = lags / length
    if target == "phase":
        return torch.cos(omega * lags)
    if target == "first_jet":
        return x * torch.cos(omega * lags)
    if target == "second_jet":
        return x.square() * torch.exp(-0.1 * x) * torch.cos(omega * lags)
    if target == "third_jet":
        return x.pow(3) * torch.exp(-0.1 * x) * torch.cos(omega * lags)
    if target == "affine":
        return -0.4 * x
    if target == "recency_weak_jet":
        return -0.4 * x + 0.15 * x.square() * torch.cos(omega * lags)
    if target == "lc_core":
        phi = tb.phi_l(lags, length)
        beta = tb.beta_l(lags, length)
        return torch.cos(omega * phi) * (1.0 + 0.4 * beta + 0.2 * beta.square())
    if target == "lc_affine":
        phi = tb.phi_l(lags, length)
        beta = tb.beta_l(lags, length)
        core = torch.cos(omega * phi) * (1.0 + 0.4 * beta + 0.2 * beta.square())
        return core - 0.25 * tb.eta_l(lags, length)
    raise ValueError(f"unknown target: {target}")


@torch.no_grad()
def evaluate(
    model: QueryLMModel,
    args: argparse.Namespace,
    seq_len: int,
    device: torch.device,
    *,
    eval_batches: int,
) -> EvalMetrics:
    model.eval()
    losses = []
    correct = 0
    total = 0
    label_balance = []
    confidence = []
    entropies = []
    supports = []
    dtype = next(model.parameters()).dtype
    for _ in range(eval_batches):
        batch = sample_batch(args, seq_len, device=device, dtype=dtype)
        logits, weights = model(batch.tokens, need_weights=True)
        losses.append(F.cross_entropy(logits, batch.labels))
        probs = torch.softmax(logits, dim=-1)
        pred = probs.argmax(dim=-1)
        correct += int((pred == batch.labels).sum().detach().cpu())
        total += int(batch.labels.numel())
        label_balance.append(batch.labels.to(torch.float32).mean())
        confidence.append(probs.max(dim=-1).values.mean())
        entropy, support = final_query_attention_stats(weights)
        entropies.append(entropy)
        supports.append(support)
    model.train()
    return EvalMetrics(
        loss=float(torch.stack(losses).mean().detach().cpu()),
        accuracy=correct / total,
        label_balance=float(torch.stack(label_balance).mean().detach().cpu()),
        mean_confidence=float(torch.stack(confidence).mean().detach().cpu()),
        attention_entropy=float(torch.stack(entropies).mean().detach().cpu()),
        attention_effective_support=float(torch.stack(supports).mean().detach().cpu()),
    )


def final_query_attention_stats(weights: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
    if weights is None:
        zero = torch.zeros(())
        return zero, zero
    final = weights[:, :, -1, :-1]
    final = final / final.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(final.dtype).tiny)
    entropy = -(final * final.clamp_min(torch.finfo(final.dtype).tiny).log()).sum(dim=-1)
    return entropy.mean(), entropy.exp().mean()


def teacher_distribution_stats(
    args: argparse.Namespace,
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, float]:
    kernel = teacher_kernel_by_position(
        target=args.target,
        seq_len=seq_len,
        train_len=args.train_len,
        omega_cycles=args.omega_cycles,
        device=device,
        dtype=dtype,
    )
    probs = torch.softmax(args.attention_lambda * kernel, dim=0)
    entropy = -(probs * probs.clamp_min(torch.finfo(dtype).tiny).log()).sum()
    return {
        "entropy": float(entropy.detach().cpu()),
        "effective_support": float(entropy.exp().detach().cpu()),
        "kernel_norm": float(kernel.square().sum().sqrt().detach().cpu()),
    }


def make_pj_bias(
    *,
    sector: str,
    target: str,
    teacher: str,
    num_heads: int,
    max_order: int,
    seq_len: int,
    omega_cycles: float,
    init_affine_slope: float,
    gate_init: str,
    order_init: str,
) -> PJBias | None:
    if sector == "none":
        return None
    # The GRAPE rows are exact special-case controls, not full learned GRAPE.
    # GRAPE-M/RoPE is represented here by the order-zero scalar character
    # branch; GRAPE-A/ALiBi by the affine branch; GRAPE-M+A by their direct sum.
    special_order_zero = sector in ("grape_m_rope", "grape_ma_rope_alibi")
    use_fj = sector in ("fj", "fj_affine", "full", "grape_m_rope", "grape_ma_rope_alibi")
    use_affine = sector in ("affine", "fj_affine", "lc_affine", "full", "grape_a_alibi", "grape_ma_rope_alibi")
    use_lc = sector in ("lc_affine", "full")
    pj_bias = PJBias(
        num_heads=num_heads,
        max_order=0 if special_order_zero else max_order,
        train_length=seq_len,
        use_fj=use_fj,
        use_affine=use_affine,
        use_lc=use_lc,
        init_omega_cycles=omega_cycles,
    )
    with torch.no_grad():
        pj_bias.raw_affine_slope.fill_(inverse_softplus(init_affine_slope))
        if gate_init == "auto":
            init_gate_logits(pj_bias, preferred_sector(target, teacher))
        if order_init == "auto":
            init_order_logits(pj_bias, preferred_order(target))
    return pj_bias


def init_gate_logits(pj_bias: PJBias, preferred: str) -> None:
    index = {"fj": 0, "affine": 1, "lc": 2}[preferred]
    if not bool(pj_bias.sector_mask[index]):
        return
    pj_bias.gate_logits.fill_(-2.0)
    pj_bias.gate_logits[:, index].fill_(2.0)


def init_order_logits(pj_bias: PJBias, order: int) -> None:
    if order < 0 or order > pj_bias.max_order:
        return
    if pj_bias.use_fj:
        pj_bias.fj_alpha_logits.fill_(-2.0)
        pj_bias.fj_alpha_logits[:, order].fill_(2.0)
    if pj_bias.use_lc:
        pj_bias.lc_alpha_logits.fill_(-2.0)
        pj_bias.lc_alpha_logits[:, order].fill_(2.0)


def preferred_sector(target: str, teacher: str) -> str:
    if target in ("lc_core", "lc_affine"):
        return "lc"
    if target == "affine":
        return "affine"
    if target == "recency_weak_jet" and teacher == "attention":
        return "affine"
    return "fj"


def preferred_order(target: str) -> int:
    return {
        "phase": 0,
        "first_jet": 1,
        "second_jet": 2,
        "third_jet": 3,
        "affine": 0,
        "recency_weak_jet": 2,
        "lc_core": 0,
        "lc_affine": 0,
    }[target]


def pj_summary(model: QueryLMModel) -> tuple[list[float] | None, list[float], dict[str, list[float]]]:
    gates = []
    slopes = []
    fj_masses = []
    lc_masses = []
    for block in model.blocks:
        pj_bias = block.attn.pj_bias
        if pj_bias is None:
            continue
        gates.append(pj_bias.sector_gates().detach().mean(dim=0).cpu())
        slopes.extend(float(v) for v in pj_bias.affine_slopes().detach().cpu().tolist())
        masses = pj_bias.order_masses()
        if "fj" in masses:
            fj_masses.append(masses["fj"].detach().mean(dim=0).cpu())
        if "lc" in masses:
            lc_masses.append(masses["lc"].detach().mean(dim=0).cpu())
    if not gates:
        return None, [], {}
    mass_summary = {}
    if fj_masses:
        mass_summary["fj"] = [float(v) for v in torch.stack(fj_masses).mean(dim=0).tolist()]
    if lc_masses:
        mass_summary["lc"] = [float(v) for v in torch.stack(lc_masses).mean(dim=0).tolist()]
    return [float(v) for v in torch.stack(gates).mean(dim=0).tolist()], slopes, mass_summary


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
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "config": config,
                "rows": rows,
                "summary": summarize_rows(rows),
            },
            file,
            indent=2,
        )
    print(f"Wrote CSV: {csv_path}")
    print(f"Wrote summary: {json_path}")


def summarize_rows(rows: list[dict[str, float | int | str]]) -> dict[str, dict[str, float]]:
    summary = {}
    keys = sorted(set((str(row["sector"]), int(row["eval_len"])) for row in rows))
    for sector, eval_len in keys:
        subset = [row for row in rows if row["sector"] == sector and int(row["eval_len"]) == eval_len]
        name = f"{sector}@{eval_len}"
        summary[name] = {
            "n": len(subset),
            "accuracy_mean": mean_row(subset, "accuracy"),
            "loss_mean": mean_row(subset, "loss"),
            "label_balance_mean": mean_row(subset, "label_balance"),
            "attention_effective_support_mean": mean_row(subset, "attention_effective_support"),
        }
    return summary


def print_summary(rows: list[dict[str, float | int | str]]) -> None:
    print("Synthetic query-LM summary")
    print("sector     eval_len  n  acc_mean  loss_mean  balance  gates(fj/a/lc)")
    print("-" * 78)
    seen = sorted(set((str(row["sector"]), int(row["eval_len"])) for row in rows))
    for sector, eval_len in seen:
        subset = [row for row in rows if row["sector"] == sector and int(row["eval_len"]) == eval_len]
        gates = gate_string(subset)
        print(
            f"{sector:<10} {eval_len:<8d} {len(subset):<2d} "
            f"{mean_row(subset, 'accuracy'):.4f}    {mean_row(subset, 'loss'):.4f}    "
            f"{mean_row(subset, 'label_balance'):.3f}    {gates}"
        )


def mean_row(rows: list[dict[str, float | int | str]], key: str) -> float:
    values = [float(row[key]) for row in rows if key in row]
    return statistics.fmean(values) if values else 0.0


def gate_string(rows: list[dict[str, float | int | str]]) -> str:
    if not rows or "gate_fj" not in rows[0]:
        return "n/a"
    return f"{mean_row(rows, 'gate_fj'):.2f}/{mean_row(rows, 'gate_affine'):.2f}/{mean_row(rows, 'gate_lc'):.2f}"


def parse_int_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in str(raw).split(",") if item.strip()]


def parse_str_list(raw: str) -> list[str]:
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def inverse_softplus(value: float) -> float:
    if value > 20.0:
        return value
    return math.log(math.expm1(value))


def freeze_query_key(attn: PJCausalSelfAttention) -> None:
    embed_dim = attn.embed_dim
    with torch.no_grad():
        attn.qkv.weight[: 2 * embed_dim].zero_()
        if attn.qkv.bias is not None:
            attn.qkv.bias[: 2 * embed_dim].zero_()
    attn.qkv.weight.register_hook(lambda grad: zero_qk_grad(grad, embed_dim))
    if attn.qkv.bias is not None:
        attn.qkv.bias.register_hook(lambda grad: zero_qk_grad(grad, embed_dim))


def zero_qk_grad(grad: torch.Tensor, embed_dim: int) -> torch.Tensor:
    grad = grad.clone()
    grad[: 2 * embed_dim].zero_()
    return grad


if __name__ == "__main__":
    main()
