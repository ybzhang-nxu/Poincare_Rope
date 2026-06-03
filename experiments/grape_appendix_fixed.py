"""GRAPE appendix fixed primitive-basis projection.

This is a primitive-containment probe, not a full GRAPE performance comparison.
All bases use the same frequency grid; PJ variants add finite jet orders at the
same frequency rather than unrelated frequencies.
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
        "/home/riven/JordanKac/.venv/bin/python experiments/grape_appendix_fixed.py"
    ) from exc

from pjrope.grape_special import (
    GrapeAALiBi,
    GrapeMA,
    GrapeMCanonicalRoPE,
    pj_fj_basis,
    pj_lc_basis,
    r2_score,
    solve_projection,
    target_kernel,
)


BASIS_LABELS = {
    "grape_m_rope": "GRAPE-M/RoPE",
    "grape_a_alibi": "GRAPE-A/ALiBi",
    "grape_ma_rope_alibi": "GRAPE-M+A",
    "pj_fj_R1": "PJ-FJ R1",
    "pj_fj_R2": "PJ-FJ R2",
    "pj_lc_R3": "PJ-LC",
}

TARGET_LABELS = {
    "phase": r"\(\cos\omega d\)",
    "affine": r"\(-d/L\)",
    "first_jet": r"\((d/L)\cos\omega d\)",
    "second_jet": r"\((d/L)^2\cos\omega d\)",
    "lc_core": "LC core",
}


@dataclass(frozen=True)
class ProjectionRow:
    omega_cycles: float
    train_len: int
    eval_len: int
    target: str
    target_label: str
    basis: str
    basis_label: str
    basis_dim: int
    train_mse: float
    mse: float
    r2: float
    condition_number: float


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype = torch.float64 if args.float64 else torch.float32
    torch.manual_seed(args.seed)

    train_lags = torch.arange(args.train_len, device=device, dtype=dtype)
    rows: list[ProjectionRow] = []
    for omega_cycles in parse_float_list(args.omega_cycles):
        omega = 2.0 * math.pi * omega_cycles / float(args.train_len)
        basis_fns = make_basis_fns(args.train_len, omega, args.max_order)
        for target_name in parse_str_list(args.targets):
            y_train = target_kernel(target_name, train_lags, length=args.train_len, omega=omega)
            for basis_name, basis_fn in basis_fns.items():
                x_train_raw = basis_fn(train_lags)
                x_train, stats = standardize(x_train_raw)
                weights, pred_train = solve_projection(x_train, y_train)
                train_mse = mse(pred_train, y_train)
                cond = condition_number(append_intercept(x_train))

                for eval_len in parse_int_list(args.eval_lens):
                    eval_lags = torch.arange(eval_len, device=device, dtype=dtype)
                    y_eval = target_kernel(target_name, eval_lags, length=args.train_len, omega=omega)
                    x_eval = apply_standardize(basis_fn(eval_lags), stats)
                    pred_eval = append_intercept(x_eval) @ weights
                    pred_eval = pred_eval.squeeze(-1)
                    rows.append(
                        ProjectionRow(
                            omega_cycles=omega_cycles,
                            train_len=args.train_len,
                            eval_len=eval_len,
                            target=target_name,
                            target_label=TARGET_LABELS[target_name],
                            basis=basis_name,
                            basis_label=BASIS_LABELS[basis_name],
                            basis_dim=int(x_train_raw.shape[-1]),
                            train_mse=float(train_mse.detach().cpu()),
                            mse=float(mse(pred_eval, y_eval).detach().cpu()),
                            r2=float(r2_score(pred_eval, y_eval).detach().cpu()),
                            condition_number=cond,
                        )
                    )

    write_outputs(rows, args)
    print_summary(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-len", type=int, default=1024)
    parser.add_argument("--eval-lens", default="1024,4096,32768")
    parser.add_argument("--omega-cycles", default="17")
    parser.add_argument(
        "--targets",
        default="phase,affine,first_jet,second_jet,lc_core",
        help="Comma-separated target kernels.",
    )
    parser.add_argument("--max-order", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--float64", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=Path("runs/grape_appendix_fixed"))
    return parser.parse_args()


def make_basis_fns(length: int, omega: float, max_order: int):
    lc_order = min(max_order, 3)
    return {
        "grape_m_rope": lambda d: GrapeMCanonicalRoPE(length, omega).features(d),
        "grape_a_alibi": lambda d: GrapeAALiBi(length).features(d),
        "grape_ma_rope_alibi": lambda d: GrapeMA(length, omega).features(d),
        "pj_fj_R1": lambda d: pj_fj_basis(d, length=length, omega=omega, max_order=1),
        "pj_fj_R2": lambda d: pj_fj_basis(d, length=length, omega=omega, max_order=2),
        "pj_lc_R3": lambda d: pj_lc_basis(d, length=length, omega=omega, max_order=lc_order),
    }


def standardize(x: torch.Tensor, eps: float = 1e-8):
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True).clamp_min(eps)
    return (x - mean) / std, (mean, std)


def apply_standardize(x: torch.Tensor, stats):
    mean, std = stats
    return (x - mean) / std


def append_intercept(x: torch.Tensor) -> torch.Tensor:
    return torch.cat([x, torch.ones(x.shape[0], 1, device=x.device, dtype=x.dtype)], dim=-1)


def condition_number(design: torch.Tensor, eps: float = 1e-12) -> float:
    singular_values = torch.linalg.svdvals(design.to(torch.float64))
    value = singular_values.max() / singular_values.min().clamp_min(eps)
    return float(value.detach().cpu())


def mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (prediction - target).square().mean()


def write_outputs(rows: list[ProjectionRow], args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "results.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    payload = {
        "config": {
            "train_len": args.train_len,
            "eval_lens": args.eval_lens,
            "omega_cycles": args.omega_cycles,
            "targets": args.targets,
            "max_order": args.max_order,
            "float64": bool(args.float64),
            "device": str(args.device),
            "out_dir": str(args.out_dir),
        },
        "notes": [
            "GRAPE rows are exact special-case controls, not full learned GRAPE.",
            "All projections use the same frequency grid; PJ-FJ adds jet orders at that frequency.",
            "This is a primitive-containment probe, not a leaderboard comparison.",
        ],
        "rows": [asdict(row) for row in rows],
    }
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_readme(rows, args.out_dir)


def write_readme(rows: list[ProjectionRow], out_dir: Path) -> None:
    final_len = max(row.eval_len for row in rows)
    lines = [
        "# GRAPE Appendix Fixed Projection",
        "",
        "This directory contains primitive-basis least-squares projections for the",
        "GRAPE appendix.  `GRAPE-M/RoPE`, `GRAPE-A/ALiBi`, and `GRAPE-M+A` are",
        "exact special-case controls, not full learned GRAPE implementations.",
        "",
        "All projections use the same frequency grid.  PJ-FJ variants add finite",
        "jet orders at the same frequency, so the table should be read as a",
        "primitive-containment probe rather than a full method comparison.",
        "",
        f"## R2 at eval length {final_len}",
        "",
        "| Target | Basis | R2 | MSE |",
        "|---|---|---:|---:|",
    ]
    for row in sorted([r for r in rows if r.eval_len == final_len], key=lambda r: (r.target, r.basis)):
        lines.append(f"| {row.target_label} | {row.basis_label} | {row.r2:.6f} | {row.mse:.3e} |")
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_summary(rows: list[ProjectionRow]) -> None:
    final_len = max(row.eval_len for row in rows)
    print(f"GRAPE appendix fixed projection @ eval_len={final_len}")
    print("target       basis                 R2        MSE")
    print("-" * 62)
    for row in sorted([r for r in rows if r.eval_len == final_len], key=lambda r: (r.target, r.basis)):
        print(f"{row.target:<12} {row.basis_label:<20} {row.r2:8.5f} {row.mse:10.3e}")


def parse_int_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in str(raw).split(",") if item.strip()]


def parse_float_list(raw: str | float) -> list[float]:
    return [float(item.strip()) for item in str(raw).split(",") if item.strip()]


def parse_str_list(raw: str) -> list[str]:
    values = [item.strip() for item in str(raw).split(",") if item.strip()]
    unknown = [value for value in values if value not in TARGET_LABELS]
    if unknown:
        raise SystemExit(f"unknown targets: {', '.join(unknown)}")
    return values


if __name__ == "__main__":
    main()

