"""Sweep attention-teacher support for Phase C synthetic query-LM."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import torch
except Exception as exc:  # pragma: no cover - command-line guard
    raise SystemExit(
        "This experiment requires PyTorch. Try:\n"
        "/home/riven/JordanKac/.venv/bin/python experiments/phase_c_attention_support_sweep.py"
    ) from exc

from experiments.synthetic_query_lm import parse_int_list, parse_str_list, run_one, validate_args


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype = torch.float64 if args.float64 else torch.float32
    lambdas = parse_float_list(args.attention_lambdas)
    sectors = parse_str_list(args.sectors)
    seeds = parse_int_list(args.seeds)

    rows = []
    for attention_lambda in lambdas:
        run_args = SimpleNamespace(**vars(args))
        run_args.teacher = "attention"
        run_args.attention_lambda = attention_lambda
        run_args.sectors = ",".join(sectors)
        run_args.seeds = ",".join(str(seed) for seed in seeds)
        run_args.out_dir = args.out_dir / f"lambda_{format_lambda(attention_lambda)}"
        validate_args(run_args)
        for seed in seeds:
            for sector in sectors:
                rows.extend(run_one(run_args, seed=seed, sector=sector, device=device, dtype=dtype))

    write_outputs(rows, args, lambdas, sectors, seeds)
    print_summary(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="recency_weak_jet")
    parser.add_argument("--attention-lambdas", default="4,8,16,24")
    parser.add_argument("--train-len", type=int, default=96)
    parser.add_argument("--train-seq-lens", default="")
    parser.add_argument("--train-length-sampling", choices=["cycle", "random"], default="cycle")
    parser.add_argument("--eval-lens", default="96,192")
    parser.add_argument("--omega-cycles", type=float, default=17.0)
    parser.add_argument("--sectors", default="none,affine,fj_affine,full")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--embed-dim", type=int, default=96)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--max-order", type=int, default=3)
    parser.add_argument("--mlp-ratio", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=220)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--init-affine-slope", type=float, default=6.4)
    parser.add_argument("--gate-init", choices=["uniform", "auto"], default="uniform")
    parser.add_argument("--order-init", choices=["uniform", "auto"], default="uniform")
    parser.add_argument("--freeze-qk", action="store_true")
    parser.add_argument("--median-threshold", action="store_true")
    parser.add_argument("--eval-batches", type=int, default=6)
    parser.add_argument("--log-every", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--float64", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=Path("runs/phase_c_attention_support_sweep"))
    return parser.parse_args()


def write_outputs(
    rows: list[dict[str, float | int | str]],
    args: argparse.Namespace,
    lambdas: list[float],
    sectors: list[str],
    seeds: list[int],
) -> None:
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
                "attention_lambdas": lambdas,
                "sectors": sectors,
                "seeds": seeds,
                "rows": rows,
                "summary": summarize(rows),
            },
            file,
            indent=2,
        )
    print(f"Wrote CSV: {csv_path}")
    print(f"Wrote summary: {json_path}")


def summarize(rows: list[dict[str, float | int | str]]) -> dict[str, dict[str, float | int]]:
    summary = {}
    keys = sorted(
        set((float(row["attention_lambda"]), str(row["sector"]), int(row["eval_len"])) for row in rows)
    )
    for attention_lambda, sector, eval_len in keys:
        subset = [
            row
            for row in rows
            if float(row["attention_lambda"]) == attention_lambda
            and row["sector"] == sector
            and int(row["eval_len"]) == eval_len
        ]
        name = f"lambda={attention_lambda:g}/{sector}@{eval_len}"
        summary[name] = {
            "n": len(subset),
            "accuracy_mean": mean_row(subset, "accuracy"),
            "accuracy_std": std_row(subset, "accuracy"),
            "loss_mean": mean_row(subset, "loss"),
            "label_balance_mean": mean_row(subset, "label_balance"),
            "teacher_effective_support_mean": mean_row(subset, "teacher_effective_support"),
            "gate_fj_mean": mean_row(subset, "gate_fj"),
            "gate_affine_mean": mean_row(subset, "gate_affine"),
            "gate_lc_mean": mean_row(subset, "gate_lc"),
        }
    return summary


def print_summary(rows: list[dict[str, float | int | str]]) -> None:
    print("Phase C attention support sweep")
    print("lambda  support  sector      eval_len  n  acc_mean  balance  gates(fj/a/lc)")
    print("-" * 88)
    keys = sorted(set((float(row["attention_lambda"]), int(row["eval_len"])) for row in rows))
    for attention_lambda, eval_len in keys:
        sectors = sorted({str(row["sector"]) for row in rows if float(row["attention_lambda"]) == attention_lambda})
        for sector in sectors:
            subset = [
                row
                for row in rows
                if float(row["attention_lambda"]) == attention_lambda
                and int(row["eval_len"]) == eval_len
                and row["sector"] == sector
            ]
            print(
                f"{attention_lambda:<7g} {mean_row(subset, 'teacher_effective_support'):<8.1f} "
                f"{sector:<10} {eval_len:<8d} {len(subset):<2d} "
                f"{mean_row(subset, 'accuracy'):.4f}    {mean_row(subset, 'label_balance'):.3f}    "
                f"{gate_string(subset)}"
            )


def mean_row(rows: list[dict[str, float | int | str]], key: str) -> float:
    values = numeric_values(rows, key)
    return statistics.fmean(values) if values else 0.0


def std_row(rows: list[dict[str, float | int | str]], key: str) -> float:
    values = numeric_values(rows, key)
    return statistics.stdev(values) if len(values) > 1 else 0.0


def numeric_values(rows: list[dict[str, float | int | str]], key: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(key, "")
        if value == "":
            continue
        values.append(float(value))
    return values


def gate_string(rows: list[dict[str, float | int | str]]) -> str:
    if not rows or "gate_fj" not in rows[0]:
        return "n/a"
    return "{:.2f}/{:.2f}/{:.2f}".format(
        mean_row(rows, "gate_fj"),
        mean_row(rows, "gate_affine"),
        mean_row(rows, "gate_lc"),
    )


def parse_float_list(raw: str | float) -> list[float]:
    return [float(item.strip()) for item in str(raw).split(",") if item.strip()]


def format_lambda(value: float) -> str:
    return f"{value:g}".replace(".", "p")


if __name__ == "__main__":
    main()
