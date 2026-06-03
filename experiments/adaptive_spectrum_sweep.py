"""Multi-seed adaptive spectrum sweep for Phase B diagnostics."""

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
        "/home/riven/JordanKac/.venv/bin/python experiments/adaptive_spectrum_sweep.py"
    ) from exc

from experiments.adaptive_spectrum_diagnostic import fit_target, make_targets, pick_omega, result_metric, select_targets


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype = torch.float64 if args.float64 else torch.float32
    seeds = parse_int_list(args.seeds)
    omega_cycles_values = parse_float_list(args.omega_cycles)

    rows = []
    for omega_cycles in omega_cycles_values:
        omega = pick_omega(omega_cycles, args.train_len)
        targets = select_targets(make_targets(args.train_len, omega), args.targets)
        for seed in seeds:
            run_args = SimpleNamespace(**vars(args))
            run_args.seed = seed
            for target_name, target_fn in targets.items():
                best = None
                for restart in range(args.restarts):
                    result = fit_target(target_name, target_fn, run_args, restart, device, dtype, omega)
                    if best is None or result_metric(result, args.selection_metric) < result_metric(
                        best, args.selection_metric
                    ):
                        best = result
                if best is None:
                    raise RuntimeError("no diagnostic run completed")
                rows.append(flatten_result(best, seed=seed, omega_cycles=omega_cycles))

    print_summary(rows)
    write_outputs(rows, args, seeds, omega_cycles_values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-len", type=int, default=1024)
    parser.add_argument("--eval-len", type=int, default=4096)
    parser.add_argument("--omega-cycles", default="17")
    parser.add_argument("--targets", default="all")
    parser.add_argument("--max-order", type=int, default=3)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=3e-2)
    parser.add_argument("--restarts", type=int, default=2)
    parser.add_argument("--order-init-cycle", action="store_true")
    parser.add_argument("--selection-metric", choices=["train_mse", "eval_mse"], default="train_mse")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--float64", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=Path("runs/adaptive_spectrum_sweep"))
    return parser.parse_args()


def flatten_result(result, *, seed: int, omega_cycles: float) -> dict[str, float | int | str]:
    row: dict[str, float | int | str] = {
        "seed": seed,
        "omega_cycles": omega_cycles,
        "target": result.target,
        "restart": result.restart,
        "train_mse": result.train_mse,
        "train_r2": result.train_r2,
        "eval_mse": result.eval_mse,
        "eval_r2": result.eval_r2,
        "gate_fj": result.gates["fj"],
        "gate_affine": result.gates["affine"],
        "gate_lc": result.gates["lc"],
        "affine_slope": result.affine_slope,
        "top_order": max(range(len(result.normalized_order_mass)), key=result.normalized_order_mass.__getitem__),
    }
    for idx, value in enumerate(result.fj_mass):
        row[f"fj_mass_r{idx}"] = value
    for idx, value in enumerate(result.lc_mass):
        row[f"lc_mass_r{idx}"] = value
    for idx, value in enumerate(result.normalized_order_mass):
        row[f"mass_r{idx}"] = value
    for idx, value in enumerate(result.functional_energy):
        row[f"energy_r{idx}"] = value
    for idx, value in enumerate(result.leave_one_order_out_delta_mse):
        row[f"loo_delta_r{idx}"] = value
    return row


def print_summary(rows: list[dict[str, float | int | str]]) -> None:
    print("Adaptive spectrum sweep summary")
    print("omega   target              n  train_mse       eval_mse        gates(fj/a/lc)       top_order")
    print("-" * 102)
    for omega_cycles in sorted({float(row["omega_cycles"]) for row in rows}):
        targets = sorted({str(row["target"]) for row in rows if float(row["omega_cycles"]) == omega_cycles})
        for target in targets:
            subset = [row for row in rows if float(row["omega_cycles"]) == omega_cycles and row["target"] == target]
            gate_fj = mean_float(subset, "gate_fj")
            gate_affine = mean_float(subset, "gate_affine")
            gate_lc = mean_float(subset, "gate_lc")
            top_order = mode_int([int(row["top_order"]) for row in subset])
            print(
                f"{omega_cycles:<7g} {target:<19} {len(subset):<2d} "
                f"{mean_float(subset, 'train_mse'):12.4e} "
                f"{mean_float(subset, 'eval_mse'):12.4e} "
                f"{gate_fj:.2f}/{gate_affine:.2f}/{gate_lc:.2f}          r={top_order}"
            )


def write_outputs(
    rows: list[dict[str, float | int | str]],
    args: argparse.Namespace,
    seeds: list[int],
    omega_cycles_values: list[float],
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

    grouped = {}
    for omega_cycles in sorted({float(row["omega_cycles"]) for row in rows}):
        omega_summary = {}
        targets = sorted({str(row["target"]) for row in rows if float(row["omega_cycles"]) == omega_cycles})
        for target in targets:
            subset = [row for row in rows if float(row["omega_cycles"]) == omega_cycles and row["target"] == target]
            omega_summary[target] = summarize_subset(subset)
        grouped[f"{omega_cycles:g}"] = omega_summary

    config = vars(args).copy()
    config["out_dir"] = str(args.out_dir)
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "config": config,
                "seeds": seeds,
                "omega_cycles": omega_cycles_values,
                "summary": grouped,
                "rows": rows,
            },
            file,
            indent=2,
        )

    print(f"Wrote CSV: {csv_path}")
    print(f"Wrote summary: {json_path}")


def summarize_subset(subset: list[dict[str, float | int | str]]) -> dict[str, float | int]:
    numeric_keys = [
        key
        for key in subset[0].keys()
        if key not in ("seed", "omega_cycles", "target", "restart", "top_order")
        and isinstance(subset[0][key], (float, int))
    ]
    summary: dict[str, float | int] = {
        "n": len(subset),
        "top_order_mode": mode_int([int(row["top_order"]) for row in subset]),
    }
    for key in numeric_keys:
        values = [float(row[key]) for row in subset]
        summary[f"{key}_mean"] = statistics.fmean(values)
        summary[f"{key}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
    return summary


def mean_float(rows: list[dict[str, float | int | str]], key: str) -> float:
    return statistics.fmean(float(row[key]) for row in rows)


def mode_int(values: list[int]) -> int:
    return max(sorted(set(values)), key=values.count)


def parse_int_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in str(raw).split(",") if item.strip()]


def parse_float_list(raw: str | float) -> list[float]:
    return [float(item.strip()) for item in str(raw).split(",") if item.strip()]


if __name__ == "__main__":
    main()
