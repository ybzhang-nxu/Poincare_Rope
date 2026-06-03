"""Export plot-ready Phase B tables from completed sweep outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


ORDER_KEYS = ("mass", "energy", "loo_delta", "fj_mass", "lc_mass")
GATE_KEYS = ("gate_fj", "gate_affine", "gate_lc")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    fixed_rows = export_fixed_tables(args.fixed_summary, args.fixed_results, args.out_dir)
    adaptive_rows = export_adaptive_tables(args.adaptive_summary, args.out_dir)
    multifreq_rows = export_multifreq_tables(args.adaptive_multifreq_summary, args.out_dir)
    write_markdown_summary(fixed_rows, adaptive_rows, multifreq_rows, args.out_dir)

    print(f"Wrote Phase B tables to: {args.out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed-summary", type=Path, default=Path("runs/fixed_kernel_phaseb_sweep/summary.json"))
    parser.add_argument("--fixed-results", type=Path, default=Path("runs/fixed_kernel_phaseb_sweep/results.csv"))
    parser.add_argument("--adaptive-summary", type=Path, default=Path("runs/adaptive_spectrum_phaseb_3seed/summary.json"))
    parser.add_argument(
        "--adaptive-multifreq-summary",
        type=Path,
        default=Path("runs/adaptive_spectrum_phaseb_multifreq_probe/summary.json"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("runs/phase_b_tables"))
    return parser.parse_args()


def export_fixed_tables(summary_path: Path, results_path: Path, out_dir: Path) -> list[dict]:
    summary = read_json(summary_path)
    fixed_best = []
    for omega_cycles, targets in summary["best_by_frequency_and_target"].items():
        for target, row in targets.items():
            out = dict(row)
            out["omega_cycles"] = float(omega_cycles)
            out["target"] = target
            out["log10_mse"] = safe_log10(float(row["mse"]))
            out["log10_condition_number"] = safe_log10(float(row["condition_number"]))
            out["log10_extrapolation_ratio"] = safe_log10(float(row["extrapolation_ratio"]))
            fixed_best.append(out)
    fixed_best.sort(key=lambda row: (row["omega_cycles"], row["target"]))
    write_csv(out_dir / "fixed_best_by_target.csv", fixed_best)

    all_rows = read_csv(results_path)
    final_len = int(summary["final_eval_len"])
    heatmap_rows = []
    for row in all_rows:
        if int(row["eval_len"]) != final_len:
            continue
        out = {
            "omega_cycles": float(row["omega_cycles"]),
            "target": row["target"],
            "basis": row["basis"],
            "eval_len": final_len,
            "mse": float(row["mse"]),
            "r2": float(row["r2"]),
            "condition_number": float(row["condition_number"]),
            "log10_mse": safe_log10(float(row["mse"])),
            "log10_condition_number": safe_log10(float(row["condition_number"])),
        }
        heatmap_rows.append(out)
    heatmap_rows.sort(key=lambda row: (row["omega_cycles"], row["target"], row["basis"]))
    write_csv(out_dir / "fixed_heatmap_long.csv", heatmap_rows)
    return fixed_best


def export_adaptive_tables(summary_path: Path, out_dir: Path) -> list[dict]:
    summary = read_json(summary_path)
    rows = []
    gate_rows = []
    order_rows = []
    teacher_order_rows = []

    for omega_cycles, targets in summary["summary"].items():
        for target, stats in targets.items():
            gate_row = {
                "omega_cycles": float(omega_cycles),
                "target": target,
                "n": int(stats["n"]),
                "train_mse_mean": stats["train_mse_mean"],
                "train_mse_std": stats["train_mse_std"],
                "eval_mse_mean": stats["eval_mse_mean"],
                "eval_mse_std": stats["eval_mse_std"],
                "train_r2_mean": stats["train_r2_mean"],
                "eval_r2_mean": stats["eval_r2_mean"],
                "top_order_mode": int(stats["top_order_mode"]),
                "gate_fj_mean": stats["gate_fj_mean"],
                "gate_fj_std": stats["gate_fj_std"],
                "gate_affine_mean": stats["gate_affine_mean"],
                "gate_affine_std": stats["gate_affine_std"],
                "gate_lc_mean": stats["gate_lc_mean"],
                "gate_lc_std": stats["gate_lc_std"],
            }
            gate_rows.append(gate_row)
            rows.append(gate_row)

            for gate in GATE_KEYS:
                teacher_order_rows.append(
                    {
                        "omega_cycles": float(omega_cycles),
                        "target": target,
                        "kind": "gate",
                        "component": gate.removeprefix("gate_"),
                        "mean": stats[f"{gate}_mean"],
                        "std": stats[f"{gate}_std"],
                    }
                )

            for order in detected_orders(stats):
                order_row = {
                    "omega_cycles": float(omega_cycles),
                    "target": target,
                    "order": order,
                    "mass_mean": stats.get(f"mass_r{order}_mean", 0.0),
                    "mass_std": stats.get(f"mass_r{order}_std", 0.0),
                    "energy_mean": stats.get(f"energy_r{order}_mean", 0.0),
                    "energy_std": stats.get(f"energy_r{order}_std", 0.0),
                    "loo_delta_mean": stats.get(f"loo_delta_r{order}_mean", 0.0),
                    "loo_delta_std": stats.get(f"loo_delta_r{order}_std", 0.0),
                    "fj_mass_mean": stats.get(f"fj_mass_r{order}_mean", 0.0),
                    "fj_mass_std": stats.get(f"fj_mass_r{order}_std", 0.0),
                    "lc_mass_mean": stats.get(f"lc_mass_r{order}_mean", 0.0),
                    "lc_mass_std": stats.get(f"lc_mass_r{order}_std", 0.0),
                }
                order_rows.append(order_row)
                for key in ORDER_KEYS:
                    teacher_order_rows.append(
                        {
                            "omega_cycles": float(omega_cycles),
                            "target": target,
                            "kind": key,
                            "component": f"r{order}",
                            "mean": order_row[f"{key}_mean"],
                            "std": order_row[f"{key}_std"],
                        }
                    )

    gate_rows.sort(key=lambda row: (row["omega_cycles"], row["target"]))
    order_rows.sort(key=lambda row: (row["omega_cycles"], row["target"], row["order"]))
    teacher_order_rows.sort(key=lambda row: (row["omega_cycles"], row["target"], row["kind"], row["component"]))
    write_csv(out_dir / "adaptive_gate_heatmap.csv", gate_rows)
    write_csv(out_dir / "adaptive_order_heatmap.csv", order_rows)
    write_csv(out_dir / "adaptive_long_heatmap.csv", teacher_order_rows)
    return rows


def export_multifreq_tables(summary_path: Path, out_dir: Path) -> list[dict]:
    if not summary_path.exists():
        return []
    summary = read_json(summary_path)
    rows = []
    for omega_cycles, targets in summary["summary"].items():
        for target, stats in targets.items():
            rows.append(
                {
                    "omega_cycles": float(omega_cycles),
                    "target": target,
                    "eval_mse_mean": stats["eval_mse_mean"],
                    "gate_fj_mean": stats["gate_fj_mean"],
                    "gate_affine_mean": stats["gate_affine_mean"],
                    "gate_lc_mean": stats["gate_lc_mean"],
                    "top_order_mode": int(stats["top_order_mode"]),
                }
            )
    rows.sort(key=lambda row: (row["omega_cycles"], row["target"]))
    write_csv(out_dir / "adaptive_multifreq_probe.csv", rows)
    return rows


def write_markdown_summary(
    fixed_rows: list[dict],
    adaptive_rows: list[dict],
    multifreq_rows: list[dict],
    out_dir: Path,
) -> None:
    lines = [
        "# Phase B Plot Tables",
        "",
        "Generated from completed sweep outputs.",
        "",
        "## Files",
        "",
        "- `fixed_best_by_target.csv`: best fixed basis per frequency/target.",
        "- `fixed_heatmap_long.csv`: all fixed basis MSE/R2/condition rows at final eval length.",
        "- `adaptive_gate_heatmap.csv`: sector gates and fit metrics per teacher.",
        "- `adaptive_order_heatmap.csv`: mass/energy/leave-one-out rows per teacher/order.",
        "- `adaptive_long_heatmap.csv`: long-form gate/order values for heatmap libraries.",
        "- `adaptive_multifreq_probe.csv`: one-seed multi-frequency adaptive sanity check.",
        "",
        "## Adaptive Main Table",
        "",
        "| Target | Eval MSE | Gates FJ/A/LC | Top order |",
        "|---|---:|---|---:|",
    ]
    for row in sorted(adaptive_rows, key=lambda r: r["target"]):
        lines.append(
            "| `{target}` | `{eval_mse:.3e}` | `{fj:.2f}/{aff:.2f}/{lc:.2f}` | `{order}` |".format(
                target=row["target"],
                eval_mse=float(row["eval_mse_mean"]),
                fj=float(row["gate_fj_mean"]),
                aff=float(row["gate_affine_mean"]),
                lc=float(row["gate_lc_mean"]),
                order=int(row["top_order_mode"]),
            )
        )
    lines.extend(
        [
            "",
            "## Fixed Kernel Best Basis Snapshot",
            "",
            "| Omega | Target | Best basis | R2 | log10 MSE | log10 cond |",
            "|---:|---|---|---:|---:|---:|",
        ]
    )
    for row in fixed_rows:
        lines.append(
            "| `{omega:g}` | `{target}` | `{basis}` | `{r2:.5f}` | `{log_mse:.2f}` | `{log_cond:.2f}` |".format(
                omega=float(row["omega_cycles"]),
                target=row["target"],
                basis=row["basis"],
                r2=float(row["r2"]),
                log_mse=float(row["log10_mse"]),
                log_cond=float(row["log10_condition_number"]),
            )
        )
    if multifreq_rows:
        lines.extend(
            [
                "",
                "## Adaptive Multi-Frequency Probe",
                "",
                "| Omega | Target | Gates FJ/A/LC | Top order |",
                "|---:|---|---|---:|",
            ]
        )
        for row in multifreq_rows:
            lines.append(
                "| `{omega:g}` | `{target}` | `{fj:.2f}/{aff:.2f}/{lc:.2f}` | `{order}` |".format(
                    omega=float(row["omega_cycles"]),
                    target=row["target"],
                    fj=float(row["gate_fj_mean"]),
                    aff=float(row["gate_affine_mean"]),
                    lc=float(row["gate_lc_mean"]),
                    order=int(row["top_order_mode"]),
                )
            )
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def detected_orders(stats: dict) -> list[int]:
    orders = set()
    for key in stats:
        for prefix in ("mass_r", "energy_r", "loo_delta_r", "fj_mass_r", "lc_mass_r"):
            if key.startswith(prefix) and key.endswith("_mean"):
                orders.add(int(key.removeprefix(prefix).removesuffix("_mean")))
    return sorted(orders)


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_log10(value: float) -> float:
    return math.log10(max(value, 1e-300))


if __name__ == "__main__":
    main()
