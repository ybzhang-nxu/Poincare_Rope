"""Export plot-ready Phase C tables from synthetic query-LM runs."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path


MAIN_RUNS = (
    (
        "attention_affine_3seed",
        "attention_affine_single_len",
        Path("runs/synthetic_query_lm_attention_affine_3seed_phasec/results.csv"),
    ),
    (
        "attention_recency_support_sweep",
        "attention_support_sweep",
        Path("runs/phase_c_attention_recency_support_sweep/results.csv"),
    ),
    (
        "signed_firstjet_singlelen_3seed",
        "signed_single_len",
        Path("runs/synthetic_query_lm_signed_firstjet_3seed_phasec/results.csv"),
    ),
    (
        "signed_firstjet_multilen_3seed",
        "signed_multi_len",
        Path("runs/synthetic_query_lm_signed_firstjet_multilen_3seed_phasec/results.csv"),
    ),
    (
        "signed_firstjet_full_auto_multilen_3seed",
        "signed_full_auto_multi_len",
        Path("runs/synthetic_query_lm_signed_firstjet_full_auto_multilen_3seed_phasec/results.csv"),
    ),
    (
        "signed_secondjet_multilen_3seed",
        "signed_multi_len",
        Path("runs/synthetic_query_lm_signed_secondjet_multilen_3seed_phasec/results.csv"),
    ),
    (
        "signed_secondjet_full_auto_multilen_3seed",
        "signed_full_auto_multi_len",
        Path("runs/synthetic_query_lm_signed_secondjet_full_auto_multilen_3seed_phasec/results.csv"),
    ),
    (
        "signed_thirdjet_multilen_3seed",
        "signed_multi_len",
        Path("runs/synthetic_query_lm_signed_thirdjet_multilen_3seed_phasec/results.csv"),
    ),
    (
        "signed_thirdjet_full_auto_multilen_3seed",
        "signed_full_auto_multi_len",
        Path("runs/synthetic_query_lm_signed_thirdjet_full_auto_multilen_3seed_phasec/results.csv"),
    ),
    (
        "signed_lccore_multilen_3seed",
        "signed_lc_multi_len",
        Path("runs/synthetic_query_lm_signed_lccore_multilen_3seed_phasec/results.csv"),
    ),
)

METRIC_KEYS = (
    "accuracy",
    "loss",
    "label_balance",
    "mean_confidence",
    "attention_entropy",
    "attention_effective_support",
    "teacher_entropy",
    "teacher_effective_support",
    "teacher_kernel_norm",
    "gate_fj",
    "gate_affine",
    "gate_lc",
    "affine_slope_mean",
)
MASS_PATTERN = re.compile(r"^(fj|lc)_mass_r(\d+)$")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_main_rows(args)
    summary = summarize(rows)
    mass_rows = summarize_order_masses(rows)
    full_auto_rows = [row for row in summary if row["protocol"] == "signed_full_auto_multi_len"]

    write_csv(args.out_dir / "phase_c_combined_results.csv", rows)
    write_csv(args.out_dir / "phase_c_summary_by_sector.csv", summary)
    write_csv(args.out_dir / "phase_c_order_mass.csv", mass_rows)
    write_csv(args.out_dir / "phase_c_full_auto_summary.csv", full_auto_rows)
    write_markdown(rows, summary, mass_rows, args.out_dir)

    print(f"Wrote Phase C tables to: {args.out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("runs/phase_c_tables"))
    parser.add_argument(
        "--include-missing",
        action="store_true",
        help="Skip missing default result files instead of failing.",
    )
    return parser.parse_args()


def load_main_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    rows = []
    for run_id, protocol, path in MAIN_RUNS:
        if not path.exists():
            if args.include_missing:
                continue
            raise FileNotFoundError(path)
        with path.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                out = dict(row)
                out["run_id"] = run_id
                out["protocol"] = protocol
                out["source_path"] = str(path)
                if not out.get("train_seq_lens"):
                    out["train_seq_lens"] = out.get("train_len", "")
                rows.append(out)
    return rows


def summarize(rows: list[dict[str, str]]) -> list[dict[str, str | int | float]]:
    groups = {}
    for row in rows:
        key = (
            row["run_id"],
            row["protocol"],
            row["teacher"],
            row["target"],
            row["sector"],
            int(float(row["eval_len"])),
            row.get("train_seq_lens", ""),
            row.get("attention_lambda", ""),
        )
        groups.setdefault(key, []).append(row)

    summary = []
    for key in sorted(groups):
        run_id, protocol, teacher, target, sector, eval_len, train_seq_lens, attention_lambda = key
        subset = groups[key]
        out: dict[str, str | int | float] = {
            "run_id": run_id,
            "protocol": protocol,
            "teacher": teacher,
            "target": target,
            "sector": sector,
            "eval_len": eval_len,
            "train_seq_lens": train_seq_lens,
            "attention_lambda": attention_lambda,
            "n": len(subset),
        }
        for metric in METRIC_KEYS:
            values = numeric_values(subset, metric)
            if values:
                out[f"{metric}_mean"] = statistics.fmean(values)
                out[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        top_order = dominant_mass_order(subset, "fj")
        if top_order is not None:
            out["top_fj_mass_order"] = top_order
        summary.append(out)
    return summary


def summarize_order_masses(rows: list[dict[str, str]]) -> list[dict[str, str | int | float]]:
    groups = {}
    for row in rows:
        for key, value in row.items():
            match = MASS_PATTERN.match(key)
            if match is None or value == "":
                continue
            component, order_raw = match.groups()
            group_key = (
                row["run_id"],
                row["protocol"],
                row["teacher"],
                row["target"],
                row["sector"],
                int(float(row["eval_len"])),
                row.get("train_seq_lens", ""),
                row.get("attention_lambda", ""),
                component,
                int(order_raw),
            )
            groups.setdefault(group_key, []).append(float(value))

    out = []
    for key in sorted(groups):
        run_id, protocol, teacher, target, sector, eval_len, train_seq_lens, attention_lambda, component, order = key
        values = groups[key]
        out.append(
            {
                "run_id": run_id,
                "protocol": protocol,
                "teacher": teacher,
                "target": target,
                "sector": sector,
                "eval_len": eval_len,
                "train_seq_lens": train_seq_lens,
                "attention_lambda": attention_lambda,
                "component": component,
                "order": order,
                "mass_mean": statistics.fmean(values),
                "mass_std": statistics.stdev(values) if len(values) > 1 else 0.0,
                "n": len(values),
            }
        )
    return out


def dominant_mass_order(rows: list[dict[str, str]], component: str) -> int | None:
    means = {}
    for row in rows:
        for key, value in row.items():
            match = MASS_PATTERN.match(key)
            if match is None or value == "":
                continue
            matched_component, order_raw = match.groups()
            if matched_component != component:
                continue
            means.setdefault(int(order_raw), []).append(float(value))
    if not means:
        return None
    order_means = {order: statistics.fmean(values) for order, values in means.items()}
    return max(order_means, key=order_means.__getitem__)


def write_markdown(
    rows: list[dict[str, str]],
    summary: list[dict[str, str | int | float]],
    mass_rows: list[dict[str, str | int | float]],
    out_dir: Path,
) -> None:
    lines = [
        "# Phase C Plot Tables",
        "",
        "Generated from main synthetic query-LM result CSVs.",
        "",
        "## Files",
        "",
        "- `phase_c_combined_results.csv`: seed-level rows from selected Phase C runs.",
        "- `phase_c_summary_by_sector.csv`: mean/std metrics grouped by run, sector, and eval length.",
        "- `phase_c_order_mass.csv`: long-form FJ/LC order masses.",
        "- `phase_c_full_auto_summary.csv`: compact full-PJ auto-initialized rows.",
        "",
        "## Accuracy Summary",
        "",
        "| Run | Target | Sector | Lambda | Eval length | Accuracy | Gates FJ/A/LC |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for row in summary:
        lines.append(
            "| `{run}` | `{target}` | `{sector}` | {lambda_label} | `{eval_len}` | `{acc:.3f}` | {gates} |".format(
                run=row["run_id"],
                target=row["target"],
                sector=row["sector"],
                lambda_label=format_lambda(row.get("attention_lambda", "")),
                eval_len=int(row["eval_len"]),
                acc=float(row.get("accuracy_mean", 0.0)),
                gates=format_gates(row),
            )
        )

    full_mass = [
        row
        for row in mass_rows
        if row["protocol"] == "signed_full_auto_multi_len" and row["component"] == "fj"
    ]
    if full_mass:
        lines.extend(
            [
                "",
                "## Full Auto FJ Mass",
                "",
                "| Target | Eval length | Order | Mass |",
                "|---|---:|---:|---:|",
            ]
        )
        for row in full_mass:
            lines.append(
                "| `{target}` | `{eval_len}` | `r{order}` | `{mass:.3f}` |".format(
                    target=row["target"],
                    eval_len=int(row["eval_len"]),
                    order=int(row["order"]),
                    mass=float(row["mass_mean"]),
                )
            )

    metadata = {
        "source_rows": len(rows),
        "summary_rows": len(summary),
        "order_mass_rows": len(mass_rows),
    }
    lines.extend(["", "## Metadata", "", "```json", json.dumps(metadata, indent=2), "```"])
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_gates(row: dict[str, str | int | float]) -> str:
    if "gate_fj_mean" not in row:
        return "n/a"
    return "`{:.2f}/{:.2f}/{:.2f}`".format(
        float(row.get("gate_fj_mean", 0.0)),
        float(row.get("gate_affine_mean", 0.0)),
        float(row.get("gate_lc_mean", 0.0)),
    )


def format_lambda(value: str | int | float) -> str:
    if value == "":
        return "n/a"
    return f"`{float(value):g}`"


def numeric_values(rows: list[dict[str, str]], key: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(key, "")
        if value == "":
            continue
        values.append(float(value))
    return values


def write_csv(path: Path, rows: list[dict[str, str | int | float]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    main()
