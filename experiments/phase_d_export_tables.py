"""Export plot-ready Phase D tables from byte-level LM runs."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path


MAIN_RUNS = (
    (
        "fallback_minirun_128",
        "fallback",
        "smoke",
        Path("runs/byte_lm_phase_d_minirun/results.csv"),
    ),
    (
        "tinyshakespeare_core_512",
        "tiny_shakespeare",
        "core_512",
        Path("runs/byte_lm_tinyshakespeare_phase_d_core_512/results.csv"),
    ),
    (
        "tinyshakespeare_core_1024",
        "tiny_shakespeare",
        "core_1024",
        Path("runs/byte_lm_tinyshakespeare_phase_d_core_1024/results.csv"),
    ),
    (
        "wikitext2_core_512",
        "wikitext2",
        "core_512",
        Path("runs/byte_lm_wikitext2_phase_d_core_512/results.csv"),
    ),
    (
        "wikitext2_core_1024",
        "wikitext2",
        "core_1024",
        Path("runs/byte_lm_wikitext2_phase_d_core_1024/results.csv"),
    ),
    (
        "tinyshakespeare_sector_ablation_1024",
        "tiny_shakespeare",
        "sector_ablation_1024",
        Path("runs/byte_lm_tinyshakespeare_phase_d_sector_ablation_1024/results.csv"),
        3,
    ),
    (
        "wikitext2_sector_ablation_1024",
        "wikitext2",
        "sector_ablation_1024",
        Path("runs/byte_lm_wikitext2_phase_d_sector_ablation_1024/results.csv"),
        3,
    ),
    (
        "tinyshakespeare_order_ablation_o0_1024",
        "tiny_shakespeare",
        "order_ablation_1024",
        Path("runs/byte_lm_tinyshakespeare_phase_d_order_ablation_o0_1024/results.csv"),
    ),
    (
        "tinyshakespeare_order_ablation_o1_1024",
        "tiny_shakespeare",
        "order_ablation_1024",
        Path("runs/byte_lm_tinyshakespeare_phase_d_order_ablation_o1_1024/results.csv"),
    ),
    (
        "wikitext2_order_ablation_o0_1024",
        "wikitext2",
        "order_ablation_1024",
        Path("runs/byte_lm_wikitext2_phase_d_order_ablation_o0_1024/results.csv"),
    ),
    (
        "wikitext2_order_ablation_o1_1024",
        "wikitext2",
        "order_ablation_1024",
        Path("runs/byte_lm_wikitext2_phase_d_order_ablation_o1_1024/results.csv"),
    ),
    (
        "wikitext2_stress_16384",
        "wikitext2",
        "stress_16384",
        Path("runs/byte_lm_wikitext2_phase_d_stress_16384/results.csv"),
    ),
    (
        "tinyshakespeare_stress_16384",
        "tiny_shakespeare",
        "stress_16384",
        Path("runs/byte_lm_tinyshakespeare_phase_d_stress_16384/results.csv"),
    ),
    (
        "tinyshakespeare_stress_32768",
        "tiny_shakespeare",
        "stress_32768_sampled",
        Path("runs/byte_lm_tinyshakespeare_phase_d_stress_32768/results.csv"),
    ),
    (
        "wikitext2_stress_32768",
        "wikitext2",
        "stress_32768_sampled",
        Path("runs/byte_lm_wikitext2_phase_d_stress_32768/results.csv"),
    ),
    (
        "gutenberg_warpeace_stress_32768",
        "gutenberg_warpeace",
        "stress_32768_sampled",
        Path("runs/byte_lm_gutenberg_warpeace_phase_d_stress_32768/results.csv"),
    ),
)

METRIC_KEYS = (
    "loss",
    "ppl",
    "delta_loss",
    "attention_entropy",
    "attention_effective_support",
    "average_attention_distance",
    "logit_std",
    "logit_abs_max",
    "gate_fj",
    "gate_affine",
    "gate_lc",
    "affine_slope_mean",
    "fj_mass_r0",
    "fj_mass_r1",
    "fj_mass_r2",
    "fj_mass_r3",
    "lc_mass_r0",
    "lc_mass_r1",
    "lc_mass_r2",
    "lc_mass_r3",
)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_main_rows(args)
    summary = summarize(rows)
    write_csv(args.out_dir / "phase_d_combined_results.csv", rows)
    write_csv(args.out_dir / "phase_d_summary_by_method.csv", summary)
    write_markdown(rows, summary, args.out_dir)
    print(f"Wrote Phase D tables to: {args.out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("runs/phase_d_tables"))
    parser.add_argument(
        "--include-missing",
        action="store_true",
        help="Skip missing default result files instead of failing.",
    )
    return parser.parse_args()


def load_main_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    rows = []
    for run_id, corpus, protocol, path, *metadata in MAIN_RUNS:
        default_max_order = metadata[0] if metadata else None
        if not path.exists():
            if args.include_missing:
                continue
            raise FileNotFoundError(path)
        with path.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                out = dict(row)
                out["run_id"] = run_id
                out["corpus"] = corpus
                out["protocol"] = protocol
                out["source_path"] = str(path)
                if default_max_order is not None and not out.get("max_order"):
                    out["max_order"] = str(default_max_order)
                rows.append(out)
    return rows


def summarize(rows: list[dict[str, str]]) -> list[dict[str, str | int | float]]:
    groups = {}
    for row in rows:
        key = (
            row["run_id"],
            row["corpus"],
            row["protocol"],
            row["method"],
            int(float(row["max_order"])) if row.get("max_order") not in (None, "") else "",
            int(float(row["train_len"])),
            int(float(row["eval_len"])),
        )
        groups.setdefault(key, []).append(row)

    summary = []
    for key in sorted(groups):
        run_id, corpus, protocol, method, max_order, train_len, eval_len = key
        subset = groups[key]
        out: dict[str, str | int | float] = {
            "run_id": run_id,
            "corpus": corpus,
            "protocol": protocol,
            "method": method,
            "max_order": max_order,
            "train_len": train_len,
            "eval_len": eval_len,
            "n": len(subset),
        }
        for metric in METRIC_KEYS:
            values = numeric_values(subset, metric)
            if values:
                out[f"{metric}_mean"] = statistics.fmean(values)
                out[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        summary.append(out)
    return summary


def write_markdown(
    rows: list[dict[str, str]],
    summary: list[dict[str, str | int | float]],
    out_dir: Path,
) -> None:
    path = out_dir / "README.md"
    lines = [
        "# Phase D Tables",
        "",
        f"Combined rows: `{len(rows)}`",
        f"Summary rows: `{len(summary)}`",
        "",
        "Files:",
        "",
        "- `phase_d_combined_results.csv`",
        "- `phase_d_summary_by_method.csv`",
    ]

    snapshots = (
        ("Tiny Shakespeare 512-to-8192", "tinyshakespeare_core_512", (512, 8192)),
        ("Tiny Shakespeare 1024-to-8192", "tinyshakespeare_core_1024", (1024, 8192)),
        ("WikiText-2 512-to-8192", "wikitext2_core_512", (512, 8192)),
        ("WikiText-2 1024-to-8192", "wikitext2_core_1024", (1024, 8192)),
        ("Tiny Shakespeare 1024 sector ablation", "tinyshakespeare_sector_ablation_1024", (1024, 8192)),
        ("WikiText-2 1024 sector ablation", "wikitext2_sector_ablation_1024", (1024, 8192)),
        ("WikiText-2 1024-to-16384 stress", "wikitext2_stress_16384", (1024, 8192, 16384)),
        ("Tiny Shakespeare 1024-to-16384 stress", "tinyshakespeare_stress_16384", (1024, 8192, 16384)),
        ("Tiny Shakespeare 1024-to-32768 sampled-diagnostic stress", "tinyshakespeare_stress_32768", (1024, 16384, 32768)),
        ("WikiText-2 1024-to-32768 sampled-diagnostic stress", "wikitext2_stress_32768", (1024, 16384, 32768)),
        ("Gutenberg War and Peace 1024-to-32768 sampled-diagnostic stress", "gutenberg_warpeace_stress_32768", (1024, 16384, 32768)),
    )
    for title, run_id, eval_lens in snapshots:
        subset = [
            row
            for row in summary
            if row["run_id"] == run_id and int(row["eval_len"]) in eval_lens
        ]
        if not subset:
            continue
        lines.extend(
            [
                "",
                f"{title} snapshot:",
                "",
                "| Method | Eval Len | Loss | Delta Loss | PPL | Gates FJ/A/LC |",
                "|---|---:|---:|---:|---:|---|",
            ]
        )
        for row in sorted(subset, key=lambda item: (str(item["method"]), int(item["eval_len"]))):
            lines.append(
                "| {method} | {eval_len} | {loss:.4f} | {delta:.4f} | {ppl:.2f} | {gates} |".format(
                    method=row["method"],
                    eval_len=row["eval_len"],
                    loss=float(row.get("loss_mean", 0.0)),
                    delta=float(row.get("delta_loss_mean", 0.0)),
                    ppl=float(row.get("ppl_mean", 0.0)),
                    gates=gate_string(row),
                )
            )
    append_order_ablation_snapshot(
        lines,
        summary,
        title="Tiny Shakespeare 1024 order ablation",
        corpus="tiny_shakespeare",
    )
    append_order_ablation_snapshot(
        lines,
        summary,
        title="WikiText-2 1024 order ablation",
        corpus="wikitext2",
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_order_ablation_snapshot(
    lines: list[str],
    summary: list[dict[str, str | int | float]],
    *,
    title: str,
    corpus: str,
) -> None:
    methods = {"fj_affine", "fj_lc", "full", "lc_affine"}
    subset = [
        row
        for row in summary
        if row["corpus"] == corpus
        and row["protocol"] in ("order_ablation_1024", "sector_ablation_1024")
        and row["method"] in methods
        and row.get("max_order") not in (None, "")
        and int(row["eval_len"]) in (1024, 8192)
    ]
    if not subset:
        return
    lines.extend(
        [
            "",
            f"{title} snapshot:",
            "",
            "| Method | Max Order | Eval Len | Loss | Delta Loss | PPL | AvgDist | Gates FJ/A/LC |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in sorted(
        subset,
        key=lambda item: (str(item["method"]), int(float(item["max_order"])), int(item["eval_len"])),
    ):
        lines.append(
            "| {method} | {max_order} | {eval_len} | {loss:.4f} | {delta:.4f} | {ppl:.2f} | {dist:.2f} | {gates} |".format(
                method=row["method"],
                max_order=int(float(row["max_order"])),
                eval_len=row["eval_len"],
                loss=float(row.get("loss_mean", 0.0)),
                delta=float(row.get("delta_loss_mean", 0.0)),
                ppl=float(row.get("ppl_mean", 0.0)),
                dist=float(row.get("average_attention_distance_mean", 0.0)),
                gates=gate_string(row),
            )
        )


def gate_string(row: dict[str, str | int | float]) -> str:
    if "gate_affine_mean" not in row:
        return "n/a"
    return "{:.2f}/{:.2f}/{:.2f}".format(
        float(row.get("gate_fj_mean", 0.0)),
        float(row.get("gate_affine_mean", 0.0)),
        float(row.get("gate_lc_mean", 0.0)),
    )


def numeric_values(rows: list[dict[str, str]], key: str) -> list[float]:
    values = []
    for row in rows:
        raw = row.get(key, "")
        if raw == "":
            continue
        value = float(raw)
        if math.isnan(value):
            continue
        values.append(value)
    return values


def write_csv(path: Path, rows: list[dict[str, str | int | float]]) -> None:
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
