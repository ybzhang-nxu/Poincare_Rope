"""Export appendix-facing GRAPE comparison tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


BASIS_ORDER = [
    ("grape_m_rope", "GRAPE-M/RoPE"),
    ("grape_a_alibi", "GRAPE-A/ALiBi"),
    ("grape_ma_rope_alibi", "GRAPE-M+A"),
    ("pj_fj_R1", "PJ-FJ R1"),
    ("pj_fj_R2", "PJ-FJ R2"),
    ("pj_lc_R3", "PJ-LC"),
]

TARGET_ORDER = ["phase", "affine", "first_jet", "second_jet", "lc_core"]

EXACTNESS_ROWS = [
    {
        "method": "GRAPE-M special case",
        "exact_relative_law": "yes",
        "norm_preserving": "yes",
        "additive_recency": "no",
        "fourier_jet_primitive": "no",
        "lc_bounded": "no",
    },
    {
        "method": "GRAPE-A special case",
        "exact_relative_law": "yes",
        "norm_preserving": "n/a",
        "additive_recency": "yes",
        "fourier_jet_primitive": "no",
        "lc_bounded": "no",
    },
    {
        "method": "GRAPE-M+A special case",
        "exact_relative_law": "yes, separate",
        "norm_preserving": "partial",
        "additive_recency": "yes",
        "fourier_jet_primitive": "no",
        "lc_bounded": "no",
    },
    {
        "method": "PJ-rotary exact",
        "exact_relative_law": "yes",
        "norm_preserving": "no generally",
        "additive_recency": "no",
        "fourier_jet_primitive": "yes",
        "lc_bounded": "no",
    },
    {
        "method": "PJ-bias",
        "exact_relative_law": "scalar kernel",
        "norm_preserving": "n/a",
        "additive_recency": "yes",
        "fourier_jet_primitive": "yes",
        "lc_bounded": "optional",
    },
    {
        "method": "LC-PJ bias",
        "exact_relative_law": "scalar kernel",
        "norm_preserving": "n/a",
        "additive_recency": "yes",
        "fourier_jet_primitive": "compactified jets",
        "lc_bounded": "yes",
    },
]


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv(args.fixed_results)
    if not rows:
        raise SystemExit(f"no rows found in {args.fixed_results}")

    summary = summarize_fixed(rows)
    write_csv(args.out_dir / "fixed_projection_summary.csv", summary)
    write_csv(args.out_dir / "exactness_sanity.csv", EXACTNESS_ROWS)
    write_fixed_tex(args.out_dir / "fixed_projection_summary_table.tex", summary)
    write_exactness_tex(args.out_dir / "exactness_sanity_table.tex", EXACTNESS_ROWS)
    write_readme(args.out_dir, summary)
    write_json(args.out_dir / "summary.json", {"fixed_rows": len(rows), "summary_rows": len(summary)})
    print(f"Wrote GRAPE appendix tables to: {args.out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed-results", type=Path, default=Path("runs/grape_appendix_fixed/results.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("runs/grape_appendix_tables"))
    return parser.parse_args()


def summarize_fixed(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    final_eval_len = max(int(float(row["eval_len"])) for row in rows)
    by_key = {(row["target"], row["basis"]): row for row in rows if int(float(row["eval_len"])) == final_eval_len}
    output = []
    for target in TARGET_ORDER:
        target_rows = [row for row in rows if row["target"] == target]
        if not target_rows:
            continue
        target_label = target_rows[0]["target_label"]
        out = {
            "target": target,
            "target_label": target_label,
            "eval_len": str(final_eval_len),
        }
        for basis, label in BASIS_ORDER:
            row = by_key.get((target, basis))
            out[f"{basis}_label"] = label
            out[f"{basis}_r2"] = row["r2"] if row else ""
            out[f"{basis}_mse"] = row["mse"] if row else ""
        output.append(out)
    return output


def write_fixed_tex(path: Path, rows: list[dict[str, str]]) -> None:
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{GRAPE special-case and PJ primitive-basis projections.  All rows use the same frequency grid; PJ-FJ variants add jet orders at that frequency.  The table reports primitive containment through fixed projection.  Values are \(R^2\) at the longest evaluation length; em dashes mark failed extrapolations with \(R^2<-10\).}",
        r"\label{tab:grape-fixed-projection}",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Target & GRAPE-M/RoPE & GRAPE-A/ALiBi & GRAPE-M+A & PJ-FJ R1 & PJ-FJ R2 & PJ-LC \\",
        r"\midrule",
    ]
    for row in rows:
        values = [format_r2(row[f"{basis}_r2"]) for basis, _ in BASIS_ORDER]
        lines.append(f"{row['target_label']} & " + " & ".join(values) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_exactness_tex(path: Path, rows: list[dict[str, str]]) -> None:
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Exactness and primitive-mode sanity table.  GRAPE special-case rows cover exact group-action laws; PJ rows identify the Fourier-jet and LC axes used in this paper.}",
        r"\label{tab:grape-exactness}",
        r"\small",
        r"\begin{tabularx}{\linewidth}{>{\raggedright\arraybackslash}Xccccc}",
        r"\toprule",
        r"Method & Exact relative law & Norm preserving & Additive recency & \(d^r e^{i\omega d}\) & LC bounded \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            "{method} & {exact} & {norm} & {affine} & {jet} & {lc} \\\\".format(
                method=escape_tex(row["method"]),
                exact=escape_tex(row["exact_relative_law"]),
                norm=escape_tex(row["norm_preserving"]),
                affine=escape_tex(row["additive_recency"]),
                jet=escape_tex(row["fourier_jet_primitive"]),
                lc=escape_tex(row["lc_bounded"]),
            )
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabularx}",
            r"\end{table}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_readme(out_dir: Path, summary: list[dict[str, str]]) -> None:
    lines = [
        "# GRAPE Appendix Tables",
        "",
        "These tables support the GRAPE comparison appendix.  The GRAPE rows are",
        "exact special-case controls: GRAPE-M/RoPE, GRAPE-A/ALiBi, and",
        "GRAPE-M+A/RoPE+ALiBi.  The rows report restricted controls for the",
        "appendix comparison.",
        "",
        "Files:",
        "",
        "- `fixed_projection_summary.csv`",
        "- `fixed_projection_summary_table.tex`",
        "- `exactness_sanity.csv`",
        "- `exactness_sanity_table.tex`",
        "- `summary.json`",
        "",
        "## Fixed Projection R2",
        "",
        "| Target | GRAPE-M/RoPE | GRAPE-A/ALiBi | GRAPE-M+A | PJ-FJ R1 | PJ-FJ R2 | PJ-LC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        values = [format_r2(row[f"{basis}_r2"]) for basis, _ in BASIS_ORDER]
        lines.append(f"| {row['target_label']} | " + " | ".join(values) + " |")
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def format_r2(value: str) -> str:
    if value == "":
        return "--"
    number = float(value)
    if number > 0.99995:
        return "1.000"
    if number < -10:
        return r"\textemdash"
    return f"{number:.3f}"


def escape_tex(value: str) -> str:
    return value.replace("_", r"\_")


if __name__ == "__main__":
    main()
