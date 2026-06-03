"""Export cross-task GRAPE special-case comparison summaries."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


DEFAULT_RUNS = (
    ("synthetic_grape_controls", "synthetic", Path("runs/grape_appendix_synthetic/results.csv")),
    ("language_tiny_grape_controls", "language", Path("runs/grape_rerun_language_tiny/results.csv")),
    ("language_wikitext_grape_controls", "language", Path("runs/grape_rerun_language_wikitext/results.csv")),
    ("music_motif_grape_controls", "music", Path("runs/grape_rerun_music_motif/results.csv")),
    ("maestro_grape_controls", "music", Path("runs/grape_rerun_maestro/results.csv")),
    ("musicnet_grape_controls", "music", Path("runs/grape_rerun_musicnet/results.csv")),
)


METHOD_LABELS = {
    "grape_m_rope": "GRAPE-M/RoPE",
    "grape_a_alibi": "GRAPE-A/ALiBi",
    "grape_ma_rope_alibi": "GRAPE-M+A",
    "fj": "PJ-FJ",
    "affine": "PJ/ALiBi affine",
    "fj_affine": "PJ-FJ+A",
    "full": "PJ full",
    "lc_affine": "LC-PJ+A",
}


RUN_LABELS = {
    "synthetic_grape_controls": "Synthetic bridge",
    "language_tiny_grape_controls": "Tiny Shakespeare",
    "language_wikitext_grape_controls": "WikiText-2",
    "music_motif_grape_controls": "Motif-rich music",
    "maestro_grape_controls": "MAESTRO controls",
    "musicnet_grape_controls": "MusicNet selector A",
}


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args)
    summary = summarize(rows)
    write_csv(args.out_dir / "grape_task_rows.csv", rows)
    write_csv(args.out_dir / "grape_task_summary.csv", summary)
    write_csv(args.out_dir / "grape_task_pivot.csv", pivot_rows(summary))
    write_tex(args.out_dir / "grape_task_summary_table.tex", summary)
    write_readme(args.out_dir, summary)
    write_json(args.out_dir / "summary.json", {"source_rows": len(rows), "summary_rows": len(summary)})
    print(f"Wrote GRAPE task comparison to: {args.out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("runs/grape_task_comparison"))
    parser.add_argument("--include-missing", action="store_true", default=True)
    return parser.parse_args()


def load_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    rows = []
    for run_id, task_family, path in DEFAULT_RUNS:
        if not path.exists():
            if args.include_missing:
                continue
            raise FileNotFoundError(path)
        with path.open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                out = dict(row)
                out["run_id"] = out.get("run_id") or run_id
                out["task_family"] = task_family
                out["source_path"] = str(path)
                if "method" not in out and "sector" in out:
                    out["method"] = out["sector"]
                if "target" not in out:
                    out["target"] = infer_target(path)
                rows.append(out)
    return rows


def summarize(rows: list[dict[str, str]]) -> list[dict[str, str | int | float]]:
    groups = {}
    for row in rows:
        eval_len = int(float(row.get("eval_len", 0)))
        key = (
            row["task_family"],
            row.get("run_id", ""),
            row.get("target", ""),
            row["method"],
            eval_len,
        )
        groups.setdefault(key, []).append(row)

    summary = []
    for key in sorted(groups):
        task_family, run_id, target, method, eval_len = key
        subset = groups[key]
        out: dict[str, str | int | float] = {
            "task_family": task_family,
            "run_id": run_id,
            "target": target,
            "method": method,
            "method_label": METHOD_LABELS.get(method, method),
            "eval_len": eval_len,
            "n": len(subset),
        }
        for metric in ("loss", "accuracy", "ppl", "gate_fj", "gate_affine", "gate_lc"):
            values = numeric_values(subset, metric)
            if values:
                out[f"{metric}_mean"] = statistics.fmean(values)
                out[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        summary.append(out)
    return summary


def write_tex(path: Path, rows: list[dict[str, str | int | float]]) -> None:
    selected = pivot_rows(rows)
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Cross-task stress reruns with GRAPE special-case controls.  Rows use restricted GRAPE-M/A exact special-case controls.  Synthetic rows report accuracy over three seeds; language and music rows report validation cross-entropy over two seeds at the maximum length available in this appendix sweep, 8192 tokens for natural-task rows.  The table reports finite-budget trainability and extrapolation; Table~\ref{tab:grape-fixed-projection} reports primitive containment.  The PJ reference column is PJ-FJ for synthetic rows and LC-PJ+A for natural rows.}",
        r"\label{tab:grape-task-reruns}",
        r"\scriptsize",
        r"\begin{tabularx}{\linewidth}{lXrrrrrr}",
        r"\toprule",
        r"Task & Target / run & Eval & GRAPE-M & GRAPE-A & GRAPE-M+A & PJ full & PJ ref \\",
        r"\midrule",
    ]
    for row in selected:
        lines.append(
            "{task} & {target} & {eval_len} & {grape_m} & {grape_a} & {grape_ma} & {pj_full} & {pj_ref} \\\\".format(
                task=escape_tex(str(row["task"])),
                target=escape_tex(str(row["target_or_run"])),
                eval_len=row["eval_len"],
                grape_m=format_cell(row.get("grape_m_rope")),
                grape_a=format_cell(row.get("grape_a_alibi")),
                grape_ma=format_cell(row.get("grape_ma_rope_alibi")),
                pj_full=format_cell(row.get("full")),
                pj_ref=format_cell(row.get("pj_ref")),
            )
        )
    lines.extend([r"\bottomrule", r"\end{tabularx}", r"\end{table}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def pivot_rows(rows: list[dict[str, str | int | float]]) -> list[dict[str, str | int | float]]:
    pivoted = []
    for group in group_max_eval(rows):
        by_method = {str(row["method"]): row for row in group}
        first = group[0]
        task_family = str(first["task_family"])
        target = str(first.get("target") or "")
        run_id = str(first.get("run_id") or "")
        eval_len = int(first["eval_len"])
        metric_name = "accuracy_mean" if task_family == "synthetic" else "loss_mean"
        pj_ref_method = "fj" if task_family == "synthetic" else "lc_affine"
        pivoted.append(
            {
                "task": display_task(task_family),
                "target_or_run": display_target(run_id, target),
                "eval_len": eval_len,
                "metric": "accuracy" if task_family == "synthetic" else "loss",
                "grape_m_rope": metric_value(by_method.get("grape_m_rope"), metric_name),
                "grape_a_alibi": metric_value(by_method.get("grape_a_alibi"), metric_name),
                "grape_ma_rope_alibi": metric_value(by_method.get("grape_ma_rope_alibi"), metric_name),
                "full": metric_value(by_method.get("full"), metric_name),
                "pj_ref": metric_value(by_method.get(pj_ref_method), metric_name),
                "pj_ref_method": METHOD_LABELS.get(pj_ref_method, pj_ref_method),
            }
        )
    return pivoted


def metric_value(row: dict[str, str | int | float] | None, metric_name: str) -> float | str:
    if row is None or metric_name not in row:
        return ""
    return float(row[metric_name])


def display_task(task_family: str) -> str:
    if task_family == "synthetic":
        return "Synthetic"
    if task_family == "language":
        return "Language"
    if task_family == "music":
        return "Music"
    return task_family


def display_target(run_id: str, target: str) -> str:
    if target:
        return target.replace("_", " ")
    return RUN_LABELS.get(run_id, run_id).replace("_", " ")


def format_cell(value: object) -> str:
    if value == "" or value is None:
        return "--"
    return f"{float(value):.3f}"


def group_max_eval(rows: list[dict[str, str | int | float]]) -> list[list[dict[str, str | int | float]]]:
    grouped = {}
    for row in rows:
        key = (row["task_family"], row["run_id"], row["target"])
        grouped.setdefault(key, []).append(row)
    out = []
    for key in sorted(grouped):
        group = grouped[key]
        max_eval = max(int(row["eval_len"]) for row in group)
        out.append([row for row in group if int(row["eval_len"]) == max_eval])
    return out


def format_metric(row: dict[str, str | int | float]) -> str:
    if "accuracy_mean" in row:
        return f"acc {float(row['accuracy_mean']):.3f}"
    if "loss_mean" in row:
        return f"loss {float(row['loss_mean']):.4f}"
    return "n/a"


def infer_target(path: Path) -> str:
    parent = path.parent.name
    if parent in {"phase", "affine", "first_jet"}:
        return parent
    return ""


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
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_readme(out_dir: Path, summary: list[dict[str, str | int | float]]) -> None:
    lines = [
        "# GRAPE Task Comparison",
        "",
        "Cross-task reruns for restricted GRAPE-M/A exact special-case controls.",
        "The rows report finite-budget trainability and extrapolation for the",
        "appendix comparison.",
        "",
        "Files:",
        "",
        "- `grape_task_rows.csv`",
        "- `grape_task_summary.csv`",
        "- `grape_task_pivot.csv`",
        "- `grape_task_summary_table.tex`",
        "",
        "## Max-eval Summary",
        "",
        "| Task | Target / run | Eval | GRAPE-M | GRAPE-A | GRAPE-M+A | PJ full | PJ ref |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in pivot_rows(summary):
        lines.append(
            "| {task} | {target} | {eval_len} | {grape_m} | {grape_a} | {grape_ma} | {pj_full} | {pj_ref} |".format(
                task=row["task"],
                target=row["target_or_run"],
                eval_len=row["eval_len"],
                grape_m=format_cell(row["grape_m_rope"]),
                grape_a=format_cell(row["grape_a_alibi"]),
                grape_ma=format_cell(row["grape_ma_rope_alibi"]),
                pj_full=format_cell(row["full"]),
                pj_ref=format_cell(row["pj_ref"]),
            )
        )
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def escape_tex(value: str) -> str:
    return value.replace("_", r"\_")


if __name__ == "__main__":
    main()
