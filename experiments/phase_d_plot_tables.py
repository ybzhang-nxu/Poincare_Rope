"""Render Phase D figures from exported table CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


METHOD_COLORS = {
    "none": "#7A7A7A",
    "rope": "#2A9D8F",
    "rope_pi": "#6A994E",
    "rope_ntk": "#90BE6D",
    "rope_yarn": "#B5C99A",
    "rope_ntk_affine": "#A7C957",
    "rope_yarn_affine": "#386641",
    "affine": "#E9C46A",
    "fj_affine": "#C8B6FF",
    "fj_lc": "#4D908E",
    "rope_affine": "#457B9D",
    "full": "#8D5A97",
    "lc_affine": "#E76F51",
}
METHOD_ORDER = [
    "none",
    "rope",
    "rope_pi",
    "rope_ntk",
    "rope_yarn",
    "affine",
    "fj_affine",
    "fj_lc",
    "rope_affine",
    "rope_ntk_affine",
    "rope_yarn_affine",
    "full",
    "lc_affine",
]
RUNS = {
    "tinyshakespeare_core_512": ("tinyshakespeare", "Tiny Shakespeare"),
    "tinyshakespeare_core_1024": ("tinyshakespeare_1024", "Tiny Shakespeare train len 1024"),
    "wikitext2_core_512": ("wikitext2", "WikiText-2"),
    "wikitext2_core_1024": ("wikitext2_1024", "WikiText-2 train len 1024"),
    "tinyshakespeare_sector_ablation_1024": (
        "tinyshakespeare_sector_ablation_1024",
        "Tiny Shakespeare sector ablation 1024",
    ),
    "wikitext2_sector_ablation_1024": (
        "wikitext2_sector_ablation_1024",
        "WikiText-2 sector ablation 1024",
    ),
    "wikitext2_stress_16384": (
        "wikitext2_stress_16384",
        "WikiText-2 stress 16384",
    ),
    "tinyshakespeare_stress_16384": (
        "tinyshakespeare_stress_16384",
        "Tiny Shakespeare stress 16384",
    ),
    "tinyshakespeare_stress_32768": (
        "tinyshakespeare_stress_32768",
        "Tiny Shakespeare sampled stress 32768",
    ),
    "wikitext2_stress_32768": (
        "wikitext2_stress_32768",
        "WikiText-2 sampled stress 32768",
    ),
    "gutenberg_warpeace_stress_32768": (
        "gutenberg_warpeace_stress_32768",
        "Gutenberg War and Peace sampled stress 32768",
    ),
}


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(args.tables_dir / "phase_d_summary_by_method.csv")
    for column in (
        "train_len",
        "eval_len",
        "max_order",
        "loss_mean",
        "delta_loss_mean",
        "ppl_mean",
        "average_attention_distance_mean",
        "attention_entropy_mean",
        "gate_fj_mean",
        "gate_affine_mean",
        "gate_lc_mean",
    ):
        if column in summary:
            summary[column] = pd.to_numeric(summary[column], errors="coerce")

    outputs = []
    for run_id, (prefix, title) in RUNS.items():
        frame = summary[summary["run_id"] == run_id].copy()
        if frame.empty:
            continue
        outputs.extend(
            [
                plot_metric(frame, "loss_mean", "loss", title, args.out_dir / f"{prefix}_loss.png"),
                plot_metric(frame, "delta_loss_mean", "delta loss", title, args.out_dir / f"{prefix}_delta_loss.png"),
                plot_metric(
                    frame,
                    "average_attention_distance_mean",
                    "average attention distance",
                    title,
                    args.out_dir / f"{prefix}_attention_distance.png",
                ),
                plot_gates(frame, title, args.out_dir / f"{prefix}_gates.png"),
            ]
        )
    outputs.extend(plot_order_ablation(summary, args.out_dir))
    write_manifest(outputs, args.out_dir)
    print(f"Wrote Phase D figures to: {args.out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables-dir", type=Path, default=Path("runs/phase_d_tables"))
    parser.add_argument("--out-dir", type=Path, default=Path("runs/phase_d_figures"))
    return parser.parse_args()


def plot_metric(frame: pd.DataFrame, column: str, ylabel: str, title: str, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    for method in METHOD_ORDER:
        subset = frame[frame["method"] == method].sort_values("eval_len")
        if subset.empty or column not in subset:
            continue
        ax.plot(
            subset["eval_len"],
            subset[column],
            marker="o",
            linewidth=2,
            color=METHOD_COLORS.get(method, "#333333"),
            label=method,
        )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("eval length")
    ax.set_ylabel(ylabel)
    if "stress 32768" in title.lower() and ylabel == "loss":
        plot_title = "Language 32768-token stress loss"
    else:
        plot_title = f"{title}: {ylabel}"
    ax.set_title(plot_title)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncols=2)
    return save_figure(fig, path)


def plot_gates(frame: pd.DataFrame, title: str, path: Path) -> Path:
    gate_frame = frame[(frame["eval_len"] == frame["train_len"]) & frame["gate_affine_mean"].notna()].copy()
    methods = [method for method in METHOD_ORDER if method in set(gate_frame["method"])]
    x = range(len(methods))
    fj = [float(gate_frame[gate_frame["method"] == method]["gate_fj_mean"].iloc[0]) for method in methods]
    affine = [float(gate_frame[gate_frame["method"] == method]["gate_affine_mean"].iloc[0]) for method in methods]
    lc = [float(gate_frame[gate_frame["method"] == method]["gate_lc_mean"].iloc[0]) for method in methods]

    fig, ax = plt.subplots(figsize=(6.8, 4.2), constrained_layout=True)
    ax.bar(x, fj, color="#2A9D8F", label="FJ")
    ax.bar(x, affine, bottom=fj, color="#E9C46A", label="Affine")
    ax.bar(x, lc, bottom=[a + b for a, b in zip(fj, affine, strict=True)], color="#E76F51", label="LC")
    ax.set_xticks(list(x), methods, rotation=20, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("sector gate")
    if "stress 32768" in title.lower():
        plot_title = "Language 32768-token sector gates"
    else:
        plot_title = f"{title}: learned sector gates"
    ax.set_title(plot_title)
    ax.legend(frameon=False, ncols=3)
    ax.grid(axis="y", alpha=0.25)
    return save_figure(fig, path)


def plot_order_ablation(summary: pd.DataFrame, out_dir: Path) -> list[Path]:
    outputs = []
    specs = (
        ("tiny_shakespeare", "tinyshakespeare_order_ablation_1024", "Tiny Shakespeare order ablation 1024"),
        ("wikitext2", "wikitext2_order_ablation_1024", "WikiText-2 order ablation 1024"),
    )
    for corpus, prefix, title in specs:
        frame = summary[
            (summary["corpus"] == corpus)
            & (summary["protocol"].isin(["order_ablation_1024", "sector_ablation_1024"]))
            & (summary["eval_len"] == 8192)
            & (summary["max_order"].notna())
            & (summary["method"].isin(["fj_affine", "fj_lc", "full", "lc_affine"]))
        ].copy()
        if frame.empty:
            continue
        frame["max_order"] = frame["max_order"].astype(int)
        outputs.extend(
            [
                plot_order_metric(frame, "loss_mean", "loss@8192", title, out_dir / f"{prefix}_loss8192.png"),
                plot_order_metric(
                    frame,
                    "delta_loss_mean",
                    "delta loss@8192",
                    title,
                    out_dir / f"{prefix}_delta_loss8192.png",
                ),
                plot_order_metric(
                    frame,
                    "average_attention_distance_mean",
                    "average attention distance@8192",
                    title,
                    out_dir / f"{prefix}_attention_distance8192.png",
                ),
            ]
        )
    return outputs


def plot_order_metric(frame: pd.DataFrame, column: str, ylabel: str, title: str, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.6, 4.2), constrained_layout=True)
    for method in ["fj_affine", "fj_lc", "full", "lc_affine"]:
        subset = frame[frame["method"] == method].sort_values("max_order")
        if subset.empty or column not in subset:
            continue
        ax.plot(
            subset["max_order"],
            subset[column],
            marker="o",
            linewidth=2,
            color=METHOD_COLORS.get(method, "#333333"),
            label=method,
        )
    ax.set_xticks([0, 1, 3])
    ax.set_xlabel("max order")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}: {ylabel}")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncols=2)
    return save_figure(fig, path)


def save_figure(fig: plt.Figure, path: Path) -> Path:
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def write_manifest(paths: list[Path], out_dir: Path) -> None:
    lines = ["# Phase D Figures", ""]
    for path in paths:
        lines.append(f"- `{path.name}`")
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
