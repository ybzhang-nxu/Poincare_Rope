"""Render Phase F stability figures from ``phase_f_stability.py`` outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


VARIANT_ORDER = [
    "raw",
    "scaled",
    "clipped",
    "log",
    "lc_amp_only",
    "lc_phase_only",
    "lc",
    "lc_wrong_scale",
]
VARIANT_COLORS = {
    "raw": "#B23A48",
    "scaled": "#E76F51",
    "clipped": "#F4A261",
    "log": "#E9C46A",
    "lc_amp_only": "#2A9D8F",
    "lc_phase_only": "#457B9D",
    "lc": "#264653",
    "lc_wrong_scale": "#7A7A7A",
}
TITLE_BY_COLUMN = {
    "qk_norm_proxy_final": "LC stability: Q/K proxy",
    "bias_abs_max": "LC stability: bias magnitude",
    "final_effective_support": "LC stability: effective support",
    "local_phase_ratio_final": "LC stability: phase-resolution ratio",
    "far_phase_span_radians": "LC stability: phase span",
    "int8_quant_error_tensor": "LC stability: int8 quantization error",
    "far_gram_condition_number": "LC stability: far Gram condition",
}


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.results)
    numeric_columns = [
        "eval_len",
        "qk_norm_proxy_final",
        "bias_abs_max",
        "final_effective_support",
        "final_average_distance",
        "int8_quant_error_tensor",
        "local_phase_ratio_final",
        "far_phase_span_radians",
        "far_local_collision_rate",
        "wrapped_phase_collision_rate",
        "far_gram_condition_number",
    ]
    for column in numeric_columns:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    outputs = [
        plot_metric(
            frame,
            "qk_norm_proxy_final",
            "final q/k norm proxy",
            args.out_dir / "phase_f_qk_norm.png",
            yscale="log",
        ),
        plot_metric(
            frame,
            "bias_abs_max",
            "max scalar bias magnitude",
            args.out_dir / "phase_f_bias_abs_max.png",
            yscale="log",
        ),
        plot_metric(
            frame,
            "final_effective_support",
            "final-query effective support",
            args.out_dir / "phase_f_effective_support.png",
        ),
        plot_metric(
            frame,
            "local_phase_ratio_final",
            "final local phase-resolution ratio",
            args.out_dir / "phase_f_phase_ratio.png",
            yscale="log",
        ),
        plot_metric(
            frame,
            "far_phase_span_radians",
            "far-window phase span (radians)",
            args.out_dir / "phase_f_phase_span.png",
            yscale="log",
        ),
        plot_metric(
            frame,
            "int8_quant_error_tensor",
            "whole-tensor int8 quantization error",
            args.out_dir / "phase_f_int8_quant_error.png",
            yscale="log",
        ),
        plot_metric(
            frame,
            "far_gram_condition_number",
            "far-window Gram condition number",
            args.out_dir / "phase_f_gram_condition.png",
            yscale="log",
        ),
        plot_collision(frame, args.out_dir / "phase_f_collision_rates.png"),
    ]
    write_manifest(outputs, args.out_dir)
    print(f"Wrote Phase F figures to: {args.out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("runs/phase_f_stability/results.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("runs/phase_f_figures"))
    return parser.parse_args()


def plot_metric(
    frame: pd.DataFrame,
    column: str,
    ylabel: str,
    path: Path,
    *,
    yscale: str = "linear",
) -> Path:
    fig, ax = plt.subplots(figsize=(7.0, 4.4), constrained_layout=True)
    for variant in VARIANT_ORDER:
        subset = frame[frame["variant"] == variant].sort_values("eval_len")
        if subset.empty:
            continue
        ax.plot(
            subset["eval_len"],
            subset[column],
            marker="o",
            linewidth=2,
            color=VARIANT_COLORS.get(variant, "#333333"),
            label=variant,
        )
    ax.set_xscale("log", base=2)
    ax.set_yscale(yscale)
    ax.set_xlabel("eval length")
    ax.set_ylabel(ylabel)
    ax.set_title(TITLE_BY_COLUMN.get(column, f"LC stability: {ylabel}"))
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncols=2)
    return save_figure(fig, path)


def plot_collision(frame: pd.DataFrame, path: Path) -> Path:
    final_len = frame["eval_len"].max()
    subset = frame[frame["eval_len"] == final_len].copy()
    variants = [variant for variant in VARIANT_ORDER if variant in set(subset["variant"])]
    x = range(len(variants))
    local = [
        float(subset[subset["variant"] == variant]["far_local_collision_rate"].iloc[0])
        for variant in variants
    ]
    wrapped = [
        float(subset[subset["variant"] == variant]["wrapped_phase_collision_rate"].iloc[0])
        for variant in variants
    ]

    fig, ax = plt.subplots(figsize=(7.0, 4.0), constrained_layout=True)
    width = 0.36
    ax.bar([idx - width / 2 for idx in x], local, width=width, color="#457B9D", label="local")
    ax.bar([idx + width / 2 for idx in x], wrapped, width=width, color="#E76F51", label="wrapped")
    ax.set_xticks(list(x), variants, rotation=20, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("collision rate")
    ax.set_title(f"LC stability: collision rates at {int(final_len)}")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncols=2)
    return save_figure(fig, path)


def save_figure(fig: plt.Figure, path: Path) -> Path:
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def write_manifest(paths: list[Path], out_dir: Path) -> None:
    lines = ["# Phase F Figures", ""]
    for path in paths:
        lines.append(f"- `{path.name}`")
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
