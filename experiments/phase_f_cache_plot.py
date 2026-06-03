"""Render Phase F cache-quantization proxy figures."""

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


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.results)
    for column in (
        "eval_len",
        "key_norm_final",
        "key_norm_max",
        "logit_std",
        "int8_cache_error_tensor",
        "int4_cache_error_tensor",
        "int8_logit_rel_error_tensor",
        "int4_logit_rel_error_tensor",
    ):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    outputs = [
        plot_metric(
            frame,
            "key_norm_final",
            "final transformed-key norm",
            args.out_dir / "phase_f_cache_key_norm_final.png",
            yscale="log",
        ),
        plot_metric(
            frame,
            "logit_std",
            "final-query logit std",
            args.out_dir / "phase_f_cache_logit_std.png",
            yscale="log",
        ),
        plot_metric(
            frame,
            "int8_cache_error_tensor",
            "int8 cache relative error",
            args.out_dir / "phase_f_cache_int8_error.png",
            yscale="log",
        ),
        plot_metric(
            frame,
            "int4_cache_error_tensor",
            "int4 cache relative error",
            args.out_dir / "phase_f_cache_int4_error.png",
            yscale="log",
        ),
        plot_metric(
            frame,
            "int4_logit_rel_error_tensor",
            "int4 logit relative error",
            args.out_dir / "phase_f_cache_int4_logit_error.png",
            yscale="log",
        ),
        plot_final_bars(
            frame,
            "key_norm_final",
            "final transformed-key norm",
            args.out_dir / "phase_f_cache_final_norm_bars.png",
            yscale="log",
        ),
    ]
    write_manifest(outputs, args.out_dir)
    print(f"Wrote Phase F cache figures to: {args.out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("runs/phase_f_cache_quantization/results.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("runs/phase_f_cache_figures"))
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
    ax.set_title(f"Phase F cache proxy: {ylabel}")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncols=2)
    return save_figure(fig, path)


def plot_final_bars(
    frame: pd.DataFrame,
    column: str,
    ylabel: str,
    path: Path,
    *,
    yscale: str = "linear",
) -> Path:
    final_len = frame["eval_len"].max()
    subset = frame[frame["eval_len"] == final_len]
    variants = [variant for variant in VARIANT_ORDER if variant in set(subset["variant"])]
    values = [float(subset[subset["variant"] == variant][column].iloc[0]) for variant in variants]
    fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    ax.bar(
        range(len(variants)),
        values,
        color=[VARIANT_COLORS.get(variant, "#333333") for variant in variants],
    )
    ax.set_xticks(range(len(variants)), variants, rotation=20, ha="right")
    ax.set_yscale(yscale)
    ax.set_ylabel(ylabel)
    ax.set_title(f"Phase F cache proxy {ylabel}@{int(final_len)}")
    ax.grid(axis="y", alpha=0.25)
    return save_figure(fig, path)


def save_figure(fig: plt.Figure, path: Path) -> Path:
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def write_manifest(paths: list[Path], out_dir: Path) -> None:
    lines = ["# Phase F Cache Figures", ""]
    for path in paths:
        lines.append(f"- `{path.name}`")
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
