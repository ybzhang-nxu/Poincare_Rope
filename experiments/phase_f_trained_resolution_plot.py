"""Render Phase F trained-resolution figures."""

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
        "train_pair_accuracy",
        "positive_accept_rate",
        "random_negative_accept_rate",
        "hard_negative_accept_rate",
        "hard_pair_accuracy",
        "hard_negative_mean_lag_gap",
        "hard_negative_mean_feature_distance",
    ):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    final_len = int(frame["eval_len"].max())
    bucket = args.bucket
    if frame[(frame["eval_len"] == final_len) & (frame["bucket"] == bucket)].empty:
        bucket = str(frame[frame["eval_len"] == final_len]["bucket"].iloc[-1])

    outputs = [
        plot_final_bars(
            frame,
            final_len=final_len,
            bucket=bucket,
            precision="fp32",
            metric="hard_pair_accuracy",
            ylabel="hard pair accuracy",
            path=args.out_dir / "phase_f_trained_fp32_hard_pair_accuracy.png",
        ),
        plot_final_bars(
            frame,
            final_len=final_len,
            bucket=bucket,
            precision="int8",
            metric="hard_pair_accuracy",
            ylabel="hard pair accuracy",
            path=args.out_dir / "phase_f_trained_int8_hard_pair_accuracy.png",
        ),
        plot_final_bars(
            frame,
            final_len=final_len,
            bucket=bucket,
            precision="int4",
            metric="hard_pair_accuracy",
            ylabel="hard pair accuracy",
            path=args.out_dir / "phase_f_trained_int4_hard_pair_accuracy.png",
        ),
        plot_final_bars(
            frame,
            final_len=final_len,
            bucket=bucket,
            precision="fp32",
            metric="hard_negative_accept_rate",
            ylabel="hard negative false-positive rate",
            path=args.out_dir / "phase_f_trained_fp32_hard_false_positive.png",
        ),
        plot_final_bars(
            frame,
            final_len=final_len,
            bucket=bucket,
            precision="int4",
            metric="hard_negative_accept_rate",
            ylabel="hard negative false-positive rate",
            path=args.out_dir / "phase_f_trained_int4_hard_false_positive.png",
        ),
        plot_metric_by_length(
            frame,
            bucket=bucket,
            precision="fp32",
            metric="hard_pair_accuracy",
            ylabel="hard pair accuracy",
            path=args.out_dir / "phase_f_trained_fp32_hard_pair_by_length.png",
        ),
        plot_metric_by_length(
            frame,
            bucket=bucket,
            precision="int4",
            metric="hard_pair_accuracy",
            ylabel="hard pair accuracy",
            path=args.out_dir / "phase_f_trained_int4_hard_pair_by_length.png",
        ),
    ]
    write_manifest(outputs, args.out_dir)
    print(f"Wrote Phase F trained-resolution figures to: {args.out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("runs/phase_f_trained_resolution/results.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("runs/phase_f_trained_resolution_figures"))
    parser.add_argument("--bucket", default="x16_32")
    return parser.parse_args()


def plot_final_bars(
    frame: pd.DataFrame,
    *,
    final_len: int,
    bucket: str,
    precision: str,
    metric: str,
    ylabel: str,
    path: Path,
) -> Path:
    subset = frame[
        (frame["eval_len"] == final_len)
        & (frame["bucket"] == bucket)
        & (frame["precision"] == precision)
    ]
    variants = [variant for variant in VARIANT_ORDER if variant in set(subset["variant"])]
    values = [float(subset[subset["variant"] == variant][metric].iloc[0]) for variant in variants]
    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    ax.bar(
        range(len(variants)),
        values,
        color=[VARIANT_COLORS.get(variant, "#333333") for variant in variants],
    )
    ax.set_xticks(range(len(variants)), variants, rotation=20, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel(ylabel)
    ax.set_title(f"Phase F trained {precision} {bucket}@{final_len}: {ylabel}")
    ax.grid(axis="y", alpha=0.25)
    return save_figure(fig, path)


def plot_metric_by_length(
    frame: pd.DataFrame,
    *,
    bucket: str,
    precision: str,
    metric: str,
    ylabel: str,
    path: Path,
) -> Path:
    subset = frame[(frame["bucket"] == bucket) & (frame["precision"] == precision)]
    fig, ax = plt.subplots(figsize=(7.0, 4.4), constrained_layout=True)
    for variant in VARIANT_ORDER:
        part = subset[subset["variant"] == variant].sort_values("eval_len")
        if part.empty:
            continue
        ax.plot(
            part["eval_len"],
            part[metric],
            marker="o",
            linewidth=2,
            color=VARIANT_COLORS.get(variant, "#333333"),
            label=variant,
        )
    ax.set_xscale("log", base=2)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("eval length")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Phase F trained {precision} {bucket}: {ylabel}")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncols=2)
    return save_figure(fig, path)


def save_figure(fig: plt.Figure, path: Path) -> Path:
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def write_manifest(paths: list[Path], out_dir: Path) -> None:
    lines = ["# Phase F Trained Resolution Figures", ""]
    for path in paths:
        lines.append(f"- `{path.name}`")
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
