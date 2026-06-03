"""Render Phase F retrieval-resolution figures."""

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
        "top1_exact",
        "within_64",
        "bucket_accuracy",
        "mean_abs_error",
        "mean_margin",
        "mean_tie_count",
    ):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    final_len = int(frame["eval_len"].max())
    final_bucket = args.bucket
    if frame[(frame["eval_len"] == final_len) & (frame["bucket"] == final_bucket)].empty:
        final_bucket = str(frame[frame["eval_len"] == final_len]["bucket"].iloc[-1])

    outputs = [
        plot_final_bars(
            frame,
            final_len=final_len,
            bucket=final_bucket,
            precision="int8",
            metric="top1_exact",
            ylabel="top-1 exact",
            path=args.out_dir / "phase_f_retrieval_int8_top1.png",
        ),
        plot_final_bars(
            frame,
            final_len=final_len,
            bucket=final_bucket,
            precision="int4",
            metric="top1_exact",
            ylabel="top-1 exact",
            path=args.out_dir / "phase_f_retrieval_int4_top1.png",
        ),
        plot_final_bars(
            frame,
            final_len=final_len,
            bucket=final_bucket,
            precision="int4",
            metric="within_64",
            ylabel="within 64 lags",
            path=args.out_dir / "phase_f_retrieval_int4_within64.png",
        ),
        plot_final_bars(
            frame,
            final_len=final_len,
            bucket=final_bucket,
            precision="int4",
            metric="mean_tie_count",
            ylabel="mean tie count",
            path=args.out_dir / "phase_f_retrieval_int4_ties.png",
            yscale="log",
        ),
        plot_margin_by_precision(
            frame,
            final_len=final_len,
            bucket=final_bucket,
            path=args.out_dir / "phase_f_retrieval_margin_by_precision.png",
        ),
    ]
    write_manifest(outputs, args.out_dir)
    print(f"Wrote Phase F retrieval figures to: {args.out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("runs/phase_f_retrieval_resolution/results.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("runs/phase_f_retrieval_figures"))
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
    yscale: str = "linear",
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
    ax.set_yscale(yscale)
    ax.set_ylabel(ylabel)
    ax.set_title(f"Phase F retrieval {precision} {bucket}@{final_len}: {ylabel}")
    ax.grid(axis="y", alpha=0.25)
    return save_figure(fig, path)


def plot_margin_by_precision(
    frame: pd.DataFrame,
    *,
    final_len: int,
    bucket: str,
    path: Path,
) -> Path:
    subset = frame[(frame["eval_len"] == final_len) & (frame["bucket"] == bucket)]
    variants = [variant for variant in VARIANT_ORDER if variant in set(subset["variant"])]
    precisions = [precision for precision in ("fp32", "int8", "int4") if precision in set(subset["precision"])]
    width = 0.24
    offsets = {"fp32": -width, "int8": 0.0, "int4": width}
    colors = {"fp32": "#7A7A7A", "int8": "#457B9D", "int4": "#E76F51"}

    fig, ax = plt.subplots(figsize=(7.6, 4.4), constrained_layout=True)
    x = list(range(len(variants)))
    for precision in precisions:
        values = []
        for variant in variants:
            row = subset[(subset["variant"] == variant) & (subset["precision"] == precision)]
            values.append(float(row["mean_margin"].iloc[0]))
        ax.bar(
            [idx + offsets[precision] for idx in x],
            values,
            width=width,
            color=colors[precision],
            label=precision,
        )
    ax.axhline(0.0, color="#333333", linewidth=1)
    ax.set_xticks(x, variants, rotation=20, ha="right")
    ax.set_ylabel("target minus best non-target score")
    ax.set_title(f"Phase F retrieval margin {bucket}@{final_len}")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncols=3)
    return save_figure(fig, path)


def save_figure(fig: plt.Figure, path: Path) -> Path:
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def write_manifest(paths: list[Path], out_dir: Path) -> None:
    lines = ["# Phase F Retrieval Figures", ""]
    for path in paths:
        lines.append(f"- `{path.name}`")
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
