"""Render Phase E music figures from exported table CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


METHOD_ORDER = ["affine", "full", "lc_affine", "rope_ntk_affine", "pj_rotary"]
METHOD_COLORS = {
    "affine": "#E9C46A",
    "full": "#8D5A97",
    "lc_affine": "#E76F51",
    "rope_ntk_affine": "#6A994E",
    "pj_rotary": "#264653",
}
MUSICNET_SELECTORS = {
    "seed29": ("musicnet_stress_32768_eval", "musicnet_pjrotary_stress_32768_eval"),
    "seed37": ("musicnet_seed37_stress_32768_eval", "musicnet_seed37_pjrotary_stress_32768_eval"),
}
MUSICNET_SELECTOR_LABELS = {
    "seed29": "selector A",
    "seed37": "selector B",
}
MAESTRO_SPLIT_LABELS = {
    "train": "controls",
    "validation": "validation",
    "test": "test",
}
MAESTRO_SPLITS = {
    "train": ("maestro_random128_controls_stress_32768_eval", "maestro_random128_controls_pjrotary_stress_32768_eval"),
    "validation": ("maestro_validation_controls_stress_32768_eval", "maestro_validation_controls_pjrotary_stress_32768_eval"),
    "test": ("maestro_test_controls_stress_32768_eval", "maestro_test_controls_pjrotary_stress_32768_eval"),
}


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(args.tables_dir / "phase_e_summary_by_method.csv")
    for column in (
        "eval_len",
        "train_len",
        "loss_mean",
        "loss_std",
        "delta_loss_mean",
        "total_high_mass_mean",
        "gate_fj_mean",
        "gate_affine_mean",
        "gate_lc_mean",
    ):
        if column in summary:
            summary[column] = pd.to_numeric(summary[column], errors="coerce")

    outputs = []
    for selector in MUSICNET_SELECTORS:
        outputs.append(
            plot_loss_curves(
                selector_frame(summary, MUSICNET_SELECTORS[selector]),
                f"MusicNet {MUSICNET_SELECTOR_LABELS[selector]} loss",
                args.out_dir / f"musicnet_{selector}_loss_by_length.png",
            )
        )
    outputs.extend(
        [
            plot_grouped_32768(
                musicnet_32768_frame(summary),
                "selector",
                "MusicNet 32768 loss by selector",
                args.out_dir / "musicnet_32768_selector_ranking.png",
            ),
            plot_grouped_32768(
                maestro_32768_frame(summary),
                "split",
                "MAESTRO 32768 loss by split",
                args.out_dir / "maestro_32768_split_ranking.png",
            ),
            plot_pj_rotary_negative_control(
                summary,
                args.out_dir / "exact_pj_rotary_negative_control.png",
            ),
            plot_high_mass_32768(
                summary,
                args.out_dir / "phase_e_high_mass_32768.png",
            ),
        ]
    )
    write_manifest(outputs, args.out_dir)
    print(f"Wrote Phase E figures to: {args.out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables-dir", type=Path, default=Path("runs/phase_e_tables"))
    parser.add_argument("--out-dir", type=Path, default=Path("runs/phase_e_figures"))
    return parser.parse_args()


def selector_frame(summary: pd.DataFrame, protocols: tuple[str, str]) -> pd.DataFrame:
    frame = summary[summary["protocol"].isin(protocols)].copy()
    return frame[frame["method"].isin(METHOD_ORDER)]


def musicnet_32768_frame(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for selector, protocols in MUSICNET_SELECTORS.items():
        frame = selector_frame(summary, protocols)
        frame = frame[frame["eval_len"] == 32768].copy()
        frame["selector"] = selector
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def maestro_32768_frame(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split, protocols in MAESTRO_SPLITS.items():
        frame = selector_frame(summary, protocols)
        frame = frame[frame["eval_len"] == 32768].copy()
        frame["split"] = split
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def plot_loss_curves(frame: pd.DataFrame, title: str, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.4, 4.6), constrained_layout=True)
    for method in METHOD_ORDER:
        subset = frame[frame["method"] == method].sort_values("eval_len")
        if subset.empty:
            continue
        ax.plot(
            subset["eval_len"],
            subset["loss_mean"],
            marker="o",
            linewidth=2.1,
            color=METHOD_COLORS.get(method, "#333333"),
            label=method,
        )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("eval length")
    ax.set_ylabel("loss")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncols=2)
    return save_figure(fig, path)


def plot_grouped_32768(frame: pd.DataFrame, group_col: str, title: str, path: Path) -> Path:
    groups = list(dict.fromkeys(frame[group_col].tolist()))
    tick_labels = [display_group(group_col, group) for group in groups]
    x = list(range(len(groups)))
    width = 0.16

    fig, ax = plt.subplots(figsize=(7.8, 4.6), constrained_layout=True)
    for method_index, method in enumerate(METHOD_ORDER):
        offsets = [idx + (method_index - (len(METHOD_ORDER) - 1) / 2.0) * width for idx in x]
        values = []
        for group in groups:
            subset = frame[(frame[group_col] == group) & (frame["method"] == method)]
            values.append(float(subset["loss_mean"].iloc[0]) if not subset.empty else float("nan"))
        ax.bar(
            offsets,
            values,
            width=width,
            color=METHOD_COLORS.get(method, "#333333"),
            label=method,
        )
    ax.set_xticks(x, tick_labels)
    ax.set_ylabel("loss@32768")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncols=2)
    return save_figure(fig, path)


def plot_pj_rotary_negative_control(summary: pd.DataFrame, path: Path) -> Path:
    specs = (
        ("MAESTRO controls", "maestro_random128_controls_pjrotary_stress_32768_eval"),
        ("MAESTRO val", "maestro_validation_controls_pjrotary_stress_32768_eval"),
        ("MAESTRO test", "maestro_test_controls_pjrotary_stress_32768_eval"),
        ("MusicNet A", "musicnet_pjrotary_stress_32768_eval"),
        ("MusicNet B", "musicnet_seed37_pjrotary_stress_32768_eval"),
    )
    labels = []
    train_losses = []
    long_losses = []
    for label, protocol in specs:
        frame = summary[(summary["protocol"] == protocol) & (summary["method"] == "pj_rotary")]
        if frame.empty:
            continue
        train = frame[frame["eval_len"] == 512]
        long = frame[frame["eval_len"] == 32768]
        if train.empty or long.empty:
            continue
        labels.append(label)
        train_losses.append(float(train["loss_mean"].iloc[0]))
        long_losses.append(float(long["loss_mean"].iloc[0]))

    fig, ax = plt.subplots(figsize=(8.0, 4.6), constrained_layout=True)
    x = list(range(len(labels)))
    width = 0.34
    ax.bar([idx - width / 2 for idx in x], train_losses, width=width, color="#7A7A7A", label="512")
    ax.bar([idx + width / 2 for idx in x], long_losses, width=width, color=METHOD_COLORS["pj_rotary"], label="32768")
    ax.set_xticks(x, labels, rotation=18, ha="right")
    ax.set_ylabel("loss")
    ax.set_title("Exact PJ-rotary short-fit vs 32768 extrapolation")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncols=2)
    return save_figure(fig, path)


def plot_high_mass_32768(summary: pd.DataFrame, path: Path) -> Path:
    specs = (
        ("MAESTRO controls", "maestro_random128_controls_stress_32768_eval"),
        ("MAESTRO val", "maestro_validation_controls_stress_32768_eval"),
        ("MAESTRO test", "maestro_test_controls_stress_32768_eval"),
        ("MusicNet A", "musicnet_stress_32768_eval"),
        ("MusicNet B", "musicnet_seed37_stress_32768_eval"),
    )
    methods = ["full", "lc_affine"]
    labels = []
    rows = []
    for label, protocol in specs:
        frame = summary[(summary["protocol"] == protocol) & (summary["eval_len"] == 32768)]
        if frame.empty:
            continue
        labels.append(label)
        rows.append(
            [
                float(frame[frame["method"] == method]["total_high_mass_mean"].iloc[0])
                if not frame[frame["method"] == method].empty
                else 0.0
                for method in methods
            ]
        )

    fig, ax = plt.subplots(figsize=(7.6, 4.4), constrained_layout=True)
    x = list(range(len(labels)))
    width = 0.32
    for idx, method in enumerate(methods):
        ax.bar(
            [value + (idx - 0.5) * width for value in x],
            [row[idx] for row in rows],
            width=width,
            color=METHOD_COLORS[method],
            label=method,
        )
    ax.set_xticks(x, labels, rotation=18, ha="right")
    ax.set_ylabel("total high-order mass")
    ax.set_title("High-order mass at 32768")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncols=2)
    return save_figure(fig, path)


def save_figure(fig: plt.Figure, path: Path) -> Path:
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def display_group(group_col: str, value: str) -> str:
    if group_col == "selector":
        return MUSICNET_SELECTOR_LABELS.get(value, value)
    if group_col == "split":
        return MAESTRO_SPLIT_LABELS.get(value, value)
    return value


def write_manifest(paths: list[Path], out_dir: Path) -> None:
    lines = ["# Phase E Figures", ""]
    for path in paths:
        lines.append(f"- `{path.name}`")
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
