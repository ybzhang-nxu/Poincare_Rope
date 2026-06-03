"""Render Phase C figures from exported table CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SECTOR_COLORS = {
    "none": "#7A7A7A",
    "affine": "#E9C46A",
    "fj": "#2A9D8F",
    "full": "#457B9D",
}
GATE_COLORS = {
    "FJ": "#2A9D8F",
    "Affine": "#E9C46A",
    "LC": "#457B9D",
}
TARGET_LABELS = {
    "affine": "attention affine",
    "first_jet": "signed first jet",
    "second_jet": "signed second jet",
    "third_jet": "signed third jet",
    "lc_core": "signed LC core",
}


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(args.tables_dir / "phase_c_summary_by_sector.csv")
    mass = pd.read_csv(args.tables_dir / "phase_c_order_mass.csv")
    for column in (
        "accuracy_mean",
        "accuracy_std",
        "eval_len",
        "gate_fj_mean",
        "gate_affine_mean",
        "gate_lc_mean",
        "attention_lambda",
        "teacher_effective_support_mean",
    ):
        if column in summary:
            summary[column] = pd.to_numeric(summary[column], errors="coerce")
    for column in ("mass_mean", "mass_std", "eval_len", "order"):
        if column in mass:
            mass[column] = pd.to_numeric(mass[column], errors="coerce")

    outputs = [
        plot_attention_affine(summary, args.out_dir),
        plot_attention_support_sweep(summary, args.out_dir),
        plot_signed_multilen(summary, args.out_dir),
        plot_full_auto_accuracy(summary, args.out_dir),
        plot_full_auto_mass(mass, args.out_dir),
        plot_lc_core_accuracy(summary, args.out_dir),
        plot_lc_core_mass(mass, args.out_dir),
        plot_gate_snapshot(summary, args.out_dir),
    ]
    write_manifest(outputs, args.out_dir)
    print(f"Wrote Phase C figures to: {args.out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables-dir", type=Path, default=Path("runs/phase_c_tables"))
    parser.add_argument("--out-dir", type=Path, default=Path("runs/phase_c_figures"))
    return parser.parse_args()


def plot_attention_affine(summary: pd.DataFrame, out_dir: Path) -> Path:
    frame = summary[summary["run_id"] == "attention_affine_3seed"].copy()
    sectors = ["none", "affine", "full"]
    return plot_accuracy_group(
        frame,
        sectors=sectors,
        title="Attention teacher: affine recency",
        path=out_dir / "attention_affine_accuracy.png",
    )


def plot_attention_support_sweep(summary: pd.DataFrame, out_dir: Path) -> Path:
    frame = summary[
        (summary["run_id"] == "attention_recency_support_sweep")
        & (summary["eval_len"] == 192)
    ].copy()
    sectors = ["none", "affine", "fj_affine", "full"]
    fig, ax = plt.subplots(figsize=(7.4, 4.8), constrained_layout=True)
    for sector in sectors:
        subset = frame[frame["sector"] == sector].sort_values("teacher_effective_support_mean")
        if subset.empty:
            continue
        ax.errorbar(
            subset["teacher_effective_support_mean"],
            subset["accuracy_mean"],
            yerr=subset["accuracy_std"].fillna(0.0),
            marker="o",
            linewidth=2,
            capsize=3,
            color=SECTOR_COLORS.get(sector, "#333333"),
            label=sector,
        )
        for row in subset.itertuples():
            ax.text(
                row.teacher_effective_support_mean,
                row.accuracy_mean + 0.018,
                f"l={row.attention_lambda:g}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
    ax.set_title("Attention recency+weak-jet support sweep at T=192")
    ax.set_xlabel("teacher effective support")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0.45, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncols=2)
    return save_figure(fig, out_dir / "attention_recency_support_sweep.png")


def plot_signed_multilen(summary: pd.DataFrame, out_dir: Path) -> Path:
    frame = summary[summary["protocol"] == "signed_multi_len"].copy()
    targets = ["first_jet", "second_jet", "third_jet"]
    sectors = ["none", "affine", "fj"]
    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.8), sharey=True, constrained_layout=True)
    for ax, target in zip(axes, targets, strict=True):
        plot_accuracy_group(
            frame[frame["target"] == target],
            sectors=sectors,
            title=TARGET_LABELS[target],
            path=None,
            ax=ax,
        )
    axes[0].set_ylabel("accuracy")
    fig.suptitle("Signed teachers with multi-length training")
    return save_figure(fig, out_dir / "signed_multilen_accuracy.png")


def plot_full_auto_accuracy(summary: pd.DataFrame, out_dir: Path) -> Path:
    frame = summary[summary["protocol"] == "signed_full_auto_multi_len"].copy()
    targets = ["first_jet", "second_jet", "third_jet"]
    eval_lens = sorted(frame["eval_len"].dropna().astype(int).unique())
    x = np.arange(len(targets))
    width = 0.34

    fig, ax = plt.subplots(figsize=(7.0, 4.6), constrained_layout=True)
    for idx, eval_len in enumerate(eval_lens):
        values = []
        errors = []
        for target in targets:
            row = frame[(frame["target"] == target) & (frame["eval_len"] == eval_len)].iloc[0]
            values.append(float(row["accuracy_mean"]))
            errors.append(float(row.get("accuracy_std", 0.0)))
        offset = (idx - (len(eval_lens) - 1) / 2) * width
        ax.bar(
            x + offset,
            values,
            width,
            yerr=errors,
            capsize=3,
            label=f"T={eval_len}",
            color=["#2A9D8F", "#457B9D"][idx % 2],
        )
        for xpos, value in zip(x + offset, values, strict=True):
            ax.text(xpos, value + 0.018, f"{value:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_title("Full PJ auto-initialized signed teachers")
    ax.set_ylabel("accuracy")
    ax.set_xticks(x, ["first jet", "second jet", "third jet"])
    ax.set_ylim(0.0, 1.08)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    return save_figure(fig, out_dir / "full_auto_accuracy.png")


def plot_full_auto_mass(mass: pd.DataFrame, out_dir: Path) -> Path:
    frame = mass[
        (mass["protocol"] == "signed_full_auto_multi_len")
        & (mass["component"] == "fj")
    ].copy()
    grouped = (
        frame.groupby(["target", "order"], as_index=False)["mass_mean"]
        .mean()
        .pivot(index="target", columns="order", values="mass_mean")
    )
    targets = [target for target in ("first_jet", "second_jet", "third_jet") if target in grouped.index]
    orders = sorted(grouped.columns.astype(int).tolist())
    values = grouped.reindex(index=targets, columns=orders).to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(6.8, 3.8), constrained_layout=True)
    image = ax.imshow(values, aspect="auto", cmap="YlGnBu")
    ax.set_title("Full PJ auto FJ order mass")
    ax.set_xlabel("order")
    ax.set_ylabel("target")
    ax.set_xticks(np.arange(len(orders)), [f"r{order}" for order in orders])
    ax.set_yticks(np.arange(len(targets)), [TARGET_LABELS[target] for target in targets])
    for y, target in enumerate(targets):
        for x, order in enumerate(orders):
            value = grouped.loc[target, order]
            ax.text(x, y, f"{value:.3f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="mean FJ mass")
    return save_figure(fig, out_dir / "full_auto_fj_mass.png")


def plot_lc_core_accuracy(summary: pd.DataFrame, out_dir: Path) -> Path:
    frame = summary[summary["run_id"] == "signed_lccore_multilen_3seed"].copy()
    sectors = ["none", "fj", "lc_affine", "full"]
    return plot_accuracy_group(
        frame,
        sectors=sectors,
        title="Signed-teacher LC-core task",
        path=out_dir / "lc_core_accuracy.png",
    )


def plot_lc_core_mass(mass: pd.DataFrame, out_dir: Path) -> Path:
    frame = mass[
        (mass["run_id"] == "signed_lccore_multilen_3seed")
        & (mass["component"] == "lc")
        & (mass["sector"].isin(["lc_affine", "full"]))
    ].copy()
    grouped = (
        frame.groupby(["sector", "order"], as_index=False)["mass_mean"]
        .mean()
        .pivot(index="sector", columns="order", values="mass_mean")
    )
    sectors = [sector for sector in ("lc_affine", "full") if sector in grouped.index]
    orders = sorted(grouped.columns.astype(int).tolist())
    values = grouped.reindex(index=sectors, columns=orders).to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(6.2, 3.4), constrained_layout=True)
    image = ax.imshow(values, aspect="auto", cmap="PuBuGn")
    ax.set_title("LC core order mass")
    ax.set_xlabel("order")
    ax.set_ylabel("sector")
    ax.set_xticks(np.arange(len(orders)), [f"r{order}" for order in orders])
    ax.set_yticks(np.arange(len(sectors)), sectors)
    for y, sector in enumerate(sectors):
        for x, order in enumerate(orders):
            value = grouped.loc[sector, order]
            ax.text(x, y, f"{value:.3f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="mean LC mass")
    return save_figure(fig, out_dir / "lc_core_lc_mass.png")


def plot_gate_snapshot(summary: pd.DataFrame, out_dir: Path) -> Path:
    frame = summary[
        (summary["eval_len"] == 96)
        & summary["gate_fj_mean"].notna()
        & summary["run_id"].isin(
            [
                "attention_affine_3seed",
                "signed_firstjet_multilen_3seed",
                "signed_firstjet_full_auto_multilen_3seed",
                "signed_secondjet_multilen_3seed",
                "signed_secondjet_full_auto_multilen_3seed",
                "signed_thirdjet_multilen_3seed",
                "signed_thirdjet_full_auto_multilen_3seed",
                "signed_lccore_multilen_3seed",
            ]
        )
    ].copy()
    frame = frame.sort_values(["target", "sector"])
    labels = [f"{row.target}\n{row.sector}" for row in frame.itertuples()]
    y = np.arange(len(frame))
    fj = frame["gate_fj_mean"].to_numpy(dtype=float)
    affine = frame["gate_affine_mean"].to_numpy(dtype=float)
    lc = frame["gate_lc_mean"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(8.2, 5.6), constrained_layout=True)
    left = np.zeros_like(fj)
    ax.barh(y, fj, left=left, color=GATE_COLORS["FJ"], label="FJ")
    left += fj
    ax.barh(y, affine, left=left, color=GATE_COLORS["Affine"], label="Affine")
    left += affine
    ax.barh(y, lc, left=left, color=GATE_COLORS["LC"], label="LC")
    ax.set_title("Synthetic sector gates")
    ax.set_xlabel("gate probability")
    ax.set_yticks(y, labels)
    ax.set_xlim(0.0, 1.0)
    ax.invert_yaxis()
    ax.legend(loc="lower right", ncols=3, frameon=False)
    ax.grid(axis="x", alpha=0.25)
    return save_figure(fig, out_dir / "sector_gates.png")


def plot_accuracy_group(
    frame: pd.DataFrame,
    *,
    sectors: list[str],
    title: str,
    path: Path | None,
    ax: plt.Axes | None = None,
) -> Path | None:
    own_fig = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    else:
        fig = ax.figure

    eval_lens = sorted(frame["eval_len"].dropna().astype(int).unique())
    x = np.arange(len(sectors))
    width = 0.34 if len(eval_lens) == 2 else 0.26
    colors = ["#2A9D8F", "#457B9D", "#E76F51"]
    for idx, eval_len in enumerate(eval_lens):
        values = []
        errors = []
        for sector in sectors:
            subset = frame[(frame["sector"] == sector) & (frame["eval_len"] == eval_len)]
            if subset.empty:
                values.append(np.nan)
                errors.append(0.0)
                continue
            row = subset.iloc[0]
            values.append(float(row["accuracy_mean"]))
            errors.append(float(row.get("accuracy_std", 0.0)))
        offset = (idx - (len(eval_lens) - 1) / 2) * width
        ax.bar(
            x + offset,
            values,
            width,
            yerr=errors,
            capsize=3,
            label=f"T={eval_len}",
            color=colors[idx % len(colors)],
        )
        for xpos, value in zip(x + offset, values, strict=True):
            if np.isnan(value):
                continue
            ax.text(xpos, value + 0.018, f"{value:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_title(title)
    ax.set_xticks(x, sectors)
    ax.set_ylim(0.0, 1.08)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    if own_fig:
        ax.set_ylabel("accuracy")
        return save_figure(fig, path)
    return None


def save_figure(fig: plt.Figure, path: Path | None) -> Path:
    if path is None:
        raise ValueError("path is required when saving a figure")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_manifest(outputs: list[Path], out_dir: Path) -> None:
    lines = [
        "# Phase C Figures",
        "",
        "Generated from `runs/phase_c_tables`.",
        "",
        "## Files",
        "",
    ]
    for path in outputs:
        lines.append(f"- `{path.name}`")
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
