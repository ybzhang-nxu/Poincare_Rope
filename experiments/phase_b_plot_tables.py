"""Render Phase B figures from exported table CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TARGET_ORDER = [
    "phase",
    "first_jet",
    "second_jet",
    "third_jet",
    "linear",
    "recency_weak_jet",
    "lc_core",
    "lc_affine",
]

GATE_COLORS = {
    "FJ": "#2A9D8F",
    "Affine": "#E9C46A",
    "LC": "#457B9D",
}


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    outputs = []
    fixed_best = pd.read_csv(args.tables_dir / "fixed_best_by_target.csv")
    adaptive_gates = pd.read_csv(args.tables_dir / "adaptive_gate_heatmap.csv")
    adaptive_orders = pd.read_csv(args.tables_dir / "adaptive_order_heatmap.csv")

    outputs.append(plot_fixed_best(fixed_best, "log10_mse", "Best fixed basis log10 MSE", "fixed_best_log10_mse", args.out_dir))
    outputs.append(
        plot_fixed_best(
            fixed_best,
            "log10_condition_number",
            "Best fixed basis log10 condition number",
            "fixed_best_condition",
            args.out_dir,
            cmap="cividis",
        )
    )
    outputs.append(plot_adaptive_gates(adaptive_gates, args.out_dir))
    outputs.append(
        plot_order_heatmap(
            adaptive_orders,
            "mass_mean",
            "Adaptive effective mass by order",
            "adaptive_order_mass",
            args.out_dir,
            cmap="YlGnBu",
        )
    )
    outputs.append(
        plot_order_heatmap(
            adaptive_orders,
            "energy_mean",
            "Adaptive functional energy by order",
            "adaptive_order_energy",
            args.out_dir,
            cmap="YlOrRd",
        )
    )
    outputs.append(
        plot_order_heatmap(
            adaptive_orders,
            "loo_delta_mean",
            "Leave-one-order-out delta",
            "adaptive_loo_delta",
            args.out_dir,
            cmap="magma",
            log_magnitude=True,
        )
    )

    multifreq_path = args.tables_dir / "adaptive_multifreq_probe.csv"
    if multifreq_path.exists():
        outputs.append(plot_multifreq_probe(pd.read_csv(multifreq_path), args.out_dir))

    write_manifest(outputs, args.out_dir)
    print(f"Wrote Phase B figures to: {args.out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables-dir", type=Path, default=Path("runs/phase_b_tables"))
    parser.add_argument("--out-dir", type=Path, default=Path("runs/phase_b_figures"))
    return parser.parse_args()


def plot_fixed_best(
    frame: pd.DataFrame,
    value_col: str,
    title: str,
    stem: str,
    out_dir: Path,
    cmap: str = "viridis",
) -> Path:
    frame = add_target_rank(frame)
    frame = frame.sort_values(["target_rank", "omega_cycles"])
    targets = ordered_targets(frame["target"])
    omegas = sorted(frame["omega_cycles"].unique())
    values = frame.pivot(index="target", columns="omega_cycles", values=value_col).reindex(index=targets, columns=omegas)
    bases = frame.pivot(index="target", columns="omega_cycles", values="basis").reindex(index=targets, columns=omegas)

    fig, ax = plt.subplots(figsize=(7.2, 5.6), constrained_layout=True)
    image = ax.imshow(values.to_numpy(dtype=float), aspect="auto", cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("omega cycles")
    ax.set_ylabel("target")
    ax.set_xticks(np.arange(len(omegas)), [format_omega(v) for v in omegas])
    ax.set_yticks(np.arange(len(targets)), targets)

    for y, target in enumerate(targets):
        for x, omega in enumerate(omegas):
            value = values.loc[target, omega]
            basis = bases.loc[target, omega]
            if pd.isna(value) or pd.isna(basis):
                continue
            ax.text(x, y, f"{short_basis(str(basis))}\n{float(value):.2f}", ha="center", va="center", fontsize=6.5)

    fig.colorbar(image, ax=ax, label=value_col)
    return save_figure(fig, out_dir / f"{stem}.png")


def plot_adaptive_gates(frame: pd.DataFrame, out_dir: Path) -> Path:
    frame = add_target_rank(frame).sort_values("target_rank")
    targets = frame["target"].tolist()
    y = np.arange(len(targets))
    fj = frame["gate_fj_mean"].to_numpy(dtype=float)
    affine = frame["gate_affine_mean"].to_numpy(dtype=float)
    lc = frame["gate_lc_mean"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(7.4, 5.4), constrained_layout=True)
    left = np.zeros_like(fj)
    ax.barh(y, fj, left=left, label="FJ", color=GATE_COLORS["FJ"])
    left += fj
    ax.barh(y, affine, left=left, label="Affine", color=GATE_COLORS["Affine"])
    left += affine
    ax.barh(y, lc, left=left, label="LC", color=GATE_COLORS["LC"])

    ax.set_title("Adaptive sector gates")
    ax.set_xlabel("gate probability")
    ax.set_yticks(y, targets)
    ax.set_xlim(0.0, 1.0)
    ax.invert_yaxis()
    ax.legend(loc="lower right", ncols=3, frameon=False)
    ax.grid(axis="x", alpha=0.25)
    return save_figure(fig, out_dir / "adaptive_gates.png")


def plot_order_heatmap(
    frame: pd.DataFrame,
    value_col: str,
    title: str,
    stem: str,
    out_dir: Path,
    cmap: str,
    log_magnitude: bool = False,
) -> Path:
    frame = add_target_rank(frame)
    targets = ordered_targets(frame["target"])
    orders = sorted(frame["order"].unique())
    values = frame.pivot(index="target", columns="order", values=value_col).reindex(index=targets, columns=orders)
    matrix = values.to_numpy(dtype=float)
    color_label = value_col
    annotate = matrix
    if log_magnitude:
        matrix = np.log10(np.maximum(np.abs(matrix), 1e-8))
        color_label = f"log10 abs {value_col}"

    fig, ax = plt.subplots(figsize=(7.0, 5.4), constrained_layout=True)
    image = ax.imshow(matrix, aspect="auto", cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("jet order r")
    ax.set_ylabel("target")
    ax.set_xticks(np.arange(len(orders)), [f"r{int(order)}" for order in orders])
    ax.set_yticks(np.arange(len(targets)), targets)

    for y, target in enumerate(targets):
        for x, order in enumerate(orders):
            raw = annotate[y, x]
            if np.isnan(raw):
                continue
            ax.text(x, y, format_cell(raw), ha="center", va="center", fontsize=7)

    fig.colorbar(image, ax=ax, label=color_label)
    return save_figure(fig, out_dir / f"{stem}.png")


def plot_multifreq_probe(frame: pd.DataFrame, out_dir: Path) -> Path:
    frame = add_target_rank(frame)
    targets = ordered_targets(frame["target"])
    omegas = sorted(frame["omega_cycles"].unique())
    values = frame.pivot(index="target", columns="omega_cycles", values="top_order_mode").reindex(index=targets, columns=omegas)

    fig, ax = plt.subplots(figsize=(6.8, 4.4), constrained_layout=True)
    image = ax.imshow(values.to_numpy(dtype=float), aspect="auto", cmap="PuBuGn", vmin=0, vmax=3)
    ax.set_title("Adaptive multi-frequency top order")
    ax.set_xlabel("omega cycles")
    ax.set_ylabel("target")
    ax.set_xticks(np.arange(len(omegas)), [format_omega(v) for v in omegas])
    ax.set_yticks(np.arange(len(targets)), targets)

    for y, target in enumerate(targets):
        for x, omega in enumerate(omegas):
            value = values.loc[target, omega]
            if pd.isna(value):
                continue
            ax.text(x, y, f"r{int(value)}", ha="center", va="center", fontsize=8)

    fig.colorbar(image, ax=ax, label="top order mode")
    return save_figure(fig, out_dir / "adaptive_multifreq_top_order.png")


def add_target_rank(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    order = {target: rank for rank, target in enumerate(TARGET_ORDER)}
    out["target_rank"] = out["target"].map(lambda target: order.get(target, len(order)))
    return out


def ordered_targets(values: pd.Series) -> list[str]:
    present = set(values.tolist())
    ordered = [target for target in TARGET_ORDER if target in present]
    ordered.extend(sorted(present.difference(ordered)))
    return ordered


def format_omega(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:g}"


def short_basis(value: str) -> str:
    replacements = {
        "scaled_R": "sR",
        "jordan_R": "jR",
        "_affine": "+A",
        "rope_alibi": "rope+A",
        "lc_R": "lcR",
    }
    out = value
    for old, new in replacements.items():
        out = out.replace(old, new)
    return out


def format_cell(value: float) -> str:
    if abs(value) >= 0.01:
        return f"{value:.2f}"
    return f"{value:.1e}"


def save_figure(fig: plt.Figure, path: Path) -> Path:
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_manifest(outputs: list[Path], out_dir: Path) -> None:
    lines = [
        "# Phase B Figures",
        "",
        "Generated from `runs/phase_b_tables`.",
        "",
        "## Files",
        "",
    ]
    for path in outputs:
        lines.append(f"- `{path.name}`")
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
