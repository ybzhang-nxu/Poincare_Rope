"""Export Phase E music comparison tables."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path


RUNS = (
    (
        "motif_rich_smoke_160",
        "motif_rich",
        "smoke_160",
        Path("runs/byte_lm_phase_e_music_motif_rich_smoke_512/results.csv"),
    ),
    (
        "bar_shuffle_control_smoke_160",
        "bar_shuffle_control",
        "smoke_160",
        Path("runs/byte_lm_phase_e_music_bar_shuffle_control_smoke_512/results.csv"),
    ),
    (
        "motif_rich_confirm_420",
        "motif_rich",
        "confirm_420",
        Path("runs/byte_lm_phase_e_music_motif_rich_420_512/results.csv"),
    ),
    (
        "bar_shuffle_control_confirm_420",
        "bar_shuffle_control",
        "confirm_420",
        Path("runs/byte_lm_phase_e_music_bar_shuffle_control_420_512/results.csv"),
    ),
    (
        "motif_rich_confirm_420_3seed",
        "motif_rich",
        "confirm_420_3seed",
        Path("runs/byte_lm_phase_e_music_motif_rich_420_512_3seed/results.csv"),
    ),
    (
        "bar_shuffle_control_confirm_420_3seed",
        "bar_shuffle_control",
        "confirm_420_3seed",
        Path("runs/byte_lm_phase_e_music_bar_shuffle_control_420_512_3seed/results.csv"),
    ),
    (
        "motif_variation_confirm_420",
        "motif_variation",
        "variation_confirm_420",
        Path("runs/byte_lm_phase_e_music_motif_variation_420_512/results.csv"),
    ),
    (
        "variation_shuffle_control_confirm_420",
        "variation_shuffle_control",
        "variation_confirm_420",
        Path("runs/byte_lm_phase_e_music_variation_shuffle_control_420_512/results.csv"),
    ),
    (
        "motif_variation_confirm_420_3seed",
        "motif_variation",
        "variation_confirm_420_3seed",
        Path("runs/byte_lm_phase_e_music_motif_variation_420_512_3seed/results.csv"),
    ),
    (
        "variation_shuffle_control_confirm_420_3seed",
        "variation_shuffle_control",
        "variation_confirm_420_3seed",
        Path("runs/byte_lm_phase_e_music_variation_shuffle_control_420_512_3seed/results.csv"),
    ),
    (
        "motif_rich_stress_8192_3seed",
        "motif_rich",
        "stress_8192_3seed",
        Path("runs/byte_lm_phase_e_music_motif_rich_stress_8192_3seed/results.csv"),
    ),
    (
        "bar_shuffle_control_stress_8192_3seed",
        "bar_shuffle_control",
        "stress_8192_3seed",
        Path("runs/byte_lm_phase_e_music_bar_shuffle_control_stress_8192_3seed/results.csv"),
    ),
    (
        "motif_variation_stress_8192_3seed",
        "motif_variation",
        "variation_stress_8192_3seed",
        Path("runs/byte_lm_phase_e_music_motif_variation_stress_8192_3seed/results.csv"),
    ),
    (
        "variation_shuffle_control_stress_8192_3seed",
        "variation_shuffle_control",
        "variation_stress_8192_3seed",
        Path("runs/byte_lm_phase_e_music_variation_shuffle_control_stress_8192_3seed/results.csv"),
    ),
    (
        "maestro_train64_smoke_160",
        "maestro_train64",
        "maestro_smoke_160",
        Path("runs/byte_lm_phase_e_maestro_midi_train64_smoke_512/results.csv"),
    ),
    (
        "maestro_train64_confirm_420",
        "maestro_train64",
        "maestro_confirm_420",
        Path("runs/byte_lm_phase_e_maestro_midi_train64_420_512/results.csv"),
    ),
    (
        "maestro_train64_confirm_420_3seed",
        "maestro_train64",
        "maestro_confirm_420_3seed",
        Path("runs/byte_lm_phase_e_maestro_midi_train64_420_512_3seed/results.csv"),
    ),
    (
        "maestro_train64_stress_8192",
        "maestro_train64",
        "maestro_stress_8192",
        Path("runs/byte_lm_phase_e_maestro_midi_train64_stress_8192/results.csv"),
    ),
    (
        "maestro_train64_stress_8192_3seed",
        "maestro_train64",
        "maestro_stress_8192_3seed",
        Path("runs/byte_lm_phase_e_maestro_midi_train64_stress_8192_3seed/results.csv"),
    ),
    (
        "maestro_train64_controls_confirm_420_3seed",
        "maestro_train64_controls",
        "maestro_controls_confirm_420_3seed",
        Path("runs/byte_lm_phase_e_maestro_midi_train64_controls_420_512_3seed/results.csv"),
    ),
    (
        "maestro_train64_controls_stress_8192_3seed",
        "maestro_train64_controls",
        "maestro_controls_stress_8192_3seed",
        Path("runs/byte_lm_phase_e_maestro_midi_train64_controls_stress_8192_3seed/results.csv"),
    ),
    (
        "maestro_train_random64_controls_confirm_420_3seed",
        "maestro_train_random64_controls",
        "maestro_random_controls_confirm_420_3seed",
        Path("runs/byte_lm_phase_e_maestro_midi_train_random64_controls_420_512_3seed/results.csv"),
    ),
    (
        "maestro_train_random64_controls_stress_8192_3seed",
        "maestro_train_random64_controls",
        "maestro_random_controls_stress_8192_3seed",
        Path("runs/byte_lm_phase_e_maestro_midi_train_random64_controls_stress_8192_3seed/results.csv"),
    ),
    (
        "maestro_train_random128_controls_confirm_420_3seed",
        "maestro_train_random128_controls",
        "maestro_random128_controls_confirm_420_3seed",
        Path("runs/byte_lm_phase_e_maestro_midi_train_random128_controls_420_512_3seed/results.csv"),
    ),
    (
        "maestro_train_random128_controls_stress_8192_3seed",
        "maestro_train_random128_controls",
        "maestro_random128_controls_stress_8192_3seed",
        Path("runs/byte_lm_phase_e_maestro_midi_train_random128_controls_stress_8192_3seed/results.csv"),
    ),
    (
        "maestro_train_random128_controls_stress_16384_train",
        "maestro_train_random128_controls",
        "maestro_random128_controls_stress_16384_train",
        Path("runs/byte_lm_phase_e_maestro_midi_train_random128_controls_stress_16384/results.csv"),
    ),
    (
        "maestro_train_random128_controls_stress_16384_eval",
        "maestro_train_random128_controls",
        "maestro_random128_controls_stress_16384_eval",
        Path("runs/byte_lm_phase_e_maestro_midi_train_random128_controls_stress_16384_eval/results.csv"),
    ),
    (
        "maestro_train_random128_controls_stress_32768_eval",
        "maestro_train_random128_controls",
        "maestro_random128_controls_stress_32768_eval",
        Path("runs/byte_lm_phase_e_maestro_midi_train_random128_controls_stress_32768_eval/results.csv"),
    ),
    (
        "maestro_validation_random64_controls_stress_32768_eval",
        "maestro_validation_random64_controls",
        "maestro_validation_controls_stress_32768_eval",
        Path("runs/byte_lm_phase_e_maestro_midi_validation_random64_controls_stress_32768_eval/results.csv"),
    ),
    (
        "maestro_test_random64_controls_stress_32768_eval",
        "maestro_test_random64_controls",
        "maestro_test_controls_stress_32768_eval",
        Path("runs/byte_lm_phase_e_maestro_midi_test_random64_controls_stress_32768_eval/results.csv"),
    ),
    (
        "maestro_train_random128_controls_pjrotary_stress_32768_train",
        "maestro_train_random128_controls_pjrotary",
        "maestro_random128_controls_pjrotary_stress_32768_train",
        Path("runs/byte_lm_phase_e_maestro_midi_train_random128_controls_pjrotary_stress_32768/results.csv"),
    ),
    (
        "maestro_train_random128_controls_pjrotary_stress_32768_eval",
        "maestro_train_random128_controls_pjrotary",
        "maestro_random128_controls_pjrotary_stress_32768_eval",
        Path("runs/byte_lm_phase_e_maestro_midi_train_random128_controls_pjrotary_stress_32768_eval/results.csv"),
    ),
    (
        "maestro_validation_random64_controls_pjrotary_stress_32768_eval",
        "maestro_validation_random64_controls_pjrotary",
        "maestro_validation_controls_pjrotary_stress_32768_eval",
        Path("runs/byte_lm_phase_e_maestro_midi_validation_random64_controls_pjrotary_stress_32768_eval/results.csv"),
    ),
    (
        "maestro_test_random64_controls_pjrotary_stress_32768_eval",
        "maestro_test_random64_controls_pjrotary",
        "maestro_test_controls_pjrotary_stress_32768_eval",
        Path("runs/byte_lm_phase_e_maestro_midi_test_random64_controls_pjrotary_stress_32768_eval/results.csv"),
    ),
    (
        "musicnet_random64_programs_smoke_160",
        "musicnet_random64_programs",
        "musicnet_smoke_160",
        Path("runs/byte_lm_phase_e_musicnet_random64_programs_smoke_512/results.csv"),
    ),
    (
        "musicnet_random64_programs_stress_8192_3seed",
        "musicnet_random64_programs",
        "musicnet_stress_8192_3seed",
        Path("runs/byte_lm_phase_e_musicnet_random64_programs_stress_8192_3seed/results.csv"),
    ),
    (
        "musicnet_random64_programs_stress_32768_train",
        "musicnet_random64_programs",
        "musicnet_stress_32768_train",
        Path("runs/byte_lm_phase_e_musicnet_random64_programs_stress_32768/results.csv"),
    ),
    (
        "musicnet_random64_programs_stress_32768_eval",
        "musicnet_random64_programs",
        "musicnet_stress_32768_eval",
        Path("runs/byte_lm_phase_e_musicnet_random64_programs_stress_32768_eval/results.csv"),
    ),
    (
        "musicnet_random64_programs_pjrotary_stress_32768_train",
        "musicnet_random64_programs_pjrotary",
        "musicnet_pjrotary_stress_32768_train",
        Path("runs/byte_lm_phase_e_musicnet_random64_programs_pjrotary_stress_32768/results.csv"),
    ),
    (
        "musicnet_random64_programs_pjrotary_stress_32768_eval",
        "musicnet_random64_programs_pjrotary",
        "musicnet_pjrotary_stress_32768_eval",
        Path("runs/byte_lm_phase_e_musicnet_random64_programs_pjrotary_stress_32768_eval/results.csv"),
    ),
    (
        "musicnet_random64_seed37_programs_stress_32768_train",
        "musicnet_random64_seed37_programs",
        "musicnet_seed37_stress_32768_train",
        Path("runs/byte_lm_phase_e_musicnet_random64_seed37_programs_stress_32768/results.csv"),
    ),
    (
        "musicnet_random64_seed37_programs_stress_32768_eval",
        "musicnet_random64_seed37_programs",
        "musicnet_seed37_stress_32768_eval",
        Path("runs/byte_lm_phase_e_musicnet_random64_seed37_programs_stress_32768_eval/results.csv"),
    ),
    (
        "musicnet_random64_seed37_programs_pjrotary_stress_32768_train",
        "musicnet_random64_seed37_programs_pjrotary",
        "musicnet_seed37_pjrotary_stress_32768_train",
        Path("runs/byte_lm_phase_e_musicnet_random64_seed37_programs_pjrotary_stress_32768/results.csv"),
    ),
    (
        "musicnet_random64_seed37_programs_pjrotary_stress_32768_eval",
        "musicnet_random64_seed37_programs_pjrotary",
        "musicnet_seed37_pjrotary_stress_32768_eval",
        Path("runs/byte_lm_phase_e_musicnet_random64_seed37_programs_pjrotary_stress_32768_eval/results.csv"),
    ),
)

PAIRS = {
    "exact": ("motif_rich", "bar_shuffle_control"),
    "variation": ("motif_variation", "variation_shuffle_control"),
}

METRIC_KEYS = (
    "loss",
    "ppl",
    "delta_loss",
    "attention_entropy",
    "attention_effective_support",
    "average_attention_distance",
    "logit_std",
    "logit_abs_max",
    "gate_fj",
    "gate_affine",
    "gate_lc",
    "affine_slope_mean",
    "fj_high_mass",
    "lc_high_mass",
    "total_high_mass",
)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args)
    summary = summarize(rows)
    paired = paired_confirm_comparison(summary)

    write_csv(args.out_dir / "phase_e_combined_results.csv", rows)
    write_csv(args.out_dir / "phase_e_summary_by_method.csv", summary)
    write_csv(args.out_dir / "phase_e_motif_vs_control.csv", paired)
    write_markdown(rows, summary, paired, args.out_dir)
    print(f"Wrote Phase E tables to: {args.out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("runs/phase_e_tables"))
    parser.add_argument(
        "--include-missing",
        action="store_true",
        help="Skip missing default result files instead of failing.",
    )
    return parser.parse_args()


def load_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    rows = []
    for run_id, corpus_variant, protocol, path in RUNS:
        if not path.exists():
            if args.include_missing:
                continue
            raise FileNotFoundError(path)
        with path.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                out = dict(row)
                out["run_id"] = run_id
                out["corpus"] = corpus_name(corpus_variant)
                out["corpus_variant"] = corpus_variant
                out["protocol"] = protocol
                out["source_path"] = str(path)
                add_high_mass_columns(out)
                rows.append(out)
    return rows


def add_high_mass_columns(row: dict[str, str]) -> None:
    fj_high = sum_numeric_prefix(row, "fj_mass_r", start_order=2)
    lc_high = sum_numeric_prefix(row, "lc_mass_r", start_order=2)
    total = fj_high + lc_high
    row["fj_high_mass"] = str(fj_high) if fj_high > 0.0 else ""
    row["lc_high_mass"] = str(lc_high) if lc_high > 0.0 else ""
    row["total_high_mass"] = str(total) if total > 0.0 else ""


def sum_numeric_prefix(row: dict[str, str], prefix: str, *, start_order: int) -> float:
    total = 0.0
    for key, raw in row.items():
        if not key.startswith(prefix) or raw == "":
            continue
        try:
            order = int(key.removeprefix(prefix))
        except ValueError:
            continue
        if order >= start_order:
            value = float(raw)
            if not math.isnan(value):
                total += value
    return total


def summarize(rows: list[dict[str, str]]) -> list[dict[str, str | int | float]]:
    groups = {}
    for row in rows:
        key = (
            row["run_id"],
            row["corpus_variant"],
            row["protocol"],
            row["method"],
            int(float(row["max_order"])) if row.get("max_order") not in (None, "") else "",
            int(float(row["train_len"])),
            int(float(row["eval_len"])),
        )
        groups.setdefault(key, []).append(row)

    summary = []
    for key in sorted(groups):
        run_id, corpus_variant, protocol, method, max_order, train_len, eval_len = key
        subset = groups[key]
        out: dict[str, str | int | float] = {
            "run_id": run_id,
            "corpus": corpus_name(corpus_variant),
            "corpus_variant": corpus_variant,
            "protocol": protocol,
            "method": method,
            "max_order": max_order,
            "train_len": train_len,
            "eval_len": eval_len,
            "n": len(subset),
        }
        for metric in METRIC_KEYS:
            values = numeric_values(subset, metric)
            if values:
                out[f"{metric}_mean"] = statistics.fmean(values)
                out[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        summary.append(out)
    return summary


def corpus_name(corpus_variant: str) -> str:
    if corpus_variant.startswith("maestro"):
        return "phase_e_maestro_midi"
    if corpus_variant.startswith("musicnet"):
        return "phase_e_musicnet_midi"
    return "phase_e_symbolic_music"


def paired_confirm_comparison(
    summary: list[dict[str, str | int | float]],
) -> list[dict[str, str | int | float]]:
    indexed = {}
    for row in summary:
        if row["protocol"] not in (
            "confirm_420",
            "confirm_420_3seed",
            "variation_confirm_420",
            "variation_confirm_420_3seed",
            "stress_8192_3seed",
            "variation_stress_8192_3seed",
        ):
            continue
        for pair_name, variants in PAIRS.items():
            if row["corpus_variant"] in variants:
                key = (pair_name, row["protocol"], row["method"], int(row["eval_len"]))
                indexed.setdefault(key, {})[row["corpus_variant"]] = row

    paired = []
    for key in sorted(indexed, key=lambda item: (str(item[0]), str(item[1]), str(item[2]), int(item[3]))):
        pair_name, protocol, method, eval_len = key
        structured_variant, control_variant = PAIRS[pair_name]
        variants = indexed[key]
        if structured_variant not in variants or control_variant not in variants:
            continue
        motif = variants[structured_variant]
        control = variants[control_variant]
        motif_loss = float(motif["loss_mean"])
        control_loss = float(control["loss_mean"])
        out: dict[str, str | int | float] = {
            "pair": pair_name,
            "protocol": protocol,
            "structured_variant": structured_variant,
            "control_variant": control_variant,
            "method": method,
            "eval_len": eval_len,
            "motif_n": int(motif["n"]),
            "control_n": int(control["n"]),
            "motif_loss": motif_loss,
            "motif_loss_std": float(motif.get("loss_std", 0.0)),
            "control_loss": control_loss,
            "control_loss_std": float(control.get("loss_std", 0.0)),
            "control_minus_motif_loss": control_loss - motif_loss,
            "motif_ppl": float(motif.get("ppl_mean", 0.0)),
            "control_ppl": float(control.get("ppl_mean", 0.0)),
            "motif_delta_loss": float(motif.get("delta_loss_mean", 0.0)),
            "control_delta_loss": float(control.get("delta_loss_mean", 0.0)),
            "motif_avg_dist": float(motif.get("average_attention_distance_mean", 0.0)),
            "control_avg_dist": float(control.get("average_attention_distance_mean", 0.0)),
            "motif_gates_fj_affine_lc": gate_string(motif),
            "control_gates_fj_affine_lc": gate_string(control),
            "motif_total_high_mass": float(motif.get("total_high_mass_mean", 0.0)),
            "control_total_high_mass": float(control.get("total_high_mass_mean", 0.0)),
        }
        paired.append(out)
    return paired


def write_markdown(
    rows: list[dict[str, str]],
    summary: list[dict[str, str | int | float]],
    paired: list[dict[str, str | int | float]],
    out_dir: Path,
) -> None:
    lines = [
        "# Phase E Tables",
        "",
        "Phase E music tables: symbolic motif-vs-control scaffold plus MAESTRO MIDI snapshots.",
        "",
        f"Combined rows: `{len(rows)}`",
        f"Summary rows: `{len(summary)}`",
        f"Paired rows: `{len(paired)}`",
        "",
        "Files:",
        "",
        "- `phase_e_combined_results.csv`",
        "- `phase_e_summary_by_method.csv`",
        "- `phase_e_motif_vs_control.csv`",
    ]

    for protocol, title in (
        ("confirm_420_3seed", "420-step 3-seed motif-rich vs control snapshot"),
        ("stress_8192_3seed", "8192 sampled-stats 3-seed motif-rich stress snapshot"),
        ("confirm_420", "420-step single-seed motif-rich vs control snapshot"),
        ("variation_confirm_420_3seed", "420-step 3-seed motif-variation vs control snapshot"),
        ("variation_stress_8192_3seed", "8192 sampled-stats 3-seed motif-variation stress snapshot"),
        ("variation_confirm_420", "420-step single-seed motif-variation vs control snapshot"),
    ):
        subset = [row for row in paired if row["protocol"] == protocol]
        if not subset:
            continue
        lines.extend(
            [
                "",
                f"{title}:",
                "",
                "| Pair | Method | Eval Len | N | Structured Loss | Control Loss | Control - Structured | Gates FJ/A/LC | High Mass |",
                "|---|---|---:|---:|---:|---:|---:|---|---:|",
            ]
        )
        for row in subset:
            lines.append(
                "| {pair} | {method} | {eval_len} | {n} | {motif_loss:.4f} | {control_loss:.4f} | {gain:.4f} | {gates} | {high:.4f} |".format(
                    pair=row["pair"],
                    method=row["method"],
                    eval_len=row["eval_len"],
                    n=row["motif_n"],
                    motif_loss=float(row["motif_loss"]),
                    control_loss=float(row["control_loss"]),
                    gain=float(row["control_minus_motif_loss"]),
                    gates=row["motif_gates_fj_affine_lc"],
                    high=float(row["motif_total_high_mass"]),
                )
            )
    for protocol, title, eval_lens in (
        ("maestro_smoke_160", "MAESTRO train64 160-step single-seed MIDI smoke snapshot", (4096,)),
        ("maestro_confirm_420_3seed", "MAESTRO train64 420-step 3-seed MIDI snapshot", (4096,)),
        ("maestro_stress_8192_3seed", "MAESTRO train64 8192 3-seed MIDI stress snapshot", (8192,)),
        ("maestro_controls_confirm_420_3seed", "MAESTRO train64 controls 420-step 3-seed MIDI snapshot", (4096,)),
        ("maestro_controls_stress_8192_3seed", "MAESTRO train64 controls 8192 3-seed MIDI stress snapshot", (8192,)),
        ("maestro_random_controls_confirm_420_3seed", "MAESTRO train random64 controls 420-step 3-seed MIDI snapshot", (4096,)),
        ("maestro_random_controls_stress_8192_3seed", "MAESTRO train random64 controls 8192 3-seed MIDI stress snapshot", (8192,)),
        ("maestro_random128_controls_confirm_420_3seed", "MAESTRO train random128 controls 420-step 3-seed MIDI snapshot", (4096,)),
        ("maestro_random128_controls_stress_8192_3seed", "MAESTRO train random128 controls 8192 3-seed MIDI stress snapshot", (8192,)),
        ("maestro_random128_controls_stress_16384_eval", "MAESTRO train random128 controls 16384 checkpoint eval snapshot", (16384,)),
        ("maestro_random128_controls_stress_32768_eval", "MAESTRO train random128 controls 32768 checkpoint eval snapshot", (32768,)),
        ("maestro_validation_controls_stress_32768_eval", "MAESTRO validation random64 controls 32768 checkpoint eval snapshot", (32768,)),
        ("maestro_test_controls_stress_32768_eval", "MAESTRO test random64 controls 32768 checkpoint eval snapshot", (32768,)),
        ("maestro_random128_controls_pjrotary_stress_32768_eval", "MAESTRO train random128 controls exact PJ-rotary 32768 checkpoint eval snapshot", (512, 32768)),
        ("maestro_validation_controls_pjrotary_stress_32768_eval", "MAESTRO validation controls exact PJ-rotary 32768 checkpoint eval snapshot", (512, 32768)),
        ("maestro_test_controls_pjrotary_stress_32768_eval", "MAESTRO test controls exact PJ-rotary 32768 checkpoint eval snapshot", (512, 32768)),
        ("musicnet_smoke_160", "MusicNet random64 reference-MIDI 160-step single-seed smoke snapshot", (4096,)),
        ("musicnet_stress_8192_3seed", "MusicNet random64 reference-MIDI 8192 3-seed stress snapshot", (8192,)),
        ("musicnet_stress_32768_eval", "MusicNet random64 reference-MIDI 32768 checkpoint eval snapshot", (16384, 32768)),
        ("musicnet_pjrotary_stress_32768_eval", "MusicNet exact PJ-rotary 32768 checkpoint eval snapshot", (512, 8192, 16384, 32768)),
        ("musicnet_seed37_stress_32768_eval", "MusicNet random64 seed37 reference-MIDI 32768 checkpoint eval snapshot", (16384, 32768)),
        ("musicnet_seed37_pjrotary_stress_32768_eval", "MusicNet seed37 exact PJ-rotary 32768 checkpoint eval snapshot", (512, 8192, 16384, 32768)),
        ("maestro_confirm_420", "MAESTRO train64 420-step single-seed MIDI snapshot", (4096,)),
        ("maestro_stress_8192", "MAESTRO train64 8192 single-seed MIDI stress snapshot", (8192,)),
    ):
        append_standalone_snapshot(lines, summary, protocol, title, eval_lens)
    out_dir.joinpath("README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_standalone_snapshot(
    lines: list[str],
    summary: list[dict[str, str | int | float]],
    protocol: str,
    title: str,
    eval_lens: tuple[int, ...],
) -> None:
    subset = [
        row
        for row in summary
        if row["protocol"] == protocol and int(row["eval_len"]) in eval_lens
    ]
    if not subset:
        return

    lines.extend(
        [
            "",
            f"{title}:",
            "",
            "| Variant | Method | Eval Len | N | Loss | Delta Loss | Gates FJ/A/LC | High Mass |",
            "|---|---|---:|---:|---:|---:|---|---:|",
        ]
    )
    for row in sorted(subset, key=lambda item: (int(item["eval_len"]), str(item["method"]))):
        lines.append(
            "| {variant} | {method} | {eval_len} | {n} | {loss:.4f} | {delta:.4f} | {gates} | {high:.4f} |".format(
                variant=row["corpus_variant"],
                method=row["method"],
                eval_len=row["eval_len"],
                n=row["n"],
                loss=float(row["loss_mean"]),
                delta=float(row.get("delta_loss_mean", 0.0)),
                gates=gate_string(row),
                high=float(row.get("total_high_mass_mean", 0.0)),
            )
        )


def gate_string(row: dict[str, str | int | float]) -> str:
    if "gate_affine_mean" not in row:
        return "n/a"
    return "{:.2f}/{:.2f}/{:.2f}".format(
        float(row.get("gate_fj_mean", 0.0)),
        float(row.get("gate_affine_mean", 0.0)),
        float(row.get("gate_lc_mean", 0.0)),
    )


def numeric_values(rows: list[dict[str, str]], key: str) -> list[float]:
    values = []
    for row in rows:
        raw = row.get(key, "")
        if raw == "":
            continue
        value = float(raw)
        if math.isnan(value):
            continue
        values.append(value)
    return values


def write_csv(path: Path, rows: list[dict[str, str | int | float]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    main()
