"""Export consolidated Phase F tables from completed probe outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


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


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    stability = read_csv(args.stability_results)
    retrieval = read_csv(args.retrieval_results)
    cache = read_csv(args.cache_results)
    trained = read_csv(args.trained_results)
    attention = read_csv(args.attention_results) if args.attention_results.exists() else []

    final_len = max_int(row["eval_len"] for row in stability)
    overview_rows = build_overview(stability, retrieval, cache, trained, attention, final_len)
    write_csv(args.out_dir / "phase_f_32768_overview.csv", overview_rows)
    write_csv(args.out_dir / "phase_f_stability_32768.csv", final_rows(stability, final_len))
    write_csv(args.out_dir / "phase_f_retrieval_x16_32_32768.csv", final_bucket_rows(retrieval, final_len))
    write_csv(args.out_dir / "phase_f_cache_32768.csv", final_rows(cache, final_len))
    write_csv(args.out_dir / "phase_f_trained_x16_32_32768.csv", final_bucket_rows(trained, final_len))
    if attention:
        write_csv(args.out_dir / "phase_f_attention_x16_32_32768.csv", final_bucket_rows(attention, final_len))
    write_summary_json(args.out_dir / "summary.json", overview_rows, final_len)
    write_readme(args.out_dir / "README.md", overview_rows, final_len)

    print(f"Wrote Phase F tables to: {args.out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stability-results", type=Path, default=Path("runs/phase_f_stability/results.csv"))
    parser.add_argument(
        "--retrieval-results",
        type=Path,
        default=Path("runs/phase_f_retrieval_resolution/results.csv"),
    )
    parser.add_argument(
        "--cache-results",
        type=Path,
        default=Path("runs/phase_f_cache_quantization/results.csv"),
    )
    parser.add_argument(
        "--trained-results",
        type=Path,
        default=Path("runs/phase_f_trained_resolution/results.csv"),
    )
    parser.add_argument(
        "--attention-results",
        type=Path,
        default=Path("runs/phase_f_attention_retrieval/results.csv"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("runs/phase_f_tables"))
    return parser.parse_args()


def build_overview(
    stability: list[dict[str, str]],
    retrieval: list[dict[str, str]],
    cache: list[dict[str, str]],
    trained: list[dict[str, str]],
    attention: list[dict[str, str]],
    final_len: int,
) -> list[dict[str, object]]:
    stability_map = {row["variant"]: row for row in final_rows(stability, final_len)}
    cache_map = {row["variant"]: row for row in final_rows(cache, final_len)}
    retrieval_maps = precision_maps(final_bucket_rows(retrieval, final_len))
    trained_maps = precision_maps(final_bucket_rows(trained, final_len))
    attention_maps = precision_maps(final_bucket_rows(attention, final_len)) if attention else {}

    rows = []
    for variant in VARIANT_ORDER:
        srow = stability_map.get(variant, {})
        crow = cache_map.get(variant, {})
        r_fp32 = retrieval_maps.get("fp32", {}).get(variant, {})
        r_int8 = retrieval_maps.get("int8", {}).get(variant, {})
        r_int4 = retrieval_maps.get("int4", {}).get(variant, {})
        t_fp32 = trained_maps.get("fp32", {}).get(variant, {})
        t_int8 = trained_maps.get("int8", {}).get(variant, {})
        t_int4 = trained_maps.get("int4", {}).get(variant, {})
        a_fp32 = attention_maps.get("fp32", {}).get(variant, {})
        a_int8 = attention_maps.get("int8", {}).get(variant, {})
        a_int4 = attention_maps.get("int4", {}).get(variant, {})
        rows.append(
            {
                "variant": variant,
                "eval_len": final_len,
                "stability_qk_norm_final": fnum(srow.get("qk_norm_proxy_final")),
                "stability_support": fnum(srow.get("final_effective_support")),
                "phase_ratio_final": fnum(srow.get("local_phase_ratio_final")),
                "far_phase_span": fnum(srow.get("far_phase_span_radians")),
                "local_collision": fnum(srow.get("far_local_collision_rate")),
                "wrapped_collision": fnum(srow.get("wrapped_phase_collision_rate")),
                "retrieval_fp32_top1": fnum(r_fp32.get("top1_exact")),
                "retrieval_int8_top1": fnum(r_int8.get("top1_exact")),
                "retrieval_int4_top1": fnum(r_int4.get("top1_exact")),
                "retrieval_int4_within64": fnum(r_int4.get("within_64")),
                "retrieval_int4_ties": fnum(r_int4.get("mean_tie_count")),
                "cache_key_norm_final": fnum(crow.get("key_norm_final")),
                "cache_logit_std": fnum(crow.get("logit_std")),
                "cache_int8_error": fnum(crow.get("int8_cache_error_tensor")),
                "cache_int4_error": fnum(crow.get("int4_cache_error_tensor")),
                "trained_fp32_hard_pair_acc": fnum(t_fp32.get("hard_pair_accuracy")),
                "trained_int8_hard_pair_acc": fnum(t_int8.get("hard_pair_accuracy")),
                "trained_int4_hard_pair_acc": fnum(t_int4.get("hard_pair_accuracy")),
                "trained_fp32_hard_neg_accept": fnum(t_fp32.get("hard_negative_accept_rate")),
                "trained_int4_hard_neg_accept": fnum(t_int4.get("hard_negative_accept_rate")),
                "attention_fp32_value_acc": fnum(a_fp32.get("value_accuracy")),
                "attention_int8_value_acc": fnum(a_int8.get("value_accuracy")),
                "attention_int4_value_acc": fnum(a_int4.get("value_accuracy")),
                "attention_fp32_top_position": fnum(a_fp32.get("top_position_accuracy")),
                "attention_int8_top_position": fnum(a_int8.get("top_position_accuracy")),
                "attention_int4_top_position": fnum(a_int4.get("top_position_accuracy")),
                "attention_fp32_target_mass": fnum(a_fp32.get("target_attention_mass")),
                "attention_int8_target_mass": fnum(a_int8.get("target_attention_mass")),
                "attention_int4_target_mass": fnum(a_int4.get("target_attention_mass")),
            }
        )
    return rows


def precision_maps(rows: list[dict[str, str]]) -> dict[str, dict[str, dict[str, str]]]:
    out: dict[str, dict[str, dict[str, str]]] = {}
    for row in rows:
        out.setdefault(row["precision"], {})[row["variant"]] = row
    return out


def final_rows(rows: list[dict[str, str]], final_len: int) -> list[dict[str, str]]:
    return [row for row in rows if int(row["eval_len"]) == final_len]


def final_bucket_rows(rows: list[dict[str, str]], final_len: int) -> list[dict[str, str]]:
    out = [row for row in rows if int(row["eval_len"]) == final_len and row.get("bucket") == "x16_32"]
    if out:
        return out
    return [row for row in rows if int(row["eval_len"]) == final_len]


def write_summary_json(path: Path, overview_rows: list[dict[str, object]], final_len: int) -> None:
    focus = [row for row in overview_rows if row["variant"] in {"raw", "scaled", "lc_phase_only", "lc", "lc_wrong_scale"}]
    path.write_text(
        json.dumps(
            {
                "final_eval_len": final_len,
                "focus_rows": focus,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def write_readme(path: Path, overview_rows: list[dict[str, object]], final_len: int) -> None:
    focus_names = ["raw", "scaled", "lc_phase_only", "lc", "lc_wrong_scale"]
    focus = [row for row in overview_rows if row["variant"] in focus_names]
    lines = [
        "# Phase F Tables",
        "",
        "Consolidated 32k snapshot from completed Phase F probes.",
        "",
        "## Files",
        "",
        "- `phase_f_32768_overview.csv`: joined stability/retrieval/cache/trained-resolution metrics.",
        "- `phase_f_stability_32768.csv`: coordinate stability snapshot.",
        "- `phase_f_retrieval_x16_32_32768.csv`: no-training retrieval hard bucket snapshot.",
        "- `phase_f_cache_32768.csv`: feature-code cache proxy snapshot.",
        "- `phase_f_trained_x16_32_32768.csv`: trained pairwise hard bucket snapshot.",
        "- `phase_f_attention_x16_32_32768.csv`: trained attention value-retrieval hard bucket snapshot.",
        "- `summary.json`: compact focus rows for downstream notes.",
        "",
        f"## Focus Snapshot at `{final_len}`",
        "",
        "| Variant | QK norm | Support | Phase ratio | Retrieval int8 | Retrieval int4 | Cache key norm | Cache logit std | Trained hard acc | Attention top | Attention value |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in focus:
        lines.append(
            "| `{variant}` | `{qk}` | `{support}` | `{phase}` | `{r8}` | `{r4}` | `{cache}` | `{logit}` | `{hard}` | `{atop}` | `{aval}` |".format(
                variant=row["variant"],
                qk=fmt(row["stability_qk_norm_final"]),
                support=fmt(row["stability_support"]),
                phase=fmt(row["phase_ratio_final"]),
                r8=fmt(row["retrieval_int8_top1"]),
                r4=fmt(row["retrieval_int4_top1"]),
                cache=fmt(row["cache_key_norm_final"]),
                logit=fmt(row["cache_logit_std"]),
                hard=fmt(row["trained_fp32_hard_pair_acc"]),
                atop=fmt(row["attention_fp32_top_position"]),
                aval=fmt(row["attention_fp32_value_acc"]),
            )
        )
    lines.extend(
        [
            "",
            "## Reading",
            "",
            "- `raw`: high-order growth negative control.",
            "- `scaled`: strong resolution but poor long-context norm/logit scale.",
            "- `lc_phase_only`: phase compression alone leaves amplitude growth unresolved.",
            "- `lc`: stable norm/logit scale and int8 top-1 retrieval, but strict adjacent-lag resolution degrades.",
            "- `lc_wrong_scale`: invalid scale control; apparent stability comes from destroying phase resolution.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def max_int(values) -> int:
    return max(int(value) for value in values)


def fnum(raw: str | None) -> float:
    if raw is None or raw == "":
        return math.nan
    return float(raw)


def fmt(value: object) -> str:
    if not isinstance(value, (float, int)) or not math.isfinite(float(value)):
        return "n/a"
    value_f = float(value)
    if abs(value_f) >= 1e4 or (0 < abs(value_f) < 1e-3):
        return f"{value_f:.3e}"
    return f"{value_f:.3f}"


if __name__ == "__main__":
    main()
