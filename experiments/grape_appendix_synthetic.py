"""Optional GRAPE appendix synthetic bridge runner.

This wrapper runs the existing synthetic query-LM bridge with restricted
GRAPE-M/A special-case scalar controls.  It is optional appendix evidence and
must not be described as a full learned GRAPE or GRAPE-AP reproduction.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    result_paths = []
    for target in parse_csv(args.targets):
        target_dir = args.out_dir / target
        cmd = [
            sys.executable,
            str(ROOT / "experiments/synthetic_query_lm.py"),
            "--teacher",
            "signed",
            "--target",
            target,
            "--train-len",
            str(args.train_len),
            "--train-seq-lens",
            args.train_seq_lens,
            "--train-length-sampling",
            args.train_length_sampling,
            "--eval-lens",
            args.eval_lens,
            "--sectors",
            args.sectors,
            "--seeds",
            args.seeds,
            "--embed-dim",
            str(args.embed_dim),
            "--num-heads",
            str(args.num_heads),
            "--layers",
            str(args.layers),
            "--max-order",
            str(args.max_order),
            "--batch-size",
            str(args.batch_size),
            "--steps",
            str(args.steps),
            "--lr",
            str(args.lr),
            "--weight-decay",
            str(args.weight_decay),
            "--gate-init",
            "auto",
            "--order-init",
            "auto",
            "--freeze-qk",
            "--median-threshold",
            "--eval-batches",
            str(args.eval_batches),
            "--out-dir",
            str(target_dir),
        ]
        if args.device:
            cmd.extend(["--device", args.device])
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, cwd=ROOT, check=True)
        result_paths.append(target_dir / "results.csv")
    combine_results(result_paths, args.out_dir / "results.csv")
    write_readme(args)
    print(f"Wrote GRAPE appendix synthetic results to: {args.out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", default="phase,affine,first_jet")
    parser.add_argument(
        "--sectors",
        default="grape_m_rope,grape_a_alibi,grape_ma_rope_alibi,fj,affine,fj_affine,lc_affine,full",
    )
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--train-len", type=int, default=96)
    parser.add_argument("--train-seq-lens", default="48,96")
    parser.add_argument("--train-length-sampling", choices=["cycle", "random"], default="cycle")
    parser.add_argument("--eval-lens", default="96,192,384")
    parser.add_argument("--embed-dim", type=int, default=96)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--max-order", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--steps", type=int, default=350)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--device", default="")
    parser.add_argument("--out-dir", type=Path, default=Path("runs/grape_appendix_synthetic"))
    return parser.parse_args()


def combine_results(paths: list[Path], out_path: Path) -> None:
    rows = []
    fieldnames = set()
    for path in paths:
        with path.open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                out = dict(row)
                out["source_path"] = str(path)
                rows.append(out)
                fieldnames.update(out)
    ordered = sorted(fieldnames)
    with out_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=ordered)
        writer.writeheader()
        writer.writerows(rows)


def write_readme(args: argparse.Namespace) -> None:
    lines = [
        "# GRAPE Appendix Synthetic Bridge",
        "",
        "Optional appendix bridge using restricted scalar/logit GRAPE special-case",
        "controls inside the existing synthetic query-LM setup.  These rows are not",
        "full learned GRAPE-M, GRAPE-A, or GRAPE-AP reproductions.",
        "",
        "Default interpretation:",
        "",
        "- GRAPE-M/RoPE covers phase teachers.",
        "- GRAPE-A/ALiBi covers affine teachers.",
        "- GRAPE-M+A is tested as a direct-sum special-case control.",
        "- PJ-FJ/LC sectors are expected to recover distance-modulated phase more directly.",
        "",
        "Outputs:",
        "",
        "- `results.csv`: combined seed-level rows.",
        "- `<target>/results.csv`: per-target synthetic query-LM rows.",
    ]
    (args.out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


if __name__ == "__main__":
    main()

