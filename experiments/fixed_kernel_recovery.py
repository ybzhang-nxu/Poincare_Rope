"""Fixed kernel basis recovery experiment.

This implements the low-cost Phase B probe from the experiment plan. It fits a
linear readout over fixed scalar basis features on ``d < L`` and evaluates
interpolation/extrapolation MSE and R^2.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import torch
except Exception as exc:  # pragma: no cover - command-line guard
    raise SystemExit(
        "This experiment requires PyTorch. Try:\n"
        "/home/riven/JordanKac/.venv/bin/python experiments/fixed_kernel_recovery.py"
    ) from exc

from pjrope import torch_backend as tb


@dataclass(frozen=True)
class FitResult:
    omega_cycles: float
    target: str
    basis: str
    eval_len: int
    train_mse: float
    mse: float
    r2: float
    condition_number: float


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype = torch.float64 if args.float64 else torch.float32
    torch.manual_seed(args.seed)

    train_d = torch.arange(args.train_len, device=device, dtype=dtype)
    eval_lengths = parse_int_list(args.eval_lengths)
    omega_cycles_values = parse_float_list(args.omega_cycles)

    rows: list[FitResult] = []
    for omega_cycles in omega_cycles_values:
        omega = pick_omega(omega_cycles, args.train_len)
        targets = make_targets(args.train_len, omega)
        basis_fns = make_basis_fns(args.train_len, omega, args.max_order)

        for target_name, target_fn in targets.items():
            y_train = target_fn(train_d)
            for basis_name, basis_fn in basis_fns.items():
                x_train_raw = basis_fn(train_d)
                x_train, stats = standardize(x_train_raw)
                design = append_intercept(x_train)
                cond = condition_number(design)
                weights = solve_linear_readout(x_train, y_train)
                pred_train = design @ weights
                train_mse = float(mse(pred_train, y_train).detach().cpu())

                for eval_len in eval_lengths:
                    d_eval = torch.arange(eval_len, device=device, dtype=dtype)
                    y_eval = target_fn(d_eval)
                    x_eval = apply_standardize(basis_fn(d_eval), stats)
                    pred = append_intercept(x_eval) @ weights
                    rows.append(
                        FitResult(
                            omega_cycles=omega_cycles,
                            target=target_name,
                            basis=basis_name,
                            eval_len=eval_len,
                            train_mse=train_mse,
                            mse=float(mse(pred, y_eval).detach().cpu()),
                            r2=float(r2_score(pred, y_eval).detach().cpu()),
                            condition_number=cond,
                        )
                    )

    print_results(rows, eval_lengths, omega_cycles_values)
    write_outputs(rows, args.out_dir, eval_lengths)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-len", type=int, default=1024)
    parser.add_argument("--eval-lengths", default="1024,2048,4096,8192")
    parser.add_argument("--omega-cycles", default="5,17,61")
    parser.add_argument("--max-order", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--float64", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=Path("runs/fixed_kernel_recovery"))
    return parser.parse_args()


def pick_omega(cycles: float, train_len: int) -> float:
    # Avoid near-zero and near-Nyquist by default: cycles=17 over L=1024 is a clean mid frequency.
    return float(2.0 * torch.pi * cycles / train_len)


def parse_int_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in str(raw).split(",") if item.strip()]


def parse_float_list(raw: str | float) -> list[float]:
    return [float(item.strip()) for item in str(raw).split(",") if item.strip()]


def make_targets(length: int, omega: float):
    def x(d):
        return d / float(length)

    def lc_core(d):
        phi = tb.phi_l(d, float(length))
        beta = tb.beta_l(d, float(length))
        return torch.cos(omega * phi) * (1.0 + 0.4 * beta + 0.2 * beta.square())

    def lc_affine(d):
        return lc_core(d) - 0.25 * tb.eta_l(d, float(length))

    return {
        "phase": lambda d: torch.cos(omega * d),
        "linear": lambda d: x(d),
        "first_jet": lambda d: x(d) * torch.cos(omega * d),
        "second_jet": lambda d: x(d).square() * torch.exp(-0.1 * x(d)) * torch.cos(omega * d),
        "third_jet": lambda d: x(d).pow(3) * torch.exp(-0.1 * x(d)) * torch.cos(omega * d),
        "recency_weak_jet": lambda d: -0.4 * x(d) + 0.15 * x(d).square() * torch.cos(omega * d),
        "lc_core": lc_core,
        "lc_affine": lc_affine,
    }


def make_basis_fns(length: int, omega: float, max_order: int):
    def x(d):
        return d / float(length)

    def trig_orders(d, order: int, *, damped: bool = False):
        features = []
        envelope = torch.exp(-0.1 * x(d)) if damped else torch.ones_like(d)
        for r in range(order + 1):
            scale = x(d).pow(r)
            features.append(scale * envelope * torch.cos(omega * d))
            features.append(scale * envelope * torch.sin(omega * d))
        return torch.stack(features, dim=-1)

    def lc_basis(d):
        phi = tb.phi_l(d, float(length))
        beta = tb.beta_l(d, float(length))
        features = []
        for r in range(max_order + 1):
            scale = beta.pow(r)
            features.append(scale * torch.cos(omega * phi))
            features.append(scale * torch.sin(omega * phi))
        return torch.stack(features, dim=-1)

    basis = {
        "rope": lambda d: torch.stack([torch.cos(omega * d), torch.sin(omega * d)], dim=-1),
        "alibi": lambda d: x(d)[:, None],
        "alibi_lc": lambda d: tb.eta_l(d, float(length))[:, None],
        "rope_alibi": lambda d: torch.cat(
            [torch.cos(omega * d)[:, None], torch.sin(omega * d)[:, None], x(d)[:, None]],
            dim=-1,
        ),
    }
    for order in range(1, max_order + 1):
        basis[f"jordan_R{order}"] = lambda d, order=order: trig_orders(d, order)
        basis[f"scaled_R{order}"] = lambda d, order=order: trig_orders(d, order, damped=True)
        basis[f"scaled_R{order}_affine"] = lambda d, order=order: torch.cat(
            [trig_orders(d, order, damped=True), x(d)[:, None]],
            dim=-1,
        )
    basis[f"lc_R{max_order}"] = lc_basis
    basis[f"lc_R{max_order}_affine"] = lambda d: torch.cat(
        [lc_basis(d), tb.eta_l(d, float(length))[:, None]],
        dim=-1,
    )
    return basis


def standardize(x: torch.Tensor, eps: float = 1e-8):
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True).clamp_min(eps)
    return (x - mean) / std, (mean, std)


def apply_standardize(x: torch.Tensor, stats):
    mean, std = stats
    return (x - mean) / std


def solve_linear_readout(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.linalg.lstsq(append_intercept(x), y[:, None]).solution


def append_intercept(x: torch.Tensor) -> torch.Tensor:
    return torch.cat([x, torch.ones(x.shape[0], 1, device=x.device, dtype=x.dtype)], dim=-1)


def condition_number(design: torch.Tensor, eps: float = 1e-12) -> float:
    singular_values = torch.linalg.svdvals(design.to(torch.float64))
    value = singular_values.max() / singular_values.min().clamp_min(eps)
    return float(value.detach().cpu())


def mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = pred.squeeze(-1)
    return (pred - target).square().mean()


def r2_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    pred = pred.squeeze(-1)
    residual = (pred - target).square().sum()
    total = (target - target.mean()).square().sum()
    return 1.0 - residual / (total + eps)


def print_results(rows: list[FitResult], eval_lengths: list[int], omega_cycles_values: list[float]) -> None:
    final_len = max(eval_lengths)
    print(f"Fixed kernel recovery best basis @ T={final_len}")
    print("omega   target              basis              mse          r2       cond")
    print("-" * 78)
    targets = sorted({row.target for row in rows})
    for omega_cycles in omega_cycles_values:
        for target in targets:
            subset = [
                row
                for row in rows
                if row.omega_cycles == omega_cycles and row.target == target and row.eval_len == final_len
            ]
            best = min(subset, key=lambda r: r.mse)
            print(
                f"{omega_cycles:<7g} {target:<19} {best.basis:<17} "
                f"{best.mse:10.4e} {best.r2:8.5f} {best.condition_number:9.2e}"
            )


def write_outputs(rows: list[FitResult], out_dir: Path, eval_lengths: list[int]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "results.csv"
    json_path = out_dir / "summary.json"

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "omega_cycles",
                "target",
                "basis",
                "eval_len",
                "train_mse",
                "mse",
                "r2",
                "condition_number",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    final_len = max(eval_lengths)
    summary = {}
    min_len = min(eval_lengths)
    for omega_cycles in sorted({r.omega_cycles for r in rows}):
        omega_summary = {}
        for target in sorted({r.target for r in rows}):
            subset = [
                r for r in rows if r.omega_cycles == omega_cycles and r.target == target and r.eval_len == final_len
            ]
            best = min(subset, key=lambda r: r.mse)
            start = next(
                r
                for r in rows
                if r.omega_cycles == omega_cycles
                and r.target == target
                and r.basis == best.basis
                and r.eval_len == min_len
            )
            best_dict = asdict(best)
            best_dict["extrapolation_ratio"] = best.mse / (start.mse + 1e-12)
            omega_summary[target] = best_dict
        summary[f"{omega_cycles:g}"] = omega_summary

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "final_eval_len": final_len,
                "best_by_frequency_and_target": summary,
                "rows": [asdict(row) for row in rows],
            },
            file,
            indent=2,
        )

    print(f"\nWrote CSV: {csv_path}")
    print(f"Wrote summary: {json_path}")


if __name__ == "__main__":
    main()
