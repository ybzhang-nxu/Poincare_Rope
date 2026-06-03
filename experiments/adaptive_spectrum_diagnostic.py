"""Adaptive spectrum diagnostic for PJ-RoPE scalar kernels.

This is a small Phase B learner: it fits a three-sector PJ kernel to synthetic
teacher kernels, then reports sector gates, order mass, functional energy, and
extrapolation error. It is intentionally light enough to run as a smoke test on
CPU, while using CUDA when available.
"""

from __future__ import annotations

import argparse
import json
import math
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
        "/home/riven/JordanKac/.venv/bin/python experiments/adaptive_spectrum_diagnostic.py"
    ) from exc

from pjrope import torch_backend as tb


SECTOR_NAMES = ("fj", "affine", "lc")


@dataclass(frozen=True)
class DiagnosticResult:
    target: str
    restart: int
    train_mse: float
    train_r2: float
    eval_mse: float
    eval_r2: float
    gates: dict[str, float]
    affine_slope: float
    fj_mass: list[float]
    lc_mass: list[float]
    normalized_order_mass: list[float]
    functional_energy: list[float]
    leave_one_order_out_delta_mse: list[float]


class AdaptivePJKernel(torch.nn.Module):
    def __init__(self, *, length: int, max_order: int, omega: float, dtype: torch.dtype) -> None:
        super().__init__()
        self.length = float(length)
        self.max_order = max_order
        self.omega = float(omega)

        order_count = max_order + 1
        self.gate_logits = torch.nn.Parameter(torch.tensor([0.0, -0.2, -0.2], dtype=dtype))
        self.fj_alpha_logits = torch.nn.Parameter(torch.zeros(order_count, dtype=dtype))
        self.lc_alpha_logits = torch.nn.Parameter(torch.zeros(order_count, dtype=dtype))

        self.fj_zeta_re = torch.nn.Parameter(0.05 * torch.randn(order_count, dtype=dtype))
        self.fj_zeta_im = torch.nn.Parameter(0.05 * torch.randn(order_count, dtype=dtype))
        self.lc_zeta_re = torch.nn.Parameter(0.05 * torch.randn(order_count, dtype=dtype))
        self.lc_zeta_im = torch.nn.Parameter(0.05 * torch.randn(order_count, dtype=dtype))
        self.affine_slope = torch.nn.Parameter(torch.tensor(0.0, dtype=dtype))
        self.raw_damping = torch.nn.Parameter(torch.tensor(-2.0, dtype=dtype))

    def forward(self, d: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        order = torch.arange(self.max_order + 1, device=d.device, dtype=d.dtype)
        factorial = torch.tensor(
            [math.factorial(int(r)) for r in range(self.max_order + 1)],
            device=d.device,
            dtype=d.dtype,
        )

        gates = torch.softmax(self.gate_logits, dim=0)
        fj_alpha = torch.softmax(self.fj_alpha_logits, dim=0)
        lc_alpha = torch.softmax(self.lc_alpha_logits, dim=0)

        x = d / self.length
        phase = self.omega * d
        cos = torch.cos(phase)
        sin = torch.sin(phase)
        damping = torch.nn.functional.softplus(self.raw_damping)

        fj_scale = x[None, :].pow(order[:, None]) / factorial[:, None]
        fj_env = torch.exp(-damping * x)[None, :]
        fj_orders = fj_scale * fj_env * fj_alpha[:, None] * (
            self.fj_zeta_re[:, None] * cos[None, :] - self.fj_zeta_im[:, None] * sin[None, :]
        )

        affine = self.affine_slope * x

        phi = tb.phi_l(d, self.length)
        beta = tb.beta_l(d, self.length)
        lc_phase = self.omega * phi
        lc_cos = torch.cos(lc_phase)
        lc_sin = torch.sin(lc_phase)
        lc_scale = beta[None, :].pow(order[:, None])
        lc_orders = lc_scale * lc_alpha[:, None] * (
            self.lc_zeta_re[:, None] * lc_cos[None, :] - self.lc_zeta_im[:, None] * lc_sin[None, :]
        )

        pred = gates[0] * fj_orders.sum(dim=0) + gates[1] * affine + gates[2] * lc_orders.sum(dim=0)
        components = {
            "gates": gates,
            "fj_alpha": fj_alpha,
            "lc_alpha": lc_alpha,
            "fj_orders": fj_orders,
            "lc_orders": lc_orders,
            "affine": affine,
        }
        return pred, components

    def parameter_masses(self) -> tuple[torch.Tensor, torch.Tensor]:
        gates = torch.softmax(self.gate_logits, dim=0)
        fj_alpha = torch.softmax(self.fj_alpha_logits, dim=0)
        lc_alpha = torch.softmax(self.lc_alpha_logits, dim=0)
        fj_zeta = torch.sqrt(self.fj_zeta_re.square() + self.fj_zeta_im.square())
        lc_zeta = torch.sqrt(self.lc_zeta_re.square() + self.lc_zeta_im.square())
        return gates[0] * fj_alpha * fj_zeta, gates[2] * lc_alpha * lc_zeta


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype = torch.float64 if args.float64 else torch.float32
    omega = pick_omega(args.omega_cycles, args.train_len)
    targets = select_targets(make_targets(args.train_len, omega), args.targets)

    results = []
    for target_name, target_fn in targets.items():
        best = None
        for restart in range(args.restarts):
            result = fit_target(target_name, target_fn, args, restart, device, dtype, omega)
            if best is None or result_metric(result, args.selection_metric) < result_metric(best, args.selection_metric):
                best = result
        if best is None:
            raise RuntimeError("no diagnostic run completed")
        results.append(best)

    print_results(results, args.train_len, args.eval_len)
    write_outputs(results, args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-len", type=int, default=1024)
    parser.add_argument("--eval-len", type=int, default=4096)
    parser.add_argument("--omega-cycles", type=float, default=17.0)
    parser.add_argument("--targets", default="all")
    parser.add_argument("--max-order", type=int, default=3)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=3e-2)
    parser.add_argument("--restarts", type=int, default=2)
    parser.add_argument("--order-init-cycle", action="store_true")
    parser.add_argument("--selection-metric", choices=["train_mse", "eval_mse"], default="train_mse")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--float64", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=Path("runs/adaptive_spectrum_diagnostic"))
    return parser.parse_args()


def fit_target(
    target_name: str,
    target_fn,
    args: argparse.Namespace,
    restart: int,
    device: torch.device,
    dtype: torch.dtype,
    omega: float,
) -> DiagnosticResult:
    torch.manual_seed(args.seed + 7919 * restart)
    model = AdaptivePJKernel(length=args.train_len, max_order=args.max_order, omega=omega, dtype=dtype).to(device)
    if getattr(args, "order_init_cycle", False):
        apply_order_biased_init(model, restart % (args.max_order + 1))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_d = torch.arange(args.train_len, device=device, dtype=dtype)
    y_train = target_fn(train_d)

    for _ in range(args.steps):
        pred, _ = model(train_d)
        loss = (pred - y_train).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        eval_d = torch.arange(args.eval_len, device=device, dtype=dtype)
        y_eval = target_fn(eval_d)
        pred_train, _ = model(train_d)
        pred_eval, eval_components = model(eval_d)
        train_mse = (pred_train - y_train).square().mean()
        train_r2 = r2_score(pred_train, y_train)
        eval_mse = (pred_eval - y_eval).square().mean()
        eval_r2 = r2_score(pred_eval, y_eval)

        fj_mass, lc_mass = model.parameter_masses()
        total_mass = fj_mass + lc_mass
        normalized_mass = total_mass / total_mass.sum().clamp_min(torch.finfo(dtype).eps)

        gates = eval_components["gates"]
        order_components = gates[0] * eval_components["fj_orders"] + gates[2] * eval_components["lc_orders"]
        functional_energy = tb.functional_energy(order_components.unsqueeze(0)).squeeze(0)
        loo_delta = leave_one_order_out_delta_mse(y_eval, pred_eval, order_components)

    return DiagnosticResult(
        target=target_name,
        restart=restart,
        train_mse=to_float(train_mse),
        train_r2=to_float(train_r2),
        eval_mse=to_float(eval_mse),
        eval_r2=to_float(eval_r2),
        gates={name: to_float(value) for name, value in zip(SECTOR_NAMES, gates)},
        affine_slope=to_float(model.affine_slope),
        fj_mass=to_float_list(fj_mass),
        lc_mass=to_float_list(lc_mass),
        normalized_order_mass=to_float_list(normalized_mass),
        functional_energy=to_float_list(functional_energy),
        leave_one_order_out_delta_mse=to_float_list(loo_delta),
    )


def pick_omega(cycles: float, train_len: int) -> float:
    return float(2.0 * math.pi * cycles / train_len)


def result_metric(result: DiagnosticResult, metric: str) -> float:
    if metric == "train_mse":
        return result.train_mse
    if metric == "eval_mse":
        return result.eval_mse
    raise ValueError(f"unknown selection metric: {metric}")


def apply_order_biased_init(model: AdaptivePJKernel, order: int) -> None:
    with torch.no_grad():
        model.fj_alpha_logits.fill_(-2.0)
        model.lc_alpha_logits.fill_(-2.0)
        model.fj_alpha_logits[order].fill_(2.0)
        model.lc_alpha_logits[order].fill_(2.0)


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


def select_targets(targets: dict, raw: str) -> dict:
    if raw == "all":
        return targets
    names = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [name for name in names if name not in targets]
    if unknown:
        raise SystemExit(f"unknown target(s): {', '.join(unknown)}")
    return {name: targets[name] for name in names}


def r2_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    residual = (pred - target).square().sum()
    total = (target - target.mean()).square().sum()
    return 1.0 - residual / (total + eps)


def leave_one_order_out_delta_mse(
    target: torch.Tensor,
    pred: torch.Tensor,
    order_components: torch.Tensor,
) -> torch.Tensor:
    base = (pred - target).square().mean()
    values = []
    for order in range(order_components.shape[0]):
        removed = pred - order_components[order]
        values.append((removed - target).square().mean() - base)
    return torch.stack(values)


def print_results(results: list[DiagnosticResult], train_len: int, eval_len: int) -> None:
    print(f"Adaptive spectrum diagnostic summary @ L={train_len}, T={eval_len}")
    print("target              train_mse   train_r2  eval_mse    eval_r2  gates(fj/a/lc)       top_order")
    print("-" * 104)
    for row in results:
        gate_text = f"{row.gates['fj']:.2f}/{row.gates['affine']:.2f}/{row.gates['lc']:.2f}"
        top_order = max(range(len(row.normalized_order_mass)), key=row.normalized_order_mass.__getitem__)
        print(
            f"{row.target:<19} {row.train_mse:9.2e} {row.train_r2:8.4f} "
            f"{row.eval_mse:9.2e} {row.eval_r2:8.4f}  {gate_text:<18} r={top_order}"
        )


def write_outputs(results: list[DiagnosticResult], args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / "results.json"
    config = vars(args).copy()
    config["out_dir"] = str(args.out_dir)
    with path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "config": config,
                "results": [asdict(result) for result in results],
            },
            file,
            indent=2,
        )
    print(f"\nWrote diagnostic JSON: {path}")


def to_float(value: torch.Tensor) -> float:
    return float(value.detach().cpu())


def to_float_list(value: torch.Tensor) -> list[float]:
    return [float(v) for v in value.detach().cpu().tolist()]


if __name__ == "__main__":
    main()
