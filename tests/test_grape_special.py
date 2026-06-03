import math
import unittest

try:
    import torch
    from pjrope.grape_special import (
        GrapeAALiBi,
        GrapeMA,
        GrapeMCanonicalRoPE,
        pj_fj_basis,
        r2_score,
        solve_projection,
        target_kernel,
    )
except Exception:  # pragma: no cover - exercised only without torch
    torch = None


@unittest.skipIf(torch is None, "torch is not installed")
class GrapeSpecialTest(unittest.TestCase):
    def test_grape_special_basis_shapes(self):
        lags = torch.arange(16, dtype=torch.float64)
        omega = 2.0 * math.pi * 3.0 / 64.0
        self.assertEqual(tuple(GrapeMCanonicalRoPE(64, omega).features(lags).shape), (16, 2))
        self.assertEqual(tuple(GrapeAALiBi(64).features(lags).shape), (16, 1))
        self.assertEqual(tuple(GrapeMA(64, omega).features(lags).shape), (16, 3))

    def test_grape_m_projects_phase_target(self):
        length = 128
        lags = torch.arange(length, dtype=torch.float64)
        omega = 2.0 * math.pi * 7.0 / float(length)
        target = target_kernel("phase", lags, length=length, omega=omega)
        _, pred = solve_projection(GrapeMCanonicalRoPE(length, omega).features(lags), target)
        self.assertGreater(float(r2_score(pred, target)), 0.999999)

    def test_grape_a_projects_affine_target(self):
        length = 128
        lags = torch.arange(length, dtype=torch.float64)
        omega = 2.0 * math.pi * 7.0 / float(length)
        target = target_kernel("affine", lags, length=length, omega=omega)
        _, pred = solve_projection(GrapeAALiBi(length).features(lags), target)
        self.assertGreater(float(r2_score(pred, target)), 0.999999)

    def test_grape_ma_has_nonzero_projection_residual_on_first_jet(self):
        length = 256
        lags = torch.arange(4 * length, dtype=torch.float64)
        omega = 2.0 * math.pi * 17.0 / float(length)
        target = target_kernel("first_jet", lags, length=length, omega=omega)

        _, grape_pred = solve_projection(GrapeMA(length, omega).features(lags), target)
        grape_mse = float((grape_pred - target).square().mean())

        _, pj_pred = solve_projection(pj_fj_basis(lags, length=length, omega=omega, max_order=1), target)
        pj_mse = float((pj_pred - target).square().mean())

        self.assertGreater(grape_mse, 1e-3)
        self.assertLess(pj_mse, 1e-20)


if __name__ == "__main__":
    unittest.main()

