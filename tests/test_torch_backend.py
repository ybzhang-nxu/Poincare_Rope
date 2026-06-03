import math
import unittest

try:
    import torch
    from pjrope import torch_backend as tb
except Exception:  # pragma: no cover - exercised only without torch
    torch = None
    tb = None


@unittest.skipIf(torch is None, "torch is not installed")
class TorchBackendTest(unittest.TestCase):
    def test_fj_matches_scalar_rope_character(self):
        lags, mask = tb.causal_lag_matrix(8)
        omega = torch.tensor([0.17])
        alpha = torch.tensor([[1.0]])
        zeta_re = torch.tensor([[1.0]])
        zeta_im = torch.tensor([[0.0]])
        comp = tb.fourier_jet_components(
            lags,
            omega,
            alpha,
            zeta_re,
            zeta_im,
            length=128.0,
        )
        self.assertEqual(tuple(comp.shape), (1, 1, 8, 8))
        self.assertAlmostEqual(float(comp[0, 0, 7, 0]), math.cos(0.17 * 7), places=6)
        gates = torch.tensor([[1.0, 0.0, 0.0]])
        bias = tb.combine_pj_bias(gates, fj_components=comp, causal_mask=mask)
        self.assertEqual(tuple(bias.shape), (1, 8, 8))
        self.assertEqual(float(bias[0, 0, 7]), 0.0)

    def test_affine_and_lc_shapes(self):
        lags, mask = tb.causal_lag_matrix(16)
        affine = tb.affine_bias(lags, torch.tensor([0.5, 1.0]), length=64.0)
        self.assertEqual(tuple(affine.shape), (2, 16, 16))
        self.assertLess(float(affine[1, 15, 0]), 0.0)

        omega = torch.tensor([0.11, 0.13])
        alpha = torch.softmax(torch.ones(2, 3), dim=-1)
        zeta_re = torch.ones(2, 3)
        zeta_im = torch.zeros(2, 3)
        lc = tb.lightcone_components(lags, omega, alpha, zeta_re, zeta_im, length=64.0)
        self.assertEqual(tuple(lc.shape), (2, 3, 16, 16))
        gates = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.5, 0.5]])
        bias = tb.combine_pj_bias(gates, affine=affine, lc_components=lc, causal_mask=mask)
        self.assertEqual(tuple(bias.shape), (2, 16, 16))

    def test_diagnostics(self):
        alpha = torch.tensor([[0.25, 0.75]])
        zeta_re = torch.tensor([[2.0, 4.0]])
        zeta_im = torch.zeros_like(zeta_re)
        mass = tb.parameter_effective_mass(torch.tensor([0.5]), alpha, zeta_re, zeta_im)
        self.assertGreater(float(mass[0, 1]), float(mass[0, 0]))

        components = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        energy = tb.functional_energy(components)
        self.assertEqual(tuple(energy.shape), (1, 2))
        loo = tb.leave_one_order_out_mse(torch.tensor([1.0, 1.0]), components)
        self.assertEqual(tuple(loo.shape), (1, 2))

    def test_exact_pj_rotary_relative_scores(self):
        query_base = torch.tensor([0.2, -0.3, 0.5, 0.7, -0.1, 0.4, 0.8, -0.2], dtype=torch.float64)
        key_base = torch.tensor([-0.6, 0.1, 0.3, -0.4, 0.9, 0.2, -0.7, 0.5], dtype=torch.float64)
        query = query_base.repeat(6).reshape(1, 1, 6, 8)
        key = key_base.repeat(6).reshape(1, 1, 6, 8)
        query_rot, key_rot = tb.apply_exact_pj_rotary(query, key, base=100.0, train_length=4)
        scores = query_rot[0, 0] @ key_rot[0, 0].T
        self.assertAlmostEqual(float(scores[3, 1]), float(scores[4, 2]), places=10)
        self.assertAlmostEqual(float(scores[5, 2]), float(scores[4, 1]), places=10)
        self.assertEqual(tuple(query_rot.shape), tuple(query.shape))

    @unittest.skipIf(torch is None or not torch.cuda.is_available(), "cuda is not available")
    def test_cuda_smoke(self):
        lags, _ = tb.causal_lag_matrix(32, device="cuda")
        omega = torch.tensor([0.07], device="cuda")
        alpha = torch.tensor([[0.5, 0.5]], device="cuda")
        zeta_re = torch.ones(1, 2, device="cuda")
        zeta_im = torch.zeros(1, 2, device="cuda")
        comp = tb.fourier_jet_components(lags, omega, alpha, zeta_re, zeta_im, length=128.0)
        self.assertEqual(comp.device.type, "cuda")
        self.assertTrue(torch.isfinite(comp).all())


if __name__ == "__main__":
    unittest.main()
