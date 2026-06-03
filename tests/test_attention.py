import unittest

try:
    import torch
    from pjrope.attention import PJBias, PJCausalSelfAttention, causal_attention_mask
except Exception:  # pragma: no cover - exercised only without torch
    torch = None
    PJBias = None
    PJCausalSelfAttention = None
    causal_attention_mask = None


@unittest.skipIf(torch is None, "torch is not installed")
class AttentionTest(unittest.TestCase):
    def test_causal_attention_mask(self):
        inclusive = causal_attention_mask(4, include_self=True)
        strict = causal_attention_mask(4, include_self=False)
        self.assertTrue(bool(inclusive[0, 0]))
        self.assertFalse(bool(strict[0, 0]))
        self.assertTrue(bool(strict[3, 2]))
        self.assertFalse(bool(strict[2, 3]))

    def test_pj_bias_shape_and_sector_mask(self):
        pj_bias = PJBias(num_heads=2, max_order=2, train_length=16, use_fj=False, use_affine=True, use_lc=False)
        bias = pj_bias(8)
        self.assertEqual(tuple(bias.shape), (2, 8, 8))
        gates = pj_bias.sector_gates()
        self.assertTrue(torch.allclose(gates[:, 0], torch.zeros(2)))
        self.assertTrue(torch.allclose(gates[:, 1], torch.ones(2)))
        self.assertTrue(torch.allclose(gates[:, 2], torch.zeros(2)))
        self.assertEqual(float(bias[0, 0, 7]), 0.0)

    def test_attention_shape_strict_mask_and_grad(self):
        torch.manual_seed(0)
        pj_bias = PJBias(num_heads=2, max_order=1, train_length=16, use_fj=True, use_affine=True, use_lc=False)
        attn = PJCausalSelfAttention(embed_dim=8, num_heads=2, pj_bias=pj_bias, include_self=False)
        x = torch.randn(3, 6, 8)
        out, weights = attn(x, need_weights=True)
        self.assertEqual(tuple(out.shape), (3, 6, 8))
        self.assertEqual(tuple(weights.shape), (3, 2, 6, 6))
        diag = torch.diagonal(weights, dim1=-2, dim2=-1)
        self.assertTrue(torch.allclose(diag, torch.zeros_like(diag)))
        self.assertTrue(torch.allclose(weights[:, :, 0], torch.zeros_like(weights[:, :, 0])))

        loss = out.square().mean()
        loss.backward()
        self.assertIsNotNone(attn.out_proj.weight.grad)
        self.assertIsNotNone(pj_bias.gate_logits.grad)


if __name__ == "__main__":
    unittest.main()
