"""Tests: causal masking, padding masks, GQA, RoPE, and numerical behaviour.

All tests run offline on a tiny Qwen2 model.  "identical" claims apply only to the
exact path (short sequences / padded batches / decode); the long-context CEPL path
is approximate and is only required to be causal and finite.
"""

from __future__ import annotations

import unittest

import torch

from tests._helpers import ROOT  # noqa: F401
from tests.tiny_model import make_tiny_qwen, make_pair, logits

from vcpa import VCPAInjector, CEPLAttention


class TestCausalMask(unittest.TestCase):
    """No token may ever attend to a future token."""

    @staticmethod
    def _bare_attn(chunk_size: int = 8) -> CEPLAttention:
        attn = CEPLAttention.__new__(CEPLAttention)
        attn.chunk_size = chunk_size
        attn._mask_cache = {}
        return attn

    def test_chunk_causal_mask(self):
        attn = self._bare_attn(8)
        n_chunks = 6
        m = attn._get_chunk_causal_mask(n_chunks, torch.device("cpu"), torch.float32)
        expected = torch.arange(n_chunks).view(-1, 1) >= torch.arange(n_chunks).view(1, -1)
        self.assertTrue(torch.equal(m, expected.float()))

    def test_token_chunk_causal_mask(self):
        attn = self._bare_attn(8)
        m = attn._get_token_chunk_causal_mask(24, 3, torch.device("cpu"), torch.float32)
        # token i may attend to chunk c iff c*8 <= i
        for i in range(24):
            for c in range(3):
                self.assertEqual(bool(m[i, c]), c * 8 <= i)

    def test_local_block_mask_is_lower_triangular(self):
        attn = self._bare_attn(8)
        m = attn._get_local_block_mask(24, 8, 8, torch.device("cpu"), torch.float32)
        self.assertEqual(m.shape, (24, 24))
        self.assertTrue(torch.all(torch.triu(m, diagonal=1) == 0))
        # block diagonal: no cross-chunk attention
        self.assertTrue(torch.all(m[:8, 8:] == 0))
        self.assertTrue(torch.all(m[8:16, :8] == 0))

    def test_future_tokens_do_not_influence_past_logits(self):
        """Changing tokens at the end must not change earlier logits (causal)."""
        baseline, model = make_pair()
        injector = VCPAInjector(model, chunk_size=16, local_window=32, threshold=16)
        injector.inject()
        try:
            x = torch.randint(0, 200, (1, 160))
            x2 = x.clone()
            x2[:, 128:] = torch.randint(0, 200, (1, 32))

            y1 = logits(model, x)
            y2 = logits(model, x2)

            # positions 0..127 must be unaffected by tokens 128..159
            err = (y1[:, :128] - y2[:, :128]).abs().max().item()
            self.assertLess(err, 1e-5, f"future tokens leaked into the past: {err}")

            # sanity: the changed positions *do* change
            err_late = (y1[:, 128:] - y2[:, 128:]).abs().max().item()
            self.assertGreater(err_late, 1e-6)
        finally:
            injector.remove()

    def test_padding_mask_does_not_leak_and_stays_on_cepl(self):
        """Padding is applied inside CEPL; it must not fall back, and must not leak.

        A mask of all ones is not padding, so it must match the unmasked CEPL path.
        A short sequence is still exact. A padded token must not move logits at
        positions that cannot see it.
        """
        baseline, model = make_pair()
        injector = VCPAInjector(model, chunk_size=16, local_window=16, threshold=16)
        injector.inject()
        try:
            attn = model.model.layers[0].self_attn
            calls = {"n": 0}
            orig = attn._unified_attention

            def counted(*a, **k):
                calls["n"] += 1
                return orig(*a, **k)

            attn._unified_attention = counted
            x = torch.randint(0, 200, (2, 64))
            ones = torch.ones_like(x)
            y_nomask = logits(model, x)
            y_ones = logits(model, x, attention_mask=ones)
            self.assertLess((y_nomask - y_ones).abs().max().item(), 1e-5)
            self.assertGreater(calls["n"], 0, "an all-ones mask must not force exact SDPA")

            mask = torch.ones_like(x)
            mask[:, :8] = 0
            x_pad = x.clone()
            x_pad[:, :8] = torch.randint(0, 200, (2, 8))
            y1 = logits(model, x, attention_mask=mask)
            y2 = logits(model, x_pad, attention_mask=mask)
            err = (y1[:, 8:] - y2[:, 8:]).abs().max().item()
            self.assertLess(err, 1e-5, f"padding tokens leaked into real positions: {err}")
            self.assertTrue(torch.isfinite(y1).all())

            # Below the threshold the exact path is unchanged, mask included.
            short = torch.randint(0, 200, (2, 12))
            short_mask = torch.ones_like(short)
            short_mask[1, :3] = 0
            y_base = logits(baseline, short, attention_mask=short_mask)
            y_cepl = logits(model, short, attention_mask=short_mask)
            self.assertLess((y_base - y_cepl).abs().max().item(), 1e-5)
        finally:
            injector.remove()

    def test_no_nan_on_long_sequence(self):
        _, model = make_pair()
        injector = VCPAInjector(model, chunk_size=16, local_window=32, threshold=16)
        injector.inject()
        try:
            x = torch.randint(0, 200, (1, 200))   # non-divisible by chunk_size
            y = logits(model, x)
            self.assertEqual(y.shape, (1, 200, model.config.vocab_size))
            self.assertTrue(torch.isfinite(y).all(), "non-finite logits on long sequence")
        finally:
            injector.remove()


    def test_cepl_path_is_actually_exercised(self):
        """Guard against silently falling back to exact attention in long prefills."""
        _, model = make_pair()
        injector = VCPAInjector(model, chunk_size=16, local_window=32, threshold=8)
        injector.inject()
        try:
            attn = model.model.layers[0].self_attn
            calls = {"n": 0}
            orig = attn._unified_attention

            def counted(*a, **k):
                calls["n"] += 1
                return orig(*a, **k)

            attn._unified_attention = counted
            x = torch.randint(0, 200, (1, 128))
            _ = logits(model, x)
            self.assertGreater(calls["n"], 0,
                               "the CEPL path was not used for a long prefill")
        finally:
            injector.remove()


    def test_blend_mode_is_causal(self):
        """The legacy convex-blend path must not leak future tokens either."""
        baseline, model = make_pair()
        injector = VCPAInjector(
            model, chunk_size=16, local_window=32, threshold=16, blend_mode="blend",
        )
        injector.inject()
        try:
            x = torch.randint(0, 200, (1, 160))
            x2 = x.clone()
            x2[:, 128:] = torch.randint(0, 200, (1, 32))
            y1 = logits(model, x)
            y2 = logits(model, x2)
            err = (y1[:, :128] - y2[:, :128]).abs().max().item()
            self.assertLess(err, 1e-5, f"future tokens leaked in blend mode: {err}")
            self.assertTrue(torch.isfinite(y1).all())
        finally:
            injector.remove()

    def test_local_only_blend_is_causal(self):
        """use_landmarks=False in blend mode must still be finite and causal."""
        _, model = make_pair()
        injector = VCPAInjector(
            model, chunk_size=16, local_window=32, threshold=16,
            blend_mode="blend", use_landmarks=False,
        )
        injector.inject()
        try:
            x = torch.randint(0, 200, (1, 128))
            x2 = x.clone()
            x2[:, 96:] = torch.randint(0, 200, (1, 32))
            y1 = logits(model, x)
            y2 = logits(model, x2)
            self.assertTrue(torch.isfinite(y1).all())
            err = (y1[:, :96] - y2[:, :96]).abs().max().item()
            self.assertLess(err, 1e-5, f"future tokens leaked in local-only blend: {err}")
        finally:
            injector.remove()

    def test_landmark_only_blend_is_causal(self):
        """use_local=False in blend mode falls back to exact attention for chunk 0."""
        _, model = make_pair()
        injector = VCPAInjector(
            model, chunk_size=16, local_window=32, threshold=16,
            blend_mode="blend", use_local=False,
        )
        injector.inject()
        try:
            x = torch.randint(0, 200, (1, 128))
            x2 = x.clone()
            x2[:, 96:] = torch.randint(0, 200, (1, 32))
            y1 = logits(model, x)
            y2 = logits(model, x2)
            self.assertTrue(torch.isfinite(y1).all(), "landmark-only blend produced non-finite logits")
            err = (y1[:, :96] - y2[:, :96]).abs().max().item()
            self.assertLess(err, 1e-5, f"future tokens leaked in landmark-only blend: {err}")
        finally:
            injector.remove()


class TestGQA(unittest.TestCase):
    def test_gqa_shapes_and_no_replication(self):
        _, model = make_pair(num_attention_heads=4, num_key_value_heads=2)
        injector = VCPAInjector(model, chunk_size=8, local_window=16, threshold=8)
        injector.inject()
        try:
            attn = model.model.layers[0].self_attn
            self.assertIsInstance(attn, CEPLAttention)
            self.assertEqual(attn._num_key_value_groups, 2)
            self.assertTrue(attn._gqa_enabled)
            self.assertEqual(attn.head_dim, 16)

            # the wrapped projections are the original module objects
            self.assertIs(attn.q_proj, attn.original_attn.q_proj)

            x = torch.randint(0, 200, (1, 64))
            y = logits(model, x)
            self.assertEqual(y.shape, (1, 64, model.config.vocab_size))
        finally:
            injector.remove()

    def test_mha_when_kv_heads_equal_q_heads(self):
        _, model = make_pair(num_attention_heads=4, num_key_value_heads=4)
        injector = VCPAInjector(model, chunk_size=8, local_window=16, threshold=8)
        injector.inject()
        try:
            attn = model.model.layers[0].self_attn
            self.assertEqual(attn._num_key_value_groups, 1)
            y = logits(model, torch.randint(0, 200, (1, 40)))
            self.assertTrue(torch.isfinite(y).all())
        finally:
            injector.remove()


class TestExactWindow(unittest.TestCase):
    """When every chunk fits in the exact window, CEPL must be mathematically exact."""

    def test_exact_window_matches_baseline(self):
        baseline, model = make_pair()
        # chunk 16, window 64 -> L = 4 exact chunks; seq 64 -> fully exact
        injector = VCPAInjector(model, chunk_size=16, local_window=64, threshold=8)
        injector.inject()
        try:
            x = torch.randint(0, 200, (1, 64))
            y_base = logits(baseline, x)
            y_cepl = logits(model, x)
            err = (y_base - y_cepl).abs().max().item()
            self.assertLess(err, 1e-5, f"exact-window mismatch: {err}")
        finally:
            injector.remove()

    def test_approximation_is_finite(self):
        baseline, model = make_pair()
        injector = VCPAInjector(model, chunk_size=16, local_window=32, threshold=8)
        injector.inject()
        try:
            x = torch.randint(0, 200, (1, 256))
            y_cepl = logits(model, x)
            self.assertTrue(torch.isfinite(y_cepl).all())
            # smoothed logits should stay in a plausible range
            self.assertLess(y_cepl.abs().max().item(), 1e4)
        finally:
            injector.remove()


class TestRoPE(unittest.TestCase):
    def test_rope_applied_for_different_lengths(self):
        baseline, model = make_pair()
        injector = VCPAInjector(model, threshold=10**9)   # exact path
        injector.inject()
        try:
            for sl in (8, 17, 64):
                x = torch.randint(0, 200, (1, sl))
                err = (logits(baseline, x) - logits(model, x)).abs().max().item()
                self.assertLess(err, 1e-5, f"RoPE mismatch at seq_len={sl}: {err}")
        finally:
            injector.remove()


if __name__ == "__main__":
    unittest.main(verbosity=2)
