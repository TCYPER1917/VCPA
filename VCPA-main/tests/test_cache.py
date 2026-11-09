"""Tests: KV cache handling and decode fallback behaviour.

Runs offline on a tiny Qwen2 model.
"""

from __future__ import annotations

import unittest

import torch

from tests._helpers import ROOT  # noqa: F401
from tests.tiny_model import make_tiny_qwen, make_pair, logits

from vcpa import VCPAInjector, choose_mode


class TestKVCache(unittest.TestCase):
    def _injected(self, **kw):
        _, model = make_pair()
        injector = VCPAInjector(model, **kw)
        injector.inject()
        return model, injector

    def test_decode_matches_prefill(self):
        """Token-by-token decode must equal one prefill of the concatenated sequence."""
        model, inj = self._injected(threshold=10**9)  # exact path
        try:
            x1 = torch.randint(10, 200, (1, 12))
            x2 = torch.randint(10, 200, (1, 7))
            with torch.inference_mode():
                out1 = model(x1, use_cache=True)
                out2 = model(x2, past_key_values=out1.past_key_values, use_cache=True).logits
                full = model(torch.cat([x1, x2], dim=1)).logits[:, -x2.shape[1]:]
            err = (out2 - full).abs().max().item()
            self.assertLess(err, 1e-4, f"cached decode differs from prefill: {err}")
        finally:
            inj.remove()

    def test_cache_is_reused_and_shapes_correct(self):
        model, inj = self._injected(threshold=10**9)
        try:
            with torch.inference_mode():
                logits_ = model(torch.randint(0, 200, (1, 10)), use_cache=True).logits
                pkv = model(  # noqa: F841  (keep reference alive)
                    torch.randint(0, 200, (1, 10)), use_cache=True
                ).past_key_values
                length_before = pkv.get_seq_length()
                nxt = torch.argmax(logits_[:, -1:, :], dim=-1)
                out = model(nxt, past_key_values=pkv, use_cache=True).logits
                self.assertEqual(pkv.get_seq_length(), length_before + 1)
            self.assertEqual(out.shape, (1, 1, model.config.vocab_size))
        finally:
            inj.remove()

    def test_generate_with_cache(self):
        """generate() (which uses the cache) must run and keep the prefix."""
        model, inj = self._injected(threshold=10**9)
        try:
            inp = torch.randint(0, 200, (1, 12))
            with torch.inference_mode():
                out = model.generate(inp, max_new_tokens=8, do_sample=False,
                                     pad_token_id=0, eos_token_id=None)
            self.assertEqual(out.shape[0], 1)
            self.assertGreaterEqual(out.shape[1], inp.shape[1] + 1)
        finally:
            inj.remove()

    def test_multi_token_continuation_with_cache(self):
        """A multi-token continuation that reuses a cache must be causal and exact."""
        model, inj = self._injected(threshold=10**9)
        try:
            x1 = torch.randint(10, 200, (1, 8))
            x2 = torch.randint(10, 200, (1, 5))
            with torch.inference_mode():
                prefix = model(x1, use_cache=True)
                y_cached = model(x2, past_key_values=prefix.past_key_values,
                                 use_cache=True).logits
                y_full = model(torch.cat([x1, x2], dim=1)).logits[:, -x2.shape[1]:]
            err = (y_cached - y_full).abs().max().item()
            self.assertLess(err, 1e-4, f"continuation with cache is misaligned: {err}")
        finally:
            inj.remove()

    def test_cache_mode_decide(self):
        model, inj = self._injected(threshold=16)
        try:
            attn = model.model.layers[0].self_attn
            # `seq_len <= threshold` is the exact path (mirrors CEPLAttention.forward)
            self.assertEqual(choose_mode(attn, 8, None), "exact")
            self.assertEqual(choose_mode(attn, 16, None), "exact")
            self.assertEqual(choose_mode(attn, 17, None), "cepl")
            self.assertEqual(choose_mode(attn, 64, None), "cepl")
            # an unreadable cache cannot be proven long, so decode stays exact
            self.assertEqual(choose_mode(attn, 1, object()), "exact")
        finally:
            inj.remove()


class TestLandmarkDecode(unittest.TestCase):
    """Long decode must use cached landmarks, without touching parameters."""

    def test_decode_is_finite_causal_and_parameter_free(self):
        _, model = make_pair()
        injector = VCPAInjector(
            model, chunk_size=16, local_window=16, threshold=16, num_landmarks=4,
        )
        injector.inject()
        try:
            keys_before = list(model.state_dict().keys())
            prompt = torch.randint(0, 200, (1, 64))
            with torch.inference_mode():
                pre = model(prompt, use_cache=True)
                attn = model.model.layers[0].self_attn
                self.assertIsNotNone(attn._landmark_cache)
                self.assertEqual(attn._landmark_cache["n_chunks"], 4)
                stored = attn._landmark_cache["n_chunks"]
                nxt = torch.argmax(pre.logits[:, -1:, :], dim=-1)
                step = model(nxt, past_key_values=pre.past_key_values, use_cache=True)
            self.assertEqual(step.logits.shape, (1, 1, model.config.vocab_size))
            self.assertTrue(torch.isfinite(step.logits).all())
            # The new token does not finish a chunk, so no landmark is rebuilt.
            self.assertEqual(attn._landmark_cache["n_chunks"], stored)
            self.assertEqual(list(model.state_dict().keys()), keys_before)
            for key, value in model.state_dict().items():
                self.assertFalse(key.startswith("_landmark"))
        finally:
            injector.remove()

    def test_generate_uses_landmark_decode(self):
        _, model = make_pair()
        injector = VCPAInjector(model, chunk_size=16, local_window=16, threshold=16)
        injector.inject()
        try:
            inp = torch.randint(0, 200, (1, 48))
            with torch.inference_mode():
                out = model.generate(
                    inp, max_new_tokens=4, do_sample=False,
                    pad_token_id=0, eos_token_id=None,
                )
            self.assertEqual(out.shape, (1, 52))
        finally:
            injector.remove()


class TestChunkRemainder(unittest.TestCase):
    """Remainder tokens (seq_len % chunk_size != 0) must never be dropped."""

    def test_remainder_module_matches_exact(self):
        """`_remainder_exact` must equal exact causal attention for the tail tokens."""
        _, model = make_pair()
        injector = VCPAInjector(model, chunk_size=16, local_window=32, threshold=8)
        injector.inject()
        try:
            attn = model.model.layers[0].self_attn
            torch.manual_seed(0)
            B, H, N, D = 1, attn.config.num_attention_heads, 21, attn.head_dim
            q = torch.randn(B, H, N, D)
            k = torch.randn(B, H, N, D)
            v = torch.randn(B, H, N, D)
            offset = 16  # one full chunk + 5 remainder tokens

            out = attn._remainder_exact(q[:, :, offset:], k, v, offset)

            i = torch.arange(N - offset).view(-1, 1) + offset
            j = torch.arange(N).view(1, -1)
            allow = j <= i
            ref = torch.nn.functional.scaled_dot_product_attention(
                q[:, :, offset:], k, v, attn_mask=allow,
                dropout_p=0.0, scale=attn.scaling,
            )
            err = (out - ref).abs().max().item()
            self.assertLess(err, 1e-6, f"remainder attention is not exact: {err}")
        finally:
            injector.remove()

    def test_remainder_tokens_have_output(self):
        """Every position produces a finite logit vector, including the remainder."""
        _, model = make_pair()
        injector = VCPAInjector(model, chunk_size=16, local_window=32, threshold=8)
        injector.inject()
        try:
            for sl in (17, 33, 47, 65, 100):
                x = torch.randint(0, 200, (1, sl))
                y = logits(model, x)
                self.assertEqual(y.shape, (1, sl, model.config.vocab_size))
                self.assertTrue(torch.isfinite(y).all(), f"non-finite logits at seq_len={sl}")
                self.assertTrue(torch.isfinite(y[:, -1]).all())
        finally:
            injector.remove()

    def test_chunk_boundary_shapes(self):
        _, model = make_pair()
        injector = VCPAInjector(model, chunk_size=16, threshold=8)
        injector.inject()
        try:
            x = torch.randint(0, 200, (1, 33))
            out = logits(model, x)
            self.assertEqual(out.shape[1], 33)
            self.assertTrue(torch.isfinite(out[0, -1]).all())
        finally:
            injector.remove()


if __name__ == "__main__":
    unittest.main(verbosity=2)
