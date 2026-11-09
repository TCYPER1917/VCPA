"""Offline tests for VCPA injection into OpenAI Whisper.

No checkpoint is downloaded. A tiny randomly-initialised Whisper is enough to
check parameter identity, reversibility, the exact fallback, causal masking,
and that the CEPL path is actually taken on long encoder sequences.
"""

from __future__ import annotations

import os
import sys
import unittest

import torch

from tests._helpers import ROOT  # noqa: F401

WHISPER_ROOT = os.environ.get("WHISPER_ROOT", r"C:\Users\K5\Documents\whisper-main")
if WHISPER_ROOT not in sys.path:
    sys.path.insert(0, WHISPER_ROOT)

from whisper.model import ModelDimensions, MultiHeadAttention, Whisper, disable_sdpa  # noqa: E402

from vcpa.whisper_inject import WhisperVCPAInjector  # noqa: E402


def tiny_whisper() -> Whisper:
    dims = ModelDimensions(
        n_mels=80,
        n_audio_ctx=64,
        n_audio_state=64,
        n_audio_head=4,
        n_audio_layer=2,
        n_vocab=200,
        n_text_ctx=64,
        n_text_state=64,
        n_text_head=4,
        n_text_layer=2,
    )
    model = Whisper(dims)
    model.eval()
    return model


def mel(batch: int = 1, n_ctx: int = 64) -> torch.Tensor:
    # conv2 stride 2 maps 2 * n_audio_ctx frames onto n_audio_ctx.
    return torch.randn(batch, 80, n_ctx * 2)


class TestWhisperInjection(unittest.TestCase):
    def test_inject_remove_preserves_parameters_and_keys(self):
        model = tiny_whisper()
        before_ptr = {n: p.data_ptr() for n, p in model.named_parameters()}
        before_keys = set(model.state_dict().keys())
        before_bytes = {
            n: p.detach().float().numpy().tobytes() for n, p in model.named_parameters()
        }

        injector = WhisperVCPAInjector(model, chunk_size=16, local_window=32, threshold=16)
        injector.inject()
        self.assertTrue(injector.is_injected)
        self.assertTrue(all(getattr(b.attn, "is_vcpa", False) for b in model.encoder.blocks))
        self.assertTrue(all(getattr(b.attn, "is_vcpa", False) for b in model.decoder.blocks))
        self.assertTrue(all(getattr(b.cross_attn, "is_vcpa", False) for b in model.decoder.blocks))
        self.assertEqual(set(model.state_dict().keys()), before_keys)
        self.assertEqual({n: p.data_ptr() for n, p in model.named_parameters()}, before_ptr)
        ok, details = injector.verify_parameter_integrity()
        self.assertTrue(ok, details)

        # load_state_dict must still address the same tensors while injected.
        model.load_state_dict(model.state_dict())
        self.assertEqual({n: p.data_ptr() for n, p in model.named_parameters()}, before_ptr)

        injector.remove()
        self.assertFalse(injector.is_injected)
        for block in model.encoder.blocks:
            self.assertIs(type(block.attn), MultiHeadAttention)
        self.assertEqual({n: p.data_ptr() for n, p in model.named_parameters()}, before_ptr)
        after_bytes = {
            n: p.detach().float().numpy().tobytes() for n, p in model.named_parameters()
        }
        self.assertEqual(before_bytes, after_bytes)
        self.assertIn("REMOVED", injector.report())

    def test_exact_fallback_matches_original(self):
        torch.manual_seed(0)
        model = tiny_whisper()
        tokens = torch.randint(0, 200, (1, 8))
        audio = mel()
        with torch.no_grad():
            baseline = model(audio, tokens).clone()
        WhisperVCPAInjector(model, threshold=10**9).inject()
        with torch.no_grad():
            injected = model(audio, tokens)
        self.assertTrue(torch.equal(baseline, injected))
        self.assertEqual(sum(b.attn.cepl_calls for b in model.encoder.blocks), 0)

    def test_encoder_cepl_is_used_and_bidirectional(self):
        torch.manual_seed(1)
        model = tiny_whisper()
        injector = WhisperVCPAInjector(
            model, chunk_size=16, local_window=32, threshold=16
        ).inject()
        audio = mel()
        with torch.no_grad():
            encoded = model.encoder(audio)
        self.assertEqual(encoded.shape, (1, 64, 64))
        self.assertTrue(torch.isfinite(encoded).all())
        self.assertEqual(sum(b.attn.cepl_calls for b in model.encoder.blocks), 2)

        block = model.encoder.blocks[0]
        x1 = torch.randn(1, 64, 64)
        x2 = x1.clone()
        x2[:, -1] = torch.randn(64)
        with torch.no_grad():
            y1 = block.attn(block.attn_ln(x1))[0]
            y2 = block.attn(block.attn_ln(x2))[0]
        # A causal mask would hide the last frame from position 0. Bidirectional
        # CEPL must let it through, via the landmark of that chunk.
        self.assertFalse(torch.allclose(y1[:, 0], y2[:, 0], atol=1e-6))
        self.assertIn("bidirectional", injector.report())

    def test_causal_decoder_does_not_leak_future_tokens(self):
        torch.manual_seed(2)
        model = tiny_whisper()
        WhisperVCPAInjector(model, chunk_size=8, local_window=16, threshold=8).inject()
        xa = torch.randn(1, 64, 64)
        tokens_a = torch.randint(0, 200, (1, 32))
        tokens_b = tokens_a.clone()
        tokens_b[:, -1] = (tokens_b[:, -1] + 17) % 200
        with torch.no_grad():
            logits_a = model.decoder(tokens_a, xa)
            logits_b = model.decoder(tokens_b, xa)
        self.assertTrue(torch.allclose(logits_a[:, :-1], logits_b[:, :-1], atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.isfinite(logits_a).all())
        self.assertGreater(sum(b.attn.cepl_calls for b in model.decoder.blocks), 0)

    def test_causal_inside_exact_window_matches_sdpa(self):
        torch.manual_seed(3)
        model = tiny_whisper()
        ref = tiny_whisper()
        ref.load_state_dict(model.state_dict())
        # N == local_window > threshold, so the kernel runs, but every past token
        # is inside the exact window and landmarks are fully masked.
        WhisperVCPAInjector(
            model, chunk_size=8, local_window=32, threshold=8, approximate_cross=False
        ).inject()
        xa = torch.randn(1, 64, 64)
        tokens = torch.randint(0, 200, (1, 32))
        with torch.no_grad():
            approx = model.decoder(tokens, xa)
            exact = ref.decoder(tokens, xa)
        self.assertTrue(torch.allclose(approx, exact, atol=1e-4, rtol=1e-4))

    def test_kv_cache_exact_decode_matches_full_forward(self):
        torch.manual_seed(4)
        model = tiny_whisper()
        WhisperVCPAInjector(model, threshold=10**9).inject()
        xa = torch.randn(1, 64, 64)
        tokens = torch.randint(0, 200, (1, 6))
        with torch.no_grad():
            full = model.decoder(tokens, xa)
            cache, hooks = model.install_kv_cache_hooks()
            step = None
            for t in range(tokens.shape[1]):
                step = model.decoder(tokens[:, t:t + 1], xa, kv_cache=cache)
            for hook in hooks:
                hook.remove()
        self.assertTrue(torch.allclose(step[:, -1], full[:, -1], atol=1e-5, rtol=1e-5))

    def test_long_decode_runs_and_matches_prefill_last_token(self):
        torch.manual_seed(5)
        model = tiny_whisper()
        WhisperVCPAInjector(
            model, chunk_size=8, local_window=16, threshold=8, approximate_cross=False
        ).inject()
        xa = torch.randn(1, 64, 64)
        tokens = torch.randint(0, 200, (1, 32))
        with torch.no_grad():
            full = model.decoder(tokens, xa)
            cache, hooks = model.install_kv_cache_hooks()
            step = None
            for t in range(tokens.shape[1]):
                step = model.decoder(tokens[:, t:t + 1], xa, kv_cache=cache)
            for hook in hooks:
                hook.remove()
        self.assertTrue(torch.isfinite(step).all())
        # Chunk-aligned length: the last decode step sees the same key set as
        # the last prefill query. Cross-attention is exact so it does not add
        # a second approximation on top of the causal comparison.
        self.assertTrue(torch.allclose(step[:, -1], full[:, -1], atol=1e-4, rtol=1e-4))
        self.assertGreater(sum(b.attn.cepl_calls for b in model.decoder.blocks), 0)

    def test_cross_attention_and_word_timestamp_qk(self):
        torch.manual_seed(6)
        model = tiny_whisper()
        WhisperVCPAInjector(model, chunk_size=16, local_window=32, threshold=16).inject()
        text = torch.randn(2, 4, 64)
        audio = torch.randn(1, 64, 64)
        block = model.decoder.blocks[0]
        with torch.no_grad():
            out, qk = block.cross_attn(text, audio)
        self.assertEqual(out.shape, text.shape)
        self.assertIsNone(qk)
        self.assertGreater(block.cross_attn.cepl_calls, 0)
        with torch.no_grad(), disable_sdpa():
            out_exact, qk_exact = block.cross_attn(text, audio)
        self.assertIsNotNone(qk_exact)
        self.assertEqual(qk_exact.shape[0], 2)
        self.assertEqual(qk_exact.shape[1], 4)  # heads
        self.assertEqual(qk_exact.shape[-1], 64)  # audio frames
        self.assertTrue(torch.isfinite(out_exact).all())

    def test_remainder_tokens_are_causal_and_finite(self):
        torch.manual_seed(7)
        model = tiny_whisper()
        WhisperVCPAInjector(model, chunk_size=8, local_window=16, threshold=8).inject()
        xa = torch.randn(1, 64, 64)
        tokens_a = torch.randint(0, 200, (1, 35))  # 35 % 8 != 0
        tokens_b = tokens_a.clone()
        tokens_b[:, -1] = (tokens_b[:, -1] + 3) % 200
        with torch.no_grad():
            logits_a = model.decoder(tokens_a, xa)
            logits_b = model.decoder(tokens_b, xa)
        self.assertTrue(torch.isfinite(logits_a).all())
        self.assertTrue(torch.allclose(logits_a[:, :-1], logits_b[:, :-1], atol=1e-5, rtol=1e-5))

    def test_full_forward_fp16_activations(self):
        # Whisper keeps parameters in fp32 and casts them to the activation dtype
        # inside Linear / Conv. model.half() is not the inference path.
        torch.manual_seed(8)
        model = tiny_whisper()
        WhisperVCPAInjector(model, chunk_size=16, local_window=32, threshold=16).inject()
        audio = mel().half()
        tokens = torch.randint(0, 200, (1, 12))
        with torch.no_grad():
            logits = model(audio, tokens)
        self.assertEqual(logits.dtype, torch.float32)
        self.assertTrue(torch.isfinite(logits).all())
        self.assertGreater(sum(b.attn.cepl_calls for b in model.encoder.blocks), 0)

    def test_repeated_cycles(self):
        model = tiny_whisper()
        injector = WhisperVCPAInjector(model, chunk_size=16, threshold=16)
        for _ in range(3):
            injector.inject()
            self.assertTrue(injector.is_injected)
            injector.remove()
            self.assertFalse(injector.is_injected)
            self.assertIs(type(model.encoder.blocks[0].attn), MultiHeadAttention)


if __name__ == "__main__":
    unittest.main()
