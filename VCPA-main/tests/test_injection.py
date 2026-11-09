"""Tests: injection API, model compatibility, and configuration handling.

Runs offline on a tiny Qwen2 model.  Real checkpoints are covered separately by
``examples/inject_qwen.py`` and the Kaggle benchmark runner.
"""

from __future__ import annotations

import unittest

import torch

from tests._helpers import ROOT  # noqa: F401
from tests.tiny_model import make_tiny_qwen, make_pair, logits

from vcpa import VCPAInjector, CEPLAttention
from vcpa.config import DEFAULT_BLEND_MODE, DEFAULT_CHUNK_SIZE


class TestInjectionAPI(unittest.TestCase):
    def test_basic_inject_remove(self):
        model = make_tiny_qwen()
        injector = VCPAInjector(model)
        self.assertIs(injector.inject(), injector)
        for layer in model.model.layers:
            self.assertIsInstance(layer.self_attn, CEPLAttention)
        injector.remove()
        for layer in model.model.layers:
            self.assertNotIsInstance(layer.self_attn, CEPLAttention)

    def test_report(self):
        model = make_tiny_qwen()
        injector = VCPAInjector(model, chunk_size=16, local_window=32)
        injector.inject()
        report = injector.report()
        self.assertIn("INJECTED", report)
        self.assertIn("chunk_size=16", report)
        injector.remove()
        self.assertIn("REMOVED", injector.report())

    def test_custom_config_applied(self):
        model = make_tiny_qwen()
        injector = VCPAInjector(
            model, chunk_size=16, threshold=512,
            use_landmarks=False, use_local=False, blend_mode="blend",
        )
        injector.inject()
        try:
            for layer in model.model.layers:
                w = layer.self_attn
                self.assertEqual(w.chunk_size, 16)
                self.assertEqual(w.threshold, 512)
                self.assertFalse(w.use_landmarks)
                self.assertFalse(w.use_local)
                self.assertEqual(w.blend_mode, "blend")
        finally:
            injector.remove()

    def test_defaults(self):
        model = make_tiny_qwen()
        injector = VCPAInjector(model)
        self.assertEqual(injector.chunk_size, DEFAULT_CHUNK_SIZE)
        self.assertEqual(injector.blend_mode, DEFAULT_BLEND_MODE)

    def test_invalid_arguments(self):
        model = make_tiny_qwen()
        with self.assertRaises(TypeError):
            VCPAInjector("not a model")
        with self.assertRaises(ValueError):
            VCPAInjector(torch.nn.Module())
        with self.assertRaises(ValueError):
            VCPAInjector(model, chunk_size=0)
        with self.assertRaises(ValueError):
            VCPAInjector(model, local_window=0)
        with self.assertRaises(ValueError):
            VCPAInjector(model, threshold=0)
        with self.assertRaises(ValueError):
            VCPAInjector(model, landmark_weight=1.5)
        with self.assertRaises(ValueError):
            VCPAInjector(model, blend_mode="nope")
        with self.assertRaises(ValueError):
            # unified is a single softmax over both streams
            VCPAInjector(model, use_local=False)
        with self.assertRaises(ValueError):
            VCPAInjector(model, use_landmarks=False)

    def test_double_inject_is_safe(self):
        model = make_tiny_qwen()
        injector = VCPAInjector(model)
        injector.inject()
        injector.inject()   # must not double-wrap
        try:
            for layer in model.model.layers:
                self.assertIsInstance(layer.self_attn, CEPLAttention)
                self.assertIsInstance(layer.self_attn.original_attn, torch.nn.Module)
                self.assertNotIsInstance(layer.self_attn.original_attn, CEPLAttention)
        finally:
            injector.remove()

    def test_remove_then_forward_still_works(self):
        baseline, model = make_pair()
        injector = VCPAInjector(model, threshold=8)
        injector.inject()
        x = torch.randint(0, 200, (1, 40))
        _ = logits(model, x)
        injector.remove()
        err = (logits(baseline, x) - logits(model, x)).abs().max().item()
        self.assertLess(err, 1e-6)


class TestCompatibility(unittest.TestCase):
    def test_detection_and_head_dim(self):
        model = make_tiny_qwen(hidden_size=64, num_attention_heads=4, num_key_value_heads=2)
        injector = VCPAInjector(model)
        attn = injector._wrap(model.model.layers[0].self_attn)
        self.assertEqual(attn._arch, "qwen2")
        self.assertEqual(attn.head_dim, 16)
        self.assertEqual(attn._num_key_value_groups, 2)
        self.assertTrue(attn._gqa_enabled)

    def test_custom_head_dim(self):
        """Qwen2.5 can decouple head_dim from hidden_size/num_heads."""
        model = make_tiny_qwen(hidden_size=64, num_attention_heads=4,
                               num_key_value_heads=2, head_dim=8)
        injector = VCPAInjector(model)
        attn = injector._wrap(model.model.layers[0].self_attn)
        self.assertEqual(attn.head_dim, 8)

    def test_all_layers_injected(self):
        model = make_tiny_qwen(num_hidden_layers=4)
        injector = VCPAInjector(model)
        injector.inject()
        try:
            self.assertEqual(sum(injector.state_dict().values()), 4)
        finally:
            injector.remove()


class TestGeneration(unittest.TestCase):
    def test_generate_runs_after_inject(self):
        _, model = make_pair()
        injector = VCPAInjector(model, threshold=1024)  # stays exact for this length
        injector.inject()
        try:
            inp = torch.randint(0, 200, (1, 8))
            with torch.inference_mode():
                out = model.generate(inp, max_new_tokens=6, do_sample=False,
                                     pad_token_id=0, eos_token_id=None)
            self.assertEqual(out.shape[0], 1)
            self.assertTrue(out.shape[1] > inp.shape[1])
        finally:
            injector.remove()


if __name__ == "__main__":
    unittest.main(verbosity=2)
