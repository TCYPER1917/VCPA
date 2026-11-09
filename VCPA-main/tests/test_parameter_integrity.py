"""Tests: parameter integrity and reversible injection.

Proves the non-negotiable constraints:
  1. All pretrained parameter values preserved
  2. Original q_proj/k_proj/v_proj/o_proj weights preserved (same objects)
  3. No trainable parameters introduced
  4. No silent model configuration change
  5. state_dict parameter values and key structure preserved
  6. Reversible injection and clean restoration
  7. No weight duplication/copies

Runs entirely offline on a tiny Qwen2 model (see tests/tiny_model.py).
"""

from __future__ import annotations

import unittest

import torch

from tests._helpers import ROOT  # noqa: F401  (adds repo root to sys.path)
from tests.tiny_model import make_tiny_qwen, make_pair

from vcpa import VCPAInjector, CEPLAttention


class TestParameterIntegrity(unittest.TestCase):
    """Verify that injection never touches pretrained parameters."""

    def test_snapshot_before_injection(self):
        model = make_tiny_qwen()
        injector = VCPAInjector(model)
        snap = injector.verify_and_store_pre_injection_snapshot(model)
        self.assertGreater(snap["keys"], 0)
        self.assertGreater(snap["size_bytes"], 0)

    def test_parameters_hash_unchanged_after_inject(self):
        model = make_tiny_qwen()
        injector = VCPAInjector(model)
        injector.verify_and_store_pre_injection_snapshot(model)
        injector.inject()
        try:
            ok, details = injector.verify_against_snapshot()
            self.assertTrue(ok, f"parameters changed after inject: {details}")
        finally:
            injector.remove()

    def test_parameters_hash_unchanged_after_remove(self):
        model = make_tiny_qwen()
        injector = VCPAInjector(model)
        injector.verify_and_store_pre_injection_snapshot(model)
        injector.inject()
        injector.remove()
        ok, details = injector.verify_against_snapshot()
        self.assertTrue(ok, f"parameters changed after inject+remove: {details}")

    def test_proj_weights_unchanged_by_value(self):
        """q/k/v/o projections must be the SAME module and tensor objects."""
        model = make_tiny_qwen()
        before_mods, before_tensors = {}, {}
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            mod = getattr(model.model.layers[0].self_attn, proj)
            before_mods[proj] = mod
            before_tensors[proj] = (mod.weight, getattr(mod, "bias", None))

        injector = VCPAInjector(model)
        injector.inject()
        try:
            attn = model.model.layers[0].self_attn
            self.assertIsInstance(attn, CEPLAttention)
            for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
                after_mod = getattr(attn, proj)
                self.assertIs(before_mods[proj], after_mod, f"{proj} module was replaced")
                self.assertIs(before_tensors[proj][0], after_mod.weight, f"{proj}.weight was copied")
                if before_tensors[proj][1] is not None:
                    self.assertIs(before_tensors[proj][1], after_mod.bias, f"{proj}.bias was copied")
                self.assertTrue(torch.equal(before_tensors[proj][0], after_mod.weight))
        finally:
            injector.remove()

    def test_no_trainable_parameters(self):
        model = make_tiny_qwen()
        n_trainable_before = sum(1 for p in model.parameters() if p.requires_grad)
        n_params_before = sum(p.numel() for p in model.parameters())

        injector = VCPAInjector(model)
        injector.inject()
        try:
            self.assertEqual(n_trainable_before, sum(1 for p in model.parameters() if p.requires_grad))
            self.assertEqual(n_params_before, sum(p.numel() for p in model.parameters()))
            for layer in model.model.layers:
                w = layer.self_attn
                self.assertIsInstance(w, CEPLAttention)
                # CEPLAttention owns no *new* parameters/submodules: its registries are
                # the original module's registries, shared by reference.
                self.assertIs(w._parameters, w.original_attn._parameters)
                self.assertIs(w._modules, w.original_attn._modules)
                # every parameter reachable from the wrapper is the original tensor
                orig_params = dict(w.original_attn.named_parameters())
                for name, p in w.named_parameters():
                    self.assertIs(p, orig_params[name])
        finally:
            injector.remove()

    def test_state_dict_key_structure_preserved(self):
        model = make_tiny_qwen()
        keys_before = list(model.state_dict().keys())
        values_before = {k: v.clone() for k, v in model.state_dict().items()}

        injector = VCPAInjector(model)
        injector.inject()
        try:
            self.assertEqual(list(model.state_dict().keys()), keys_before)
            for k, v in model.state_dict().items():
                self.assertTrue(torch.equal(v, values_before[k]), f"state_dict value changed: {k}")
        finally:
            injector.remove()

    def test_state_dict_reload_while_injected(self):
        """load_state_dict must still work while the model is injected."""
        model = make_tiny_qwen()
        sd = {k: v.clone() for k, v in model.state_dict().items()}
        injector = VCPAInjector(model)
        injector.inject()
        try:
            missing, unexpected = model.load_state_dict(sd, strict=False)
            self.assertEqual(missing, [])
            self.assertEqual(unexpected, [])
        finally:
            injector.remove()

    def test_reversible_clean_restoration(self):
        model = make_tiny_qwen()
        originals = [layer.self_attn for layer in model.model.layers]
        injector = VCPAInjector(model)
        injector.inject()
        self.assertTrue(injector.is_injected)
        injector.remove()
        self.assertFalse(injector.is_injected)
        for layer, orig in zip(model.model.layers, originals):
            self.assertIs(layer.self_attn, orig, "remove() did not restore the original module object")

    def test_repeated_injection_cycle(self):
        model = make_tiny_qwen()
        injector = VCPAInjector(model)
        injector.verify_and_store_pre_injection_snapshot(model)
        for _ in range(3):
            injector.inject()
            injector.remove()
        ok, details = injector.verify_against_snapshot()
        self.assertTrue(ok, f"parameters drifted after repeated cycles: {details}")

    def test_original_module_object_retrieved(self):
        model = make_tiny_qwen()
        before = {i: id(m.self_attn) for i, m in enumerate(model.model.layers)}
        injector = VCPAInjector(model)
        injector.inject()
        try:
            for i, module in injector.get_original():
                self.assertEqual(before[i], id(module))
        finally:
            injector.remove()

    def test_no_weight_duplication_in_model(self):
        model = make_tiny_qwen()
        n_before = sum(p.numel() for p in model.parameters())
        injector = VCPAInjector(model)
        injector.inject()
        try:
            self.assertEqual(n_before, sum(p.numel() for p in model.parameters()))
        finally:
            injector.remove()

    def test_config_not_silently_changed(self):
        model = make_tiny_qwen()
        cfg_before = {k: v for k, v in model.config.to_dict().items() if not k.startswith("_")}
        injector = VCPAInjector(model)
        injector.inject()
        try:
            cfg_after = {k: v for k, v in model.config.to_dict().items() if not k.startswith("_")}
            self.assertEqual(cfg_before, cfg_after, "model config changed during injection")
        finally:
            injector.remove()


class TestReversibleInference(unittest.TestCase):
    """inject -> forward -> remove -> forward must be numerically identical."""

    def test_before_and_after_injection_match(self):
        baseline, model = make_pair()
        inp = torch.randint(0, 200, (1, 64))

        with torch.inference_mode():
            out_before = model(inp).logits.detach().clone()

        injector = VCPAInjector(model, threshold=10**9)   # force the exact path
        injector.inject()
        with torch.inference_mode():
            out_during = model(inp).logits.detach().clone()
        injector.remove()

        with torch.inference_mode():
            out_after = model(inp).logits.detach().clone()

        with torch.inference_mode():
            out_ref = baseline(inp).logits.detach().clone()

        self.assertLess((out_before - out_during).abs().max().item(), 1e-6,
                        "exact path differs from baseline")
        self.assertLess((out_before - out_after).abs().max().item(), 1e-6,
                        "model changed permanently after remove()")
        self.assertLess((out_before - out_ref).abs().max().item(), 1e-6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
