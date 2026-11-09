"""Reversible runtime injection layer for VCPA-Inject.

Target API (illustrative; actual API is implemented below):

    model = AutoModelForCausalLM.from_pretrained(model_id)
    injector = VCPAInjector(model)
    injector.inject()

    # normal inference / generation
    outputs = model.generate(...)

    injector.remove()

The injector:
- wraps each attention module with CEPLAttention WITHOUT touching any weights
- preserves the original attention module object (accessible via injector.get_original())
- guarantees parameter integrity with programmatic verification
- records whether injection succeeded per layer for partial-failure robustness
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .attention import CEPLAttention
from .cache import VCPACacheMode
from .config import (
    DEFAULT_BLEND_LANDMARK_WEIGHT, DEFAULT_BLEND_MODE, DEFAULT_CHUNK_SIZE,
    DEFAULT_LOCAL_WINDOW, DEFAULT_NUM_LANDMARKS, DEFAULT_THRESHOLD,
    DEFAULT_USE_LANDMARKS, DEFAULT_USE_LOCAL,
)
from .pooling import POOL_MULTIVEC


@dataclass
class InjectResult:
    """Per-layer results of an inject / remove cycle."""

    injected: bool
    layer_idx: int
    module_path: str
    error: Optional[str] = None


class VCPAInjector:
    """Runtime VCPA (CEPL) injector for pretrained Hugging Face causal language models."""

    def __init__(
        self,
        model: nn.Module,
        *,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        local_window: int = DEFAULT_LOCAL_WINDOW,
        threshold: int = DEFAULT_THRESHOLD,
        use_landmarks: bool = DEFAULT_USE_LANDMARKS,
        use_local: bool = DEFAULT_USE_LOCAL,
        landmark_weight: float = DEFAULT_BLEND_LANDMARK_WEIGHT,
        pool_type: str = POOL_MULTIVEC,
        num_landmarks: int = DEFAULT_NUM_LANDMARKS,
        blend_mode: str = DEFAULT_BLEND_MODE,
        auto_gqa: bool = True,
    ):
        if not isinstance(model, nn.Module):
            raise TypeError("model must be a torch.nn.Module (e.g. AutoModelForCausalLM).")
        if not hasattr(model, "model"):
            raise ValueError(
                "model must be a transformer-style model exposing 'model' (e.g. 'model.model.layers'). "
                f"Got type {type(model).__name__}."
            )

        self.model = model
        # Validate configuration up-front so errors surface at construction time,
        # not halfway through a 24-layer injection.
        if int(chunk_size) <= 0:
            raise ValueError("chunk_size must be a positive integer")
        if int(local_window) <= 0:
            raise ValueError("local_window must be a positive integer")
        if int(threshold) <= 0:
            raise ValueError("threshold must be a positive integer")
        if not 0.0 <= float(landmark_weight) <= 1.0:
            raise ValueError("landmark_weight must be in [0, 1]")
        if blend_mode not in ("unified", "blend"):
            raise ValueError("blend_mode must be 'unified' or 'blend'")
        if blend_mode == "unified" and not (use_landmarks and use_local):
            raise ValueError(
                "blend_mode='unified' requires use_local=True and use_landmarks=True "
                "(it is a single softmax over both streams). Use blend_mode='blend' "
                "to disable one stream."
            )

        self.chunk_size = int(chunk_size)
        self.local_window = int(local_window)
        self.threshold = int(threshold)
        self.use_landmarks = bool(use_landmarks)
        self.use_local = bool(use_local)
        self.landmark_weight = float(landmark_weight)
        self.pool_type = pool_type
        self.num_landmarks = max(1, int(num_landmarks))
        self.blend_mode = blend_mode
        self.auto_gqa = bool(auto_gqa)

        self._injected: Dict[int, bool] = {}       # layer_idx -> whether injected
        self._results: List[InjectResult] = []     # full per-layer report
        self._cache_mode = VCPACacheMode()

    # ------------------------------------------------------------------
    # Injection
    # ------------------------------------------------------------------

    def inject(self) -> "VCPAInjector":
        """Wrap every attention module with CEPLAttention.

        Returns self for chaining. No parameters are modified, copied, or allocated
        in the model (CEPLAttention only holds a reference to the original modules).
        """
        # Guard against double-injection
        if any(self._injected.values()):
            # If already partially injected, finish the cycle safely
            self.remove()

        layers = self._iter_layers()
        for layer_idx, layer in layers:
            attn = self._find_attn(layer)
            if attn is None:
                self._results.append(InjectResult(
                    injected=False, layer_idx=layer_idx, module_path=self._module_path(layer),
                    error="no attention module found",
                ))
                continue
            try:
                wrapped = self._wrap(attn)
            except Exception as exc:  # noqa: BLE001
                self._results.append(InjectResult(
                    injected=False, layer_idx=layer_idx, module_path=self._module_path(layer),
                    error=str(exc),
                ))
                continue
            try:
                self._set_attn(layer, wrapped)
            except Exception as exc:  # noqa: BLE001
                self._results.append(InjectResult(
                    injected=False, layer_idx=layer_idx, module_path=self._module_path(layer),
                    error=str(exc),
                ))
                continue
            self._injected[layer_idx] = True
            self._results.append(InjectResult(
                injected=True, layer_idx=layer_idx, module_path=self._module_path(layer),
            ))

        self._cache_mode.use_cepl = True
        return self

    def _iter_layers(self):
        """Yield (layer_idx, layer) tuples across model.model.layers.

        Supports list-like layers and models with a .layers attribute.
        """
        for i, layer in enumerate(getattr(self.model, "model", nn.Module()).layers):
            yield i, layer

    def _find_attn(self, layer: nn.Module) -> Optional[nn.Module]:
        """Find the attention submodule inside a transformer layer.

        Handles self_attn (Llama/Mistral/Qwen2) and, as a fallback, attn.
        """
        if hasattr(layer, "self_attn"):
            return layer.self_attn
        if hasattr(layer, "attn"):
            return layer.attn
        return None

    def _wrap(self, attn: nn.Module) -> CEPLAttention:
        """Wrap a single attention module into CEPLAttention."""
        if isinstance(attn, CEPLAttention):
            # Already wrapped by this injector (should be guarded, but be safe)
            return attn
        return CEPLAttention(
            original_attn=attn,
            chunk_size=self.chunk_size,
            local_window=self.local_window,
            threshold=self.threshold,
            use_landmarks=self.use_landmarks,
            use_local=self.use_local,
            landmark_weight=self.landmark_weight,
            pool_type=self.pool_type,
            blend_mode=self.blend_mode,
            num_landmarks=self.num_landmarks,
        )

    def _set_attn(self, layer: nn.Module, wrapped: CEPLAttention) -> None:
        if hasattr(layer, "self_attn"):
            layer.self_attn = wrapped
        elif hasattr(layer, "attn"):
            layer.attn = wrapped

    def _module_path(self, layer: nn.Module) -> str:
        return f"model.layers[{getattr(layer, 'layer_idx', '?')}] self_attn"

    # ------------------------------------------------------------------
    # Removal (reversible)
    # ------------------------------------------------------------------

    def remove(self) -> "VCPAInjector":
        """Restore every wrapped attention module to its original pretrained state.

        The original module objects are returned (not recreated), so their weights are
        exactly the pretrained weights. Returns self for chaining.
        """
        layers = list(self._iter_layers())
        for layer_idx, layer in layers:
            attn = self._find_attn(layer)
            if attn is not None and isinstance(attn, CEPLAttention):
                original = attn.remove()
                self._set_attn(layer, original)
                self._injected[layer_idx] = False
            elif attn is not None and not isinstance(attn, CEPLAttention):
                # Nothing to restore; module was never wrapped (or already restored)
                self._injected[layer_idx] = False
        self._cache_mode.use_cepl = False
        self._cache_mode._step = 0
        return self

    def state_dict(self) -> Dict[int, bool]:
        """Return a snapshot of the injection state: layer_idx -> injected."""
        return dict(self._injected)

    @property
    def is_injected(self) -> bool:
        """True if at least one attention module is currently wrapped."""
        return any(self._injected.values())

    def get_original(self, layer_idx: Optional[int] = None) -> List[Tuple[int, nn.Module]]:
        """Return the original (pretrained) attention modules.

        When ``layer_idx`` is given, return only that layer's original module.
        The modules are the same objects referenced by the wrapped CEPLAttention,
        i.e. no copies, no weight duplication.
        """
        if layer_idx is not None:
            layer = getattr(self.model, "model").layers[layer_idx]
            attn = self._find_attn(layer)
            if isinstance(attn, CEPLAttention):
                return [(layer_idx, attn.remove())]
            return []
        out = []
        for i, layer in enumerate(getattr(self.model, "model").layers):
            attn = self._find_attn(layer)
            if isinstance(attn, CEPLAttention):
                out.append((i, attn.remove()))
        return out

    # ------------------------------------------------------------------
    # Verification (parameter integrity)
    # ------------------------------------------------------------------

    def verify_parameter_integrity(
        self,
        *,
        check_qkv_o: bool = True,
        check_state_dict_keys: bool = True,
        atol: float = 0.0,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Verify that all pretrained parameters are unchanged after injection.

        Two levels of checking:

        * structural: every wrapped projection is the *same module object* as the
          original module's projection (no copies), and the state_dict surface is
          non-empty/stable;
        * value-level: if ``verify_and_store_pre_injection_snapshot`` was called before
          injection, every parameter is compared byte-for-byte against that snapshot.

        Without a pre-injection snapshot the value-level check is impossible, so only
        the structural guarantee is reported.
        """
        details: Dict[str, Any] = {"checks": [], "equal": 0, "unequal": 0, "errors": []}

        if check_state_dict_keys:
            keys = {k for k in self.model.state_dict().keys()}
            details["checks"].append({"name": "state_dict_keys_present", "keys": len(keys)})
            details["equal"] += 1

        n_total = 0
        n_equal = 0
        n_unequal = 0
        for i, layer in enumerate(getattr(self.model, "model").layers):
            attn = self._find_attn(layer)
            if isinstance(attn, CEPLAttention):
                for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
                    if hasattr(attn, proj):
                        p = getattr(attn, proj)
                        original = getattr(attn.original_attn, proj, None)
                        if check_qkv_o:
                            n_total += 1
                            if p is original:
                                n_equal += 1
                            else:
                                n_unequal += 1
        details["proj_refs_preserved"] = {"total": n_total, "equal": n_equal, "unequal": n_unequal}

        ok = n_unequal == 0
        if hasattr(self, "_pre_injection_snapshot"):
            snap_ok, snap_details = self.verify_against_snapshot()
            details["snapshot_check"] = snap_details
            ok = ok and snap_ok
        else:
            details["snapshot_check"] = {"note": "no pre-injection snapshot stored"}

        details["all_equal"] = ok
        return ok, details

    def verify_and_store_pre_injection_snapshot(self, model: nn.Module) -> Dict[str, Any]:
        """Hash all parameter values before injection, so post-injection checks are meaningful."""
        snap = {}
        for name, p in model.named_parameters():
            snap[name] = p.detach().cpu().float().numpy().tobytes()
        self._pre_injection_snapshot = snap  # noqa: WPS432
        self._snapshot_keys = set(snap.keys())
        return {"keys": len(snap), "size_bytes": sum(len(v) for v in snap.values())}

    def verify_against_snapshot(self) -> Tuple[bool, Dict[str, Any]]:
        """Compare current parameters against the stored pre-injection snapshot."""
        if not hasattr(self, "_pre_injection_snapshot"):
            return False, {"error": "no pre-injection snapshot stored; call verify_and_store_pre_injection_snapshot first"}
        snap = self._pre_injection_snapshot
        keys = {k for k, _ in self.model.named_parameters()}
        new_snap = {n: p.detach().cpu().float().numpy().tobytes() for n, p in self.model.named_parameters()}

        equal = 0
        mismatches = []
        for k, v in new_snap.items():
            if k not in snap:
                mismatches.append(("missing_in_snapshot", k))
                continue
            if snap[k] != v:
                mismatches.append(("value_changed", k))
            else:
                equal += 1
        added = [k for k in snap if k not in new_snap]
        ok = not mismatches and not added
        return ok, {"equal": equal, "mismatches": mismatches, "added": added}

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def report(self) -> str:
        """Return a human-readable summary of the last inject / remove cycle."""
        lines = [f"VCPAInjector report: {sum(self._injected.values())}/{len(self._injected)} layers injected"]
        lines.append(f"  config: chunk_size={self.chunk_size}, local_window={self.local_window}, "
                     f"threshold={self.threshold}, landmarks={self.use_landmarks}, "
                     f"local={self.use_local}, blend={self.landmark_weight:.2f}, "
                     f"blend_mode={self.blend_mode}, pool={self.pool_type}")
        status = "INJECTED" if any(self._injected.values()) else "REMOVED"
        lines.append(f"  overall status: {status}")
        if self._results:
            failed = [r for r in self._results if not r.injected]
            if failed:
                lines.append(f"  failures: {len(failed)}")
                for r in failed:
                    lines.append(f"    layer {r.layer_idx}: {r.error}")
        return "\n".join(lines)
