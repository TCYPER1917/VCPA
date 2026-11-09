"""VCPA-Inject: training-free runtime attention integration for pretrained LLMs.

Version 12.0.0-alpha ("Generation 12" — VCPA-Inject)

A reversible runtime attention adapter that lets VCPA (implemented here as
"chunked-exact + pooled-landmarks" attention, CEPL) be injected into an already
pretrained Hugging Face causal language model WITHOUT:

    - Retraining the model
    - Fine-tuning the model
    - Updating or modifying pretrained parameters
    - Weight conversion or checkpoint reconstruction
    - Permanent changes to the original architecture

Usage:

    from transformers import AutoModelForCausalLM
    from vcpa import VCPAInjector

    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
    injector = VCPAInjector(model)
    injector.inject()

    # normal inference and generation
    outputs = model.generate(...)

    injector.remove()   # restore original attention

IMPORTANT NOTES:

    - VCPA (CEPL) is an APPROXIMATION of exact self-attention. It is NOT equivalent.
      The approximation only kicks in for long contexts (seq_len > threshold); short
      sequences run through exact SDPA and are identical to the baseline.
    - During autoregressive DECODE (single-token steps with past_key_values) VCPA
      falls back to exact SDPA using the cached KV. Chunk-pooled landmarks are applied
      only during PREFILL (long-context forward passes).
    - All pretrained parameters are preserved unchanged; verified programmatically.

See the README.md for benchmarks, installation, and limitations.
"""

from __future__ import annotations

from .attention import CEPLAttention
from .injector import VCPAInjector, InjectResult
from .pooling import (
    build_pooler, MaxPooler, MeanPooler, Pooler, SoftmaxWeightedPooler, TopKPooler,
)
from .cache import VCPACacheMode, choose_mode
from .compatibility import get_model_arch, get_head_dim, enable_gqa
from .config import __version__

__all__ = [
    "VCPAInjector",
    "InjectResult",
    "CEPLAttention",
    "WhisperVCPAInjector",
    "build_pooler",
    "Pooler",
    "MeanPooler",
    "MaxPooler",
    "SoftmaxWeightedPooler",
    "TopKPooler",
    "VCPACacheMode",
    "choose_mode",
    "get_model_arch",
    "get_head_dim",
    "enable_gqa",
    "__version__",
]


def __getattr__(name: str):
    # Lazy: importing the Whisper adapter must not require whisper for Qwen users.
    if name == "WhisperVCPAInjector":
        from .whisper_inject import WhisperVCPAInjector
        return WhisperVCPAInjector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
