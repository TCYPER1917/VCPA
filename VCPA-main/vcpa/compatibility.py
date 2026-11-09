"""Architecture-specific compatibility helpers for VCPA-Inject.

Provides:
- ``get_model_arch(attn_module)`` -> model family name
- ``get_head_dim(attn_module, config)`` -> per-model head_dim
- ``rope_func(config)`` -> (cos_fn, sin_fn) callable pair, or a ``transformers``-style
  ``(cos, sin)`` factory depending on how the model applies RoPE.
- ``enable_gqa(attn_module)`` -> bool
- ``num_key_value_groups(attn_module)`` -> int
"""

from __future__ import annotations

from typing import Callable, Tuple

import torch

ROPE_REGISTRY: dict[str, Callable] = {}


def register_rope(name: str) -> Callable:
    def decorator(fn):
        ROPE_REGISTRY[name] = fn
        return fn
    return decorator


@register_rope("qwen2")
def qwen2_rope(cos: torch.Tensor, sin: torch.Tensor, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings as used by Qwen2."""
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
    return apply_rotary_pos_emb(q, k, cos, sin)


@register_rope("llama")
def llama_rope(cos: torch.Tensor, sin: torch.Tensor, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings as used by Llama architectures."""
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
    return apply_rotary_pos_emb(q, k, cos, sin)


@register_rope("mistral")
def mistral_rope(cos: torch.Tensor, sin: torch.Tensor, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings as used by Mistral (same contract as Llama)."""
    from transformers.models.mistral.modeling_mistral import apply_rotary_pos_emb
    return apply_rotary_pos_emb(q, k, cos, sin)


def get_model_arch(attn_module) -> str:
    """Detect the model family from the attention module / config."""
    config = getattr(attn_module, "config", None)
    if config is None:
        return "unknown"
    name = getattr(config, "model_type", None)
    if not name:
        return "unknown"
    return str(name).lower().replace("_", "")


def get_head_dim(attn_module, config) -> int:
    """Derive head_dim for the given module/config without hardcoding 64."""
    if hasattr(attn_module, "head_dim"):
        return int(attn_module.head_dim)
    head_dim = getattr(config, "head_dim", None)
    if head_dim:
        return int(head_dim)
    if hasattr(config, "hidden_size") and hasattr(config, "num_attention_heads"):
        return int(config.hidden_size // config.num_attention_heads)
    raise AttributeError(
        "Cannot determine head_dim from attention module or config. "
        "Set head_dim in config or add a 'head_dim' attribute to the attention module."
    )


def rope_factory(attn_module):
    """Return a callable ``apply_rope(cos, sin, q, k)`` suitable for this model.

    Returns None if RoPE could not be resolved (e.g. unknown architecture or no rotary embedding).
    """
    arch = get_model_arch(attn_module)
    fn = ROPE_REGISTRY.get(arch)
    return fn


def num_key_value_heads(attn_module, config) -> int:
    if hasattr(attn_module, "num_key_value_heads"):
        return int(attn_module.num_key_value_heads)
    if hasattr(config, "num_key_value_heads"):
        return int(config.num_key_value_heads)
    # default: MQA-like (1 KV head) if nothing else
    num_q = config.num_attention_heads if hasattr(config, "num_attention_heads") else 1
    return max(1, num_q)


def num_key_value_groups(attn_module, config) -> int:
    if hasattr(attn_module, "num_key_value_groups"):
        return int(attn_module.num_key_value_groups)
    n_q = config.num_attention_heads if hasattr(config, "num_attention_heads") else 1
    n_kv = num_key_value_heads(attn_module, config)
    groups = n_q // n_kv
    assert n_q % n_kv == 0, f"GQA/MQA requires num_attention_heads divisible by num_key_value_heads; " \
                            f"got {n_q} vs {n_kv}"
    return groups


def enable_gqa(attn_module, config) -> bool:
    """Return True when native GQA support in SDPA should be used."""
    return num_key_value_groups(attn_module, config) > 1
