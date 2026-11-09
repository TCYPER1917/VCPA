"""Tiny, randomly-initialised Qwen2 model used by the offline test-suite.

The tests must run in seconds and without network access.  Real checkpoints
(Qwen2.5-0.5B and larger) are exercised separately on GPU via ``benchmarks/``
and ``examples/inject_qwen.py``.

The tiny model keeps the architectural properties that matter for VCPA:
  * GQA (num_attention_heads != num_key_value_heads)
  * RoPE applied inside the attention module
  * KV cache managed inside the attention module
  * head_dim != hidden_size / num_attention_heads when configured explicitly
"""

from __future__ import annotations

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM


def make_tiny_qwen(
    *,
    vocab_size: int = 256,
    hidden_size: int = 64,
    num_hidden_layers: int = 3,
    num_attention_heads: int = 4,
    num_key_value_heads: int = 2,
    head_dim: int | None = None,
    max_position_embeddings: int = 4096,
    seed: int = 0,
) -> Qwen2ForCausalLM:
    """Build a small, deterministic Qwen2 model for tests."""
    torch.manual_seed(seed)
    kwargs = dict(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 2,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        max_position_embeddings=max_position_embeddings,
        tie_word_embeddings=False,
        attention_dropout=0.0,
        use_cache=True,
        bos_token_id=0,
        eos_token_id=1,
    )
    if head_dim is not None:
        kwargs["head_dim"] = head_dim
    cfg = Qwen2Config(**kwargs)
    model = Qwen2ForCausalLM._from_config(cfg, attn_implementation="sdpa")
    model.eval()
    return model


def make_pair(**kwargs) -> tuple[Qwen2ForCausalLM, Qwen2ForCausalLM]:
    """Return two models with identical weights: (baseline, to_be_injected)."""
    seed = kwargs.pop("seed", 0)
    return make_tiny_qwen(seed=seed, **kwargs), make_tiny_qwen(seed=seed, **kwargs)


def logits(model, input_ids, **kw) -> torch.Tensor:
    with torch.inference_mode():
        return model(input_ids, **kw).logits.detach()
