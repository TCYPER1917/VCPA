"""End-to-end smoke test for VCPA-Inject.

Default: uses a tiny randomly-initialised Qwen2 model, runs in a couple of seconds,
requires no network access.  Use ``--real`` to run against a real checkpoint instead
(recommended on a GPU; slow on CPU).

    python smoke_test.py
    python smoke_test.py --real --model Qwen/Qwen2.5-0.5B
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from vcpa import VCPAInjector, CEPLAttention  # noqa: E402


def build_tiny():
    from transformers import Qwen2Config, Qwen2ForCausalLM
    cfg = Qwen2Config(
        vocab_size=256, hidden_size=64, intermediate_size=128,
        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=8192, tie_word_embeddings=False,
    )
    return Qwen2ForCausalLM._from_config(cfg, attn_implementation="sdpa").eval()


def build_real(model_id: str, dtype: torch.dtype):
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, attn_implementation="sdpa",
    ).eval()


def fingerprint(model) -> tuple[int, float]:
    n = sum(p.numel() for p in model.parameters())
    s = sum(float(p.detach().float().sum()) for p in model.parameters())
    return n, s


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", action="store_true", help="use a real HF checkpoint")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    args = parser.parse_args()

    torch.manual_seed(0)
    print("== load ==")
    if args.real:
        dtype = {"float32": torch.float32, "float16": torch.float16,
                 "bfloat16": torch.bfloat16}[args.dtype]
        model = build_real(args.model, dtype)
    else:
        model = build_tiny()

    vocab = model.config.vocab_size
    fp_before = fingerprint(model)
    keys_before = set(model.state_dict().keys())
    print(f"params: {fp_before[0]:,}")

    injector = VCPAInjector(model, chunk_size=32, local_window=64, threshold=32)
    injector.inject()
    print(injector.report())
    print("state_dict keys equal:", keys_before == set(model.state_dict().keys()))
    print("param count equal:", fp_before[0] == fingerprint(model)[0])
    assert all(isinstance(layer.self_attn, CEPLAttention) for layer in model.model.layers)

    with torch.inference_mode():
        short = model(torch.randint(0, vocab, (2, 24))).logits
        print("short fwd:", tuple(short.shape))

        long = model(torch.randint(0, vocab, (1, 128))).logits
        print("long fwd:", tuple(long.shape), "finite:", bool(torch.isfinite(long).all()))

        odd = model(torch.randint(0, vocab, (1, 100))).logits
        print("non-divisible fwd:", tuple(odd.shape), "finite:", bool(torch.isfinite(odd).all()))

        out1 = model(torch.randint(0, vocab, (1, 16)), use_cache=True)
        out2 = model(torch.randint(0, vocab, (1, 1)),
                     past_key_values=out1.past_key_values, use_cache=True)
        print("decode:", tuple(out2.logits.shape), "finite:", bool(torch.isfinite(out2.logits).all()))

        gen = model.generate(torch.randint(0, vocab, (1, 8)), max_new_tokens=4,
                             do_sample=False, pad_token_id=0, eos_token_id=None)
        print("generate:", tuple(gen.shape))

    injector.remove()
    print("removed. is_injected:", injector.is_injected)
    print("state_dict keys restored:", keys_before == set(model.state_dict().keys()))
    print("parameters unchanged:", fp_before == fingerprint(model))
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
