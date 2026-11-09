"""Decode benchmark: tokens/s for autoregressive generation with a KV cache.

VCPA intentionally falls back to exact attention during decode (see README), so the
expected result is: **same speed as baseline** (the wrapper adds a thin Python layer).
This benchmark verifies that claim instead of assuming it.

    python benchmarks/benchmark_decode.py --model Qwen/Qwen2.5-1.5B --device cuda \
        --prompt-len 512 --new-tokens 64 --json results/decode.json
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import (  # noqa: E402
    add_common_args, environment_info, load_model, make_injector, param_fingerprint,
    print_table, resolve_device, resolve_dtype, save_json,
)


@torch.inference_mode()
def greedy_decode(model, input_ids, new_tokens: int, device) -> float:
    """Greedy decode, returning seconds for the decode loop only."""
    out = model(input_ids, use_cache=True)
    past = out.past_key_values
    nxt = out.logits[:, -1:, :].argmax(dim=-1)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for _ in range(new_tokens - 1):
        out = model(nxt, past_key_values=past, use_cache=True)
        past = out.past_key_values
        nxt = out.logits[:, -1:, :].argmax(dim=-1)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter() - t0


def main() -> int:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--prompt-len", type=int, default=512)
    parser.add_argument("--new-tokens", type=int, default=64)
    args = parser.parse_args()

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    print("=" * 78)
    print("VCPA-Inject :: decode benchmark")
    for k, v in environment_info(device).items():
        print(f"  {k}: {v}")
    print("=" * 78)

    model = load_model(args.model, device, dtype,
                       attn_implementation=args.attn_implementation,
                       load_in_4bit=args.load_in_4bit)
    fp_before = param_fingerprint(model)
    vocab = model.config.vocab_size

    results = {
        "benchmark": "decode",
        "model": args.model,
        "config": vars(args),
        "environment": environment_info(device),
        "fingerprint_before": fp_before,
        "rows": [],
    }

    for prompt_len in _as_list(args.prompt_len):
        input_ids = torch.randint(0, vocab, (1, prompt_len), device=device)

        for _ in range(args.warmup):
            greedy_decode(model, input_ids, min(args.new_tokens, 8), device)
        base_times = [greedy_decode(model, input_ids, args.new_tokens, device)
                      for _ in range(max(1, args.repeats // 2))]
        base_s = sorted(base_times)[len(base_times) // 2]

        injector = make_injector(model, args).inject()
        try:
            for _ in range(args.warmup):
                greedy_decode(model, input_ids, min(args.new_tokens, 8), device)
            cepl_times = [greedy_decode(model, input_ids, args.new_tokens, device)
                          for _ in range(max(1, args.repeats // 2))]
        finally:
            injector.remove()
        cepl_s = sorted(cepl_times)[len(cepl_times) // 2]

        row = {
            "prompt_len": prompt_len,
            "new_tokens": args.new_tokens,
            "baseline_s": round(base_s, 4),
            "injected_s": round(cepl_s, 4),
            "baseline_tokens_per_s": round(args.new_tokens / base_s, 1),
            "injected_tokens_per_s": round(args.new_tokens / cepl_s, 1),
            "ratio": round(cepl_s / base_s, 3),
        }
        results["rows"].append(row)
        print(f"prompt={prompt_len:5d}  baseline={row['baseline_tokens_per_s']:.1f} tok/s  "
              f"injected={row['injected_tokens_per_s']:.1f} tok/s  ratio={row['ratio']}x")

    fp_after = param_fingerprint(model)
    results["fingerprint_after"] = fp_after
    results["parameters_unchanged"] = fp_before == fp_after

    print()
    print_table(results["rows"], ["prompt_len", "new_tokens", "baseline_tokens_per_s",
                                  "injected_tokens_per_s", "ratio"])
    if args.json:
        save_json(results, args.json)
        print(f"results written to {args.json}")
    print(f"parameters unchanged: {results['parameters_unchanged']}")
    return 0


def _as_list(value):
    return value if isinstance(value, list) else [value]


if __name__ == "__main__":
    raise SystemExit(main())
