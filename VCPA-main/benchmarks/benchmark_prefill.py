"""Prefill benchmark: baseline vs VCPA-injected long-context forward passes.

Measures per-sequence latency and tokens/s for several context lengths, plus a
"forced-exact" injected run that isolates the Python-adapter overhead from the
algorithmic saving.

Example (Kaggle GPU):

    python benchmarks/benchmark_prefill.py \
        --model Qwen/Qwen2.5-1.5B --device cuda --dtype float16 \
        --seq-lens 512 1024 2048 4096 --json results/prefill_qwen1.5b.json
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import (  # noqa: E402
    add_common_args, environment_info, load_model, make_injector, measure,
    param_fingerprint, print_table, resolve_device, resolve_dtype, save_json,
)


def main() -> int:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--seq-lens", type=int, nargs="+",
                        default=[512, 1024, 2048, 4096])
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    print("=" * 78)
    print("VCPA-Inject :: prefill benchmark")
    for k, v in environment_info(device).items():
        print(f"  {k}: {v}")
    print("=" * 78)

    model = load_model(args.model, device, dtype,
                       attn_implementation=args.attn_implementation,
                       load_in_4bit=args.load_in_4bit)
    fingerprint_before = param_fingerprint(model)
    vocab = model.config.vocab_size

    results = {
        "benchmark": "prefill",
        "model": args.model,
        "config": vars(args),
        "environment": environment_info(device),
        "fingerprint_before": fingerprint_before,
        "rows": [],
    }

    for seq_len in args.seq_lens:
        input_ids = torch.randint(0, vocab, (args.batch_size, seq_len), device=device)

        # ---- baseline -------------------------------------------------
        def run_baseline():
            with torch.inference_mode():
                model(input_ids, use_cache=False)

        base = measure(run_baseline, device, warmup=args.warmup,
                       repeats=args.repeats, name=f"baseline_{seq_len}")

        # ---- injected (CEPL) ------------------------------------------
        injector = make_injector(model, args).inject()
        try:
            def run_cepl():
                with torch.inference_mode():
                    model(input_ids, use_cache=False)

            cepl = measure(run_cepl, device, warmup=args.warmup,
                           repeats=args.repeats, name=f"cepl_{seq_len}")
        finally:
            injector.remove()

        # ---- injected, forced exact (adapter overhead only) -----------
        # threshold >= seq_len disables CEPL, so we measure the wrapper cost.
        args.threshold, saved = max(args.threshold, seq_len), args.threshold
        injector = make_injector(model, args).inject()
        try:
            def run_exact():
                with torch.inference_mode():
                    model(input_ids, use_cache=False)

            exact = measure(run_exact, device, warmup=args.warmup,
                            repeats=args.repeats, name=f"exact_{seq_len}")
        finally:
            injector.remove()
            args.threshold = saved

        row = {
            "seq_len": seq_len,
            "batch_size": args.batch_size,
            "baseline_s": round(base.median, 4),
            "cepl_s": round(cepl.median, 4),
            "adapter_exact_s": round(exact.median, 4),
            "speedup_cepl_vs_baseline": round(base.median / cepl.median, 3) if cepl.median else None,
            "adapter_overhead_pct": round((exact.median / base.median - 1.0) * 100.0, 2),
            "baseline_peak_mb": _mb(base.peak_memory_bytes),
            "cepl_peak_mb": _mb(cepl.peak_memory_bytes),
            "baseline_tokens_per_s": round(seq_len * args.batch_size / base.median, 1),
            "cepl_tokens_per_s": round(seq_len * args.batch_size / cepl.median, 1),
        }
        results["rows"].append(row)
        print(f"seq_len={seq_len:5d}  baseline={row['baseline_s']:.3f}s  "
              f"cepl={row['cepl_s']:.3f}s  speedup={row['speedup_cepl_vs_baseline']}x  "
              f"peak={row['cepl_peak_mb']}MB")

    fingerprint_after = param_fingerprint(model)
    results["fingerprint_after"] = fingerprint_after
    results["parameters_unchanged"] = fingerprint_before == fingerprint_after

    print()
    print_table(results["rows"], [
        "seq_len", "baseline_s", "cepl_s", "adapter_exact_s",
        "speedup_cepl_vs_baseline", "adapter_overhead_pct",
        "baseline_peak_mb", "cepl_peak_mb",
    ])
    print(f"\nparameters unchanged: {results['parameters_unchanged']}")

    if args.json:
        save_json(results, args.json)
        print(f"results written to {args.json}")
    return 0


def _mb(value):
    return round(value / 1e6, 1) if value is not None else None


if __name__ == "__main__":
    raise SystemExit(main())
