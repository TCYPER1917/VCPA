"""Memory benchmark: peak GPU memory for long-context prefill, baseline vs VCPA.

On CUDA the dominant term is the attention score matrix O(B*H*N^2).  VCPA replaces it
with O(B*H*N*(W + N/C)); this benchmark measures the actual peak difference.

    python benchmarks/benchmark_memory.py --model Qwen/Qwen2.5-1.5B --device cuda \
        --seq-lens 1024 2048 4096 8192 --json results/memory.json

CPU runs are supported but ``peak_mb`` will be null (no CUDA allocator stats).
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import (  # noqa: E402
    add_common_args, environment_info, load_model, make_injector, param_fingerprint,
    peak_memory, print_table, reset_peak_memory, resolve_device, resolve_dtype, save_json,
    Timer,
)


def main() -> int:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[1024, 2048, 4096, 8192])
    args = parser.parse_args()

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    print("=" * 78)
    print("VCPA-Inject :: memory benchmark")
    for k, v in environment_info(device).items():
        print(f"  {k}: {v}")
    print("=" * 78)

    model = load_model(args.model, device, dtype,
                       attn_implementation=args.attn_implementation,
                       load_in_4bit=args.load_in_4bit)
    fp_before = param_fingerprint(model)
    vocab = model.config.vocab_size

    results = {
        "benchmark": "memory",
        "model": args.model,
        "config": vars(args),
        "environment": environment_info(device),
        "fingerprint_before": fp_before,
        "rows": [],
    }

    for seq_len in args.seq_lens:
        input_ids = torch.randint(0, vocab, (1, seq_len), device=device)
        try:
            # ---- baseline ------------------------------------------------
            with torch.inference_mode():
                model(input_ids, use_cache=False)      # warm-up allocator
            reset_peak_memory(device)
            with torch.inference_mode(), Timer(device) as t_base:
                model(input_ids, use_cache=False)
            base_peak = peak_memory(device)

            # ---- injected ------------------------------------------------
            injector = make_injector(model, args).inject()
            try:
                with torch.inference_mode():
                    model(input_ids, use_cache=False)
                reset_peak_memory(device)
                with torch.inference_mode(), Timer(device) as t_cepl:
                    model(input_ids, use_cache=False)
                cepl_peak = peak_memory(device)
            finally:
                injector.remove()
        except torch.cuda.OutOfMemoryError:  # pragma: no cover - GPU only
            torch.cuda.empty_cache()
            results["rows"].append({"seq_len": seq_len, "error": "CUDA OOM"})
            print(f"seq_len={seq_len:5d}  OOM")
            break

        row = {
            "seq_len": seq_len,
            "baseline_peak_mb": _mb(base_peak),
            "cepl_peak_mb": _mb(cepl_peak),
            "baseline_s": round(t_base.elapsed, 4),
            "cepl_s": round(t_cepl.elapsed, 4),
            "reduction_pct": (round((1 - cepl_peak / base_peak) * 100, 2)
                              if base_peak and cepl_peak else None),
        }
        results["rows"].append(row)
        print(f"seq_len={seq_len:5d}  baseline={row['baseline_peak_mb']}MB  "
              f"cepl={row['cepl_peak_mb']}MB  reduction={row['reduction_pct']}%")

    fp_after = param_fingerprint(model)
    results["fingerprint_after"] = fp_after
    results["parameters_unchanged"] = fp_before == fp_after

    print()
    print_table(results["rows"], ["seq_len", "baseline_peak_mb", "cepl_peak_mb", "reduction_pct"])
    if args.json:
        save_json(results, args.json)
        print(f"results written to {args.json}")
    return 0


def _mb(value):
    return round(value / 1e6, 1) if value is not None else None


if __name__ == "__main__":
    raise SystemExit(main())
