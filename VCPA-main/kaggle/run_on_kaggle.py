"""One-command Kaggle runner for VCPA-Inject.

Upload this repository as a Kaggle Dataset (or clone it into the notebook), then:

    !python kaggle/run_on_kaggle.py --models Qwen/Qwen2.5-1.5B

It runs the full suite in isolated subprocesses (clean CUDA allocator per benchmark):

  1. parameter-integrity check on the real checkpoint
  2. accuracy / perplexity (wikitext-2 when ``datasets`` is installed)
  3. prefill latency + throughput
  4. decode throughput (VCPA falls back to exact attention here)
  5. peak-memory comparison

Results are written to ``/kaggle/working/vcpa_results/*.json`` (or ``./results`` when
not on Kaggle) and a markdown summary is printed at the end.

Useful flags:
    --models Qwen/Qwen2.5-0.5B Qwen/Qwen2.5-1.5B Qwen/Qwen2.5-3B
    --seq-lens 512 1024 2048 4096        # prefill / accuracy lengths
    --memory-seq-lens 1024 2048 4096 8192
    --load-in-4bit                        # for models that do not fit in 16-bit
    --skip accuracy memory                # run only some benchmarks
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

IS_KAGGLE = os.path.isdir("/kaggle/working")
DEFAULT_OUT = "/kaggle/working/vcpa_results" if IS_KAGGLE else os.path.join(ROOT, "results")


def run(cmd: list[str], label: str) -> int:
    print("\n" + "=" * 80)
    print(f">>> {label}")
    print("    " + " ".join(cmd))
    print("=" * 80, flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd)
    print(f"<<< {label}: exit={proc.returncode} in {time.time() - t0:.1f}s", flush=True)
    return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["Qwen/Qwen2.5-0.5B"])
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[512, 1024, 2048])
    parser.add_argument("--memory-seq-lens", type=int, nargs="+", default=[1024, 2048, 4096])
    parser.add_argument("--prompt-len", type=int, default=1024)
    parser.add_argument("--new-tokens", type=int, default=64)
    parser.add_argument("--max-windows", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--local-window", type=int, default=256)
    parser.add_argument("--threshold", type=int, default=256)
    parser.add_argument("--pool", default="mean")
    parser.add_argument("--blend-mode", default="unified")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--skip", nargs="*", default=[],
                        choices=["accuracy", "prefill", "decode", "memory"])
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    py = sys.executable
    bench = os.path.join(ROOT, "benchmarks")

    def common(model: str) -> list[str]:
        cmd = [
            "--model", model,
            "--dtype", args.dtype,
            "--chunk-size", str(args.chunk_size),
            "--local-window", str(args.local_window),
            "--threshold", str(args.threshold),
            "--pool", args.pool,
            "--blend-mode", args.blend_mode,
            "--warmup", str(args.warmup),
            "--repeats", str(args.repeats),
        ]
        if args.load_in_4bit:
            cmd.append("--load-in-4bit")
        return cmd

    summary = {}
    for model in args.models:
        tag = model.replace("/", "__").replace(":", "_")
        summary[model] = {}

        # 1. integrity -----------------------------------------------------
        if run([py, os.path.join(HERE, "check_integrity.py"), "--model", model,
                "--dtype", args.dtype] + (["--load-in-4bit"] if args.load_in_4bit else []),
               f"integrity :: {model}") != 0:
            print(f"!! integrity check failed for {model}; continuing to benchmarks")

        # 2. accuracy ------------------------------------------------------
        if "accuracy" not in args.skip:
            out = os.path.join(args.out_dir, f"accuracy_{tag}.json")
            run([py, os.path.join(bench, "benchmark_accuracy.py"), *common(model),
                 "--seq-lens", *map(str, args.seq_lens),
                 "--max-windows", str(args.max_windows),
                 "--json", out], f"accuracy :: {model}")
            summary[model]["accuracy"] = out

        # 3. prefill -------------------------------------------------------
        if "prefill" not in args.skip:
            out = os.path.join(args.out_dir, f"prefill_{tag}.json")
            run([py, os.path.join(bench, "benchmark_prefill.py"), *common(model),
                 "--seq-lens", *map(str, args.seq_lens),
                 "--json", out], f"prefill :: {model}")
            summary[model]["prefill"] = out

        # 4. decode --------------------------------------------------------
        if "decode" not in args.skip:
            out = os.path.join(args.out_dir, f"decode_{tag}.json")
            run([py, os.path.join(bench, "benchmark_decode.py"), *common(model),
                 "--prompt-len", str(args.prompt_len),
                 "--new-tokens", str(args.new_tokens),
                 "--json", out], f"decode :: {model}")
            summary[model]["decode"] = out

        # 5. memory --------------------------------------------------------
        if "memory" not in args.skip:
            out = os.path.join(args.out_dir, f"memory_{tag}.json")
            run([py, os.path.join(bench, "benchmark_memory.py"), *common(model),
                 "--seq-lens", *map(str, args.memory_seq_lens),
                 "--json", out], f"memory :: {model}")
            summary[model]["memory"] = out

    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print_summary(summary)
    print(f"\nAll results saved under {args.out_dir}")
    return 0


def print_summary(summary: dict) -> None:
    print("\n" + "#" * 80)
    print("# VCPA-Inject :: results summary")
    print("#" * 80)
    for model, files in summary.items():
        print(f"\n## {model}")
        for kind, path in files.items():
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
            except Exception as exc:  # noqa: BLE001
                print(f"- {kind}: could not read {path} ({exc})")
                continue
            rows = data.get("rows", [])
            ok = data.get("parameters_unchanged")
            print(f"- {kind}: parameters_unchanged={ok}")
            for r in rows:
                if "ppl_baseline" in r:
                    print(f"    seq={r['seq_len']:>5}  ppl {r['ppl_baseline']:.3f} -> "
                          f"{r['ppl_injected']:.3f} ({r['ppl_delta_pct']:+.2f}%)  "
                          f"top1={r['top1_agreement_pct']:.1f}%")
                elif "reduction_pct" in r:
                    print(f"    seq={r['seq_len']:>5}  {r['baseline_peak_mb']}MB -> "
                          f"{r['cepl_peak_mb']}MB ({r['reduction_pct']}%)")
                elif "speedup_cepl_vs_baseline" in r:
                    print(f"    seq={r['seq_len']:>5}  {r['baseline_s']:.3f}s -> "
                          f"{r['cepl_s']:.3f}s ({r['speedup_cepl_vs_baseline']}x)")
                elif "baseline_tokens_per_s" in r:
                    print(f"    prompt={r['prompt_len']:>5}  "
                          f"{r['baseline_tokens_per_s']:.1f} -> "
                          f"{r['injected_tokens_per_s']:.1f} tok/s")


if __name__ == "__main__":
    raise SystemExit(main())
