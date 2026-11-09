"""Minimal end-to-end example: inject VCPA into a pretrained Qwen model.

    # CPU smoke test (small model, short prompt)
    python examples/inject_qwen.py --model Qwen/Qwen2.5-0.5B

    # GPU, long context, with timing
    python examples/inject_qwen.py --model Qwen/Qwen2.5-1.5B --device cuda \
        --dtype float16 --prompt-tokens 2048

The script:
  1. loads the model,
  2. records a parameter fingerprint,
  3. runs the prompt before injection,
  4. injects VCPA and runs the same prompt again,
  5. prints error metrics and the injector report,
  6. removes VCPA and proves the parameters are untouched.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from vcpa import VCPAInjector  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dtype", default="auto",
                        choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--prompt", default=None,
                        help="prompt text; defaults to repeating a public-domain sentence")
    parser.add_argument("--prompt-tokens", type=int, default=512,
                        help="if --prompt is omitted, build a prompt of this many tokens")
    parser.add_argument("--new-tokens", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--local-window", type=int, default=256)
    parser.add_argument("--threshold", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    dtype = {"auto": torch.float16 if device.type == "cuda" else torch.float32,
             "float32": torch.float32,
             "float16": torch.float16,
             "bfloat16": torch.bfloat16}[args.dtype]

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"loading {args.model} on {device} as {dtype} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, attn_implementation="sdpa",
    ).to(device).eval()

    text = args.prompt or (
        "Attention lets every position look at every other position. "
        "The cost grows with the square of the sequence length. "
    )
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    if ids.shape[1] < 8:
        ids = ids.repeat(1, 8)
    if args.prompt is None:
        reps = max(1, args.prompt_tokens // ids.shape[1] + 1)
        ids = ids.repeat(1, reps)[:, :args.prompt_tokens]
    print(f"prompt tokens: {ids.shape[1]}")

    def fingerprint() -> tuple[int, float]:
        n = 0
        s = 0.0
        for p in model.parameters():
            n += p.numel()
            s += float(p.detach().float().sum())
        return n, s

    fp_before = fingerprint()

    with torch.inference_mode():
        t0 = time.perf_counter()
        base_logits = model(ids).logits.detach().float()
        base_time = time.perf_counter() - t0
    print(f"baseline forward: {base_time:.3f}s")

    injector = VCPAInjector(
        model,
        chunk_size=args.chunk_size,
        local_window=args.local_window,
        threshold=args.threshold,
    ).inject()
    print(injector.report())

    try:
        with torch.inference_mode():
            t0 = time.perf_counter()
            vcpa_logits = model(ids).logits.detach().float()
            vcpa_time = time.perf_counter() - t0
        diff = (base_logits - vcpa_logits).abs()
        rel = float(diff.mean() / base_logits.abs().mean())
        print(f"vcpa forward:     {vcpa_time:.3f}s  (speedup {base_time / vcpa_time:.2f}x)")
        print(f"logit deviation:  max={float(diff.max()):.4f}  mean={float(diff.mean()):.4f}  rel={rel:.4f}")
        print(f"last-token top-1 agreement: "
              f"{float((base_logits[:, -1].argmax(-1) == vcpa_logits[:, -1].argmax(-1)).float().mean()):.2f}")

        with torch.inference_mode():
            out = model.generate(ids[:, :64], max_new_tokens=args.new_tokens,
                                 do_sample=False, pad_token_id=tokenizer.eos_token_id)
        print("generated:", tokenizer.decode(out[0, -args.new_tokens:], skip_special_tokens=True)[:200])
    finally:
        injector.remove()

    fp_after = fingerprint()
    print(f"parameters unchanged: {fp_before == fp_after}")
    print("VCPA removed; model restored to the original attention implementation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
