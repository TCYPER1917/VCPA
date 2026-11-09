"""Accuracy benchmark: perplexity and next-token divergence, baseline vs VCPA.

VCPA is an APPROXIMATION.  This benchmark measures how much quality is lost, on real
text, instead of hiding it.  Two complementary views:

  1. Perplexity over held-out windows (what users actually feel).
  2. Divergence of the next-token distribution at each window boundary
     (mean |Δlogit|, max |Δlogit|, top-1 agreement, symmetric KL, cosine similarity).

Corpus resolution order:
  * ``--text-file`` (plain text)
  * Hugging Face ``datasets`` wikitext-2-raw-v1 (test split) when available
  * a small built-in public-domain sample (enough for a smoke check)

    python benchmarks/benchmark_accuracy.py --model Qwen/Qwen2.5-1.5B --device cuda \
        --dtype float16 --seq-lens 1024 2048 --max-windows 8 --json results/accuracy.json
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import (  # noqa: E402
    add_common_args, environment_info, load_model, make_injector, param_fingerprint,
    print_table, resolve_device, resolve_dtype, save_json,
)

FALLBACK_TEXT = (
    "The history of computing is a history of abstractions. Each layer hides the "
    "complexity beneath it, and each new abstraction makes the next one possible. "
    "Attention mechanisms changed the way sequence models are built because they let "
    "every position look directly at every other position instead of passing information "
    "through a fixed-size hidden state. That direct access is powerful, but it is also "
    "expensive: the cost of comparing every pair of positions grows with the square of "
    "the sequence length. Much of modern systems work on language models is therefore "
    "about approximating that comparison without losing too much of its accuracy. "
) * 40


def load_corpus(args) -> str:
    if args.text_file:
        with open(args.text_file, "r", encoding="utf-8") as fh:
            return fh.read()
    # The wikitext script dataset was removed from the Hub; the canonical copy is now
    # the "Salesforce/wikitext" dataset repo. Try the current name first, then the
    # legacy one, and fall back to a built-in sample only if both fail.
    candidates = [
        ("Salesforce/wikitext", "wikitext-2-raw-v1"),
        ("wikitext", "wikitext-2-raw-v1"),
    ]
    last_error = None
    for repo, config in candidates:
        try:
            from datasets import load_dataset
            ds = load_dataset(repo, config, split="test")
            text = "\n".join(t for t in ds["text"] if t.strip())
            if text:
                print(f"corpus: {repo} / {config} (test), {len(text)} chars")
                return text
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    print(f"corpus: built-in sample (wikitext unavailable: "
          f"{type(last_error).__name__}: {last_error})")
    return FALLBACK_TEXT


def make_windows(tokenizer, text: str, seq_len: int, max_windows: int, device):
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    usable = (len(ids) // seq_len) * seq_len
    if usable < seq_len:
        raise SystemExit(f"corpus too short for seq_len={seq_len}: {len(ids)} tokens available")
    return ids[:usable].view(-1, seq_len)[:max_windows].to(device)


@torch.inference_mode()
def evaluate(model, windows) -> dict:
    """Perplexity plus the last-position logits of every window."""
    nll = 0.0
    n_tokens = 0
    last_logits = []
    for i in range(windows.shape[0]):
        ids = windows[i:i + 1]          # keep the batch dimension: (1, seq_len)
        logits = model(ids, use_cache=False).logits
        # Score in chunks: the full fp32 logit matrix is vocab_size wide and, cast all at
        # once, can exhaust a 16 GB GPU on long sequences.
        target = ids[:, 1:]
        for start in range(0, ids.shape[1] - 1, 256):
            sl = logits[:, start:start + 256].float()
            tg = target[:, start:start + 256]
            nll += float(F.cross_entropy(sl.reshape(-1, sl.shape[-1]), tg.reshape(-1),
                                         reduction="sum"))
            n_tokens += tg.numel()
        last_logits.append(logits[:, -1, :].float().detach().cpu())
    return {
        "ppl": math.exp(nll / n_tokens),
        "tokens": n_tokens,
        "last_logits": last_logits,
    }


def divergence(logits_a, logits_b) -> dict:
    sum_abs = 0.0
    max_abs = 0.0
    agree = 0
    total = 0
    kl = 0.0
    cos = 0.0
    width = 1
    for la, lb in zip(logits_a, logits_b):
        la, lb = la.float(), lb.float()
        width = la.shape[-1]
        sum_abs += float((la - lb).abs().sum())
        max_abs = max(max_abs, float((la - lb).abs().max()))
        agree += int((la.argmax(-1) == lb.argmax(-1)).sum())
        total += la.shape[0]
        logp_a, logp_b = F.log_softmax(la, -1), F.log_softmax(lb, -1)
        kl += float((logp_a.exp() * (logp_a - logp_b)).sum())
        cos += float(F.cosine_similarity(la, lb, dim=-1).mean())
    n = max(1, len(logits_a))
    return {
        "mean_abs_logit_diff": sum_abs / max(1, total * width),
        "max_abs_logit_diff": max_abs,
        "top1_agreement_pct": 100.0 * agree / max(1, total),
        "mean_kl_baseline_to_injected": kl / n,
        "mean_cosine_similarity": cos / n,
    }


class _StubTokenizer:
    """Deterministic byte-level stub tokenizer used only for the ``tiny:`` model."""

    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size

    def __call__(self, text: str, return_tensors=None, add_special_tokens=False):
        import torch as _torch
        ids = _torch.tensor([b % self.vocab_size for b in text.encode("utf-8")], dtype=_torch.long)
        return type("TokOut", (), {"input_ids": ids.unsqueeze(0)})()


def main() -> int:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[512, 1024, 2048])
    parser.add_argument("--max-windows", type=int, default=8)
    parser.add_argument("--text-file", default=None)
    args = parser.parse_args()

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    print("=" * 78)
    print("VCPA-Inject :: accuracy benchmark")
    for k, v in environment_info(device).items():
        print(f"  {k}: {v}")
    print("=" * 78)

    from transformers import AutoTokenizer

    if args.model.startswith("tiny:"):
        tokenizer = _StubTokenizer(256)
        print("tokenizer: stub (tiny model)")
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    text = load_corpus(args)

    model = load_model(args.model, device, dtype,
                       attn_implementation=args.attn_implementation,
                       load_in_4bit=args.load_in_4bit)
    fp_before = param_fingerprint(model)

    results = {
        "benchmark": "accuracy",
        "model": args.model,
        "config": vars(args),
        "environment": environment_info(device),
        "fingerprint_before": fp_before,
        "rows": [],
    }

    for seq_len in args.seq_lens:
        windows = make_windows(tokenizer, text, seq_len, args.max_windows, device)

        base = evaluate(model, windows)

        injector = make_injector(model, args).inject()
        try:
            cepl = evaluate(model, windows)
        finally:
            injector.remove()

        div = divergence(base["last_logits"], cepl["last_logits"])
        row = {
            "seq_len": seq_len,
            "windows": len(windows),
            "tokens": base["tokens"],
            "ppl_baseline": round(base["ppl"], 4),
            "ppl_injected": round(cepl["ppl"], 4),
            "ppl_delta_pct": round(cepl["ppl"] / base["ppl"] * 100 - 100, 3),
            "mean_abs_logit_diff": round(div["mean_abs_logit_diff"], 6),
            "max_abs_logit_diff": round(div["max_abs_logit_diff"], 4),
            "top1_agreement_pct": round(div["top1_agreement_pct"], 3),
            "mean_kl": round(div["mean_kl_baseline_to_injected"], 6),
            "mean_cosine_similarity": round(div["mean_cosine_similarity"], 6),
        }
        results["rows"].append(row)
        print(f"seq_len={seq_len:5d}  ppl {row['ppl_baseline']:.3f} -> {row['ppl_injected']:.3f} "
              f"({row['ppl_delta_pct']:+.2f}%)  top1={row['top1_agreement_pct']:.2f}%  "
              f"cos={row['mean_cosine_similarity']:.4f}")

    fp_after = param_fingerprint(model)
    results["fingerprint_after"] = fp_after
    results["parameters_unchanged"] = fp_before == fp_after

    print()
    print_table(results["rows"], ["seq_len", "ppl_baseline", "ppl_injected", "ppl_delta_pct",
                                  "top1_agreement_pct", "mean_cosine_similarity"])
    print(f"\nparameters unchanged: {results['parameters_unchanged']}")
    if args.json:
        save_json(results, args.json)
        print(f"results written to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
