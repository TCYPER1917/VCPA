# VCPA-Inject (Generation 12)

**A training-free, reversible runtime attention adapter for pretrained causal language models.**

VCPA-Inject wraps the existing attention modules of a Hugging Face model at runtime and
replaces exact full-context attention with **CEPL** — *Chunked-Exact + Pooled-Landmarks*
attention — **without changing, copying, or adding a single model parameter**.

```python
from transformers import AutoModelForCausalLM
from vcpa import VCPAInjector

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
injector = VCPAInjector(model).inject()

# ... normal inference / generation, unchanged call sites ...

injector.remove()          # attention is exactly the original implementation again
```

---

## Table of contents

- [What this is (and is not)](#what-this-is-and-is-not)
- [Guarantees](#guarantees)
- [How it works](#how-it-works)
- [Quick start](#quick-start)
- [Configuration](#configuration)
- [Testing](#testing)
- [Benchmarks](#benchmarks)
- [Supported models and limitations](#supported-models-and-limitations)
- [Project layout](#project-layout)
- [Generation 11 audit](#generation-11-audit)

---

## What this is (and is not)

**Is:**

- A runtime wrapper around existing attention modules.
- A long-context approximation: recent tokens keep *exact* token-level attention; older
  chunks are represented by pooled landmarks inside a **single softmax**.
- Fully reversible: `inject()` / `remove()`, repeated any number of times.
- Parameter-preserving: verified by SHA-256 over every parameter before, during, and
  after injection.

**Is not:**

- Not a training method, not a fine-tuning method, not a new checkpoint format.
- **Not claimed to be numerically equivalent to exact attention.** CEPL is an
  approximation. Its error is measured, not hidden.
- Not claimed to be compatible with every architecture (see
  [Supported models](#supported-models-and-limitations)).

---

## Guarantees

These are enforced by the test-suite (`tests/`) and by `kaggle/check_integrity.py`:

| Constraint | How it is enforced |
| :--- | :--- |
| Pretrained parameter values unchanged | SHA-256 per parameter before/during/after injection |
| `q_proj` / `k_proj` / `v_proj` / `o_proj` unchanged | Same **module and tensor objects** are reused (`assertIs`) |
| No trainable parameters introduced | Total and trainable parameter counts are identical |
| No weight duplication | Wrapper aliases the original `_parameters` / `_buffers` / `_modules` registries |
| `state_dict` key structure preserved | Keys and values compared before/during/after; `load_state_dict` works while injected |
| No silent config change | `model.config.to_dict()` compared before/after |
| Reversible | `remove()` restores the *same module objects* that were wrapped |
| No future-token leakage | Changing suffix tokens leaves earlier logits bit-identical |
| No dropped tokens | Non-chunk-aligned tails are computed with exact causal attention |

Run them yourself:

```bash
python -m unittest discover -s tests -p "test_*.py" -v   # ~2 s, offline, 47 tests
```

---

## How it works

For a sequence of `N` tokens split into chunks of size `C`, a query token in chunk `c`
attends over the **union** of:

1. **Exact window** — every token of the `L` most recent chunks
   (`L = local_window // C`), with a proper causal mask inside the current chunk.
2. **Landmarks** — one pooled key/value per chunk for chunks `0 … c-L` (strictly older
   than the exact window, so nothing is counted twice).

Both sets share **one softmax**, which is the principled sparse-attention formulation:

```
p = softmax( [ q·K_exact ,  q·K_landmark + log(C) ] / sqrt(d) )
```

The `log(C)` term is mass preservation: a landmark summarises `C` original keys, so a
single landmark must contribute the same total probability mass as `C` identical keys.
Without it, old context is silently under-weighted (this was the main error source in
early prototypes).

Cost per query drops from `N` keys to `W + N/C` keys, where `W = L·C` is the exact
window. Peak score memory drops from `O(B·H·N²)` to `O(B·H·N·(W + N/C))`.

### Exact paths (correctness first)

CEPL runs **only** for unpadded, cache-free prefills above `threshold`. Everything else
uses the original exact attention through SDPA:

- `seq_len <= threshold` → exact
- a mask that is not plain padding-plus-causal → exact
- multi-token continuation that reuses a non-empty KV cache → exact
- single-token decode inside `threshold` → exact; longer decode uses the landmark cache
- both approximation streams disabled → exact
- non-chunk-aligned tail tokens → exact causal attention over the full past

### Grouped-query attention

GQA/MQA is handled by reshaping queries to `(B, H_kv, G, …)` and **broadcasting** KV
heads. Keys and values are never replicated. `enable_gqa=True` is used on the SDPA paths.

### Decode

The KV cache is still updated exactly as the original module does
(`past_key_values.update(...)`). Short decode stays on exact SDPA. Once the cache is
longer than `threshold`, each new token attends to the exact recent window plus landmarks
of completed older chunks, updated only when a chunk finishes. That path is an
approximation; `benchmarks/benchmark_decode.py` measures it rather than assuming a speedup.

---

## Quick start

```bash
python -m pip install -r requirements.txt
```

```python
import torch
from transformers import AutoModelForCausalLM
from vcpa import VCPAInjector

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B", torch_dtype=torch.float16,
).cuda().eval()

injector = VCPAInjector(
    model,
    chunk_size=64,
    local_window=256,   # tokens kept exact
    threshold=256,      # below this, exact attention
).inject()

print(injector.report())

with torch.inference_mode():
    out = model.generate(input_ids, max_new_tokens=64)

injector.remove()
```

Full runnable example: [`examples/inject_qwen.py`](examples/inject_qwen.py).

---

## Configuration

| Argument | Default | Meaning |
| :--- | :--- | :--- |
| `chunk_size` | `64` | Tokens per pooled chunk (`C`). |
| `local_window` | `256` | Tokens kept at exact token-level attention (`W`). `L = W // C`. |
| `threshold` | `256` | Sequences at or below this length use exact SDPA. |
| `use_landmarks` | `True` | Include pooled landmarks for old chunks (legacy `blend` mode only). |
| `use_local` | `True` | Include the exact recent window (legacy `blend` mode only). |
| `pool_type` | `"multivec"` | `multivec` (K segment landmarks), `mean`, `max`, `softmax`, `topk`. |
| `num_landmarks` | `4` | Vectors kept per chunk. Only used by `multivec`. |
| `blend_mode` | `"unified"` | `"unified"` (single softmax over both streams) or `"blend"` (legacy convex mixture, for comparison). |
| `landmark_weight` | `0.5` | Only used by the legacy `"blend"` mode. |

> `blend_mode="unified"` is a single softmax over **both** streams, so it requires
> `use_local=True` and `use_landmarks=True`. To disable a stream (for ablation), use
> `blend_mode="blend"`; that path is causal but less faithful.

`injector.report()` prints the active configuration and per-layer status.

---

## Testing

```bash
# Offline, deterministic, ~2 s (tiny randomly-initialised Qwen2; no downloads)
python -m unittest discover -s tests -p "test_*.py" -v

# End-to-end smoke test
python smoke_test.py

# Real-checkpoint integrity (works on CPU for 0.5B)
python kaggle/check_integrity.py --model Qwen/Qwen2.5-0.5B --dtype float32
```

The committed test-suite covers:

- parameter integrity, state-dict surface, reversibility, repeated cycles
- chunk/token causal masks, future-token independence, padding fallback
- GQA and MHA shapes, RoPE, exact-window equivalence
- KV cache decode, multi-token continuation, remainder tokens, generation
- a regression guard that **the CEPL path is actually used** for long prefills

---

## Benchmarks

> **No benchmark numbers are hard-coded in this repository.**
> Numbers depend on GPU, dtype, model and driver. Run the suite on your target hardware
> and report what you measure.

```bash
python benchmarks/benchmark_prefill.py --model Qwen/Qwen2.5-1.5B --device cuda --dtype float16
python benchmarks/benchmark_decode.py  --model Qwen/Qwen2.5-1.5B --device cuda --dtype float16
python benchmarks/benchmark_memory.py  --model Qwen/Qwen2.5-1.5B --device cuda --dtype float16
python benchmarks/benchmark_accuracy.py --model Qwen/Qwen2.5-1.5B --device cuda --dtype float16
```

Each script prints a table and (with `--json`) writes machine-readable results including
`parameters_unchanged: true/false`.

### Kaggle

Everything heavy is packaged for Kaggle:

- [`kaggle/VCPA_Kaggle.ipynb`](kaggle/VCPA_Kaggle.ipynb) — notebook
- [`kaggle/run_on_kaggle.py`](kaggle/run_on_kaggle.py) — one command runs all benchmarks
- [`kaggle/check_integrity.py`](kaggle/check_integrity.py) — real-checkpoint integrity

```bash
python kaggle/run_on_kaggle.py \
    --models Qwen/Qwen2.5-0.5B Qwen/Qwen2.5-1.5B \
    --seq-lens 512 1024 2048 4096 \
    --memory-seq-lens 1024 2048 4096 8192
```

Results land in `/kaggle/working/vcpa_results/`.

### Results template

Fill this after running on your hardware (do not copy numbers from elsewhere):

Measured on a Tesla T4 (16 GB), torch 2.11, transformers 5.22, fp16, batch size 1,
defaults `chunk_size=64`, `local_window=256`, `threshold=256`, `blend_mode="unified"`,
`pool="mean"`. Perplexity is wikitext-2-raw-v1 (test), 4 non-overlapping windows.

**Prefill latency** (`benchmark_prefill.py`, median of 5 repeats after 2 warm-ups):

| Model | N | Baseline s | CEPL s | Speedup | Adapter overhead |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-0.5B | 512 | 0.051 | 0.051 | 0.99x | +0.7% |
| Qwen2.5-0.5B | 1024 | 0.142 | 0.093 | 1.53x | +0.0% |
| Qwen2.5-0.5B | 2048 | 0.445 | 0.190 | 2.35x | +0.0% |
| Qwen2.5-0.5B | 4096 | 1.550 | 0.389 | 3.99x | +1.0% |
| Qwen2.5-1.5B | 512 | 0.120 | 0.114 | 1.05x | +1.4% |
| Qwen2.5-1.5B | 1024 | 0.283 | 0.225 | 1.25x | +0.8% |
| Qwen2.5-1.5B | 2048 | 0.781 | 0.479 | 1.63x | +4.6% |
| Qwen2.5-1.5B | 4096 | 2.509 | 0.910 | 2.76x | -5.3% |

**Peak memory** (`benchmark_memory.py`):

| Model | N | Baseline MB | CEPL MB | Reduction |
| :--- | ---: | ---: | ---: | ---: |
| Qwen2.5-0.5B | 2048 | 1632 | 1632 | 0% |
| Qwen2.5-0.5B | 4096 | 3283 | 2259 | 31% |
| Qwen2.5-0.5B | 8192 | 9921 | 3510 | 65% |
| Qwen2.5-1.5B | 4096 | 5144 | 4356 | 15% |
| Qwen2.5-1.5B | 8192 | 10947 | 5613 | 49% |

**Decode** (prompt 1024, 64 new tokens): 0.5B 36.5 → 34.9 tok/s (1.04x slower), 1.5B
29.9 → 30.2 tok/s (0.99x). Decode uses the exact fallback, so parity is the expected and
correct outcome.

**Quality — this is the important number.** On wikitext-2 the default configuration
degrades perplexity severely:

| Model | N | Baseline PPL | CEPL PPL | ΔPPL | top-1 agreement |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-0.5B | 2048 | 14.48 | 98.75 | +582% | 0% |
| Qwen2.5-0.5B | 4096 | 10.93 | 65.66 | +501% | 25% |
| Qwen2.5-1.5B | 2048 | 10.03 | 139.38 | +1290% | 25% |
| Qwen2.5-1.5B | 4096 | 7.62 | 79.01 | +937% | 25% |

Widening the exact window helps but does not fix it (Qwen2.5-1.5B, single window):

| N | exact window | CEPL PPL | ΔPPL | speedup |
| ---: | ---: | ---: | ---: | ---: |
| 2048 | 256 | 92.8 | +1225% | 5.1x |
| 2048 | 512 | 58.3 | +732% | 5.8x |
| 2048 | 1024 | 22.4 | +220% | 4.3x |
| 4096 | 1024 | 36.8 | +353% | 1.7x |

The other poolers are worse, not better: at N=2048, window 512, `max` pooling gave
+30,217% and `softmax` pooling overflowed to NaN. Mean pooling is the least destructive
of the three.

**Reading:** CEPL delivers its promised speed and memory gains, and it does so without
touching a single parameter (SHA-256 verified on both checkpoints before, during and
after injection). But a single pooled landmark per chunk of 64 is far too lossy to
preserve a pretrained model's output distribution, at any exact window that still yields
a speedup. Treat the current defaults as a correct and fast *mechanism*, not as a
quality-neutral drop-in. See `TECHNICAL_REPORT.md` §6 for the full discussion.

Interpretation notes:

- A speedup below 1 is a valid, reportable outcome; CEPL only pays off when the exact
  window is much smaller than the context.
- `ppl_delta_pct` is the honest quality cost, and here it is large.
- Padded batches and decode are intentionally *not* accelerated.

---

## Supported models and limitations

**Verified:**

- `Qwen2ForCausalLM` (Qwen2 / Qwen2.5), including GQA and explicit `head_dim`.
- Architectures with the same attention contract (Llama/Mistral) are supported in
  principle by the compatibility layer (RoPE registry, `head_dim` derivation, GQA
  helpers), but are **not yet validated end-to-end**. Do not assume compatibility.

**Limitations:**

- **Quality loss is large at the default settings.** On wikitext-2, CEPL raises
  perplexity by 5–13x (see [Benchmarks](#benchmarks)). It is a correct and fast
  approximation, not a quality-neutral one. Increase `local_window` to trade speed for
  accuracy, and measure the result on your own data before relying on it.
- CEPL is approximate; it is not equivalent to exact attention.
- A mask that is not padding-plus-causal, and any multi-token continuation with a
  non-empty KV cache, use the exact fallback. Standard padded batches stay on CEPL.
  Long decode uses cached landmarks plus an exact recent window, not full attention.
- Sliding-window models (`sliding_window` config) are not specifically handled.
- Position encodings other than RoPE (qwen2/llama registries) are unsupported.
- Chunk pooling with `mean` is lossy by design; `local_window` and `chunk_size` control
  the trade-off. Use the accuracy benchmark to choose them for your workload.
- The adapter adds a thin Python wrapper; for very short contexts this can be marginally
  slower than the native module (measured by `adapter_overhead_pct`).

---

## Project layout

```
vcpa/                      # the package
  attention.py             # CEPLAttention: wrapper, unified softmax, exact paths
  injector.py              # VCPAInjector: inject/remove/verify/report
  pooling.py               # mean/max/softmax/top-k landmark poolers
  compatibility.py         # arch detection, head_dim, RoPE registry, GQA helpers
  cache.py                 # VCPA vs exact mode decisions
  config.py                # defaults and version
tests/                     # 47 offline tests on a tiny Qwen2 model
benchmarks/                # prefill / decode / memory / accuracy harness
kaggle/                    # Kaggle notebook + one-command runner + integrity check
examples/inject_qwen.py    # minimal end-to-end usage
gen11/                     # original Generation 11 implementation (preserved)
AUDIT_GEN11.md             # audit of Generation 11 (bugs and false claims)
TECHNICAL_REPORT.md        # design, math, error analysis, limitations
```

---

## Generation 11 audit

Generation 11 is preserved unchanged in `gen11/` for comparison. Its implementation had
correctness bugs that made its reported numbers non-comparable; see
[`AUDIT_GEN11.md`](AUDIT_GEN11.md). Generation 12 is a clean re-implementation with:

- no intra-chunk future leakage (Gen 11 pooled the query's own chunk),
- no dropped remainder tokens,
- correct KV-cache handling,
- no K/V replication for GQA,
- `head_dim` derived from the model rather than hard-coded,
- padding-mask correctness (exact fallback),
- reversible injection with programmatic parameter-integrity verification.

---

## Version

`12.0.0-alpha` — see `vcpa/config.py`.

VCPA-Inject is research code. Validate accuracy and speed on your own model,
data and hardware before relying on it.
