# VCPA-Inject — Technical Report (Generation 12)

**Status:** `12.0.0-alpha` · research prototype · benchmarked on a Tesla T4 (see §6).

---

## Abstract

VCPA-Inject is a training-free runtime adapter that replaces full quadratic self-attention
in a pretrained causal language model with **CEPL** (*Chunked-Exact + Pooled-Landmarks*)
attention. Recent tokens are attended exactly; older context is summarised by one pooled
landmark per chunk; both sets share a **single softmax**. The adapter introduces no
parameters, copies no weights, preserves the `state_dict` surface, and is fully
reversible. Exact attention is retained for short sequences, padded batches, decoding,
and cache continuations, so causal correctness is never traded for speed. We derive the
mass-preserving `log C` landmark bias, analyse the approximation error, document the
failure modes, and provide an offline test-suite plus a GPU benchmark harness.

---

## 1. Motivation

Exact causal self-attention costs `O(N²)` time and memory in the sequence length `N`.
For long-context inference the attention score matrix dominates both latency and peak
memory, even though most queries place negligible mass on most keys.

Existing remedies fall into three families:

1. **Train a new model** with sparse/linear attention — expensive, changes weights.
2. **Approximate at inference** with sliding windows, sinks, or block-sparse patterns —
   cheap, but usually drops distant context entirely.
3. **Compress the KV cache** (quantisation, eviction, merging) — effective, but often
   intrusive and hard to reverse.

VCPA-Inject targets a fourth point: a **reversible, parameter-preserving runtime
adapter** that keeps a configurable recent window exact and compresses only the distant
past into landmarks, with an explicit, measurable approximation error.

The non-negotiable constraints for this generation were:

- preserve every pretrained parameter value;
- introduce no trainable parameters and no weight copies;
- preserve the `state_dict` key structure;
- remain reversible (`inject` / `remove`);
- never sacrifice causal correctness;
- always keep a fallback to the original attention;
- verify integrity programmatically;
- report honestly when the method is slower or worse.

---

## 2. Method

### 2.1 Notation

- `N` sequence length, `C` chunk size, `L` number of exact chunks,
  `W = L·C` exact window, `H` query heads, `H_kv` KV heads, `G = H/H_kv`,
  `d` head dimension, `s = 1/√d`.

Chunks are `[0,C), [C,2C), …`. A query in chunk `c` may attend to keys in chunks `≤ c`
(causality).

### 2.2 Key set

For a query in chunk `c`:

```
K_exact(c)    = tokens of chunks max(0, c-L+1) … c
K_landmark(c) = pooled landmarks of chunks 0 … c-L
```

The two sets are disjoint by construction: landmarks stop exactly where the exact window
starts. No token is counted twice, and no chunk is omitted.

### 2.3 Single softmax with mass-preserving bias

Exact attention is

```
y_i = Σ_j exp(s_ij) v_j / Σ_j exp(s_ij),      s_ij = s·q_i·k_j.
```

For an old chunk `m` with `C` keys, suppose the scores within the chunk are approximately
constant around `s̄ = s·q_i·k̄_m` (the score against the mean key). Then

```
Σ_{j∈m} exp(s_ij) ≈ C · exp(s̄) = exp(s̄ + log C).
```

So replacing the chunk by a single landmark requires adding `log C` to its logit. The
implementation therefore computes

```
logits = [ s·q·K_exact ,  s·q·K_landmark + log C ]
p      = softmax(logits)
y      = p · [ V_exact , V_landmark ].
```

**Why this matters.** Without the bias, each old chunk contributes the mass of *one* key
instead of `C`, systematically under-weighting distant context. In early prototypes this
alone produced ~0.75 relative logit error at `N = 128–512`. With the bias and the single
softmax, the error at `N = 128` (fully inside the exact window) is at float32 noise level,
and long-context error is dominated by within-chunk score variance rather than a global
mass shift.

### 2.4 Error intuition

A second-order expansion of `log Σ_j exp(s_j)` around `s̄` gives

```
log Σ_j exp(s_j) ≈ s̄ + log C + Var(s)/2,
```

so the landmark approximation errs by roughly `Var_j(s_ij)/2` in log-mass per chunk and
by the within-chunk covariance between `exp(s)` and `v` in the numerator. Both vanish
when scores and values are homogeneous inside a chunk, i.e. when `C` is small relative to
the semantic granularity of the data. This is why `C` and `W` are exposed as knobs and
why the accuracy benchmark reports both perplexity and distribution divergence.

### 2.5 Exact paths

CEPL is used **only** when all of the following hold:

- `seq_len > threshold`;
- no `attention_mask` is provided (padded batches are not supported by CEPL);
- this layer's KV cache is empty (fresh prefill);
- at least one of `use_local` / `use_landmarks` is enabled.

Otherwise the adapter calls `torch.nn.functional.scaled_dot_product_attention` on the
original `q/k/v`, which is numerically the baseline computation:

| Situation | Path |
| :--- | :--- |
| `N ≤ threshold` | exact SDPA |
| padding / 4D mask present | exact SDPA with the given mask |
| empty cache, long prefill | **CEPL** |
| cache already has past tokens | exact SDPA; bottom-right causal mask if none supplied |
| single-token decode | exact SDPA over cached KV |
| tail `N mod C ≠ 0` | exact causal attention over the full past |
| both streams disabled | exact SDPA |

The bottom-right mask matters: for a continuation with `q_len > 1` and `kv_len > q_len`,
PyTorch's `is_causal=True` top-left alignment would be wrong. The adapter detects this
case and builds an explicit `(q_len, kv_len)` mask.

### 2.6 Grouped-query attention

For `H_kv < H`, queries are viewed as `(B, H_kv, G, N, d)` and KV as
`(B, H_kv, 1, N, d)`. Broadcasting performs GQA without materialising repeated K/V.
SDPA paths use `enable_gqa=True`.

### 2.7 Decode

Decoding is **not** approximated. The adapter calls
`past_key_values.update(k, v, layer_idx)` exactly as the native module does, then runs
exact SDPA. The expected decode throughput is therefore baseline-level; the harness
measures it rather than assuming it.

---

## 3. Complexity

Per query chunk with exact window `W` and chunk size `C`:

| Quantity | Exact attention | CEPL |
| :--- | :--- | :--- |
| Keys per query | `N` | `W + N/C` |
| Time | `O(N² d)` | `O(N (W + N/C) d)` |
| Score memory | `O(B H N²)` | `O(B H N (W + N/C))` |
| KV memory | `O(B H_kv N d)` | unchanged |

CEPL is beneficial when `W + N/C < N`, i.e. roughly `N > W·C/(C-1)`.

---

## 4. Injection mechanism

`CEPLAttention` is an `nn.Module` that holds the original attention module as a **plain,
non-registered reference** (`object.__setattr__`, so `nn.Module` does not add an
`original_attn.*` branch to the tree). To make the parameter surface identical to the
original, the wrapper **aliases the original module's registries by reference**:

```python
self._parameters = original._parameters
self._buffers = original._buffers
self._modules = original._modules
self._non_persistent_buffers_set = original._non_persistent_buffers_set
```

Consequences:

- `model.parameters()` yields the same tensor objects, same names, same count;
- `model.state_dict()` has the same keys and values;
- `model.load_state_dict()` still works while injected;
- `remove()` returns the original module object, so restoration is exact.

`__getattr__` delegates any other attribute (`config`, `layer_idx`, `num_key_value_heads`,
…) to the original module.

An earlier attempt overrode `named_modules` / `named_parameters`; that broke PyTorch's
internal `named_modules(memo, prefix, remove_duplicate)` signature and produced duplicate
or missing parameters. Registry aliasing is simpler and strictly better.

---

## 5. Experimental protocol

All measurements are produced by `benchmarks/` and `kaggle/run_on_kaggle.py`; no numbers
are hard-coded in this repository.

**Integrity** (`kaggle/check_integrity.py`): SHA-256 per parameter and per `state_dict`
entry, before / while injected / after removal, plus parameter counts and layer counts.

**Accuracy** (`benchmark_accuracy.py`): wikitext-2 test perplexity over non-overlapping
windows, plus next-token divergence at each window boundary:

- mean and max `|Δlogit|`,
- top-1 agreement,
- symmetric KL,
- cosine similarity.

**Prefill** (`benchmark_prefill.py`): median latency over warm-up + repeats with CUDA
synchronisation, for baseline, CEPL, and "forced exact" injected mode (isolates adapter
overhead from algorithmic saving), plus peak memory.

**Decode** (`benchmark_decode.py`): greedy decode with KV cache; tokens/s baseline vs
injected. Expect ratio ≈ 1.0.

**Memory** (`benchmark_memory.py`): peak CUDA allocated memory per context length.

Recommended Kaggle run:

```bash
python kaggle/run_on_kaggle.py \
    --models Qwen/Qwen2.5-0.5B Qwen/Qwen2.5-1.5B \
    --seq-lens 512 1024 2048 4096 \
    --memory-seq-lens 1024 2048 4096 8192
```

and a configuration sweep (chunk size × exact window) reported by the notebook.

---

## 6. Results

Measured on a Tesla T4 (16 GB), torch 2.11.0+cu130, transformers 5.22.1, fp16, batch
size 1, defaults `C=64`, `W=256`, `threshold=256`, `blend_mode="unified"`, `pool="mean"`.
Every row reported `parameters_unchanged: true`, and `kaggle/check_integrity.py`
confirmed SHA-256 equality of all parameters and of the `state_dict` before, during and
after injection on both checkpoints.

| Model | N | baseline s | CEPL s | speedup | base MB | CEPL MB | ΔPPL % | top-1 % |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-0.5B | 2048 | 0.445 | 0.190 | 2.35x | 1632 | 1632 | +582 | 0 |
| Qwen2.5-0.5B | 4096 | 1.550 | 0.389 | 3.99x | 3283 | 2259 | +501 | 25 |
| Qwen2.5-1.5B | 2048 | 0.781 | 0.479 | 1.63x | 3727 | 3726 | +1290 | 25 |
| Qwen2.5-1.5B | 4096 | 2.509 | 0.910 | 2.76x | 5144 | 4356 | +937 | 25 |

Perplexity is wikitext-2-raw-v1 (test split), 4 non-overlapping windows. Decode
throughput (prompt 1024, 64 tokens) was 1.04x and 0.99x of baseline for the two models,
i.e. the exact decode fallback costs nothing measurable.

### 6.2 The quality result, stated plainly

The speed and memory columns match the complexity analysis. The perplexity column does
not match the hope. A pooled landmark per 64-token chunk discards enough information
that the pretrained next-token distribution collapses: top-1 agreement at the window
boundary falls to 0–25%.

This is not an implementation defect. Three checks argue against that:

1. The approximation error is exactly zero for `N ≤ W` on a random model (§6.1), so the
   mask, the RoPE application and the softmax are correct.
2. Enlarging the exact window reduces the error monotonically but slowly. On
   Qwen2.5-1.5B at N=2048, windows of 256, 512 and 1024 tokens gave +1225%, +732% and
   +220% perplexity. The speedup at the 1024 window was still 4.3x, but +220% is not a
   usable model.
3. The pooling strategy is not the lever. At N=2048 with a 512 window, `max` pooling
   gave +30,217% and `softmax`-weighted pooling overflowed to NaN. Mean pooling, the
   default, is the least bad of the three.

The conclusion is about the method, not the code: **one landmark per chunk cannot
represent a chunk's keys well enough to preserve a pretrained model's output, at any
compression ratio that is still worth computing.** The `log C` bias corrects the mean
probability mass of a landmark; it cannot correct the variance lost by averaging 64
distinct keys into one.

What would have to change: either many landmarks per chunk (defeating the compression),
a window so large that little is compressed (defeating the speedup), or landmarks that
are trained rather than pooled. None of those is a small change, and none is claimed
here.

Reporting rules used in this project:

- If CEPL is slower, report the slowdown.
- If perplexity degrades more than acceptable, report it and prefer a larger
  `local_window` or smaller `chunk_size`.
- Never compare numbers across different GPUs, dtypes, or batch sizes.
- Always report `parameters_unchanged`.

### 6.1 Mechanics validation (not a language-quality benchmark)

To verify that the implementation is numerically well-behaved before spending GPU time,
the unified path was compared against exact SDPA on a **4-layer randomly-initialised
Qwen2** (hidden 128, 4 heads / 2 KV heads, random token ids·). This checks the
implementation and the `log C` bias; it says nothing about language quality.

| C | W | N | mean relative logit error |
| ---: | ---: | ---: | ---: |
| 64 | 256 | 128 | 0.0000 (inside the exact window) |
| 64 | 256 | 256 | 0.0000 (inside the exact window) |
| 64 | 256 | 512 | 0.0023 |
| 64 | 256 | 1024 | 0.0040 |
| 64 | 128 | 256 | 0.0026 |
| 64 | 128 | 512 | 0.0056 |
| 64 | 128 | 1024 | 0.0073 |

The error is exactly zero while `N ≤ W`, as expected from the construction, and grows
slowly beyond it. Real perplexity/divergence numbers must still come from
`benchmark_accuracy.py` on a pretrained checkpoint.

· The random model's logits are near-uniform (`ppl ≈ vocab_size`), so perplexity and
top-1 agreement on it are not informative and are deliberately not reported.

---

## 7. Known limitations and failure modes

1. **Approximation error** grows with within-chunk heterogeneity, i.e. larger `C` and more
   diverse content. The `log C` bias corrects the mean mass, not the variance.
2. **Exotic masks are not supported by CEPL.** A mask that is not padding-plus-causal
   still forces exact attention. Standard 2D/4D padding does not.
3. **Decode is approximate once the cache exceeds `threshold`.** Each new token attends
   to the exact recent window plus landmarks cached from completed chunks. Short decode
   and multi-token continuations stay exact. The landmark path is not claimed to match
   full attention.
4. **Cache continuation with `q_len > 1`** is exact; long continuation prefills therefore
   pay full attention cost.
5. **Sliding-window models** (`sliding_window`) are not specifically handled; the original
   module's window semantics are not reproduced by CEPL.
6. **Non-RoPE position encodings** and architectures outside the qwen2/llama registry are
   unsupported.
7. **Adapter overhead**: a thin Python wrapper exists even on exact paths. On very short
   sequences it can be marginally slower than the native module.
8. **`mean` pooling is a lossy summary.** `max`, `softmax` and `topk` poolers exist but
   have not been validated at scale.
9. **Dtype range**: attention scores are accumulated in float32. In fp16 a single
   RoPE-rotated query-key dot product routinely exceeds 65504, which overflows to
   infinity and turns `softmax` into NaN — this was observed on Qwen2.5-1.5B and fixed by
   upcasting the score computation, exactly as SDPA does internally. The `log C` bias
   itself is small and is not the source of the overflow.
10. **No training-time support.** The wrapper is inference-oriented; gradients may flow
    through the original projections, but the approximation is not designed or tested for
    training.

---

## 8. Generation 11 audit (summary)

`gen11/VCPA-Efficient-Attention.py` is preserved unchanged. The audit
([`AUDIT_GEN11.md`](AUDIT_GEN11.md)) found:

- pooling the query's own chunk leaked future tokens within that chunk;
- remainder tokens (`N mod C`) were dropped;
- `past_key_values` was ignored, breaking generation;
- `local_window` was unused;
- device was a global constant;
- GQA used `repeat_interleave` (K/V replication);
- `head_dim` was hard-coded to 64;
- `attention_mask` was ignored;
- replacement was irreversible;
- no parameter-integrity verification;
- benchmarks ran in training mode with gradients, inflating memory comparisons.

Generation 12 fixes each of these and adds a regression test that the CEPL path is
actually exercised (guarding against a silent exact-path fallback).

---

## 9. Future work

- Padded-batch CEPL with per-sequence chunk masks.
- Landmark compression of the KV cache itself (so decode can also benefit).
- Learned or score-aware landmarks (top-k / attention-weighted) validated at scale.
- Chunked decode: approximate over landmark memory during generation.
- Validation on Llama/Mistral checkpoints and on non-RoPE architectures.
- Block-sparse kernels (FlexAttention / FlashAttention varlen) for the exact window.

---

## Appendix A — Derivation of the landmark bias

Let a chunk contain scores `s_1…s_C` and values `v_1…v_C`. Exact contribution:

```
numerator   = Σ_j e^{s_j} v_j
denominator = Σ_j e^{s_j}
```

Landmark approximation uses `k̄ = (1/C)Σ_j k_j`, `v̄ = (1/C)Σ_j v_j`, and score
`s̄ = s·q·k̄`. With one landmark key in the softmax, its unnormalised weight is `e^{s̄}`.
Matching the denominator to first order:

```
Σ_j e^{s_j} = e^{s̄} Σ_j e^{s_j - s̄} ≈ e^{s̄} · C · (1 + Var(s)/2) ≈ e^{s̄ + log C}.
```

Hence the `+ log C` bias. The numerator uses `v̄`, which is exact when `v_j` is constant
within the chunk and otherwise incurs a covariance error.

## Appendix B — Why a single softmax beats a convex blend

Blending two independently normalised outputs,

```
y = α·softmax(QK_exact)V_exact + (1-α)·softmax(QK_land)V_land,
```

forces each stream to sum to one, so neither can dominate even when one is clearly more
relevant; the result is a fixed mixture that distorts the attention distribution at every
query. A single softmax over the union lets the model choose per query and preserves the
exact distribution whenever the exact window already covers the context. Empirically the
blend produced ~0.74 relative logit error at `N = 128–512`, while the unified formulation
is exact when `N ≤ W` and substantially closer beyond it.

## Appendix C — Reproducing the offline test-suite

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

The suite builds a tiny Qwen2 model from a config (no downloads), with GQA, RoPE and a KV
cache, and asserts the guarantees listed in the README. It runs in about two seconds and
must pass before any GPU run is trusted.
