# VCPA-Inject — Phase 1 Audit Report: Generation 11

**Repository:** VCPA-Efficient-Attention.py (standalone script, no modular structure)
**Target:** `Qwen/Qwen2.5-0.5B` — 24 layers, hidden=896, 14 heads, 2 KV heads (GQA, 7 groups), head_dim=64, no sliding window
**Transformers version used:** 5.22.1 (CPU torch 2.15.0)

---

## 1. Repository structure

The repo contains only two files: `VCPA-Efficient-Attention.py` (the Gen 11 implementation + a benchmark) and `README.md`.
No tests, no modular code, no installation requirements, no environment file, no LICENSE, and no `requirements.txt`.
The README's "code.py / GEN11_REPORT.md / results.json" artifacts are referenced but do not exist in the repo.

---

## 2. Mathematical formulation of Gen 11

For `seq_len > chunk_size`:

1. Only the first `trunc_len = floor(seq_len / chunk_size) * chunk_size` tokens are used; tokens
   at positions `trunc_len .. seq_len-1` are **silently dropped**.
2. Keys and values are reshaped into chunks of size `chunk_size` and **mean-pooled**, producing
   per-chunk landmark keys/values of shape `(B, H, N/C)`.
3. Full query sequence is multiplied against the *chunked* key landmarks to form an
   `(B, H, N, N/C)` score matrix.
4. **Chunk-level causal masking:** `mask_global[i, c] = -inf` when `i < c*C`. Tokens are allowed
   to attend to chunks `0 … floor(i/C)` (their own chunk and all earlier ones). This is correct
   **at chunk granularity**, but chunk `floor(i/C)` is mean-pooled over *all* its tokens, including
   tokens that come **after** position `i`. Tokens early in a chunk therefore attend to the mean of
   their own future → intra-chunk future leak (see §3.1).
5. Softmax, `nan_to_num` (replaces NaN with 0), matmul with chunk values → output.

The remainder of the sequence (`trunc_len … N-1`) is never pooled, so those tokens are absent from
every landmark. Every query also loses token-level detail: all keys/values inside a chunk are collapsed
to a single mean-pooled landmark, and there is no exact local stream. This is an approximation; it is
**not** equivalent to exact self-attention.

---

## 3. Implementation errors and bugs

### 3.1 Current-chunk pooling leaks future tokens (critical)

```python
token_indices     = torch.arange(seq_len).view(1, 1, seq_len, 1)                 # i
chunk_end_indices = (torch.arange(num_chunks) + 1).view(1, 1, 1, num_chunks) * C # (c+1)*C
mask_global       = token_indices < (chunk_end_indices - C)                      # i < c*C
scores_global     = scores_global.masked_fill(mask_global, -inf)
```

Entries are masked when `i < c*C`, so token `i` may attend to chunks `c ≤ floor(i/C)`: its own chunk
and every earlier one. That part is correct. The problem is **what a landmark represents**.

`k_chunked[c]` is the mean of the **whole** chunk `c` (all `C` tokens). For a query token `i` inside
chunk `c_i`, the allowed landmark `c_i` was built from tokens `c_i*C … c_i*C + C - 1`, including
positions `> i`. The output at position `i` therefore depends on **future tokens**:

```
token i=0, C=64  ->  attends to the mean of tokens 0..63  (future leak)
```

This is not a rounding issue; it changes the model's output whenever the suffix of a chunk changes.
A causal model must never do this. Gen 12 fixes it by keeping the query's own chunk at exact
token-level resolution with a proper causal mask, and by pooling **only strictly older chunks**
(`m ≤ c - L`, where `L` chunks are kept exact).

A second, non-causal-but-lossy consequence: even for the last token of a chunk (where pooling is
causally safe), the entire chunk is reduced to one vector, so token-level structure is lost. There is
no exact local stream in Gen 11, despite `local_window` being accepted by the constructor.

### 3.2 Remainder tokens silently dropped (critical)

Only `key_states[:, :, :trunc_len]` is pooled. Tokens beyond `trunc_len` contribute nothing to the
output. For seq_len=4096, chunk_size=64: 64 tokens (1.5%) dropped; for seq_len not divisible, more.

### 3.3 KV cache / `past_key_values` completely ignored

`past_key_values` is accepted as a parameter but never read. Generation with `generate(...)` would
recompute the entire history each step, destroying autoregression and making long generation impossible.

### 3.4 `local_window` parameter unused

The constructor accepts `local_window=128` but it never enters the forward path. The README and the code
promise "multi-scale" / local attention that is not implemented.

### 3.5 Global DEVICE assumption

```python
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
token_indices = torch.arange(seq_len, device=DEVICE)
```

Device is fixed at import time. If the model is moved to another device (`.to("cuda")`, `device_map`,
or `torch.compile`), the mask indices would be on the wrong device → runtime error or silent
misbehavior. Attention masks must be created on `hidden_states.device`.

### 3.6 GQA handling is wasteful

```python
if self.num_key_value_groups > 1:
    key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
    value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)
```

Keys/values are replicated from 2 KV heads to 14 heads (7x memory increase for the K/V tensors) and
then pooled. The SDPA path supports native GQA (`enable_gqa=True`), so this replication is unnecessary.
For large KV budgets this inflates memory use precisely when VCPA aims to reduce it.

### 3.7 `head_dim` / `num_heads` defaults are hard-coded

```python
self.head_dim = getattr(original_module, "head_dim", 64)
self.num_heads = getattr(original_module, "num_heads", 14)
```

Hard-coded defaults `64` and `14` are the Qwen2.5-0.5B values. Any model with a different geometry
(e.g. Llama-3.1 uses `head_dim=128`) silently computes wrong shapes or wrong attention. Both must be
derived from the config/module (as Gen 12 does).

### 3.8 Attention mask ignored

`attention_mask` is accepted but never used. Padding masks / batched masks with different lengths
are therefore silently broken.

### 3.9 Not reversible

The benchmark does `layer.self_attn = VCPAQwen2Attention(...)` — the original module object is
replaced in place and garbage-collected. There is no way to recover the original attention class,
no `remove()` API, no preservation of the original module object.

### 3.10 Parameter integrity not preserved / verified

Weights are *shared* (references), so numerically unchanged, but there is no verification of
integrity (no hashes, no `torch.equal` checks, no state_dict key-structure validation).

### 3.11 Returns attention weights as `None`

The Qwen2 forward calls the attention interface and captures `(attn_output, attn_weights)`. SDPA always
returns `None` for weights, so this is compatible, but the module should document this behavior.

### 3.12 `torch.nan_to_num(attn_global, nan=0.0)` masks numerical issues instead of fixing them

If the mask blocks all attendable positions for some token, softmax produces NaN, and `nan_to_num`
silently zeroes it. Better to guard against all-masked rows.

### 3.13 No batch support verification

`bsz` is used correctly in vectorized ops, so batching is fine in principle, but nothing tests it.

### 3.14 RoPE handling is Qwen2-specific and brittle

It imports `apply_rotary_pos_emb` from `transformers.models.qwen2.modeling_qwen2`. For other
architectures (Llama, Mistral) the function lives in a different module and the position-embedding
format differs (`rotary_emb.forward` vs `(cos, sin)` tuple). No architecture detection.

### 3.15 Benchmark methodology flaws

- The "baseline" at 4096 is reported as OOM while the actual test sets `model_baseline.train()` and
  does `loss.backward()`, a training graph far larger than inference. Not comparable to the "optimized"
  run which also trains. Both run training, so the *relative* claim (Gen 11 survived where baseline OOM'd)
  is internally consistent but does not prove memory efficiency of the attention algorithm itself,
  because backward on the chunked ops may use different memory than exact attention backward.
- **Both runs use `device_map=DEVICE`** and `train()` mode, so weight gradients are computed; the claim
  "training-free" is contradicted by the benchmark setting.
- Single-run timings, no warmup, no `torch.cuda.synchronize()`, no `inference_mode`.
- No output quality comparison, no perplexity, no latency breakdown (prefill vs decode).

**Bottom line:** The README benchmark claims ("48.22% VRAM reduction", "2.7x faster") are not
independently reproducible from this code and mix training-mode memory with inference benchmarks.

---

## 4. Architectural limitations vs. the VCPA-Inject goals

| Requirement | Gen 11 status |
|---|---|
| No parameter modification | ✅ weights shared, never overwritten |
| Preserves original modules | ❌ replaces in place; original lost |
| Reversible injection | ❌ no remove() |
| No weight duplication | ⚠️ GQA replication inflates KV memory |
| Compatible with HF attention interface | ⚠️ accepts the right kwargs but ignores several (mask, pkv) |
| GQA/MQA support | ⚠️ works only via wasteful replication |
| RoPE support | ⚠️ Qwen2-only import path |
| Causal masking | ❌ chunk mask is correct, but pooling the current chunk leaks intra-chunk future tokens (3.1) |
| KV cache / autoregression | ❌ ignored |
| Remainder tokens | ❌ dropped |
| local_window | ❌ ignored |
| Padding masks | ❌ ignored |
| Multi-device / dtype | ⚠️ global DEVICE constant |
| Chunk size divides sequence | ❌ remainder dropped |
| Parameter integrity verification | ❌ none |
| Modular / testable | ❌ single script |

---

## 5. Design decisions for VCPA-Inject

**Injection approach: attention adapter/wrapper (Approach B).**
Wrap the original attention module in a subclass instead of replacing it:

- Original module (`original_attn`) retained → reversible via `remove()`.
- q/k/v/o_proj remain the original Linear modules under the same attribute names → state_dict key
  structure preserved (`layer.self_attn.q_proj.weight` unchanged).
- No weights copied or modified → `torch.jit` / `state_dict()` integrity verified programmatically.
- The wrapper replicates the exact interface expected by the Transformers attention dispatch:
  `forward(hidden_states, position_embeddings, attention_mask, past_key_values, **kwargs)`.

**Algorithm (Gen 12):** "Chunked exact + pooled landmarks" (CEPL):

- `seq_len <= threshold` → exact SDPA (`is_causal`) on the full sequence (matches baseline exactly).
- `seq_len > threshold`:
  - Split into full chunks of size `C` plus a remainder chunk of size `< C`.
  - Remainder: exact token-level causal attention (no pooling loss on the tail).
  - **Exact window:** the `L = max(1, local_window // C)` most recent chunks are kept at exact
    token-level resolution, with a proper causal mask inside the current chunk.
  - **Landmarks:** one pooled key/value per chunk for chunks `0 … c-L` (strictly older than the exact
    window), so the current chunk is never pooled and no future token can leak in.
  - **Single softmax:** the exact keys and the landmarks share one softmax, with a mass-preserving
    `+log(C)` bias on landmark logits (a landmark stands for `C` original keys). This is the shipped
    `blend_mode="unified"`. The earlier convex two-softmax "blend" is retained only for comparison.
- **GQA:** never replicate; reshape queries to `(B, H_kv, G, …)` and broadcast KV; pass
  `enable_gqa=True` to SDPA where applicable.
- **RoPE:** apply via the architecture-specific function (Qwen2: `apply_rotary_pos_emb`; Llama/Mistral:
  `apply_rotary_pos_emb` from `modeling_llama`/`modeling_mistral`).
- **Masking:** build masks on `hidden_states.device`; respect `attention_mask` (padded batches fall
  back to exact SDPA); never let a query see future positions (verified by test).
- **Decode fallback:** during autoregressive generation the history grows; recomputing pooled landmarks
  each step is O(seq_len) work per token. VCPA is therefore used only in prefill (when
  `past_key_values` is absent/empty and seq_len large); decode runs through exact SDPA using the
  cached KV. This is explicitly documented; no unsupported claim of full-generation equivalence.

**Modularity:** pooling strategies are pluggable classes under `vcpa/pooling.py`; attention backends
under `vcpa/attention.py`; architecture adapters under `vcpa/compatibility.py`; an `injector.py` exposes
`VCPAInjector(model)` → `.inject()` / `.remove()` / `.state()` APIs.

---

## 6. Remaining open questions

1. Best pooling function: mean vs max vs learned-weighted vs top-k per chunk? (investigate in Phase 3)
2. Optimal `local_window` size and whether local+landmark blending hurts quality.
3. Whether chunk-pooled attention during a *large prefill* meaningfully reduces KV memory (the KV
   cache is not allocated until decode in HF's eager path, so the main win is attention-matrix memory).
4. Numerical stability of softmax-weighted pooling at float16.
