"""Chunked-exact + pooled-landmarks (CEPL) attention for VCPA-Inject.

Gen 12 design (default ``blend_mode="unified"``):

For a sequence of ``N`` tokens split into chunks of size ``C``, a query in chunk ``c``
attends over the **union** of

  1. the exact keys/values of the ``L`` most recent chunks
     (``L = max(1, local_window // C)``), with a proper causal mask inside chunk ``c``,
     and
  2. one pooled landmark key/value for each older chunk ``m <= c - L``,
     with a mass-preserving ``+log(C)`` bias on the landmark logits.

Both sets share **one softmax**, so recent context keeps the exact distribution and
older context is compressed rather than dropped.  The exact window and the landmark set
are disjoint by construction, so nothing is double-counted.

Exact paths (correctness first):

* ``seq_len <= threshold`` -> exact SDPA;
* a mask that is not plain padding-plus-causal -> exact SDPA with that mask;
* a short cache (``kv_len <= threshold``) or a multi-token continuation -> exact SDPA;
* a long single-token decode -> landmark cache (exact recent window + stored landmarks);
* ``seq_len % C != 0`` -> the tail tokens use exact causal attention over the full past;
* both streams disabled (legacy blend mode only) -> exact SDPA.

The legacy ``blend_mode="blend"`` (two separately-softmaxed streams, convex mixture) is
kept only for comparison.  It is causal but markedly less faithful than ``"unified"``;
see ``TECHNICAL_REPORT.md``.

GQA is handled by reshaping queries to ``(B, H_kv, G, ...)`` and broadcasting KV heads;
keys/values are never replicated.  ``enable_gqa=True`` is used on the SDPA paths.

CEPL is an approximation and is never claimed to be equivalent to exact attention.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import (
    DEFAULT_BLEND_LANDMARK_WEIGHT, DEFAULT_BLEND_MODE,
    DEFAULT_CHUNK_SIZE, DEFAULT_LOCAL_WINDOW, DEFAULT_NUM_LANDMARKS, DEFAULT_THRESHOLD,
    DEFAULT_USE_LANDMARKS, DEFAULT_USE_LOCAL,
    POOL_MEAN, POOL_MULTIVEC,
)
from .compatibility import (
    enable_gqa, get_head_dim, get_model_arch, num_key_value_groups,
    rope_factory,
)
from .pooling import build_pooler


def _cache_has_content(past_key_values: Any, layer_idx: Optional[int] = None) -> bool:
    """True if *this layer's* KV cache already holds at least one past token.

    An empty cache (e.g. ``DynamicCache()`` created for a fresh prefill) must not
    disable the CEPL path.  The check is per-layer because a shared ``DynamicCache``
    is filled layer-by-layer during the same forward pass; checking the global length
    would make layer 0 look "empty" and layer 1 look "full".
    """
    if past_key_values is None:
        return False
    if layer_idx is not None:
        try:
            return int(past_key_values.get_seq_length(layer_idx)) > 0
        except TypeError:
            pass          # older signature without layer_idx
        except Exception:  # noqa: BLE001
            pass
    try:
        return int(past_key_values.get_seq_length()) > 0
    except Exception:  # noqa: BLE001 - cache object without get_seq_length
        try:
            return any(int(t.shape[-2]) > 0 for t in past_key_values.key_cache)
        except Exception:  # noqa: BLE001
            return True


class CEPLAttention(nn.Module):
    """Vectorized chunk-pooled attention wrapper around a wrapped original attention module.

    Args:
        original_attn: the pretrained attention module to wrap (Qwen2Attention, LlamaAttention, etc.).
        chunk_size: size of each pooled chunk.
        local_window: tokens of past context kept at exact token-level attention per chunk (block-diagonal).
        threshold: below this seq_len, use exact SDPA on the whole sequence.
        use_landmarks: blend in global chunk landmarks.
        use_local: blend in block-diagonal local attention.
        landmark_weight: weight of the landmark stream (1 - local_weight for the local stream).
        pool_type: ``multivec`` (default), or mean/max/softmax/topk.
        num_landmarks: vectors kept per chunk when ``pool_type="multivec"``.
    """

    def __init__(
        self,
        original_attn: nn.Module,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        local_window: int = DEFAULT_LOCAL_WINDOW,
        threshold: int = DEFAULT_THRESHOLD,
        use_landmarks: bool = DEFAULT_USE_LANDMARKS,
        use_local: bool = DEFAULT_USE_LOCAL,
        landmark_weight: float = DEFAULT_BLEND_LANDMARK_WEIGHT,
        pool_type: str = POOL_MULTIVEC,
        blend_mode: str = DEFAULT_BLEND_MODE,
        num_landmarks: int = DEFAULT_NUM_LANDMARKS,
    ):
        super().__init__()
        if chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")
        if local_window <= 0:
            raise ValueError("local_window must be a positive integer")
        if threshold <= 0:
            raise ValueError("threshold must be a positive integer")
        if not 0.0 <= landmark_weight <= 1.0:
            raise ValueError("landmark_weight must be in [0, 1]")

        # The original attention module is kept as a *plain* reference, not registered as an
        # nn.Module child. This is intentional: we must preserve the model's exact
        # state_dict key structure (no "original_attn.*" leakage) while remaining reversible.
        # nn.Module.__init__ automatically registers attributes that are nn.Module
        # instances, so we store the reference AFTER the parent __init__ ran.
        object.__setattr__(self, "original_attn", original_attn)
        self.config = original_attn.config
        self.layer_idx = original_attn.layer_idx
        self._arch = get_model_arch(original_attn)
        self.head_dim = get_head_dim(original_attn, self.config)
        self._num_key_value_groups = num_key_value_groups(original_attn, self.config)
        self._gqa_enabled = enable_gqa(original_attn, self.config)
        self.scaling = getattr(original_attn, "scaling", self.head_dim ** -0.5)

        self.chunk_size = int(chunk_size)
        self.local_window = int(local_window)
        self.threshold = int(threshold)
        self._log_chunk_size = math.log(self.chunk_size) if self.chunk_size > 1 else 0.0
        self.use_landmarks = bool(use_landmarks)
        self.use_local = bool(use_local)
        self.landmark_weight = float(landmark_weight)
        self.local_weight = 1.0 - float(landmark_weight)
        if blend_mode not in ("unified", "blend"):
            raise ValueError("blend_mode must be 'unified' or 'blend'")
        if blend_mode == "unified" and not (use_landmarks and use_local):
            raise ValueError(
                "blend_mode='unified' is a single softmax over the exact window AND the "
                "landmarks, so it requires use_local=True and use_landmarks=True. "
                "Use blend_mode='blend' to disable one stream."
            )
        self.blend_mode = blend_mode

        self.num_landmarks = max(1, int(num_landmarks))
        self.pooler = build_pooler(pool_type, num_landmarks=self.num_landmarks)
        if self.pooler is None:
            self.pooler = build_pooler(POOL_MEAN)
        # Activations, not parameters. Kept out of state_dict on purpose: a registered
        # buffer would change the checkpoint surface the injector promises to preserve.
        object.__setattr__(self, "_landmark_cache", None)

        # Cached per-layer causal masks (materialized when needed, shared across forward calls).
        self._mask_cache: dict[tuple[int, int], torch.Tensor] = {}

        # Expose the original module's parameter tree identically (by reference).
        self._share_registries_from(original_attn)

    # ------------------------------------------------------------------
    # Surface-preserving module tree.
    #
    # We keep the original module as a plain *reference* (not a registered child), then
    # share its parameter / buffer / submodule registries with this wrapper **by reference**.
    # The result is that the injected module exposes exactly the same tree as the original:
    #
    #   model.layers[i].self_attn.q_proj.weight   (identical object, not a copy)
    #   model.layers[i].self_attn.k_proj.*
    #   ...
    #
    # Concretely, for a standard Qwen2/Llama attention module:
    #   - self._modules   is the original's {q_proj, k_proj, v_proj, o_proj, ...}
    #   - self._parameters/self._buffers are the original's
    # so model.parameters(), named_parameters(), state_dict() and load_state_dict()
    # behave exactly as before injection, with zero tensor copies.
    #
    # This guarantees non-negotiable constraints #5 (state_dict key structure) and
    # #7 (no weight duplication).
    # ------------------------------------------------------------------

    # noinspection PyProtectedMember
    def _share_registries_from(self, original: nn.Module) -> None:
        """Make this module's parameter tree identical to ``original``'s, by reference."""
        self._parameters = original._parameters
        self._buffers = original._buffers
        self._modules = original._modules
        self._non_persistent_buffers_set = original._non_persistent_buffers_set

    def __getattr__(self, name: str) -> Any:
        # Delegates attributes not found on this wrapper to the wrapped original module.
        # Guard against recursion during unpickling / early construction.
        if name in ("original_attn", "__dict__", "__class__"):
            raise AttributeError(name)
        try:
            return super().__getattr__(name)
        except AttributeError:
            original = self.__dict__.get("original_attn")
            if original is None:
                raise
            return getattr(original, name)

    # ------------------------------------------------------------------
    # Public API for the inject / remove cycle
    # ------------------------------------------------------------------

    def remove(self) -> nn.Module:
        """Return the original (unwrapped) attention module for the caller to re-assign."""
        return self.original_attn

    # ------------------------------------------------------------------
    # Mask helpers (small, cached, device-aware)
    # ------------------------------------------------------------------

    def _get_chunk_causal_mask(self, n_chunks: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Causal mask over chunks: (n_queries, n_chunks) -> True means allow.

        Query token in chunk q may attend to landmark chunk c iff c <= q.
        """
        key = (n_chunks, device)
        if key in self._mask_cache:
            return self._mask_cache[key]
        q = torch.arange(n_chunks, device=device).view(n_chunks, 1)   # (C,1)
        c = torch.arange(n_chunks, device=device).view(1, n_chunks)   # (1,C)
        m = (c <= q)                                                  # (C,C): c <= q
        m = m.to(dtype=dtype)
        self._mask_cache[key] = m
        return m

    def _get_token_chunk_causal_mask(
        self, seq_len: int, n_chunks: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Mask (seq_len, n_chunks): token i may attend to chunk c iff c*C <= i."""
        key = ("token_chunk", seq_len, n_chunks, device)
        if key in self._mask_cache:
            return self._mask_cache[key]
        i = torch.arange(seq_len, device=device).view(seq_len, 1)
        starts = (torch.arange(n_chunks, device=device) * self.chunk_size).view(1, n_chunks)
        m = (starts <= i).to(dtype=dtype)
        self._mask_cache[key] = m
        return m

    def _get_local_block_mask(
        self, seq_len: int, chunk_size: int, local_window: int,
        device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Block-diagonal local mask (block-diagonal + causal + sliding window).

        Within each chunk, token i attends to tokens j in the same chunk with
        max(chunk_start, i - local_window) <= j <= i.  Shape: (seq_len, seq_len).
        """
        n_chunks = (seq_len + chunk_size - 1) // chunk_size
        blocks = []
        for c in range(n_chunks):
            start = c * chunk_size
            end = min(start + chunk_size, seq_len)
            sl = end - start
            i = torch.arange(sl, device=device).view(sl, 1)
            j = torch.arange(sl, device=device).view(1, sl)
            # causal and within a local window (window measured from chunk start here)
            blk = (j <= i) & (j > (i - local_window))
            blocks.append(blk)
        full = torch.block_diag(*blocks).to(dtype=dtype)
        return full

    # ------------------------------------------------------------------
    # Padding masks and the landmark cache
    # ------------------------------------------------------------------

    def _interpret_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        bsz: int,
        q_len: int,
        kv_len: int,
        device: torch.device,
    ) -> tuple[Optional[torch.Tensor], bool]:
        """Reduce a mask to per-key validity, or refuse it.

        Returns ``(key_valid, exotic)``. ``key_valid`` is ``(B, kv_len)`` True
        for real tokens, or None when every key is real. ``exotic`` is True when
        the mask is not padding-plus-causal; the caller must then use exact SDPA
        rather than invent a sparse pattern the mask does not have.

        A 4D mask is accepted only if the rows we sample match
        ``j <= query_position AND valid[j]``. That is the mask Hugging Face builds
        for a padded causal batch. A sliding-window or document mask fails the
        sample and stays on the exact path.
        """
        if attention_mask is None:
            return None, False
        mask = attention_mask
        if mask.dim() == 2:
            keep = mask[:, -kv_len:].to(device=device)
            valid = keep if keep.dtype == torch.bool else keep > 0
            if bool(valid.all()):
                return None, False
            return valid, False
        if mask.dim() != 4:
            return None, True
        mask = mask.to(device=device)
        if mask.shape[-1] >= kv_len:
            mask = mask[..., -kv_len:]
        else:
            return None, True
        if mask.shape[-2] == 1:
            row = mask[:, 0, 0, :]
            valid = row if row.dtype == torch.bool else row > -1.0e4
            if bool(valid.all()):
                return None, False
            return valid, False

        # Square prefill, or a continuation whose queries sit at the end of the cache.
        if mask.shape[-2] != q_len:
            return None, True
        if mask.dtype == torch.bool:
            allowed = mask[:, 0]
        else:
            allowed = mask[:, 0] > -1.0e4
        q_pos0 = kv_len - q_len
        diag_q = torch.arange(q_len, device=device)
        diag_k = diag_q + q_pos0
        in_range = diag_k < kv_len
        valid = torch.ones(bsz, kv_len, dtype=torch.bool, device=device)
        valid[:, diag_k[in_range]] = allowed[:, diag_q[in_range], diag_k[in_range]]
        # Sample rows instead of materialising an (N, N) expected mask.
        sample = {0, q_len // 2, q_len - 1}
        keys = torch.arange(kv_len, device=device).view(1, -1)
        for row_i in sample:
            q_pos = q_pos0 + row_i
            expected = (keys <= q_pos) & valid
            if not torch.equal(allowed[:, row_i, :], expected):
                return None, True
        if bool(valid.all()):
            return None, False
        return valid, False

    def _valid_window(
        self, key_valid: torch.Tensor, n_recent: int, n_chunks: int, chunk: int,
    ) -> torch.Tensor:
        """Validity of the exact-window keys, aligned with ``k_win``.

        Returns ``(B, 1, 1, n_chunks, 1, n_recent * chunk)`` so it broadcasts
        over heads, groups and the query position inside the chunk.
        """
        bsz = key_valid.shape[0]
        usable = n_chunks * chunk
        valid = key_valid[:, :usable].view(bsz, 1, 1, n_chunks, chunk)
        padded = F.pad(valid, (0, 0, n_recent, 0))
        window = padded.unfold(3, n_recent, 1)[:, :, :, 1:n_chunks + 1]
        window = window.permute(0, 1, 2, 3, 5, 4)
        return window.reshape(bsz, 1, 1, n_chunks, 1, n_recent * chunk)

    def _remember_landmarks(self, past_key_values, key, value, key_valid) -> None:
        """Keep the landmarks just computed, keyed to this cache object."""
        packed = getattr(self, "_last_landmarks", None)
        if packed is None:
            return
        pk, pv, mass = packed
        object.__setattr__(self, "_landmark_cache", {
            "cache_id": id(past_key_values),
            "pk": pk, "pv": pv, "mass": mass,
            "n_chunks": pk.shape[2],
            "batch": pk.shape[0],
        })

    def _ensure_landmarks(self, past_key_values, key, value, key_valid, n_complete: int):
        """Return landmarks for the first ``n_complete`` chunks, building only new ones."""
        cache = getattr(self, "_landmark_cache", None)
        same = (
            cache is not None
            and cache.get("cache_id") == id(past_key_values)
            and cache.get("batch") == key.shape[0]
            and cache["pk"].device == key.device
            and cache["pk"].dtype == key.dtype
        )
        if not same:
            cache = None
        have = 0 if cache is None else int(cache["n_chunks"])
        if have > n_complete:
            # Cache shrank (a new, shorter sequence reused the object). Rebuild.
            cache = None
            have = 0
        if have < n_complete:
            start = have * self.chunk_size
            end = n_complete * self.chunk_size
            valid = None if key_valid is None else key_valid[:, start:end]
            pk, pv, mass = self.pooler.pool_landmarks(
                key[:, :, start:end], value[:, :, start:end], self.chunk_size, valid=valid,
            )
            pk, pv, mass = pk.detach(), pv.detach(), mass.detach()
            if cache is None:
                cache = {
                    "cache_id": id(past_key_values),
                    "pk": pk, "pv": pv, "mass": mass,
                    "n_chunks": n_complete,
                    "batch": key.shape[0],
                }
            else:
                cache = {
                    **cache,
                    "pk": torch.cat([cache["pk"], pk], dim=2),
                    "pv": torch.cat([cache["pv"], pv], dim=2),
                    "mass": torch.cat([cache["mass"], mass], dim=2),
                    "n_chunks": n_complete,
                }
            object.__setattr__(self, "_landmark_cache", cache)
        return cache

    def _landmark_decode(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_valid: Optional[torch.Tensor],
        input_shape: Tuple[int, ...],
        past_key_values: Any,
        **gqa_kw: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """One decode step over the exact recent window plus cached landmarks.

        The new token is already in ``key``/``value`` (the cache was updated by
        the caller, exactly as Qwen2 does). When that token finishes a chunk,
        only that chunk is pooled and appended. Older landmarks are not rebuilt.

        This is still an approximation. It matches the prefill rule: a query in
        chunk ``c`` sees exact tokens of chunks ``c-L+1 .. c`` and landmarks of
        chunks ``m <= c-L``. It does not claim to equal full attention.
        """
        bsz, n_head, _q_len, head_dim = query.shape
        kv_len = key.shape[2]
        n_kv = key.shape[1]
        groups = n_head // n_kv
        C = self.chunk_size
        pos = kv_len - 1
        chunk = pos // C
        n_recent = max(1, min(self.local_window // C if self.local_window >= C else 1, chunk + 1))
        exact_start = max(0, (chunk - n_recent + 1) * C)
        n_land_chunks = max(0, chunk - n_recent + 1)
        n_complete = kv_len // C

        k_exact = key[:, :, exact_start:kv_len, :]
        v_exact = value[:, :, exact_start:kv_len, :]
        q = query.view(bsz, n_kv, groups, 1, head_dim)
        scores_exact = torch.matmul(
            q.float(), k_exact.float().unsqueeze(2).transpose(-1, -2),
        ) * self.scaling
        allow_exact = torch.ones(
            bsz, n_kv, 1, 1, k_exact.shape[2], dtype=torch.bool, device=query.device,
        )
        if key_valid is not None:
            allow_exact = allow_exact & key_valid[:, exact_start:kv_len].view(bsz, 1, 1, 1, -1)

        parts_s = [scores_exact]
        parts_v = [v_exact.unsqueeze(2).expand(-1, -1, groups, -1, -1)]
        parts_a = [allow_exact]
        if n_land_chunks > 0:
            cache = self._ensure_landmarks(
                past_key_values, key, value, key_valid, n_complete,
            )
            pk = cache["pk"][:, :, :n_land_chunks].reshape(bsz, n_kv, 1, -1, head_dim)
            pv = cache["pv"][:, :, :n_land_chunks].reshape(bsz, n_kv, 1, -1, head_dim)
            mass = cache["mass"][:, :, :n_land_chunks].reshape(bsz, n_kv, 1, 1, -1)
            scores_land = torch.matmul(q.float(), pk.float().transpose(-1, -2)) * self.scaling
            scores_land = scores_land + torch.log(mass.clamp(min=1).float())
            parts_s.append(scores_land)
            parts_v.append(pv.expand(-1, -1, groups, -1, -1))
            parts_a.append((mass > 0).view(bsz, n_kv, 1, 1, -1))

        scores = torch.cat(parts_s, dim=-1)
        allow = torch.cat(parts_a, dim=-1)
        row_ok = allow.any(dim=-1, keepdim=True)
        scores = scores.masked_fill(~allow, torch.finfo(torch.float32).min)
        scores = scores.masked_fill(~row_ok, 0.0)
        attn = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
        attn = attn.masked_fill(~row_ok, 0.0)
        values = torch.cat(parts_v, dim=3)
        out = torch.matmul(attn, values).reshape(bsz, n_head, 1, head_dim)
        # A chunk may have completed on this step. Pool it now so the next step
        # does not rescan the prefix. The current query did not use it (its own
        # chunk is inside the exact window).
        if n_complete > 0:
            self._ensure_landmarks(past_key_values, key, value, key_valid, n_complete)
        out = out.transpose(1, 2).contiguous().reshape(*input_shape, -1)
        return self.o_proj(out), None

    # ------------------------------------------------------------------
    # Main forward: mirrors the Qwen2/Llama attention module interface
    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Any = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        bsz, seq_len = input_shape
        hidden_shape = (*input_shape, -1, self.head_dim)

        # --- projections (original pretrained Linear modules, unchanged) ---
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # --- RoPE (applied inside the attention module, as in Qwen2/Llama/Mistral) ---
        if position_embeddings is not None:
            cos, sin = position_embeddings
            rope_fn = rope_factory(self.original_attn)
            if rope_fn is not None:
                query_states, key_states = rope_fn(cos, sin, query_states, key_states)

        # --- decide CEPL vs exact BEFORE updating the cache ---------------------
        # A shared DynamicCache is filled layer-by-layer, so the per-layer length must
        # be sampled before this layer's update.
        object.__setattr__(self, "_last_landmarks", None)
        has_past = _cache_has_content(past_key_values, self.layer_idx)
        if past_key_values is not None and not has_past:
            # New sequence: the cache object is empty. Drop landmarks from the previous one.
            object.__setattr__(self, "_landmark_cache", None)

        # --- KV cache (exactly as the original module does: update BEFORE attention) ---
        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx
            )

        # --- GQA: native enable_gqa, never replicate ---
        gqa_kw: dict[str, Any] = {}
        if self._gqa_enabled:
            gqa_kw["enable_gqa"] = True

        kv_len = key_states.shape[2]
        key_valid, exotic_mask = self._interpret_mask(
            attention_mask, bsz, seq_len, kv_len, hidden_states.device,
        )

        # Exact causal path is used when:
        #   * the sequence is short, OR
        #   * the mask is not a plain padding+causal mask (we will not guess), OR
        #   * a multi-token continuation reuses a non-empty cache, OR
        #   * decode is still inside the exact threshold, OR
        #   * both approximation streams are disabled.
        # A 2D/4D *padding* mask no longer forces this path: CEPL applies it per key.
        # A long single-token decode uses the landmark cache instead of full SDPA.
        long_decode = has_past and seq_len == 1 and kv_len > self.threshold
        use_exact = (
            exotic_mask
            or (has_past and seq_len > 1)
            or (not long_decode and seq_len <= self.threshold)
            or (long_decode and self.blend_mode != "unified")
            or (not self.use_landmarks and not self.use_local)
        )
        if long_decode and not use_exact:
            return self._landmark_decode(
                query_states, key_states, value_states, key_valid,
                input_shape, past_key_values, **gqa_kw,
            )
        if use_exact:
            mask = attention_mask
            if mask is not None and mask.dim() == 2:
                # (B, K) padding mask -> (B, 1, 1, K) boolean mask for SDPA
                mask = mask[:, None, None, :].to(torch.bool)
            is_causal = False
            if mask is None:
                kv_len = key_states.shape[2]
                if seq_len == 1:
                    # Single-token decode: the one query sees the whole cache, so no
                    # mask is required (and building one per step would waste time).
                    is_causal = False
                elif has_past and kv_len > seq_len:
                    # Continuation prefill without an explicit mask: align the causal
                    # mask to the bottom-right of the (query, kv) rectangle.
                    i = torch.arange(seq_len, device=hidden_states.device).view(-1, 1) + (kv_len - seq_len)
                    j = torch.arange(kv_len, device=hidden_states.device).view(1, -1)
                    mask = j <= i
                else:
                    is_causal = seq_len > 1
            attn_output = F.scaled_dot_product_attention(
                query_states, key_states, value_states,
                attn_mask=mask,
                dropout_p=0.0,
                scale=self.scaling,
                is_causal=is_causal,
                **gqa_kw,
            )
            attn_output = attn_output.transpose(1, 2).contiguous().reshape(*input_shape, -1)
            return self.o_proj(attn_output), None

        # --- long prefill: CEPL, padding applied inside the score mask ---
        out, _weights = self._forward_cepl(
            query_states, key_states, value_states,
            input_shape, hidden_states.device, key_valid=key_valid, **gqa_kw,
        )
        if past_key_values is not None:
            self._remember_landmarks(past_key_values, key_states, value_states, key_valid)
        return out, _weights

    # ------------------------------------------------------------------
    # CEPL core
    # ------------------------------------------------------------------

    def _forward_cepl(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        input_shape: Tuple[int, ...],
        device: torch.device,
        key_valid: Optional[torch.Tensor] = None,
        **gqa_kw: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, seq_len = input_shape
        C = self.chunk_size
        n_full = (seq_len // C) * C

        if n_full == 0:
            # Sequence shorter than one chunk: nothing to pool, use exact attention.
            short_mask = None
            is_causal = True
            if key_valid is not None:
                i = torch.arange(seq_len, device=device).view(1, -1, 1)
                j = torch.arange(key_states.shape[2], device=device).view(1, 1, -1)
                short_mask = (j <= (i + key_states.shape[2] - seq_len)) & key_valid[:, None, :]
                is_causal = False
            out = F.scaled_dot_product_attention(
                query_states, key_states, value_states,
                attn_mask=short_mask, dropout_p=0.0, scale=self.scaling,
                is_causal=is_causal, **gqa_kw,
            )
            out = out.transpose(1, 2).contiguous().reshape(*input_shape, -1)
            return self.o_proj(out), None

        # --- main chunk-aligned prefix: chunked-exact + pooled landmarks ---
        q_main = query_states[:, :, :n_full, :]
        k_main = key_states[:, :, :n_full, :]
        v_main = value_states[:, :, :n_full, :]
        valid_main = None if key_valid is None else key_valid[:, :n_full]
        out_main = self._cepl_prefill(
            q_main, k_main, v_main, device, key_valid=valid_main, **gqa_kw,
        )

        # --- remainder tokens: EXACT causal attention over the full past ---
        # (never dropped, unlike Gen 11).  Only `seq_len - n_full` queries are computed.
        if n_full < seq_len:
            out_rem = self._remainder_exact(
                query_states[:, :, n_full:, :], key_states, value_states, n_full,
                key_valid=key_valid, **gqa_kw,
            )
            out = torch.cat([out_main, out_rem], dim=2)
        else:
            out = out_main

        out = out.transpose(1, 2).contiguous().reshape(*input_shape, -1)
        return self.o_proj(out), None

    def _cepl_prefill(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        device: torch.device,
        key_valid: Optional[torch.Tensor] = None,
        **gqa_kw: Any,
    ) -> torch.Tensor:
        """Run one CEPL forward over a chunk-aligned sequence.

        ``blend_mode``:
          * ``"unified"``  (default): a **single softmax** over the union of
            (a) exact keys of the L most recent chunks and (b) pooled landmarks of
            older chunks.  This is the principled sparse-attention formulation and
            preserves the exact distribution for recent context.
          * ``"blend"``: legacy convex blend of two independently-softmaxed streams.
            Cheaper conceptually but much less faithful; kept for comparison.
        """
        if getattr(self, "blend_mode", "unified") == "unified":
            return self._unified_attention(q, k, v, device, key_valid=key_valid, **gqa_kw)

        out_local = self._local_stream(q, k, v, device, **gqa_kw) if self.use_local else None
        if self.use_landmarks:
            out_land = self._landmark_attention(q, k, v, device, key_valid=key_valid, **gqa_kw)
            n_chunks = q.shape[2] // self.chunk_size
            # Landmarks are visible to a query in chunk c only for chunks m < c.  Chunk 0
            # has no past, so its landmark stream is undefined; those rows keep the exact
            # local output (or, when use_local is off, an exact causal fallback).
            valid = (torch.arange(n_chunks, device=device) >= 1)
            valid_tok = valid.repeat_interleave(self.chunk_size).view(1, 1, -1, 1)
            valid_tok = valid_tok.to(out_land.dtype)
            if out_local is None:
                out = out_land * (self.landmark_weight * valid_tok)
                q0 = q[:, :, :self.chunk_size, :]
                out[:, :, :self.chunk_size, :] = self._remainder_exact(q0, k, v, 0, **gqa_kw)
                return out
            w_land = self.landmark_weight * valid_tok
            return (1.0 - w_land) * out_local + w_land * out_land
        return out_local  # type: ignore[return-value]

    def _unified_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        device: torch.device,
        key_valid: Optional[torch.Tensor] = None,
        **gqa_kw: Any,
    ) -> torch.Tensor:
        """Single-softmax local-exact + global-landmark attention.

        For a query in chunk ``c`` the key set is:
          * exact tokens of chunks ``c-L+1 .. c``  (causal inside chunk c), and
          * pooled landmarks of chunks ``0 .. c-L`` (chunks strictly older than the
            exact window, so no information is double-counted).

        Cost per query chunk: ``L*C + n_chunks`` keys instead of ``N``.
        Peak score memory is O(B*H*N*(L*C + N/C)) instead of O(B*H*N^2).

        GQA is handled by broadcasting each KV head across its query group; no
        key/value replication is performed.
        """
        bsz, n_head, seq_len, head_dim = query.shape
        C = self.chunk_size
        n_chunks = seq_len // C
        n_kv = key.shape[1]
        groups = n_head // n_kv

        # Number of recent chunks kept at exact token-level resolution.
        L = max(1, min(self.local_window // C if self.local_window >= C else 1, n_chunks))

        qc = query.view(bsz, n_kv, groups, n_chunks, C, head_dim)
        kc = key.view(bsz, n_kv, 1, n_chunks, C, head_dim)
        vc = value.view(bsz, n_kv, 1, n_chunks, C, head_dim)

        # ---------------- exact local window over L recent chunks ----------------
        # Pad L empty chunks at the front.  Chunk c lives at padded index L+c, so the
        # window that *ends* at chunk c starts at padded index c+1.
        # NOTE: Tensor.unfold appends the window dimension at the END, hence the permute.
        kpad = F.pad(kc, (0, 0, 0, 0, L, 0))          # (B,Hkv,1,nc+L,C,D)
        vpad = F.pad(vc, (0, 0, 0, 0, L, 0))
        k_win = kpad.unfold(3, L, 1)[:, :, :, 1:n_chunks + 1]   # (B,Hkv,1,nc,C,D,L)
        v_win = vpad.unfold(3, L, 1)[:, :, :, 1:n_chunks + 1]
        k_win = k_win.permute(0, 1, 2, 3, 6, 4, 5)             # (B,Hkv,1,nc,L,C,D)
        v_win = v_win.permute(0, 1, 2, 3, 6, 4, 5)
        total = L * C
        k_win = k_win.reshape(bsz, n_kv, 1, n_chunks, total, head_dim)
        v_win = v_win.reshape(bsz, n_kv, 1, n_chunks, total, head_dim)

        # Scores are computed in float32.  fp16 overflows to inf once a dot product
        # exceeds 65504, which is routine for RoPE-rotated keys of a real pretrained
        # model (observed on Qwen2.5-1.5B), and softmax(inf) is NaN.  SDPA upcasts
        # internally; a hand-rolled score path has to do it explicitly.
        scores_exact = torch.matmul(qc.float(), k_win.float().transpose(-1, -2)) * self.scaling
        # (B,Hkv,G,nc,C,total)

        # mask_exact: (nc, C, total)
        l_idx = torch.arange(L, device=device)                      # (L,)
        c_idx = torch.arange(n_chunks, device=device)               # (nc,)
        src_chunk = c_idx.view(n_chunks, 1) - L + 1 + l_idx.view(1, L)   # (nc,L)
        allowed_chunk = src_chunk >= 0                              # (nc,L)
        i = torch.arange(C, device=device).view(C, 1)
        j = torch.arange(C, device=device).view(1, C)
        causal_in_chunk = (j <= i)                                  # (C,C)
        # For each window slot: current chunk (slot L-1) is causal, older slots full.
        slot_mask = torch.ones((L, C, C), dtype=torch.bool, device=device)
        slot_mask[L - 1] = causal_in_chunk
        mask_exact = (
            allowed_chunk.view(n_chunks, 1, L, 1, 1)               # (nc,1,L,1,1)
            & slot_mask.view(1, 1, L, C, C)                        # (1,1,L,C,C)
        ).squeeze(1).permute(0, 2, 1, 3).reshape(n_chunks, C, total)

        # ---------------- global landmarks over older chunks ----------------
        # K vectors per chunk. Bias is log(mass) per vector, not log(C) on every
        # one of them: K copies of log(C) would overweight the old context by K.
        pk, pv, mass = self.pooler.pool_landmarks(key, value, C, valid=key_valid)
        # Detached activations for the decode cache. Not a parameter, not a buffer.
        object.__setattr__(self, "_last_landmarks", (
            pk.detach(), pv.detach(), mass.detach(),
        ))
        k_land = pk.shape[3]
        pk = pk.reshape(bsz, n_kv, 1, 1, n_chunks * k_land, head_dim)
        pv = pv.reshape(bsz, n_kv, 1, 1, n_chunks * k_land, head_dim)
        scores_land = torch.matmul(qc.float(), pk.float().transpose(-1, -2)) * self.scaling
        # (B,Hkv,G,nc,C,nc*K)
        log_mass = torch.log(mass.clamp(min=1).float()).reshape(bsz, n_kv, 1, 1, 1, -1)
        scores_land = scores_land + log_mass

        # mask_land: query chunk c sees landmark chunk m iff m <= c-L, for every
        # of that chunk's K vectors.
        m_idx = torch.arange(n_chunks, device=device).view(1, 1, n_chunks)   # (1,1,nc)
        c_col = c_idx.view(n_chunks, 1, 1)                                    # (nc,1,1)
        mask_land = (m_idx <= (c_col - L)).expand(n_chunks, C, n_chunks)
        mask_land = mask_land.unsqueeze(-1).expand(n_chunks, C, n_chunks, k_land)
        mask_land = mask_land.reshape(n_chunks, C, n_chunks * k_land)

        # ---------------- join and single softmax ----------------
        scores = torch.cat([scores_exact, scores_land], dim=-1)
        allow = torch.cat([
            mask_exact.view(1, 1, 1, n_chunks, C, total),
            mask_land.view(1, 1, 1, n_chunks, C, n_chunks * k_land),
        ], dim=-1)
        allow = allow.expand(bsz, n_kv, 1, n_chunks, C, total + n_chunks * k_land).clone()
        if key_valid is not None:
            valid_win = self._valid_window(key_valid, L, n_chunks, C)
            allow[..., :total] = allow[..., :total] & valid_win
        mass_ok = (mass > 0).reshape(bsz, n_kv, 1, 1, 1, n_chunks * k_land)
        allow[..., total:] = allow[..., total:] & mass_ok
        row_ok = allow.any(dim=-1, keepdim=True)
        scores = scores.masked_fill(~allow, torch.finfo(torch.float32).min)
        scores = scores.masked_fill(~row_ok, 0.0)
        attn = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
        attn = attn.masked_fill(~row_ok, 0.0)

        values = torch.cat(
            [v_win.expand(-1, -1, groups, -1, -1, -1),
             pv.expand(-1, -1, groups, n_chunks, -1, -1)],
            dim=4,
        )  # (B,Hkv,G,nc,total+nc*K,D)

        out = torch.matmul(attn, values)                                      # (B,Hkv,G,nc,C,D)
        return out.reshape(bsz, n_head, seq_len, head_dim)

    def _local_stream(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        device: torch.device,
        **gqa_kw: Any,
    ) -> torch.Tensor:
        """Exact causal attention inside each chunk (block-diagonal).

        Implemented as a single grouped matmul over blocks:
            (B, Hkv, G, n_chunks, C, D) @ (B, Hkv, 1, n_chunks, C, D)^T
        Memory is O(B*Hq*N*C) instead of O(B*Hq*N*N), and GQA needs no K/V
        replication because each KV head is broadcast across its query group.
        """
        bsz, n_head, seq_len, head_dim = query.shape
        C = self.chunk_size
        n_chunks = seq_len // C
        n_kv = key.shape[1]
        groups = n_head // n_kv

        q_b = query.view(bsz, n_kv, groups, n_chunks, C, head_dim)
        k_b = key.view(bsz, n_kv, 1, n_chunks, C, head_dim)
        v_b = value.view(bsz, n_kv, 1, n_chunks, C, head_dim)

        scores = torch.matmul(q_b.float(), k_b.float().transpose(-1, -2)) * self.scaling

        idx = torch.arange(C, device=device)
        causal = (idx.view(C, 1) >= idx.view(1, C)).view(1, 1, 1, 1, C, C)
        scores = scores.masked_fill(~causal, torch.finfo(torch.float32).min)

        attn = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
        out = torch.matmul(attn, v_b)
        return out.reshape(bsz, n_head, seq_len, head_dim)

    def _landmark_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        device: torch.device,
        key_valid: Optional[torch.Tensor] = None,
        **gqa_kw: Any,
    ) -> torch.Tensor:
        """Causal attention of each query to pooled landmarks of chunks <= its own chunk.

        GQA is handled by reshaping queries into (H_kv, G) groups and broadcasting the
        pooled keys/values: no K/V replication, no extra memory.
        """
        bsz, n_head, seq_len, head_dim = query.shape
        C = self.chunk_size
        n_chunks = seq_len // C

        pk, pv, mass = self.pooler.pool_landmarks(key, value, C, valid=key_valid)
        n_kv = pk.shape[1]
        groups = n_head // n_kv
        k_land = pk.shape[3]
        pk = pk.reshape(bsz, n_kv, n_chunks * k_land, head_dim)
        pv = pv.reshape(bsz, n_kv, n_chunks * k_land, head_dim)

        q_grouped = query.view(bsz, n_kv, groups, seq_len, head_dim)
        pk_b = pk.unsqueeze(2)                              # (B, H_kv, 1, nc*K, D)
        scores = torch.matmul(q_grouped.float(), pk_b.float().transpose(-1, -2)) * self.scaling
        log_mass = torch.log(mass.clamp(min=1).float()).reshape(bsz, n_kv, 1, 1, -1)
        scores = scores + log_mass

        # A query token in chunk c may use landmarks of chunks m < c only.  The query's own
        # chunk is covered by the exact local stream; pooling it would leak future tokens.
        token_chunk = torch.arange(seq_len, device=device) // C          # (N,)
        m_idx = torch.arange(n_chunks, device=device).view(1, n_chunks)  # (1, nc)
        mask = (m_idx < token_chunk.view(seq_len, 1))                    # (N, nc)
        mask = mask.unsqueeze(-1).expand(seq_len, n_chunks, k_land).reshape(seq_len, -1)
        mass_ok = (mass > 0).reshape(bsz, n_kv, 1, 1, n_chunks * k_land)
        allow = mask.view(1, 1, 1, seq_len, -1) & mass_ok
        row_ok = allow.any(dim=-1, keepdim=True)
        scores = scores.masked_fill(~allow, torch.finfo(torch.float32).min)
        scores = scores.masked_fill(~row_ok, 0.0)
        attn = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
        attn = attn.masked_fill(~row_ok, 0.0)
        out = torch.matmul(attn, pv.unsqueeze(2))           # (B, H_kv, G, N, D)
        return out.reshape(bsz, n_head, seq_len, head_dim)

    def _remainder_exact(
        self,
        q_rem: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        offset: int,
        key_valid: Optional[torch.Tensor] = None,
        **gqa_kw: Any,
    ) -> torch.Tensor:
        """Exact causal attention for the remainder tokens over the full sequence.

        Token at global position ``offset + i`` attends to every key ``j <= offset + i``.
        """
        n_rem = q_rem.shape[2]
        n_kv_total = key.shape[2]
        device = q_rem.device
        i = torch.arange(n_rem, device=device).view(n_rem, 1) + offset          # (R,1)
        j = torch.arange(n_kv_total, device=device).view(1, n_kv_total)         # (1,N)
        allow = j <= i                                                         # (R,N) bool
        if key_valid is not None:
            allow = allow.unsqueeze(0) & key_valid[:, None, :]
        return F.scaled_dot_product_attention(
            q_rem, key, value,
            attn_mask=allow, dropout_p=0.0, scale=self.scaling,
            is_causal=False, **gqa_kw,
        )

    # Backwards-compatible helper used by earlier prototypes / tests.
    def _block_sdpa(self, query, key, value, causal: bool, **gqa_kw: Any) -> torch.Tensor:
        return F.scaled_dot_product_attention(
            query, key, value, attn_mask=None, dropout_p=0.0,
            scale=self.scaling, is_causal=causal, **gqa_kw,
        )
