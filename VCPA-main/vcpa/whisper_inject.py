"""Runtime VCPA injection for OpenAI Whisper.

VCPA-Inject (see README.md and TECHNICAL_REPORT.md) wraps a pretrained causal
language model's attention without copying or modifying a single parameter.
Whisper is not that architecture, so this module is the adaptation, not a call
to ``VCPAInjector``:

* Whisper attention is ``MultiHeadAttention`` (``query`` / ``key`` / ``value`` /
  ``out``), not Hugging Face ``q_proj`` / ``k_proj``. Positional information is
  an absolute sinusoid added *before* the block. RoPE is not applied here.
* The audio encoder is bidirectional self-attention over 1500 frames. CEPL's
  causal mask would hide future audio and is not used. The same single-softmax
  construction is kept, with a centered exact window plus landmarks of every
  chunk outside that window.
* The text decoder's self-attention is causal. It uses the Generation-12
  unified CEPL rule unchanged: exact tokens of the ``L`` most recent chunks,
  landmarks of strictly older chunks, one softmax, ``log(mass)`` bias.
* Cross-attention has no query-aligned window (a text position is not an audio
  frame). Completed audio chunks are compressed to landmarks; the unaligned
  tail is attended exactly so no frame is dropped. This path is an adaptation
  and is more lossy than self-attention CEPL. ``approximate_cross=False`` keeps
  it exact.
* The KV cache is a forward hook on the original ``key`` / ``value`` modules.
  Those modules stay the same objects, and projections still go through them,
  so ``install_kv_cache_hooks`` and beam rearrangement keep working.
* ``disable_sdpa()`` forces the original exact path and returns real QK scores.
  Word-timestamp alignment depends on that.

Nothing in this file writes to a parameter. ``inject()`` / ``remove()`` are
reversible. CEPL is an approximation; it is not claimed to match Whisper's
output distribution. Short sequences, and any call made while SDPA is disabled,
stay on the original implementation.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import (
    DEFAULT_BLEND_MODE,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_LOCAL_WINDOW,
    DEFAULT_NUM_LANDMARKS,
    DEFAULT_THRESHOLD,
    DEFAULT_USE_LANDMARKS,
    DEFAULT_USE_LOCAL,
)
from .pooling import POOL_MULTIVEC, build_pooler


_CEPL_CLS = None


class _CEPLImpl:
    """Method body of the Whisper attention wrapper.

    The concrete class is created at inject time so it can subclass
    ``whisper.model.MultiHeadAttention``. That keeps ``isinstance`` checks in
    ``install_kv_cache_hooks`` working without copying weights.
    """

    is_vcpa = True

    def __init__(
        self,
        original_attn: nn.Module,
        *,
        kind: str,
        chunk_size: int,
        local_window: int,
        threshold: int,
        use_landmarks: bool,
        use_local: bool,
        pool_type: str,
        num_landmarks: int,
        blend_mode: str,
        approximate_cross: bool,
        approximate_encoder: bool,
        approximate_decoder: bool,
    ):
        # Skip MultiHeadAttention.__init__: it would allocate new Linear layers.
        nn.Module.__init__(self)
        if blend_mode != "unified":
            raise ValueError(
                "Whisper injection implements blend_mode='unified' only "
                "(single softmax over the exact window and the landmarks)."
            )
        object.__setattr__(self, "original_attn", original_attn)
        # Share the original registries by reference. state_dict keys, parameter
        # identity, and the key/value modules the KV-cache hooks close over stay
        # exactly the pretrained objects.
        object.__setattr__(self, "_parameters", original_attn._parameters)
        object.__setattr__(self, "_buffers", original_attn._buffers)
        object.__setattr__(self, "_modules", original_attn._modules)
        object.__setattr__(
            self, "_non_persistent_buffers_set", original_attn._non_persistent_buffers_set
        )

        self.n_head = int(original_attn.n_head)
        n_state = int(original_attn.query.weight.shape[0])
        if n_state % self.n_head != 0:
            raise ValueError(
                f"n_state {n_state} is not divisible by n_head {self.n_head}"
            )
        self.head_dim = n_state // self.n_head
        self.scaling = self.head_dim ** -0.5
        self.kind = kind
        self.chunk_size = int(chunk_size)
        self.local_window = int(local_window)
        self.threshold = int(threshold)
        self.use_landmarks = bool(use_landmarks)
        self.use_local = bool(use_local)
        self.blend_mode = blend_mode
        self.approximate_cross = bool(approximate_cross)
        self.approximate_encoder = bool(approximate_encoder)
        self.approximate_decoder = bool(approximate_decoder)
        self.num_landmarks = max(1, int(num_landmarks))
        self.pooler = build_pooler(pool_type, num_landmarks=self.num_landmarks)
        self.cepl_calls = 0

    def remove(self) -> nn.Module:
        """Return the original attention module for the caller to re-assign."""
        return self.original_attn

    def forward(
        self,
        x: torch.Tensor,
        xa: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        # Same projection order as MultiHeadAttention.forward. Calling key/value
        # is what fires Whisper's KV-cache hooks; do not project twice.
        q = self.query(x)
        if kv_cache is None or xa is None or self.key not in kv_cache:
            src = x if xa is None else xa
            k = self.key(src)
            v = self.value(src)
        else:
            k = kv_cache[self.key]
            v = kv_cache[self.value]

        if xa is not None:
            attn_kind = "cross"
        elif mask is not None:
            attn_kind = "causal"
        else:
            attn_kind = "bidirectional"

        if self._use_exact(attn_kind, q, k):
            wv, qk = self.original_attn.qkv_attention(q, k, v, mask)
            return self.out(wv), qk

        qh = self._to_heads(q)
        kh = self._to_heads(k)
        vh = self._to_heads(v)
        if attn_kind == "cross":
            out = self._cross_cepl(qh, kh, vh)
        elif attn_kind == "causal" and kh.shape[-2] > qh.shape[-2] and qh.shape[-2] == 1:
            out = self._causal_decode(qh, kh, vh)
        elif attn_kind == "causal":
            out = self._causal_prefill(qh, kh, vh)
        else:
            out = self._bidirectional_prefill(qh, kh, vh)
        self.cepl_calls += 1
        return self.out(self._from_heads(out)), None

    def _use_exact(self, attn_kind: str, q: torch.Tensor, k: torch.Tensor) -> bool:
        # Word timestamps call disable_sdpa() and read the returned QK matrix.
        # That call must be the original attention, not an approximation.
        from whisper.model import MultiHeadAttention

        if not MultiHeadAttention.use_sdpa:
            return True
        if not self.use_landmarks or not self.use_local:
            # Unified CEPL is one softmax over both streams. A disabled stream
            # is not implemented here; the original attention is the safe path.
            if attn_kind != "cross" or not self.use_landmarks:
                return True
        q_len = int(q.shape[1])
        kv_len = int(k.shape[1])
        if attn_kind == "cross":
            return (not self.approximate_cross) or kv_len <= self.threshold or kv_len < self.chunk_size
        if attn_kind == "causal":
            if not self.approximate_decoder:
                return True
            has_past = kv_len > q_len
            if has_past and q_len != 1:
                return True
            if has_past:
                return kv_len <= self.threshold
            return q_len <= self.threshold or q_len < self.chunk_size
        if not self.approximate_encoder:
            return True
        return kv_len <= self.threshold or kv_len < self.chunk_size or q_len != kv_len

    def _to_heads(self, t: torch.Tensor) -> torch.Tensor:
        b, length, _ = t.shape
        return t.view(b, length, self.n_head, self.head_dim).permute(0, 2, 1, 3)

    def _from_heads(self, t: torch.Tensor) -> torch.Tensor:
        return t.permute(0, 2, 1, 3).contiguous().flatten(start_dim=2)

    def _attend(
        self,
        scores: torch.Tensor,
        allow: torch.Tensor,
        values: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Single softmax. ``allow`` True means the key is visible."""
        if allow.shape != scores.shape:
            allow = allow.expand_as(scores)
        row_ok = allow.any(dim=-1, keepdim=True)
        scores = scores.masked_fill(~allow, torch.finfo(torch.float32).min)
        scores = scores.masked_fill(~row_ok, 0.0)
        attn = F.softmax(scores, dim=-1, dtype=torch.float32).to(dtype)
        attn = attn.masked_fill(~row_ok, 0.0)
        return torch.matmul(attn, values)

    def _window_L(self, n_chunks: int) -> int:
        if self.local_window >= self.chunk_size:
            raw = self.local_window // self.chunk_size
        else:
            raw = 1
        return max(1, min(raw, n_chunks))

    def _pool(self, k: torch.Tensor, v: torch.Tensor, n_tokens: int):
        """Landmarks of the first ``n_tokens`` keys. Cached by storage identity.

        A decode step that reuses the same KV tensor (Whisper cross-attention)
        does not re-pool. A grown or rearranged cache is a new tensor and misses.
        """
        k_main = k[:, :, :n_tokens]
        v_main = v[:, :, :n_tokens]
        sig = (
            k_main.data_ptr(),
            tuple(k_main.shape),
            tuple(k_main.stride()),
            str(k_main.dtype),
            str(k_main.device),
        )
        if self.__dict__.get("_landmark_sig") == sig:
            return self.__dict__["_landmark_tensors"]
        pk, pv, mass = self.pooler.pool_landmarks(
            k_main.contiguous(), v_main.contiguous(), self.chunk_size
        )
        packed = (pk.detach(), pv.detach(), mass.detach())
        self.__dict__["_landmark_sig"] = sig
        self.__dict__["_landmark_tensors"] = packed
        return packed

    def _causal_exact_mask(self, n_chunks: int, L: int, C: int, device: torch.device) -> torch.Tensor:
        """(n_chunks, C, L*C) True where the exact window may be attended.

        Slot ``s`` is source chunk ``c - L + 1 + s``. The current chunk (last
        slot) is causal; older slots in the window are fully visible. Slots that
        fall before chunk 0 are masked — those are the left padding.
        """
        c = torch.arange(n_chunks, device=device).view(n_chunks, 1, 1, 1)
        s = torch.arange(L, device=device).view(1, 1, L, 1)
        i = torch.arange(C, device=device).view(1, C, 1, 1)
        j = torch.arange(C, device=device).view(1, 1, 1, C)
        src = c - L + 1 + s
        in_range = src >= 0
        is_current = s == (L - 1)
        causal = j <= i
        allow = in_range & torch.where(is_current, causal, torch.ones_like(causal))
        return allow.reshape(n_chunks, C, L * C)

    def _causal_land_mask(
        self, n_chunks: int, L: int, C: int, k_land: int, device: torch.device
    ) -> torch.Tensor:
        """(n_chunks, C, n_chunks*K) True for landmarks strictly older than the window."""
        c = torch.arange(n_chunks, device=device).view(n_chunks, 1)
        m = torch.arange(n_chunks, device=device).view(1, n_chunks)
        allow = m <= (c - L)
        allow = allow.view(n_chunks, 1, n_chunks, 1).expand(n_chunks, C, n_chunks, k_land)
        return allow.reshape(n_chunks, C, n_chunks * k_land)

    def _causal_aligned(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        bsz, n_head, seq_len, head_dim = q.shape
        C = self.chunk_size
        n_chunks = seq_len // C
        L = self._window_L(n_chunks)

        qc = q.reshape(bsz, n_head, n_chunks, C, head_dim)
        kc = k.reshape(bsz, n_head, n_chunks, C, head_dim)
        vc = v.reshape(bsz, n_head, n_chunks, C, head_dim)

        kpad = F.pad(kc, (0, 0, 0, 0, L, 0))
        vpad = F.pad(vc, (0, 0, 0, 0, L, 0))
        k_win = kpad.unfold(2, L, 1)[:, :, 1:n_chunks + 1]
        v_win = vpad.unfold(2, L, 1)[:, :, 1:n_chunks + 1]
        k_win = k_win.permute(0, 1, 2, 5, 3, 4).reshape(bsz, n_head, n_chunks, L * C, head_dim)
        v_win = v_win.permute(0, 1, 2, 5, 3, 4).reshape(bsz, n_head, n_chunks, L * C, head_dim)

        scores_exact = torch.matmul(qc.float(), k_win.float().transpose(-1, -2)) * self.scaling
        mask_exact = self._causal_exact_mask(n_chunks, L, C, q.device)

        pk, pv, mass = self._pool(k, v, seq_len)
        k_land = pk.shape[3]
        pk_f = pk.reshape(bsz, n_head, 1, n_chunks * k_land, head_dim)
        pv_f = pv.reshape(bsz, n_head, 1, n_chunks * k_land, head_dim)
        scores_land = torch.matmul(qc.float(), pk_f.float().transpose(-1, -2)) * self.scaling
        log_mass = torch.log(mass.clamp(min=1).float()).reshape(bsz, n_head, 1, 1, -1)
        scores_land = scores_land + log_mass
        mask_land = self._causal_land_mask(n_chunks, L, C, k_land, q.device)

        scores = torch.cat([scores_exact, scores_land], dim=-1)
        allow = torch.cat(
            [
                mask_exact.view(1, 1, n_chunks, C, L * C),
                mask_land.view(1, 1, n_chunks, C, n_chunks * k_land),
            ],
            dim=-1,
        ).expand(bsz, n_head, n_chunks, C, -1).clone()
        mass_ok = (mass > 0).reshape(bsz, n_head, 1, 1, n_chunks * k_land)
        allow[..., L * C:] = allow[..., L * C:] & mass_ok

        values = torch.cat(
            [v_win, pv_f.expand(bsz, n_head, n_chunks, -1, head_dim)],
            dim=3,
        )
        out = self._attend(scores, allow, values, q.dtype)
        return out.reshape(bsz, n_head, seq_len, head_dim)

    def _remainder(
        self,
        q_rem: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        offset: int,
        *,
        causal: bool,
    ) -> torch.Tensor:
        """Exact attention for the non-aligned tail. Tokens are never dropped."""
        if not causal:
            return F.scaled_dot_product_attention(
                q_rem, k, v, dropout_p=0.0, scale=self.scaling, is_causal=False
            )
        n_rem = q_rem.shape[2]
        n_kv = k.shape[2]
        i = torch.arange(n_rem, device=q_rem.device).view(1, 1, n_rem, 1) + offset
        j = torch.arange(n_kv, device=q_rem.device).view(1, 1, 1, n_kv)
        allow = j <= i
        return F.scaled_dot_product_attention(
            q_rem, k, v, attn_mask=allow, dropout_p=0.0, scale=self.scaling, is_causal=False
        )

    def _causal_prefill(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        C = self.chunk_size
        n_full = (q.shape[2] // C) * C
        out_main = self._causal_aligned(q[:, :, :n_full], k[:, :, :n_full], v[:, :, :n_full])
        if n_full == q.shape[2]:
            return out_main
        out_rem = self._remainder(q[:, :, n_full:], k, v, n_full, causal=True)
        return torch.cat([out_main, out_rem], dim=2)

    def _causal_decode(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """One new token over the exact recent window plus older landmarks.

        Matches the prefill rule: a query in chunk ``c`` sees exact tokens of
        chunks ``c-L+1 .. c`` (the new token is the last key, so the slice is
        already causal) and landmarks of chunks ``m <= c-L``.
        """
        bsz, n_head, _, head_dim = q.shape
        kv_len = k.shape[2]
        C = self.chunk_size
        chunk = (kv_len - 1) // C
        L = self._window_L(chunk + 1)
        exact_start = max(0, (chunk - L + 1) * C)
        n_land = max(0, chunk - L + 1)

        k_exact = k[:, :, exact_start:kv_len]
        v_exact = v[:, :, exact_start:kv_len]
        scores = torch.matmul(q.float(), k_exact.float().transpose(-1, -2)) * self.scaling
        values = v_exact
        allow = torch.ones(scores.shape, dtype=torch.bool, device=q.device)

        if n_land > 0:
            pk, pv, mass = self._pool(k, v, n_land * C)
            k_land = pk.shape[3]
            pk = pk.reshape(bsz, n_head, n_land * k_land, head_dim)
            pv = pv.reshape(bsz, n_head, n_land * k_land, head_dim)
            scores_l = torch.matmul(q.float(), pk.float().transpose(-1, -2)) * self.scaling
            scores_l = scores_l + torch.log(mass.clamp(min=1).float()).reshape(bsz, n_head, 1, -1)
            scores = torch.cat([scores, scores_l], dim=-1)
            values = torch.cat([values, pv], dim=2)
            allow_l = (mass > 0).reshape(bsz, n_head, 1, n_land * k_land)
            allow = torch.cat(
                [allow, allow_l.expand(bsz, n_head, 1, n_land * k_land)],
                dim=-1,
            )
        return self._attend(scores, allow, values, q.dtype)

    def _bidirectional_prefill(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        C = self.chunk_size
        n_full = (q.shape[2] // C) * C
        out_main = self._bidirectional_aligned(q[:, :, :n_full], k[:, :, :n_full], v[:, :, :n_full])
        if n_full == q.shape[2]:
            return out_main
        out_rem = self._remainder(q[:, :, n_full:], k, v, n_full, causal=False)
        return torch.cat([out_main, out_rem], dim=2)

    def _bidirectional_aligned(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Centered exact window + landmarks of every chunk outside that window.

        A query in chunk ``c`` sees every token of chunks
        ``[c - L//2, c - L//2 + L)`` and one landmark set for each other chunk.
        No causal mask: the encoder is bidirectional. The two sets are disjoint.
        """
        bsz, n_head, seq_len, head_dim = q.shape
        C = self.chunk_size
        n_chunks = seq_len // C
        L = self._window_L(n_chunks)
        left = L // 2
        right = L - 1 - left

        qc = q.reshape(bsz, n_head, n_chunks, C, head_dim)
        kc = k.reshape(bsz, n_head, n_chunks, C, head_dim)
        vc = v.reshape(bsz, n_head, n_chunks, C, head_dim)

        kpad = F.pad(kc, (0, 0, 0, 0, left, right))
        vpad = F.pad(vc, (0, 0, 0, 0, left, right))
        k_win = kpad.unfold(2, L, 1)[:, :, :n_chunks]
        v_win = vpad.unfold(2, L, 1)[:, :, :n_chunks]
        k_win = k_win.permute(0, 1, 2, 5, 3, 4).reshape(bsz, n_head, n_chunks, L * C, head_dim)
        v_win = v_win.permute(0, 1, 2, 5, 3, 4).reshape(bsz, n_head, n_chunks, L * C, head_dim)

        scores_exact = torch.matmul(qc.float(), k_win.float().transpose(-1, -2)) * self.scaling

        c = torch.arange(n_chunks, device=q.device).view(n_chunks, 1, 1, 1)
        s = torch.arange(L, device=q.device).view(1, 1, L, 1)
        src = c - left + s
        in_range = (src >= 0) & (src < n_chunks)
        mask_exact = in_range.expand(n_chunks, C, L, C).reshape(n_chunks, C, L * C)

        pk, pv, mass = self._pool(k, v, seq_len)
        k_land = pk.shape[3]
        pk_f = pk.reshape(bsz, n_head, 1, n_chunks * k_land, head_dim)
        pv_f = pv.reshape(bsz, n_head, 1, n_chunks * k_land, head_dim)
        scores_land = torch.matmul(qc.float(), pk_f.float().transpose(-1, -2)) * self.scaling
        log_mass = torch.log(mass.clamp(min=1).float()).reshape(bsz, n_head, 1, 1, -1)
        scores_land = scores_land + log_mass

        m = torch.arange(n_chunks, device=q.device).view(1, n_chunks)
        c_idx = torch.arange(n_chunks, device=q.device).view(n_chunks, 1)
        in_window = (m >= (c_idx - left)) & (m < (c_idx - left + L))
        mask_land = (~in_window).view(n_chunks, 1, n_chunks, 1).expand(n_chunks, C, n_chunks, k_land)
        mask_land = mask_land.reshape(n_chunks, C, n_chunks * k_land)

        scores = torch.cat([scores_exact, scores_land], dim=-1)
        allow = torch.cat(
            [
                mask_exact.view(1, 1, n_chunks, C, L * C),
                mask_land.view(1, 1, n_chunks, C, n_chunks * k_land),
            ],
            dim=-1,
        ).expand(bsz, n_head, n_chunks, C, -1).clone()
        mass_ok = (mass > 0).reshape(bsz, n_head, 1, 1, n_chunks * k_land)
        allow[..., L * C:] = allow[..., L * C:] & mass_ok
        values = torch.cat(
            [v_win, pv_f.expand(bsz, n_head, n_chunks, -1, head_dim)],
            dim=3,
        )
        out = self._attend(scores, allow, values, q.dtype)
        return out.reshape(bsz, n_head, seq_len, head_dim)

    def _cross_cepl(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Compress audio K/V into landmarks. Every text query sees every landmark.

        Text positions are not audio positions, so there is no exact local window
        to align. The tail that does not fill a chunk is attended exactly.
        """
        bq, n_head, _, head_dim = q.shape
        bk = k.shape[0]
        kv_len = k.shape[2]
        C = self.chunk_size
        n_full = (kv_len // C) * C

        pk, pv, mass = self._pool(k, v, n_full)
        k_land = pk.shape[3]
        n_chunks = pk.shape[2]
        pk = pk.reshape(bk, n_head, n_chunks * k_land, head_dim)
        pv = pv.reshape(bk, n_head, n_chunks * k_land, head_dim)
        mass = mass.reshape(bk, n_head, 1, n_chunks * k_land)
        if bk == 1 and bq != 1:
            pk = pk.expand(bq, -1, -1, -1)
            pv = pv.expand(bq, -1, -1, -1)
            mass = mass.expand(bq, -1, -1, -1)

        scores = torch.matmul(q.float(), pk.float().transpose(-1, -2)) * self.scaling
        scores = scores + torch.log(mass.clamp(min=1).float())
        allow = mass > 0
        values = pv

        if n_full < kv_len:
            k_tail = k[:, :, n_full:]
            v_tail = v[:, :, n_full:]
            if k_tail.shape[0] == 1 and bq != 1:
                k_tail = k_tail.expand(bq, -1, -1, -1)
                v_tail = v_tail.expand(bq, -1, -1, -1)
            scores_tail = torch.matmul(q.float(), k_tail.float().transpose(-1, -2)) * self.scaling
            scores = torch.cat([scores, scores_tail], dim=-1)
            values = torch.cat([values, v_tail], dim=2)
            allow = torch.cat(
                [
                    allow,
                    torch.ones(bq, n_head, 1, k_tail.shape[2], dtype=torch.bool, device=q.device),
                ],
                dim=-1,
            )
        return self._attend(scores, allow, values, q.dtype)


def _attention_cls():
    """Subclass MultiHeadAttention once, so KV-cache hooks still recognise it."""
    global _CEPL_CLS
    if _CEPL_CLS is not None:
        return _CEPL_CLS
    from whisper.model import MultiHeadAttention

    class WhisperCEPLAttention(_CEPLImpl, MultiHeadAttention):
        pass

    _CEPL_CLS = WhisperCEPLAttention
    return _CEPL_CLS


class WhisperVCPAInjector:
    """Reversible VCPA injection for an OpenAI ``whisper.model.Whisper`` instance."""

    def __init__(
        self,
        model: nn.Module,
        *,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        local_window: int = DEFAULT_LOCAL_WINDOW,
        threshold: int = DEFAULT_THRESHOLD,
        use_landmarks: bool = DEFAULT_USE_LANDMARKS,
        use_local: bool = DEFAULT_USE_LOCAL,
        pool_type: str = POOL_MULTIVEC,
        num_landmarks: int = DEFAULT_NUM_LANDMARKS,
        blend_mode: str = DEFAULT_BLEND_MODE,
        approximate_encoder: bool = True,
        approximate_decoder: bool = True,
        approximate_cross: bool = True,
    ):
        if not isinstance(model, nn.Module):
            raise TypeError("model must be a whisper.model.Whisper instance.")
        if not hasattr(model, "encoder") or not hasattr(model, "decoder"):
            raise ValueError(
                "model must be an OpenAI Whisper module with .encoder and .decoder. "
                f"Got {type(model).__name__}. Hugging Face causal LMs use VCPAInjector, not this class."
            )
        if int(chunk_size) <= 0 or int(local_window) <= 0 or int(threshold) <= 0:
            raise ValueError("chunk_size, local_window and threshold must be positive.")
        if blend_mode != "unified":
            raise ValueError("Whisper injection only supports blend_mode='unified'.")
        if blend_mode == "unified" and not (use_landmarks and use_local):
            raise ValueError(
                "blend_mode='unified' requires use_local=True and use_landmarks=True."
            )

        self.model = model
        self.chunk_size = int(chunk_size)
        self.local_window = int(local_window)
        self.threshold = int(threshold)
        self.use_landmarks = bool(use_landmarks)
        self.use_local = bool(use_local)
        self.pool_type = pool_type
        self.num_landmarks = max(1, int(num_landmarks))
        self.blend_mode = blend_mode
        self.approximate_encoder = bool(approximate_encoder)
        self.approximate_decoder = bool(approximate_decoder)
        self.approximate_cross = bool(approximate_cross)
        self._wrapped: List[Tuple[nn.Module, str, nn.Module]] = []
        self._failures: List[str] = []

    def inject(self) -> "WhisperVCPAInjector":
        """Wrap every Whisper attention module. Parameters are not copied or written."""
        if self._wrapped:
            self.remove()
        cls = _attention_cls()
        for parent, attr, kind in self._iter_sites():
            attn = getattr(parent, attr)
            if attn is None:
                continue
            if isinstance(attn, cls):
                continue
            try:
                wrapped = cls(
                    attn,
                    kind=kind,
                    chunk_size=self.chunk_size,
                    local_window=self.local_window,
                    threshold=self.threshold,
                    use_landmarks=self.use_landmarks,
                    use_local=self.use_local,
                    pool_type=self.pool_type,
                    num_landmarks=self.num_landmarks,
                    blend_mode=self.blend_mode,
                    approximate_cross=self.approximate_cross,
                    approximate_encoder=self.approximate_encoder,
                    approximate_decoder=self.approximate_decoder,
                )
                setattr(parent, attr, wrapped)
                self._wrapped.append((parent, attr, attn))
            except Exception as exc:  # noqa: BLE001
                self._failures.append(f"{kind} {attr}: {exc}")
        return self

    def remove(self) -> "WhisperVCPAInjector":
        """Put the original attention modules back. Same objects, same weights."""
        for parent, attr, original in self._wrapped:
            current = getattr(parent, attr)
            if getattr(current, "is_vcpa", False):
                setattr(parent, attr, current.remove())
            else:
                setattr(parent, attr, original)
        self._wrapped.clear()
        return self

    @property
    def is_injected(self) -> bool:
        return bool(self._wrapped)

    def _iter_sites(self):
        encoder = getattr(self.model, "encoder", None)
        decoder = getattr(self.model, "decoder", None)
        if encoder is not None:
            for block in encoder.blocks:
                yield block, "attn", "encoder"
        if decoder is not None:
            for block in decoder.blocks:
                yield block, "attn", "decoder"
                if getattr(block, "cross_attn", None) is not None:
                    yield block, "cross_attn", "cross"

    def verify_parameter_integrity(self) -> Tuple[bool, Dict[str, Any]]:
        """Structural check: wrapped projections are the original module objects."""
        cls = _attention_cls()
        equal = 0
        unequal = 0
        for parent, attr, _original in self._wrapped:
            attn = getattr(parent, attr)
            if not isinstance(attn, cls):
                unequal += 1
                continue
            for name in ("query", "key", "value", "out"):
                equal += 1
                if getattr(attn, name) is not getattr(attn.original_attn, name):
                    unequal += 1
                    equal -= 1
        keys_now = set(self.model.state_dict().keys())
        return unequal == 0, {
            "projection_refs_equal": equal,
            "projection_refs_unequal": unequal,
            "state_dict_keys": len(keys_now),
            "failures": list(self._failures),
        }

    def report(self) -> str:
        n = len(self._wrapped)
        calls = 0
        by_kind = {"encoder": 0, "decoder": 0, "cross": 0}
        for parent, attr, _original in self._wrapped:
            attn = getattr(parent, attr)
            by_kind[getattr(attn, "kind", "?")] = by_kind.get(getattr(attn, "kind", "?"), 0) + 1
            calls += int(getattr(attn, "cepl_calls", 0))
        lines = [
            f"WhisperVCPAInjector: {n} attention modules injected",
            "  encoder self-attention: bidirectional CEPL "
            f"({'on' if self.approximate_encoder else 'exact fallback'}) x{by_kind.get('encoder', 0)}",
            "  decoder self-attention: causal CEPL "
            f"({'on' if self.approximate_decoder else 'exact fallback'}) x{by_kind.get('decoder', 0)}",
            "  decoder cross-attention: audio-landmark CEPL "
            f"({'on' if self.approximate_cross else 'exact fallback'}) x{by_kind.get('cross', 0)}",
            "  config: "
            f"chunk_size={self.chunk_size}, local_window={self.local_window}, "
            f"threshold={self.threshold}, pool={self.pool_type}, "
            f"num_landmarks={self.num_landmarks}, blend_mode={self.blend_mode}",
            f"  CEPL forwards so far: {calls}",
            "  parameters: not copied, not modified. remove() restores the original modules.",
            "  note: CEPL is an approximation. Sequences at or below threshold, and any "
            "forward while SDPA is disabled (word timestamps), use exact attention.",
        ]
        if self._failures:
            lines.append(f"  failures: {len(self._failures)}")
            lines.extend(f"    {item}" for item in self._failures)
        lines.append("  status: INJECTED" if self.is_injected else "  status: REMOVED")
        return "\n".join(lines)
