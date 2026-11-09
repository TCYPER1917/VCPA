"""Cache-aware mode control for VCPA-Inject.

During autoregressive decode the history grows one token at a time. Recomputing
chunk-pooled landmarks over the entire history each decode step would cost O(seq_len)
per token and defeat the purpose of caching.

VCPACacheMode is a tiny context manager / state object that decides, for each forward,
whether the wrapped attention should run in "cepl" mode (prefill, long context, no cache)
or fall back to exact attention (decode, short context, or when a cache is present).

This design explicitly supports the requirement:
    "If VCPA cannot safely or efficiently support incremental decoding yet, use the
     original attention implementation as a fallback during decode."

DECODE FALLBACK: when ``past_key_values`` is present and the current step is a single
token, the attention runs through exact SDPA using the cached KV. No claim is made that
chunk-pooled attention equals exact attention.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, Optional


class VCPACacheMode:
    """Stateful object that decides CEPL vs exact-attention per forward call."""

    def __init__(
        self,
        use_cepl: bool = True,
        min_context_for_cepl: int = 256,
    ):
        self.use_cepl = bool(use_cepl)
        self.min_context_for_cepl = int(min_context_for_cepl)
        self._step = 0

    @property
    def is_decode_step(self) -> bool:
        """True if this forward represents a single-token decode step."""
        return self._step > 0

    def step(self) -> "VCPACacheMode":
        """Advance the generation step counter."""
        self._step += 1
        return self

    @contextmanager
    def prefill(self) -> Iterator[None]:
        """Context that forces CEPL mode for the duration of a prefill pass."""
        self._step = 0
        old = self.use_cepl
        self.use_cepl = True
        try:
            yield
        finally:
            self.use_cepl = old

    @contextmanager
    def decode(self) -> Iterator[None]:
        """Context that forces exact-attention fallback for decode steps."""
        self._step += 1
        old = self.use_cepl
        self.use_cepl = False
        try:
            yield
        finally:
            self.use_cepl = old
            self._step -= 1

    def decide(self, seq_len: int, past_key_values: Any | None = None) -> str:
        """Decide the mode for this forward call.

        Returns one of: "exact", "cepl", "cepl_short".
        """
        if not self.use_cepl:
            return "exact"
        if past_key_values is not None and seq_len == 1:
            # Short or unreadable caches stay exact. A long cache uses the landmark
            # path inside CEPLAttention; this label is informational.
            length = None
            try:
                length = int(past_key_values.get_seq_length())
            except Exception:  # noqa: BLE001
                length = None
            if length is not None and length > self.min_context_for_cepl:
                return "landmark_decode"
            return "exact"
        if seq_len <= self.min_context_for_cepl:
            return "exact"
        return "cepl"


def choose_mode(
    attn,
    seq_len: int,
    past_key_values: Any | None = None,
    cache_mode: Optional[VCPACacheMode] = None,
) -> str:
    """Convenience: delegate mode decision to the cache mode object, or a default."""
    if cache_mode is not None:
        return cache_mode.decide(seq_len, past_key_values)
    # default heuristic (mirrors CEPLAttention.forward: <= threshold is exact)
    if past_key_values is not None and seq_len == 1:
        length = None
        try:
            length = int(past_key_values.get_seq_length(getattr(attn, "layer_idx", None)))
        except TypeError:
            try:
                length = int(past_key_values.get_seq_length())
            except Exception:  # noqa: BLE001
                length = None
        except Exception:  # noqa: BLE001
            length = None
        if length is not None and length > attn.threshold:
            return "landmark_decode"
        return "exact"
    if seq_len <= attn.threshold:
        return "exact"
    return "cepl"
