"""Chunk pooling strategies for VCPA-Inject.

Each class implements:
    pool(k, v, chunk_size) -> (pk, pv):  B, H, N -> B, H, N//C
    gather(pk, pv, idx) -> (gk, gv):     used for remainder / special tokens

All operations are pure vectorized PyTorch, device-agnostic, and dtype-preserving.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .config import (
    DEFAULT_NUM_LANDMARKS, POOL_MEAN, POOL_MAX, POOL_MULTIVEC, POOL_SOFT, POOL_TOPK, EPS,
)


class Pooler:
    """Abstract base: implements a pooling strategy over chunks of K/V.

    ``pool`` keeps the historical contract: one vector per chunk, ``(B, H, N//C, D)``.
    ``pool_landmarks`` is what attention actually consumes. It returns ``K`` vectors per
    chunk plus the number of original tokens each vector stands for, so the attention
    bias is ``log(mass)`` rather than a flat ``log(C)`` pasted onto every landmark.
    """

    landmarks_per_chunk = 1

    def __init__(self, pool_type: str = POOL_MEAN):
        self.pool_type = pool_type

    def pool(self, k: torch.Tensor, v: torch.Tensor, chunk_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Pool key/value over chunks of size ``chunk_size``.

        Args:
            k, v: (B, H, N, D). N must be divisible by ``chunk_size``.

        Returns:
            pk, pv: (B, H, N//C, D).
        """
        raise NotImplementedError

    def pool_landmarks(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        chunk_size: int,
        valid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``K`` landmarks per chunk and the token mass each one represents.

        Args:
            k, v: (B, H, N, D), N divisible by ``chunk_size``.
            valid: optional (B, N) boolean, True where the token is real (not padding).

        Returns:
            pk, pv: (B, H, N//C, K, D)
            mass: (B, H, N//C, K) — original tokens summarised by that landmark.
                  A zero mass means the landmark is empty and must be masked out.
        """
        pk, pv = self.pool(k, v, chunk_size)
        b, h, n_chunks, _ = pk.shape
        mass = torch.full(
            (b, h, n_chunks, 1), float(chunk_size), device=pk.device, dtype=torch.float32,
        )
        if valid is not None:
            counts = valid.to(torch.float32).view(b, 1, n_chunks, chunk_size).sum(dim=-1)
            mass = counts.expand(b, h, n_chunks).unsqueeze(-1)
        return pk.unsqueeze(3), pv.unsqueeze(3), mass


def build_pooler(pool_type: str, **kw) -> Pooler:
    """Factory: create a pooler from a string name."""
    if pool_type == POOL_MEAN:
        return MeanPooler()
    if pool_type == POOL_MAX:
        return MaxPooler()
    if pool_type == POOL_SOFT:
        return SoftmaxWeightedPooler()
    if pool_type == POOL_TOPK:
        return TopKPooler(**kw)
    if pool_type == POOL_MULTIVEC:
        return QueryAwareMultiVectorPooler(**kw)
    raise ValueError(
        f"Unknown pooling strategy: {pool_type!r}. "
        f"Valid options: {POOL_MEAN}, {POOL_MAX}, {POOL_SOFT}, {POOL_TOPK}, {POOL_MULTIVEC}."
    )


class MeanPooler(Pooler):
    """Simple mean pooling: each chunk collapses to its arithmetic mean."""

    def pool(self, k: torch.Tensor, v: torch.Tensor, chunk_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        b, h, n, d = k.shape
        n_chunks = n // chunk_size
        k_ = k.view(b, h, n_chunks, chunk_size, d).mean(dim=3)
        v_ = v.view(b, h, n_chunks, chunk_size, d).mean(dim=3)
        return k_, v_


class MaxPooler(Pooler):
    """Maximum pooling: each chunk collapses to the elementwise maximum."""

    def pool(self, k: torch.Tensor, v: torch.Tensor, chunk_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        b, h, n, d = k.shape
        n_chunks = n // chunk_size
        k_, _ = k.view(b, h, n_chunks, chunk_size, d).max(dim=3)
        v_, _ = v.view(b, h, n_chunks, chunk_size, d).max(dim=3)
        return k_, v_


class SoftmaxWeightedPooler(Pooler):
    """Token-wise softmax-weighted pooling inside each chunk.

    For each chunk c and head h:
        w_{c,i} = softmax(k_{c,i} . k_c_mean / sqrt(D)) over i in chunk
        p_k_c = sum_i w_{c,i} k_{c,i}   (similarly for v)
    This keeps the most "representative" tokens and is more informative than plain mean.
    """

    def pool(self, k: torch.Tensor, v: torch.Tensor, chunk_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        b, h, n, d = k.shape
        n_chunks = n // chunk_size
        k_chunks = k.view(b, h, n_chunks, chunk_size, d)
        v_chunks = v.view(b, h, n_chunks, chunk_size, d)
        centroid = k_chunks.mean(dim=3, keepdim=True)                        # (B,H,C,1,D)
        # similarity of each token to its own chunk centroid
        sim = (k_chunks * centroid).sum(dim=-1, keepdim=True) / (d ** 0.5)   # (B,H,C,S,1)
        w = F.softmax(sim, dim=3)
        return (w * k_chunks).sum(dim=3), (w * v_chunks).sum(dim=3)          # (B,H,C,D)


class TopKPooler(Pooler):
    """Top-k per chunk: keep only the k largest-magnitude token vectors as landmarks.

    The landmark is the highest-magnitude token in the chunk; the remaining pool is
    the mean of the rest. ``with_flags`` returns a retain_mask so callers can attend
    only to the selected token when they need to.
    """

    def __init__(self, pool_type: str = POOL_TOPK, k: int | None = None):
        super().__init__(pool_type)
        self.k = k

    def pool(self, k: torch.Tensor, v: torch.Tensor, chunk_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return mean-pooled landmarks (same shape/semantics as mean pooling).

        For the strict top-k behavior call ``with_flags`` and attend only to retained
        landmarks.
        """
        return MeanPooler(POOL_MEAN).pool(k, v, chunk_size)

    def with_flags(self, k: torch.Tensor, v: torch.Tensor, chunk_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (pooled_k, pooled_v, retain_mask) with top-k landmark selection.

        For each chunk the retained landmark is the token with the largest squared
        norm (the strongest "landmark"). ``retain_mask`` marks which chunks produced a
        retained landmark; since we always retain one token per chunk it is all True
        and mainly exists so callers can mask out empty chunks if they wish.

        Args:
            k, v: (B, H, N, D), N divisible by chunk_size.
            chunk_size: size of each chunk.

        Returns:
            pooled_k, pooled_v: (B, H, N//C, D)
            retain_mask: (B, H, N//C) boolean
        """
        b, h, n, d = k.shape
        n_chunks = n // chunk_size
        k_view = k.view(b, h, n_chunks, chunk_size, d)
        v_view = v.view(b, h, n_chunks, chunk_size, d)

        magn = k_view.pow(2).sum(dim=-1)                  # (B,H,C,S)
        best = magn.argmax(dim=3, keepdim=True)           # (B,H,C,1)
        idx = best.unsqueeze(-1).expand(-1, -1, -1, -1, d)

        k_land = k_view.gather(3, idx).squeeze(3)         # (B,H,C,D)
        v_land = v_view.gather(3, idx).squeeze(3)
        retain_mask = torch.ones((b, h, n_chunks), dtype=torch.bool, device=k.device)
        return k_land, v_land, retain_mask

    def gather(self, pk: torch.Tensor, pv: torch.Tensor, idx: torch.Tensor, retain_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather landmark values at ``idx`` where the landmark was retained."""
        rm = retain_mask[:, :, idx].unsqueeze(-1)
        return (
            torch.where(rm, pk, torch.zeros_like(pk)),
            torch.where(rm, pv, torch.zeros_like(pv)),
        )


class QueryAwareMultiVectorPooler(Pooler):
    """K representative landmarks per chunk, instead of one static mean.

    A single mean throws away the within-chunk variance that caused the large
    perplexity jumps. This pooler splits every chunk into ``K`` nearly equal
    segments and, inside each segment, builds one landmark by a local
    magnitude softmax:

        w_i = softmax_i(||k_i||^2)   over the valid tokens of the segment
        landmark = sum_i w_i * (k_i, v_i)

    That is a tiny attention over the segment, not over the whole sequence, so
    it stays O(N) and does not depend on the downstream query. Query guidance
    happens where it is cheap and cacheable: the main softmax, which weights
    these K landmarks differently for every query.

    Why not pool with the query itself? A landmark that depends on the current
    query cannot be reused by the next query or by the decode cache, and
    computing it would read every old token again — the cost CEPL exists to
    avoid. Magnitude selects *which* directions a chunk contains; the query
    selects *which of those directions it needs*.

    Each landmark's ``mass`` is the number of valid tokens in its segment, so
    the attention bias is ``log(mass)`` per landmark. Using ``log(C)`` on all K
    of them would overweight old context by a factor of K.
    """

    def __init__(self, num_landmarks: int = DEFAULT_NUM_LANDMARKS, **_ignored):
        super().__init__(POOL_MULTIVEC)
        if int(num_landmarks) < 1:
            raise ValueError("num_landmarks must be >= 1")
        self.num_landmarks = int(num_landmarks)
        self.landmarks_per_chunk = self.num_landmarks

    def pool(self, k: torch.Tensor, v: torch.Tensor, chunk_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        pk, pv, _mass = self.pool_landmarks(k, v, chunk_size)
        # Historical callers expect one vector. Average the K landmarks so the
        # old shape still means "a summary of the chunk", not a silent truncation.
        return pk.mean(dim=3), pv.mean(dim=3)

    def pool_landmarks(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        chunk_size: int,
        valid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, h, n, d = k.shape
        n_chunks = n // chunk_size
        k_num = min(self.num_landmarks, chunk_size)
        kc = k.view(b, h, n_chunks, chunk_size, d)
        vc = v.view(b, h, n_chunks, chunk_size, d)

        seg = (torch.arange(chunk_size, device=k.device) * k_num) // chunk_size  # (C,)
        seg_oh = F.one_hot(seg, k_num).to(torch.bool)                             # (C, K)

        if valid is None:
            token_ok = torch.ones(b, 1, n_chunks, chunk_size, dtype=torch.bool, device=k.device)
        else:
            token_ok = valid[:, :n].to(torch.bool).view(b, 1, n_chunks, chunk_size)

        # (B, H, nc, C, K): token belongs to this segment and is not padding.
        member = seg_oh.view(1, 1, 1, chunk_size, k_num) & token_ok.unsqueeze(-1)
        magnitude = kc.float().pow(2).sum(dim=-1)                                # (B, H, nc, C)
        scores = magnitude.unsqueeze(-1).expand(b, h, n_chunks, chunk_size, k_num)
        member_h = member.expand(b, h, n_chunks, chunk_size, k_num)
        scores = scores.masked_fill(~member_h, torch.finfo(torch.float32).min)
        empty = ~member.any(dim=3)                                               # (B, 1 or H, nc, K)
        # A segment with no valid token must not produce NaN in softmax.
        scores = scores.masked_fill(empty.unsqueeze(3), 0.0)
        weights = F.softmax(scores, dim=3)
        weights = weights.masked_fill(empty.unsqueeze(3), 0.0)

        pk = torch.einsum("bhnck,bhncd->bhnkd", weights, kc.float()).to(dtype=k.dtype)
        pv = torch.einsum("bhnck,bhncd->bhnkd", weights, vc.float()).to(dtype=v.dtype)
        mass = member.to(torch.float32).sum(dim=3).expand(b, h, n_chunks, k_num)
        return pk, pv, mass
