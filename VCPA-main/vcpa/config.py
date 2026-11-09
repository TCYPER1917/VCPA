"""VCPA-Inject: config and constants.

A "chunked-exact + pooled-landmarks" (CEPL) attention module that:
- injects into a pretrained Hugging Face causal language model at runtime
- preserves all original parameters unchanged
- remains reversible (inject / remove)
- runs exact attention when the context is small, and chunk-pooled landmarks when it is large
"""

from __future__ import annotations

__version__ = "12.0.0-alpha"
__package_name__ = "vcpa"

# Default pooling strategies, exposed as strings for config parsing.
POOL_MEAN = "mean"
POOL_MAX = "max"
POOL_SOFT = "softmax"  # token-wise softmax-weighted mean inside a chunk
POOL_TOPK = "topk"     # top-k per chunk by value magnitude (approximate landmark)
POOL_MULTIVEC = "multivec"  # K magnitude-weighted segment landmarks per chunk

DEFAULT_CHUNK_SIZE = 64        # size of each full chunk (landmarks)
DEFAULT_LOCAL_WINDOW = 256     # tokens of recent context kept at exact token-level attention
DEFAULT_BLEND_MODE = "unified" # "unified": single softmax over exact-window + landmarks
                               # "blend":   legacy convex blend of two separate softmaxes
DEFAULT_USE_LANDMARKS = True   # whether to include global chunk landmarks
DEFAULT_USE_LOCAL = True       # whether to include the exact local window
DEFAULT_THRESHOLD = 256        # seq_len below which exact SDPA is used instead of CEPL
DEFAULT_NUM_LANDMARKS = 4      # representative vectors kept per chunk (multivec)
DEFAULT_BLEND_LANDMARK_WEIGHT = 0.5  # weight for the landmark stream
DEFAULT_BLEND_LOCAL_WEIGHT = 0.5     # weight for the local stream (1 - weight_landmark)

# Minimum number of full chunks needed before chunk-pooled attention becomes worthwhile.
MIN_CHUNKS_FOR_POOLING = 4

# Numerical guard.
EPS = 1e-6
