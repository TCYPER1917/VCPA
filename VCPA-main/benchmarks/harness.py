"""Shared benchmark utilities for VCPA-Inject.

Design goals:
  * honest measurements (warm-up, repeats, CUDA synchronisation, peak-memory resets)
  * identical code path on CPU and GPU
  * machine-readable JSON output
  * never mix baseline and injected timings on the same process state

Run on CPU for a smoke check, or on GPU (recommended) for real numbers:

    python benchmarks/benchmark_prefill.py --model Qwen/Qwen2.5-1.5B --device cuda
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# ----------------------------------------------------------------------
# Environment / device
# ----------------------------------------------------------------------

def resolve_device(name: str = "auto") -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(name)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return dev


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if name == "auto":
        return torch.float16 if device.type == "cuda" else torch.float32
    if name not in mapping:
        raise ValueError(f"Unknown dtype {name!r}; choose from {list(mapping)}")
    return mapping[name]


def environment_info(device: torch.device) -> Dict[str, Any]:
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": _transformers_version(),
        "device": str(device),
        "cpu_count": os.cpu_count(),
    }
    if device.type == "cuda":
        info.update({
            "gpu": torch.cuda.get_device_name(device),
            "gpu_total_memory_gb": round(
                torch.cuda.get_device_properties(device).total_memory / 1e9, 2),
            "cuda": torch.version.cuda,
        })
    return info


def _transformers_version() -> str:
    try:
        import transformers
        return transformers.__version__
    except Exception:  # noqa: BLE001
        return "unknown"


# ----------------------------------------------------------------------
# Model loading
# ----------------------------------------------------------------------

def load_model(
    model_id: str,
    device: torch.device,
    dtype: torch.dtype,
    *,
    attn_implementation: Optional[str] = None,
    trust_remote_code: bool = False,
    load_in_4bit: bool = False,
):
    """Load a causal LM for inference.

    ``model_id="tiny:qwen"`` builds a small random Qwen2 model instead of downloading
    a checkpoint.  This is used to validate the benchmark scripts on CPU before
    running them on a GPU box.

    ``load_in_4bit`` is used on Kaggle for models that do not fit in 16-bit.
    Note: 4-bit quantisation changes the baseline too; results remain comparable
    (baseline vs injected) because both sides run on the same quantised model.
    """
    from transformers import AutoModelForCausalLM

    if model_id.startswith("tiny:"):
        from transformers import Qwen2Config, Qwen2ForCausalLM
        cfg = Qwen2Config(
            vocab_size=256, hidden_size=64, intermediate_size=128,
            num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=8192, tie_word_embeddings=False,
        )
        return (Qwen2ForCausalLM._from_config(cfg, attn_implementation="sdpa")
                .to(device).to(dtype).eval())

    kwargs: Dict[str, Any] = {"torch_dtype": dtype, "trust_remote_code": trust_remote_code}
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = {"": device.index or 0}
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        model = model.to(device)
    return model.eval()


def param_fingerprint(model) -> Dict[str, Any]:
    """Cheap structural fingerprint used to prove weights were not touched."""
    n_params = 0
    checksum = 0.0
    for p in model.parameters():
        n_params += p.numel()
        checksum += float(p.detach().float().sum().item())
    return {
        "num_parameters": n_params,
        "sum_of_parameters": checksum,
        "num_tensors": sum(1 for _ in model.parameters()),
        "state_dict_keys": len(model.state_dict()),
    }


# ----------------------------------------------------------------------
# Timing
# ----------------------------------------------------------------------

@dataclass
class Measurement:
    name: str
    seconds: List[float] = field(default_factory=list)
    peak_memory_bytes: Optional[int] = None

    @property
    def mean(self) -> float:
        return statistics.fmean(self.seconds) if self.seconds else float("nan")

    @property
    def median(self) -> float:
        return statistics.median(self.seconds) if self.seconds else float("nan")

    @property
    def stdev(self) -> float:
        return statistics.stdev(self.seconds) if len(self.seconds) > 1 else 0.0

    @property
    def tokens_per_second(self) -> float:
        return float("nan")


class Timer:
    """Wall-clock timer that synchronises CUDA when needed."""

    def __init__(self, device: torch.device):
        self.device = device
        self._t0 = 0.0

    def __enter__(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.elapsed = time.perf_counter() - self._t0
        return False


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory(device: torch.device) -> Optional[int]:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        return int(torch.cuda.max_memory_allocated(device))
    return None


def measure(
    fn: Callable[[], Any],
    device: torch.device,
    *,
    warmup: int = 3,
    repeats: int = 10,
    name: str = "run",
) -> Measurement:
    """Run ``fn`` with warm-up and return timing statistics."""
    for _ in range(warmup):
        fn()
    reset_peak_memory(device)
    m = Measurement(name=name)
    for _ in range(repeats):
        with Timer(device) as t:
            fn()
        m.seconds.append(t.elapsed)
    m.peak_memory_bytes = peak_memory(device)
    return m


# ----------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------

def save_json(payload: Dict[str, Any], path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    return path


def print_table(rows: List[Dict[str, Any]], columns: List[str]) -> None:
    if not rows:
        return
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in columns}
    line = "  ".join(c.ljust(widths[c]) for c in columns)
    print(line)
    print("  ".join("-" * widths[c] for c in columns))
    for r in rows:
        print("  ".join(str(r.get(c, "")).ljust(widths[c]) for c in columns))


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B",
                        help="Hugging Face model id or local path")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--attn-implementation", default=None,
                        help="e.g. sdpa / eager / flash_attention_2")
    parser.add_argument("--load-in-4bit", action="store_true",
                        help="quantise the model (useful for large models on Kaggle)")
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--local-window", type=int, default=256)
    parser.add_argument("--threshold", type=int, default=256)
    parser.add_argument("--pool", default="multivec",
                        choices=["multivec", "mean", "max", "softmax", "topk"])
    parser.add_argument("--num-landmarks", type=int, default=4,
                        help="landmarks kept per chunk when --pool multivec")
    parser.add_argument("--blend-mode", default="unified", choices=["unified", "blend"])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--json", default=None, help="write results to this JSON file")
    return parser


def make_injector(model, args) -> Any:
    from vcpa import VCPAInjector
    return VCPAInjector(
        model,
        chunk_size=args.chunk_size,
        local_window=args.local_window,
        threshold=args.threshold,
        pool_type=args.pool,
        blend_mode=args.blend_mode,
        num_landmarks=args.num_landmarks,
    )
