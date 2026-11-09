"""Real-checkpoint integrity check used by the Kaggle runner.

Loads the model twice worth of information? No: loads it once, fingerprints every
parameter, injects, fingerprints again, removes, fingerprints a third time.  The
fingerprint is a SHA-256 over each parameter's raw bytes, so it detects any change
in value, dtype, shape or ordering.

    python kaggle/check_integrity.py --model Qwen/Qwen2.5-1.5B --dtype float16
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from vcpa import VCPAInjector  # noqa: E402


def _tensor_bytes(t: torch.Tensor) -> bytes:
    """Exact raw bytes of a tensor, robust to bfloat16 and quantised dtypes."""
    t = t.detach()
    try:
        t = t.cpu().contiguous()
        # view(torch.uint8) is byte-exact for every standard dtype (incl. bfloat16).
        return t.view(torch.uint8).numpy().tobytes()
    except Exception:  # noqa: BLE001 - quantised / exotic parameter type
        try:
            return t.to(torch.float32).numpy().tobytes()
        except Exception:  # noqa: BLE001
            return repr((tuple(t.shape), str(t.dtype))).encode()


def hash_parameters(model) -> dict[str, str]:
    out = {}
    for name, p in model.named_parameters():
        h = hashlib.sha256()
        h.update(repr((tuple(p.shape), str(p.dtype))).encode())
        h.update(_tensor_bytes(p))
        out[name] = h.hexdigest()
    return out


def hash_state_dict(model) -> dict[str, str]:
    out = {}
    for name, t in model.state_dict().items():
        h = hashlib.sha256()
        h.update(repr((tuple(t.shape), str(t.dtype))).encode())
        h.update(_tensor_bytes(t))
        out[name] = h.hexdigest()
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--load-in-4bit", action="store_true")
    args = parser.parse_args()

    from benchmarks.harness import load_model, resolve_device, resolve_dtype

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    model = load_model(args.model, device, dtype, load_in_4bit=args.load_in_4bit)

    params_before = hash_parameters(model)
    sd_before = hash_state_dict(model)
    n_before = sum(p.numel() for p in model.parameters())

    injector = VCPAInjector(model)
    injector.inject()
    try:
        params_injected = hash_parameters(model)
        sd_injected = hash_state_dict(model)
        n_injected = sum(p.numel() for p in model.parameters())
    finally:
        injector.remove()

    params_after = hash_parameters(model)
    sd_after = hash_state_dict(model)
    n_after = sum(p.numel() for p in model.parameters())

    injected_layers = sum(1 for layer in model.model.layers
                          if type(layer.self_attn).__name__ == "CEPLAttention")
    checks = {
        "model": args.model,
        "num_parameters_before": n_before,
        "num_parameters_injected": n_injected,
        "num_parameters_after": n_after,
        "parameter_count_unchanged": n_before == n_injected == n_after,
        "parameter_hashes_unchanged_while_injected": params_before == params_injected,
        "parameter_hashes_unchanged_after_remove": params_before == params_after,
        "state_dict_keys_unchanged_while_injected": list(sd_before) == list(sd_injected),
        "state_dict_keys_unchanged_after_remove": list(sd_before) == list(sd_after),
        "state_dict_hashes_unchanged_while_injected": sd_before == sd_injected,
        "state_dict_hashes_unchanged_after_remove": sd_before == sd_after,
        "layers": len(model.model.layers),
        "layers_still_injected_after_remove": injected_layers,
    }
    checks["ok"] = all(v for k, v in checks.items() if isinstance(v, bool))

    print("\nIntegrity check")
    for k, v in checks.items():
        print(f"  {k}: {v}")

    out_dir = "/kaggle/working/vcpa_results" if os.path.isdir("/kaggle/working") else os.path.join(ROOT, "results")
    os.makedirs(out_dir, exist_ok=True)
    tag = args.model.replace("/", "__").replace(":", "_")
    path = os.path.join(out_dir, f"integrity_{tag}.json")
    import json
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(checks, fh, indent=2)
    print(f"written: {path}")

    return 0 if checks["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
