"""Inject VCPA into an OpenAI Whisper checkpoint, without touching its weights.

    python examples/inject_whisper.py --model tiny
    python examples/inject_whisper.py --model C:\\path\\to\\tiny.pt --device cpu

The official Whisper tree is expected at ../whisper-main or WHISPER_ROOT.
Injection is reversible: parameters are the same objects before, during, and after.
"""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WHISPER_ROOT = os.environ.get("WHISPER_ROOT", r"C:\Users\K5\Documents\whisper-main")
for path in (ROOT, WHISPER_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

import torch  # noqa: E402
import whisper  # noqa: E402


def fingerprint(model) -> dict:
    return {name: param.data_ptr() for name, param in model.named_parameters()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="tiny", help="whisper model name or checkpoint path")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--local-window", type=int, default=256)
    parser.add_argument("--threshold", type=int, default=256)
    args = parser.parse_args()

    # load_model injects by default; load plain weights first so the fingerprint
    # is taken on the original modules, then inject explicitly.
    model = whisper.load_model(args.model, device=args.device, vcpa=False)
    before = fingerprint(model)
    model = whisper.inject_vcpa(
        model,
        chunk_size=args.chunk_size,
        local_window=args.local_window,
        threshold=args.threshold,
    )
    during = fingerprint(model)
    print(model.vcpa.report())
    if before != during:
        print("FAIL: parameter identity changed during injection")
        return 1

    mel = torch.randn(1, model.dims.n_mels, model.dims.n_audio_ctx * 2, device=model.device)
    if args.device == "cpu":
        mel = mel.float()
    with torch.no_grad():
        features = model.encoder(mel)
    print(f"encoder output: {tuple(features.shape)} finite={torch.isfinite(features).all().item()}")
    print(f"encoder CEPL calls: {sum(block.attn.cepl_calls for block in model.encoder.blocks)}")

    whisper.remove_vcpa(model)
    after = fingerprint(model)
    if before != after:
        print("FAIL: parameter identity changed after remove()")
        return 1
    print("parameters unchanged: True")
    print(model.vcpa.report())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
