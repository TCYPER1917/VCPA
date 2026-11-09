"""Shared test helpers."""

from __future__ import annotations

import os
import sys

# Make the repository root importable when tests are run from anywhere.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Real checkpoints are only downloaded when explicitly requested (e.g. on Kaggle).
REAL_MODEL_ID = os.environ.get("VCPA_REAL_MODEL", "Qwen/Qwen2.5-0.5B")
RUN_REAL_MODEL_TESTS = os.environ.get("VCPA_TEST_REAL_MODEL", "0") == "1"
