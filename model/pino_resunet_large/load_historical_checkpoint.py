#!/usr/bin/env python3
"""Load the retained PINO-ResU-Net-large model from its source snapshot.

The completed result is bound to the exact trainer kept under ``provenance/``.
That compact snapshot intentionally does not duplicate the vendored network
directory.  This adapter imports the byte-exact trainer, then points its model
lookup at the repository's hash-bound ``src/vendor`` implementation.  All
scientific functions continue to come from the retained trainer.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


BENCHMARK_ROOT = Path(__file__).resolve().parents[2]
TRAINER_SNAPSHOT = (
    BENCHMARK_ROOT
    / "provenance"
    / "training_sources"
    / "pino_depth4_plateau_v2_original"
    / "train_pino_depth4_plateau_v2.py"
)
VENDOR_ROOT = (BENCHMARK_ROOT / "src" / "vendor").resolve()


def _load_retained_trainer():
    specification = importlib.util.spec_from_file_location(
        "retained_pino_depth4_plateau_v2_trainer", TRAINER_SNAPSHOT
    )
    if specification is None or specification.loader is None:
        raise ImportError(TRAINER_SNAPSHOT)
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    module.PINO_ROOT = VENDOR_ROOT
    return module


_TRAINER = _load_retained_trainer()


def __getattr__(name: str):
    """Delegate the evaluator's required interface to the exact trainer."""

    return getattr(_TRAINER, name)


def source_identity() -> dict[str, str]:
    """Return readable paths for evaluation provenance and diagnostics."""

    return {
        "trainer_snapshot": str(TRAINER_SNAPSHOT),
        "vendor_root": str(VENDOR_ROOT),
    }
