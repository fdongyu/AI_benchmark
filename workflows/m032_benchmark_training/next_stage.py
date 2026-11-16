#!/usr/bin/env python3
"""Validate one m032 depth4/plateau lineage and print its next stage."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import torch


STAGES = (4, 8, 16, 32)
PROTOCOL_ID = os.environ.get(
    "AI_BENCHMARK_PROTOCOL_ID", "houston-m032-common13-depth4-plateau-v1"
)
REGISTRY = Path(__file__).resolve().parent / "model_registry.json"
MODEL_LIST_SUPPORT = Path(__file__).resolve().parents[2] / "src" / "model_list.py"


def load_model_list_support():
    spec = importlib.util.spec_from_file_location(
        "next_stage_model_list_support", MODEL_LIST_SUPPORT
    )
    if spec is None or spec.loader is None:
        raise ImportError(MODEL_LIST_SUPPORT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def model_keys() -> tuple[str, ...]:
    payload = json.loads(REGISTRY.read_text(encoding="utf-8"))
    entries = load_model_list_support().validate_model_list(payload)
    return tuple(str(entry["model"]) for entry in entries)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=model_keys(), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def next_stage(output_root: Path, model: str) -> str:
    if model not in model_keys():
        raise ValueError(f"unknown model: {model}")
    roots = [output_root / model / f"stage_ar{hours:02d}" for hours in STAGES]
    progress_names = (
        "stage_summary.json",
        "checkpoint_final.pt",
        "checkpoint_last.pt",
        "checkpoint_best_validation_supervised.pt",
        "training_config.json",
        "provenance.json",
        "history.csv",
        "slice_status.json",
    )
    for index, (hours, stage) in enumerate(zip(STAGES, roots)):
        summary_path = stage / "stage_summary.json"
        if not summary_path.is_file():
            if any(
                any((later / name).exists() for name in progress_names)
                for later in roots[index + 1 :]
            ):
                raise RuntimeError("a later stage cannot precede an incomplete stage")
            return str(hours)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        expected = {
            "protocol_id": PROTOCOL_ID,
            "subset": "m032",
            "model": model,
            "rollout_hours": hours,
            "complete": True,
            "stop_reason": "validation_plateau",
        }
        mismatches = {
            key: (summary.get(key), value)
            for key, value in expected.items()
            if summary.get(key) != value
        }
        epochs = int(summary.get("epochs_completed", -1))
        if not 12 <= epochs <= 240:
            mismatches["epochs_completed"] = (epochs, "12..240")
        if mismatches:
            raise RuntimeError(f"invalid {summary_path}: {mismatches}")
        paths = {
            "last": stage / "checkpoint_last.pt",
            "final": stage / "checkpoint_final.pt",
            "best": stage / "checkpoint_best_validation_supervised.pt",
        }
        expected_hashes = summary.get("checkpoint_sha256", {})
        if set(expected_hashes) != set(paths):
            raise RuntimeError(f"incomplete checkpoint hashes in {summary_path}")
        for role, checkpoint in paths.items():
            if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
                raise RuntimeError(f"missing or empty checkpoint: {checkpoint}")
            if sha256(checkpoint) != expected_hashes[role]:
                raise RuntimeError(f"{role} checkpoint hash mismatch: {checkpoint}")
        if expected_hashes["last"] != expected_hashes["final"]:
            raise RuntimeError(f"final is not the convergence-last checkpoint: {stage}")
        best = torch.load(paths["best"], map_location="cpu", weights_only=False)
        final = torch.load(paths["final"], map_location="cpu", weights_only=False)
        for role, payload in (("best", best), ("final", final)):
            config = payload.get("config", {})
            payload_expected = {
                "protocol_id": PROTOCOL_ID,
                "subset": "m032",
                    "model": model,
                "rollout_hours": hours,
            }
            payload_mismatches = {
                key: (config.get(key), value)
                for key, value in payload_expected.items()
                if config.get(key) != value
            }
            if payload_mismatches:
                raise RuntimeError(
                    f"{role} checkpoint lineage mismatch: {payload_mismatches}"
                )
        if int(best.get("epoch", -1)) != int(summary.get("best_epoch", -2)):
            raise RuntimeError(f"best checkpoint selection mismatch: {stage}")
        if int(final.get("epoch", -1)) != epochs:
            raise RuntimeError(f"final checkpoint epoch mismatch: {stage}")
        if int(final.get("early_stopping_stale_epochs", -1)) < 8:
            raise RuntimeError(f"final checkpoint lacks plateau convergence: {stage}")
        if hours == 4:
            if summary.get("initial_checkpoint") is not None:
                raise RuntimeError("AR4 cannot have a curriculum-parent checkpoint")
        else:
            previous = STAGES[index - 1]
            parent = roots[index - 1] / "checkpoint_best_validation_supervised.pt"
            if Path(str(summary.get("initial_checkpoint"))).resolve() != parent.resolve():
                raise RuntimeError(f"parent path mismatch: {stage}")
            if summary.get("initial_checkpoint_sha256") != sha256(parent):
                raise RuntimeError(f"parent checkpoint hash mismatch: {stage}")
    return "complete"


def main() -> None:
    args = arguments()
    print(next_stage(args.output_root.resolve(), args.model))


if __name__ == "__main__":
    main()
