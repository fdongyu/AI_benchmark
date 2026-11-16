#!/usr/bin/env python3
"""Validate resumable or converged multi-model depth4/plateau artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path

import torch


PROTOCOL_ID = os.environ.get(
    "AI_BENCHMARK_PROTOCOL_ID", "houston-m032-common13-depth4-plateau-v1"
)
REGISTRY = Path(__file__).resolve().parent / "model_registry.json"
MODEL_LIST_SUPPORT = Path(__file__).resolve().parents[2] / "src" / "model_list.py"


def load_model_list_support():
    spec = importlib.util.spec_from_file_location(
        "validate_stage_model_list_support", MODEL_LIST_SUPPORT
    )
    if spec is None or spec.loader is None:
        raise ImportError(MODEL_LIST_SUPPORT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def model_specs() -> dict[str, dict]:
    payload = json.loads(REGISTRY.read_text(encoding="utf-8"))
    entries = load_model_list_support().validate_model_list(payload)
    return {str(entry["model"]): dict(entry) for entry in entries}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("progress", "complete"), required=True)
    parser.add_argument("--model", choices=tuple(model_specs()), required=True)
    parser.add_argument("--hours", type=int, choices=(4, 8, 16, 32), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--invocation-id", default="")
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_json(path: Path) -> dict:
    require(path.is_file(), f"missing {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_checkpoint(payload: dict, config: dict, label: str) -> None:
    saved = payload.get("config", {})
    keys = (
        "protocol_id",
        "subset",
        "model",
        "rollout_hours",
        "input_features",
        "depth_weight",
        "momentum_weight",
        "supervised_scale",
        "learning_rate_schedule",
        "plateau_factor",
        "plateau_patience",
        "plateau_threshold",
        "plateau_threshold_mode",
        "plateau_cooldown",
        "plateau_min_lr",
        "plateau_eps",
    )
    mismatches = {
        key: (saved.get(key), config.get(key))
        for key in keys
        if saved.get(key) != config.get(key)
    }
    require(not mismatches, f"{label} checkpoint config mismatch: {mismatches}")
    require("scheduler_state" in payload, f"{label} checkpoint lacks scheduler state")
    require("optimizer_state" in payload, f"{label} checkpoint lacks optimizer state")


def main() -> None:
    args = arguments()
    specs = model_specs()
    spec = specs[args.model]
    stage = args.output_root.resolve() / args.model / f"stage_ar{args.hours:02d}"
    config = load_json(stage / "training_config.json")
    expected = {
        "protocol_id": PROTOCOL_ID,
        "subset": "m032",
        "model": args.model,
        "rollout_hours": args.hours,
        "epochs": 240,
        "transitions_per_epoch": 8192,
        "validation_transitions": 2048,
        "depth_weight": 4.0,
        "momentum_weight": 1.0,
        "supervised_scale": 3.0,
        "learning_rate": float(spec["learning_rate"]),
        "endpoint_learning_rate": float(spec["endpoint_learning_rate"]),
        "weight_decay": 1e-5,
        "physics_enabled": bool(spec["physics_regularizer"]),
        "learning_rate_schedule": "reduce_on_plateau",
        "plateau_factor": 0.5,
        "plateau_patience": 2,
        "plateau_threshold": 0.0001,
        "plateau_threshold_mode": "rel",
        "plateau_cooldown": 0,
        "plateau_min_lr": 0.0,
        "plateau_eps": 1e-8,
        "cudnn_deterministic": True,
        "torch_deterministic_algorithms": False,
    }
    mismatches = {
        key: (config.get(key), value)
        for key, value in expected.items()
        if config.get(key) != value
    }
    require(not mismatches, f"training config mismatch: {mismatches}")
    provenance = load_json(stage / "provenance.json")
    require(
        provenance.get("protocol", {}).get("protocol_id") == PROTOCOL_ID,
        "provenance protocol mismatch",
    )
    require(
        provenance.get("model", {}).get("physics_regularizer")
        is bool(spec["physics_regularizer"]),
        "provenance physics-treatment mismatch",
    )

    history_path = stage / "history.csv"
    require(history_path.is_file(), f"missing {history_path}")
    with history_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    epochs = [int(row["epoch"]) for row in rows]
    require(epochs == list(range(1, len(rows) + 1)), "history epochs are not contiguous")
    require(rows, "history is empty")
    for row in rows:
        for column in (
            "train_supervised",
            "train_objective",
            "validation_supervised",
            "validation_objective",
        ):
            require(math.isfinite(float(row[column])), f"non-finite {column}")

    last_path = stage / "checkpoint_last.pt"
    require(last_path.is_file() and last_path.stat().st_size > 0, f"missing {last_path}")
    checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
    validate_checkpoint(checkpoint, config, "last")
    require(int(checkpoint.get("epoch", -1)) == epochs[-1], "last checkpoint/history mismatch")
    require(checkpoint.get("config", {}).get("protocol_id") == PROTOCOL_ID, "checkpoint protocol mismatch")
    require(
        int(checkpoint.get("best_epoch", -1)) <= epochs[-1],
        "last checkpoint best epoch is invalid",
    )

    status = load_json(stage / "slice_status.json")
    require(int(status.get("epochs_completed", -1)) == epochs[-1], "status/history mismatch")
    if args.mode == "progress":
        require(status.get("complete") is False, "progress stage is unexpectedly complete")
        if args.invocation_id:
            require(status.get("invocation_id") == args.invocation_id, "invocation mismatch")
    else:
        summary = load_json(stage / "stage_summary.json")
        require(summary.get("complete") is True, "stage is not complete")
        require(summary.get("stop_reason") == "validation_plateau", "wrong stop reason")
        require(12 <= epochs[-1] <= 240, "converged epoch is outside 12..240")
        require(int(summary.get("best_epoch", 0)) <= epochs[-1], "best epoch is invalid")
        require(
            int(summary.get("epochs_completed", -1)) == epochs[-1],
            "summary/history epoch mismatch",
        )
        require(
            int(summary.get("early_stopping_stale_epochs", -1)) >= 8,
            "summary does not satisfy plateau patience",
        )
        require(
            int(checkpoint.get("early_stopping_stale_epochs", -1)) >= 8,
            "last checkpoint does not satisfy plateau patience",
        )
        require(
            int(checkpoint.get("best_epoch", -1)) == int(summary["best_epoch"]),
            "summary/last best epoch mismatch",
        )
        require(
            math.isclose(
                float(checkpoint.get("best_validation_supervised", math.inf)),
                float(summary["best_validation_supervised"]),
                rel_tol=0.0,
                abs_tol=0.0,
            ),
            "summary/last best score mismatch",
        )
        payloads = {"last": checkpoint}
        paths = {"last": last_path}
        for role, name in (
            ("final", "checkpoint_final.pt"),
            ("best", "checkpoint_best_validation_supervised.pt"),
        ):
            path = stage / name
            require(path.is_file() and path.stat().st_size > 0, f"missing {path}")
            paths[role] = path
            payloads[role] = torch.load(path, map_location="cpu", weights_only=False)
            validate_checkpoint(payloads[role], config, role)
        require(int(payloads["final"].get("epoch", -1)) == epochs[-1], "final epoch mismatch")
        require(
            int(payloads["best"].get("epoch", -1)) == int(summary["best_epoch"]),
            "best checkpoint epoch mismatch",
        )
        require(
            math.isclose(
                float(payloads["best"].get("best_validation_supervised", math.inf)),
                float(summary["best_validation_supervised"]),
                rel_tol=0.0,
                abs_tol=0.0,
            ),
            "best checkpoint score mismatch",
        )
        expected_hashes = summary.get("checkpoint_sha256", {})
        require(set(expected_hashes) == set(paths), "summary checkpoint hashes are incomplete")
        for role, path in paths.items():
            require(sha256(path) == expected_hashes[role], f"{role} checkpoint hash mismatch")
        require(expected_hashes["last"] == expected_hashes["final"], "final is not convergence-last")

        initial = summary.get("initial_checkpoint")
        initial_hash = summary.get("initial_checkpoint_sha256")
        if args.hours == 4:
            require(
                initial is None and initial_hash is None,
                "AR4 cannot have a curriculum-parent checkpoint",
            )
            require(
                "initial_checkpoint" not in provenance.get("files", {}),
                "AR4 provenance must not have a parent checkpoint",
            )
            require(
                provenance.get("initialization_policy")
                == spec["initialization_policy"],
                "AR4 model initialization policy mismatch",
            )
        else:
            previous = {8: 4, 16: 8, 32: 16}[args.hours]
            expected_parent = (
                args.output_root.resolve()
                / args.model
                / f"stage_ar{previous:02d}"
                / "checkpoint_best_validation_supervised.pt"
            )
            require(Path(str(initial)).resolve() == expected_parent, "parent path mismatch")
            require(expected_parent.is_file(), f"missing parent checkpoint {expected_parent}")
            require(sha256(expected_parent) == initial_hash, "parent checkpoint hash mismatch")
            require(config.get("stage_initialization_sha256") == initial_hash, "config parent hash mismatch")
            parent_identity = provenance.get("files", {}).get("initial_checkpoint", {})
            require(parent_identity.get("sha256") == initial_hash, "provenance parent hash mismatch")
    print(
        json.dumps(
            {
                "mode": args.mode,
                "model": args.model,
                "hours": args.hours,
                "epochs": epochs[-1],
            }
        )
    )


if __name__ == "__main__":
    main()
