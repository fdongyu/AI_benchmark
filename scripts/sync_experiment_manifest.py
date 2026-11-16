#!/usr/bin/env python3
"""Check or update model-list fields in a versioned experiment manifest."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / "workflows"
MODEL_LIST_SUPPORT = ROOT / "src" / "model_list.py"
FROZEN_EXPERIMENT = (WORKFLOWS / "m032_benchmark_training").resolve()
FROZEN_EXPERIMENT_ID = "houston-common13-m032-depth4-plateau-v1"
FROZEN_PROTOCOL_ID = "houston-m032-common13-depth4-plateau-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_model_list_support():
    spec = importlib.util.spec_from_file_location(
        "sync_model_list_support", MODEL_LIST_SUPPORT
    )
    if spec is None or spec.loader is None:
        raise ImportError(MODEL_LIST_SUPPORT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument(
        "--write",
        action="store_true",
        help="atomically update the new experiment; default is read-only checking",
    )
    parser.add_argument(
        "--experiment-id",
        help="check that campaign_manifest.json uses this experiment ID",
    )
    parser.add_argument(
        "--protocol-id",
        help="check that campaign_manifest.json uses this m032 training-recipe ID",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_dir = args.experiment_dir.expanduser().resolve()
    try:
        experiment_dir.relative_to(WORKFLOWS.resolve())
    except ValueError as error:
        raise ValueError(f"experiment must be inside {WORKFLOWS}") from error
    if args.write and experiment_dir == FROZEN_EXPERIMENT:
        raise ValueError("refusing to rewrite the completed m032 benchmark experiment")
    if (args.experiment_id is None) != (args.protocol_id is None):
        raise ValueError("--experiment-id and --protocol-id must be provided together")
    registry_path = experiment_dir / "model_registry.json"
    manifest_path = experiment_dir / "campaign_manifest.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    models = load_model_list_support().validate_model_list(registry)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if args.experiment_id is not None:
        if experiment_dir != FROZEN_EXPERIMENT and (
            args.experiment_id == FROZEN_EXPERIMENT_ID
            or args.protocol_id == FROZEN_PROTOCOL_ID
        ):
            raise ValueError(
                "a new experiment must not reuse the completed v1 experiment or "
                "training-recipe ID"
            )
        identity_mismatches = {}
        if manifest.get("campaign_id") != args.experiment_id:
            identity_mismatches["campaign_id"] = {
                "found": manifest.get("campaign_id"),
                "expected": args.experiment_id,
            }
        manifest_protocol = manifest.get("protocol_ids", {}).get("m032")
        if manifest_protocol != args.protocol_id:
            identity_mismatches["protocol_ids.m032"] = {
                "found": manifest_protocol,
                "expected": args.protocol_id,
            }
        if identity_mismatches:
            print(json.dumps(identity_mismatches, indent=2, sort_keys=True))
            raise SystemExit(1)
    expected = {
        "model_registry_file": registry_path.name,
        "model_registry_sha256": sha256(registry_path),
        "model_keys": [entry["model"] for entry in models],
        "physics_treatments": sorted(
            entry["model"] for entry in models if entry["physics_regularizer"]
        ),
    }
    model_count = len(models)
    expected_array = f"0-{model_count - 1}%4"
    mismatches = {
        key: {"found": manifest.get(key), "expected": value}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    found_array = manifest.get("slurm", {}).get("array")
    if found_array != expected_array:
        mismatches["slurm.array"] = {
            "found": found_array,
            "expected": expected_array,
        }
    if not mismatches:
        print(f"{experiment_dir}: model list and manifest agree ({model_count} models)")
        return
    if not args.write:
        print(json.dumps(mismatches, indent=2, sort_keys=True))
        raise SystemExit(1)
    manifest.update(expected)
    if not isinstance(manifest.get("slurm"), dict):
        manifest["slurm"] = {}
    manifest["slurm"]["array"] = expected_array
    temporary = manifest_path.with_name(
        f".{manifest_path.name}.tmp.{os.getpid()}.{time.time_ns()}"
    )
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, manifest_path)
    print(f"updated {manifest_path} for {model_count} models")


if __name__ == "__main__":
    main()
