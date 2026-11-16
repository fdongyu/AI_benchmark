#!/usr/bin/env python3
"""Atomically record an accepted m032 depth4/plateau submission."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = Path(__file__).resolve().parent
MODEL_LIST_SUPPORT = ROOT / "src" / "model_list.py"
MODEL_CONNECTOR_SUPPORT = ROOT / "src" / "model_connector.py"


def load_model_list_support():
    spec = importlib.util.spec_from_file_location(
        "receipt_model_list_support", MODEL_LIST_SUPPORT
    )
    if spec is None or spec.loader is None:
        raise ImportError(MODEL_LIST_SUPPORT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_model_connector_support():
    spec = importlib.util.spec_from_file_location(
        "receipt_model_connector_support", MODEL_CONNECTOR_SUPPORT
    )
    if spec is None or spec.loader is None:
        raise ImportError(MODEL_CONNECTOR_SUPPORT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-job", required=True)
    parser.add_argument("--training-job", required=True)
    parser.add_argument("--preflight-command", required=True)
    parser.add_argument("--training-command", required=True)
    parser.add_argument("--receipt-dir", type=Path, required=True)
    parser.add_argument("--array-spec", default="0-4%4")
    parser.add_argument(
        "--tasks",
        default="",
        help="Optional comma-separated task subset for a targeted recovery array.",
    )
    parser.add_argument("--dependency", default="")
    parser.add_argument("--parent-training-job", default="")
    parser.add_argument("--reason", default="")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    source_paths = [ROOT / "src" / "train_benchmark_model.py"]
    source_paths.append(MODEL_CONNECTOR_SUPPORT)
    source_paths.append(MODEL_LIST_SUPPORT)
    source_paths.extend(sorted((ROOT / "src" / "vendor" / "baselines").glob("*.py")))
    source_paths.extend(sorted((ROOT / "src" / "vendor" / "pino").glob("*.py")))
    source_paths.extend(sorted(WORKFLOW.glob("*.py")))
    source_paths.extend(sorted(WORKFLOW.glob("*.sh")))
    source_paths.extend(sorted(WORKFLOW.glob("*.sbatch")))
    source_paths.extend(sorted(WORKFLOW.glob("*.md")))
    source_paths.append(WORKFLOW / "campaign_manifest.json")
    source_paths.append(WORKFLOW / "model_registry.json")
    registry = json.loads(
        (WORKFLOW / "model_registry.json").read_text(encoding="utf-8")
    )
    models_from_list = load_model_list_support().validate_model_list(registry)
    connector_support = load_model_connector_support()
    for model in models_from_list:
        if "connector" in model:
            source_paths.extend(connector_support.connector_source_files(model))
            continue
        declared = model.get("source_files", [])
        for relative_name in declared:
            relative = Path(relative_name)
            if relative.is_absolute():
                raise ValueError(f"model source must be repository-relative: {relative}")
            source = (ROOT / relative).resolve()
            try:
                source.relative_to(ROOT)
            except ValueError as error:
                raise ValueError(f"model source escapes repository: {relative}") from error
            if not source.is_file():
                raise FileNotFoundError(source)
            source_paths.append(source)
    unique_paths = sorted(set(path.resolve() for path in source_paths))
    experiment = json.loads(
        (WORKFLOW / "campaign_manifest.json").read_text(encoding="utf-8")
    )
    models = experiment["model_keys"]
    if args.tasks:
        selected_tasks = [int(value) for value in args.tasks.split(",")]
    else:
        selected_tasks = list(range(len(models)))
    if len(set(selected_tasks)) != len(selected_tasks) or any(
        task < 0 or task >= len(models) for task in selected_tasks
    ):
        raise ValueError(f"invalid task selection: {selected_tasks}")
    task_map = []
    for task in selected_tasks:
        task_map.append(
            {
                "task": task,
                "subset": "m032",
                "model": models[task],
            }
        )
    payload = {
        "campaign_id": experiment["campaign_id"],
        "submitted_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "user": os.environ.get("USER"),
        "preflight_job_id": args.preflight_job,
        "training_job_id": args.training_job,
        "dependency": args.dependency or f"afterok:{args.preflight_job}",
        "array": args.array_spec,
        "account": "m3780_g",
        "commands": {
            "preflight": args.preflight_command,
            "training": args.training_command,
        },
        "task_map": task_map,
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in unique_paths
        },
    }
    if "pino_resunet_large_reference" in experiment:
        payload["pino_resunet_large_reference"] = experiment[
            "pino_resunet_large_reference"
        ]
    if args.parent_training_job or args.reason:
        payload["recovery"] = {
            "parent_training_job_id": args.parent_training_job or None,
            "reason": args.reason or None,
        }
    args.receipt_dir.mkdir(parents=True, exist_ok=True)
    destination = args.receipt_dir / f"submission_{args.training_job}.json"
    temporary = destination.with_name(
        f".{destination.name}.tmp.{os.getpid()}.{time.time_ns()}"
    )
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, destination)
    print(destination)


if __name__ == "__main__":
    main()
