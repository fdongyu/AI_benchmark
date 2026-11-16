#!/usr/bin/env python3
"""Resolve one immutable Slurm array task from the campaign model registry."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


REGISTRY = Path(__file__).resolve().parent / "model_registry.json"
MODEL_LIST_SUPPORT = Path(__file__).resolve().parents[2] / "src" / "model_list.py"


def load_model_list_support():
    spec = importlib.util.spec_from_file_location(
        "workflow_model_list_support", MODEL_LIST_SUPPORT
    )
    if spec is None or spec.loader is None:
        raise ImportError(MODEL_LIST_SUPPORT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_models() -> list[dict]:
    payload = json.loads(REGISTRY.read_text(encoding="utf-8"))
    return list(load_model_list_support().validate_model_list(payload))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-index", type=int)
    parser.add_argument(
        "--field", choices=("count", "model", "has_connector", "tsv"), default="tsv"
    )
    args = parser.parse_args()
    models = load_models()
    if args.field == "count":
        if args.task_index is not None:
            parser.error("--task-index is not used with --field count")
        print(len(models))
        return
    if args.task_index is None:
        parser.error("--task-index is required for model and tsv fields")
    if not 0 <= args.task_index < len(models):
        raise ValueError(
            f"task index {args.task_index} is outside 0..{len(models) - 1}"
        )
    entry = models[args.task_index]
    if args.field == "model":
        print(entry["model"])
        return
    if args.field == "has_connector":
        print("true" if "connector" in entry else "false")
        return
    values = (
        entry["model"],
        entry["learning_rate"],
        entry["endpoint_learning_rate"],
        entry["slice_epochs"],
    )
    if any("\t" in str(value) or "\n" in str(value) for value in values):
        raise RuntimeError("registry values cannot contain tabs or newlines")
    print("\t".join(str(value) for value in values))


if __name__ == "__main__":
    main()
