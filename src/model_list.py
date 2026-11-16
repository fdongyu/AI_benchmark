#!/usr/bin/env python3
"""Validate the readable JSON model list used by a benchmark experiment."""

from __future__ import annotations

import math
import re
from typing import Any, Mapping


MODEL_ID = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
REQUIRED_FIELDS = (
    "model",
    "architecture_key",
    "physics_regularizer",
    "initialization_policy",
    "learning_rate",
    "endpoint_learning_rate",
    "slice_epochs",
    "description",
)


def validate_model_list(payload: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    """Return validated model entries or fail before paths/jobs are constructed."""

    if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
        raise ValueError("model list must be a schema_version=1 JSON object")
    entries = payload.get("models")
    if not isinstance(entries, list) or not entries:
        raise ValueError("model list must contain at least one model")
    validated: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(entries):
        if not isinstance(raw, Mapping):
            raise ValueError(f"models[{index}] must be a JSON object")
        missing = [field for field in REQUIRED_FIELDS if field not in raw]
        if missing:
            raise ValueError(f"models[{index}] is missing fields: {missing}")
        entry = dict(raw)
        model = entry["model"]
        if not isinstance(model, str) or MODEL_ID.fullmatch(model) is None:
            raise ValueError(
                f"models[{index}].model must match {MODEL_ID.pattern}: {model!r}"
            )
        if model in seen:
            raise ValueError(f"model list contains duplicate model id: {model}")
        seen.add(model)
        for field in ("architecture_key", "initialization_policy", "description"):
            if not isinstance(entry[field], str) or not entry[field].strip():
                raise ValueError(f"{model}.{field} must be a non-empty string")
        if not isinstance(entry["physics_regularizer"], bool):
            raise ValueError(f"{model}.physics_regularizer must be true or false")
        for field in ("learning_rate", "endpoint_learning_rate"):
            value = entry[field]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{model}.{field} must be a number")
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{model}.{field} must be finite and non-negative")
        if float(entry["learning_rate"]) <= 0.0:
            raise ValueError(f"{model}.learning_rate must be positive")
        if float(entry["endpoint_learning_rate"]) > float(entry["learning_rate"]):
            raise ValueError(
                f"{model}.endpoint_learning_rate cannot exceed learning_rate"
            )
        if isinstance(entry["slice_epochs"], bool) or not isinstance(
            entry["slice_epochs"], int
        ) or entry["slice_epochs"] < 1:
            raise ValueError(f"{model}.slice_epochs must be a positive integer")
        if "connector" in entry:
            if not isinstance(entry["connector"], str) or not entry["connector"].strip():
                raise ValueError(f"{model}.connector must be a non-empty path")
            if not isinstance(entry.get("settings", {}), Mapping):
                raise ValueError(f"{model}.settings must be a JSON object")
            sources = entry.get("source_files", [])
            if not isinstance(sources, list) or any(
                not isinstance(source, str) or not source.strip() for source in sources
            ):
                raise ValueError(f"{model}.source_files must be a list of paths")
        validated.append(entry)
    return tuple(validated)
