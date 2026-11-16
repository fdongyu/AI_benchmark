#!/usr/bin/env python3
"""Load and validate a model connector for a versioned benchmark experiment.

A connector is a small Python file that constructs one neural network.  The
common trainer remains responsible for data, state updates, losses, staged
training, and checkpointing. Physical evaluation is a separate integration.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn


SOURCE_DIR = Path(__file__).resolve().parent
BENCHMARK_ROOT = SOURCE_DIR.parent
MODEL_ROOT = BENCHMARK_ROOT / "model"


def _repository_file(value: object, *, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty repository-relative path")
    relative = Path(value)
    if relative.is_absolute():
        raise ValueError(f"{field} must be repository-relative: {relative}")
    resolved = (BENCHMARK_ROOT / relative).resolve()
    try:
        resolved.relative_to(BENCHMARK_ROOT)
    except ValueError as error:
        raise ValueError(f"{field} escapes the benchmark repository: {relative}") from error
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def connector_path(model_spec: Mapping[str, Any]) -> Path:
    """Return a connector at ``model/<model_id>/connector.py``.

    Keeping every adapter beside its model description makes the plug-in
    boundary easy to inspect.  Resolving the path before checking the model
    root also rejects traversal and symlink escapes.
    """

    path = _repository_file(model_spec.get("connector"), field="connector")
    try:
        relative = path.relative_to(MODEL_ROOT)
    except ValueError as error:
        raise ValueError(
            f"connector must be inside {MODEL_ROOT.relative_to(BENCHMARK_ROOT)}: {path}"
        ) from error
    if len(relative.parts) != 2 or relative.name != "connector.py":
        raise ValueError(
            "connector must use model/<model_id>/connector.py: "
            f"{path.relative_to(BENCHMARK_ROOT)}"
        )
    return path


def connector_source_files(model_spec: Mapping[str, Any]) -> list[Path]:
    """Return the connector and model sources for reproducibility.

    A connector may expose ``source_files()`` so its implementation files stay
    beside the model description.  A versioned model list may additionally
    declare sources; both lists are validated and fingerprinted.
    """

    connector = connector_path(model_spec)
    paths = [connector]
    module = _load_module(connector)
    discover = getattr(module, "source_files", None)
    if discover is not None:
        if not callable(discover):
            raise TypeError(f"{connector}: source_files must be callable")
        discovered = discover()
        if not isinstance(discovered, (list, tuple)) or any(
            not isinstance(item, str) or not item.strip() for item in discovered
        ):
            raise TypeError(
                f"{connector}: source_files() must return a list or tuple of paths"
            )
        paths.extend(
            _repository_file(item, field=f"source_files()[{index}]")
            for index, item in enumerate(discovered)
        )
    declared = model_spec.get("source_files", [])
    if not isinstance(declared, list) or any(not isinstance(item, str) for item in declared):
        raise ValueError("source_files must be a list of repository-relative paths")
    paths.extend(
        _repository_file(item, field=f"source_files[{index}]")
        for index, item in enumerate(declared)
    )
    if len(paths) != len(set(paths)):
        raise ValueError("connector/source_files contains a duplicate path")
    return paths


def _load_module(path: Path):
    identity = hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:12]
    name = f"benchmark_model_connector_{path.stem}_{identity}"
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import model connector from {path}")
    module = importlib.util.module_from_spec(spec)
    inserted: list[str] = []
    for directory in (str(BENCHMARK_ROOT), str(SOURCE_DIR)):
        if directory not in sys.path:
            sys.path.insert(0, directory)
            inserted.append(directory)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    finally:
        for directory in inserted:
            sys.path.remove(directory)
    return module


def build_connected_model(
    model_spec: Mapping[str, Any],
    *,
    input_channels: int,
    output_channels: int,
) -> tuple[nn.Module, list[dict[str, Any]] | None, dict[str, Any]]:
    """Construct a model through the connector's small public interface."""

    path = connector_path(model_spec)
    module = _load_module(path)
    create_model = getattr(module, "create_model", None)
    if not callable(create_model):
        raise AttributeError(f"{path} must define create_model(...)")
    settings = model_spec.get("settings", {})
    if not isinstance(settings, Mapping):
        raise ValueError("connector settings must be a JSON object")
    try:
        json.dumps(settings, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("connector settings must contain finite JSON values") from error
    model = create_model(
        in_channels=int(input_channels),
        out_channels=int(output_channels),
        settings=dict(settings),
    )
    if not isinstance(model, nn.Module):
        raise TypeError(f"{path}: create_model() must return torch.nn.Module")
    for name, value in (*model.named_parameters(), *model.named_buffers()):
        if value.is_floating_point() and value.dtype != torch.float32:
            raise TypeError(f"{path}: {name} has {value.dtype}; the shared recipe requires float32")
        if value.is_complex() and value.dtype != torch.complex64:
            raise TypeError(f"{path}: {name} has {value.dtype}; expected complex64")
        if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(value).all():
            raise FloatingPointError(f"{path}: {name} contains a non-finite value")

    details: dict[str, Any] = {
        "connector": str(path.relative_to(BENCHMARK_ROOT)),
        "connector_settings": dict(settings),
    }
    describe = getattr(module, "model_details", None)
    if describe is not None:
        if not callable(describe):
            raise TypeError(f"{path}: model_details must be callable")
        extra = describe(model=model, settings=dict(settings))
        if not isinstance(extra, Mapping):
            raise TypeError(f"{path}: model_details() must return a mapping")
        try:
            json.dumps(extra, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValueError("model_details() must contain finite JSON values") from error
        details["connector_details"] = dict(extra)
    return model, None, details
