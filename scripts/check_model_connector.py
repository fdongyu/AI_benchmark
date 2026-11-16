#!/usr/bin/env python3
"""Check a model connector's shape, finite values, gradients, and metadata."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
LOADER_PATH = ROOT / "src" / "model_connector.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_connector_support():
    spec = importlib.util.spec_from_file_location("benchmark_connector_support", LOADER_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(LOADER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    location = parser.add_mutually_exclusive_group(required=True)
    location.add_argument(
        "--model",
        help="model folder name; resolves to model/<name>/connector.py",
    )
    location.add_argument(
        "--connector",
        help="repository-relative connector path (advanced use)",
    )
    parser.add_argument("--settings-json", default="{}")
    parser.add_argument("--source-file", action="append", default=[])
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = json.loads(args.settings_json)
    if not isinstance(settings, dict):
        raise ValueError("--settings-json must contain a JSON object")
    if args.height < 1 or args.width < 1:
        raise ValueError("--height and --width must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    support = load_connector_support()
    connector = (
        Path("model") / args.model / "connector.py"
        if args.model is not None
        else Path(args.connector)
    )
    if connector.is_absolute():
        try:
            connector = connector.resolve().relative_to(ROOT)
        except ValueError as error:
            raise ValueError("connector must be inside the benchmark repository") from error
    model_spec = {
        "connector": str(connector),
        "settings": settings,
        "source_files": list(args.source_file),
    }
    torch.manual_seed(20260917)
    model, _groups, details = support.build_connected_model(
        model_spec, input_channels=13, output_channels=3
    )
    device = torch.device(args.device)
    model.to(device)
    model.train()
    values = torch.randn(1, 13, args.height, args.width, device=device)
    prediction = model(values)
    expected = (1, 3, args.height, args.width)
    if tuple(prediction.shape) != expected:
        raise RuntimeError(f"connector output {tuple(prediction.shape)} != {expected}")
    if prediction.dtype != torch.float32:
        raise TypeError(
            f"connector output has {prediction.dtype}; the shared recipe requires float32"
        )
    if not torch.isfinite(prediction).all():
        raise FloatingPointError("connector output contains a non-finite value")
    prediction.square().mean().backward()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("connector model has no trainable parameters")
    if any(parameter.grad is None or not torch.isfinite(parameter.grad).all() for parameter in trainable):
        raise FloatingPointError("connector produced a missing or non-finite gradient")

    result = {
        "status": "valid",
        "connector": model_spec["connector"],
        "input_shape": list(values.shape),
        "output_shape": list(prediction.shape),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "details": details,
        "source_files": [
            {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
            }
            for path in support.connector_source_files(model_spec)
        ],
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
