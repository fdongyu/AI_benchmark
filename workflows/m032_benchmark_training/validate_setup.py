#!/usr/bin/env python3
"""GPU preflight for one treatment in the m032 depth4/plateau campaign."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import torch


BENCHMARK_ROOT = Path(
    os.environ.get(
        "AI_BENCHMARK_ROOT",
        Path(__file__).resolve().parents[2],
    )
).resolve()
FOUNDATION_ROOT = Path(
    os.environ.get(
        "AI_FOUNDATION_ROOT",
        BENCHMARK_ROOT.parent / "AI_foundation",
    )
).resolve()
TRAINER_PATH = Path(
    os.environ.get(
        "M032_DEPTH_TRAINER",
        BENCHMARK_ROOT / "src" / "train_benchmark_model.py",
    )
).resolve()
DATASET_ROOT = Path(
    os.environ.get(
        "HOUSTON_DATASET_ROOT",
        FOUNDATION_ROOT / "DATA" / "rdycore_ml_dataset_core",
    )
).resolve()
EXPERIMENT_ROOT = Path(
    os.environ.get(
        "AI_BENCHMARK_EXPERIMENT_DIR",
        BENCHMARK_ROOT / "workflows" / "m032_benchmark_training",
    )
).resolve()
OUTPUT_ROOT = Path(
    os.environ.get(
        "AI_BENCHMARK_OUTPUT_ROOT",
        BENCHMARK_ROOT / "results" / EXPERIMENT_ROOT.name,
    )
).resolve()


def load_trainer():
    spec = importlib.util.spec_from_file_location("common13_trainer", TRAINER_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(TRAINER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def probe_stage_directory_lock(model: str) -> dict[str, str]:
    """Exercise the same atomic-directory primitive used by train.sbatch."""

    lock_parent = EXPERIMENT_ROOT / "locks" / "stages" / model
    lock_parent.mkdir(parents=True, exist_ok=True)
    lock_dir = lock_parent / "ar04"
    owner_file = lock_dir / "owner"
    owner = (
        "preflight_"
        f"{os.environ.get('SLURM_ARRAY_JOB_ID', 'manual')}_"
        f"{os.environ.get('SLURM_ARRAY_TASK_ID', '0')}_{os.getpid()}"
    )
    created = False
    try:
        lock_dir.mkdir()
        created = True
        owner_file.write_text(f"{owner}\n", encoding="utf-8")
        if owner_file.read_text(encoding="utf-8").strip() != owner:
            raise RuntimeError(f"stage lock owner round trip failed: {owner_file}")
        try:
            lock_dir.mkdir()
        except FileExistsError:
            pass
        else:
            raise RuntimeError("atomic stage lock did not reject a second owner")
    except FileExistsError as error:
        existing = "unknown"
        if owner_file.is_file():
            existing = owner_file.read_text(encoding="utf-8").strip() or "unknown"
        raise RuntimeError(
            f"preflight found an active or stale stage lock at {lock_dir}: {existing}"
        ) from error
    finally:
        if created:
            if owner_file.is_file() and owner_file.read_text(encoding="utf-8").strip() == owner:
                owner_file.unlink()
            lock_dir.rmdir()

    # Confirm that owner-checked cleanup made the lock reusable.
    lock_dir.mkdir()
    lock_dir.rmdir()
    return {
        "status": "valid",
        "primitive": "atomic_directory",
        "path": str(lock_dir),
        "contention_rejected": "true",
        "reacquire_succeeded": "true",
    }


def main() -> None:
    cli = parse_args()
    trainer = load_trainer()
    if cli.model not in trainer.MODEL_KEYS:
        raise ValueError(f"unknown treatment: {cli.model}")
    metadata = BENCHMARK_ROOT / "data" / "metadata" / "m032"
    parsed = trainer.parse_args(
        [
            "--subset",
            "m032",
            "--model",
            cli.model,
            "--rollout-hours",
            "4",
            "--dataset-root",
            str(DATASET_ROOT),
            "--shared-artifact-dir",
            str(metadata / "shared"),
            "--channel-artifact-dir",
            str(metadata / "channels"),
            "--output-root",
            str(OUTPUT_ROOT),
            "--device",
            cli.device,
            "--depth-weight",
            "4",
            "--momentum-weight",
            "1",
            "--supervised-scale",
            "3",
            "--weight-decay",
            "1e-5",
            "--no-deterministic",
        ]
    )
    trainer.resolve_protocol_defaults(parsed)
    trainer.validate_arguments(parsed)
    if cli.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA preflight requested but CUDA is unavailable")
    device = torch.device(cli.device)
    stage_lock_probe = probe_stage_directory_lock(cli.model)
    split_file = parsed.shared_artifact_dir / "trajectory_split.csv"
    normalization_file = parsed.shared_artifact_dir / "normalization.json"
    static_file = parsed.dataset_root / "static_grid.h5"
    masks_file = parsed.channel_artifact_dir / "channel_masks_fullres.npz"
    for path in (split_file, normalization_file, static_file, masks_file):
        if not path.is_file():
            raise FileNotFoundError(path)

    train = trainer.TrajectoryWindowDataset(
        split_file, "train", 4, parsed.dataset_root
    )
    validation = trainer.TrajectoryWindowDataset(
        split_file, "validation", 4, parsed.dataset_root
    )
    trainer.validate_dataset_contract("m032", train, validation)
    normalization = trainer.load_json(normalization_file)
    static = trainer.load_static(static_file, masks_file, device)
    sample = validation[0]
    current = sample["current"][None].to(device)
    features = trainer.make_features(
        current,
        sample["rainfall"][0:1].to(device),
        sample["ocean"][0:1].to(device),
        static,
        normalization,
    )
    expected_shape = (1, len(trainer.FEATURE_NAMES), *trainer.RASTER_SHAPE)
    if tuple(features.shape) != expected_shape:
        raise RuntimeError(f"feature shape {tuple(features.shape)} != {expected_shape}")
    if not torch.isfinite(features).all():
        raise FloatingPointError("common feature tensor is non-finite")

    model, groups, report = trainer.build_model(parsed, device)
    optimizer, _ = trainer.optimizer_for(parsed, model, groups)
    training_batch = {
        name: value[None].to(device)
        for name, value in sample.items()
    }
    optimizer.zero_grad(set_to_none=True)
    training_metrics = trainer.rollout_batch(
        parsed,
        model,
        training_batch,
        static,
        normalization,
        epoch=1,
        optimizer=optimizer,
    )
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        parsed.gradient_clip,
        error_if_nonfinite=True,
    )
    optimizer.step()
    if not all(np.isfinite(value) for value in training_metrics.values()):
        raise FloatingPointError("preflight training rollout returned non-finite metrics")
    model.eval()
    with torch.no_grad():
        prediction = model(features)
    if tuple(prediction.shape) != (1, 3, *trainer.RASTER_SHAPE):
        raise RuntimeError(f"prediction has invalid shape {tuple(prediction.shape)}")
    if prediction.dtype != torch.float32:
        raise TypeError(
            f"prediction has dtype {prediction.dtype}; shared training requires float32"
        )
    if not torch.isfinite(prediction).all():
        raise FloatingPointError("preflight prediction is non-finite")

    domain = static["domain"] > 0.5
    x_values = static["x_coordinate"][domain]
    y_values = static["y_coordinate"][domain]
    result = {
        "status": "valid",
        "subset": "m032",
        "model": cli.model,
        "protocol_id": trainer.protocol_spec("m032")["protocol_id"],
        "feature_names": list(trainer.FEATURE_NAMES),
        "feature_shape": list(features.shape),
        "x_range": [float(x_values.min()), float(x_values.max())],
        "y_range": [float(y_values.min()), float(y_values.max())],
        "prediction_shape": list(prediction.shape),
        "prediction_abs_max": float(prediction.abs().max()),
        "training_rollout_metrics": training_metrics,
        "gradient_norm_before_clip": float(gradient_norm),
        "stage_lock_probe": stage_lock_probe,
        "parameters": int(report["parameters"]),
        "trainable_parameters": int(report["trainable_parameters"]),
        "model_report": report,
    }
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
