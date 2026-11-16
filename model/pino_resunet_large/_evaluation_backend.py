#!/usr/bin/env python3
"""Internal rollout and metric backend for PINO-ResU-Net-large evaluation.

The public entry point is ``evaluate.py`` in this model folder. This module
adapts the 13-feature checkpoint and residual state update to the shared flood
evaluation library. It remains separate so the strict model adapter can reuse
the rollout implementation.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")

BENCHMARK_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_LIB = BENCHMARK_ROOT / "analysis" / "lib"
if str(ANALYSIS_LIB) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_LIB))

import pandas as pd
import torch

from flood_evaluation import evaluate as legacy


EXPECTED_MODEL = "pino_resunet_large"
EXPECTED_SUBSET = "m032"
EXPECTED_HORIZON = 32
EXPECTED_PROTOCOL = "houston-m032-common13-residual-v1"
MODEL_KEY = "pino_resunet_large_common13"
MODEL_LABEL = "PINO-ResU-Net-large (common13)"
CLI_DESCRIPTION = "Evaluate PINO-ResU-Net-large on the common m032 test set."


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _write_json(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def _import_module(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, str(path))
    if specification is None or specification.loader is None:
        raise ImportError(f"cannot import {name} from {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def _config_proxy(checkpoint_config: dict[str, Any]) -> SimpleNamespace:
    dataset_root = Path(checkpoint_config["dataset_root"]).resolve()
    shared = Path(checkpoint_config["shared_artifact_dir"]).resolve()
    return SimpleNamespace(
        data=SimpleNamespace(
            dataset_root=dataset_root,
            split_csv=(shared / "trajectory_split.csv").resolve(),
            normalization_json=(shared / "normalization.json").resolve(),
            static_grid=(dataset_root / "static_grid.h5").resolve(),
        ),
        physics=SimpleNamespace(
            manning_n=float(checkpoint_config["manning_n"]),
            dry_tolerance_m=float(checkpoint_config["dry_tolerance_m"]),
            max_velocity_m_s=float(checkpoint_config["max_velocity_m_s"]),
            dt_s=float(checkpoint_config["dt_s"]),
            dx_m=float(checkpoint_config["dx_m"]),
            dy_m=float(checkpoint_config["dy_m"]),
        ),
        training=SimpleNamespace(seed=int(checkpoint_config["seed"])),
    )


def _require_equal(actual: Any, expected: Any, name: str) -> None:
    if actual != expected:
        raise RuntimeError(f"{name} mismatch: found {actual!r}, expected {expected!r}")


def _validate_scientific_contract(
    *,
    common: Any,
    checkpoint: Path,
    payload: dict[str, Any],
    stage_summary: dict[str, Any],
    provenance: dict[str, Any],
    checkpoint_role: str,
) -> None:
    config = payload.get("config", {})
    _require_equal(config.get("model"), EXPECTED_MODEL, "checkpoint model")
    _require_equal(config.get("subset"), EXPECTED_SUBSET, "checkpoint subset")
    _require_equal(config.get("rollout_hours"), EXPECTED_HORIZON, "rollout horizon")
    _require_equal(config.get("protocol_id"), EXPECTED_PROTOCOL, "protocol")
    _require_equal(config.get("architecture_key"), "houston_resunet_large", "architecture")
    _require_equal(config.get("external_residual_update"), True, "residual update")
    _require_equal(
        config.get("prediction_target"),
        "normalized conservative-state increment",
        "prediction target",
    )
    _require_equal(list(config.get("input_features", ())), list(common.FEATURE_NAMES), "features")
    _require_equal(list(config.get("auxiliary_losses", ())), [], "auxiliary losses")

    _require_equal(stage_summary.get("complete"), True, "stage completion")
    _require_equal(stage_summary.get("stop_reason"), "fixed_budget_complete", "stop reason")
    _require_equal(stage_summary.get("epochs_completed"), 48, "completed epochs")
    _require_equal(stage_summary.get("epochs_required"), 48, "required epochs")
    _require_equal(stage_summary.get("model"), EXPECTED_MODEL, "stage model")
    _require_equal(stage_summary.get("protocol_id"), EXPECTED_PROTOCOL, "stage protocol")
    if checkpoint_role == "best":
        _require_equal(payload.get("epoch"), stage_summary.get("best_epoch"), "best epoch")
        expected_name = "checkpoint_best_validation_supervised.pt"
    else:
        _require_equal(payload.get("epoch"), stage_summary.get("epochs_completed"), "final epoch")
        expected_name = "checkpoint_final.pt"
    _require_equal(checkpoint.name, expected_name, "checkpoint filename")

    files = provenance.get("files", {})
    required_files = (
        "campaign_manifest",
        "channel_masks",
        "normalization",
        "source_1",
        "source_2",
        "static_grid",
        "trainer",
        "trajectory_split",
    )
    for name in required_files:
        identity = files.get(name)
        if not isinstance(identity, dict) or "path" not in identity or "sha256" not in identity:
            raise RuntimeError(f"provenance lacks a hashed {name} identity")
        path = Path(identity["path"]).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = common.sha256_file(path)
        _require_equal(digest, identity["sha256"], f"provenance hash for {name}")

    model_record = provenance.get("model", {})
    _require_equal(model_record.get("treatment"), EXPECTED_MODEL, "provenance treatment")
    _require_equal(model_record.get("strict_fp32"), True, "strict FP32")
    _require_equal(model_record.get("tf32_enabled"), False, "TF32 setting")
    _require_equal(model_record.get("prediction_target"), config["prediction_target"], "provenance target")


def _rewrite_identity_columns(output_dir: Path, run_key: str, epoch: int) -> None:
    products = (
        ("domain_metrics_by_trajectory.csv", None),
        ("gauge_metrics_by_trajectory.csv", None),
        ("rollout_status.csv", None),
        ("gauge_timeseries.csv.gz", "gzip"),
    )
    for filename, compression in products:
        path = output_dir / filename
        frame = pd.read_csv(path, compression=compression)
        replacements = {
            "run_key": run_key,
            "model_key": MODEL_KEY,
            "model": MODEL_LABEL,
            "checkpoint_epoch": epoch,
        }
        for column, value in replacements.items():
            if column in frame.columns:
                frame[column] = value
        temporary = path.with_suffix(path.suffix + ".tmp")
        frame.to_csv(temporary, index=False, compression=compression)
        os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    benchmark = BENCHMARK_ROOT
    parser = argparse.ArgumentParser(description=CLI_DESCRIPTION)
    parser.add_argument(
        "--model-loader",
        type=Path,
        required=True,
        help="Python module that reconstructs the recorded PINO-ResU-Net-large model.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--checkpoint-role", choices=("best", "final"), default="best"
    )
    parser.add_argument(
        "--campaign-primary-checkpoint",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--gauge-file",
        type=Path,
        default=benchmark / "data/metadata/validation/rdycore_usgs_gauge_cells.csv",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=("test",))
    parser.add_argument("--evaluation-members", type=int, default=32)
    parser.add_argument("--max-events", type=int, default=0)
    parser.add_argument("--max-lead-hours", type=float, default=0.0)
    parser.add_argument("--event-id", action="append", default=[])
    parser.add_argument("--snapshot-member", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cuda")
    parser.add_argument("--threads", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_loader_path = args.model_loader.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    primary_checkpoint = args.campaign_primary_checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    for path in (model_loader_path, checkpoint, primary_checkpoint, args.gauge_file.resolve()):
        if not path.is_file():
            raise FileNotFoundError(path)
    stage_dir = checkpoint.parent
    stage_summary_path = stage_dir / "stage_summary.json"
    provenance_path = stage_dir / "provenance.json"
    training_config_path = stage_dir / "training_config.json"
    for path in (stage_summary_path, provenance_path, training_config_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    common = _import_module("pino_resunet_large_model_loader", model_loader_path)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model_state" not in payload or "config" not in payload:
        raise RuntimeError(f"unsupported checkpoint payload: {checkpoint}")
    stage_summary = _load_json(stage_summary_path)
    provenance = _load_json(provenance_path)
    _validate_scientific_contract(
        common=common,
        checkpoint=checkpoint,
        payload=payload,
        stage_summary=stage_summary,
        provenance=provenance,
        checkpoint_role=args.checkpoint_role,
    )
    checkpoint_config = payload["config"]
    proxy = _config_proxy(checkpoint_config)
    normalization = common.load_json(proxy.data.normalization_json)
    masks_path = (
        Path(checkpoint_config["channel_artifact_dir"]).resolve()
        / "channel_masks_fullres.npz"
    )
    if not masks_path.is_file():
        raise FileNotFoundError(masks_path)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    def load_config_adapter(_: Path) -> SimpleNamespace:
        return proxy

    def build_checkpoint_model(
        _config: SimpleNamespace, selected_checkpoint: Path, device: torch.device
    ) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any]]:
        if selected_checkpoint.resolve() != checkpoint:
            raise RuntimeError("legacy evaluator requested an unexpected checkpoint")
        model_args = Namespace(**checkpoint_config)
        model, _, model_report = common.build_model(model_args, device)
        incompatibility = model.load_state_dict(payload["model_state"], strict=True)
        if incompatibility.missing_keys or incompatibility.unexpected_keys:
            raise RuntimeError(f"strict checkpoint load failed: {incompatibility}")
        expected_parameters = provenance.get("model", {}).get("parameters")
        actual_parameters = sum(parameter.numel() for parameter in model.parameters())
        _require_equal(actual_parameters, expected_parameters, "model parameter count")
        _require_equal(model_report.get("parameters"), expected_parameters, "model report")
        for name, value in model.state_dict().items():
            if not torch.isfinite(value).all():
                raise RuntimeError(f"checkpoint tensor is non-finite: {name}")
        model.eval()
        return model, normalization, payload

    def torch_static_adapter(
        _static: dict[str, Any], device: torch.device, _manning_n: float
    ) -> dict[str, torch.Tensor]:
        return common.load_static(proxy.data.static_grid, masks_path, device)

    def one_step_adapter(
        model: torch.nn.Module,
        current: torch.Tensor,
        rainfall: torch.Tensor,
        ocean_height: torch.Tensor,
        static: dict[str, torch.Tensor],
        selected_normalization: dict[str, Any],
        config: SimpleNamespace,
    ) -> torch.Tensor:
        features = common.make_features(
            current, rainfall, ocean_height, static, selected_normalization
        )
        if features.shape[1] != len(common.FEATURE_NAMES):
            raise RuntimeError(f"constructed {features.shape[1]} features")
        normalized_delta = model(features)
        following, _ = common.residual_state_update(
            current,
            normalized_delta.float(),
            selected_normalization,
            static["domain"],
            dry_tolerance=config.physics.dry_tolerance_m,
            max_velocity=config.physics.max_velocity_m_s,
            straight_through=False,
        )
        return following

    legacy.load_config = load_config_adapter
    legacy._validate_config = lambda _config: None
    legacy._checkpoint_and_model = build_checkpoint_model
    legacy._torch_static = torch_static_adapter
    legacy._one_step = one_step_adapter
    legacy.project_wet_dry = common.project_wet_dry

    metadata = legacy.evaluate_rollouts(
        config_path=training_config_path,
        checkpoint=checkpoint,
        gauge_file=args.gauge_file,
        output_dir=output_dir,
        split=args.split,
        evaluation_members=args.evaluation_members,
        max_events=args.max_events,
        max_lead_hours=args.max_lead_hours,
        event_ids=tuple(args.event_id),
        snapshot_member=args.snapshot_member,
        device_name=args.device,
        threads=args.threads,
        strict=True,
    )

    epoch = int(payload["epoch"])
    run_key = f"houston_{MODEL_KEY}_epoch{epoch:04d}"
    _rewrite_identity_columns(output_dir, run_key, epoch)
    primary_payload = torch.load(primary_checkpoint, map_location="cpu", weights_only=False)
    primary_epoch = int(primary_payload.get("epoch", -1))
    primary_history = primary_payload.get("history_row", {})
    selected_history = payload.get("history_row", {})
    metadata.update(
        {
            "run_key": run_key,
            "model_key": MODEL_KEY,
            "model": MODEL_LABEL,
            "checkpoint_role": args.checkpoint_role,
            "checkpoint_epoch": epoch,
            "checkpoint_sha256": common.sha256_file(checkpoint),
            "training_config": str(training_config_path),
            "training_config_sha256": common.sha256_file(training_config_path),
            "training_provenance": str(provenance_path),
            "training_provenance_sha256": common.sha256_file(provenance_path),
            "stage_summary": str(stage_summary_path),
            "stage_summary_sha256": common.sha256_file(stage_summary_path),
            "evaluation_adapter": str(Path(__file__).resolve()),
            "evaluation_adapter_sha256": common.sha256_file(Path(__file__).resolve()),
            "evaluation_model_loader": str(model_loader_path),
            "evaluation_model_loader_sha256": common.sha256_file(model_loader_path),
            "protocol_id": EXPECTED_PROTOCOL,
            "input_features": list(common.FEATURE_NAMES),
            "prediction_parameterization": checkpoint_config["prediction_target"],
            "external_residual_update": True,
            "auxiliary_losses": [],
            "selected_validation_supervised": float(
                selected_history.get(
                    "validation_supervised",
                    stage_summary.get("best_validation_supervised", math.nan),
                )
            ),
            "campaign_primary_checkpoint": str(primary_checkpoint),
            "campaign_primary_checkpoint_epoch": primary_epoch,
            "campaign_primary_checkpoint_sha256": common.sha256_file(primary_checkpoint),
            "campaign_primary_validation_supervised": float(
                primary_history.get("validation_supervised", math.nan)
            ),
            "checkpoint_selection_note": (
                "The common13 campaign declares the fixed-budget final checkpoint primary. "
                "This export uses the validation-selected best checkpoint for a symmetric "
                "best-to-best comparison with the previous PINO-ResU-Net-large plot; "
                "both identities are recorded."
            ),
        }
    )
    if not math.isfinite(metadata["selected_validation_supervised"]):
        raise RuntimeError("selected checkpoint lacks a finite validation score")
    if not math.isfinite(metadata["campaign_primary_validation_supervised"]):
        raise RuntimeError("campaign-primary checkpoint lacks a finite validation score")
    _write_json(metadata, output_dir / "export_metadata.json")
    print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)
