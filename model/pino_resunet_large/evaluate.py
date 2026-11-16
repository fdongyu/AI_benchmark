#!/usr/bin/env python3
"""Evaluate the retained PINO-ResU-Net-large on the common m032 test set.

This is the model's public evaluation entry point. The rollout implementation
lives in the private ``_evaluation_backend.py`` module, while this file validates
the depth-emphasized, validation-plateau contract instead of weakening the
historical backend's fixed-budget checks.
"""

from __future__ import annotations

import json
import hashlib
import math
import os
from pathlib import Path
from typing import Any

import _evaluation_backend as base


EXPECTED_PROTOCOL = "houston-m032-common13-pino-depth4-plateau-v2"
EXPECTED_MODEL = "pino_resunet_large"
MODEL_KEY = "pino_resunet_large_depth4_plateau_v2"
MODEL_LABEL = "PINO-ResU-Net-large (depth4 plateau v2)"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_scientific_contract(
    *,
    common: Any,
    checkpoint: Path,
    payload: dict[str, Any],
    stage_summary: dict[str, Any],
    provenance: dict[str, Any],
    checkpoint_role: str,
) -> None:
    """Fail closed unless this is the converged AR32 validation-best model."""

    config = payload.get("config", {})
    base._require_equal(config.get("model"), EXPECTED_MODEL, "checkpoint model")
    base._require_equal(config.get("subset"), "m032", "checkpoint subset")
    base._require_equal(config.get("rollout_hours"), 32, "rollout horizon")
    base._require_equal(config.get("protocol_id"), EXPECTED_PROTOCOL, "protocol")
    base._require_equal(
        config.get("architecture_key"), "houston_resunet_large", "architecture"
    )
    base._require_equal(config.get("external_residual_update"), True, "residual update")
    base._require_equal(
        config.get("prediction_target"),
        "normalized conservative-state increment",
        "prediction target",
    )
    base._require_equal(list(config.get("input_features", ())), list(common.FEATURE_NAMES), "features")
    base._require_equal(list(config.get("auxiliary_losses", ())), [], "auxiliary losses")
    base._require_equal(float(config.get("depth_weight", math.nan)), 4.0, "depth weight")
    base._require_equal(float(config.get("momentum_weight", math.nan)), 1.0, "momentum weight")
    base._require_equal(float(config.get("supervised_scale", math.nan)), 3.0, "supervised scale")
    base._require_equal(config.get("learning_rate_schedule"), "reduce_on_plateau", "LR schedule")
    base._require_equal(int(config.get("epochs", -1)), 240, "stage safety ceiling")
    base._require_equal(int(config.get("early_stopping_patience", -1)), 8, "early-stop patience")
    base._require_equal(
        float(config.get("minimum_relative_improvement", math.nan)),
        1.0e-3,
        "minimum relative improvement",
    )

    base._require_equal(stage_summary.get("complete"), True, "stage completion")
    base._require_equal(stage_summary.get("stop_reason"), "validation_plateau", "stop reason")
    base._require_equal(stage_summary.get("model"), EXPECTED_MODEL, "stage model")
    base._require_equal(stage_summary.get("protocol_id"), EXPECTED_PROTOCOL, "stage protocol")
    epochs_completed = int(stage_summary.get("epochs_completed", -1))
    if not 12 <= epochs_completed <= 240:
        raise RuntimeError(f"completed epochs outside validated range: {epochs_completed}")
    if int(stage_summary.get("early_stopping_stale_epochs", -1)) < 8:
        raise RuntimeError("stage summary does not satisfy plateau patience")

    if checkpoint_role != "best":
        raise RuntimeError("depth4/plateau-v2 evaluation must select the validation-best checkpoint")
    base._require_equal(payload.get("epoch"), stage_summary.get("best_epoch"), "best epoch")
    base._require_equal(
        checkpoint.name, "checkpoint_best_validation_supervised.pt", "checkpoint filename"
    )
    base._require_equal(
        float(payload.get("best_validation_supervised", math.nan)),
        float(stage_summary.get("best_validation_supervised", math.nan)),
        "best validation score",
    )
    expected_hash = stage_summary.get("checkpoint_sha256", {}).get("best")
    base._require_equal(common.sha256_file(checkpoint), expected_hash, "best checkpoint hash")

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
        base._require_equal(
            common.sha256_file(path), identity["sha256"], f"provenance hash for {name}"
        )

    model_record = provenance.get("model", {})
    base._require_equal(model_record.get("treatment"), EXPECTED_MODEL, "provenance treatment")
    base._require_equal(model_record.get("strict_fp32"), True, "strict FP32")
    base._require_equal(model_record.get("tf32_enabled"), False, "TF32 setting")
    base._require_equal(
        model_record.get("prediction_target"), config["prediction_target"], "provenance target"
    )
    base._require_equal(model_record.get("external_residual_update"), True, "provenance residual update")
    base._require_equal(model_record.get("physics_regularizer"), True, "PINO physics regularizer")


def _write_json(payload: dict[str, Any], path: Path) -> None:
    """Replace inherited fixed-budget wording with the exact v2 identity."""

    if path.name == "export_metadata.json":
        payload.update(
            {
                "model_key": MODEL_KEY,
                "model": MODEL_LABEL,
                "protocol_id": EXPECTED_PROTOCOL,
                "checkpoint_role": "best",
                "campaign_primary_checkpoint": payload.get("checkpoint"),
                "campaign_primary_checkpoint_epoch": payload.get("checkpoint_epoch"),
                "campaign_primary_checkpoint_sha256": payload.get("checkpoint_sha256"),
                "campaign_primary_validation_supervised": payload.get(
                    "selected_validation_supervised"
                ),
                "state_loss_relative_component_weights": {"h": 4.0, "qx": 1.0, "qy": 1.0},
                "state_loss_expanded_formula": "2*MSE(h)+0.5*MSE(qx)+0.5*MSE(qy)",
                "wetness_or_inundation_loss": False,
                "checkpoint_selection_note": (
                    "The validation-best AR32 checkpoint is the declared evaluation checkpoint "
                    "for the converged depth4/plateau-v2 protocol."
                ),
            }
        )
        payload["evaluation_model_factory"] = payload.pop(
            "evaluation_model_loader", None
        )
        payload["evaluation_model_factory_sha256"] = payload.pop(
            "evaluation_model_loader_sha256", None
        )
        provenance_path = Path(payload["training_provenance"])
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        trainer_identity = provenance["files"]["trainer"]
        payload["training_trainer"] = trainer_identity["path"]
        payload["training_trainer_sha256"] = trainer_identity["sha256"]
        wrapper = Path(__file__).resolve()
        shared_adapter = Path(base.__file__).resolve()
        payload["evaluation_adapter"] = str(wrapper)
        payload["evaluation_adapter_sha256"] = _sha256(wrapper)
        payload["shared_rollout_adapter"] = str(shared_adapter)
        payload["shared_rollout_adapter_sha256"] = _sha256(shared_adapter)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def main() -> None:
    base.CLI_DESCRIPTION = __doc__
    base.EXPECTED_PROTOCOL = EXPECTED_PROTOCOL
    base.MODEL_KEY = MODEL_KEY
    base.MODEL_LABEL = MODEL_LABEL
    base._validate_scientific_contract = _validate_scientific_contract
    base._write_json = _write_json
    base.main()


if __name__ == "__main__":
    main()
