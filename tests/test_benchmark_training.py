"""Contract tests for the m032 multi-model depth4/plateau campaign."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "src" / "train_benchmark_model.py"
WORKFLOW = ROOT / "workflows" / "m032_benchmark_training"
MANIFEST = WORKFLOW / "campaign_manifest.json"
REGISTRY = WORKFLOW / "model_registry.json"
NEXT_STAGE = WORKFLOW / "next_stage.py"
CONNECTOR_SUPPORT = ROOT / "src" / "model_connector.py"
CONNECTOR_TEMPLATE = ROOT / "model" / "model_new0" / "connector.py"
MODEL_LIST_SUPPORT = ROOT / "src" / "model_list.py"
SYNC_EXPERIMENT = ROOT / "scripts" / "sync_experiment_manifest.py"

EXPECTED_MODELS = (
    "unet_v2",
    "fno_v2",
    "resunet_small",
    "pino_resunet_small",
    "resunet_large",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_trainer():
    return load_module("benchmark_trainer", TRAINER)


def protocol_args(trainer, model: str = "unet_v2"):
    args = trainer.parse_args(
        ["--subset", "m032", "--model", model, "--rollout-hours", "4"]
    )
    trainer.resolve_protocol_defaults(args)
    return args


def test_supervised_term_matches_pino_resunet_large_formula() -> None:
    trainer = load_trainer()
    prediction = torch.zeros(1, 3, 2, 2)
    target = torch.empty_like(prediction)
    target[:, 0] = 1.0
    target[:, 1] = 2.0
    target[:, 2] = 3.0
    domain = torch.ones(1, 1, 2, 2)
    values = trainer.supervised_delta_loss(
        prediction, target, domain, depth_weight=4.0, momentum_weight=1.0
    )
    reproduced = trainer.SUPERVISED_SCALE * values["supervised"]
    assert reproduced.item() == pytest.approx(2.0 * 1.0 + 0.5 * 4.0 + 0.5 * 9.0)


def test_protocol_locks_loss_plateau_handoff_and_exposure() -> None:
    trainer = load_trainer()
    protocol = trainer.protocol_spec("m032")
    assert protocol["protocol_id"] == "houston-m032-common13-depth4-plateau-v1"
    assert protocol["model_keys"] == list(EXPECTED_MODELS)
    assert protocol["supervised_loss"]["component_weights"] == [4.0, 1.0, 1.0]
    assert protocol["supervised_loss"]["supervised_scale"] == 3.0
    assert protocol["supervised_loss"]["literal_formula"] == "2*D + 0.5*Qx + 0.5*Qy"
    assert protocol["supervised_loss"]["inundation_loss"] is False
    assert protocol["auxiliary_losses"] == []
    assert protocol["epochs_by_stage"] == {"4": 240, "8": 240, "16": 240, "32": 240}
    assert protocol["forecast_transitions_per_epoch"] == 8192
    schedule = protocol["optimizer"]["schedule"]
    assert schedule == {
        "name": "ReduceLROnPlateau",
        "mode": "min",
        "factor": 0.5,
        "patience": 2,
        "threshold": 1e-4,
        "threshold_mode": "rel",
        "cooldown": 0,
        "minimum_learning_rate": 0.0,
        "epsilon": 1e-8,
        "restart_each_curriculum_stage": True,
    }
    assert protocol["stage_transition"].startswith("preceding validation-best")


def test_registry_is_the_exact_five_model_training_set() -> None:
    trainer = load_trainer()
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))["models"]
    assert tuple(entry["model"] for entry in registry) == EXPECTED_MODELS
    assert trainer.MODEL_KEYS == EXPECTED_MODELS
    assert trainer.PHYSICS_TREATMENTS == frozenset(("pino_resunet_small",))
    specs = {entry["model"]: entry for entry in registry}
    assert specs["resunet_large"]["architecture_key"] == "houston_resunet_large"


def test_baseline_sources_are_exact_vendored_training_definitions() -> None:
    assert sha256(ROOT / "src" / "vendor" / "baselines" / "unet_model.py") == (
        "b6761003de6a3efe447f4e9d60440ab49a85d9aec11d1d2421fb2118cdce8919"
    )
    assert sha256(ROOT / "src" / "vendor" / "baselines" / "fno_model.py") == (
        "139a964f43023c5f4b356b9978f50e4d6f0755ee6e7133ad2eafe5a4c108a686"
    )


def test_model_connector_has_the_common_shape_and_is_used_by_trainer() -> None:
    support = load_module("model_connector_support_test", CONNECTOR_SUPPORT)
    model_spec = {
        "model": "connector_test",
        "architecture_key": "connector_test",
        "physics_regularizer": False,
        "connector": "model/model_new0/connector.py",
        "settings": {"width": 7},
        "source_files": [],
    }
    model, groups, details = support.build_connected_model(
        model_spec, input_channels=13, output_channels=3
    )
    prediction = model(torch.zeros(2, 13, 16, 24))
    assert prediction.shape == (2, 3, 16, 24)
    assert torch.isfinite(prediction).all()
    assert groups is None
    assert details["connector_details"]["width"] == 7
    assert support.connector_source_files(model_spec) == [CONNECTOR_TEMPLATE]

    trainer = load_trainer()
    trainer.MODEL_SPECS["connector_test"] = dict(model_spec)
    trainer.ARCHITECTURE_KEYS["connector_test"] = "connector_test"
    try:
        built, built_groups, report = trainer.build_model(
            SimpleNamespace(model="connector_test"), torch.device("cpu")
        )
        assert built(torch.zeros(1, 13, 16, 24)).shape == (1, 3, 16, 24)
        assert built_groups is None
        assert report["connector"] == "model/model_new0/connector.py"
        assert trainer.model_source_files("connector_test") == [
            CONNECTOR_SUPPORT,
            CONNECTOR_TEMPLATE,
        ]
    finally:
        del trainer.MODEL_SPECS["connector_test"]
        del trainer.ARCHITECTURE_KEYS["connector_test"]


def test_model_connector_rejects_paths_outside_connector_directory() -> None:
    support = load_module("model_connector_path_test", CONNECTOR_SUPPORT)
    with pytest.raises(ValueError, match="inside model"):
        support.connector_path({"connector": "src/train_benchmark_model.py"})


@pytest.mark.parametrize(
    ("model_id", "parameters", "vendor_source"),
    (
        ("unet_v2", 489_475, "src/vendor/baselines/unet_model.py"),
        ("fno_v2", 613_851, "src/vendor/baselines/fno_model.py"),
        ("resunet_small", 535_825, "src/vendor/pino/model.py"),
        ("pino_resunet_small", 535_825, "src/vendor/pino/model.py"),
        ("resunet_large", 1_570_934, "src/vendor/pino/model.py"),
        ("pino_resunet_large", 1_570_934, "src/vendor/pino/model.py"),
    ),
)
def test_every_retained_model_folder_has_a_working_connector(
    model_id: str, parameters: int, vendor_source: str
) -> None:
    support = load_module(f"model_connector_{model_id}_test", CONNECTOR_SUPPORT)
    specification = {
        "connector": f"model/{model_id}/connector.py",
        "settings": {},
        "source_files": [],
    }
    model, groups, details = support.build_connected_model(
        specification, input_channels=13, output_channels=3
    )
    prediction = model(torch.zeros(1, 13, 32, 64))
    assert prediction.shape == (1, 3, 32, 64)
    assert torch.isfinite(prediction).all()
    assert groups is None
    assert sum(value.numel() for value in model.parameters()) == parameters
    sources = support.connector_source_files(specification)
    assert sources == [
        ROOT / f"model/{model_id}/connector.py",
        ROOT / vendor_source,
    ]
    assert "short_description" in details["connector_details"]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("model", "../escape", "must match"),
        ("physics_regularizer", "false", "must be true or false"),
        ("learning_rate", float("nan"), "finite and non-negative"),
        ("slice_epochs", 0, "positive integer"),
    ),
)
def test_model_list_rejects_unsafe_or_ambiguous_values(
    field: str, value, message: str
) -> None:
    support = load_module("model_list_validation_test", MODEL_LIST_SUPPORT)
    entry = {
        "model": "safe_model",
        "architecture_key": "safe_model",
        "physics_regularizer": False,
        "initialization_policy": "seeded_random_cold_start",
        "learning_rate": 1.0e-4,
        "endpoint_learning_rate": 0.0,
        "slice_epochs": 12,
        "description": "Test model.",
    }
    entry[field] = value
    with pytest.raises(ValueError, match=message):
        support.validate_model_list({"schema_version": 1, "models": [entry]})


def test_model_list_rejects_duplicate_ids() -> None:
    support = load_module("model_list_duplicate_test", MODEL_LIST_SUPPORT)
    entry = {
        "model": "duplicate",
        "architecture_key": "duplicate",
        "physics_regularizer": False,
        "initialization_policy": "seeded_random_cold_start",
        "learning_rate": 1.0e-4,
        "endpoint_learning_rate": 0.0,
        "slice_epochs": 12,
        "description": "Test model.",
    }
    with pytest.raises(ValueError, match="duplicate model id"):
        support.validate_model_list(
            {"schema_version": 1, "models": [entry, dict(entry)]}
        )


def test_registry_resolver_reports_model_count() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            str(WORKFLOW / "resolve_model.py"),
            "--field",
            "count",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == str(len(EXPECTED_MODELS))


def test_registry_resolver_reports_connector_status() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            str(WORKFLOW / "resolve_model.py"),
            "--task-index",
            "0",
            "--field",
            "has_connector",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == "false"


def test_experiment_manifest_is_synchronized_with_model_list() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            str(SYNC_EXPERIMENT),
            "--experiment-dir",
            str(WORKFLOW),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "model list and manifest agree (5 models)" in completed.stdout


def test_experiment_identity_is_checked_against_manifest() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            str(SYNC_EXPERIMENT),
            "--experiment-dir",
            str(WORKFLOW),
            "--experiment-id",
            "houston-common13-m032-depth4-plateau-v1",
            "--protocol-id",
            "houston-m032-common13-depth4-plateau-v1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "model list and manifest agree (5 models)" in completed.stdout


def test_dataset_prefers_portable_relative_trajectory_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trainer = load_trainer()
    split = tmp_path / "trajectory_split.csv"
    split.write_text(
        "split,scenario_id,timesteps,trajectory_file,trajectory_relpath\n"
        "train,example,8,/obsolete/absolute.h5,events/example.h5\n",
        encoding="utf-8",
    )
    portable_root = tmp_path / "dataset"
    monkeypatch.setenv("HOUSTON_DATASET_ROOT", str(portable_root))
    dataset = trainer.TrajectoryWindowDataset(split, "train", 4)
    assert dataset.trajectory_paths == [
        (portable_root / "events" / "example.h5").resolve()
    ]


def test_dataset_relocates_frozen_absolute_path_without_rewriting_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trainer = load_trainer()
    split = tmp_path / "trajectory_split.csv"
    split.write_text(
        "split,scenario_id,timesteps,trajectory_file\n"
        "train,example,8,/old/site/dataset/trajectories/event/member_001.h5\n",
        encoding="utf-8",
    )
    portable_root = tmp_path / "dataset"
    monkeypatch.setenv("HOUSTON_DATASET_ROOT", str(portable_root))
    dataset = trainer.TrajectoryWindowDataset(split, "train", 4)
    assert dataset.trajectory_paths == [
        (portable_root / "trajectories" / "event" / "member_001.h5").resolve()
    ]


def test_manifest_registry_and_pino_resunet_large_reference_are_hash_bound() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["model_keys"] == list(EXPECTED_MODELS)
    assert manifest["model_registry_sha256"] == sha256(REGISTRY)
    pino_large = manifest["pino_resunet_large_reference"]
    assert pino_large["model"] == "pino_resunet_large"
    assert pino_large["protocol_id"] == "houston-m032-common13-pino-depth4-plateau-v2"
    checkpoint = Path(pino_large["primary_checkpoint"])
    assert checkpoint.is_file()
    assert sha256(checkpoint) == pino_large["primary_checkpoint_sha256"]
    summary = checkpoint.parent / "stage_summary.json"
    assert sha256(summary) == pino_large["stage_anchors"]["32"]["stage_summary_sha256"]
    assert pino_large["best_epoch"] == 52


@pytest.mark.parametrize("model", EXPECTED_MODELS)
def test_every_model_passes_locked_argument_validation(model: str) -> None:
    trainer = load_trainer()
    args = protocol_args(trainer, model)
    trainer.validate_arguments(args)
    rates = trainer.TREATMENT_LEARNING_RATES[model]
    assert args.learning_rate == rates["learning_rate"]
    assert args.endpoint_learning_rate == rates["endpoint_learning_rate"]
    assert args.weight_decay == pytest.approx(1e-5)


def test_completed_v1_trainer_refuses_non_smoke_training() -> None:
    trainer = load_trainer()
    args = protocol_args(trainer)
    with pytest.raises(RuntimeError, match="completed, read-only experiment"):
        trainer.refuse_completed_training(args)


def test_missing_best_is_repaired_only_from_selected_last(tmp_path: Path) -> None:
    trainer = load_trainer()
    args = protocol_args(trainer)
    payload = {
        "epoch": 3,
        "best_epoch": 3,
        "best_validation_supervised": 0.75,
        "config": vars(args).copy(),
    }
    last = tmp_path / "checkpoint_last.pt"
    best = tmp_path / "checkpoint_best_validation_supervised.pt"
    trainer.atomic_checkpoint(last, payload)
    repaired = trainer.ensure_best_checkpoint_consistency(last, best, payload, args)
    assert repaired["epoch"] == 3
    assert sha256(best) == sha256(last)


def test_unrecoverable_stale_best_fails_closed(tmp_path: Path) -> None:
    trainer = load_trainer()
    args = protocol_args(trainer)
    payload = {
        "epoch": 4,
        "best_epoch": 3,
        "best_validation_supervised": 0.75,
        "config": vars(args).copy(),
    }
    last = tmp_path / "checkpoint_last.pt"
    best = tmp_path / "checkpoint_best_validation_supervised.pt"
    trainer.atomic_checkpoint(last, payload)
    with pytest.raises(RuntimeError, match="cannot be reconstructed"):
        trainer.ensure_best_checkpoint_consistency(last, best, payload, args)


def test_next_stage_accepts_only_hashed_converged_model_lineage(tmp_path: Path) -> None:
    trainer = load_trainer()
    module = load_module("m032_depth_next_stage", NEXT_STAGE)
    model = "unet_v2"
    stage = tmp_path / model / "stage_ar04"
    stage.mkdir(parents=True)
    config = {
        "protocol_id": "houston-m032-common13-depth4-plateau-v1",
        "subset": "m032",
        "model": model,
        "rollout_hours": 4,
    }
    best_payload = {
        "epoch": 4,
        "best_epoch": 4,
        "best_validation_supervised": 0.5,
        "config": config,
    }
    final_payload = {
        "epoch": 12,
        "best_epoch": 4,
        "best_validation_supervised": 0.5,
        "early_stopping_stale_epochs": 8,
        "config": config,
    }
    best = stage / "checkpoint_best_validation_supervised.pt"
    last = stage / "checkpoint_last.pt"
    final = stage / "checkpoint_final.pt"
    trainer.atomic_checkpoint(best, best_payload)
    trainer.atomic_checkpoint(last, final_payload)
    trainer.atomic_link(last, final)
    summary = {
        "protocol_id": config["protocol_id"],
        "subset": "m032",
        "model": model,
        "rollout_hours": 4,
        "complete": True,
        "stop_reason": "validation_plateau",
        "epochs_completed": 12,
        "best_epoch": 4,
        "initial_checkpoint": None,
        "initial_checkpoint_sha256": None,
        "checkpoint_sha256": {
            "best": sha256(best),
            "last": sha256(last),
            "final": sha256(final),
        },
    }
    (stage / "stage_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    assert module.next_stage(tmp_path, model) == "8"
    best.write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        module.next_stage(tmp_path, model)


def test_slurm_array_uses_registry_and_has_safe_restart_floor() -> None:
    training = (WORKFLOW / "train.sbatch").read_text(encoding="utf-8")
    preflight = (WORKFLOW / "preflight.sbatch").read_text(encoding="utf-8")
    submit = (WORKFLOW / "submit_campaign.sh").read_text(encoding="utf-8")
    runtime = (WORKFLOW / "runtime_env.sh").read_text(encoding="utf-8")
    assert "#SBATCH --array=0-4%4" in training
    assert "#SBATCH --array=0-4%4" in preflight
    assert "resolve_model.py" in training and "resolve_model.py" in preflight
    assert "MAX_RESTARTS=520" in training
    assert '--export="ALL,M032_DEPTH_ENV=$ENV_FILE"' in submit
    assert '--field count' in submit
    assert '--array="$ARRAY_SPEC"' in submit
    assert "task_index < MODEL_COUNT" in submit
    assert '--array-spec "$ARRAY_SPEC"' in submit
    assert "completed, read-only experiment" in submit
    assert "completed, read-only experiment" in runtime
    assert "must replace both completed v1 IDs" in runtime
    assert '--experiment-id "$AI_BENCHMARK_EXPERIMENT_ID"' in submit
    assert '--protocol-id "$AI_BENCHMARK_PROTOCOL_ID"' in submit
    assert "--field has_connector" in submit


def test_immutable_campaigns_are_not_imported_or_overwritten() -> None:
    source = TRAINER.read_text(encoding="utf-8")
    assert "train_pino_depth4_plateau_v2" not in source
    assert "CONTINUATION_PARENT" not in source
    assert "pino_resunet_large_reference" in source
