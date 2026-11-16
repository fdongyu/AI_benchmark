"""Integrity checks for the retained PINO-ResU-Net-large result."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from argparse import Namespace
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
TRAINING_SNAPSHOT = (
    ROOT / "provenance/training_sources/pino_depth4_plateau_v2_original"
)
EVALUATION_SNAPSHOT = (
    ROOT / "provenance/evaluation_sources/pino_depth4_plateau_v2_original"
)
RESULT = ROOT / "results/m032_benchmark/pino_resunet_large"
EXPORT = ROOT / "data/evaluations/pino_resunet_large_m032_ar32"
TRAINER_SHA = "0c1b01460c2cd0cf7d8a1a375eec5a14f64471254b8157448229cd362cc16ea9"
CAMPAIGN_SHA = "e079d0b62d9bc3e1b05d965c93a1cea2c827266a07ce4746ca041249556edd6c"
CHECKPOINT_SHA = "189f6e4a7b5a5493d82ba17f7f563d50d5890660f8c390c0ab74cc6ed59e90eb"
PROTOCOL = "houston-m032-common13-pino-depth4-plateau-v2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checksum_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        value, name = line.split(maxsplit=1)
        values[name.strip()] = value
    return values


def test_pino_resunet_large_sources_are_byte_exact_and_stage_bound() -> None:
    expected = {
        "trainer": (
            TRAINING_SNAPSHOT / "train_pino_depth4_plateau_v2.py",
            TRAINER_SHA,
        ),
        "campaign_manifest": (
            TRAINING_SNAPSHOT / "campaign_manifest.json",
            CAMPAIGN_SHA,
        ),
    }
    declared = checksum_file(TRAINING_SNAPSHOT / "SHA256SUMS")
    assert declared["train_pino_depth4_plateau_v2.py"] == TRAINER_SHA
    assert declared["campaign_manifest.json"] == CAMPAIGN_SHA
    for path, expected_hash in expected.values():
        assert path.is_file()
        assert sha256(path) == expected_hash

    stages = sorted(RESULT.glob("stage_ar*/provenance.json"))
    assert [path.parent.name for path in stages] == [
        "stage_ar04",
        "stage_ar08",
        "stage_ar16",
        "stage_ar32",
    ]
    for path in stages:
        provenance = json.loads(path.read_text(encoding="utf-8"))
        assert provenance["cleanup_migration"]["migration_id"] == (
            "pino-v2-coldstart-repair-2026-11-15"
        )
        assert provenance["protocol"]["protocol_id"] == PROTOCOL
        for key, (destination, expected_hash) in expected.items():
            identity = provenance["files"][key]
            assert Path(identity["path"]).resolve() == destination.resolve()
            assert identity["sha256"] == expected_hash
            assert sha256(Path(identity["path"])) == expected_hash


def test_pino_resunet_large_receipts_are_byte_exact() -> None:
    migration = json.loads(
        (TRAINING_SNAPSHOT / "receipt_cleanup_migration.json").read_text(
            encoding="utf-8"
        )
    )
    assert migration["migration_id"] == "pino-v2-coldstart-repair-2026-11-15"
    expected_receipts = {
        "provenance/submission_receipts/pino_depth4_plateau_v2/submission_60863469.json": (
            "512d9facebd1544aed87fa780900bd2e89fa449f8bf74fcb3e2beb0aa395e3e4"
        ),
        "provenance/submission_receipts/pino_depth4_plateau_v2/submission_60887630.json": (
            "89ecb508eba9a1538ae7d047c46465e4a03dac6ec2d027c9d5482f2aa3be0397"
        ),
    }
    for relative, expected_hash in expected_receipts.items():
        record = migration["historical_receipts"][relative]
        receipt_path = ROOT / relative
        assert record["payload_unchanged"] is True
        assert record["payload_sha256"] == expected_hash
        assert sha256(receipt_path) == expected_hash
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        assert receipt["source_sha256"]["src/train_pino_depth4_plateau_v2.py"] == TRAINER_SHA
        assert receipt["source_sha256"][
            "workflows/pino_depth4_plateau_v2/campaign_manifest.json"
        ] == CAMPAIGN_SHA

    relocations = migration["scientific_source_relocations"]
    assert relocations["src/train_pino_depth4_plateau_v2.py"] == {
        "sha256": TRAINER_SHA,
        "snapshot": (
            "provenance/training_sources/pino_depth4_plateau_v2_original/"
            "train_pino_depth4_plateau_v2.py"
        ),
    }
    assert relocations["workflows/pino_depth4_plateau_v2/campaign_manifest.json"] == {
        "sha256": CAMPAIGN_SHA,
        "snapshot": (
            "provenance/training_sources/pino_depth4_plateau_v2_original/"
            "campaign_manifest.json"
        ),
    }


def test_pino_resunet_large_evaluation_resolves_immutable_sources() -> None:
    common = EVALUATION_SNAPSHOT / "evaluate_common13_results.py"
    wrapper = EVALUATION_SNAPSHOT / "evaluate_pino_depth4_plateau_v2.py"
    common_sha = "fded3129d2b01ec0d7470429e72ab42ccd6f671ce1814b6af94352117df8fb21"
    wrapper_sha = "655b8f3bcfa8365f9b4732ff4fa1f3f75eb81aae2b3ceabbe9f1951c1f88c8ab"
    declared = checksum_file(EVALUATION_SNAPSHOT / "SHA256SUMS")
    assert declared[common.name] == common_sha == sha256(common)
    assert declared[wrapper.name] == wrapper_sha == sha256(wrapper)

    metadata = json.loads((EXPORT / "export_metadata.json").read_text(encoding="utf-8"))
    assert metadata["protocol_id"] == PROTOCOL
    assert metadata["complete_rollouts"] == 192
    assert metadata["checkpoint_role"] == "best"
    assert metadata["checkpoint_epoch"] == 52
    assert metadata["checkpoint_sha256"] == CHECKPOINT_SHA
    expected_sources = {
        "evaluation_adapter": (wrapper, wrapper_sha),
        "shared_rollout_adapter": (common, common_sha),
        "evaluation_model_factory": (
            TRAINING_SNAPSHOT / "train_pino_depth4_plateau_v2.py",
            TRAINER_SHA,
        ),
        "training_trainer": (
            TRAINING_SNAPSHOT / "train_pino_depth4_plateau_v2.py",
            TRAINER_SHA,
        ),
    }
    for field, (path, expected_hash) in expected_sources.items():
        assert Path(metadata[field]).resolve() == path.resolve()
        assert metadata[f"{field}_sha256"] == expected_hash
        assert sha256(path) == expected_hash


def test_pino_resunet_large_checkpoint_and_publication_record_agree() -> None:
    publication = json.loads((ROOT / "publication_manifest.json").read_text(encoding="utf-8"))
    assert publication["reference_model_id"] == "pino_resunet_large"
    model = next(item for item in publication["models"] if item["id"] == "pino_resunet_large")
    checkpoint = ROOT / model["artifacts"]["checkpoint"]["path"]
    assert model["artifacts"]["checkpoint"]["sha256"] == CHECKPOINT_SHA
    assert sha256(checkpoint) == CHECKPOINT_SHA

    summary = json.loads((checkpoint.parent / "stage_summary.json").read_text(encoding="utf-8"))
    assert summary["protocol_id"] == PROTOCOL
    assert summary["model"] == "pino_resunet_large"
    assert summary["rollout_hours"] == 32
    assert summary["best_epoch"] == 52
    assert summary["complete"] is True
    assert summary["stop_reason"] == "validation_plateau"
    assert f"{CHECKPOINT_SHA}  {checkpoint.relative_to(ROOT)}" in (
        ROOT / "ARTIFACTS.sha256"
    ).read_text(encoding="utf-8")


def test_pino_resunet_large_model_loader_uses_snapshot_and_stable_vendor() -> None:
    adapter_path = ROOT / "model/pino_resunet_large/load_historical_checkpoint.py"
    specification = importlib.util.spec_from_file_location(
        "pino_resunet_large_factory_test", adapter_path
    )
    assert specification is not None and specification.loader is not None
    factory = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(factory)
    assert factory.TRAINER_SNAPSHOT.resolve() == (
        TRAINING_SNAPSHOT / "train_pino_depth4_plateau_v2.py"
    ).resolve()
    assert factory.VENDOR_ROOT == (ROOT / "src/vendor").resolve()

    payload = torch.load(
        RESULT / "stage_ar32/checkpoint_best_validation_supervised.pt",
        map_location="cpu",
        weights_only=False,
    )
    model, groups, report = factory.build_model(
        Namespace(**payload["config"]), torch.device("cpu")
    )
    assert groups is None
    assert report["parameters"] == 1_570_934
    assert report["trainable_parameters"] == 1_570_875
    assert sum(parameter.numel() for parameter in model.parameters()) == 1_570_934


def test_evaluation_workflow_uses_model_folder_and_never_overwrites() -> None:
    workflow = (ROOT / "workflows/evaluate_pino_resunet_large.sbatch").read_text(
        encoding="utf-8"
    )
    assert "model/pino_resunet_large/evaluate.py" in workflow
    assert "model/pino_resunet_large/load_historical_checkpoint.py" in workflow
    assert "Refusing to overwrite completed export" in workflow
    assert "PINO_RESUNET_LARGE_EVAL_OUTPUT_DIR" in workflow
