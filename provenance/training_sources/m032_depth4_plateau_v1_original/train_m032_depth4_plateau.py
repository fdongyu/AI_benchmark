#!/usr/bin/env python3
"""Train the non-reference m032 emulators under the selected PINO protocol.

This is a separate, provenance-locked campaign.  Every treatment receives the
same common13 features, residual conservative-state target, 4:1:1 relative
h:qx:qy emphasis, literal supervised loss, transition sampling, validation
criterion, plateau scheduler, and validation-best curriculum handoff used by
the selected PINO.  Data-only/PINO pairs differ only by the declared SWE term.
Published Poseidon and MORPH weights remain model-specific hot starts.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import random
import shutil
import signal
import subprocess
import sys
import time
from collections import OrderedDict, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional, Sequence

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler


SOURCE_DIR = Path(__file__).resolve().parent
BENCHMARK_ROOT = SOURCE_DIR.parent
FOUNDATION_ROOT = Path(
    os.environ.get(
        "AI_FOUNDATION_ROOT",
        BENCHMARK_ROOT.parent / "AI_foundation",
    )
).resolve()
MODEL_ROOT = Path(
    os.environ.get(
        "RDYCORE_AI_ROOT",
        FOUNDATION_ROOT / "Perlmutter_transfer" / "code" / "AI_foundation",
    )
).resolve()
PINO_ROOT = (SOURCE_DIR / "vendor").resolve()
ADAPTER_ROOT = (SOURCE_DIR / "adapters").resolve()
DEFAULT_CAMPAIGN_MANIFEST = (
    BENCHMARK_ROOT
    / "workflows"
    / "m032_depth4_plateau_v1"
    / "campaign_manifest.json"
).resolve()
MODEL_REGISTRY_PATH = (
    BENCHMARK_ROOT
    / "workflows"
    / "m032_depth4_plateau_v1"
    / "model_registry.json"
).resolve()

RASTER_SHAPE = (256, 512)
if not MODEL_REGISTRY_PATH.is_file():
    raise FileNotFoundError(f"missing model registry: {MODEL_REGISTRY_PATH}")
with MODEL_REGISTRY_PATH.open("r", encoding="utf-8") as _registry_stream:
    _registry_payload = json.load(_registry_stream)
MODEL_REGISTRY = tuple(_registry_payload["models"])
MODEL_KEYS = tuple(str(entry["model"]) for entry in MODEL_REGISTRY)
if len(MODEL_KEYS) != len(set(MODEL_KEYS)):
    raise RuntimeError("model registry contains duplicate model keys")
MODEL_SPECS = {str(entry["model"]): dict(entry) for entry in MODEL_REGISTRY}
PHYSICS_TREATMENTS = frozenset(
    str(entry["model"]) for entry in MODEL_REGISTRY if bool(entry["physics_regularizer"])
)
STAGE_EPOCHS = {4: 240, 8: 240, 16: 240, 32: 240}
SUBSET_TRANSITIONS_PER_EPOCH = {"m032": 8192}
SUBSET_TRAIN_TRAJECTORIES = {"m032": 864}
SUBSET_MEMBERS_PER_EVENT = {"m032": 32}
VALIDATION_TRANSITIONS_PER_EPOCH = 2048
TREATMENT_LEARNING_RATES = {
    str(entry["model"]): {
        "learning_rate": float(entry["learning_rate"]),
        "endpoint_learning_rate": float(entry["endpoint_learning_rate"]),
    }
    for entry in MODEL_REGISTRY
}
COMMON_WEIGHT_DECAY = 1.0e-5
COMMON_GRADIENT_CLIP = 1.0
COMMON_DETACH_INTERVAL = 4
SUPERVISED_SCALE = 3.0
EARLY_STOPPING_MINIMUM_EPOCHS = 12
EARLY_STOPPING_PATIENCE = 8
MINIMUM_RELATIVE_IMPROVEMENT = 1.0e-3
PLATEAU_FACTOR = 0.5
PLATEAU_PATIENCE = 2
PLATEAU_THRESHOLD = 1.0e-4
PLATEAU_THRESHOLD_MODE = "rel"
PLATEAU_COOLDOWN = 0
PLATEAU_MIN_LR = 0.0
PLATEAU_EPS = 1.0e-8
SWE_CONTRACT = {
    "mass_weight": 1.0e-6,
    "momentum_x_weight": 1.0e-6,
    "momentum_y_weight": 1.0e-6,
    "mass_scale_m_s": 1.0e-5,
    "momentum_x_scale_m2_s2": 1.0e-3,
    "momentum_y_scale_m2_s2": 1.0e-3,
    "warmup_epochs": 2,
    "dx_m": 147.302734375,
    "dy_m": 144.244140625,
    "dt_s": 3600.0,
    "y_axis_sign": -1.0,
    "gravity_m_s2": 9.81,
    "manning_n": 0.015,
}
ARCHITECTURE_KEYS = {
    str(entry["model"]): str(entry["architecture_key"])
    for entry in MODEL_REGISTRY
}
FEATURE_NAMES = (
    "height_t",
    "momentum_x_t",
    "momentum_y_t",
    "log1p_rainfall_t",
    "ocean_height_t",
    "bed_elevation",
    "domain_mask",
    "ocean_boundary_mask",
    "x_coordinate",
    "y_coordinate",
    "stream_core_mask",
    "channel_corridor_mask",
    "stream_proximity",
)

# These names are an executable declaration that no hidden task-specific loss
# is part of this protocol.  Keep the tuple empty.
AUXILIARY_LOSSES: tuple[str, ...] = ()
LOSS_COMPONENTS = (
    "supervised",
    "depth_mse",
    "momentum_x_mse",
    "momentum_y_mse",
    "physics",
    "physics_mass",
    "physics_momentum_x",
    "physics_momentum_y",
    "objective",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def import_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path: Path, *, hash_contents: bool = True) -> dict[str, Any]:
    path = path.resolve()
    stat = path.stat()
    result: dict[str, Any] = {
        "path": str(path),
        "bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if hash_contents:
        result["sha256"] = sha256_file(path)
    return result


def git_identity(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"root": str(path.resolve())}
    try:
        completed = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        result["commit"] = completed.stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(path), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        )
        result["dirty"] = bool(dirty.stdout.strip())
    except (OSError, subprocess.CalledProcessError) as error:
        result["unavailable"] = str(error)
    return result


def protocol_spec(subset: str) -> dict[str, Any]:
    """Return the immutable multi-model depth-emphasis protocol contract."""

    if subset not in SUBSET_TRANSITIONS_PER_EPOCH:
        raise ValueError(f"unsupported subset: {subset}")
    transitions_per_epoch = SUBSET_TRANSITIONS_PER_EPOCH[subset]
    return {
        "protocol_id": "houston-m032-common13-depth4-plateau-v1",
        "subset": subset,
        "model_keys": list(MODEL_KEYS),
        "model_registry": [dict(entry) for entry in MODEL_REGISTRY],
        "architecture_by_treatment": dict(ARCHITECTURE_KEYS),
        "curriculum_hours": [4, 8, 16, 32],
        "epochs_by_stage": {str(key): value for key, value in STAGE_EPOCHS.items()},
        "forecast_transitions_per_epoch": transitions_per_epoch,
        "validation_transitions_per_epoch": VALIDATION_TRANSITIONS_PER_EPOCH,
        "maximum_forecast_transitions_per_model": (
            sum(STAGE_EPOCHS.values()) * transitions_per_epoch
        ),
        "exposure_policy": (
            "8,192 training forecast transitions per epoch, maximum 240 epochs "
            "per stage, stopped after validation-plateau convergence"
        ),
        "training_trajectories": SUBSET_TRAIN_TRAJECTORIES[subset],
        "training_members_per_event": SUBSET_MEMBERS_PER_EVENT[subset],
        "seed": 20260917,
        "batch_size": 1,
        "truncated_bptt_detach_interval": COMMON_DETACH_INTERVAL,
        "teacher_forcing": False,
        "reproducibility": {
            "strict_fp32": True,
            "tf32": False,
            "cudnn_deterministic": True,
            "torch_deterministic_algorithms": False,
        },
        "prediction_parameterization": "normalized conservative-state increment",
        "state_update": "U_next = wet_dry_project(U_current + denormalize_delta(model(x)))",
        "normalization_note": (
            "subset-specific train-only mean+std delta normalization is retained; "
            "raw network output zero denormalizes to the subset's training mean "
            "increment, not exact persistence"
        ),
        "input_features": list(FEATURE_NAMES),
        "input_feature_count": len(FEATURE_NAMES),
        "sampler": {
            "training": "exact hierarchical event/member/window balance",
            "validation": "fixed seeded exact hierarchical event/member/window balance",
            "replacement": True,
        },
        "projector": {
            "dry_tolerance_m": 1.0e-4,
            "zero_dry_momentum": True,
            "max_velocity_m_s": 10.0,
            "training_gradient": "straight-through identity",
        },
        "supervised_loss": {
            "type": "domain-masked normalized-delta MSE",
            "component_weights": [4.0, 1.0, 1.0],
            "depth_weight": 4.0,
            "momentum_weight": 1.0,
            "supervised_scale": SUPERVISED_SCALE,
            "literal_formula": "2*D + 0.5*Qx + 0.5*Qy",
            "inundation_loss": False,
            "selection_metric": "validation supervised MSE only",
        },
        "auxiliary_losses": list(AUXILIARY_LOSSES),
        "physics_treatments": sorted(PHYSICS_TREATMENTS),
        "swe_regularizer": {
            "enabled_treatments": sorted(PHYSICS_TREATMENTS),
            **SWE_CONTRACT,
            "positivity_weight": 0.0,
            "dry_momentum_weight": 0.0,
            "wet_only_momentum": True,
            "project_states_inside_residual": False,
            "open_boundary_exclusion_cells": 1,
            "momentum_shoreline_exclusion_cells": 1,
        },
        "optimizer": {
            "name": "AdamW",
            "weight_decay": COMMON_WEIGHT_DECAY,
            "gradient_clip_norm": COMMON_GRADIENT_CLIP,
            "learning_rates_by_treatment": {
                key: dict(value) for key, value in TREATMENT_LEARNING_RATES.items()
            },
            "schedule": {
                "name": "ReduceLROnPlateau",
                "mode": "min",
                "factor": PLATEAU_FACTOR,
                "patience": PLATEAU_PATIENCE,
                "threshold": PLATEAU_THRESHOLD,
                "threshold_mode": PLATEAU_THRESHOLD_MODE,
                "cooldown": PLATEAU_COOLDOWN,
                "minimum_learning_rate": PLATEAU_MIN_LR,
                "epsilon": PLATEAU_EPS,
                "restart_each_curriculum_stage": True,
            },
        },
        "architecture_contract": {
            "unet_v2_base_channels": 16,
            "fno_v2": {
                "hidden_channels": 24,
                "modes_height": 12,
                "modes_width": 20,
                "layers": 4,
                "lifting_channels": 48,
                "projection_channels": 48,
                "domain_padding": 0.0625,
            },
            "houston_resunet_small": {
                "base_channels": 14,
                "depth": 4,
                "dropout": 0.0,
                "adapter_width": 0,
                "parameters": 535825,
            },
            "houston_resunet_large": {
                "base_channels": 24,
                "depth": 4,
                "dropout": 0.0,
                "adapter_width": 8,
                "adapter_training": "frozen neutral adapter, matching old m032 PINO",
                "parameters": 1570934,
                "trainable_parameters": 1570875,
                "old_m032_difference": (
                    "+480 input-endpoint weights required by 13 rather than 11 channels"
                ),
            },
            "poseidon_refiner_width": 24,
            "poseidon_initial_lead_time": 0.05,
            "poseidon_initialization": (
                "published pretrained backbone with fresh 13-channel task endpoints"
            ),
            "poseidon_b_learning_rate_note": (
                "hot-start stability rate 5e-6 backbone / 5e-5 endpoints, retained "
                "from the documented m032 numerical-stability correction"
            ),
            "morph_tuning_mode": "full",
        },
        "convergence": {
            "maximum_epochs": 240,
            "minimum_epochs": EARLY_STOPPING_MINIMUM_EPOCHS,
            "patience": EARLY_STOPPING_PATIENCE,
            "minimum_relative_improvement": MINIMUM_RELATIVE_IMPROVEMENT,
        },
        "stage_transition": "preceding validation-best checkpoint; fresh optimizer",
        "primary_evaluation": "validation-best converged AR32 checkpoint",
        "secondary_checkpoint": "last checkpoint at convergence",
        "numerics": {
            "seeded": True,
            "float_precision": "float32",
            "automatic_mixed_precision": False,
            "cuda_matmul_tf32": False,
            "cudnn_tf32": False,
            "nvidia_tf32_override": 0,
            "deterministic_algorithms": False,
            "determinism_exception": (
                "CUDA bilinear-interpolation backward lacks a supported deterministic kernel"
            ),
        },
        "architecture_internal_position_information": (
            "allowed when native to the architecture; the external 13-channel tensor is identical"
        ),
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    default_data = FOUNDATION_ROOT / "DATA" / "rdycore_ml_dataset_core"
    default_pretrained = FOUNDATION_ROOT / "pretrained"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", required=True, choices=tuple(SUBSET_TRANSITIONS_PER_EPOCH))
    parser.add_argument("--model", required=True, choices=MODEL_KEYS)
    parser.add_argument("--rollout-hours", required=True, type=int, choices=(4, 8, 16, 32))
    parser.add_argument("--dataset-root", type=Path, default=default_data)
    parser.add_argument("--shared-artifact-dir", type=Path, default=None)
    parser.add_argument(
        "--channel-artifact-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--output-root", type=Path, default=None
    )
    parser.add_argument(
        "--poseidon-t-pretrained", type=Path, default=default_pretrained / "Poseidon-T"
    )
    parser.add_argument(
        "--poseidon-b-pretrained", type=Path, default=default_pretrained / "Poseidon-B"
    )
    parser.add_argument(
        "--morph-pretrained",
        type=Path,
        default=default_pretrained / "morph-Ti-FM-max_ar1_ep225.pth",
    )
    parser.add_argument(
        "--campaign-manifest",
        type=Path,
        default=DEFAULT_CAMPAIGN_MANIFEST,
        help="Immutable campaign contract included in every stage provenance record.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=0,
        help="Must equal the stage budget (0 selects the locked stage default).",
    )
    parser.add_argument("--epochs-per-invocation", type=int, default=0)
    parser.add_argument("--transitions-per-epoch", type=int, default=0)
    parser.add_argument(
        "--validation-transitions",
        type=int,
        default=VALIDATION_TRANSITIONS_PER_EPOCH,
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--detach-interval", type=int, default=COMMON_DETACH_INTERVAL)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--mixed-precision", choices=("off", "bfloat16"), default="off")
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--learning-rate", type=float, default=0.0)
    parser.add_argument("--endpoint-learning-rate", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=-1.0)
    parser.add_argument("--gradient-clip", type=float, default=COMMON_GRADIENT_CLIP)
    parser.add_argument("--depth-weight", type=float, default=4.0)
    parser.add_argument("--momentum-weight", type=float, default=1.0)
    parser.add_argument("--supervised-scale", type=float, default=SUPERVISED_SCALE)
    parser.add_argument("--dry-tolerance-m", type=float, default=1.0e-4)
    parser.add_argument("--max-velocity-m-s", type=float, default=10.0)
    parser.add_argument("--physics-mass-weight", type=float, default=SWE_CONTRACT["mass_weight"])
    parser.add_argument(
        "--physics-momentum-weight",
        type=float,
        default=SWE_CONTRACT["momentum_x_weight"],
    )
    parser.add_argument("--physics-mass-scale", type=float, default=SWE_CONTRACT["mass_scale_m_s"])
    parser.add_argument(
        "--physics-momentum-scale",
        type=float,
        default=SWE_CONTRACT["momentum_x_scale_m2_s2"],
    )
    parser.add_argument("--physics-warmup-epochs", type=int, default=SWE_CONTRACT["warmup_epochs"])
    parser.add_argument("--dx-m", type=float, default=SWE_CONTRACT["dx_m"])
    parser.add_argument("--dy-m", type=float, default=SWE_CONTRACT["dy_m"])
    parser.add_argument("--dt-s", type=float, default=SWE_CONTRACT["dt_s"])
    parser.add_argument(
        "--y-axis-sign", type=float, choices=(-1.0, 1.0), default=SWE_CONTRACT["y_axis_sign"]
    )
    parser.add_argument("--gravity", type=float, default=SWE_CONTRACT["gravity_m_s2"])
    parser.add_argument("--manning-n", type=float, default=SWE_CONTRACT["manning_n"])
    parser.add_argument("--invocation-id", default="")
    parser.add_argument("--skip-checkpoints", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run one synthetic CPU optimization step through the common protocol.",
    )
    return parser.parse_args(argv)


class TrajectoryWindowDataset(Dataset):
    """Full-resolution autoregressive windows indexed by trajectory and origin."""

    def __init__(self, split_file: Path, split: str, horizon: int):
        table = pd.read_csv(split_file)
        self.table = table.loc[table["split"] == split].reset_index(drop=True)
        if self.table.empty:
            raise RuntimeError(f"no trajectories for split={split}")
        self.horizon = int(horizon)
        self.samples: list[tuple[int, int]] = []
        self.row_sample_indices: list[np.ndarray] = []
        for row_index, row in self.table.iterrows():
            start_index = len(self.samples)
            count = int(row["timesteps"]) - self.horizon
            if count < 1:
                raise RuntimeError(
                    f"trajectory {row['scenario_id']} is shorter than AR{self.horizon}"
                )
            self.samples.extend((row_index, start) for start in range(count))
            self.row_sample_indices.append(
                np.arange(start_index, start_index + count, dtype=np.int64)
            )
        self._files: OrderedDict[str, h5py.File] = OrderedDict()

    def __len__(self) -> int:
        return len(self.samples)

    def _file(self, path: str) -> h5py.File:
        handle = self._files.pop(path, None)
        if handle is None:
            handle = h5py.File(path, "r", locking=False)
        self._files[path] = handle
        while len(self._files) > 6:
            _, old = self._files.popitem(last=False)
            old.close()
        return handle

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row_index, start = self.samples[index]
        row = self.table.iloc[row_index]
        h5 = self._file(str(row["trajectory_file"]))
        stop = start + self.horizon
        return {
            "current": torch.from_numpy(np.asarray(h5["state"][start], dtype=np.float32)),
            "following": torch.from_numpy(
                np.asarray(h5["state"][start + 1 : stop + 1], dtype=np.float32)
            ),
            "rainfall": torch.from_numpy(
                np.asarray(h5["rainfall"][start:stop], dtype=np.float32)
            ),
            "ocean": torch.from_numpy(
                np.asarray(h5["ocean_boundary_height"][start:stop], dtype=np.float32)
            ),
        }

    def close(self) -> None:
        for handle in self._files.values():
            handle.close()
        self._files.clear()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def _balanced_counts(total: int, groups: int, generator: torch.Generator) -> list[int]:
    if total < 0 or groups < 1:
        raise ValueError("invalid balanced-count request")
    base, remainder = divmod(total, groups)
    result = [base] * groups
    if remainder:
        order = torch.randperm(groups, generator=generator).tolist()
        for index in order[:remainder]:
            result[index] += 1
    return result


class EventMemberWindowSampler(Sampler[int]):
    """Exactly balance events, then members, then sample windows uniformly."""

    def __init__(
        self,
        dataset: TrajectoryWindowDataset,
        num_samples: int,
        seed: int,
        *,
        fixed: bool = False,
    ) -> None:
        if num_samples < 1:
            raise ValueError("num_samples must be positive")
        self.dataset = dataset
        self.num_samples = int(num_samples)
        self.generator = torch.Generator().manual_seed(int(seed))
        grouped: dict[str, list[int]] = defaultdict(list)
        for row_index, row in dataset.table.iterrows():
            grouped[str(row["event_id"])].append(int(row_index))
        self.events = tuple(sorted(grouped))
        self.event_rows = tuple(
            tuple(
                sorted(
                    grouped[event],
                    key=lambda index: int(dataset.table.iloc[index]["member_id"]),
                )
            )
            for event in self.events
        )
        self.fixed = bool(fixed)
        # Materialize validation indices immediately.  Reconstructing a
        # sampler after a sliced-job restart then regenerates the identical
        # held-out windows from the declared seed, independent of RNG restore.
        self._fixed_indices: Optional[tuple[int, ...]] = (
            self._draw() if self.fixed else None
        )

    def __len__(self) -> int:
        return self.num_samples

    def _draw(self) -> tuple[int, ...]:
        selected: list[int] = []
        event_counts = _balanced_counts(
            self.num_samples, len(self.events), self.generator
        )
        for rows, event_count in zip(self.event_rows, event_counts):
            member_counts = _balanced_counts(event_count, len(rows), self.generator)
            for row_index, member_count in zip(rows, member_counts):
                if member_count == 0:
                    continue
                candidates = self.dataset.row_sample_indices[row_index]
                positions = torch.randint(
                    len(candidates),
                    (member_count,),
                    generator=self.generator,
                ).tolist()
                selected.extend(int(candidates[position]) for position in positions)
        order = torch.randperm(len(selected), generator=self.generator).tolist()
        return tuple(selected[index] for index in order)

    def __iter__(self) -> Iterator[int]:
        if self.fixed and self._fixed_indices is None:
            self._fixed_indices = self._draw()
        return iter(self._fixed_indices if self.fixed else self._draw())


def _worker_seed(_: int) -> None:
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def build_loaders(
    args: argparse.Namespace, split_file: Path
) -> tuple[TrajectoryWindowDataset, TrajectoryWindowDataset, DataLoader, DataLoader]:
    train_samples = args.transitions_per_epoch // args.rollout_hours
    validation_samples = args.validation_transitions // args.rollout_hours
    train = TrajectoryWindowDataset(split_file, "train", args.rollout_hours)
    validation = TrajectoryWindowDataset(split_file, "validation", args.rollout_hours)
    train_sampler = EventMemberWindowSampler(
        train, train_samples, args.seed + args.rollout_hours
    )
    validation_sampler = EventMemberWindowSampler(
        validation,
        validation_samples,
        args.seed + args.rollout_hours + 10_000,
        fixed=True,
    )
    common = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": bool(args.pin_memory),
        "persistent_workers": args.num_workers > 0,
        "worker_init_fn": _worker_seed,
    }
    train_loader = DataLoader(
        train,
        sampler=train_sampler,
        generator=torch.Generator().manual_seed(args.seed + 1_000_003),
        **common,
    )
    validation_loader = DataLoader(
        validation,
        sampler=validation_sampler,
        generator=torch.Generator().manual_seed(args.seed + 2_000_003),
        **common,
    )
    return train, validation, train_loader, validation_loader


def validate_dataset_contract(
    subset: str,
    train: TrajectoryWindowDataset,
    validation: TrajectoryWindowDataset,
) -> None:
    """Fail closed if the requested subset is paired with the wrong split file."""

    expected_train = SUBSET_TRAIN_TRAJECTORIES[subset]
    expected_members = SUBSET_MEMBERS_PER_EVENT[subset]
    if len(train.table) != expected_train:
        raise RuntimeError(
            f"{subset} requires {expected_train} training trajectories; "
            f"found {len(train.table)}"
        )
    train_counts = train.table.groupby("event_id")["member_id"].nunique()
    if len(train_counts) != 27 or not train_counts.eq(expected_members).all():
        raise RuntimeError(
            f"{subset} training membership contract failed: "
            f"events={len(train_counts)}, counts={sorted(train_counts.unique())}"
        )
    validation_counts = validation.table.groupby("event_id")["member_id"].nunique()
    if (
        len(validation.table) != 192
        or len(validation_counts) != 6
        or not validation_counts.eq(32).all()
    ):
        raise RuntimeError("validation population must be exactly 6 events x 32 members")


def load_static(
    static_path: Path, masks_path: Path, device: torch.device
) -> dict[str, torch.Tensor]:
    with h5py.File(static_path, "r") as h5:
        arrays = {
            "bed": np.asarray(h5["bed_elevation"], dtype=np.float32),
            "domain": np.asarray(h5["domain_mask"], dtype=np.float32),
            "ocean_mask": np.asarray(h5["ocean_boundary_mask"], dtype=np.float32),
        }
    with np.load(masks_path) as masks:
        arrays["stream_core"] = masks["stream_core_mask"].astype(np.float32)
        arrays["channel"] = masks["channel_corridor_mask"].astype(np.float32)
        distance = masks["stream_distance_m"].astype(np.float32)
        arrays["stream_proximity"] = np.exp(-distance / 1000.0).astype(np.float32)
    y_coordinate = np.linspace(1.0, -1.0, RASTER_SHAPE[0], dtype=np.float32)
    x_coordinate = np.linspace(-1.0, 1.0, RASTER_SHAPE[1], dtype=np.float32)
    arrays["x_coordinate"], arrays["y_coordinate"] = np.meshgrid(
        x_coordinate, y_coordinate
    )
    for name, array in arrays.items():
        if array.shape != RASTER_SHAPE:
            raise RuntimeError(f"{name} has shape {array.shape}; expected {RASTER_SHAPE}")
    result = {
        name: torch.from_numpy(array)[None, None].to(device)
        for name, array in arrays.items()
    }
    result["stream_proximity"] *= result["domain"]
    result["x_coordinate"] *= result["domain"]
    result["y_coordinate"] *= result["domain"]
    return result


def norm_vector(
    normalization: Mapping[str, Any], key: str, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    values = normalization[key]
    mean = torch.tensor(values["mean"], dtype=torch.float32, device=device)[
        None, :, None, None
    ]
    std = torch.tensor(values["std"], dtype=torch.float32, device=device)[
        None, :, None, None
    ]
    return mean, std


def make_features(
    current: torch.Tensor,
    rainfall: torch.Tensor,
    ocean: torch.Tensor,
    static: Mapping[str, torch.Tensor],
    normalization: Mapping[str, Any],
) -> torch.Tensor:
    """Construct the same ordered 13-channel tensor for every treatment."""

    domain = static["domain"]
    batch = current.shape[0]
    state_mean, state_std = norm_vector(normalization, "state", current.device)
    state = ((current - state_mean) / state_std) * domain
    rain_mean = float(normalization["log1p_rainfall"]["mean"][0])
    rain_std = float(normalization["log1p_rainfall"]["std"][0])
    rain = (
        (torch.log1p(torch.clamp_min(rainfall, 0.0)) - rain_mean) / rain_std
    ) * domain
    ocean_mean = float(normalization["ocean_height"]["mean"][0])
    ocean_std = float(normalization["ocean_height"]["std"][0])
    ocean_field = (ocean[:, None, None, None] - ocean_mean) / ocean_std
    bed_mean = float(normalization["bed_elevation"]["mean"][0])
    bed_std = float(normalization["bed_elevation"]["std"][0])
    bed = ((static["bed"] - bed_mean) / bed_std) * domain
    return torch.cat(
        (
            state,
            rain,
            ocean_field.expand(batch, -1, *current.shape[-2:]) * domain,
            bed.expand(batch, -1, -1, -1),
            domain.expand(batch, -1, -1, -1),
            static["ocean_mask"].expand(batch, -1, -1, -1),
            static["x_coordinate"].expand(batch, -1, -1, -1),
            static["y_coordinate"].expand(batch, -1, -1, -1),
            static["stream_core"].expand(batch, -1, -1, -1),
            static["channel"].expand(batch, -1, -1, -1),
            static["stream_proximity"].expand(batch, -1, -1, -1),
        ),
        dim=1,
    )


def normalize_delta(
    delta: torch.Tensor, normalization: Mapping[str, Any]
) -> torch.Tensor:
    mean, std = norm_vector(normalization, "state_delta", delta.device)
    return (delta - mean) / std


def denormalize_delta(
    normalized: torch.Tensor, normalization: Mapping[str, Any]
) -> torch.Tensor:
    mean, std = norm_vector(normalization, "state_delta", normalized.device)
    return normalized * std + mean


def project_wet_dry(
    state: torch.Tensor,
    domain_mask: torch.Tensor,
    *,
    dry_tolerance: float = 1.0e-4,
    max_velocity: float = 10.0,
) -> torch.Tensor:
    """Hard nonnegative/dry-momentum/velocity projection in conservative form."""

    domain = domain_mask > 0.5
    h = torch.clamp_min(state[:, :1], 0.0)
    wet = (h > dry_tolerance) & domain
    h = torch.where(wet, h, torch.zeros_like(h))
    momentum = torch.where(wet.expand(-1, 2, -1, -1), state[:, 1:], 0.0)
    speed = torch.linalg.vector_norm(momentum, dim=1, keepdim=True)
    limit = h * float(max_velocity)
    scale = torch.clamp(
        limit / speed.clamp_min(torch.finfo(state.dtype).eps), max=1.0
    )
    momentum = momentum * scale
    return torch.cat((h, momentum), dim=1) * domain.to(state.dtype)


def straight_through_project(
    state: torch.Tensor,
    domain_mask: torch.Tensor,
    *,
    dry_tolerance: float = 1.0e-4,
    max_velocity: float = 10.0,
) -> torch.Tensor:
    projected = project_wet_dry(
        state,
        domain_mask,
        dry_tolerance=dry_tolerance,
        max_velocity=max_velocity,
    )
    return state + (projected - state).detach()


def residual_state_update(
    current: torch.Tensor,
    normalized_delta: torch.Tensor,
    normalization: Mapping[str, Any],
    domain_mask: torch.Tensor,
    *,
    dry_tolerance: float,
    max_velocity: float,
    straight_through: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    proposal = current + denormalize_delta(normalized_delta, normalization)
    projector = straight_through_project if straight_through else project_wet_dry
    following = projector(
        proposal,
        domain_mask,
        dry_tolerance=dry_tolerance,
        max_velocity=max_velocity,
    )
    return following, normalize_delta(following - current, normalization)


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.shape[0] == 1 and values.shape[0] != 1:
        mask = mask.expand(values.shape[0], -1, -1, -1)
    if mask.shape[1] == 1 and values.shape[1] != 1:
        mask = mask.expand(-1, values.shape[1], -1, -1)
    weights = mask.to(values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def supervised_delta_loss(
    predicted_delta: torch.Tensor,
    target_delta: torch.Tensor,
    domain_mask: torch.Tensor,
    *,
    depth_weight: float,
    momentum_weight: float,
) -> dict[str, torch.Tensor]:
    squared = (predicted_delta - target_delta).square()
    depth = masked_mean(squared[:, :1], domain_mask)
    momentum_x = masked_mean(squared[:, 1:2], domain_mask)
    momentum_y = masked_mean(squared[:, 2:3], domain_mask)
    # Normalize by the sum of per-component weights.  The default therefore is
    # ordinary MSE across h, qx, and qy, with no privileged state component.
    denominator = float(depth_weight + 2.0 * momentum_weight)
    if denominator <= 0.0:
        raise ValueError("at least one supervised component weight must be positive")
    supervised = (
        depth_weight * depth + momentum_weight * (momentum_x + momentum_y)
    ) / denominator
    return {
        "depth_mse": depth,
        "momentum_x_mse": momentum_x,
        "momentum_y_mse": momentum_y,
        "supervised": supervised,
    }


def physics_evaluation_mask(
    domain: torch.Tensor, ocean_mask: torch.Tensor
) -> torch.Tensor:
    neighborhood = F.max_pool2d(ocean_mask, kernel_size=3, stride=1, padding=1)
    return domain * (neighborhood < 0.5).to(domain.dtype)


def fully_wet_stencil_mask(
    state_t: torch.Tensor,
    state_tp1: torch.Tensor,
    domain: torch.Tensor,
    dry_tolerance: float,
) -> torch.Tensor:
    midpoint = 0.5 * (state_t[:, :1] + state_tp1[:, :1])
    wet = (midpoint > dry_tolerance) & (domain > 0.5)
    touches_dry = F.max_pool2d(
        (~wet).to(midpoint.dtype), kernel_size=3, stride=1, padding=1
    )
    return (touches_dry < 0.5).to(midpoint.dtype)


_PHYSICS_MODULE = None


def pino_physics_module():
    global _PHYSICS_MODULE
    if _PHYSICS_MODULE is None:
        _PHYSICS_MODULE = import_file(
            "m008_fair_residual_pino_physics", PINO_ROOT / "pino" / "physics.py"
        )
    return _PHYSICS_MODULE


def step_objective(
    args: argparse.Namespace,
    current: torch.Tensor,
    predicted: torch.Tensor,
    projected_delta: torch.Tensor,
    target: torch.Tensor,
    rainfall: torch.Tensor,
    static: Mapping[str, torch.Tensor],
    normalization: Mapping[str, Any],
    epoch: int,
) -> dict[str, torch.Tensor]:
    target_delta = normalize_delta(target - current, normalization)
    values = supervised_delta_loss(
        projected_delta,
        target_delta,
        static["domain"],
        depth_weight=args.depth_weight,
        momentum_weight=args.momentum_weight,
    )
    # supervised_delta_loss normalizes by 4+1+1.  Multiplying by three gives
    # exactly 2*D + 0.5*Qx + 0.5*Qy, the previous PINO's state-data term.  The
    # earlier 0.1 inundation BCE is intentionally omitted by user request.
    values["supervised"] = args.supervised_scale * values["supervised"]
    zero = values["supervised"].new_zeros(())
    values.update(
        {
            "physics": zero,
            "physics_mass": zero,
            "physics_momentum_x": zero,
            "physics_momentum_y": zero,
        }
    )
    if args.model in PHYSICS_TREATMENTS:
        physics = pino_physics_module().physics_loss(
            current.float(),
            predicted.float(),
            rainfall.float(),
            static["bed"].float(),
            torch.full_like(static["domain"], args.manning_n).float(),
            static["domain"],
            args.dt_s,
            args.dx_m,
            args.dy_m,
            evaluation_mask=physics_evaluation_mask(
                static["domain"], static["ocean_mask"]
            ),
            momentum_evaluation_mask=fully_wet_stencil_mask(
                current, predicted, static["domain"], args.dry_tolerance_m
            ),
            gravity=args.gravity,
            dry_tolerance=args.dry_tolerance_m,
            y_axis_sign=args.y_axis_sign,
            component_weights=(1.0, 1.0, 1.0),
            component_scales=(
                args.physics_mass_scale,
                args.physics_momentum_scale,
                args.physics_momentum_scale,
            ),
            positivity_weight=0.0,
            dry_momentum_weight=0.0,
            wet_only_momentum=True,
            project_states=False,
            return_components=True,
        )
        ramp = min(1.0, float(epoch) / max(1, args.physics_warmup_epochs))
        physics_objective = ramp * (
            args.physics_mass_weight * physics["mass"]
            + args.physics_momentum_weight
            * (physics["momentum_x"] + physics["momentum_y"])
        )
        values.update(
            {
                "physics": physics_objective,
                "physics_mass": physics["mass"],
                "physics_momentum_x": physics["momentum_x"],
                "physics_momentum_y": physics["momentum_y"],
            }
        )
    values["objective"] = values["supervised"] + values["physics"]
    return values


class TinySmokeOperator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(len(FEATURE_NAMES), 8, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(8, 3, 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.net(values)


def model_source_files(model_key: str) -> list[Path]:
    architecture = ARCHITECTURE_KEYS[model_key]
    if architecture == "unet_v2":
        return [MODEL_ROOT / "UNet" / "model.py"]
    if architecture == "fno_v2":
        return [MODEL_ROOT / "FNO" / "model.py"]
    if architecture.startswith("poseidon"):
        return [
            ADAPTER_ROOT / "poseidon_common13.py",
            MODEL_ROOT / "Poseidon" / "upstream" / "scOT" / "model.py",
        ]
    if architecture == "morph_common13":
        morph_utils = MODEL_ROOT / "MORPH" / "upstream" / "src" / "utils"
        return [ADAPTER_ROOT / "morph_common13.py", *sorted(morph_utils.glob("*.py"))]
    return [PINO_ROOT / "pino" / "model.py"]


def build_model(
    args: argparse.Namespace, device: torch.device
) -> tuple[nn.Module, list[dict[str, Any]] | None, dict[str, Any]]:
    architecture = ARCHITECTURE_KEYS[args.model]
    report: dict[str, Any] = {
        "treatment": args.model,
        "architecture_key": architecture,
        "physics_regularizer": args.model in PHYSICS_TREATMENTS,
    }
    groups = None
    if architecture == "unet_v2":
        module = import_file("m008_residual_unet", MODEL_ROOT / "UNet" / "model.py")
        model = module.RDycoreUNet(
            in_channels=len(FEATURE_NAMES), out_channels=3, base_channels=16
        )
        report["base_channels"] = 16
    elif architecture == "fno_v2":
        module = import_file("m008_residual_fno", MODEL_ROOT / "FNO" / "model.py")
        settings = {
            "hidden_channels": 24,
            "modes_height": 12,
            "modes_width": 20,
            "layers": 4,
            "lifting_channels": 48,
            "projection_channels": 48,
            "domain_padding": 0.0625,
        }
        model = module.RDycoreFNO(
            in_channels=len(FEATURE_NAMES), out_channels=3, **settings
        )
        report.update(settings)
    elif architecture in ("poseidon_t_common13", "poseidon_b_common13"):
        module = import_file(
            "common13_residual_poseidon", ADAPTER_ROOT / "poseidon_common13.py"
        )
        pretrained = (
            args.poseidon_t_pretrained
            if architecture == "poseidon_t_common13"
            else args.poseidon_b_pretrained
        )
        model, adapter_report = module.build_poseidon_v2(
            pretrained,
            image_size=RASTER_SHAPE,
            refiner_width=24,
            initial_lead_time=0.05,
        )
        locked_rates = TREATMENT_LEARNING_RATES[args.model]
        backbone_lr = args.learning_rate or locked_rates["learning_rate"]
        endpoint_lr = (
            args.endpoint_learning_rate or locked_rates["endpoint_learning_rate"]
        )
        weight_decay = (
            args.weight_decay if args.weight_decay >= 0 else COMMON_WEIGHT_DECAY
        )
        groups = module.parameter_groups(
            model,
            backbone_lr=backbone_lr,
            endpoint_lr=endpoint_lr,
            time_lr=endpoint_lr,
            refiner_lr=endpoint_lr,
            weight_decay=weight_decay,
        )
        # The original adapter exempts time conditioning from decay.  This
        # campaign intentionally keeps one 1e-5 AdamW decay across every
        # treatment/group so only the necessary learning-rate split remains.
        for group in groups:
            group["weight_decay"] = weight_decay
        report.update(adapter_report)
        # The adapter's historical report says absolute state.  This workflow
        # owns the output semantics and supersedes that field.
        report["adapter_report_prediction_target"] = report.pop(
            "prediction_target", None
        )
    elif architecture == "morph_common13":
        morph_dir = MODEL_ROOT / "MORPH"
        sys.path[:0] = [str(morph_dir / "upstream"), str(morph_dir)]
        module = import_file(
            "common13_residual_morph", ADAPTER_ROOT / "morph_common13.py"
        )
        backbone, adapter_report = module.build_rdycore_morph(args.morph_pretrained)
        model = module.RDycoreMORPHV2(backbone, image_size=RASTER_SHAPE)
        module.set_tuning_mode(model, "full")
        locked_rates = TREATMENT_LEARNING_RATES[args.model]
        backbone_lr = args.learning_rate or locked_rates["learning_rate"]
        endpoint_lr = (
            args.endpoint_learning_rate or locked_rates["endpoint_learning_rate"]
        )
        weight_decay = (
            args.weight_decay if args.weight_decay >= 0 else COMMON_WEIGHT_DECAY
        )
        groups = module.parameter_groups(model, backbone_lr, endpoint_lr, weight_decay)
        report.update(adapter_report)
    elif architecture in ("houston_resunet_small", "houston_resunet_large"):
        module = import_file(
            "m008_residual_houston_resunet", PINO_ROOT / "pino" / "model.py"
        )
        large = architecture == "houston_resunet_large"
        base_channels = 24 if large else 14
        adapter_width = 8 if large else 0
        model = module.HoustonPINO(
            in_channels=len(FEATURE_NAMES),
            out_channels=3,
            base_channels=base_channels,
            depth=4,
            dropout=0.0,
            adapter_width=adapter_width,
        )
        model.freeze_adapter()
        report.update(
            {
                "base_channels": base_channels,
                "depth": 4,
                "dropout": 0.0,
                "adapter_width": adapter_width,
                "adapter_training": "frozen neutral" if large else "absent",
                "capacity_role": (
                    "old m032 PINO hyperparameters (base=24, depth=4, adapter=8)"
                    if large
                    else "small capacity-matched residual U-Net (base=14, depth=4)"
                ),
            }
        )
    else:
        raise AssertionError(architecture)
    model.to(device)
    report.update(
        {
            "prediction_target": "normalized conservative-state increment",
            "external_residual_update": True,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "trainable_parameters": sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            ),
        }
    )
    return model, groups, report


def optimizer_for(
    args: argparse.Namespace,
    model: nn.Module,
    groups: list[dict[str, Any]] | None,
) -> tuple[torch.optim.Optimizer, dict[str, Any]]:
    if groups is None:
        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(groups)
    description = {
        "name": "AdamW",
        "parameter_groups": [
            {
                "name": group.get("name", f"group_{index}"),
                "initial_learning_rate": float(group["lr"]),
                "weight_decay": float(group["weight_decay"]),
                "parameters": int(
                    sum(parameter.numel() for parameter in group["params"])
                ),
            }
            for index, group in enumerate(optimizer.param_groups)
        ],
        "schedule": (
            "ReduceLROnPlateau(mode=min, factor=0.5, patience=2, "
            "threshold=1e-4 relative, cooldown=0, min_lr=0, eps=1e-8), "
            "restarted at each curriculum stage"
        ),
    }
    return optimizer, description


def cosine_factor(epoch_index: int, total_epochs: int) -> float:
    progress = min(max(epoch_index, 0), total_epochs) / float(total_epochs)
    return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress))


def rollout_batch(
    args: argparse.Namespace,
    model: nn.Module,
    batch: Mapping[str, torch.Tensor],
    static: Mapping[str, torch.Tensor],
    normalization: Mapping[str, Any],
    epoch: int,
    optimizer: Optional[torch.optim.Optimizer],
) -> dict[str, float]:
    training = optimizer is not None
    device = static["domain"].device
    current = batch["current"].to(device, non_blocking=True)
    following = batch["following"].to(device, non_blocking=True)
    rainfall = batch["rainfall"].to(device, non_blocking=True)
    ocean = batch["ocean"].to(device, non_blocking=True)
    totals: defaultdict[str, float] = defaultdict(float)
    chunk: list[torch.Tensor] = []
    amp = args.mixed_precision == "bfloat16" and device.type == "cuda"
    for lead in range(args.rollout_hours):
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=amp,
        ):
            features = make_features(
                current, rainfall[:, lead], ocean[:, lead], static, normalization
            )
            normalized_delta = model(features)
        predicted, projected_delta = residual_state_update(
            current,
            normalized_delta.float(),
            normalization,
            static["domain"],
            dry_tolerance=args.dry_tolerance_m,
            max_velocity=args.max_velocity_m_s,
            straight_through=training,
        )
        values = step_objective(
            args,
            current,
            predicted,
            projected_delta,
            following[:, lead],
            rainfall[:, lead],
            static,
            normalization,
            epoch,
        )
        for name in LOSS_COMPONENTS:
            value = values[name]
            if not bool(torch.isfinite(value.detach()).all().item()):
                raise FloatingPointError(
                    f"non-finite {name} at epoch={epoch} lead={lead + 1}"
                )
            totals[name] += float(value.detach()) / args.rollout_hours
        if training:
            chunk.append(values["objective"] / args.rollout_hours)
            chunk_end = (
                (lead + 1) % args.detach_interval == 0
                or lead + 1 == args.rollout_hours
            )
            if chunk_end:
                torch.stack(chunk).sum().backward()
                current = predicted.detach()
                chunk = []
            else:
                current = predicted
        else:
            current = predicted
    return dict(totals)


def mean_records(records: Sequence[Mapping[str, float]]) -> dict[str, float]:
    return {
        name: float(np.mean([record[name] for record in records]))
        for name in LOSS_COMPONENTS
    }


def run_epoch(
    args: argparse.Namespace,
    model: nn.Module,
    loader: DataLoader,
    static: Mapping[str, torch.Tensor],
    normalization: Mapping[str, Any],
    epoch: int,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    records = []
    windows = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            if training:
                optimizer.zero_grad(set_to_none=True)
            metrics = rollout_batch(
                args, model, batch, static, normalization, epoch, optimizer
            )
            if training:
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad],
                    args.gradient_clip,
                    error_if_nonfinite=True,
                )
                optimizer.step()
            records.append(metrics)
            windows += int(batch["current"].shape[0])
    if not records:
        raise RuntimeError("data loader produced no batches")
    result = mean_records(records)
    result["windows"] = windows
    result["forecast_transitions"] = windows * args.rollout_hours
    return result


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}")
    try:
        with temporary.open("wb") as stream:
            torch.save(dict(payload), stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_link(source: Path, destination: Path) -> None:
    """Atomically retain a checkpoint inode without another multi-GB write."""

    temporary = destination.with_name(
        f".{destination.name}.tmp.{os.getpid()}.{time.time_ns()}"
    )
    try:
        os.link(source, temporary)
        os.replace(temporary, destination)
    except OSError:
        temporary.unlink(missing_ok=True)
        copy_path = destination.with_name(
            f".{destination.name}.copy.{os.getpid()}.{time.time_ns()}"
        )
        try:
            shutil.copy2(source, copy_path)
            os.replace(copy_path, destination)
        finally:
            copy_path.unlink(missing_ok=True)
    finally:
        temporary.unlink(missing_ok=True)


def ensure_best_checkpoint_consistency(
    last_path: Path,
    best_path: Path,
    last_checkpoint: Mapping[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Repair only the crash window in which last was committed before best.

    If the current last checkpoint is itself the selected best, it contains the
    exact recoverable model state and may safely replace a missing/stale best
    link.  Otherwise a missing or inconsistent best is unrecoverable and the
    stage fails closed instead of silently handing off the wrong weights.
    """

    last_epoch = int(last_checkpoint.get("epoch", -1))
    best_epoch = int(last_checkpoint.get("best_epoch", -1))
    best_score = float(last_checkpoint.get("best_validation_supervised", math.inf))
    if last_epoch < 1 or best_epoch < 1 or best_epoch > last_epoch:
        raise RuntimeError("last checkpoint has invalid best-checkpoint counters")
    repairable = last_epoch == best_epoch

    def load_and_validate() -> dict[str, Any]:
        payload = torch.load(best_path, map_location="cpu", weights_only=False)
        validate_resume(payload.get("config", {}), args)
        mismatches = {}
        if int(payload.get("epoch", -1)) != best_epoch:
            mismatches["epoch"] = (payload.get("epoch"), best_epoch)
        if int(payload.get("best_epoch", -1)) != best_epoch:
            mismatches["best_epoch"] = (payload.get("best_epoch"), best_epoch)
        if float(payload.get("best_validation_supervised", math.inf)) != best_score:
            mismatches["best_validation_supervised"] = (
                payload.get("best_validation_supervised"),
                best_score,
            )
        if mismatches:
            raise RuntimeError(f"best checkpoint does not match last counters: {mismatches}")
        return payload

    try:
        if not best_path.is_file() or best_path.stat().st_size <= 0:
            raise RuntimeError("best checkpoint is missing or empty")
        return load_and_validate()
    except Exception as error:
        if not repairable:
            raise RuntimeError(
                "best checkpoint is inconsistent and cannot be reconstructed from "
                f"last epoch {last_epoch} (selected best epoch {best_epoch})"
            ) from error
        atomic_link(last_path, best_path)
        repaired = load_and_validate()
        print(
            f"repaired best checkpoint from atomically committed last epoch {last_epoch}",
            flush=True,
        )
        return repaired
def update_history(path: Path, row: Mapping[str, Any]) -> None:
    epoch = int(row["epoch"])
    if path.exists():
        frame = pd.read_csv(path)
        if (frame["epoch"].astype(int) > epoch).any():
            raise RuntimeError("history is ahead of the resumable checkpoint")
        frame = frame.loc[frame["epoch"].astype(int) != epoch]
        frame = pd.concat((frame, pd.DataFrame([row])), ignore_index=True)
    else:
        frame = pd.DataFrame([row])
    frame = frame.sort_values("epoch")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}")
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def sampler_state(loader: DataLoader) -> Optional[torch.Tensor]:
    generator = getattr(loader.sampler, "generator", None)
    return generator.get_state() if generator is not None else None


def capture_rng(train_loader: DataLoader, validation_loader: DataLoader) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "train_sampler": sampler_state(train_loader),
        "validation_sampler": sampler_state(validation_loader),
    }


def restore_rng(
    state: Mapping[str, Any], train_loader: DataLoader, validation_loader: DataLoader
) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    for key, loader in (
        ("train_sampler", train_loader),
        ("validation_sampler", validation_loader),
    ):
        generator = getattr(loader.sampler, "generator", None)
        if generator is not None and state.get(key) is not None:
            generator.set_state(state[key])


RESUME_INVARIANTS = (
    "subset",
    "model",
    "rollout_hours",
    "dataset_root",
    "shared_artifact_dir",
    "channel_artifact_dir",
    "output_root",
    "campaign_manifest",
    "epochs",
    "transitions_per_epoch",
    "validation_transitions",
    "batch_size",
    "detach_interval",
    "seed",
    "mixed_precision",
    "learning_rate",
    "endpoint_learning_rate",
    "weight_decay",
    "gradient_clip",
    "depth_weight",
    "momentum_weight",
    "supervised_scale",
    "dry_tolerance_m",
    "max_velocity_m_s",
    "physics_mass_weight",
    "physics_momentum_weight",
    "physics_mass_scale",
    "physics_momentum_scale",
    "physics_warmup_epochs",
    "dx_m",
    "dy_m",
    "dt_s",
    "y_axis_sign",
    "gravity",
    "manning_n",
)


def validate_resume(saved: Mapping[str, Any], args: argparse.Namespace) -> None:
    current = vars(args)
    mismatches = []
    for key in RESUME_INVARIANTS:
        left = saved.get(key)
        right = current[key]
        if isinstance(right, Path):
            left, right = str(left), str(right)
        if left != right:
            mismatches.append(f"{key}: checkpoint={left!r}, requested={right!r}")
    if mismatches:
        raise RuntimeError(
            "refusing scientifically incompatible resume:\n  " + "\n  ".join(mismatches)
        )


def validate_provenance_files(
    saved: Mapping[str, Any],
    current: Mapping[str, Any],
    context: str,
    ignored_names: Sequence[str] = (),
) -> None:
    """Fail closed if hashed scientific inputs changed within a curriculum."""

    mismatches = []
    saved_files = saved.get("files", {})
    current_files = current.get("files", {})
    for name, saved_identity in saved_files.items():
        if name in ignored_names:
            continue
        if "sha256" not in saved_identity or name not in current_files:
            continue
        current_digest = current_files[name].get("sha256")
        if saved_identity["sha256"] != current_digest:
            mismatches.append(name)
    if mismatches:
        raise RuntimeError(
            f"scientific provenance changed relative to {context}: {sorted(mismatches)}"
        )


def build_provenance(
    args: argparse.Namespace,
    split_file: Path,
    norm_file: Path,
    static_file: Path,
    masks_file: Path,
    train_data: TrajectoryWindowDataset,
    validation_data: TrajectoryWindowDataset,
    model_report: Mapping[str, Any],
    optimizer_report: Mapping[str, Any],
    initial_checkpoint: Optional[Path],
) -> dict[str, Any]:
    source_files = [Path(__file__).resolve(), PINO_ROOT / "pino" / "physics.py"]
    source_files.extend(model_source_files(args.model))
    files = {
        "trajectory_split": file_identity(split_file),
        "normalization": file_identity(norm_file),
        "static_grid": file_identity(static_file),
        "channel_masks": file_identity(masks_file),
        "trainer": file_identity(Path(__file__)),
        "campaign_manifest": file_identity(args.campaign_manifest),
        "model_registry": file_identity(MODEL_REGISTRY_PATH),
    }
    manifest = load_json(args.campaign_manifest)
    canonical_pino = manifest["canonical_pino"]
    files["canonical_pino_checkpoint"] = file_identity(
        Path(canonical_pino["primary_checkpoint"])
    )
    for index, source in enumerate(source_files[1:], start=1):
        files[f"source_{index}"] = file_identity(source)
    architecture = ARCHITECTURE_KEYS[args.model]
    if architecture in ("poseidon_t_common13", "poseidon_b_common13"):
        pretrained_dir = (
            args.poseidon_t_pretrained
            if architecture == "poseidon_t_common13"
            else args.poseidon_b_pretrained
        )
        files["pretrained_config"] = file_identity(pretrained_dir / "config.json")
        candidates = (
            pretrained_dir / "model.safetensors",
            pretrained_dir / "pytorch_model.bin",
        )
        weight_file = next((path for path in candidates if path.is_file()), None)
        if weight_file is None:
            raise FileNotFoundError(f"no Poseidon weight file in {pretrained_dir}")
        files["pretrained_weights"] = file_identity(weight_file)
    elif architecture == "morph_common13":
        files["pretrained_weights"] = file_identity(args.morph_pretrained)
    if initial_checkpoint is not None:
        files["initial_checkpoint"] = file_identity(initial_checkpoint)
    packages = {}
    for name in ("torch", "numpy", "pandas", "h5py", "neuraloperator", "transformers"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    config = vars(args).copy()
    for key, value in list(config.items()):
        if isinstance(value, Path):
            config[key] = str(value.resolve())
    return {
        "created_utc": utc_now(),
        "protocol": protocol_spec(args.subset),
        "resolved_arguments": config,
        "command": [sys.executable, *sys.argv],
        "host": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "packages": packages,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "resolved_device": model_report.get("resolved_device"),
        "git": {
            "benchmark": git_identity(BENCHMARK_ROOT),
            "foundation": git_identity(FOUNDATION_ROOT),
        },
        "files": files,
        "model": dict(model_report),
        "optimizer": dict(optimizer_report),
        "initial_checkpoint": str(initial_checkpoint) if initial_checkpoint else None,
        "initialization_policy": (
            "preceding validation-best checkpoint; optimizer and schedule restarted"
            if initial_checkpoint
            else str(MODEL_SPECS[args.model]["initialization_policy"])
        ),
        "data": {
            "training_trajectories": len(train_data.table),
            "training_events": int(train_data.table["event_id"].nunique()),
            "training_members": sorted(
                int(value) for value in train_data.table["member_id"].unique()
            ),
            "validation_trajectories": len(validation_data.table),
            "validation_events": int(validation_data.table["event_id"].nunique()),
            "validation_members": sorted(
                int(value) for value in validation_data.table["member_id"].unique()
            ),
            "train_windows_available": len(train_data),
            "validation_windows_available": len(validation_data),
        },
        "loss_audit": {
            "components": list(LOSS_COMPONENTS),
            "auxiliary_losses": list(AUXILIARY_LOSSES),
            "validation_selection_excludes_physics": True,
        },
    }


def resolve_protocol_defaults(args: argparse.Namespace) -> None:
    local_metadata = BENCHMARK_ROOT / "data" / "metadata" / args.subset
    if args.shared_artifact_dir is None:
        args.shared_artifact_dir = local_metadata / "shared"
    if args.channel_artifact_dir is None:
        args.channel_artifact_dir = local_metadata / "channels"
    if args.output_root is None:
        args.output_root = BENCHMARK_ROOT / "results" / "m032_depth4_plateau_v1"
    if args.transitions_per_epoch == 0:
        args.transitions_per_epoch = SUBSET_TRANSITIONS_PER_EPOCH[args.subset]
    expected_epochs = STAGE_EPOCHS[int(args.rollout_hours)]
    if args.epochs == 0:
        args.epochs = expected_epochs
    rates = TREATMENT_LEARNING_RATES[args.model]
    if args.learning_rate == 0.0:
        args.learning_rate = rates["learning_rate"]
    if args.endpoint_learning_rate == 0.0:
        args.endpoint_learning_rate = rates["endpoint_learning_rate"]
    if args.weight_decay < 0.0:
        args.weight_decay = COMMON_WEIGHT_DECAY


def validate_arguments(args: argparse.Namespace) -> None:
    positive = (
        "epochs",
        "transitions_per_epoch",
        "validation_transitions",
        "batch_size",
        "threads",
        "detach_interval",
    )
    for name in positive:
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.epochs_per_invocation < 0:
        raise ValueError("--epochs-per-invocation cannot be negative")
    if args.transitions_per_epoch % args.rollout_hours:
        raise ValueError("training transitions must be divisible by rollout hours")
    if args.validation_transitions % args.rollout_hours:
        raise ValueError("validation transitions must be divisible by rollout hours")
    if args.detach_interval > args.rollout_hours:
        raise ValueError("detach interval cannot exceed rollout hours")
    if args.rollout_hours % args.detach_interval:
        raise ValueError("rollout hours must be divisible by detach interval")
    for name in ("depth_weight", "momentum_weight", "physics_mass_weight", "physics_momentum_weight"):
        if getattr(args, name) < 0:
            raise ValueError(f"{name} cannot be negative")
    if args.dry_tolerance_m != 1.0e-4 or args.max_velocity_m_s != 10.0:
        raise ValueError("the protocol locks dry tolerance=1e-4 and velocity cap=10 m/s")
    expected_epochs = STAGE_EPOCHS[args.rollout_hours]
    if (
        args.epochs != expected_epochs
        or args.transitions_per_epoch
        != SUBSET_TRANSITIONS_PER_EPOCH[args.subset]
        or args.validation_transitions != VALIDATION_TRANSITIONS_PER_EPOCH
    ):
        raise ValueError(
            f"the reproduction protocol locks AR{args.rollout_hours} to at most "
            f"{expected_epochs} epochs, "
            f"{SUBSET_TRANSITIONS_PER_EPOCH[args.subset]} training transitions, "
            f"and {VALIDATION_TRANSITIONS_PER_EPOCH} validation transitions per pass"
        )
    if args.seed != 20260917:
        raise ValueError("the reproduction protocol locks seed 20260917")
    if args.batch_size != 1:
        raise ValueError("the reproduction protocol locks batch size 1")
    if args.detach_interval != COMMON_DETACH_INTERVAL:
        raise ValueError(
            f"the reproduction protocol locks detach interval {COMMON_DETACH_INTERVAL}"
        )
    if (
        args.depth_weight != 4.0
        or args.momentum_weight != 1.0
        or args.supervised_scale != SUPERVISED_SCALE
    ):
        raise ValueError(
            "the reproduction protocol locks h:qx:qy=4:1:1 and "
            f"supervised_scale={SUPERVISED_SCALE}"
        )
    expected_rates = TREATMENT_LEARNING_RATES[args.model]
    if (
        args.learning_rate != expected_rates["learning_rate"]
        or args.endpoint_learning_rate != expected_rates["endpoint_learning_rate"]
    ):
        raise ValueError(
            f"the primary protocol locks {args.model} learning rates to "
            f"{expected_rates}"
        )
    if args.weight_decay != COMMON_WEIGHT_DECAY:
        raise ValueError(
            f"the primary protocol locks AdamW weight decay to {COMMON_WEIGHT_DECAY}"
        )
    if args.gradient_clip != COMMON_GRADIENT_CLIP:
        raise ValueError(
            f"the primary protocol locks gradient clipping to {COMMON_GRADIENT_CLIP}"
        )
    expected_swe = {
        "physics_mass_weight": SWE_CONTRACT["mass_weight"],
        "physics_momentum_weight": SWE_CONTRACT["momentum_x_weight"],
        "physics_mass_scale": SWE_CONTRACT["mass_scale_m_s"],
        "physics_momentum_scale": SWE_CONTRACT["momentum_x_scale_m2_s2"],
        "physics_warmup_epochs": SWE_CONTRACT["warmup_epochs"],
        "dx_m": SWE_CONTRACT["dx_m"],
        "dy_m": SWE_CONTRACT["dy_m"],
        "dt_s": SWE_CONTRACT["dt_s"],
        "y_axis_sign": SWE_CONTRACT["y_axis_sign"],
        "gravity": SWE_CONTRACT["gravity_m_s2"],
        "manning_n": SWE_CONTRACT["manning_n"],
    }
    changed_swe = {
        name: (getattr(args, name), expected)
        for name, expected in expected_swe.items()
        if getattr(args, name) != expected
    }
    if changed_swe:
        raise ValueError(f"the primary protocol locks SWE constants: {changed_swe}")
    if args.mixed_precision != "off":
        raise ValueError("the reproduction protocol requires strict FP32 with AMP disabled")
    if args.device != "cuda":
        raise ValueError("the production reproduction protocol requires CUDA")
    if args.skip_checkpoints:
        raise ValueError("the reproduction protocol requires atomic checkpoints")
    if args.deterministic:
        raise ValueError(
            "the reproduction protocol locks deterministic algorithms off because CUDA "
            "bilinear-interpolation backward is unsupported; seeded execution remains enabled"
        )
    if not args.campaign_manifest.is_file():
        raise FileNotFoundError(args.campaign_manifest)
    manifest = load_json(args.campaign_manifest)
    expected_protocol_ids = manifest.get("protocol_ids", {})
    if manifest.get("campaign_id") != "houston-common13-m032-depth4-plateau-v1":
        raise ValueError("campaign manifest has the wrong campaign_id")
    if expected_protocol_ids.get(args.subset) != protocol_spec(args.subset)["protocol_id"]:
        raise ValueError("campaign manifest protocol_id does not match executable contract")
    if manifest.get("feature_names") != list(FEATURE_NAMES):
        raise ValueError("campaign manifest feature order does not match executable contract")
    if manifest.get("model_keys") != list(MODEL_KEYS):
        raise ValueError("campaign manifest treatment order does not match executable contract")
    if manifest.get("physics_treatments") != sorted(PHYSICS_TREATMENTS):
        raise ValueError("campaign manifest physics treatments do not match executable contract")
    if manifest.get("model_registry_file") != MODEL_REGISTRY_PATH.name:
        raise ValueError("campaign manifest points to the wrong model registry")
    if manifest.get("model_registry_sha256") != sha256_file(MODEL_REGISTRY_PATH):
        raise ValueError("campaign manifest model registry hash mismatch")
    canonical = manifest.get("canonical_pino", {})
    canonical_checkpoint = Path(str(canonical.get("primary_checkpoint", "")))
    if (
        canonical.get("model") != "pino_resunet_large"
        or canonical.get("protocol_id")
        != "houston-m032-common13-pino-depth4-plateau-v2"
        or int(canonical.get("best_epoch", -1)) != 52
        or not canonical_checkpoint.is_file()
        or sha256_file(canonical_checkpoint)
        != canonical.get("primary_checkpoint_sha256")
    ):
        raise ValueError("canonical PINO anchor is missing or inconsistent")
    stage32 = canonical_checkpoint.parent
    summary_path = stage32 / "stage_summary.json"
    summary_anchor = canonical.get("stage_anchors", {}).get("32", {})
    if (
        not summary_path.is_file()
        or sha256_file(summary_path) != summary_anchor.get("stage_summary_sha256")
    ):
        raise ValueError("canonical PINO AR32 summary anchor mismatch")
    canonical_summary = load_json(summary_path)
    if (
        canonical_summary.get("protocol_id") != canonical["protocol_id"]
        or int(canonical_summary.get("best_epoch", -1)) != canonical["best_epoch"]
        or not math.isclose(
            float(canonical_summary.get("best_validation_supervised", math.inf)),
            float(canonical.get("best_validation_supervised", math.inf)),
            rel_tol=0.0,
            abs_tol=0.0,
        )
    ):
        raise ValueError("canonical PINO AR32 metadata does not match its anchor")


def smoke_test(args: argparse.Namespace) -> None:
    args.device = "cpu"
    args.mixed_precision = "off"
    args.rollout_hours = 4
    args.detach_interval = 4
    height, width = 16, 24
    domain = torch.ones(1, 1, height, width)
    static = {
        "domain": domain,
        "bed": torch.zeros_like(domain),
        "ocean_mask": torch.zeros_like(domain),
        "x_coordinate": torch.linspace(-1.0, 1.0, width)[None, None, None, :].expand(
            1, 1, height, width
        ),
        "y_coordinate": torch.linspace(1.0, -1.0, height)[None, None, :, None].expand(
            1, 1, height, width
        ),
        "stream_core": torch.zeros_like(domain),
        "channel": torch.zeros_like(domain),
        "stream_proximity": torch.zeros_like(domain),
    }
    normalization = {
        "state": {"mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]},
        "state_delta": {"mean": [0.0, 0.0, 0.0], "std": [0.1, 0.1, 0.1]},
        "log1p_rainfall": {"mean": [0.0], "std": [1.0]},
        "ocean_height": {"mean": [0.0], "std": [1.0]},
        "bed_elevation": {"mean": [0.0], "std": [1.0]},
    }
    current = torch.zeros(1, 3, height, width)
    current[:, 0] = 0.2
    following = current[:, None].expand(-1, 4, -1, -1, -1).clone()
    batch = {
        "current": current,
        "following": following,
        "rainfall": torch.zeros(1, 4, 1, height, width),
        "ocean": torch.zeros(1, 4),
    }
    model = TinySmokeOperator()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4)
    metrics = rollout_batch(
        args, model, batch, static, normalization, epoch=1, optimizer=optimizer
    )
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    print(json.dumps({"smoke": "passed", "model_treatment": args.model, **metrics}, indent=2))


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    resolve_protocol_defaults(args)
    if args.smoke:
        validate_arguments(args)
        smoke_test(args)
        return
    validate_arguments(args)
    if not args.invocation_id:
        args.invocation_id = f"manual.{os.getpid()}.{time.time_ns()}"
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(
        "cuda"
        if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    if args.mixed_precision != "off" and device.type != "cuda":
        raise ValueError("bfloat16 mode requires CUDA")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        # Match the earlier PINO's practical determinism without enabling
        # torch.use_deterministic_algorithms, whose bilinear backward is not
        # supported on this CUDA path.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.set_float32_matmul_precision("highest")
    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    split_file = args.shared_artifact_dir / "trajectory_split.csv"
    norm_file = args.shared_artifact_dir / "normalization.json"
    static_file = args.dataset_root / "static_grid.h5"
    masks_file = args.channel_artifact_dir / "channel_masks_fullres.npz"
    required = [split_file, norm_file, static_file, masks_file, Path(__file__)]
    required.append(args.campaign_manifest)
    if ARCHITECTURE_KEYS[args.model].startswith("poseidon"):
        pretrained = (
            args.poseidon_t_pretrained
            if args.model == "poseidon_t_common13"
            else args.poseidon_b_pretrained
        )
        if not pretrained.is_dir():
            raise FileNotFoundError(pretrained)
    if args.model == "morph_common13":
        required.append(args.morph_pretrained)
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    stage_dir = args.output_root / args.model / f"stage_ar{args.rollout_hours:02d}"
    stage_dir.mkdir(parents=True, exist_ok=True)
    # Production duplicate-writer protection is the owner-checked atomic
    # directory lock in workflows/common13_v1/train.sbatch.  Do not add a
    # descriptor flock here: GPFS compute nodes return errno 524 for that call.
    last_path = stage_dir / "checkpoint_last.pt"
    best_path = stage_dir / "checkpoint_best_validation_supervised.pt"
    final_path = stage_dir / "checkpoint_final.pt"
    history_path = stage_dir / "history.csv"
    summary_path = stage_dir / "stage_summary.json"
    if summary_path.exists() and final_path.exists():
        summary = load_json(summary_path)
        if summary.get("complete") is True:
            last_checkpoint = torch.load(
                last_path, map_location="cpu", weights_only=False
            )
            validate_resume(last_checkpoint.get("config", {}), args)
            ensure_best_checkpoint_consistency(
                last_path, best_path, last_checkpoint, args
            )
            expected_hashes = summary.get("checkpoint_sha256", {})
            paths_by_role = {
                "best": best_path,
                "last": last_path,
                "final": final_path,
            }
            if set(expected_hashes) != set(paths_by_role):
                raise RuntimeError("completed stage has incomplete checkpoint hashes")
            for role, path in paths_by_role.items():
                if sha256_file(path) != expected_hashes[role]:
                    raise RuntimeError(f"completed stage {role} checkpoint hash mismatch")
            print(json.dumps({"status": "already_complete", **summary}, indent=2))
            return
        raise RuntimeError(f"inconsistent existing stage summary: {summary_path}")

    initial_checkpoint: Optional[Path] = None
    if args.rollout_hours > 4:
        previous = {8: 4, 16: 8, 32: 16}[args.rollout_hours]
        initial_checkpoint = (
            args.output_root
            / args.model
            / f"stage_ar{previous:02d}"
            / "checkpoint_best_validation_supervised.pt"
        )
        if not initial_checkpoint.is_file():
            raise FileNotFoundError(
                f"preceding validation-best checkpoint is missing: {initial_checkpoint}"
            )

    normalization = load_json(norm_file)
    static = load_static(static_file, masks_file, device)
    train_data, validation_data, train_loader, validation_loader = build_loaders(
        args, split_file
    )
    validate_dataset_contract(args.subset, train_data, validation_data)
    model, groups, model_report = build_model(args, device)
    model_report["resolved_device"] = str(device)
    model_report["strict_fp32"] = True
    model_report["tf32_enabled"] = False
    model_report["deterministic_algorithms"] = bool(args.deterministic)
    model_report["cudnn_deterministic"] = bool(
        device.type == "cuda" and torch.backends.cudnn.deterministic
    )
    if initial_checkpoint is not None:
        source = torch.load(initial_checkpoint, map_location="cpu", weights_only=False)
        source_config = source["config"]
        if source_config["model"] != args.model:
            raise RuntimeError("preceding checkpoint treatment mismatch")
        if source_config.get("subset") != args.subset:
            raise RuntimeError("preceding checkpoint subset mismatch")
        if source_config.get("protocol_id") != protocol_spec(args.subset)["protocol_id"]:
            raise RuntimeError("preceding checkpoint protocol mismatch")
        if list(source_config.get("input_features", ())) != list(FEATURE_NAMES):
            raise RuntimeError("preceding checkpoint feature schema mismatch")
        model.load_state_dict(source["model_state"], strict=True)
    optimizer, optimizer_report = optimizer_for(args, model, groups)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=PLATEAU_FACTOR,
        patience=PLATEAU_PATIENCE,
        threshold=PLATEAU_THRESHOLD,
        threshold_mode=PLATEAU_THRESHOLD_MODE,
        cooldown=PLATEAU_COOLDOWN,
        min_lr=PLATEAU_MIN_LR,
        eps=PLATEAU_EPS,
    )

    config = vars(args).copy()
    for key, value in list(config.items()):
        if isinstance(value, Path):
            config[key] = str(value.resolve())
    config.update(
        {
            "protocol_id": protocol_spec(args.subset)["protocol_id"],
            "architecture_key": ARCHITECTURE_KEYS[args.model],
            "physics_enabled": args.model in PHYSICS_TREATMENTS,
            "input_features": list(FEATURE_NAMES),
            "prediction_target": "normalized conservative-state increment",
            "external_residual_update": True,
            "auxiliary_losses": list(AUXILIARY_LOSSES),
            "train_windows_per_epoch": args.transitions_per_epoch // args.rollout_hours,
            "validation_windows_per_epoch": args.validation_transitions // args.rollout_hours,
            "primary_checkpoint": "checkpoint_best_validation_supervised.pt",
            "selection_metric": "validation_supervised",
            "learning_rate_schedule": "reduce_on_plateau",
            "plateau_factor": PLATEAU_FACTOR,
            "plateau_patience": PLATEAU_PATIENCE,
            "plateau_threshold": PLATEAU_THRESHOLD,
            "plateau_threshold_mode": PLATEAU_THRESHOLD_MODE,
            "plateau_cooldown": PLATEAU_COOLDOWN,
            "plateau_min_lr": PLATEAU_MIN_LR,
            "plateau_eps": PLATEAU_EPS,
            "minimum_epochs": EARLY_STOPPING_MINIMUM_EPOCHS,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "minimum_relative_improvement": MINIMUM_RELATIVE_IMPROVEMENT,
            "stage_initialization": str(initial_checkpoint) if initial_checkpoint else None,
            "stage_initialization_sha256": (
                sha256_file(initial_checkpoint) if initial_checkpoint else None
            ),
            "cudnn_deterministic": bool(
                device.type == "cuda" and torch.backends.cudnn.deterministic
            ),
            "torch_deterministic_algorithms": bool(args.deterministic),
        }
    )
    provenance = build_provenance(
        args,
        split_file,
        norm_file,
        static_file,
        masks_file,
        train_data,
        validation_data,
        model_report,
        optimizer_report,
        initial_checkpoint,
    )
    config_path = stage_dir / "training_config.json"
    provenance_path = stage_dir / "provenance.json"
    if config_path.exists():
        validate_resume(load_json(config_path), args)
    else:
        atomic_json(config_path, config)
    if provenance_path.exists():
        validate_provenance_files(
            load_json(provenance_path), provenance, "this stage's first invocation"
        )
    else:
        atomic_json(provenance_path, provenance)
    if initial_checkpoint is not None:
        source_provenance_path = Path(source.get("provenance_file", ""))
        if not source_provenance_path.is_file():
            raise FileNotFoundError(
                f"preceding stage provenance is missing: {source_provenance_path}"
            )
        validate_provenance_files(
            load_json(source_provenance_path),
            provenance,
            "preceding curriculum stage",
            ignored_names=("initial_checkpoint",),
        )
        del source

    start_epoch = 1
    best_validation_supervised = float("inf")
    best_epoch = 0
    stale_epochs = 0
    if not last_path.exists() and (
        history_path.exists() or best_path.exists() or final_path.exists()
    ):
        raise RuntimeError(
            "stage has history/best/final artifacts but no atomic last checkpoint"
        )
    if last_path.exists():
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
        validate_resume(checkpoint["config"], args)
        ensure_best_checkpoint_consistency(last_path, best_path, checkpoint, args)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        restore_rng(checkpoint["rng_state"], train_loader, validation_loader)
        start_epoch = int(checkpoint["epoch"]) + 1
        best_validation_supervised = float(checkpoint["best_validation_supervised"])
        best_epoch = int(checkpoint["best_epoch"])
        stale_epochs = int(checkpoint.get("early_stopping_stale_epochs", 0))
        if checkpoint.get("history_row") is not None:
            update_history(history_path, checkpoint["history_row"])

    stop_requested = {"value": False}

    def request_stop(signum, _frame):
        stop_requested["value"] = True
        print(f"received signal {signum}; stopping safely after the current epoch", flush=True)

    signal.signal(signal.SIGUSR1, request_stop)
    converged = (
        start_epoch - 1 >= EARLY_STOPPING_MINIMUM_EPOCHS
        and stale_epochs >= EARLY_STOPPING_PATIENCE
    )
    invocation_end = start_epoch - 1 if converged else args.epochs
    if args.epochs_per_invocation and not converged:
        invocation_end = min(
            args.epochs, start_epoch + args.epochs_per_invocation - 1
        )
    invocation_start_epoch = start_epoch
    final_epoch = start_epoch - 1
    started_invocation = time.time()
    for epoch in range(start_epoch, invocation_end + 1):
        started = time.time()
        train_metrics = run_epoch(
            args, model, train_loader, static, normalization, epoch, optimizer
        )
        validation_metrics = run_epoch(
            args, model, validation_loader, static, normalization, epoch
        )
        score = float(validation_metrics["supervised"])
        scheduler.step(score)
        final_epoch = epoch
        threshold = best_validation_supervised * (1.0 - MINIMUM_RELATIVE_IMPROVEMENT)
        significant = math.isfinite(score) and (
            not math.isfinite(best_validation_supervised) or score < threshold
        )
        selected = math.isfinite(score) and (
            not math.isfinite(best_validation_supervised)
            or score < best_validation_supervised
        )
        if selected:
            best_validation_supervised = score
            best_epoch = epoch
        stale_epochs = 0 if significant else stale_epochs + 1
        converged = (
            epoch >= EARLY_STOPPING_MINIMUM_EPOCHS
            and stale_epochs >= EARLY_STOPPING_PATIENCE
        )
        row = {
            "epoch": epoch,
            "elapsed_minutes": (time.time() - started) / 60.0,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "best_validation_supervised": best_validation_supervised,
            "significant_validation_improvement": significant,
            "early_stopping_stale_epochs": stale_epochs,
            "plateau_converged": converged,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
        }
        for group_index, group in enumerate(optimizer.param_groups):
            group_name = str(group.get("name", f"group_{group_index}"))
            row[f"learning_rate_{group_name}"] = float(group["lr"])
        payload = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_validation_supervised": best_validation_supervised,
            "best_epoch": best_epoch,
            "early_stopping_stale_epochs": stale_epochs,
            "config": config,
            "provenance_file": str(stage_dir / "provenance.json"),
            "rng_state": capture_rng(train_loader, validation_loader),
            "history_row": row,
            "invocation_id": args.invocation_id,
        }
        if not args.skip_checkpoints:
            atomic_checkpoint(last_path, payload)
            if selected:
                atomic_link(last_path, best_path)
        update_history(history_path, row)
        print(json.dumps(row), flush=True)
        if stop_requested["value"] or converged:
            break

    complete = converged
    if complete and not args.skip_checkpoints:
        committed_last = torch.load(last_path, map_location="cpu", weights_only=False)
        ensure_best_checkpoint_consistency(
            last_path, best_path, committed_last, args
        )
        atomic_link(last_path, final_path)
    reached_maximum_without_convergence = (
        final_epoch == args.epochs and not complete
    )
    checkpoint_sha256 = {}
    if complete:
        checkpoint_sha256 = {
            "best": sha256_file(best_path),
            "last": sha256_file(last_path),
            "final": sha256_file(final_path),
        }
    summary = {
        "protocol_id": protocol_spec(args.subset)["protocol_id"],
        "subset": args.subset,
        "model": args.model,
        "architecture_key": ARCHITECTURE_KEYS[args.model],
        "rollout_hours": args.rollout_hours,
        "epochs_completed": final_epoch,
        "epochs_required": args.epochs,
        "complete": complete,
        "stop_reason": (
            "validation_plateau"
            if complete
            else "maximum_epochs_without_convergence"
            if reached_maximum_without_convergence
            else "time_slice_signal"
            if stop_requested["value"]
            else "invocation_slice_complete"
        ),
        "invocation_id": args.invocation_id,
        "invocation_start_epoch": invocation_start_epoch,
        "invocation_end_epoch": final_epoch,
        "epochs_this_invocation": max(0, final_epoch - invocation_start_epoch + 1),
        "best_validation_supervised": best_validation_supervised,
        "best_epoch": best_epoch,
        "early_stopping_stale_epochs": stale_epochs,
        "convergence_rule": {
            "maximum_epochs": args.epochs,
            "minimum_epochs": EARLY_STOPPING_MINIMUM_EPOCHS,
            "patience": EARLY_STOPPING_PATIENCE,
            "minimum_relative_improvement": MINIMUM_RELATIVE_IMPROVEMENT,
        },
        "primary_checkpoint": str(best_path),
        "secondary_checkpoint": str(final_path),
        "checkpoint_sha256": checkpoint_sha256,
        "initial_checkpoint": str(initial_checkpoint) if initial_checkpoint else None,
        "initial_checkpoint_sha256": (
            sha256_file(initial_checkpoint) if initial_checkpoint else None
        ),
        "primary_evaluation": bool(complete and args.rollout_hours == 32),
        "forecast_transitions_completed_this_stage": final_epoch
        * args.transitions_per_epoch,
        "invocation_elapsed_minutes": (time.time() - started_invocation) / 60.0,
        "completed_utc": utc_now(),
    }
    atomic_json(stage_dir / "slice_status.json", summary)
    if complete:
        atomic_json(summary_path, summary)
    print(json.dumps(summary, indent=2), flush=True)
    if not complete:
        raise SystemExit(86 if reached_maximum_without_convergence else 85)


if __name__ == "__main__":
    main()
