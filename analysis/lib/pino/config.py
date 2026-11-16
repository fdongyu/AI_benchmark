"""Typed experiment configuration with reproducible serialization."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import os

import yaml


@dataclass
class DataConfig:
    dataset_root: str
    split_csv: str
    normalization_json: str
    require_normalization_provenance: bool = False
    static_grid: Optional[str] = None
    train_split: str = "train"
    validation_split: str = "validation"
    context_steps: int = 1
    forecast_steps: int = 4
    stride: int = 1
    tile_size: Optional[Tuple[int, int]] = (128, 128)
    tile_stride: Optional[Tuple[int, int]] = None
    halo: int = 2
    batch_size: int = 1
    num_workers: int = 4
    pin_memory: bool = True
    train_windows_per_epoch: Optional[int] = 2048
    max_train_windows: Optional[int] = None
    max_validation_windows: Optional[int] = 128


@dataclass
class ModelConfig:
    input_channels: int = 11
    output_channels: int = 3
    base_channels: int = 24
    depth: int = 4
    dropout: float = 0.0
    adapter_width: int = 8
    predict_delta: bool = True


@dataclass
class PhysicsConfig:
    enabled: bool = True
    dx_m: float = 147.302734375
    dy_m: float = 144.244140625
    # Raster rows run north to south while physical y/northing increases north.
    y_axis_sign: float = -1.0
    dt_s: float = 3600.0
    gravity: float = 9.81
    manning_n: float = 0.015
    dry_tolerance_m: float = 1.0e-4
    max_velocity_m_s: float = 10.0
    # Provisional weak priors for hourly point samples. Recalibrate from
    # conservative, sub-hourly training data before increasing them.
    mass_weight: float = 1.0e-6
    momentum_weight: float = 1.0e-6
    # Disabled by default: a closed-domain volume penalty is invalid on tiles
    # and at the open Gulf boundary unless boundary fluxes are supplied.
    global_mass_weight: float = 0.0
    mass_scale_m_s: float = 1.0e-5
    momentum_scale_m2_s2: float = 1.0e-3
    warmup_epochs: int = 2


@dataclass
class LossConfig:
    supervised_weight: float = 1.0
    depth_weight: float = 2.0
    momentum_weight: float = 1.0
    wet_threshold_m: float = 0.05
    # Keep the historical objective enabled by default.  Fair-comparison
    # campaigns can switch it off explicitly while retaining the threshold and
    # weight in the resolved configuration for complete provenance.
    inundation_enabled: bool = True
    inundation_weight: float = 0.1
    # The present mask marks a boundary face mapped to cells, not a verified
    # cell-center Dirichlet target, so this remains opt-in.
    ocean_stage_weight: float = 0.0


@dataclass
class TrainingConfig:
    seed: int = 2025
    device: str = "cuda"
    require_cuda: bool = True
    epochs: int = 40
    learning_rate: float = 2.0e-4
    weight_decay: float = 1.0e-5
    grad_clip_norm: float = 1.0
    detach_interval: int = 4
    mixed_precision: bool = False
    deterministic: bool = True
    adaptive_loss_balancing: bool = False
    loss_balance_decay: float = 0.95
    loss_weight_min: float = 1.0e-4
    loss_weight_max: float = 10.0
    checkpoint_dir: str = "runs/houston_pino"
    resume: Optional[str] = None
    save_every: int = 1
    validation_every: int = 1
    early_stopping_patience: int = 8
    minimum_relative_improvement: float = 1.0e-3


@dataclass
class FineTuneConfig:
    event_id: Optional[str] = None
    checkpoint: Optional[str] = None
    split: str = "test"
    mode: str = "adapter"
    observation_hours: int = 24
    epochs: int = 10
    learning_rate: float = 1.0e-3
    output_dir: str = "runs/event_adaptation"


@dataclass
class ExperimentConfig:
    data: DataConfig
    model: ModelConfig = field(default_factory=ModelConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    finetune: FineTuneConfig = field(default_factory=FineTuneConfig)
    experiment_name: str = "houston_pino"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _tuple_or_none(value: Any) -> Optional[Tuple[int, int]]:
    if value is None:
        return None
    if len(value) != 2:
        raise ValueError("tile_size must contain exactly two integers")
    return int(value[0]), int(value[1])


def _expand_path(value: Optional[str], base: Path) -> Optional[str]:
    if value is None:
        return None
    expanded = Path(os.path.expandvars(os.path.expanduser(value)))
    if not expanded.is_absolute():
        expanded = (base / expanded).resolve()
    return str(expanded)


def load_config(path: str | Path) -> ExperimentConfig:
    """Load YAML and resolve filesystem paths relative to the YAML file."""

    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    if "data" not in raw:
        raise ValueError("configuration must contain a data section")

    data_values = dict(raw["data"])
    data_values["tile_size"] = _tuple_or_none(data_values.get("tile_size", (128, 128)))
    data_values["tile_stride"] = _tuple_or_none(data_values.get("tile_stride"))
    base = config_path.parent
    for key in ("dataset_root", "split_csv", "normalization_json", "static_grid"):
        if key in data_values:
            data_values[key] = _expand_path(data_values[key], base)

    training_values = dict(raw.get("training", {}))
    if "checkpoint_dir" in training_values:
        training_values["checkpoint_dir"] = _expand_path(
            training_values["checkpoint_dir"], base
        )
    if training_values.get("resume"):
        training_values["resume"] = _expand_path(training_values["resume"], base)

    fine_values = dict(raw.get("finetune", {}))
    for key in ("checkpoint", "output_dir"):
        if fine_values.get(key):
            fine_values[key] = _expand_path(fine_values[key], base)

    return ExperimentConfig(
        data=DataConfig(**data_values),
        model=ModelConfig(**raw.get("model", {})),
        physics=PhysicsConfig(**raw.get("physics", {})),
        loss=LossConfig(**raw.get("loss", {})),
        training=TrainingConfig(**training_values),
        finetune=FineTuneConfig(**fine_values),
        experiment_name=str(raw.get("experiment_name", "houston_pino")),
    )
