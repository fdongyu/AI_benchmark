"""GPU-aware autoregressive training and event adaptation for Houston PINO."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence
import json
import math
import platform
import random
import shutil
import socket
import time
import warnings

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler, Sampler, Subset

from .checkpoint import load_checkpoint, save_checkpoint
from .config import ExperimentConfig
from .data import HoustonTrajectoryDataset, Normalization
from .features import denormalize_delta, make_features, normalize_delta
from .losses import (
    AdaptiveLossBalancer,
    ocean_stage_loss,
    soft_inundation_loss,
    supervised_state_loss,
)
from .metrics import (
    finalize_metric_statistics,
    merge_metric_statistics,
    metric_statistics,
)
from .model import HoustonPINO
from .physics import physics_loss, straight_through_wet_dry
from .utils import (
    git_revision,
    seed_everything,
    select_device,
    sha256_file,
    sha256_files,
    sha256_manifest_split,
    write_jsonl,
)


def build_model(config: ExperimentConfig) -> HoustonPINO:
    """Construct the configured grid operator."""

    values = config.model
    if not values.predict_delta:
        raise ValueError(
            "the Houston rollout currently requires model.predict_delta=true"
        )
    return HoustonPINO(
        in_channels=values.input_channels,
        out_channels=values.output_channels,
        base_channels=values.base_channels,
        depth=values.depth,
        dropout=values.dropout,
        adapter_width=values.adapter_width,
    )


def _validate_config(config: ExperimentConfig) -> None:
    if config.model.input_channels != 11 or config.model.output_channels != 3:
        raise ValueError(
            "Houston PINO requires 11 input and 3 conservative-state output channels"
        )
    if config.data.context_steps != 1:
        raise ValueError(
            "this operator consumes one state frame; set data.context_steps=1"
        )
    if config.data.forecast_steps < 1:
        raise ValueError("data.forecast_steps must be positive")
    if (
        config.data.train_windows_per_epoch is not None
        and config.data.train_windows_per_epoch < 1
    ):
        raise ValueError("data.train_windows_per_epoch must be positive or null")
    if config.data.batch_size < 1 or config.data.num_workers < 0:
        raise ValueError("data.batch_size must be positive and num_workers nonnegative")
    if config.data.train_split == config.data.validation_split:
        raise ValueError("data.train_split and data.validation_split must differ")
    if config.training.detach_interval < 1:
        raise ValueError("training.detach_interval must be positive")
    if config.training.epochs < 1 or config.finetune.epochs < 1:
        raise ValueError("training and fine-tune epochs must be positive")
    if config.training.save_every < 1 or config.training.validation_every < 1:
        raise ValueError("save_every and validation_every must be positive")
    if config.training.early_stopping_patience < 1:
        raise ValueError("early_stopping_patience must be positive")
    if config.training.grad_clip_norm <= 0.0:
        raise ValueError("grad_clip_norm must be positive")
    if not 0.0 <= config.training.minimum_relative_improvement < 1.0:
        raise ValueError("minimum_relative_improvement must be in [0, 1)")
    if config.physics.global_mass_weight != 0.0:
        raise ValueError(
            "global_mass_weight must remain zero until an open-boundary flux diagnostic "
            "is supplied; the local mass residual already includes rainfall"
        )
    if config.physics.y_axis_sign not in (-1.0, 1.0):
        raise ValueError("physics.y_axis_sign must be -1 or +1")
    if config.physics.max_velocity_m_s <= 0.0:
        raise ValueError("physics.max_velocity_m_s must be positive")
    for name, value in (
        ("physics.dx_m", config.physics.dx_m),
        ("physics.dy_m", config.physics.dy_m),
        ("physics.dt_s", config.physics.dt_s),
        ("physics.mass_scale_m_s", config.physics.mass_scale_m_s),
        ("physics.momentum_scale_m2_s2", config.physics.momentum_scale_m2_s2),
        ("training.learning_rate", config.training.learning_rate),
        ("finetune.learning_rate", config.finetune.learning_rate),
    ):
        if value <= 0.0:
            raise ValueError(f"{name} must be positive")
    for name, value in (
        ("loss.supervised_weight", config.loss.supervised_weight),
        ("loss.depth_weight", config.loss.depth_weight),
        ("loss.momentum_weight", config.loss.momentum_weight),
        ("loss.inundation_weight", config.loss.inundation_weight),
        ("loss.ocean_stage_weight", config.loss.ocean_stage_weight),
        ("physics.mass_weight", config.physics.mass_weight),
        ("physics.momentum_weight", config.physics.momentum_weight),
    ):
        if value < 0.0:
            raise ValueError(f"{name} must be nonnegative")


def _dataset(
    config: ExperimentConfig,
    split: str,
    *,
    event_ids: Optional[Iterable[str]] = None,
    observation_hours: Optional[float] = None,
    start_stride: Optional[int] = None,
) -> HoustonTrajectoryDataset:
    values = config.data
    dataset = HoustonTrajectoryDataset(
        split_manifest=values.split_csv,
        dataset_root=values.dataset_root,
        split=split,
        context_steps=values.context_steps,
        forecast_steps=values.forecast_steps,
        start_stride=values.stride if start_stride is None else start_stride,
        tile_size=values.tile_size,
        tile_stride=values.tile_stride,
        halo=values.halo,
        static_file=values.static_grid,
        event_ids=event_ids,
        observation_hours=observation_hours,
    )
    metadata = dataset.grid_metadata
    if metadata.get("dx_m") is not None:
        if not math.isclose(
            float(metadata["dx_m"]), config.physics.dx_m, rel_tol=0.0, abs_tol=1.0e-9
        ) or not math.isclose(
            float(metadata["dy_m"]), config.physics.dy_m, rel_tol=0.0, abs_tol=1.0e-9
        ):
            raise ValueError(
                "configured physics spacing does not match static-grid coordinates"
            )
        expected_y_sign = -1.0 if metadata.get("north_up") else 1.0
        if config.physics.y_axis_sign != expected_y_sign:
            raise ValueError(
                "configured physics.y_axis_sign contradicts static-grid orientation"
            )
    return dataset


def _normalization(config: ExperimentConfig) -> dict[str, Any]:
    normalization = Normalization.from_json(config.data.normalization_json)
    metadata = normalization.metadata
    source = str(metadata.get("source_split", "")).strip().lower()
    if source not in ("train", "train only"):
        raise ValueError(
            "normalization metadata must declare source_split as train or train only"
        )
    expected_binding = sha256_manifest_split(
        config.data.split_csv, config.data.train_split
    )
    binding = metadata.get("training_manifest_sha256")
    if binding is not None and str(binding) != expected_binding:
        raise ValueError(
            "normalization provenance does not match the current training manifest rows"
        )
    if binding is None:
        message = (
            "normalization metadata is not cryptographically bound to the training "
            "manifest; recompute it with training_manifest_sha256 before making "
            "held-out adaptation claims"
        )
        if config.data.require_normalization_provenance:
            raise ValueError(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)
    return normalization.as_dict()


def _limited_dataset(
    dataset: Dataset[Any], maximum: Optional[int], seed: int
) -> Dataset[Any]:
    if maximum is None or len(dataset) <= maximum:
        return dataset
    if maximum < 1:
        raise ValueError("maximum dataset windows must be positive when specified")
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:maximum].tolist()
    return Subset(dataset, indices)


def _event_balanced_validation_subset(
    dataset: HoustonTrajectoryDataset, maximum: Optional[int], seed: int
) -> Dataset[Any]:
    """Choose a fixed validation subset with nearly equal windows per event."""

    if maximum is None or len(dataset) <= maximum:
        return dataset
    if maximum < 1:
        raise ValueError("maximum validation windows must be positive when specified")
    grouped: dict[str, list[int]] = defaultdict(list)
    preceding = 0
    for trajectory, count in zip(dataset.trajectories, dataset._samples_per_trajectory):
        grouped[trajectory.event_id].extend(range(preceding, preceding + count))
        preceding += count
    generator = torch.Generator().manual_seed(seed)
    events = sorted(grouped)
    if maximum < len(events):
        raise ValueError(
            "max_validation_windows must be at least the number of validation events"
        )
    event_order = torch.randperm(len(events), generator=generator).tolist()
    base, remainder = divmod(maximum, len(events))
    selected: list[int] = []
    for position, event_index in enumerate(event_order):
        candidates = grouped[events[event_index]]
        quota = base + int(position < remainder)
        if quota > len(candidates):
            raise ValueError(
                "validation event has fewer windows than its balanced quota; "
                "reduce max_validation_windows"
            )
        order = torch.randperm(len(candidates), generator=generator)[:quota]
        selected.extend(candidates[index] for index in order.tolist())
    return Subset(dataset, selected)


def _adaptation_subsets(
    dataset: HoustonTrajectoryDataset,
    *,
    maximum_train: Optional[int],
    maximum_validation: Optional[int],
    seed: int,
) -> tuple[Dataset[Any], Dataset[Any]]:
    """Partition only the observed prefix for adaptation and model selection.

    Ensemble members are held out as complete validation groups whenever at
    least two are available.  This is stricter than a random window split and
    ensures that no post-observation target selects an adapted checkpoint.
    """

    trajectory_count = len(dataset.trajectories)
    generator = torch.Generator().manual_seed(seed)
    train_indices: list[int] = []
    validation_indices: list[int] = []
    if trajectory_count >= 2:
        order = torch.randperm(trajectory_count, generator=generator).tolist()
        validation_count = max(1, int(round(0.2 * trajectory_count)))
        validation_trajectories = set(order[:validation_count])
        for index in range(len(dataset)):
            trajectory_index, _, _ = dataset._locate(index)
            destination = (
                validation_indices
                if trajectory_index in validation_trajectories
                else train_indices
            )
            destination.append(index)
    else:
        # A single-member event cannot support member holdout. Use a blocked
        # temporal tail and leave a full window-span gap so adjacent sliding
        # windows cannot share context or targets across selection folds.
        located = [dataset._locate(index) for index in range(len(dataset))]
        starts = sorted({start for _, start, _ in located})
        validation_start_count = max(1, int(round(0.2 * len(starts))))
        validation_start_count = min(validation_start_count, len(starts) - 1)
        validation_starts = set(starts[-validation_start_count:])
        first_validation_start = min(validation_starts)
        window_span = dataset.context_steps + dataset.forecast_steps
        maximum_train_start = first_validation_start - window_span
        for index, (_, start, _) in enumerate(located):
            if start in validation_starts:
                validation_indices.append(index)
            elif start <= maximum_train_start:
                train_indices.append(index)
    if not train_indices or not validation_indices:
        raise ValueError(
            "the observed event prefix is too small to form adaptation train/validation sets"
        )
    train: Dataset[Any] = Subset(dataset, train_indices)
    validation: Dataset[Any] = Subset(dataset, validation_indices)
    return (
        _limited_dataset(train, maximum_train, seed + 1),
        _limited_dataset(validation, maximum_validation, seed + 2),
    )


def _forecast_origin_subset(
    dataset: HoustonTrajectoryDataset,
    observation_hours: float,
    maximum: Optional[int],
    seed: int,
) -> Dataset[Any]:
    """Select forecasts whose context ends at the last observed event frame.

    Selecting exactly one temporal origin per member/tile avoids evaluating
    adaptation with future target states used as later-window context.
    """

    selected: list[int] = []
    for index in range(len(dataset)):
        trajectory_index, start, _ = dataset._locate(index)
        trajectory = dataset.trajectories[trajectory_index]
        last_observed = int(
            math.floor(
                observation_hours * 3600.0 / trajectory.interval_seconds + 1.0e-9
            )
        )
        first_target = start + dataset.context_steps
        if first_target == last_observed + 1:
            selected.append(index)
    if not selected:
        raise ValueError(
            "no post-observation forecast origins are available; reduce observation_hours "
            "or forecast_steps"
        )
    subset: Dataset[Any] = Subset(dataset, selected)
    return _limited_dataset(subset, maximum, seed)


def _seed_worker(worker_id: int) -> None:
    del worker_id
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)


class EventBalancedSampler(Sampler[int]):
    """Sample events uniformly, then one member/window/tile within the event."""

    def __init__(
        self,
        dataset: HoustonTrajectoryDataset,
        num_samples: int,
        generator: torch.Generator,
    ) -> None:
        if num_samples < 1:
            raise ValueError("num_samples must be positive")
        self.dataset = dataset
        self.num_samples = int(num_samples)
        self.generator = generator
        groups: dict[str, list[int]] = {}
        for index, trajectory in enumerate(dataset.trajectories):
            groups.setdefault(trajectory.event_id, []).append(index)
        self.events = tuple(sorted(groups))
        self.trajectory_groups = tuple(tuple(groups[event]) for event in self.events)

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self):
        for _ in range(self.num_samples):
            event_index = int(
                torch.randint(len(self.events), (1,), generator=self.generator).item()
            )
            trajectories = self.trajectory_groups[event_index]
            group_index = int(
                torch.randint(len(trajectories), (1,), generator=self.generator).item()
            )
            trajectory_index = trajectories[group_index]
            preceding = (
                0
                if trajectory_index == 0
                else self.dataset._cumulative[trajectory_index - 1]
            )
            count = self.dataset._samples_per_trajectory[trajectory_index]
            local_index = int(
                torch.randint(count, (1,), generator=self.generator).item()
            )
            yield preceding + local_index


def _loader(
    dataset: Dataset[Any],
    config: ExperimentConfig,
    *,
    shuffle: bool,
    device: torch.device,
    seed_offset: int = 0,
    event_balanced_samples: Optional[int] = None,
) -> DataLoader[Any]:
    workers = config.data.num_workers
    # Keep sampling independent of DataLoader's worker-seeding stream.  With
    # persistent workers, the latter is consumed on process creation but not
    # on every subsequent epoch; sharing it would make a resumed run draw
    # different windows from an uninterrupted run.
    worker_generator = torch.Generator().manual_seed(config.training.seed + seed_offset)
    sampler_generator = torch.Generator().manual_seed(
        config.training.seed + seed_offset + 1_000_003
    )
    sampler = None
    if event_balanced_samples is not None:
        if not isinstance(dataset, HoustonTrajectoryDataset):
            raise TypeError("event-balanced sampling requires HoustonTrajectoryDataset")
        sampler = EventBalancedSampler(
            dataset, event_balanced_samples, sampler_generator
        )
        shuffle = False
    elif shuffle:
        # An explicit generator keeps shuffle order independent of worker
        # process creation and therefore restartable with persistent workers.
        sampler = RandomSampler(dataset, generator=sampler_generator)
        shuffle = False
    options: dict[str, Any] = {
        "batch_size": config.data.batch_size,
        "shuffle": shuffle,
        "num_workers": workers,
        "pin_memory": bool(config.data.pin_memory and device.type == "cuda"),
        "drop_last": False,
        "worker_init_fn": _seed_worker,
        "generator": worker_generator,
    }
    if sampler is not None:
        options["sampler"] = sampler
    if workers > 0:
        options["persistent_workers"] = True
    return DataLoader(dataset, **options)


def _move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def interior_evaluation_mask(batch: Mapping[str, Any]) -> torch.Tensor:
    """Return ``domain_mask`` restricted to each tile's non-halo valid interior."""

    domain = batch["domain_mask"]
    if domain.ndim != 4 or domain.shape[1] != 1:
        raise ValueError("domain_mask must have shape [B,1,H,W]")
    slices = batch.get("interior_slices")
    valid_shapes = batch.get("valid_interior_shape")
    if slices is None or valid_shapes is None:
        return domain
    if slices.ndim == 1:
        slices = slices.unsqueeze(0)
        valid_shapes = valid_shapes.unsqueeze(0)
    interior = torch.zeros_like(domain)
    for sample in range(domain.shape[0]):
        y0 = int(slices[sample, 0].item())
        x0 = int(slices[sample, 2].item())
        valid_h = int(valid_shapes[sample, 0].item())
        valid_w = int(valid_shapes[sample, 1].item())
        interior[sample, :, y0 : y0 + valid_h, x0 : x0 + valid_w] = 1
    return domain * interior


def _physics_evaluation_mask(
    evaluation_mask: torch.Tensor, ocean_boundary_mask: torch.Tensor
) -> torch.Tensor:
    # The open boundary has no exterior/ghost state in the training raster.
    # Remove it and one neighboring cell from PDE residual evaluation while
    # retaining those cells in the flux stencil and ocean-stage objective.
    neighborhood = F.max_pool2d(
        ocean_boundary_mask.to(evaluation_mask.dtype),
        kernel_size=3,
        stride=1,
        padding=1,
    )
    return evaluation_mask * (neighborhood < 0.5).to(evaluation_mask.dtype)


def _fully_wet_stencil_mask(
    state_t: torch.Tensor,
    state_tp1: torch.Tensor,
    domain_mask: torch.Tensor,
    dry_tolerance: float,
) -> torch.Tensor:
    """Exclude a one-cell shoreline band from momentum residual evaluation.

    The current centered pressure stencil is well balanced only when every
    neighboring cell is wet.  Continuity remains evaluated at wetting fronts.
    """

    midpoint_depth = 0.5 * (state_t[:, :1] + state_tp1[:, :1])
    wet = (midpoint_depth > dry_tolerance) & (domain_mask > 0.5)
    touches_dry = F.max_pool2d(
        (~wet).to(midpoint_depth.dtype), kernel_size=3, stride=1, padding=1
    )
    return (touches_dry < 0.5).to(midpoint_depth.dtype)


def _constant_dt(
    batch_dt: torch.Tensor, lead: int, expected_dt_s: Optional[float] = None
) -> float:
    values = batch_dt[:, lead].detach().to(dtype=torch.float64)
    first = float(values[0].item())
    if not torch.allclose(
        values, torch.full_like(values, first), rtol=0.0, atol=1.0e-6
    ):
        raise ValueError("a batch cannot mix output intervals for the SWE residual")
    if expected_dt_s is not None and not math.isclose(
        first, expected_dt_s, rel_tol=0.0, abs_tol=1.0e-6
    ):
        raise ValueError(
            f"trajectory interval {first} s differs from configured physics.dt_s "
            f"{expected_dt_s} s"
        )
    return first


def _objective_names(config: ExperimentConfig) -> list[str]:
    names = ["supervised"]
    if config.physics.enabled and (
        config.physics.mass_weight > 0.0 or config.physics.momentum_weight > 0.0
    ):
        names.append("physics")
    return names


def _loss_balancer(
    config: ExperimentConfig, device: torch.device
) -> Optional[AdaptiveLossBalancer]:
    names = _objective_names(config)
    if not config.training.adaptive_loss_balancing or len(names) == 1:
        return None
    return AdaptiveLossBalancer(
        names,
        decay=config.training.loss_balance_decay,
        minimum=config.training.loss_weight_min,
        maximum=config.training.loss_weight_max,
    ).to(device)


def _combine_objectives(
    objectives: Mapping[str, torch.Tensor],
    balancer: Optional[AdaptiveLossBalancer],
    *,
    update: bool,
    multipliers: Optional[Mapping[str, float]] = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    factors = multipliers or {}
    if balancer is None:
        weights = {name: value.new_ones(()) for name, value in objectives.items()}
    else:
        if update:
            balancer.update(objectives)
        weights = balancer.current_weights()
    effective = {
        name: weight * float(factors.get(name, 1.0)) for name, weight in weights.items()
    }
    total = sum(effective[name] * objectives[name] for name in objectives)
    return total, effective


def _single_step(
    model: HoustonPINO,
    current: torch.Tensor,
    target: torch.Tensor,
    rainfall: torch.Tensor,
    ocean_height: torch.Tensor,
    bed: torch.Tensor,
    domain: torch.Tensor,
    ocean_mask: torch.Tensor,
    evaluation_mask: torch.Tensor,
    normalization: Mapping[str, Any],
    config: ExperimentConfig,
    dt_s: float,
    epoch: int,
    balancer: Optional[AdaptiveLossBalancer],
    update_balancer: bool,
    amp_enabled: bool,
    tile_origin: Optional[torch.Tensor] = None,
    global_shape: Optional[torch.Tensor] = None,
    halo: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    manning = torch.full_like(domain, config.physics.manning_n)
    with torch.autocast(
        device_type=current.device.type,
        dtype=torch.float16,
        enabled=amp_enabled,
    ):
        features = make_features(
            current,
            rainfall,
            ocean_height,
            bed,
            domain,
            ocean_mask,
            normalization,
            manning=manning,
            tile_origin=tile_origin,
            global_shape=global_shape,
            halo=halo,
        )
        normalized_delta = model(features)
        proposal = current + denormalize_delta(normalized_delta, normalization)

    predicted = straight_through_wet_dry(
        proposal.float(),
        domain,
        dry_tolerance=config.physics.dry_tolerance_m,
        max_velocity=config.physics.max_velocity_m_s,
    )
    predicted_delta = normalize_delta(predicted - current, normalization)
    target_delta = normalize_delta(target - current, normalization)
    data_terms = supervised_state_loss(
        predicted_delta,
        target_delta,
        evaluation_mask,
        depth_weight=config.loss.depth_weight,
        momentum_weight=config.loss.momentum_weight,
    )
    if config.loss.inundation_enabled and config.loss.inundation_weight > 0.0:
        inundation = soft_inundation_loss(
            predicted[:, :1],
            target[:, :1],
            evaluation_mask,
            threshold_m=config.loss.wet_threshold_m,
        )
    else:
        # Do not merely multiply the term by zero: skipping its construction
        # makes a supervised-only protocol unambiguous and avoids retaining an
        # unnecessary autograd branch.  Keep a zero diagnostic so history
        # schemas remain compatible with existing analysis code.
        inundation = predicted.new_zeros(())
    ocean = ocean_stage_loss(
        predicted,
        bed,
        ocean_height,
        ocean_mask * evaluation_mask,
    )
    supervised = config.loss.supervised_weight * (
        data_terms["depth"] + data_terms["momentum_data"]
    )
    supervised = (
        supervised
        + config.loss.inundation_weight * inundation
        + config.loss.ocean_stage_weight * ocean
    )
    objectives: dict[str, torch.Tensor] = {"supervised": supervised}
    diagnostics: dict[str, torch.Tensor] = {
        **data_terms,
        "inundation": inundation,
        "ocean_stage": ocean,
        "supervised": supervised,
    }

    if "physics" in _objective_names(config):
        physics_mask = _physics_evaluation_mask(evaluation_mask, ocean_mask)
        physics = physics_loss(
            current.float(),
            predicted.float(),
            rainfall.float(),
            bed.float(),
            manning.float(),
            domain,
            dt_s,
            config.physics.dx_m,
            config.physics.dy_m,
            evaluation_mask=physics_mask,
            momentum_evaluation_mask=_fully_wet_stencil_mask(
                current,
                predicted,
                domain,
                config.physics.dry_tolerance_m,
            ),
            gravity=config.physics.gravity,
            dry_tolerance=config.physics.dry_tolerance_m,
            y_axis_sign=config.physics.y_axis_sign,
            component_weights=(1.0, 1.0, 1.0),
            component_scales=(
                config.physics.mass_scale_m_s,
                config.physics.momentum_scale_m2_s2,
                config.physics.momentum_scale_m2_s2,
            ),
            positivity_weight=0.0,
            dry_momentum_weight=0.0,
            project_states=False,
            return_components=True,
        )
        warmup = max(config.physics.warmup_epochs, 1)
        ramp = min(1.0, float(epoch + 1) / float(warmup))
        physics_objective = config.physics.mass_weight * physics[
            "mass"
        ] + config.physics.momentum_weight * (
            physics["momentum_x"] + physics["momentum_y"]
        )
        objectives["physics"] = physics_objective
        diagnostics.update(
            {
                "physics": ramp * physics_objective,
                "physics_ramp": physics_objective.new_tensor(ramp),
                "physics_mass": physics["mass"],
                "physics_momentum_x": physics["momentum_x"],
                "physics_momentum_y": physics["momentum_y"],
            }
        )

    multipliers = {"physics": ramp} if "physics" in objectives else None
    total, weights = _combine_objectives(
        objectives,
        balancer,
        update=update_balancer,
        multipliers=multipliers,
    )
    diagnostics["objective"] = total
    diagnostics.update({f"weight_{name}": value for name, value in weights.items()})
    return predicted, total, diagnostics


def _run_batch(
    model: HoustonPINO,
    batch: Mapping[str, Any],
    normalization: Mapping[str, Any],
    config: ExperimentConfig,
    epoch: int,
    balancer: Optional[AdaptiveLossBalancer],
    *,
    training: bool,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> tuple[dict[str, float], int, list[dict[str, Any]]]:
    if training and (optimizer is None or scaler is None):
        raise ValueError("training batches require optimizer and gradient scaler")
    current = batch["current_state"].float()
    targets = batch["future_state"].float()
    rainfall = batch["rainfall"].float()
    ocean_height = batch["ocean_boundary_height"].float()
    bed = batch["bed_elevation"].float()
    domain = batch["domain_mask"]
    ocean_mask = batch["ocean_boundary_mask"]
    evaluation_mask = interior_evaluation_mask(batch)
    amp_enabled = bool(
        config.training.mixed_precision and current.device.type == "cuda"
    )
    forecast_steps = targets.shape[1]
    totals: defaultdict[str, float] = defaultdict(float)
    metric_records: list[dict[str, Any]] = []
    chunk: Optional[torch.Tensor] = None
    event_ids = batch["event_id"]
    if isinstance(event_ids, str):
        event_ids = [event_ids]

    if training:
        optimizer.zero_grad(set_to_none=True)
    grad_context = torch.enable_grad() if training else torch.no_grad()
    with grad_context:
        for lead in range(forecast_steps):
            predicted, objective, diagnostics = _single_step(
                model,
                current,
                targets[:, lead],
                rainfall[:, lead],
                ocean_height[:, lead],
                bed,
                domain,
                ocean_mask,
                evaluation_mask,
                normalization,
                config,
                _constant_dt(batch["dt_seconds"], lead, config.physics.dt_s),
                epoch,
                balancer,
                update_balancer=training,
                amp_enabled=amp_enabled,
                tile_origin=batch.get("tile_origin"),
                global_shape=batch.get("global_shape"),
                halo=batch.get("halo"),
            )
            if not torch.isfinite(objective):
                values = {
                    name: float(value.detach().cpu())
                    for name, value in diagnostics.items()
                }
                raise FloatingPointError(
                    f"non-finite rollout objective at lead {lead}: {values}"
                )

            if training:
                scaled_objective = objective / float(forecast_steps)
                chunk = scaled_objective if chunk is None else chunk + scaled_objective
                end_chunk = (
                    (lead + 1) % config.training.detach_interval == 0
                    or lead + 1 == forecast_steps
                )
                if end_chunk:
                    scaler.scale(chunk).backward()
                    chunk = None
                    predicted = predicted.detach()

            for name, value in diagnostics.items():
                totals[name] += float(value.detach().cpu())
            step_statistics = metric_statistics(
                predicted,
                targets[:, lead],
                evaluation_mask,
                thresholds_m=(config.loss.wet_threshold_m, 0.30),
            )
            for sample, statistics in enumerate(step_statistics):
                metric_records.append(
                    {
                        "event_id": str(event_ids[sample]),
                        "lead": lead + 1,
                        "statistics": statistics,
                    }
                )
            current = predicted

    if training:
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            config.training.grad_clip_norm,
        )
        scaler.step(optimizer)
        scaler.update()
    return (
        {name: value / forecast_steps for name, value in totals.items()},
        current.shape[0],
        metric_records,
    )


def _run_epoch(
    model: HoustonPINO,
    loader: DataLoader[Any],
    normalization: Mapping[str, Any],
    config: ExperimentConfig,
    device: torch.device,
    epoch: int,
    balancer: Optional[AdaptiveLossBalancer],
    *,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    totals: defaultdict[str, float] = defaultdict(float)
    pooled_statistics: dict[str, float] = {}
    statistics_by_lead: defaultdict[int, dict[str, float]] = defaultdict(dict)
    statistics_by_event: defaultdict[str, dict[str, float]] = defaultdict(dict)
    statistics_by_event_lead: defaultdict[str, defaultdict[int, dict[str, float]]] = (
        defaultdict(lambda: defaultdict(dict))
    )
    samples = 0
    started = time.monotonic()
    for raw_batch in loader:
        batch = _move_batch(raw_batch, device)
        metrics, batch_size, metric_records = _run_batch(
            model,
            batch,
            normalization,
            config,
            epoch,
            balancer,
            training=training,
            optimizer=optimizer,
            scaler=scaler,
        )
        for name, value in metrics.items():
            totals[name] += value * batch_size
        for record in metric_records:
            event = record["event_id"]
            lead = int(record["lead"])
            statistics = record["statistics"]
            merge_metric_statistics(pooled_statistics, statistics)
            merge_metric_statistics(statistics_by_lead[lead], statistics)
            if not training:
                merge_metric_statistics(statistics_by_event[event], statistics)
                merge_metric_statistics(
                    statistics_by_event_lead[event][lead], statistics
                )
        samples += batch_size
    if samples == 0:
        raise RuntimeError("data loader produced no samples")
    result = {name: value / samples for name, value in totals.items()}
    result.update(finalize_metric_statistics(pooled_statistics))
    result["metrics_by_lead"] = {
        str(lead): finalize_metric_statistics(statistics)
        for lead, statistics in sorted(statistics_by_lead.items())
    }
    if not training:
        result["metrics_by_event"] = {
            event: {
                "overall": finalize_metric_statistics(statistics),
                "by_lead": {
                    str(lead): finalize_metric_statistics(lead_statistics)
                    for lead, lead_statistics in sorted(
                        statistics_by_event_lead[event].items()
                    )
                },
            }
            for event, statistics in sorted(statistics_by_event.items())
        }
    result["samples"] = float(samples)
    result["seconds"] = time.monotonic() - started
    return result


def _operator_source_sha256() -> str:
    package = Path(__file__).resolve().parent
    return sha256_files(
        [
            package / "config.py",
            package / "features.py",
            package / "model.py",
            package / "physics.py",
        ]
    )


def _provenance(config: ExperimentConfig) -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[2]
    package = Path(__file__).resolve().parent
    static_path = config.data.static_grid or str(
        Path(config.data.dataset_root) / "static_grid.h5"
    )
    return {
        "git_revision": git_revision(repository),
        "hostname": socket.gethostname(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_device": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "experiment": config.experiment_name,
        "source_sha256": sha256_files(list(package.glob("*.py"))),
        "operator_source_sha256": _operator_source_sha256(),
        "manifest_sha256": sha256_file(config.data.split_csv),
        "training_manifest_sha256": sha256_manifest_split(
            config.data.split_csv, config.data.train_split
        ),
        "normalization_sha256": sha256_file(config.data.normalization_json),
        "static_grid_sha256": sha256_file(static_path),
    }


def _write_resolved_config(output_dir: Path, config: ExperimentConfig) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "resolved_config.json"
    temporary = destination.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(config.to_dict(), stream, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(destination)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(dict(payload), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def _resume_comparable_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Remove only fields that legitimately change when extending a run."""

    comparable = json.loads(json.dumps(dict(config)))
    training = comparable.get("training", {})
    for key in ("epochs", "resume", "checkpoint_dir"):
        training.pop(key, None)
    return comparable


def _curriculum_comparable_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Remove only fields that are allowed to change between curriculum stages."""

    comparable = json.loads(json.dumps(dict(config)))
    comparable.pop("experiment_name", None)
    data = comparable.get("data", {})
    for key in (
        "forecast_steps",
        "train_windows_per_epoch",
        "max_validation_windows",
    ):
        data.pop(key, None)
    training = comparable.get("training", {})
    for key in ("checkpoint_dir", "resume"):
        training.pop(key, None)
    return comparable


def _validate_resume_payload(
    payload: Mapping[str, Any],
    config: ExperimentConfig,
    normalization: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> None:
    if payload.get("normalization") != normalization:
        raise ValueError("resume checkpoint normalization differs from this experiment")
    previous = payload.get("config")
    if previous is None or _resume_comparable_config(
        previous
    ) != _resume_comparable_config(config.to_dict()):
        raise ValueError(
            "resume checkpoint configuration differs (only epochs, resume, and "
            "checkpoint_dir may change)"
        )
    previous_provenance = payload.get("provenance", {})
    for key in (
        "source_sha256",
        "operator_source_sha256",
        "manifest_sha256",
        "training_manifest_sha256",
        "normalization_sha256",
        "static_grid_sha256",
    ):
        if previous_provenance.get(key) != provenance.get(key):
            raise ValueError(f"resume checkpoint provenance mismatch for {key}")

    # Curriculum controls are deliberately kept outside ExperimentConfig so
    # adding staged training does not change the operator-source hash of the
    # already published epoch-79 checkpoint.  When they are present, however,
    # an exact restart must retain them.
    if (
        previous_provenance.get("workflow") == "curriculum_training"
        or provenance.get("workflow") == "curriculum_training"
    ):
        for key in (
            "curriculum_stage_hours",
            "forecast_steps_per_epoch",
            "validation_forecast_steps",
            "minimum_epochs",
            "require_convergence",
            "initialization_policy",
            "initial_checkpoint_sha256",
        ):
            if previous_provenance.get(key) != provenance.get(key):
                raise ValueError(f"resume checkpoint provenance mismatch for {key}")


def _validate_initial_checkpoint_payload(
    payload: Mapping[str, Any],
    config: ExperimentConfig,
    normalization: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> None:
    """Validate a model-only stage transition before optimizer creation.

    A curriculum transition intentionally changes the autoregressive horizon,
    sampler exposure, validation exposure, experiment name, and output path.
    It must not change the model, physical problem, loss, normalization, or
    training population.  Optimizer, scheduler, scaler, RNG, and early-stop
    state are intentionally *not* restored here.
    """

    if payload.get("normalization") != normalization:
        raise ValueError(
            "initial checkpoint normalization differs from this curriculum stage"
        )
    previous = payload.get("config")
    if not isinstance(previous, Mapping):
        raise ValueError("initial checkpoint lacks a resolved configuration")
    current = config.to_dict()
    if _curriculum_comparable_config(previous) != _curriculum_comparable_config(
        current
    ):
        raise ValueError(
            "initial checkpoint configuration differs from this curriculum stage "
            "outside the allowed experiment name, horizon/exposure, output, and "
            "resume fields"
        )
    previous_data = previous.get("data", {})
    current_data = current.get("data", {})
    previous_provenance = payload.get("provenance", {})
    for key in (
        "source_sha256",
        "operator_source_sha256",
        "manifest_sha256",
        "training_manifest_sha256",
        "normalization_sha256",
        "static_grid_sha256",
    ):
        if previous_provenance.get(key) != provenance.get(key):
            raise ValueError(f"initial checkpoint provenance mismatch for {key}")
    previous_events = previous_provenance.get("training_event_ids")
    current_events = provenance.get("training_event_ids")
    if previous_events is None or previous_events != current_events:
        raise ValueError(
            "initial checkpoint training events differ from this curriculum stage"
        )

    current_hours = int(current_data.get("forecast_steps", -1))
    previous_hours_by_stage = {8: 4, 16: 8, 32: 16}
    expected_previous_hours = previous_hours_by_stage.get(current_hours)
    if expected_previous_hours is None:
        raise ValueError(
            "a model-only curriculum warm start is valid only for AR8, AR16, or AR32"
        )
    if int(previous_data.get("forecast_steps", -1)) != expected_previous_hours:
        raise ValueError(
            "initial checkpoint is not from the immediately preceding curriculum horizon"
        )
    if previous_provenance.get("workflow") != "curriculum_training":
        raise ValueError("initial checkpoint is not a curriculum-training checkpoint")
    if provenance.get("workflow") != "curriculum_training":
        raise ValueError("current experiment is not marked as curriculum training")
    if (
        int(previous_provenance.get("curriculum_stage_hours", -1))
        != expected_previous_hours
    ):
        raise ValueError("initial checkpoint curriculum-stage provenance is invalid")

    for key in (
        "forecast_steps_per_epoch",
        "validation_forecast_steps",
        "minimum_epochs",
        "require_convergence",
    ):
        if previous_provenance.get(key) != provenance.get(key):
            raise ValueError(
                f"initial checkpoint curriculum provenance mismatch for {key}"
            )

    previous_train_windows = previous_data.get("train_windows_per_epoch")
    previous_validation_windows = previous_data.get("max_validation_windows")
    if (
        previous_train_windows is None
        or expected_previous_hours * int(previous_train_windows)
        != int(previous_provenance.get("forecast_steps_per_epoch", -1))
    ):
        raise ValueError("initial checkpoint training exposure is inconsistent")
    if (
        previous_validation_windows is None
        or expected_previous_hours * int(previous_validation_windows)
        != int(previous_provenance.get("validation_forecast_steps", -1))
    ):
        raise ValueError("initial checkpoint validation exposure is inconsistent")

    expected_policy = (
        "random_seeded_cold_start"
        if expected_previous_hours == 4
        else "preceding_curriculum_stage_best_checkpoint"
    )
    if previous_provenance.get("initialization_policy") != expected_policy:
        raise ValueError("initial checkpoint initialization-policy provenance is invalid")


def _update_validation_state(
    *,
    score: float,
    best: float,
    best_epoch: int,
    stale: int,
    epoch: int,
    minimum_relative_improvement: float,
) -> tuple[float, int, int, bool, bool]:
    """Update absolute checkpoint selection and plateau state independently."""

    finite = math.isfinite(score)
    threshold = best * (1.0 - minimum_relative_improvement)
    significant = finite and (not math.isfinite(best) or score < threshold)
    selected = finite and (not math.isfinite(best) or score < best)
    if selected:
        best = score
        best_epoch = epoch
    stale = 0 if significant else stale + 1
    return best, best_epoch, stale, selected, significant


def _plateau_converged(
    epochs_completed: int,
    stale: int,
    minimum_epochs: int,
    patience: int,
) -> bool:
    return epochs_completed >= minimum_epochs and stale >= patience


def _capture_training_state(
    *,
    best: float,
    best_epoch: int,
    stale: int,
    train_loader: DataLoader[Any],
    validation_loader: DataLoader[Any],
) -> dict[str, Any]:
    state: dict[str, Any] = {
        "best_validation_score": float(best),
        "checkpoint_selection_metric": "supervised",
        "best_validation_epoch": int(best_epoch),
        "early_stopping_stale_epochs": int(stale),
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
        "train_loader_generator_state": train_loader.generator.get_state(),
        "validation_loader_generator_state": validation_loader.generator.get_state(),
    }
    if torch.cuda.is_available():
        state["cuda_random_state_all"] = torch.cuda.get_rng_state_all()
    if isinstance(train_loader.sampler, (EventBalancedSampler, RandomSampler)):
        state["train_sampler_generator_state"] = (
            train_loader.sampler.generator.get_state()
        )
    return state


def _restore_training_state(
    state: Mapping[str, Any],
    *,
    train_loader: DataLoader[Any],
    validation_loader: DataLoader[Any],
) -> tuple[float, int, int]:
    if not state:
        raise ValueError(
            "resume checkpoint lacks RNG/early-stopping state required for an exact restart"
        )
    random.setstate(state["python_random_state"])
    np.random.set_state(state["numpy_random_state"])
    torch.set_rng_state(state["torch_random_state"].cpu())
    if torch.cuda.is_available() and "cuda_random_state_all" in state:
        torch.cuda.set_rng_state_all(
            [value.cpu() for value in state["cuda_random_state_all"]]
        )
    train_loader.generator.set_state(state["train_loader_generator_state"].cpu())
    validation_loader.generator.set_state(
        state["validation_loader_generator_state"].cpu()
    )
    if isinstance(train_loader.sampler, (EventBalancedSampler, RandomSampler)):
        sampler_state = state.get("train_sampler_generator_state")
        if sampler_state is None:
            raise ValueError(
                "resume checkpoint lacks event-sampler RNG state required for an "
                "exact restart"
            )
        train_loader.sampler.generator.set_state(sampler_state.cpu())
    return (
        float(state["best_validation_score"]),
        int(state["best_validation_epoch"]),
        int(state["early_stopping_stale_epochs"]),
    )


def _checkpoint_matches_selection(path: Path, best: float, best_epoch: int) -> bool:
    """Check that a checkpoint is the epoch selected by early stopping."""

    if not path.is_file():
        return False
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metrics = payload.get("metrics", {})
    validation = metrics.get("validation", {})
    return (
        int(payload.get("epoch", -1)) == best_epoch
        and metrics.get("checkpoint_selection_metric") == "supervised"
        and float(metrics.get("best_validation_score", math.inf)) == best
        and float(validation.get("supervised", math.inf)) == best
    )


def _atomic_copy(source: Path, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copyfile(source, temporary)
    temporary.replace(destination)


def _fit(
    *,
    model: HoustonPINO,
    train_loader: DataLoader[Any],
    validation_loader: DataLoader[Any],
    normalization: Mapping[str, Any],
    config: ExperimentConfig,
    device: torch.device,
    output_dir: Path,
    epochs: int,
    learning_rate: float,
    provenance: Mapping[str, Any],
    resume: Optional[str] = None,
    minimum_epochs: int = 1,
    require_convergence: bool = False,
) -> Path:
    output_dir = output_dir.resolve()
    if minimum_epochs < 1 or minimum_epochs > epochs:
        raise ValueError("minimum_epochs must be between one and training.epochs")
    best_path = output_dir / "best.pt"
    managed_artifacts = [
        output_dir / "last.pt",
        best_path,
        output_dir / "history.jsonl",
        output_dir / "resolved_config.json",
    ]
    managed_artifacts.extend(output_dir.glob("epoch_*.pt"))
    if resume is None and any(path.exists() for path in managed_artifacts):
        raise FileExistsError(
            f"refusing to mix a fresh run with existing artifacts in {output_dir}; "
            "choose a new checkpoint_dir or resume last.pt"
        )
    if resume is not None:
        resume_path = Path(resume).resolve()
        if resume_path.parent != output_dir or resume_path.name != "last.pt":
            raise ValueError(
                "resume must point to checkpoint_dir/last.pt so history and best "
                "checkpoint remain unambiguous"
            )

    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not parameters:
        raise ValueError("no trainable parameters remain after applying fine-tune mode")
    optimizer = torch.optim.AdamW(
        parameters,
        lr=learning_rate,
        weight_decay=config.training.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )
    balancer = _loss_balancer(config, device)
    amp_enabled = bool(config.training.mixed_precision and device.type == "cuda")
    if hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    else:  # pragma: no cover - compatibility with older supported PyTorch.
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    start_epoch = 0
    best = math.inf
    best_epoch = 0
    stale = 0
    history_records: list[dict[str, Any]] = []
    if resume:
        payload = load_checkpoint(
            resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_balancer=balancer,
            scaler=scaler,
            map_location=device,
        )
        _validate_resume_payload(payload, config, normalization, provenance)
        start_epoch = int(payload["epoch"])
        best, best_epoch, stale = _restore_training_state(
            payload.get("training_state", {}),
            train_loader=train_loader,
            validation_loader=validation_loader,
        )
        history_path = output_dir / "history.jsonl"
        if not history_path.is_file():
            raise FileNotFoundError("resume directory is missing history.jsonl")
        with history_path.open("r", encoding="utf-8") as stream:
            history_records = [json.loads(line) for line in stream if line.strip()]
        committed_history = [
            record for record in history_records if int(record["epoch"]) <= start_epoch
        ]
        committed_epochs = [int(record["epoch"]) for record in committed_history]
        if committed_epochs != list(range(1, start_epoch + 1)):
            raise ValueError("last.pt epoch does not match the final history record")
        if len(committed_history) != len(history_records):
            # History is staged before last.pt is atomically committed. A
            # preemption in that interval leaves a safe-to-remove tail.
            write_jsonl(history_path, committed_history)
        history_records = committed_history
        if math.isfinite(best) and not _checkpoint_matches_selection(
            best_path, best, best_epoch
        ):
            candidates = []
            if best_epoch == start_epoch:
                candidates.append(resume_path)
            candidates.append(output_dir / f"epoch_{best_epoch:04d}.pt")
            source = next(
                (
                    candidate
                    for candidate in candidates
                    if _checkpoint_matches_selection(candidate, best, best_epoch)
                ),
                None,
            )
            if source is None:
                raise ValueError(
                    "selected best checkpoint is stale and cannot be recovered from "
                    "the committed restart or epoch snapshot"
                )
            _atomic_copy(source, best_path)

    if resume:
        _write_json(
            output_dir / f"resume_config_epoch_{start_epoch:04d}.json",
            config.to_dict(),
        )
    else:
        _write_resolved_config(output_dir, config)
    history = output_dir / "history.jsonl"
    if resume and _plateau_converged(
        start_epoch,
        stale,
        minimum_epochs,
        config.training.early_stopping_patience,
    ):
        if not best_path.is_file():
            raise RuntimeError("the converged resumed run has no best checkpoint")
        return best_path
    if start_epoch >= epochs:
        if not best_path.is_file():
            raise RuntimeError("the resumed run has no validated best checkpoint")
        if require_convergence:
            raise RuntimeError(
                "training reached its epoch limit without satisfying the configured "
                "validation-plateau convergence rule"
            )
        return best_path
    for epoch in range(start_epoch, epochs):
        train_metrics = _run_epoch(
            model,
            train_loader,
            normalization,
            config,
            device,
            epoch,
            balancer,
            optimizer=optimizer,
            scaler=scaler,
        )
        should_validate = (
            epoch + 1
        ) % config.training.validation_every == 0 or epoch + 1 == epochs
        validation_metrics: dict[str, float] = {}
        if should_validate:
            validation_metrics = _run_epoch(
                model,
                validation_loader,
                normalization,
                config,
                device,
                epoch,
                balancer,
            )
            scheduler.step(validation_metrics["supervised"])
        # Supervised validation is stationary across epochs. The combined
        # objective is not: physics warmup (and optional adaptive balancing)
        # changes its scale, so it must not select checkpoints.
        score = validation_metrics.get("supervised", math.inf)
        selected = False
        if should_validate:
            best, best_epoch, stale, selected, _ = _update_validation_state(
                score=score,
                best=best,
                best_epoch=best_epoch,
                stale=stale,
                epoch=epoch + 1,
                minimum_relative_improvement=(
                    config.training.minimum_relative_improvement
                ),
            )

        record: dict[str, Any] = {
            "epoch": epoch + 1,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "validation": validation_metrics,
            "checkpoint_selection_metric": "supervised",
            "best_validation_score": best if math.isfinite(best) else None,
        }
        history_records.append(record)
        checkpoint_metrics = {**record, "best_validation_score": best}
        common = {
            "model": model,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "epoch": epoch + 1,
            "config": config.to_dict(),
            "normalization": normalization,
            "metrics": checkpoint_metrics,
            "provenance": dict(provenance),
            "loss_balancer": balancer,
            "scaler": scaler,
            "training_state": _capture_training_state(
                best=best,
                best_epoch=best_epoch,
                stale=stale,
                train_loader=train_loader,
                validation_loader=validation_loader,
            ),
        }
        if (epoch + 1) % config.training.save_every == 0:
            save_checkpoint(output_dir / f"epoch_{epoch + 1:04d}.pt", **common)
        # History is staged first and last.pt is the atomic commit marker.
        write_jsonl(history, history_records)
        save_checkpoint(output_dir / "last.pt", **common)
        if selected:
            _atomic_copy(output_dir / "last.pt", best_path)
        print(json.dumps(record, sort_keys=True), flush=True)
        if _plateau_converged(
            epoch + 1,
            stale,
            minimum_epochs,
            config.training.early_stopping_patience,
        ):
            break
    if not best_path.is_file():
        raise RuntimeError("training completed without a finite validation objective")
    if require_convergence and not _plateau_converged(
        len(history_records),
        stale,
        minimum_epochs,
        config.training.early_stopping_patience,
    ):
        raise RuntimeError(
            "training reached its epoch limit without satisfying the configured "
            "validation-plateau convergence rule"
        )
    return best_path


def train_experiment(
    config: ExperimentConfig,
    *,
    initial_checkpoint: Optional[str | Path] = None,
    initialization_policy: Optional[str] = None,
    minimum_epochs: int = 1,
    require_convergence: bool = False,
) -> Path:
    """Train a multi-event Houston model and return its best checkpoint.

    ``initial_checkpoint`` is a model-only warm start for a new curriculum
    stage.  It is intentionally different from ``training.resume``: a resume
    restores every optimization and RNG state in the same output directory,
    while a stage transition restores only model weights and starts a fresh
    optimizer, scheduler, sampler stream, and early-stopping record.
    """

    _validate_config(config)
    seed_everything(config.training.seed, config.training.deterministic)
    device = select_device(config.training.device, config.training.require_cuda)
    normalization = _normalization(config)
    train_dataset = _dataset(config, config.data.train_split)
    validation_dataset = _dataset(config, config.data.validation_split)
    train_data: Dataset[Any] = train_dataset
    validation_data = _event_balanced_validation_subset(
        validation_dataset,
        config.data.max_validation_windows,
        config.training.seed + 1,
    )
    model = build_model(config).to(device)
    # Keep the event adapter at its neutral zero initialization while fitting
    # the shared operator.  Fine-tuning then learns only an event correction.
    model.freeze_adapter()
    output_dir = Path(config.training.checkpoint_dir)
    provenance = _provenance(config)
    provenance.update(
        {
            "workflow": (
                "curriculum_training"
                if initialization_policy is not None
                else "base_training"
            ),
            "train_split": config.data.train_split,
            "validation_split": config.data.validation_split,
            "training_event_ids": sorted(
                {trajectory.event_id for trajectory in train_dataset.trajectories}
            ),
            "train_windows_per_epoch": config.data.train_windows_per_epoch,
            "curriculum_stage_hours": (
                config.data.forecast_steps
                if initialization_policy is not None
                else None
            ),
            "forecast_steps_per_epoch": (
                config.data.forecast_steps * config.data.train_windows_per_epoch
                if config.data.train_windows_per_epoch is not None
                else None
            ),
            "validation_forecast_steps": (
                config.data.forecast_steps * config.data.max_validation_windows
                if config.data.max_validation_windows is not None
                else None
            ),
            "minimum_epochs": int(minimum_epochs),
            "require_convergence": bool(require_convergence),
            "initialization_policy": initialization_policy,
        }
    )
    if initial_checkpoint is not None:
        initial_path = Path(initial_checkpoint).expanduser().resolve()
        if not initial_path.is_file():
            raise FileNotFoundError(
                f"initial curriculum checkpoint does not exist: {initial_path}"
            )
        if initial_path.parent == output_dir.resolve():
            raise ValueError(
                "initial curriculum checkpoint must come from a preceding stage"
            )
        provenance.update(
            {
                "initial_checkpoint": str(initial_path),
                "initial_checkpoint_sha256": sha256_file(initial_path),
            }
        )
        if config.training.resume:
            payload = torch.load(
                initial_path, map_location="cpu", weights_only=False
            )
        else:
            payload = load_checkpoint(
                initial_path,
                model=model,
                map_location=device,
            )
        _validate_initial_checkpoint_payload(
            payload, config, normalization, provenance
        )
        provenance.update(
            {
                "initial_checkpoint_epoch": int(payload.get("epoch", -1)),
                "initial_experiment": payload.get("config", {}).get(
                    "experiment_name"
                ),
                "initial_forecast_steps": payload.get("config", {})
                .get("data", {})
                .get("forecast_steps"),
            }
        )
    elif initialization_policy not in (None, "random_seeded_cold_start"):
        raise ValueError(
            "a non-cold-start initialization policy requires initial_checkpoint"
        )
    else:
        provenance.update(
            {
                "initial_checkpoint": None,
                "initial_checkpoint_sha256": None,
                "initial_checkpoint_epoch": None,
                "initial_experiment": None,
                "initial_forecast_steps": None,
            }
        )
    return _fit(
        model=model,
        train_loader=_loader(
            train_data,
            config,
            shuffle=config.data.train_windows_per_epoch is None,
            device=device,
            event_balanced_samples=config.data.train_windows_per_epoch,
        ),
        validation_loader=_loader(
            validation_data, config, shuffle=False, device=device, seed_offset=1
        ),
        normalization=normalization,
        config=config,
        device=device,
        output_dir=output_dir,
        epochs=config.training.epochs,
        learning_rate=config.training.learning_rate,
        provenance=provenance,
        resume=config.training.resume,
        minimum_epochs=minimum_epochs,
        require_convergence=require_convergence,
    )


def _set_finetune_mode(model: HoustonPINO, mode: str) -> None:
    if mode == "adapter":
        model.freeze_backbone()
    elif mode == "head":
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in model.head.parameters():
            parameter.requires_grad_(True)
        for parameter in model.adapter.parameters():
            parameter.requires_grad_(True)
    elif mode == "full":
        model.unfreeze_all()
    else:
        raise ValueError("finetune.mode must be one of: adapter, head, full")


def finetune_experiment(
    config: ExperimentConfig,
    *,
    event_id: Optional[str] = None,
    checkpoint: Optional[str] = None,
) -> Path:
    """Adapt to the observed prefix of one held-out event without future leakage."""

    _validate_config(config)
    event = event_id or config.finetune.event_id
    source = checkpoint or config.finetune.checkpoint
    if not event:
        raise ValueError("an event_id is required for fine-tuning")
    if not source:
        raise ValueError("a base checkpoint is required for fine-tuning")
    source_path = Path(source).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)

    seed_everything(config.training.seed, config.training.deterministic)
    device = select_device(config.training.device, config.training.require_cuda)
    normalization = _normalization(config)
    model = build_model(config).to(device)
    payload = load_checkpoint(source_path, model=model, map_location=device)
    if payload.get("config", {}).get("model") != config.to_dict()["model"]:
        raise ValueError("checkpoint model configuration differs from fine-tune config")
    if payload.get("normalization") != normalization:
        raise ValueError("checkpoint and configured normalization statistics differ")
    checkpoint_provenance = payload.get("provenance", {})
    training_events = checkpoint_provenance.get("training_event_ids")
    if not training_events:
        raise ValueError(
            "checkpoint lacks training_event_ids, so held-out adaptation cannot be verified"
        )
    if event in set(training_events):
        raise ValueError(
            f"event {event!r} contributed to base training and cannot be presented as unseen"
        )
    current_provenance = _provenance(config)
    for key in (
        "operator_source_sha256",
        "training_manifest_sha256",
        "normalization_sha256",
        "static_grid_sha256",
    ):
        if checkpoint_provenance.get(key) != current_provenance[key]:
            raise ValueError(f"checkpoint provenance mismatch for {key}")
    _set_finetune_mode(model, config.finetune.mode)

    observed_prefix = _dataset(
        config,
        config.finetune.split,
        event_ids=[event],
        observation_hours=config.finetune.observation_hours,
    )
    observed_train, observed_validation = _adaptation_subsets(
        observed_prefix,
        maximum_train=config.data.max_train_windows,
        maximum_validation=config.data.max_validation_windows,
        seed=config.training.seed,
    )
    complete_event = _dataset(
        config,
        config.finetune.split,
        event_ids=[event],
        # Evaluation must contain the exact first post-observation origin even
        # when training windows were subsampled with a coarser stride.
        start_stride=1,
    )
    forecast_data = _forecast_origin_subset(
        complete_event,
        config.finetune.observation_hours,
        config.data.max_validation_windows,
        config.training.seed + 1,
    )
    output_dir = Path(config.finetune.output_dir) / event
    provenance = _provenance(config)
    provenance.update(
        {
            "workflow": "event_adaptation",
            "event_id": event,
            "event_split": config.finetune.split,
            "observation_hours": config.finetune.observation_hours,
            "parent_checkpoint": str(source_path),
            "finetune_mode": config.finetune.mode,
            "trainable_parameters": model.trainable_parameter_count,
            "checkpoint_selection": "observed-prefix member holdout only",
        }
    )
    adapted_checkpoint = _fit(
        model=model,
        train_loader=_loader(observed_train, config, shuffle=True, device=device),
        validation_loader=_loader(
            observed_validation,
            config,
            shuffle=False,
            device=device,
            seed_offset=1,
        ),
        normalization=normalization,
        config=config,
        device=device,
        output_dir=output_dir,
        epochs=config.finetune.epochs,
        learning_rate=config.finetune.learning_rate,
        provenance=provenance,
    )
    # Only after training/checkpoint selection is complete do we open the
    # post-observation targets for a one-time report.  These metrics cannot
    # affect the optimizer, scheduler, early stopping, or chosen checkpoint.
    evaluation_balancer = _loss_balancer(config, device)
    load_checkpoint(
        adapted_checkpoint,
        model=model,
        loss_balancer=evaluation_balancer,
        map_location=device,
    )
    future_metrics = _run_epoch(
        model,
        _loader(forecast_data, config, shuffle=False, device=device, seed_offset=2),
        normalization,
        config,
        device,
        max(config.finetune.epochs - 1, 0),
        evaluation_balancer,
    )
    _write_json(
        output_dir / "post_observation_evaluation.json",
        {
            "event_id": event,
            "observation_hours": config.finetune.observation_hours,
            "checkpoint": str(adapted_checkpoint),
            "selection_role": "evaluation only; not used for model selection",
            "metrics": future_metrics,
        },
    )
    return adapted_checkpoint
