"""Target-free autoregressive forecasting for trained Houston operators."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
import json
import os

import h5py
import numpy as np
import torch

from .checkpoint import load_checkpoint
from .config import ExperimentConfig
from .features import denormalize_delta, make_features
from .physics import project_wet_dry
from .trainer import (
    _normalization,
    _operator_source_sha256,
    _validate_config,
    build_model,
)
from .utils import select_device, sha256_file


def _batched_field(
    value: torch.Tensor, batch: int, height: int, width: int, name: str
) -> torch.Tensor:
    if value.ndim == 2:
        value = value.unsqueeze(0).unsqueeze(0)
    if value.ndim == 3 and value.shape[0] == 1:
        value = value.unsqueeze(0)
    if value.ndim != 4 or value.shape[1:] != (1, height, width):
        raise ValueError(f"{name} must have shape [1,H,W] or [B,1,H,W]")
    if value.shape[0] == 1:
        value = value.expand(batch, -1, -1, -1)
    if value.shape[0] != batch:
        raise ValueError(f"{name} batch dimension does not match initial_state")
    return value


def forecast_rollout(
    model: torch.nn.Module,
    initial_state: torch.Tensor,
    rainfall_mm_h: torch.Tensor,
    ocean_height_m: torch.Tensor,
    bed_elevation: torch.Tensor,
    domain_mask: torch.Tensor,
    ocean_boundary_mask: torch.Tensor,
    normalization: Mapping[str, Any],
    config: ExperimentConfig,
) -> torch.Tensor:
    """Roll forward from one observed state using forcing only.

    Returns ``[B,T+1,3,H,W]`` and includes the projected initial state at
    index zero. No future target state is accepted by this interface.
    """

    if initial_state.ndim == 3:
        initial_state = initial_state.unsqueeze(0)
    if initial_state.ndim != 4 or initial_state.shape[1] != 3:
        raise ValueError("initial_state must have shape [3,H,W] or [B,3,H,W]")
    if rainfall_mm_h.ndim == 4:
        rainfall_mm_h = rainfall_mm_h.unsqueeze(0)
    if rainfall_mm_h.ndim != 5 or rainfall_mm_h.shape[2] != 1:
        raise ValueError("rainfall must have shape [T,1,H,W] or [B,T,1,H,W]")
    batch, _, height, width = initial_state.shape
    if rainfall_mm_h.shape[0] == 1:
        rainfall_mm_h = rainfall_mm_h.expand(batch, -1, -1, -1, -1)
    if rainfall_mm_h.shape[0] != batch or rainfall_mm_h.shape[-2:] != (
        height,
        width,
    ):
        raise ValueError("rainfall batch/grid does not match initial_state")
    steps = rainfall_mm_h.shape[1]
    if ocean_height_m.ndim == 1:
        ocean_height_m = ocean_height_m.unsqueeze(0)
    if ocean_height_m.ndim != 2:
        raise ValueError("ocean height must have shape [T] or [B,T]")
    if ocean_height_m.shape[0] == 1:
        ocean_height_m = ocean_height_m.expand(batch, -1)
    if ocean_height_m.shape != (batch, steps):
        raise ValueError("ocean height must have one value per batch and forecast step")

    device = next(model.parameters()).device
    current = initial_state.to(device=device, dtype=torch.float32)
    rainfall_mm_h = rainfall_mm_h.to(device=device, dtype=torch.float32)
    ocean_height_m = ocean_height_m.to(device=device, dtype=torch.float32)
    bed = _batched_field(
        bed_elevation.to(device=device, dtype=torch.float32),
        batch,
        height,
        width,
        "bed_elevation",
    )
    domain = _batched_field(
        domain_mask.to(device=device), batch, height, width, "domain_mask"
    )
    ocean_mask = _batched_field(
        ocean_boundary_mask.to(device=device),
        batch,
        height,
        width,
        "ocean_boundary_mask",
    )
    tensors = (current, rainfall_mm_h, ocean_height_m, bed, domain, ocean_mask)
    if not all(torch.isfinite(value).all() for value in tensors):
        raise ValueError("forecast inputs must contain only finite values")

    current = project_wet_dry(
        current,
        domain,
        dry_tolerance=config.physics.dry_tolerance_m,
        max_velocity=config.physics.max_velocity_m_s,
    )
    states = [current]
    manning = torch.full_like(domain, config.physics.manning_n)
    model.eval()
    with torch.no_grad():
        for lead in range(steps):
            features = make_features(
                current,
                rainfall_mm_h[:, lead],
                ocean_height_m[:, lead],
                bed,
                domain,
                ocean_mask,
                normalization,
                manning=manning,
            )
            delta = denormalize_delta(model(features), normalization)
            current = project_wet_dry(
                current + delta,
                domain,
                dry_tolerance=config.physics.dry_tolerance_m,
                max_velocity=config.physics.max_velocity_m_s,
            )
            if not torch.isfinite(current).all():
                raise FloatingPointError(
                    f"non-finite forecast state at lead {lead + 1}"
                )
            states.append(current)
    return torch.stack(states, dim=1)


def forecast_file(
    config: ExperimentConfig,
    checkpoint: str | Path,
    input_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Forecast from a forcing HDF5 file and atomically write predictions."""

    _validate_config(config)
    source = Path(checkpoint).expanduser().resolve()
    forcing_path = Path(input_path).expanduser().resolve()
    destination = Path(output_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if not forcing_path.is_file():
        raise FileNotFoundError(forcing_path)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite forecast output: {destination}")

    device = select_device(config.training.device, config.training.require_cuda)
    normalization = _normalization(config)
    model = build_model(config).to(device)
    payload = load_checkpoint(source, model=model, map_location=device)
    if payload.get("config", {}).get("model") != config.to_dict()["model"]:
        raise ValueError("checkpoint model configuration differs from forecast config")
    if payload.get("normalization") != normalization:
        raise ValueError("checkpoint and configured normalization statistics differ")
    checkpoint_physics = payload.get("config", {}).get("physics", {})
    for field in ("dt_s", "manning_n", "dry_tolerance_m", "max_velocity_m_s"):
        if checkpoint_physics.get(field) != getattr(config.physics, field):
            raise ValueError(
                f"checkpoint physics.{field} differs from the forecast config"
            )
    if (
        payload.get("provenance", {}).get("operator_source_sha256")
        != _operator_source_sha256()
    ):
        raise ValueError("checkpoint operator source differs from the current code")

    static_path = Path(
        config.data.static_grid or Path(config.data.dataset_root) / "static_grid.h5"
    )
    static_hash = sha256_file(static_path)
    if payload.get("provenance", {}).get("static_grid_sha256") != static_hash:
        raise ValueError("checkpoint provenance does not match the configured static grid")
    with h5py.File(static_path, "r") as static:
        bed = torch.from_numpy(np.asarray(static["bed_elevation"], dtype=np.float32))
        domain = torch.from_numpy(np.asarray(static["domain_mask"], dtype=np.float32))
        ocean_mask = torch.from_numpy(
            np.asarray(static["ocean_boundary_mask"], dtype=np.float32)
        )
        static_signature = str(static.attrs.get("static_grid_signature", ""))
    with h5py.File(forcing_path, "r") as forcing:
        required = ("initial_state", "rainfall", "ocean_boundary_height")
        missing = [name for name in required if name not in forcing]
        if missing:
            raise ValueError(f"forecast input is missing datasets: {missing}")
        initial_state = torch.from_numpy(
            np.asarray(forcing["initial_state"], dtype=np.float32)
        )
        rainfall = torch.from_numpy(np.asarray(forcing["rainfall"], dtype=np.float32))
        ocean = torch.from_numpy(
            np.asarray(forcing["ocean_boundary_height"], dtype=np.float32)
        )
        if "time_seconds" in forcing:
            time_seconds = np.asarray(forcing["time_seconds"], dtype=np.float64)
        else:
            time_seconds = np.arange(rainfall.shape[0] + 1, dtype=np.float64)
            time_seconds *= config.physics.dt_s
    if initial_state.ndim != 3 or initial_state.shape[0] != 3:
        raise ValueError("file initial_state must have shape [3,H,W]")
    if rainfall.ndim != 4 or rainfall.shape[1] != 1 or rainfall.shape[0] < 1:
        raise ValueError("file rainfall must have nonempty shape [T,1,H,W]")
    if ocean.ndim != 1 or ocean.shape[0] != rainfall.shape[0]:
        raise ValueError("file ocean_boundary_height must have shape [T]")
    if time_seconds.shape != (rainfall.shape[0] + 1,):
        raise ValueError("time_seconds must have T+1 entries for T forcing intervals")
    if not np.isfinite(time_seconds).all() or not np.allclose(
        np.diff(time_seconds),
        config.physics.dt_s,
        rtol=0.0,
        atol=1.0e-6,
    ):
        raise ValueError("time_seconds must use the configured fixed forecast interval")

    prediction = (
        forecast_rollout(
            model,
            initial_state,
            rainfall,
            ocean,
            bed,
            domain,
            ocean_mask,
            normalization,
            config,
        )[0]
        .cpu()
        .numpy()
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"stale temporary forecast output exists: {temporary}")
    with h5py.File(temporary, "x") as output:
        output.attrs["format"] = "houston_pino_forecast_v1"
        output.attrs["checkpoint"] = str(source)
        output.attrs["checkpoint_sha256"] = sha256_file(source)
        output.attrs["input_sha256"] = sha256_file(forcing_path)
        output.attrs["static_grid_signature"] = static_signature
        output.attrs["state_channel_names"] = "Height,MomentumX,MomentumY"
        output.attrs["state_channel_units"] = "m,m2 s-1,m2 s-1"
        output.attrs["rainfall_units"] = "mm h-1"
        output.attrs["ocean_boundary_height_units"] = "m"
        output.attrs["resolved_model_config"] = json.dumps(
            config.to_dict()["model"], sort_keys=True
        )
        output.create_dataset("state", data=prediction, compression="gzip")
        output.create_dataset("time_seconds", data=time_seconds)
        output.create_dataset("rainfall", data=rainfall.numpy(), compression="gzip")
        output.create_dataset("ocean_boundary_height", data=ocean.numpy())
    os.replace(temporary, destination)
    return destination
