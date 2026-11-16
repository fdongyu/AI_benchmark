"""Free-rollout evaluation functions for Houston flood emulators.

The file products follow the benchmark's common export schema so retained
U-Net, FNO, ResU-Net, and PINO-ResU-Net results can be evaluated without a
schema conversion step.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import h5py
import numpy as np
import pandas as pd
import torch

from pino.checkpoint import load_checkpoint
from pino.config import ExperimentConfig, load_config
from pino.features import denormalize_delta, make_features
from pino.physics import project_wet_dry
from pino.trainer import (
    _normalization,
    _operator_source_sha256,
    _validate_config,
    build_model,
)
from pino.utils import seed_everything, select_device, sha256_file

from .metrics import RunningFieldMetrics, hydrograph_metrics, summarize_events


DEPTH_THRESHOLDS_M = (0.05, 0.10, 0.30)
EXCLUDED_OUTLET_GAUGES = frozenset({"08072600", "08073100"})

DOMAIN_SUMMARY_METRICS = {
    "depth_rmse_m": ("Depth RMSE (m)", False),
    "depth_mae_m": ("Depth MAE (m)", False),
    "depth_bias_m": ("Depth bias (m)", False),
    "depth_correlation": ("Depth correlation", True),
    "depth_nse": ("Depth NSE", True),
    "depth_kge": ("Depth KGE", True),
    "depth_csi_at_0p05m": ("CSI at 0.05 m", True),
    "depth_csi_at_0p1m": ("CSI at 0.10 m", True),
    "depth_csi_at_0p3m": ("CSI at 0.30 m", True),
    "depth_f1_at_0p05m": ("F1 at 0.05 m", True),
    "depth_f1_at_0p3m": ("F1 at 0.30 m", True),
    "momentum_x_rmse_m2_s": ("x-momentum RMSE (m2/s)", False),
    "momentum_y_rmse_m2_s": ("y-momentum RMSE (m2/s)", False),
    "peak_depth_rmse_m": ("Peak-depth RMSE (m)", False),
    "peak_depth_correlation": ("Peak-depth correlation", True),
    "peak_timing_mae_h": ("Peak-time MAE (h)", False),
    "mse_skill_vs_persistence": ("MSE skill vs persistence", True),
    "volume_relative_bias": ("Water-volume relative bias", False),
}

GAUGE_SUMMARY_METRICS = {
    "depth_rmse_m": ("Gauge depth RMSE (m)", False),
    "depth_mae_m": ("Gauge depth MAE (m)", False),
    "depth_bias_m": ("Gauge depth bias (m)", False),
    "depth_correlation": ("Gauge depth correlation", True),
    "depth_nse": ("Gauge depth NSE", True),
    "depth_kge": ("Gauge depth KGE", True),
    "depth_csi_at_0p05m": ("Gauge CSI at 0.05 m", True),
    "depth_csi_at_0p1m": ("Gauge CSI at 0.10 m", True),
    "depth_csi_at_0p3m": ("Gauge CSI at 0.30 m", True),
    "absolute_peak_depth_error_m": ("Absolute peak-depth error (m)", False),
    "absolute_peak_timing_error_hours": (
        "Absolute peak-time error (h)",
        False,
    ),
    "absolute_arrival_timing_error_hours_at_0p05m": (
        "Absolute 0.05-m arrival error (h)",
        False,
    ),
    "absolute_arrival_timing_error_hours_at_0p3m": (
        "Absolute 0.30-m arrival error (h)",
        False,
    ),
}


def _read_static(path: Path) -> dict[str, np.ndarray]:
    with h5py.File(path, "r", locking=False) as handle:
        required = (
            "bed_elevation",
            "domain_mask",
            "ocean_boundary_mask",
            "x",
            "y",
        )
        missing = [name for name in required if name not in handle]
        if missing:
            raise ValueError(f"static grid is missing datasets: {missing}")
        result = {
            "bed_elevation": np.asarray(handle["bed_elevation"], dtype=np.float32),
            "domain_mask": np.asarray(handle["domain_mask"], dtype=np.float32),
            "ocean_boundary_mask": np.asarray(
                handle["ocean_boundary_mask"], dtype=np.float32
            ),
            "x_raw": np.asarray(handle["x"], dtype=np.float64),
            "y_raw": np.asarray(handle["y"], dtype=np.float64),
        }
        if "source_cell_index" in handle:
            result["source_cell_index"] = np.asarray(
                handle["source_cell_index"], dtype=np.int64
            )
    shape = result["domain_mask"].shape
    if any(result[name].shape != shape for name in required[:3]):
        raise ValueError("static raster fields have inconsistent shapes")
    x_raw = result.pop("x_raw")
    y_raw = result.pop("y_raw")
    if x_raw.ndim == 1 and y_raw.ndim == 1:
        x, y = np.meshgrid(x_raw, y_raw)
    elif x_raw.shape == shape and y_raw.shape == shape:
        x, y = x_raw, y_raw
    else:
        raise ValueError("static x/y coordinates do not match the raster")
    result["x"] = x
    result["y"] = y
    return result


def make_gauge_mapping(gauge_file: Path, static_file: Path) -> pd.DataFrame:
    """Map retained USGS locations to their nearest active ML-grid cells."""

    gauges = pd.read_csv(gauge_file, dtype={"station_id": str})
    required = {"station_id", "station_name", "gauge_x", "gauge_y"}
    missing = required - set(gauges.columns)
    if missing:
        raise ValueError(f"gauge metadata is missing columns: {sorted(missing)}")
    gauges["station_id"] = gauges["station_id"].str.zfill(8)
    gauges = gauges.loc[
        ~gauges["station_id"].isin(EXCLUDED_OUTLET_GAUGES)
    ].copy()
    if gauges["station_id"].duplicated().any():
        raise ValueError("gauge metadata contains duplicate station identifiers")

    static = _read_static(static_file)
    x = static["x"]
    y = static["y"]
    domain = static["domain_mask"] > 0.5
    valid = domain & np.isfinite(x) & np.isfinite(y)
    if not np.any(valid):
        raise ValueError("static grid has no valid active cells")
    source = static.get("source_cell_index")
    rows: list[dict[str, Any]] = []
    for gauge in gauges.itertuples(index=False):
        distance_squared = np.full(domain.shape, np.inf, dtype=np.float64)
        distance_squared[valid] = (
            (x[valid] - float(gauge.gauge_x)) ** 2
            + (y[valid] - float(gauge.gauge_y)) ** 2
        )
        flat_index = int(np.argmin(distance_squared))
        row, column = np.unravel_index(flat_index, domain.shape)
        record = {
            "station_id": str(gauge.station_id).zfill(8),
            "station_name": str(gauge.station_name),
            "gauge_x": float(gauge.gauge_x),
            "gauge_y": float(gauge.gauge_y),
            "model_row": int(row),
            "model_column": int(column),
            "model_x": float(x[row, column]),
            "model_y": float(y[row, column]),
            "mapping_distance_m": float(math.sqrt(distance_squared[row, column])),
        }
        if source is not None:
            record["model_source_cell"] = int(source[row, column])
        rows.append(record)
    result = pd.DataFrame(rows).sort_values("station_id").reset_index(drop=True)
    if result.empty:
        raise ValueError("no retained gauges remain after exclusions")
    return result


def _checkpoint_and_model(
    config: ExperimentConfig,
    checkpoint: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any]]:
    normalization = _normalization(config)
    model = build_model(config).to(device)
    payload = load_checkpoint(checkpoint, model=model, map_location=device)
    checkpoint_config = payload.get("config", {})
    if checkpoint_config.get("model") != config.to_dict()["model"]:
        raise ValueError("checkpoint model configuration differs from evaluation config")
    if payload.get("normalization") != normalization:
        raise ValueError("checkpoint normalization differs from evaluation config")
    checkpoint_physics = checkpoint_config.get("physics", {})
    for field in (
        "manning_n",
        "dry_tolerance_m",
        "max_velocity_m_s",
        "dt_s",
        "dx_m",
        "dy_m",
    ):
        if checkpoint_physics.get(field) != getattr(config.physics, field):
            raise ValueError(f"checkpoint physics.{field} differs from config")
    provenance = payload.get("provenance", {})
    checks = {
        "operator_source_sha256": _operator_source_sha256(),
        "manifest_sha256": sha256_file(config.data.split_csv),
        "normalization_sha256": sha256_file(config.data.normalization_json),
        "static_grid_sha256": sha256_file(config.data.static_grid),
    }
    for name, expected in checks.items():
        if provenance.get(name) != expected:
            raise ValueError(f"checkpoint provenance mismatch for {name}")
    model.eval()
    return model, normalization, payload


def _select_trajectories(
    config: ExperimentConfig,
    *,
    split: str,
    evaluation_members: int,
    max_events: int,
    event_ids: tuple[str, ...],
) -> pd.DataFrame:
    manifest = pd.read_csv(config.data.split_csv)
    required = {"scenario_id", "event_id", "member_id", "timesteps", "split"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"trajectory manifest is missing columns: {sorted(missing)}")
    event_split_counts = manifest.groupby("event_id")["split"].nunique()
    if (event_split_counts > 1).any():
        leaked = event_split_counts[event_split_counts > 1].index[0]
        raise ValueError(f"whole-event split violation for {leaked}")
    selected = manifest.loc[manifest["split"].astype(str) == split].copy()
    selected["event_id"] = selected["event_id"].astype(str)
    if event_ids:
        requested = set(event_ids)
        available = set(selected["event_id"])
        missing_events = requested - available
        if missing_events:
            raise ValueError(
                f"requested events are absent from split {split!r}: "
                f"{sorted(missing_events)}"
            )
        selected = selected.loc[selected["event_id"].isin(requested)]
    selected = selected.sort_values(["event_id", "member_id"])
    selected = selected.groupby("event_id", group_keys=False).head(
        evaluation_members
    )
    if max_events > 0:
        events = sorted(selected["event_id"].unique())[:max_events]
        selected = selected.loc[selected["event_id"].isin(events)]
    selected = selected.reset_index(drop=True)
    if selected.empty:
        raise ValueError(f"no trajectories selected from split {split!r}")
    return selected


def _trajectory_path(config: ExperimentConfig, row: Any) -> Path:
    path = (
        Path(config.data.dataset_root)
        / "trajectories"
        / str(row.event_id)
        / f"member_{int(row.member_id):03d}.h5"
    ).resolve()
    root = Path(config.data.dataset_root).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"trajectory escapes dataset root: {path}") from exc
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _torch_static(
    static: dict[str, np.ndarray], device: torch.device, manning_n: float
) -> dict[str, torch.Tensor]:
    result = {
        "bed": torch.from_numpy(static["bed_elevation"])[None, None].to(device),
        "domain": torch.from_numpy(static["domain_mask"])[None, None].to(device),
        "ocean_mask": torch.from_numpy(static["ocean_boundary_mask"])[
            None, None
        ].to(device),
    }
    result["manning"] = torch.full_like(result["domain"], float(manning_n))
    return result


def _one_step(
    model: torch.nn.Module,
    current: torch.Tensor,
    rainfall: torch.Tensor,
    ocean_height: torch.Tensor,
    static: dict[str, torch.Tensor],
    normalization: dict[str, Any],
    config: ExperimentConfig,
) -> torch.Tensor:
    features = make_features(
        current,
        rainfall,
        ocean_height,
        static["bed"],
        static["domain"],
        static["ocean_mask"],
        normalization,
        manning=static["manning"],
    )
    delta = denormalize_delta(model(features), normalization)
    return project_wet_dry(
        current + delta,
        static["domain"],
        dry_tolerance=config.physics.dry_tolerance_m,
        max_velocity=config.physics.max_velocity_m_s,
    )


def _metric_bundle() -> dict[str, RunningFieldMetrics]:
    return {
        "depth": RunningFieldMetrics(DEPTH_THRESHOLDS_M),
        "momentum_x": RunningFieldMetrics(),
        "momentum_y": RunningFieldMetrics(),
    }


def _update_bundle(
    bundle: dict[str, RunningFieldMetrics],
    truth: np.ndarray,
    prediction: np.ndarray,
    mask: np.ndarray,
) -> None:
    bundle["depth"].update(truth[0], prediction[0], mask)
    bundle["momentum_x"].update(truth[1], prediction[1], mask)
    bundle["momentum_y"].update(truth[2], prediction[2], mask)


def _bundle_result(
    bundle: dict[str, RunningFieldMetrics],
) -> dict[str, float | int]:
    result: dict[str, float | int] = {}
    result.update(bundle["depth"].result("depth", "m"))
    result.update(bundle["momentum_x"].result("momentum_x", "m2_s"))
    result.update(bundle["momentum_y"].result("momentum_y", "m2_s"))
    return result


def _write_csv(frame: pd.DataFrame, path: Path, **kwargs: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, **kwargs)
    os.replace(temporary, path)


def _write_json(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def evaluate_rollouts(
    *,
    config_path: Path,
    checkpoint: Path,
    gauge_file: Path,
    output_dir: Path,
    split: str = "test",
    evaluation_members: int = 32,
    max_events: int = 0,
    max_lead_hours: float = 0.0,
    event_ids: tuple[str, ...] = (),
    snapshot_member: int = 1,
    device_name: str = "auto",
    threads: int = 16,
    strict: bool = True,
) -> dict[str, Any]:
    """Run every selected trajectory freely from its initial condition."""

    if evaluation_members < 1:
        raise ValueError("evaluation_members must be positive")
    if max_events < 0 or max_lead_hours < 0.0:
        raise ValueError("max_events and max_lead_hours must be nonnegative")
    if snapshot_member < 1 or threads < 1:
        raise ValueError("snapshot_member and threads must be positive")
    config_path = config_path.expanduser().resolve()
    checkpoint = checkpoint.expanduser().resolve()
    gauge_file = gauge_file.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    for required in (config_path, checkpoint, gauge_file):
        if not required.is_file():
            raise FileNotFoundError(required)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"refusing to mix analysis products in nonempty directory: {output_dir}"
        )
    snapshot_dir = output_dir / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    config = load_config(config_path)
    _validate_config(config)
    seed_everything(config.training.seed, deterministic=True)
    if device_name == "auto":
        selected_device = "cuda" if torch.cuda.is_available() else "cpu"
        require_cuda = False
    else:
        selected_device = device_name
        require_cuda = device_name == "cuda"
    device = select_device(selected_device, require_cuda)
    model, normalization, checkpoint_payload = _checkpoint_and_model(
        config, checkpoint, device
    )

    static_path = Path(config.data.static_grid).resolve()
    static = _read_static(static_path)
    torch_static = _torch_static(static, device, config.physics.manning_n)
    domain = static["domain_mask"] > 0.5
    if not np.any(domain):
        raise ValueError("evaluation domain is empty")
    mapping = make_gauge_mapping(gauge_file, static_path)
    _write_csv(mapping, output_dir / "ml_grid_usgs_gauge_mapping.csv")
    gauge_rows_index = mapping["model_row"].to_numpy(dtype=np.int64)
    gauge_columns_index = mapping["model_column"].to_numpy(dtype=np.int64)

    selected = _select_trajectories(
        config,
        split=split,
        evaluation_members=evaluation_members,
        max_events=max_events,
        event_ids=event_ids,
    )
    shape = domain.shape
    spatial_squared_error = np.zeros(shape, dtype=np.float64)
    spatial_absolute_error = np.zeros(shape, dtype=np.float64)
    spatial_error = np.zeros(shape, dtype=np.float64)
    spatial_count = np.zeros(shape, dtype=np.int64)
    metrics_by_lead: dict[float, dict[str, RunningFieldMetrics]] = {}
    domain_rows: list[dict[str, Any]] = []
    gauge_rows: list[dict[str, Any]] = []
    gauge_timeseries: list[pd.DataFrame] = []
    status_rows: list[dict[str, Any]] = []
    checkpoint_epoch = int(checkpoint_payload.get("epoch", -1))
    run_key = f"houston_pino_epoch{checkpoint_epoch:04d}"
    started = time.monotonic()

    with torch.inference_mode():
        for number, row in enumerate(selected.itertuples(index=False), start=1):
            print(
                f"{run_key}: {number}/{len(selected)} {row.scenario_id}",
                flush=True,
            )
            path = _trajectory_path(config, row)
            trajectory_metrics = _metric_bundle()
            persistence_metrics = RunningFieldMetrics(DEPTH_THRESHOLDS_M)
            volume_metrics = RunningFieldMetrics()
            volume_truth_sum = 0.0
            volume_error_sum = 0.0
            gauge_truth: list[list[float]] = [list() for _ in range(len(mapping))]
            gauge_prediction: list[list[float]] = [
                list() for _ in range(len(mapping))
            ]
            lead_hours: list[float] = []
            utc_seconds: list[int] = []
            status = "complete"
            peak_snapshot: dict[str, Any] | None = None
            peak_truth_mean = -math.inf

            with h5py.File(path, "r", locking=False) as handle:
                required = (
                    "state",
                    "rainfall",
                    "ocean_boundary_height",
                    "time_seconds",
                )
                missing = [name for name in required if name not in handle]
                if missing:
                    raise ValueError(f"{path} is missing datasets: {missing}")
                timesteps = min(
                    int(row.timesteps),
                    int(handle["state"].shape[0]),
                    int(handle["rainfall"].shape[0]),
                    int(handle["ocean_boundary_height"].shape[0]),
                )
                if timesteps < 2:
                    raise ValueError(f"trajectory has fewer than two states: {path}")
                time_seconds = np.asarray(
                    handle["time_seconds"][:timesteps], dtype=np.float64
                )
                intervals = np.diff(time_seconds)
                if (
                    not np.isfinite(time_seconds).all()
                    or np.any(intervals <= 0.0)
                    or not np.allclose(
                        intervals,
                        config.physics.dt_s,
                        rtol=0.0,
                        atol=1.0e-6,
                    )
                ):
                    raise ValueError(f"trajectory cadence disagrees with config: {path}")
                available = timesteps - 1
                if max_lead_hours > 0.0:
                    available = min(
                        available,
                        int(math.floor(max_lead_hours * 3600.0 / intervals[0])),
                    )
                if available < 1:
                    raise ValueError("max_lead_hours selects no forecast transitions")
                initial = np.asarray(handle["state"][0], dtype=np.float32)
                if initial.shape != (3, *shape) or not np.isfinite(initial).all():
                    raise ValueError(f"invalid initial state in {path}")
                current = torch.from_numpy(initial)[None].to(device)
                current = project_wet_dry(
                    current,
                    torch_static["domain"],
                    dry_tolerance=config.physics.dry_tolerance_m,
                    max_velocity=config.physics.max_velocity_m_s,
                )
                initial_depth = current[0, 0].detach().cpu().numpy().copy()
                peak_truth = initial[0].copy()
                peak_prediction = initial_depth.copy()
                peak_truth_time = np.zeros(shape, dtype=np.float32)
                peak_prediction_time = np.zeros(shape, dtype=np.float32)
                cell_area_m2 = config.physics.dx_m * config.physics.dy_m

                for time_index in range(available):
                    rainfall_array = np.asarray(
                        handle["rainfall"][time_index], dtype=np.float32
                    )
                    if rainfall_array.shape != (1, *shape):
                        raise ValueError(f"invalid rainfall shape in {path}")
                    rainfall = torch.from_numpy(rainfall_array)[None].to(device)
                    ocean = torch.tensor(
                        [float(handle["ocean_boundary_height"][time_index])],
                        dtype=torch.float32,
                        device=device,
                    )
                    current = _one_step(
                        model,
                        current,
                        rainfall,
                        ocean,
                        torch_static,
                        normalization,
                        config,
                    )
                    if not torch.isfinite(current).all():
                        status = f"nonfinite_at_lead_{time_index + 1}"
                        break
                    prediction = (
                        current[0].detach().cpu().numpy().astype(np.float32, copy=False)
                    )
                    truth = np.asarray(
                        handle["state"][time_index + 1], dtype=np.float32
                    )
                    if truth.shape != (3, *shape) or not np.isfinite(truth).all():
                        raise ValueError(
                            f"invalid target state at index {time_index + 1} in {path}"
                        )
                    lead_hour = float(
                        (time_seconds[time_index + 1] - time_seconds[0]) / 3600.0
                    )
                    _update_bundle(trajectory_metrics, truth, prediction, domain)
                    lead_bundle = metrics_by_lead.setdefault(
                        lead_hour, _metric_bundle()
                    )
                    _update_bundle(lead_bundle, truth, prediction, domain)
                    persistence_metrics.update(truth[0], initial_depth, domain)

                    depth_error = prediction[0] - truth[0]
                    spatial_squared_error[domain] += (
                        depth_error[domain].astype(np.float64) ** 2
                    )
                    spatial_absolute_error[domain] += np.abs(
                        depth_error[domain].astype(np.float64)
                    )
                    spatial_error[domain] += depth_error[domain].astype(np.float64)
                    spatial_count[domain] += 1

                    truth_volume = float(
                        truth[0, domain].sum(dtype=np.float64) * cell_area_m2
                    )
                    predicted_volume = float(
                        prediction[0, domain].sum(dtype=np.float64) * cell_area_m2
                    )
                    volume_metrics.update(
                        np.asarray([truth_volume]), np.asarray([predicted_volume])
                    )
                    volume_truth_sum += truth_volume
                    volume_error_sum += predicted_volume - truth_volume

                    truth_peak_update = truth[0] > peak_truth
                    prediction_peak_update = prediction[0] > peak_prediction
                    peak_truth[truth_peak_update] = truth[0, truth_peak_update]
                    peak_prediction[prediction_peak_update] = prediction[
                        0, prediction_peak_update
                    ]
                    peak_truth_time[truth_peak_update] = lead_hour
                    peak_prediction_time[prediction_peak_update] = lead_hour
                    truth_mean = float(np.mean(truth[0, domain], dtype=np.float64))
                    if truth_mean > peak_truth_mean:
                        peak_truth_mean = truth_mean
                        if "time_utc_unix_seconds" in handle:
                            snapshot_utc = int(
                                handle["time_utc_unix_seconds"][time_index + 1]
                            )
                        else:
                            snapshot_utc = -1
                        peak_snapshot = {
                            "truth": truth[0].copy(),
                            "prediction": prediction[0].copy(),
                            "lead_hour": lead_hour,
                            "time_utc_unix_seconds": snapshot_utc,
                        }

                    lead_hours.append(lead_hour)
                    if "time_utc_unix_seconds" in handle:
                        utc_seconds.append(
                            int(handle["time_utc_unix_seconds"][time_index + 1])
                        )
                    else:
                        utc_seconds.append(-1)
                    gauge_truth_values = truth[
                        0, gauge_rows_index, gauge_columns_index
                    ]
                    gauge_prediction_values = prediction[
                        0, gauge_rows_index, gauge_columns_index
                    ]
                    for gauge_index in range(len(mapping)):
                        gauge_truth[gauge_index].append(
                            float(gauge_truth_values[gauge_index])
                        )
                        gauge_prediction[gauge_index].append(
                            float(gauge_prediction_values[gauge_index])
                        )

            trajectory_result = _bundle_result(trajectory_metrics)
            peak_metrics = RunningFieldMetrics(DEPTH_THRESHOLDS_M)
            peak_metrics.update(peak_truth, peak_prediction, domain)
            trajectory_result.update(peak_metrics.result("peak_depth", "m"))
            peak_timing = RunningFieldMetrics()
            peak_timing_mask = domain & (peak_truth >= DEPTH_THRESHOLDS_M[0])
            peak_timing.update(
                peak_truth_time,
                peak_prediction_time,
                peak_timing_mask,
            )
            trajectory_result.update(peak_timing.result("peak_timing", "h"))
            persistence_result = persistence_metrics.result(
                "persistence_depth", "m"
            )
            trajectory_result.update(persistence_result)
            baseline_rmse = float(
                persistence_result.get("persistence_depth_rmse_m", math.nan)
            )
            model_rmse = float(trajectory_result.get("depth_rmse_m", math.nan))
            trajectory_result["mse_skill_vs_persistence"] = (
                1.0 - model_rmse * model_rmse / (baseline_rmse * baseline_rmse)
                if math.isfinite(baseline_rmse) and baseline_rmse > 0.0
                else math.nan
            )
            trajectory_result.update(volume_metrics.result("volume", "m3"))
            trajectory_result["volume_relative_bias"] = (
                volume_error_sum / volume_truth_sum
                if volume_truth_sum > 0.0
                else math.nan
            )
            base = {
                "run_key": run_key,
                "model_key": "emulator",
                "model": "Emulator",
                "checkpoint_epoch": checkpoint_epoch,
                "split": split,
                "scenario_id": str(row.scenario_id),
                "event_id": str(row.event_id),
                "member_id": int(row.member_id),
                "forecast_hours_completed": (
                    float(lead_hours[-1]) if lead_hours else 0.0
                ),
                "forecast_steps_completed": len(lead_hours),
                "status": status,
            }
            domain_rows.append({**base, **trajectory_result})
            status_rows.append(base)

            interval_hours = config.physics.dt_s / 3600.0
            for gauge_index, gauge in mapping.iterrows():
                truth_values = np.asarray(
                    gauge_truth[gauge_index], dtype=np.float64
                )
                prediction_values = np.asarray(
                    gauge_prediction[gauge_index], dtype=np.float64
                )
                gauge_rows.append(
                    {
                        **base,
                        "station_id": str(gauge.station_id).zfill(8),
                        "station_name": str(gauge.station_name),
                        **hydrograph_metrics(
                            truth_values,
                            prediction_values,
                            interval_hours=interval_hours,
                            thresholds=DEPTH_THRESHOLDS_M,
                        ),
                    }
                )
                gauge_timeseries.append(
                    pd.DataFrame(
                        {
                            "run_key": run_key,
                            "model_key": "emulator",
                            "model": "Emulator",
                            "checkpoint_epoch": checkpoint_epoch,
                            "split": split,
                            "scenario_id": str(row.scenario_id),
                            "event_id": str(row.event_id),
                            "member_id": int(row.member_id),
                            "station_id": str(gauge.station_id).zfill(8),
                            "station_name": str(gauge.station_name),
                            "lead_hour": lead_hours,
                            "time_utc_unix_seconds": utc_seconds,
                            "rdycore_depth_m": truth_values,
                            "predicted_depth_m": prediction_values,
                        }
                    )
                )
            if (
                int(row.member_id) == snapshot_member
                and peak_snapshot is not None
                and status == "complete"
            ):
                np.savez_compressed(
                    snapshot_dir / f"{row.scenario_id}.peak.npz",
                    truth=peak_snapshot["truth"],
                    prediction=peak_snapshot["prediction"],
                    peak_truth=peak_truth,
                    peak_prediction=peak_prediction,
                    peak_truth_time_hours=peak_truth_time,
                    peak_prediction_time_hours=peak_prediction_time,
                    lead_hour=np.float32(peak_snapshot["lead_hour"]),
                    time_utc_unix_seconds=np.int64(
                        peak_snapshot["time_utc_unix_seconds"]
                    ),
                )

    valid = spatial_count > 0
    rmse = np.full(shape, np.nan, dtype=np.float32)
    mae = np.full(shape, np.nan, dtype=np.float32)
    bias = np.full(shape, np.nan, dtype=np.float32)
    rmse[valid] = np.sqrt(
        spatial_squared_error[valid] / spatial_count[valid]
    ).astype(np.float32)
    mae[valid] = (
        spatial_absolute_error[valid] / spatial_count[valid]
    ).astype(np.float32)
    bias[valid] = (spatial_error[valid] / spatial_count[valid]).astype(np.float32)
    np.savez_compressed(
        output_dir / "spatial_error_summary.npz",
        x=static["x"].astype(np.float32),
        y=static["y"].astype(np.float32),
        domain=domain,
        rmse_m=rmse,
        mae_m=mae,
        bias_m=bias,
        sample_count=spatial_count,
    )

    domain_frame = pd.DataFrame(domain_rows)
    gauge_frame = pd.DataFrame(gauge_rows)
    status_frame = pd.DataFrame(status_rows)
    _write_csv(domain_frame, output_dir / "domain_metrics_by_trajectory.csv")
    _write_csv(gauge_frame, output_dir / "gauge_metrics_by_trajectory.csv")
    _write_csv(status_frame, output_dir / "rollout_status.csv")
    timeseries_frame = pd.concat(gauge_timeseries, ignore_index=True)
    _write_csv(
        timeseries_frame,
        output_dir / "gauge_timeseries.csv.gz",
        compression="gzip",
    )

    lead_rows = []
    for lead_hour, bundle in sorted(metrics_by_lead.items()):
        lead_rows.append({"lead_hour": lead_hour, **_bundle_result(bundle)})
    _write_csv(pd.DataFrame(lead_rows), output_dir / "domain_metrics_by_lead.csv")
    complete_domain = domain_frame.loc[domain_frame["status"] == "complete"].copy()
    complete_gauge = gauge_frame.loc[gauge_frame["status"] == "complete"].copy()
    domain_event, domain_summary = summarize_events(
        complete_domain, DOMAIN_SUMMARY_METRICS
    )
    gauge_event, gauge_summary = summarize_events(
        complete_gauge, GAUGE_SUMMARY_METRICS
    )
    _write_csv(domain_event, output_dir / "domain_metrics_by_event.csv")
    _write_csv(domain_summary, output_dir / "domain_summary_metrics.csv")
    _write_csv(gauge_event, output_dir / "gauge_metrics_by_event.csv")
    _write_csv(gauge_summary, output_dir / "gauge_summary_metrics.csv")

    complete = int((status_frame["status"] == "complete").sum())
    metadata = {
        "run_key": run_key,
        "model_key": "emulator",
        "model": "Emulator",
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": checkpoint_epoch,
        "checkpoint_sha256": sha256_file(checkpoint),
        "config": str(config_path),
        "dataset_root": str(config.data.dataset_root),
        "trajectory_split": str(config.data.split_csv),
        "normalization_file": str(config.data.normalization_json),
        "static_grid": str(static_path),
        "split": split,
        "test_events": int(selected["event_id"].nunique()),
        "selected_trajectories": int(len(selected)),
        "evaluation_members_per_event": int(evaluation_members),
        "snapshot_member": int(snapshot_member),
        "max_lead_hours": float(max_lead_hours),
        "core_gauges": int(len(mapping)),
        "excluded_outlet_gauges": sorted(EXCLUDED_OUTLET_GAUGES),
        "raster_shape": list(domain.shape),
        "active_cells": int(domain.sum()),
        "device": str(device),
        "rollout_protocol": "free autoregressive rollout from state at t=0",
        "comparison_target": "RDycore water depth and conservative state",
        "gauge_metric_semantics": (
            "Emulator prediction versus RDycore depth at USGS locations; "
            "these are not "
            "comparisons with observed USGS gauge records"
        ),
        "aggregation": (
            "median within each event, then median and IQR across events"
        ),
        "depth_thresholds_m": list(DEPTH_THRESHOLDS_M),
        "complete_rollouts": complete,
        "elapsed_seconds": time.monotonic() - started,
    }
    _write_json(metadata, output_dir / "export_metadata.json")
    print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)
    if strict and complete != len(selected):
        raise RuntimeError(
            f"only {complete}/{len(selected)} rollouts completed; see rollout_status.csv"
        )
    return metadata


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "configs/houston.yaml")
    parser.add_argument(
        "--checkpoint", type=Path, default=root / "runs/houston_pino/best.pt"
    )
    parser.add_argument(
        "--gauge-file",
        type=Path,
        default=(
            root.parent
            / "Perlmutter_transfer/small_artifacts/validation/"
            "rdycore_usgs_gauge_cells.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "results/houston_pino",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--evaluation-members", type=int, default=32)
    parser.add_argument("--max-events", type=int, default=0)
    parser.add_argument("--max-lead-hours", type=float, default=0.0)
    parser.add_argument("--event-id", action="append", default=[])
    parser.add_argument("--snapshot-member", type=int, default=1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="write partial products without failing when a rollout is non-finite",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    evaluate_rollouts(
        config_path=args.config,
        checkpoint=args.checkpoint,
        gauge_file=args.gauge_file,
        output_dir=args.output_dir,
        split=args.split,
        evaluation_members=args.evaluation_members,
        max_events=args.max_events,
        max_lead_hours=args.max_lead_hours,
        event_ids=tuple(args.event_id),
        snapshot_member=args.snapshot_member,
        device_name=args.device,
        threads=args.threads,
        strict=not args.allow_incomplete,
    )


if __name__ == "__main__":
    main()
