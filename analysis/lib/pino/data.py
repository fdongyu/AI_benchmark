"""Leakage-safe access to the rasterized Houston RDycore trajectories.

The trajectory files store states at discrete output times.  Rainfall and the
ocean boundary value at index ``t`` force the transition from state ``t`` to
state ``t + 1``.  :class:`HoustonTrajectoryDataset` makes this alignment
explicit: a sample contains context states, future target states, and only the
forcing intervals needed to advance the last context state to those targets.

All arrays returned by the dataset are in physical units.  ``Normalization``
is deliberately separate so conservation and SWE residuals can continue to
operate on physical quantities.
"""

from __future__ import annotations

import bisect
import copy
import csv
import json
import math
import os
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


STATE_NAMES = ("height", "momentum_x", "momentum_y")
STATIC_NAMES = (
    "bed_elevation",
    "domain_mask",
    "ocean_boundary_mask",
    "source_cell_area",
)


def _pair(value: int | Sequence[int], name: str) -> tuple[int, int]:
    if isinstance(value, int):
        result = (value, value)
    else:
        if len(value) != 2:
            raise ValueError(f"{name} must be an int or a two-element sequence")
        result = (int(value[0]), int(value[1]))
    if result[0] <= 0 or result[1] <= 0:
        raise ValueError(f"{name} entries must be positive, found {result}")
    return result


def _is_beneath(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


class Normalization:
    """Channel-aware transformations loaded from ``normalization.json``.

    The class accepts tensors of any rank.  For vector quantities such as
    state, callers specify the channel dimension.  Rainfall follows the
    dataset convention ``log1p(max(rainfall_mm_h, 0))`` before standardizing.
    """

    REQUIRED_KEYS = (
        "state",
        "state_delta",
        "log1p_rainfall",
        "ocean_height",
        "bed_elevation",
    )

    def __init__(self, statistics: Mapping[str, Any]) -> None:
        self.statistics = dict(statistics)
        missing = [key for key in self.REQUIRED_KEYS if key not in self.statistics]
        if missing:
            raise ValueError(f"Normalization is missing keys: {missing}")
        for key in self.REQUIRED_KEYS:
            item = self.statistics[key]
            if "mean" not in item or "std" not in item:
                raise ValueError(f"Normalization entry {key!r} needs mean and std")
            mean = np.asarray(item["mean"], dtype=np.float64).reshape(-1)
            std = np.asarray(item["std"], dtype=np.float64).reshape(-1)
            if mean.size == 0 or mean.shape != std.shape:
                raise ValueError(f"Invalid normalization shape for {key!r}")
            if not np.isfinite(mean).all() or not np.isfinite(std).all():
                raise ValueError(f"Non-finite normalization values for {key!r}")
            if np.any(std <= 0.0):
                raise ValueError(f"Normalization std must be positive for {key!r}")

    @classmethod
    def from_json(cls, path: str | Path) -> "Normalization":
        with Path(path).open("r", encoding="utf-8") as stream:
            return cls(json.load(stream))

    @property
    def metadata(self) -> Mapping[str, Any]:
        return self.statistics.get("metadata", {})

    def as_dict(self) -> dict[str, Any]:
        """Return an independent mapping suitable for checkpoints/configuration."""

        return copy.deepcopy(self.statistics)

    def _parameters(
        self, key: str, tensor: torch.Tensor, channel_dim: int | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        item = self.statistics[key]
        mean = torch.as_tensor(item["mean"], dtype=tensor.dtype, device=tensor.device)
        std = torch.as_tensor(item["std"], dtype=tensor.dtype, device=tensor.device)
        if mean.numel() == 1:
            return mean.squeeze(0), std.squeeze(0)
        if channel_dim is None:
            raise ValueError(f"channel_dim is required for multi-channel entry {key!r}")
        dimension = channel_dim % tensor.ndim
        if tensor.shape[dimension] != mean.numel():
            raise ValueError(
                f"Tensor has {tensor.shape[dimension]} channels on dimension {dimension}; "
                f"{key!r} has {mean.numel()}"
            )
        shape = [1] * tensor.ndim
        shape[dimension] = mean.numel()
        return mean.reshape(shape), std.reshape(shape)

    def normalize(
        self, key: str, tensor: torch.Tensor, channel_dim: int | None = None
    ) -> torch.Tensor:
        mean, std = self._parameters(key, tensor, channel_dim)
        return (tensor - mean) / std

    def denormalize(
        self, key: str, tensor: torch.Tensor, channel_dim: int | None = None
    ) -> torch.Tensor:
        mean, std = self._parameters(key, tensor, channel_dim)
        return tensor * std + mean

    def normalize_state(self, tensor: torch.Tensor, channel_dim: int) -> torch.Tensor:
        return self.normalize("state", tensor, channel_dim)

    def denormalize_state(self, tensor: torch.Tensor, channel_dim: int) -> torch.Tensor:
        return self.denormalize("state", tensor, channel_dim)

    def normalize_state_delta(
        self, tensor: torch.Tensor, channel_dim: int
    ) -> torch.Tensor:
        return self.normalize("state_delta", tensor, channel_dim)

    def denormalize_state_delta(
        self, tensor: torch.Tensor, channel_dim: int
    ) -> torch.Tensor:
        return self.denormalize("state_delta", tensor, channel_dim)

    def normalize_rainfall(self, rainfall_mm_h: torch.Tensor) -> torch.Tensor:
        transformed = torch.log1p(torch.clamp_min(rainfall_mm_h, 0.0))
        return self.normalize("log1p_rainfall", transformed)

    def denormalize_rainfall(self, tensor: torch.Tensor) -> torch.Tensor:
        transformed = self.denormalize("log1p_rainfall", tensor)
        return torch.clamp_min(torch.expm1(transformed), 0.0)

    def normalize_ocean(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.normalize("ocean_height", tensor)

    def denormalize_ocean(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.denormalize("ocean_height", tensor)

    def normalize_bed(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.normalize("bed_elevation", tensor)

    def denormalize_bed(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.denormalize("bed_elevation", tensor)


@dataclass(frozen=True)
class _Trajectory:
    scenario_id: str
    event_id: str
    member_id: int
    timesteps: int
    interval_seconds: float
    path: Path


@dataclass(frozen=True)
class _Tile:
    y: int
    x: int
    height: int
    width: int
    valid_height: int
    valid_width: int


class HoustonTrajectoryDataset(Dataset):
    """Windowed Houston trajectories from one whole-event data split.

    Parameters
    ----------
    split_manifest:
        ``trajectory_split.csv`` containing at least ``event_id``, ``split``,
        ``timesteps``, and ``trajectory_file``.
    dataset_root:
        Current location of ``static_grid.h5`` and ``trajectories/``.  Paths in
        a relocated manifest are reconstructed beneath this root; stale
        absolute paths are never opened.
    context_steps, forecast_steps:
        Number of known state frames and future target frames.  Forcing has
        exactly ``forecast_steps`` entries and begins at the last context
        frame, matching state transitions without exposing future states.
    tile_size, tile_stride, halo:
        Optional fixed-size interior tiles.  Halo and incomplete edge tiles
        are zero padded, giving every sample the same spatial shape.  Use the
        returned ``interior_slices`` and ``valid_interior_shape`` to exclude
        halo/padding from losses.

    Returned dynamic shapes are ``[time, channel, height, width]``.  Static
    fields are ``[1, height, width]`` and ``current_state`` is ``[3,H,W]``.
    """

    def __init__(
        self,
        split_manifest: str | Path,
        dataset_root: str | Path,
        split: str,
        *,
        context_steps: int = 1,
        forecast_steps: int = 1,
        start_stride: int = 1,
        tile_size: int | Sequence[int] | None = None,
        tile_stride: int | Sequence[int] | None = None,
        halo: int = 0,
        static_file: str | Path | None = None,
        event_ids: Iterable[str] | None = None,
        max_target_time_index: int | None = None,
        observation_hours: float | None = None,
        max_open_files: int = 4,
    ) -> None:
        super().__init__()
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        if not self.dataset_root.is_dir():
            raise FileNotFoundError(f"Dataset root does not exist: {self.dataset_root}")
        self.context_steps = int(context_steps)
        self.forecast_steps = int(forecast_steps)
        self.start_stride = int(start_stride)
        self.halo = int(halo)
        self.max_open_files = int(max_open_files)
        if self.context_steps < 1 or self.forecast_steps < 1:
            raise ValueError("context_steps and forecast_steps must be positive")
        if self.start_stride < 1:
            raise ValueError("start_stride must be positive")
        if self.halo < 0:
            raise ValueError("halo cannot be negative")
        if self.max_open_files < 1:
            raise ValueError("max_open_files must be positive")
        if max_target_time_index is not None and int(max_target_time_index) < 0:
            raise ValueError("max_target_time_index cannot be negative")
        if observation_hours is not None:
            observation_hours = float(observation_hours)
            if not math.isfinite(observation_hours) or observation_hours < 0.0:
                raise ValueError("observation_hours must be finite and nonnegative")
        self.max_target_time_index = (
            None if max_target_time_index is None else int(max_target_time_index)
        )
        self.observation_hours = observation_hours

        static_path = self._resolve_static_path(static_file)
        self.static_path = static_path
        (
            self._static,
            self._static_signature,
            self.grid_metadata,
        ) = self._read_static(static_path)
        self.height, self.width = self._static["domain_mask"].shape

        if tile_size is None:
            self.tile_size = (self.height, self.width)
            if tile_stride is not None:
                raise ValueError("tile_stride requires tile_size")
            self.tile_stride = self.tile_size
        else:
            self.tile_size = _pair(tile_size, "tile_size")
            self.tile_stride = _pair(
                self.tile_size if tile_stride is None else tile_stride,
                "tile_stride",
            )
            if any(s > size for s, size in zip(self.tile_stride, self.tile_size)):
                raise ValueError(
                    "tile_stride cannot exceed tile_size (it would leave gaps)"
                )
        self._tiles = self._make_tiles()

        manifest_path = Path(split_manifest).expanduser().resolve()
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Split manifest does not exist: {manifest_path}")
        rows = self._read_manifest(manifest_path)
        self._validate_event_splits(rows)
        requested_events = (
            None if event_ids is None else {str(item) for item in event_ids}
        )
        selected = [row for row in rows if row.get("split", "").strip() == split]
        if requested_events is not None:
            available = {row["event_id"].strip() for row in selected}
            missing = requested_events - available
            if missing:
                raise ValueError(
                    f"Requested events are not in split {split!r}: {sorted(missing)}"
                )
            selected = [
                row for row in selected if row["event_id"].strip() in requested_events
            ]
        if not selected:
            raise ValueError(f"No trajectories found for split {split!r}")

        self.split = split
        self.trajectories: list[_Trajectory] = []
        self._samples_per_trajectory: list[int] = []
        minimum_steps = self.context_steps + self.forecast_steps
        for row in selected:
            trajectory = self._trajectory_from_row(row)
            if trajectory.timesteps < minimum_steps:
                continue
            if "height" in row and row["height"].strip():
                if int(row["height"]) != self.height:
                    raise ValueError(f"Height mismatch for {trajectory.scenario_id}")
            if "width" in row and row["width"].strip():
                if int(row["width"]) != self.width:
                    raise ValueError(f"Width mismatch for {trajectory.scenario_id}")
            maximum_start = trajectory.timesteps - minimum_steps
            if self.max_target_time_index is not None:
                maximum_start = min(
                    maximum_start,
                    self.max_target_time_index
                    - self.context_steps
                    - self.forecast_steps
                    + 1,
                )
            if self.observation_hours is not None:
                maximum_observed_index = int(
                    math.floor(
                        self.observation_hours * 3600.0 / trajectory.interval_seconds
                        + 1.0e-9
                    )
                )
                maximum_start = min(
                    maximum_start,
                    maximum_observed_index
                    - self.context_steps
                    - self.forecast_steps
                    + 1,
                )
            if maximum_start < 0:
                continue
            starts = 1 + maximum_start // self.start_stride
            self.trajectories.append(trajectory)
            self._samples_per_trajectory.append(starts * len(self._tiles))
        if not self.trajectories:
            raise ValueError(
                f"No {split!r} trajectory is long enough for "
                f"{self.context_steps} context + {self.forecast_steps} forecast states"
            )
        self._cumulative = np.cumsum(self._samples_per_trajectory).tolist()

        # These objects are process-local.  They are reset after fork/spawn and
        # when a Dataset instance moves between worker processes.
        self._owner_pid: int | None = None
        self._files: OrderedDict[str, h5py.File] = OrderedDict()
        self._validated_files: set[str] = set()

    @staticmethod
    def _read_manifest(path: Path) -> list[dict[str, str]]:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            required = {"event_id", "split", "timesteps", "trajectory_file"}
            missing = required - set(reader.fieldnames or ())
            if missing:
                raise ValueError(
                    f"Split manifest is missing columns: {sorted(missing)}"
                )
            return [dict(row) for row in reader]

    @staticmethod
    def _validate_event_splits(rows: Sequence[Mapping[str, str]]) -> None:
        assignments: dict[str, set[str]] = {}
        for row in rows:
            event = row["event_id"].strip()
            assignment = row["split"].strip()
            if not event or not assignment:
                raise ValueError(
                    "Every manifest row needs non-empty event_id and split"
                )
            assignments.setdefault(event, set()).add(assignment)
        leaked = {
            event: values for event, values in assignments.items() if len(values) > 1
        }
        if leaked:
            example = next(iter(leaked.items()))
            raise ValueError(
                "Whole-event split violation: event "
                f"{example[0]!r} occurs in splits {sorted(example[1])}"
            )

    def _resolve_static_path(self, value: str | Path | None) -> Path:
        candidate = (
            self.dataset_root / "static_grid.h5" if value is None else Path(value)
        )
        if not candidate.is_absolute():
            candidate = self.dataset_root / candidate
        candidate = candidate.expanduser().resolve()
        if not _is_beneath(candidate, self.dataset_root):
            raise ValueError(f"Static grid must be beneath dataset_root: {candidate}")
        if not candidate.is_file():
            raise FileNotFoundError(f"Static grid does not exist: {candidate}")
        return candidate

    @staticmethod
    def _read_static(
        path: Path,
    ) -> tuple[dict[str, np.ndarray], str | None, dict[str, Any]]:
        with h5py.File(str(path), "r") as handle:
            missing = [name for name in STATIC_NAMES if name not in handle]
            if missing:
                raise ValueError(f"Static grid is missing datasets: {missing}")
            arrays = {
                name: np.asarray(handle[name][()], dtype=np.float32)
                for name in STATIC_NAMES
            }
            signature = handle.attrs.get("static_grid_signature")
            metadata: dict[str, Any] = {
                "north_up": None,
                "dx_m": None,
                "dy_m": None,
            }
            if handle.attrs.get("format") == "rdycore_regular_grid_trajectory":
                if "x" not in handle or "y" not in handle:
                    raise ValueError(
                        "RDycore static grid must contain x and y coordinates"
                    )
                x = np.asarray(handle["x"][()], dtype=np.float64)
                y = np.asarray(handle["y"][()], dtype=np.float64)
                if (
                    x.ndim != 1
                    or y.ndim != 1
                    or x.size < 2
                    or y.size < 2
                    or not np.isfinite(x).all()
                    or not np.isfinite(y).all()
                ):
                    raise ValueError("static x/y coordinates are invalid")
                dx = np.diff(x)
                dy = np.diff(y)
                if not np.allclose(dx, dx[0], rtol=0.0, atol=1.0e-9):
                    raise ValueError("static x spacing is not uniform")
                if not np.allclose(dy, dy[0], rtol=0.0, atol=1.0e-9):
                    raise ValueError("static y spacing is not uniform")
                north_up = bool(handle.attrs.get("north_up", False))
                if dx[0] <= 0.0 or (north_up and dy[0] >= 0.0):
                    raise ValueError(
                        "static coordinate orientation contradicts north_up metadata"
                    )
                metadata = {
                    "north_up": north_up,
                    "dx_m": float(abs(dx[0])),
                    "dy_m": float(abs(dy[0])),
                }
        shapes = {array.shape for array in arrays.values()}
        if len(shapes) != 1 or len(next(iter(shapes))) != 2:
            raise ValueError(
                f"Static fields need one common [H,W] shape, found {shapes}"
            )
        return arrays, None if signature is None else str(signature), metadata

    def _relocated_path(self, row: Mapping[str, str]) -> Path:
        raw = Path(row["trajectory_file"].strip())
        candidates: list[Path] = []
        if not raw.is_absolute():
            candidates.append(self.dataset_root / raw)
        parts = raw.parts
        if "trajectories" in parts:
            index = parts.index("trajectories")
            candidates.append(self.dataset_root.joinpath(*parts[index:]))
        event = row["event_id"].strip()
        member_text = row.get("member_id", "").strip()
        if member_text:
            candidates.append(
                self.dataset_root
                / "trajectories"
                / event
                / f"member_{int(member_text):03d}.h5"
            )
        candidates.append(self.dataset_root / "trajectories" / event / raw.name)

        safe: list[Path] = []
        for candidate in candidates:
            resolved = candidate.expanduser().resolve()
            if _is_beneath(resolved, self.dataset_root) and resolved not in safe:
                safe.append(resolved)
        for candidate in safe:
            if candidate.is_file():
                return candidate
        searched = ", ".join(str(path) for path in safe)
        raise FileNotFoundError(
            f"Could not relocate trajectory for {row.get('scenario_id', event)!r} "
            f"beneath {self.dataset_root}; searched: {searched}"
        )

    def _trajectory_from_row(self, row: Mapping[str, str]) -> _Trajectory:
        event = row["event_id"].strip()
        scenario = row.get("scenario_id", "").strip() or event
        member = int(row.get("member_id", "0") or 0)
        timesteps = int(row["timesteps"])
        if timesteps < 1:
            raise ValueError(f"Invalid timestep count for {scenario!r}: {timesteps}")
        interval = float(row.get("output_interval_seconds", "3600") or 3600.0)
        if not math.isfinite(interval) or interval <= 0.0:
            raise ValueError(f"Invalid output interval for {scenario!r}: {interval}")
        return _Trajectory(
            scenario_id=scenario,
            event_id=event,
            member_id=member,
            timesteps=timesteps,
            interval_seconds=interval,
            path=self._relocated_path(row),
        )

    def _make_tiles(self) -> list[_Tile]:
        tile_h, tile_w = self.tile_size
        stride_h, stride_w = self.tile_stride
        result = []
        for y in range(0, self.height, stride_h):
            for x in range(0, self.width, stride_w):
                valid_height = min(tile_h, self.height - y)
                valid_width = min(tile_w, self.width - x)
                # Do not manufacture millions of loss-free samples over the
                # rectangular raster's inactive background.  Halos are still
                # loaded around every retained tile.
                active = self._static["domain_mask"][
                    y : y + valid_height, x : x + valid_width
                ]
                if not np.any(active > 0.5):
                    continue
                result.append(
                    _Tile(
                        y=y,
                        x=x,
                        height=tile_h,
                        width=tile_w,
                        valid_height=valid_height,
                        valid_width=valid_width,
                    )
                )
        if not result:
            raise ValueError("domain_mask has no active cells in any spatial tile")
        return result

    def __len__(self) -> int:
        return int(self._cumulative[-1])

    def _reset_process_local_state(self) -> None:
        pid = os.getpid()
        if self._owner_pid == pid:
            return
        # In a forked child, inherited HDF5 identifiers must not be reused.
        self._files = OrderedDict()
        self._validated_files = set()
        self._owner_pid = pid

    def _file(self, trajectory: _Trajectory) -> h5py.File:
        self._reset_process_local_state()
        key = str(trajectory.path)
        handle = self._files.pop(key, None)
        if handle is None:
            try:
                handle = h5py.File(key, "r", locking=False)
            except TypeError:  # h5py < 3.5 has no locking keyword.
                handle = h5py.File(key, "r")
        self._files[key] = handle
        while len(self._files) > self.max_open_files:
            _, old = self._files.popitem(last=False)
            old.close()
        if key not in self._validated_files:
            self._validate_trajectory(handle, trajectory)
            self._validated_files.add(key)
        return handle

    def _validate_trajectory(self, handle: h5py.File, trajectory: _Trajectory) -> None:
        required = ("state", "rainfall", "ocean_boundary_height")
        missing = [name for name in required if name not in handle]
        if missing:
            raise ValueError(f"{trajectory.path} is missing datasets: {missing}")
        state = handle["state"]
        rainfall = handle["rainfall"]
        ocean = handle["ocean_boundary_height"]
        if state.ndim != 4 or state.shape[1] != len(STATE_NAMES):
            raise ValueError(f"Invalid state shape in {trajectory.path}: {state.shape}")
        if rainfall.ndim != 4 or rainfall.shape[1] != 1:
            raise ValueError(
                f"Invalid rainfall shape in {trajectory.path}: {rainfall.shape}"
            )
        if state.shape[2:] != (self.height, self.width):
            raise ValueError(f"Trajectory/static shape mismatch in {trajectory.path}")
        if rainfall.shape[2:] != (self.height, self.width):
            raise ValueError(f"Rainfall/static shape mismatch in {trajectory.path}")
        if (
            min(state.shape[0], rainfall.shape[0], ocean.shape[0])
            < trajectory.timesteps
        ):
            raise ValueError(
                f"Manifest declares {trajectory.timesteps} steps but {trajectory.path} "
                "contains fewer"
            )
        signature = handle.attrs.get("static_grid_signature")
        if (
            signature is not None
            and self._static_signature is not None
            and str(signature) != self._static_signature
        ):
            raise ValueError(f"Static-grid signature mismatch in {trajectory.path}")

        attributes = handle.attrs
        if attributes.get("format") == "rdycore_regular_grid_trajectory":
            expected_strings = {
                "event_id": trajectory.event_id,
                "scenario_id": trajectory.scenario_id,
                "state_channel_names": "Height,MomentumX,MomentumY",
                "state_channel_units": "m,m2 s-1,m2 s-1",
                "rainfall_units": "mm h-1",
                "ocean_boundary_height_units": "m",
            }
            for name, expected in expected_strings.items():
                actual = attributes.get(name)
                if str(actual) != expected:
                    raise ValueError(
                        f"{trajectory.path} attribute {name!r} is {actual!r}, "
                        f"expected {expected!r}"
                    )
            if int(attributes.get("member_id", -1)) != trajectory.member_id:
                raise ValueError(f"Member identity mismatch in {trajectory.path}")
            if not bool(attributes.get("north_up", False)):
                raise ValueError(f"Trajectory is not marked north_up: {trajectory.path}")
            if bool(attributes.get("normalization_applied", True)):
                raise ValueError(f"Trajectory state must be dimensional: {trajectory.path}")
            interval = float(
                attributes.get("output_interval_seconds", float("nan"))
            )
            if not math.isclose(
                interval, trajectory.interval_seconds, rel_tol=0.0, abs_tol=1.0e-6
            ):
                raise ValueError(f"Output interval mismatch in {trajectory.path}")
            nonfinite = np.asarray(
                attributes.get("state_nonfinite_count_by_channel", [-1]),
                dtype=np.int64,
            )
            if nonfinite.shape != (3,) or np.any(nonfinite != 0):
                raise ValueError(
                    f"State non-finite metadata is invalid in {trajectory.path}"
                )
            if "time_seconds" not in handle:
                raise ValueError(
                    f"RDycore trajectory lacks time_seconds: {trajectory.path}"
                )
            times = np.asarray(handle["time_seconds"][()], dtype=np.float64)
            if (
                times.ndim != 1
                or times.size < trajectory.timesteps
                or not np.isfinite(times[: trajectory.timesteps]).all()
                or not np.allclose(
                    np.diff(times[: trajectory.timesteps]),
                    trajectory.interval_seconds,
                    rtol=0.0,
                    atol=1.0e-6,
                )
            ):
                raise ValueError(f"Invalid time_seconds cadence in {trajectory.path}")

    def _spatial_region(
        self, tile: _Tile
    ) -> tuple[slice, slice, tuple[int, int, int, int]]:
        top = tile.y - self.halo
        left = tile.x - self.halo
        bottom = tile.y + tile.height + self.halo
        right = tile.x + tile.width + self.halo
        y_slice = slice(max(0, top), min(self.height, bottom))
        x_slice = slice(max(0, left), min(self.width, right))
        pads = (
            max(0, -top),
            max(0, bottom - self.height),
            max(0, -left),
            max(0, right - self.width),
        )
        return y_slice, x_slice, pads

    @staticmethod
    def _pad(array: np.ndarray, pads: tuple[int, int, int, int]) -> np.ndarray:
        top, bottom, left, right = pads
        specification = [(0, 0)] * (array.ndim - 2) + [(top, bottom), (left, right)]
        if any(pads):
            array = np.pad(array, specification, mode="constant", constant_values=0)
        return np.ascontiguousarray(array, dtype=np.float32)

    def _locate(self, index: int) -> tuple[int, int, int]:
        length = len(self)
        if index < 0:
            index += length
        if index < 0 or index >= length:
            raise IndexError(index)
        trajectory_index = bisect.bisect_right(self._cumulative, index)
        preceding = (
            0 if trajectory_index == 0 else self._cumulative[trajectory_index - 1]
        )
        local = index - preceding
        start_number, tile_index = divmod(local, len(self._tiles))
        return trajectory_index, start_number * self.start_stride, tile_index

    def __getitem__(self, index: int) -> dict[str, Any]:
        trajectory_index, start, tile_index = self._locate(index)
        trajectory = self.trajectories[trajectory_index]
        tile = self._tiles[tile_index]
        handle = self._file(trajectory)

        context_stop = start + self.context_steps
        future_stop = context_stop + self.forecast_steps
        forcing_start = context_stop - 1
        forcing_stop = future_stop - 1
        y_slice, x_slice, pads = self._spatial_region(tile)

        context = self._pad(
            np.asarray(handle["state"][start:context_stop, :, y_slice, x_slice]), pads
        )
        future = self._pad(
            np.asarray(handle["state"][context_stop:future_stop, :, y_slice, x_slice]),
            pads,
        )
        rainfall = self._pad(
            np.asarray(
                handle["rainfall"][forcing_start:forcing_stop, :, y_slice, x_slice]
            ),
            pads,
        )
        ocean = np.asarray(
            handle["ocean_boundary_height"][forcing_start:forcing_stop],
            dtype=np.float32,
        )

        if "time_seconds" in handle:
            time = handle["time_seconds"]
            context_time = np.asarray(time[start:context_stop], dtype=np.float64)
            future_time = np.asarray(time[context_stop:future_stop], dtype=np.float64)
            forcing_time = np.asarray(
                time[forcing_start:forcing_stop], dtype=np.float64
            )
            dt = (
                np.asarray(time[forcing_start + 1 : forcing_stop + 1], dtype=np.float64)
                - forcing_time
            )
        else:
            all_indices = np.arange(trajectory.timesteps, dtype=np.float64)
            all_times = all_indices * trajectory.interval_seconds
            context_time = all_times[start:context_stop]
            future_time = all_times[context_stop:future_stop]
            forcing_time = all_times[forcing_start:forcing_stop]
            dt = np.full(
                self.forecast_steps, trajectory.interval_seconds, dtype=np.float64
            )

        static = {
            name: self._pad(array[y_slice, x_slice][None], pads)
            for name, array in self._static.items()
        }
        interior = torch.tensor(
            [
                self.halo,
                self.halo + tile.height,
                self.halo,
                self.halo + tile.width,
            ],
            dtype=torch.int64,
        )
        result: dict[str, Any] = {
            "context_state": torch.from_numpy(context),
            "current_state": torch.from_numpy(context[-1].copy()),
            "future_state": torch.from_numpy(future),
            "rainfall": torch.from_numpy(rainfall),
            "ocean_boundary_height": torch.from_numpy(ocean.copy()),
            "context_time_seconds": torch.from_numpy(context_time.copy()),
            "future_time_seconds": torch.from_numpy(future_time.copy()),
            "forcing_time_seconds": torch.from_numpy(forcing_time.copy()),
            "dt_seconds": torch.from_numpy(dt.copy()),
            "scenario_id": trajectory.scenario_id,
            "event_id": trajectory.event_id,
            "member_id": torch.tensor(trajectory.member_id, dtype=torch.int64),
            "start_index": torch.tensor(start, dtype=torch.int64),
            "context_indices": torch.arange(start, context_stop, dtype=torch.int64),
            "future_indices": torch.arange(
                context_stop, future_stop, dtype=torch.int64
            ),
            "forcing_indices": torch.arange(
                forcing_start, forcing_stop, dtype=torch.int64
            ),
            "tile_origin": torch.tensor([tile.y, tile.x], dtype=torch.int64),
            "global_shape": torch.tensor([self.height, self.width], dtype=torch.int64),
            "halo": torch.tensor(self.halo, dtype=torch.int64),
            "interior_slices": interior,
            "valid_interior_shape": torch.tensor(
                [tile.valid_height, tile.valid_width], dtype=torch.int64
            ),
        }
        result.update({name: torch.from_numpy(value) for name, value in static.items()})
        return result

    @property
    def open_file_count(self) -> int:
        """Number of trajectory handles open in the current worker process."""

        self._reset_process_local_state()
        return len(self._files)

    def close(self) -> None:
        for handle in getattr(self, "_files", {}).values():
            try:
                handle.close()
            except Exception:
                pass
        self._files = OrderedDict()
        self._validated_files = set()

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_owner_pid"] = None
        state["_files"] = OrderedDict()
        state["_validated_files"] = set()
        return state

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
