#!/usr/bin/env python3
"""Plot flood-emulator full-domain and USGS-gauge evaluation exports.

The input directory is produced by the Houston flood-rollout evaluator and
must contain ``gauge_timeseries.csv.gz``,
``ml_grid_usgs_gauge_mapping.csv``, ``spatial_error_summary.npz``, and a
``snapshots`` directory containing ``*.peak.npz`` files.
"""

from __future__ import annotations

import argparse
import math
import os
import re
from pathlib import Path
from typing import Iterable, Optional, Sequence

# Perlmutter home directories can be read-only from compute nodes.  Set a
# writable cache before importing matplotlib, while respecting any explicit
# user setting.
_CACHE_ROOT = Path(os.environ.get("TMPDIR", "/tmp"))
os.environ.setdefault(
    "XDG_CACHE_HOME", str(_CACHE_ROOT / f"ai-benchmark-cache-{os.getuid()}")
)
os.environ.setdefault(
    "MPLCONFIGDIR", str(_CACHE_ROOT / f"ai-benchmark-matplotlib-{os.getuid()}")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TRUTH_COLOR = "black"
EMULATOR_COLOR = "#0072B2"
MODEL_LABEL = "Emulator"
REQUIRED_TIMESERIES_COLUMNS = {
    "event_id",
    "scenario_id",
    "member_id",
    "station_id",
    "station_name",
    "lead_hour",
    "rdycore_depth_m",
    "predicted_depth_m",
}
REQUIRED_MAPPING_COLUMNS = {
    "station_id",
    "model_x",
    "model_y",
}
REQUIRED_SPATIAL_KEYS = {
    "x",
    "y",
    "domain",
    "rmse_m",
    "mae_m",
    "bias_m",
}
REQUIRED_PEAK_KEYS = {"peak_truth", "peak_prediction"}


class InputValidationError(RuntimeError):
    """Raised when the analysis export is absent or internally inconsistent."""


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create emulator hydrographs, peak-flood maps, and spatial error maps "
            "from an analysis export directory."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing the rollout-analysis export files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory in which figures will be created.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="PNG resolution (default: 200).",
    )
    parser.add_argument(
        "--model-label",
        default="Emulator",
        help="Emulator label used in legends and map titles (default: Emulator).",
    )
    args = parser.parse_args(argv)
    if args.dpi < 72:
        parser.error("--dpi must be at least 72")
    return args


def _require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise InputValidationError(f"Missing {description}: {path}")


def _require_columns(frame: pd.DataFrame, required: set[str], path: Path) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise InputValidationError(
            f"{path} is missing required column(s): {', '.join(missing)}"
        )


def _require_keys(keys: Iterable[str], required: set[str], path: Path) -> None:
    missing = sorted(required.difference(keys))
    if missing:
        raise InputValidationError(
            f"{path} is missing required array(s): {', '.join(missing)}"
        )


def _safe_filename(value: object) -> str:
    result = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return result or "unnamed"


def _member_is_one(values: pd.Series) -> pd.Series:
    text = values.astype(str).str.strip().str.lower().str.removeprefix("m")
    numeric = pd.to_numeric(text, errors="coerce")
    return numeric.eq(1)


def load_timeseries(path: Path) -> pd.DataFrame:
    _require_file(path, "gauge time-series table")
    try:
        frame = pd.read_csv(
            path,
            dtype={"event_id": str, "scenario_id": str, "station_id": str},
        )
    except Exception as exc:
        raise InputValidationError(f"Could not read {path}: {exc}") from exc
    _require_columns(frame, REQUIRED_TIMESERIES_COLUMNS, path)
    if frame.empty:
        raise InputValidationError(f"Gauge time-series table is empty: {path}")

    frame = frame.copy()
    frame["event_id"] = frame["event_id"].astype(str)
    frame["scenario_id"] = frame["scenario_id"].astype(str)
    frame["station_id"] = frame["station_id"].astype(str).str.zfill(8)
    for column in ("lead_hour", "rdycore_depth_m", "predicted_depth_m"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame["lead_hour"].isna().any():
        raise InputValidationError(f"{path} contains non-numeric lead_hour values")
    if not np.isfinite(frame["rdycore_depth_m"].to_numpy(dtype=float)).any():
        raise InputValidationError(f"{path} has no finite RDycore depth values")
    if not np.isfinite(frame["predicted_depth_m"].to_numpy(dtype=float)).any():
        raise InputValidationError(f"{path} has no finite predicted depth values")

    member_one = frame.loc[_member_is_one(frame["member_id"])].copy()
    if member_one.empty:
        raise InputValidationError(f"{path} contains no ensemble member 001 rows")
    duplicate_key = ["scenario_id", "station_id", "lead_hour"]
    duplicates = member_one.duplicated(duplicate_key, keep=False)
    if duplicates.any():
        example = member_one.loc[duplicates, duplicate_key].iloc[0].to_dict()
        raise InputValidationError(
            f"{path} contains duplicate member-001 samples; example key: {example}"
        )
    return member_one.sort_values(
        ["event_id", "scenario_id", "station_id", "lead_hour"]
    ).reset_index(drop=True)


def load_mapping(path: Path) -> pd.DataFrame:
    _require_file(path, "USGS gauge-to-grid mapping")
    try:
        frame = pd.read_csv(path, dtype={"station_id": str})
    except Exception as exc:
        raise InputValidationError(f"Could not read {path}: {exc}") from exc
    _require_columns(frame, REQUIRED_MAPPING_COLUMNS, path)
    if frame.empty:
        raise InputValidationError(f"Gauge mapping is empty: {path}")
    frame = frame.copy()
    frame["station_id"] = frame["station_id"].astype(str).str.zfill(8)
    if frame["station_id"].duplicated().any():
        duplicate = frame.loc[frame["station_id"].duplicated(), "station_id"].iloc[0]
        raise InputValidationError(f"{path} maps station {duplicate} more than once")
    if len(frame) != 14:
        raise InputValidationError(
            f"{path} must contain the 14 core USGS gauges; found {len(frame)}"
        )
    for column in ("model_x", "model_y"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if not np.isfinite(frame[column].to_numpy(dtype=float)).all():
            raise InputValidationError(f"{path} contains invalid {column} coordinates")
    return frame.sort_values("station_id").reset_index(drop=True)


def load_spatial(path: Path) -> dict[str, np.ndarray]:
    _require_file(path, "spatial error summary")
    try:
        with np.load(path, allow_pickle=False) as archive:
            _require_keys(archive.files, REQUIRED_SPATIAL_KEYS, path)
            arrays = {key: np.asarray(archive[key]) for key in REQUIRED_SPATIAL_KEYS}
    except InputValidationError:
        raise
    except Exception as exc:
        raise InputValidationError(f"Could not read {path}: {exc}") from exc

    domain = arrays["domain"]
    if domain.ndim != 2:
        raise InputValidationError(
            f"domain in {path} must be two-dimensional; found shape {domain.shape}"
        )
    shape = domain.shape
    for key in ("rmse_m", "mae_m", "bias_m"):
        if arrays[key].shape != shape:
            raise InputValidationError(
                f"{key} shape {arrays[key].shape} does not match domain shape {shape}"
            )
    arrays["domain"] = domain.astype(bool)
    arrays["x"], arrays["y"] = _coordinate_grids(
        arrays["x"], arrays["y"], shape, path
    )
    if not arrays["domain"].any():
        raise InputValidationError(f"domain in {path} contains no active cells")
    return arrays


def _coordinate_grids(
    x: np.ndarray, y: np.ndarray, shape: tuple[int, int], path: Path
) -> tuple[np.ndarray, np.ndarray]:
    if x.ndim == 1 and y.ndim == 1:
        if x.size != shape[1] or y.size != shape[0]:
            raise InputValidationError(
                f"Coordinate lengths ({x.size}, {y.size}) in {path} do not match "
                f"raster shape {shape}"
            )
        return np.meshgrid(x, y)
    if x.shape != shape or y.shape != shape:
        raise InputValidationError(
            f"x/y in {path} must both match raster shape {shape}, or both be "
            "one-dimensional coordinate vectors"
        )
    return x, y


def validate_station_coverage(
    timeseries: pd.DataFrame, mapping: pd.DataFrame, timeseries_path: Path
) -> None:
    observed = set(timeseries["station_id"].unique())
    mapped = set(mapping["station_id"].unique())
    missing = sorted(mapped.difference(observed))
    if missing:
        raise InputValidationError(
            f"{timeseries_path} has no member-001 records for mapped station(s): "
            f"{', '.join(missing)}"
        )


def _station_label(row: pd.Series, timeseries: pd.DataFrame) -> str:
    station_id = str(row["station_id"])
    if "station_name" in row and pd.notna(row["station_name"]):
        name = str(row["station_name"]).strip()
    else:
        candidates = timeseries.loc[
            timeseries["station_id"].eq(station_id), "station_name"
        ].dropna()
        name = str(candidates.iloc[0]).strip() if not candidates.empty else ""
    return f"USGS {station_id}: {name}" if name else f"USGS {station_id}"


def _line_panel(axis: plt.Axes, data: pd.DataFrame, title: str) -> None:
    data = data.sort_values("lead_hour")
    axis.plot(
        data["lead_hour"],
        data["rdycore_depth_m"],
        color=TRUTH_COLOR,
        linewidth=2.1,
        label="RDycore",
    )
    axis.plot(
        data["lead_hour"],
        data["predicted_depth_m"],
        color=EMULATOR_COLOR,
        linewidth=1.5,
        label=MODEL_LABEL,
    )
    axis.set_title(title, fontsize=9)
    axis.set_ylim(bottom=0.0)
    axis.grid(alpha=0.25)


def plot_event_hydrographs(
    timeseries: pd.DataFrame,
    mapping: pd.DataFrame,
    output_dir: Path,
    dpi: int,
) -> list[Path]:
    """Create one all-gauge hydrograph grid per member-001 test event."""
    directory = output_dir / "event_hydrographs"
    directory.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    nstations = len(mapping)
    ncols = 4 if nstations > 9 else min(3, nstations)
    nrows = math.ceil(nstations / ncols)

    for event_id in sorted(timeseries["event_id"].unique()):
        event = timeseries.loc[timeseries["event_id"].eq(event_id)]
        scenarios = sorted(event["scenario_id"].unique())
        if len(scenarios) != 1:
            raise InputValidationError(
                f"Event {event_id} has {len(scenarios)} member-001 scenarios; expected one"
            )
        figure, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(3.8 * ncols, 2.65 * nrows),
            squeeze=False,
        )
        for axis, (_, station) in zip(axes.flat, mapping.iterrows()):
            station_id = str(station["station_id"])
            data = event.loc[event["station_id"].eq(station_id)]
            if data.empty:
                raise InputValidationError(
                    f"Event {event_id} has no member-001 data for station {station_id}"
                )
            _line_panel(axis, data, _station_label(station, timeseries))
        for axis in axes.flat[nstations:]:
            axis.set_axis_off()
        for row in range(nrows):
            axes[row, 0].set_ylabel("Water depth (m)")
        for index, axis in enumerate(axes.flat[:nstations]):
            if index + ncols >= nstations:
                axis.set_xlabel("Forecast lead (h)")
        handles, labels = axes.flat[0].get_legend_handles_labels()
        figure.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.94),
            ncol=2,
            frameon=False,
        )
        figure.suptitle(
            f"Test-event hydrographs: {event_id}, ensemble member 001",
            fontsize=14,
            y=0.995,
        )
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))
        output = directory / f"{_safe_filename(event_id)}.hydrographs.png"
        figure.savefig(output, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
        outputs.append(output)
    return outputs


def plot_station_hydrographs(
    timeseries: pd.DataFrame,
    mapping: pd.DataFrame,
    output_dir: Path,
    dpi: int,
) -> list[Path]:
    """Create one multi-event member-001 hydrograph grid per gauge."""
    directory = output_dir / "station_hydrographs"
    directory.mkdir(parents=True, exist_ok=True)
    events = sorted(timeseries["event_id"].unique())
    ncols = min(3, len(events))
    nrows = math.ceil(len(events) / ncols)
    outputs: list[Path] = []

    for _, station in mapping.iterrows():
        station_id = str(station["station_id"])
        station_data = timeseries.loc[timeseries["station_id"].eq(station_id)]
        figure, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(5.0 * ncols, 3.5 * nrows),
            squeeze=False,
        )
        for axis, event_id in zip(axes.flat, events):
            event = station_data.loc[station_data["event_id"].eq(event_id)]
            if event.empty:
                raise InputValidationError(
                    f"Station {station_id} has no member-001 data for event {event_id}"
                )
            _line_panel(axis, event, str(event_id).replace("AORC_", ""))
            axis.set_xlabel("Forecast lead (h)")
            axis.set_ylabel("Water depth (m)")
        for axis in axes.flat[len(events):]:
            axis.set_axis_off()
        handles, labels = axes.flat[0].get_legend_handles_labels()
        figure.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.93),
            ncol=2,
            frameon=False,
        )
        figure.suptitle(
            f"{_station_label(station, timeseries)}\nHeld-out ensemble member 001",
            fontsize=14,
            y=0.995,
        )
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))
        output = directory / f"{_safe_filename(station_id)}.hydrographs.png"
        figure.savefig(output, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
        outputs.append(output)
    return outputs


def _finite_percentile(
    arrays: Iterable[np.ndarray], percentile: float, floor: float
) -> float:
    finite_parts = []
    for array in arrays:
        values = np.asarray(array, dtype=float)
        values = values[np.isfinite(values)]
        if values.size:
            finite_parts.append(values)
    if not finite_parts:
        raise InputValidationError("Cannot determine a color limit from all-NaN data")
    return max(floor, float(np.nanpercentile(np.concatenate(finite_parts), percentile)))


def _map_panel(
    axis: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    field: np.ndarray,
    title: str,
    cmap: str,
    vmin: float,
    vmax: float,
    mapping: pd.DataFrame,
):
    artist = axis.pcolormesh(
        x / 1000.0,
        y / 1000.0,
        np.ma.masked_invalid(field),
        shading="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        rasterized=True,
    )
    axis.scatter(
        mapping["model_x"] / 1000.0,
        mapping["model_y"] / 1000.0,
        s=13,
        c="white",
        edgecolors="black",
        linewidths=0.45,
        zorder=3,
    )
    axis.set_title(title, fontsize=11)
    axis.set_aspect("equal")
    axis.set_xlabel("Easting (km)")
    axis.set_ylabel("Northing (km)")
    return artist


def _snapshot_event_label(scenario: str, timeseries: pd.DataFrame) -> str:
    matches = timeseries.loc[
        timeseries["scenario_id"].eq(scenario), "event_id"
    ].unique()
    if len(matches) == 1:
        return str(matches[0])
    return scenario.rsplit("__m", 1)[0]


def plot_peak_flood_maps(
    input_dir: Path,
    spatial: dict[str, np.ndarray],
    mapping: pd.DataFrame,
    timeseries: pd.DataFrame,
    output_dir: Path,
    dpi: int,
) -> list[Path]:
    """Plot event-maximum depths and emulator-minus-RDycore error."""
    snapshot_dir = input_dir / "snapshots"
    if not snapshot_dir.is_dir():
        raise InputValidationError(f"Missing peak snapshot directory: {snapshot_dir}")
    snapshots = sorted(snapshot_dir.glob("*.peak.npz"))
    if not snapshots:
        raise InputValidationError(f"No *.peak.npz files found in {snapshot_dir}")
    directory = output_dir / "peak_flood_maps"
    directory.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    limit_rows: list[dict[str, object]] = []
    domain = spatial["domain"]
    x, y = spatial["x"], spatial["y"]

    for path in snapshots:
        try:
            with np.load(path, allow_pickle=False) as archive:
                _require_keys(archive.files, REQUIRED_PEAK_KEYS, path)
                truth_raw = np.asarray(archive["peak_truth"], dtype=float)
                prediction_raw = np.asarray(archive["peak_prediction"], dtype=float)
        except InputValidationError:
            raise
        except Exception as exc:
            raise InputValidationError(f"Could not read {path}: {exc}") from exc
        if truth_raw.shape != domain.shape or prediction_raw.shape != domain.shape:
            raise InputValidationError(
                f"Peak arrays in {path} have shapes {truth_raw.shape} and "
                f"{prediction_raw.shape}; expected {domain.shape}"
            )
        truth = np.where(domain, truth_raw, np.nan)
        prediction = np.where(domain, prediction_raw, np.nan)
        error = prediction - truth
        depth_max = _finite_percentile((truth, prediction), 99.5, 0.1)
        error_limit = _finite_percentile((np.abs(error),), 99.0, 0.02)

        figure, axes = plt.subplots(
            1, 3, figsize=(13.2, 4.5), constrained_layout=True
        )
        depth_artist = _map_panel(
            axes[0], x, y, truth, "RDycore peak depth", "Blues", 0.0,
            depth_max, mapping,
        )
        _map_panel(
            axes[1], x, y, prediction, f"{MODEL_LABEL} peak depth", "Blues", 0.0,
            depth_max, mapping,
        )
        error_artist = _map_panel(
            axes[2], x, y, error, f"{MODEL_LABEL} minus RDycore", "RdBu_r",
            -error_limit, error_limit, mapping,
        )
        depth_colorbar = figure.colorbar(
            depth_artist, ax=axes[:2], location="bottom", shrink=0.78, pad=0.12
        )
        depth_colorbar.set_label("Event-maximum water depth (m)")
        error_colorbar = figure.colorbar(
            error_artist, ax=axes[2], location="bottom", shrink=0.82, pad=0.12
        )
        error_colorbar.set_label("Peak-depth error (m)")
        scenario = path.name[: -len(".peak.npz")]
        event_label = _snapshot_event_label(scenario, timeseries)
        limit_rows.append(
            {
                "scenario_id": scenario,
                "event_id": event_label,
                "peak_depth_vmax_m": depth_max,
                "peak_depth_error_abs_limit_m": error_limit,
            }
        )
        figure.suptitle(
            f"Peak flood depth: {event_label}, ensemble member 001", fontsize=14
        )
        output = directory / f"{_safe_filename(scenario)}.peak_flood.png"
        figure.savefig(output, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
        outputs.append(output)
    pd.DataFrame(limit_rows).to_csv(
        directory / "peak_map_color_limits.csv", index=False
    )
    return outputs


def plot_spatial_errors(
    spatial: dict[str, np.ndarray],
    mapping: pd.DataFrame,
    output_dir: Path,
    dpi: int,
) -> Path:
    """Plot time/trajectory aggregated depth RMSE, MAE, and bias fields."""
    directory = output_dir / "spatial_error_maps"
    directory.mkdir(parents=True, exist_ok=True)
    domain = spatial["domain"]
    x, y = spatial["x"], spatial["y"]
    rmse = np.where(domain, spatial["rmse_m"], np.nan)
    mae = np.where(domain, spatial["mae_m"], np.nan)
    bias = np.where(domain, spatial["bias_m"], np.nan)
    error_max = _finite_percentile((rmse, mae), 99.0, 0.01)
    bias_limit = _finite_percentile((np.abs(bias),), 99.0, 0.01)

    figure, axes = plt.subplots(1, 3, figsize=(13.2, 4.5), constrained_layout=True)
    error_artist = _map_panel(
        axes[0], x, y, rmse, "Depth RMSE", "magma", 0.0, error_max, mapping
    )
    _map_panel(
        axes[1], x, y, mae, "Depth MAE", "magma", 0.0, error_max, mapping
    )
    bias_artist = _map_panel(
        axes[2], x, y, bias, "Depth bias", "RdBu_r", -bias_limit,
        bias_limit, mapping,
    )
    error_colorbar = figure.colorbar(
        error_artist, ax=axes[:2], location="bottom", shrink=0.78, pad=0.12
    )
    error_colorbar.set_label("Water-depth error (m)")
    bias_colorbar = figure.colorbar(
        bias_artist, ax=axes[2], location="bottom", shrink=0.82, pad=0.12
    )
    bias_colorbar.set_label(f"{MODEL_LABEL} minus RDycore (m)")
    figure.suptitle(f"Full-domain {MODEL_LABEL} rollout errors", fontsize=14)
    output = directory / "full_domain_spatial_errors.png"
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return output


def generate_figures(input_dir: Path, output_dir: Path, dpi: int = 200) -> list[Path]:
    input_dir = input_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise InputValidationError(f"Input directory does not exist: {input_dir}")

    timeseries_path = input_dir / "gauge_timeseries.csv.gz"
    mapping_path = input_dir / "ml_grid_usgs_gauge_mapping.csv"
    spatial_path = input_dir / "spatial_error_summary.npz"
    timeseries = load_timeseries(timeseries_path)
    mapping = load_mapping(mapping_path)
    spatial = load_spatial(spatial_path)
    validate_station_coverage(timeseries, mapping, timeseries_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    outputs.extend(plot_event_hydrographs(timeseries, mapping, output_dir, dpi))
    outputs.extend(plot_station_hydrographs(timeseries, mapping, output_dir, dpi))
    outputs.extend(
        plot_peak_flood_maps(
            input_dir, spatial, mapping, timeseries, output_dir, dpi
        )
    )
    outputs.append(plot_spatial_errors(spatial, mapping, output_dir, dpi))
    return outputs


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    global MODEL_LABEL
    MODEL_LABEL = args.model_label
    try:
        outputs = generate_figures(args.input_dir, args.output_dir, args.dpi)
    except InputValidationError as exc:
        raise SystemExit(f"Input validation failed: {exc}") from exc
    print(f"Created {len(outputs)} figure(s) in {args.output_dir.resolve()}")
    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
