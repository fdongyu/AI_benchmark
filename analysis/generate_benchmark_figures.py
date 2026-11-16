#!/usr/bin/env python3
"""Generate comparable Houston flood-emulator tables and figures.

The script consumes the standardized physical-rollout exports under
``data/evaluations``. It validates that every included model uses the same
test trajectories, gauges, truth hydrographs, grid, and peak-depth snapshots
before drawing a comparison. Generated products go to ``analysis/figures`` by
default; that directory is intentionally excluded from Git.

Only PINO-ResU-Net-large currently has an audited physical export. Use
``--allow-incomplete`` to render the available result and an explicit model
availability table. Without that flag, automatic discovery fails until every
reportable AR32 model has a compatible export.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = Path(__file__).resolve().parent
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

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

import plot_flood_evaluation as single


DEFAULT_EVALUATION_ROOT = ROOT / "data" / "evaluations"
DEFAULT_OUTPUT = ANALYSIS_DIR / "figures"
PUBLICATION_MANIFEST = ROOT / "publication_manifest.json"
DOMAIN_KEYS = ("scenario_id", "event_id", "member_id")
GAUGE_KEYS = DOMAIN_KEYS + ("station_id",)
TIMESERIES_KEYS = (
    "scenario_id",
    "event_id",
    "member_id",
    "station_id",
    "lead_hour",
    "time_utc_unix_seconds",
)
SUMMARY_COLUMNS = {
    "metric",
    "metric_label",
    "higher_is_better",
    "test_events",
    "finite_event_values",
    "median",
    "q25",
    "q75",
}
DOMAIN_PLOT_METRICS = (
    "depth_rmse_m",
    "depth_mae_m",
    "depth_correlation",
    "depth_nse",
    "depth_kge",
    "depth_csi_at_0p1m",
    "peak_depth_rmse_m",
    "mse_skill_vs_persistence",
)
GAUGE_PLOT_METRICS = (
    "depth_rmse_m",
    "depth_mae_m",
    "depth_correlation",
    "depth_nse",
    "depth_kge",
    "depth_csi_at_0p1m",
    "absolute_peak_depth_error_m",
    "absolute_peak_timing_error_hours",
)
COLORS = (
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#CC79A7",
    "#D55E00",
    "#6A3D9A",
    "#56B4E9",
)


class ComparisonInputError(RuntimeError):
    """Raised when exports cannot support a scientifically paired comparison."""


@dataclass
class Export:
    model_id: str
    label: str
    directory: Path
    metadata: dict[str, Any]
    domain_summary: pd.DataFrame
    gauge_summary: pd.DataFrame
    domain_population: pd.DataFrame
    gauge_population: pd.DataFrame
    timeseries: pd.DataFrame
    mapping: pd.DataFrame
    spatial: dict[str, np.ndarray]
    snapshots: tuple[str, ...]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluation-root",
        type=Path,
        default=DEFAULT_EVALUATION_ROOT,
        help="Root scanned for standardized evaluation directories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Ignored directory receiving generated tables and figures.",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        metavar="MODEL_ID=EXPORT_DIR",
        help=(
            "Explicit model export. Repeat for multiple models. When omitted, "
            "all valid children of --evaluation-root are discovered."
        ),
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Render available exports and record missing publication models.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Replace an existing generated output directory atomically. Without "
            "this flag, a non-empty output directory is refused."
        ),
    )
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args(argv)
    if args.dpi < 72:
        parser.error("--dpi must be at least 72")
    return args


def _load_publication_models() -> list[dict[str, Any]]:
    if not PUBLICATION_MANIFEST.is_file():
        raise ComparisonInputError(f"Missing publication manifest: {PUBLICATION_MANIFEST}")
    payload = json.loads(PUBLICATION_MANIFEST.read_text(encoding="utf-8"))
    models = payload.get("models")
    if not isinstance(models, list) or not models:
        raise ComparisonInputError("publication_manifest.json has no model records")
    return models


def _safe_filename(value: object) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return text or "unnamed"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalize_ids(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in ("scenario_id", "event_id"):
        if column in result:
            result[column] = result[column].astype(str)
    if "member_id" in result:
        member = result["member_id"].astype(str).str.lower().str.removeprefix("m")
        result["member_id"] = pd.to_numeric(member, errors="raise").astype(int)
    if "station_id" in result:
        result["station_id"] = (
            result["station_id"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(8)
        )
    return result


def _load_population(path: Path, keys: Sequence[str]) -> pd.DataFrame:
    if not path.is_file():
        raise ComparisonInputError(f"Missing population table: {path}")
    frame = _normalize_ids(pd.read_csv(path, dtype=str))
    missing = [key for key in keys if key not in frame]
    if missing:
        raise ComparisonInputError(f"{path} lacks comparison keys: {missing}")
    if "status" in frame:
        frame = frame.loc[frame["status"].astype(str).eq("complete")]
    duplicated = frame.duplicated(list(keys), keep=False)
    if duplicated.any():
        example = frame.loc[duplicated, list(keys)].iloc[0].to_dict()
        raise ComparisonInputError(
            f"Duplicate comparison records in {path}; example: {example}"
        )
    frame = frame.sort_values(list(keys))
    if frame.empty:
        raise ComparisonInputError(f"No complete comparison records in {path}")
    return frame.reset_index(drop=True)


def _load_summary(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise ComparisonInputError(f"Missing summary table: {path}")
    frame = pd.read_csv(path)
    missing = sorted(SUMMARY_COLUMNS.difference(frame.columns))
    if missing:
        raise ComparisonInputError(f"{path} lacks summary columns: {missing}")
    if frame["metric"].duplicated().any():
        raise ComparisonInputError(f"Duplicate metric rows in {path}")
    for column in ("median", "q25", "q75"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for _, row in frame.iterrows():
        test_events = int(row["test_events"])
        finite_events = int(row["finite_event_values"])
        if test_events != 6 or not 0 <= finite_events <= test_events:
            raise ComparisonInputError(
                f"Invalid event counts for {row['metric']} in {path}: "
                f"{finite_events}/{test_events}"
            )
        if finite_events:
            values = np.asarray([row["q25"], row["median"], row["q75"]], dtype=float)
            if not np.isfinite(values).all() or not values[0] <= values[1] <= values[2]:
                raise ComparisonInputError(
                    f"Invalid event quartiles for {row['metric']} in {path}"
                )
    return frame


def _load_event_metrics(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise ComparisonInputError(f"Missing event metric table: {path}")
    frame = _normalize_ids(pd.read_csv(path, dtype={"event_id": str}))
    if "event_id" not in frame:
        raise ComparisonInputError(f"{path} lacks event_id")
    if frame["event_id"].duplicated().any():
        raise ComparisonInputError(f"Duplicate event rows in {path}")
    if len(frame) != 6:
        raise ComparisonInputError(f"{path} must contain six held-out events; found {len(frame)}")
    return frame.sort_values("event_id").reset_index(drop=True)


def _verify_summary(
    summary: pd.DataFrame, event_metrics: pd.DataFrame, summary_path: Path
) -> None:
    event_ids = event_metrics["event_id"].astype(str)
    if event_ids.nunique() != 6:
        raise ComparisonInputError(f"{summary_path}: event population is not six unique events")
    for _, row in summary.iterrows():
        metric = str(row["metric"])
        if metric not in event_metrics:
            raise ComparisonInputError(
                f"{summary_path}: metric {metric} is absent from its event table"
            )
        values = pd.to_numeric(event_metrics[metric], errors="coerce").dropna()
        expected = np.asarray(
            [values.median(), values.quantile(0.25), values.quantile(0.75)],
            dtype=float,
        )
        recorded = np.asarray([row["median"], row["q25"], row["q75"]], dtype=float)
        if len(values) != int(row["finite_event_values"]) or not np.allclose(
            expected, recorded, rtol=1.0e-12, atol=1.0e-14, equal_nan=True
        ):
            raise ComparisonInputError(
                f"{summary_path}: summary values do not reproduce event metric {metric}"
            )


def _infer_model_id(
    directory: Path, metadata: dict[str, Any], models: Sequence[dict[str, Any]]
) -> str:
    candidates = sorted(
        (str(model["id"]) for model in models), key=len, reverse=True
    )
    values = (
        directory.name,
        str(metadata.get("model_key", "")),
        str(metadata.get("model", "")),
    )
    normalized = [re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") for value in values]
    for candidate in candidates:
        if any(value == candidate or value.startswith(candidate + "_") for value in normalized):
            return candidate
    return normalized[0] or "unknown_model"


def _load_export(
    directory: Path,
    models: Sequence[dict[str, Any]],
    explicit_model_id: Optional[str] = None,
) -> Export:
    directory = directory.expanduser().resolve()
    metadata_path = directory / "export_metadata.json"
    if not metadata_path.is_file():
        raise ComparisonInputError(f"Missing export metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    model_id = explicit_model_id or _infer_model_id(directory, metadata, models)
    names = {str(model["id"]): str(model["publication_name"]) for model in models}
    label = names.get(model_id, str(metadata.get("model", model_id)))

    domain_summary_path = directory / "domain_summary_metrics.csv"
    gauge_summary_path = directory / "gauge_summary_metrics.csv"
    domain_summary = _load_summary(domain_summary_path)
    gauge_summary = _load_summary(gauge_summary_path)
    domain_event = _load_event_metrics(directory / "domain_metrics_by_event.csv")
    gauge_event = _load_event_metrics(directory / "gauge_metrics_by_event.csv")
    _verify_summary(domain_summary, domain_event, domain_summary_path)
    _verify_summary(gauge_summary, gauge_event, gauge_summary_path)
    domain_population = _load_population(
        directory / "domain_metrics_by_trajectory.csv", DOMAIN_KEYS
    )
    gauge_population = _load_population(
        directory / "gauge_metrics_by_trajectory.csv", GAUGE_KEYS
    )
    timeseries = single.load_timeseries(directory / "gauge_timeseries.csv.gz")
    mapping = single.load_mapping(directory / "ml_grid_usgs_gauge_mapping.csv")
    single.validate_station_coverage(
        timeseries, mapping, directory / "gauge_timeseries.csv.gz"
    )
    spatial = single.load_spatial(directory / "spatial_error_summary.npz")
    snapshot_dir = directory / "snapshots"
    snapshots = tuple(path.name for path in sorted(snapshot_dir.glob("*.peak.npz")))
    if not snapshots:
        raise ComparisonInputError(f"No peak snapshots in {snapshot_dir}")

    complete = int(metadata.get("complete_rollouts", len(domain_population)))
    selected = int(metadata.get("selected_trajectories", len(domain_population)))
    if complete != 192 or selected != 192 or len(domain_population) != 192:
        raise ComparisonInputError(
            f"{directory}: benchmark requires 192 complete trajectories; "
            f"found complete/selected/table={complete}/{selected}/{len(domain_population)}"
        )
    events = sorted(domain_population["event_id"].unique())
    members_per_event = domain_population.groupby("event_id")["member_id"].nunique()
    rows_per_event = domain_population.groupby("event_id").size()
    if (
        len(events) != 6
        or not members_per_event.eq(32).all()
        or not rows_per_event.eq(32).all()
    ):
        raise ComparisonInputError(
            f"{directory}: benchmark requires six events with 32 unique members each"
        )
    if set(domain_event["event_id"]) != set(events) or set(gauge_event["event_id"]) != set(events):
        raise ComparisonInputError(f"{directory}: event-summary population mismatch")
    if len(mapping) != 14:
        raise ComparisonInputError(f"{directory}: benchmark requires 14 core gauges")
    expected_gauges = 192 * 14
    if len(gauge_population) != expected_gauges:
        raise ComparisonInputError(
            f"{directory}: expected {expected_gauges} trajectory-gauge records; "
            f"found {len(gauge_population)}"
        )
    gauges_per_trajectory = gauge_population.groupby(list(DOMAIN_KEYS))["station_id"].nunique()
    if not gauges_per_trajectory.eq(14).all():
        raise ComparisonInputError(
            f"{directory}: every trajectory must contain all 14 gauge locations"
        )
    mapping_stations = set(mapping["station_id"].astype(str))
    population_stations = set(gauge_population["station_id"].astype(str))
    timeseries_stations = set(timeseries["station_id"].astype(str))
    if population_stations != mapping_stations or timeseries_stations != mapping_stations:
        raise ComparisonInputError(
            f"{directory}: gauge tables do not use the exact 14-station mapping"
        )
    expected_hydrographs = {
        (event_id, station_id)
        for event_id in events
        for station_id in mapping_stations
    }
    observed_hydrographs = set(
        timeseries.loc[:, ["event_id", "station_id"]]
        .astype(str)
        .itertuples(index=False, name=None)
    )
    if observed_hydrographs != expected_hydrographs:
        raise ComparisonInputError(
            f"{directory}: member-001 hydrographs do not cover all six events "
            "and 14 gauge locations"
        )
    required_truth_diagnostics = {
        "forecast_hours_completed",
        "persistence_depth_rmse_m",
    }
    missing_truth = sorted(required_truth_diagnostics.difference(domain_population.columns))
    if missing_truth:
        raise ComparisonInputError(
            f"{directory}: domain table lacks truth diagnostics: {missing_truth}"
        )
    if set(pd.to_numeric(timeseries["member_id"], errors="raise").astype(int)) != {1}:
        raise ComparisonInputError(f"{directory}: hydrograph export is not member 001")
    return Export(
        model_id=model_id,
        label=label,
        directory=directory,
        metadata=metadata,
        domain_summary=domain_summary,
        gauge_summary=gauge_summary,
        domain_population=domain_population,
        gauge_population=gauge_population,
        timeseries=timeseries,
        mapping=mapping,
        spatial=spatial,
        snapshots=snapshots,
    )


def _parse_explicit_model(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ComparisonInputError(
            f"--model must be MODEL_ID=EXPORT_DIR; received {value!r}"
        )
    model_id, raw_path = value.split("=", 1)
    model_id = model_id.strip()
    if not model_id or not raw_path.strip():
        raise ComparisonInputError(f"Invalid --model value: {value!r}")
    return model_id, Path(raw_path.strip())


def discover_exports(
    evaluation_root: Path,
    explicit: Sequence[str],
    models: Sequence[dict[str, Any]],
) -> list[Export]:
    if explicit:
        requested = [_parse_explicit_model(value) for value in explicit]
        exports = [_load_export(path, models, model_id) for model_id, path in requested]
    else:
        root = evaluation_root.expanduser().resolve()
        if not root.is_dir():
            raise ComparisonInputError(f"Evaluation root does not exist: {root}")
        directories = sorted(
            path for path in root.iterdir() if (path / "export_metadata.json").is_file()
        )
        if not directories:
            raise ComparisonInputError(f"No evaluation exports under {root}")
        exports = [_load_export(path, models) for path in directories]
    identifiers = [export.model_id for export in exports]
    if len(identifiers) != len(set(identifiers)):
        raise ComparisonInputError(f"Duplicate model exports: {identifiers}")
    order = {str(model["id"]): index for index, model in enumerate(models)}
    return sorted(exports, key=lambda item: order.get(item.model_id, len(order)))


def _frame_key_tuples(frame: pd.DataFrame, keys: Sequence[str]) -> list[tuple[Any, ...]]:
    return list(frame.loc[:, list(keys)].itertuples(index=False, name=None))


def _require_same_array(
    reference: np.ndarray, candidate: np.ndarray, description: str
) -> None:
    if reference.shape != candidate.shape or not np.allclose(
        reference, candidate, rtol=1.0e-10, atol=1.0e-10, equal_nan=True
    ):
        raise ComparisonInputError(f"Cross-model mismatch in {description}")


def _read_snapshot(export: Export, name: str) -> dict[str, np.ndarray]:
    path = export.directory / "snapshots" / name
    with np.load(path, allow_pickle=False) as archive:
        required = {"truth", "peak_truth", "peak_prediction"}
        missing = sorted(required.difference(archive.files))
        if missing:
            raise ComparisonInputError(f"{path} lacks arrays: {missing}")
        return {key: np.asarray(archive[key], dtype=float) for key in required}


def validate_compatibility(exports: Sequence[Export]) -> dict[str, Any]:
    if not exports:
        raise ComparisonInputError("No exports were loaded")
    reference = exports[0]
    domain_keys = _frame_key_tuples(reference.domain_population, DOMAIN_KEYS)
    gauge_keys = _frame_key_tuples(reference.gauge_population, GAUGE_KEYS)
    time_keys = _frame_key_tuples(reference.timeseries, TIMESERIES_KEYS)
    truth = reference.timeseries["rdycore_depth_m"].to_numpy(dtype=float)
    mapping_values = reference.mapping[
        ["station_id", "model_x", "model_y"]
    ].to_numpy()
    snapshot_truth = {
        name: _read_snapshot(reference, name)
        for name in reference.snapshots
    }
    for candidate in exports[1:]:
        if _frame_key_tuples(candidate.domain_population, DOMAIN_KEYS) != domain_keys:
            raise ComparisonInputError(
                f"Domain trajectory population differs for {candidate.label}"
            )
        if _frame_key_tuples(candidate.gauge_population, GAUGE_KEYS) != gauge_keys:
            raise ComparisonInputError(
                f"Gauge trajectory population differs for {candidate.label}"
            )
        if _frame_key_tuples(candidate.timeseries, TIMESERIES_KEYS) != time_keys:
            raise ComparisonInputError(f"Hydrograph keys differ for {candidate.label}")
        _require_same_array(
            truth,
            candidate.timeseries["rdycore_depth_m"].to_numpy(dtype=float),
            f"RDycore hydrograph truth for {candidate.label}",
        )
        reference_diagnostics = reference.domain_population.loc[
            :, [*DOMAIN_KEYS, "forecast_hours_completed", "persistence_depth_rmse_m"]
        ].copy()
        candidate_diagnostics = candidate.domain_population.loc[
            :, [*DOMAIN_KEYS, "forecast_hours_completed", "persistence_depth_rmse_m"]
        ].copy()
        for column in ("forecast_hours_completed", "persistence_depth_rmse_m"):
            _require_same_array(
                pd.to_numeric(reference_diagnostics[column], errors="coerce").to_numpy(),
                pd.to_numeric(candidate_diagnostics[column], errors="coerce").to_numpy(),
                f"{column} for {candidate.label}",
            )
        candidate_mapping = candidate.mapping[
            ["station_id", "model_x", "model_y"]
        ].to_numpy()
        if mapping_values.shape != candidate_mapping.shape or not np.array_equal(
            mapping_values[:, 0], candidate_mapping[:, 0]
        ):
            raise ComparisonInputError(f"Gauge mapping differs for {candidate.label}")
        _require_same_array(
            mapping_values[:, 1:].astype(float),
            candidate_mapping[:, 1:].astype(float),
            f"gauge coordinates for {candidate.label}",
        )
        for key in ("x", "y"):
            _require_same_array(
                reference.spatial[key], candidate.spatial[key],
                f"{key} grid for {candidate.label}",
            )
        if not np.array_equal(reference.spatial["domain"], candidate.spatial["domain"]):
            raise ComparisonInputError(f"Cross-model mismatch in domain mask for {candidate.label}")
        if candidate.snapshots != reference.snapshots:
            raise ComparisonInputError(f"Peak snapshots differ for {candidate.label}")
        for name in reference.snapshots:
            candidate_snapshot = _read_snapshot(candidate, name)
            for key in ("truth", "peak_truth"):
                _require_same_array(
                    snapshot_truth[name][key],
                    candidate_snapshot[key],
                    f"{key} {name} for {candidate.label}",
                )
    return {
        "reference_model": reference.model_id,
        "domain_trajectories": len(domain_keys),
        "test_events": int(reference.domain_population["event_id"].nunique()),
        "members_per_event": 32,
        "core_gauges": len(reference.mapping),
        "gauge_trajectory_records": len(gauge_keys),
        "member_001_hydrograph_samples": len(time_keys),
        "snapshot_member": 1,
        "peak_snapshots": list(reference.snapshots),
        "population_and_truth_match": True,
    }


def availability_table(
    models: Sequence[dict[str, Any]], exports: Sequence[Export]
) -> pd.DataFrame:
    available = {export.model_id: export for export in exports}
    rows = []
    for model in models:
        model_id = str(model["id"])
        export = available.get(model_id)
        reportable = bool(model.get("reportable_ar32"))
        if export is not None:
            status = "physical_evaluation_available"
        elif reportable:
            status = "missing_physical_evaluation"
        else:
            status = "unavailable_no_ar32_checkpoint"
        rows.append(
            {
                "model_id": model_id,
                "model": model["publication_name"],
                "reportable_ar32": reportable,
                "physical_evaluation_status": status,
                "evaluation_directory": str(export.directory) if export else "",
            }
        )
    return pd.DataFrame(rows)


def combine_summary(exports: Sequence[Export], scope: str) -> pd.DataFrame:
    pieces = []
    for export in exports:
        source = export.domain_summary if scope == "domain" else export.gauge_summary
        frame = source.copy()
        frame.insert(0, "model", export.label)
        frame.insert(0, "model_id", export.model_id)
        frame.insert(0, "scope", scope)
        pieces.append(frame)
    return pd.concat(pieces, ignore_index=True)


def plot_metric_panels(
    summary: pd.DataFrame,
    exports: Sequence[Export],
    metric_keys: Sequence[str],
    title: str,
    output: Path,
    dpi: int,
) -> None:
    missing = [
        f"{export.model_id}:{metric}"
        for export in exports
        for metric in metric_keys
        if not (
            summary["model_id"].eq(export.model_id)
            & summary["metric"].eq(metric)
        ).any()
    ]
    if missing:
        raise ComparisonInputError(f"Missing summary metrics: {missing}")
    ncols = 4
    nrows = math.ceil(len(metric_keys) / ncols)
    figure, axes = plt.subplots(
        nrows, ncols, figsize=(17.0, 4.2 * nrows), squeeze=False,
        constrained_layout=True,
    )
    x = np.arange(len(exports))
    for panel_index, (axis, metric) in enumerate(zip(axes.flat, metric_keys)):
        panel = summary.loc[summary["metric"].eq(metric)].set_index("model_id")
        panel = panel.reindex([export.model_id for export in exports])
        median = panel["median"].to_numpy(dtype=float)
        q25 = panel["q25"].to_numpy(dtype=float)
        q75 = panel["q75"].to_numpy(dtype=float)
        axis.bar(
            x, median, color=[COLORS[index % len(COLORS)] for index in range(len(exports))],
            edgecolor="black", linewidth=0.7,
            yerr=np.vstack((median - q25, q75 - median)), capsize=3,
        )
        label = str(panel.iloc[0]["metric_label"])
        axis.set_ylabel(label)
        axis.set_xticks(x, [export.label for export in exports], rotation=28, ha="right")
        axis.grid(axis="y", alpha=0.25)
        if "bias" in metric or metric in {"depth_nse", "depth_kge", "mse_skill_vs_persistence"}:
            axis.axhline(0.0, color="0.45", linewidth=0.8)
        axis.text(
            0.015, 0.985, f"({chr(ord('a') + panel_index)})",
            transform=axis.transAxes, ha="left", va="top", fontweight="bold",
        )
    for axis in axes.flat[len(metric_keys):]:
        axis.set_visible(False)
    figure.suptitle(title + "\nmedian of event medians; whiskers show event IQR", fontsize=15)
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def _station_title(row: pd.Series, timeseries: pd.DataFrame) -> str:
    return single._station_label(row, timeseries)


def plot_hydrographs(
    exports: Sequence[Export], output_dir: Path, dpi: int
) -> list[Path]:
    directory = output_dir / "hydrographs"
    directory.mkdir(parents=True, exist_ok=True)
    reference = exports[0]
    mapping = reference.mapping
    events = sorted(reference.timeseries["event_id"].unique())
    outputs = []
    ncols = 4
    nrows = math.ceil(len(mapping) / ncols)
    for event_id in events:
        figure, axes = plt.subplots(
            nrows, ncols, figsize=(3.8 * ncols, 2.65 * nrows), squeeze=False
        )
        reference_event = reference.timeseries.loc[
            reference.timeseries["event_id"].eq(event_id)
        ]
        for axis, (_, station) in zip(axes.flat, mapping.iterrows()):
            station_id = str(station["station_id"])
            truth = reference_event.loc[
                reference_event["station_id"].eq(station_id)
            ].sort_values("lead_hour")
            axis.plot(
                truth["lead_hour"], truth["rdycore_depth_m"], color="black",
                linewidth=2.0, label="RDycore",
            )
            for index, export in enumerate(exports):
                selected = export.timeseries.loc[
                    export.timeseries["event_id"].eq(event_id)
                    & export.timeseries["station_id"].eq(station_id)
                ].sort_values("lead_hour")
                axis.plot(
                    selected["lead_hour"], selected["predicted_depth_m"],
                    color=COLORS[index % len(COLORS)], linewidth=1.35,
                    label=export.label,
                )
            axis.set_title(_station_title(station, reference.timeseries), fontsize=8.5)
            axis.set_ylim(bottom=0.0)
            axis.grid(alpha=0.25)
        for axis in axes.flat[len(mapping):]:
            axis.set_axis_off()
        for row in range(nrows):
            axes[row, 0].set_ylabel("Water depth (m)")
        for axis in axes.flat[-ncols:]:
            if axis.get_visible():
                axis.set_xlabel("Forecast lead (h)")
        handles, labels = axes.flat[0].get_legend_handles_labels()
        figure.legend(
            handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.95),
            ncol=min(4, len(labels)), frameon=False,
        )
        figure.suptitle(
            f"Held-out hydrographs: {event_id}, ensemble member 001",
            fontsize=14, y=0.995,
        )
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))
        path = directory / f"{_safe_filename(event_id)}.hydrographs.png"
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
        outputs.append(path)
    return outputs


def _finite_limit(arrays: Iterable[np.ndarray], percentile: float, floor: float) -> float:
    parts = []
    for array in arrays:
        values = np.asarray(array, dtype=float)
        values = values[np.isfinite(values)]
        if values.size:
            parts.append(values)
    if not parts:
        raise ComparisonInputError("Cannot calculate map color limits from NaN arrays")
    return max(floor, float(np.nanpercentile(np.concatenate(parts), percentile)))


def _map(
    axis: plt.Axes,
    spatial: dict[str, np.ndarray],
    field: np.ndarray,
    title: str,
    cmap: str,
    vmin: float,
    vmax: float,
    mapping: pd.DataFrame,
):
    artist = axis.pcolormesh(
        spatial["x"] / 1000.0, spatial["y"] / 1000.0,
        np.ma.masked_invalid(field), shading="auto", cmap=cmap,
        vmin=vmin, vmax=vmax, rasterized=True,
    )
    axis.scatter(
        mapping["model_x"] / 1000.0, mapping["model_y"] / 1000.0,
        s=11, c="white", edgecolors="black", linewidths=0.4, zorder=3,
    )
    axis.set_title(title, fontsize=10)
    axis.set_aspect("equal")
    axis.set_xlabel("Easting (km)")
    axis.set_ylabel("Northing (km)")
    return artist


def plot_peak_comparisons(
    exports: Sequence[Export], output_dir: Path, dpi: int
) -> list[Path]:
    directory = output_dir / "peak_depth_snapshots"
    directory.mkdir(parents=True, exist_ok=True)
    reference = exports[0]
    spatial = reference.spatial
    domain = spatial["domain"].astype(bool)
    outputs = []
    for name in reference.snapshots:
        bundles = [_read_snapshot(export, name) for export in exports]
        truth = np.where(domain, bundles[0]["peak_truth"], np.nan)
        predictions = [np.where(domain, bundle["peak_prediction"], np.nan) for bundle in bundles]
        errors = [prediction - truth for prediction in predictions]
        depth_max = _finite_limit([truth, *predictions], 99.5, 0.1)
        error_limit = _finite_limit([np.abs(error) for error in errors], 99.0, 0.02)
        columns = len(exports) + 1
        figure, axes = plt.subplots(
            2, columns, figsize=(4.0 * columns, 8.0), squeeze=False,
            constrained_layout=True,
        )
        depth_artist = _map(
            axes[0, 0], spatial, truth, "RDycore peak depth", "Blues", 0.0,
            depth_max, reference.mapping,
        )
        axes[1, 0].set_axis_off()
        error_artist = None
        for column, (export, prediction, error) in enumerate(
            zip(exports, predictions, errors), start=1
        ):
            _map(
                axes[0, column], spatial, prediction,
                f"{export.label} peak depth", "Blues", 0.0, depth_max,
                reference.mapping,
            )
            error_artist = _map(
                axes[1, column], spatial, error,
                f"{export.label} minus RDycore", "RdBu_r", -error_limit,
                error_limit, reference.mapping,
            )
        figure.colorbar(
            depth_artist, ax=list(axes[0, :]), location="bottom", shrink=0.75,
            pad=0.08, label="Event-maximum water depth (m)",
        )
        if error_artist is not None:
            figure.colorbar(
                error_artist, ax=list(axes[1, 1:]), location="bottom", shrink=0.75,
                pad=0.08, label="Peak-depth error (m)",
            )
        scenario = name[: -len(".peak.npz")]
        figure.suptitle(
            f"Peak water-depth comparison: {scenario}", fontsize=15
        )
        path = directory / f"{_safe_filename(scenario)}.peak_comparison.png"
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
        outputs.append(path)
    return outputs


def plot_spatial_comparison(
    exports: Sequence[Export], output: Path, dpi: int
) -> Path:
    reference = exports[0]
    domain = reference.spatial["domain"].astype(bool)
    rmse = [np.where(domain, export.spatial["rmse_m"], np.nan) for export in exports]
    mae = [np.where(domain, export.spatial["mae_m"], np.nan) for export in exports]
    bias = [np.where(domain, export.spatial["bias_m"], np.nan) for export in exports]
    error_max = _finite_limit([*rmse, *mae], 99.0, 0.01)
    bias_limit = _finite_limit([np.abs(value) for value in bias], 99.0, 0.01)
    figure, axes = plt.subplots(
        len(exports), 3, figsize=(13.2, 4.1 * len(exports)), squeeze=False,
        constrained_layout=True,
    )
    error_artist = None
    bias_artist = None
    for row, export in enumerate(exports):
        error_artist = _map(
            axes[row, 0], reference.spatial, rmse[row],
            f"{export.label}: depth RMSE", "magma", 0.0, error_max,
            reference.mapping,
        )
        _map(
            axes[row, 1], reference.spatial, mae[row],
            f"{export.label}: depth MAE", "magma", 0.0, error_max,
            reference.mapping,
        )
        bias_artist = _map(
            axes[row, 2], reference.spatial, bias[row],
            f"{export.label}: depth bias", "RdBu_r", -bias_limit, bias_limit,
            reference.mapping,
        )
    figure.colorbar(
        error_artist, ax=list(axes[:, :2].flat), location="bottom", shrink=0.75,
        pad=0.08, label="Water-depth error (m)",
    )
    figure.colorbar(
        bias_artist, ax=list(axes[:, 2]), location="bottom", shrink=0.75,
        pad=0.08, label="Prediction minus RDycore (m)",
    )
    figure.suptitle("Full-domain spatial error comparison", fontsize=15)
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return output


def generate(
    exports: Sequence[Export],
    models: Sequence[dict[str, Any]],
    output_dir: Path,
    dpi: int,
) -> list[Path]:
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    compatibility = validate_compatibility(exports)
    availability = availability_table(models, exports)
    missing_reportable = availability.loc[
        availability["physical_evaluation_status"].eq(
            "missing_physical_evaluation"
        ), "model_id",
    ].tolist()
    domain = combine_summary(exports, "domain")
    gauge = combine_summary(exports, "gauge")
    availability_path = output_dir / "model_evaluation_availability.csv"
    domain_path = output_dir / "full_domain_metrics_comparison.csv"
    gauge_path = output_dir / "usgs_location_metrics_comparison.csv"
    availability.to_csv(availability_path, index=False)
    domain.to_csv(domain_path, index=False)
    gauge.to_csv(gauge_path, index=False)
    outputs = [availability_path, domain_path, gauge_path]

    domain_figure = output_dir / "full_domain_metrics_comparison.png"
    gauge_figure = output_dir / "usgs_location_metrics_comparison.png"
    plot_metric_panels(
        domain, exports, DOMAIN_PLOT_METRICS,
        "Full-domain held-out rollout metrics", domain_figure, dpi,
    )
    plot_metric_panels(
        gauge, exports, GAUGE_PLOT_METRICS,
        "Held-out metrics at USGS gauge locations",
        gauge_figure, dpi,
    )
    outputs.extend((domain_figure, gauge_figure))
    outputs.extend(plot_hydrographs(exports, output_dir, dpi))
    outputs.extend(plot_peak_comparisons(exports, output_dir, dpi))
    outputs.append(
        plot_spatial_comparison(
            exports, output_dir / "full_domain_spatial_error_comparison.png", dpi
        )
    )
    manifest = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "comparison_is_complete": not missing_reportable,
        "missing_reportable_models": missing_reportable,
        "aggregation": (
            "median within each held-out event, followed by median and "
            "interquartile range across six event medians"
        ),
        "gauge_metric_semantics": (
            "Emulator predictions versus RDycore depth at mapped USGS gauge "
            "locations; not comparisons with observed USGS records"
        ),
        "test_events": compatibility["test_events"],
        "members_per_event": compatibility["members_per_event"],
        "domain_trajectories": compatibility["domain_trajectories"],
        "core_gauges": compatibility["core_gauges"],
        "snapshot_member": compatibility["snapshot_member"],
        "compatibility": compatibility,
        "exports": [
            {
                "model_id": export.model_id,
                "model": export.label,
                "directory": str(export.directory),
                "protocol_id": export.metadata.get("protocol_id"),
                "checkpoint": export.metadata.get("checkpoint"),
                "checkpoint_sha256": export.metadata.get("checkpoint_sha256"),
                "export_metadata_sha256": _sha256(
                    export.directory / "export_metadata.json"
                ),
            }
            for export in exports
        ],
        "products": [str(path.relative_to(output_dir)) for path in outputs],
    }
    manifest_path = output_dir / "comparison_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    outputs.append(manifest_path)
    return outputs


def generate_atomically(
    exports: Sequence[Export],
    models: Sequence[dict[str, Any]],
    output_dir: Path,
    dpi: int,
    overwrite: bool,
) -> list[Path]:
    destination = output_dir.expanduser().resolve()
    if destination in {Path("/"), ROOT.resolve(), ANALYSIS_DIR.resolve()}:
        raise ComparisonInputError(f"Refusing unsafe output directory: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not destination.is_dir() or destination.is_symlink():
            raise ComparisonInputError(f"Output is not a regular directory: {destination}")
        if any(destination.iterdir()) and not overwrite:
            raise ComparisonInputError(
                f"Output directory is not empty: {destination}; pass --overwrite "
                "to replace generated products"
            )
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.render-", dir=str(destination.parent)
        )
    )
    backup: Optional[Path] = None
    try:
        temporary_outputs = generate(exports, models, temporary, dpi)
        relative_outputs = [path.relative_to(temporary) for path in temporary_outputs]
        if destination.exists():
            if any(destination.iterdir()):
                backup = destination.with_name(
                    f".{destination.name}.previous-{os.getpid()}"
                )
                if backup.exists():
                    raise ComparisonInputError(f"Temporary backup already exists: {backup}")
                os.replace(destination, backup)
            else:
                destination.rmdir()
        try:
            os.replace(temporary, destination)
        except Exception:
            if backup is not None and backup.exists() and not destination.exists():
                os.replace(backup, destination)
            raise
        if backup is not None:
            shutil.rmtree(backup)
        return [destination / relative for relative in relative_outputs]
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        models = _load_publication_models()
        exports = discover_exports(args.evaluation_root, args.model, models)
        availability = availability_table(models, exports)
        missing = availability.loc[
            availability["physical_evaluation_status"].eq(
                "missing_physical_evaluation"
            ), "model",
        ].tolist()
        if missing and not args.allow_incomplete:
            raise ComparisonInputError(
                "Missing compatible physical evaluations for: "
                + ", ".join(missing)
                + ". Run the shared evaluator first or pass --allow-incomplete "
                "to render only audited exports."
            )
        outputs = generate_atomically(
            exports, models, args.output_dir, args.dpi, args.overwrite
        )
    except ComparisonInputError as exc:
        raise SystemExit(f"Comparison input validation failed: {exc}") from exc
    print(
        f"Created {len(outputs)} comparison product(s) from "
        f"{len(exports)} compatible model export(s) in "
        f"{args.output_dir.expanduser().resolve()}"
    )
    if missing:
        print("Missing physical evaluations: " + ", ".join(missing))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
