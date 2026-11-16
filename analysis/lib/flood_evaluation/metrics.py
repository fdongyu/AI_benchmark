"""Streaming flood metrics compatible with the existing emulator workflow."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd


def _threshold_label(value: float) -> str:
    return f"{float(value):g}".replace(".", "p")


def _finite_correlation(
    count: int,
    truth_sum: float,
    prediction_sum: float,
    truth_square_sum: float,
    prediction_square_sum: float,
    cross_sum: float,
) -> float:
    if count < 3:
        return math.nan
    n = float(count)
    truth_variance_sum = truth_square_sum - truth_sum * truth_sum / n
    prediction_variance_sum = (
        prediction_square_sum - prediction_sum * prediction_sum / n
    )
    if truth_variance_sum <= 0.0 or prediction_variance_sum <= 0.0:
        return math.nan
    covariance_sum = cross_sum - truth_sum * prediction_sum / n
    return float(
        covariance_sum / math.sqrt(truth_variance_sum * prediction_variance_sum)
    )


class RunningFieldMetrics:
    """Accumulate finite-pair metrics without retaining full raster time series."""

    def __init__(self, thresholds: Iterable[float] = ()) -> None:
        values = tuple(sorted({float(value) for value in thresholds}))
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("thresholds must be finite and nonnegative")
        self.thresholds = values
        self.count = 0
        self.truth_sum = 0.0
        self.prediction_sum = 0.0
        self.truth_square_sum = 0.0
        self.prediction_square_sum = 0.0
        self.cross_sum = 0.0
        self.squared_error_sum = 0.0
        self.absolute_error_sum = 0.0
        self.error_sum = 0.0
        self.contingency = {
            threshold: {"hits": 0, "misses": 0, "false_alarms": 0}
            for threshold in values
        }

    def update(
        self,
        truth: np.ndarray,
        prediction: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> None:
        truth_values = np.asarray(truth)
        prediction_values = np.asarray(prediction)
        if truth_values.shape != prediction_values.shape:
            raise ValueError(
                "truth and prediction shapes differ: "
                f"{truth_values.shape} != {prediction_values.shape}"
            )
        valid = np.isfinite(truth_values) & np.isfinite(prediction_values)
        if mask is not None:
            spatial_mask = np.asarray(mask, dtype=bool)
            try:
                spatial_mask = np.broadcast_to(spatial_mask, truth_values.shape)
            except ValueError as exc:
                raise ValueError("mask cannot broadcast to the field shape") from exc
            valid &= spatial_mask
        x = np.asarray(truth_values[valid], dtype=np.float64)
        y = np.asarray(prediction_values[valid], dtype=np.float64)
        if x.size == 0:
            return
        error = y - x
        self.count += int(x.size)
        self.truth_sum += float(x.sum(dtype=np.float64))
        self.prediction_sum += float(y.sum(dtype=np.float64))
        self.truth_square_sum += float(np.dot(x, x))
        self.prediction_square_sum += float(np.dot(y, y))
        self.cross_sum += float(np.dot(x, y))
        self.squared_error_sum += float(np.dot(error, error))
        self.absolute_error_sum += float(np.abs(error).sum(dtype=np.float64))
        self.error_sum += float(error.sum(dtype=np.float64))
        for threshold, counts in self.contingency.items():
            truth_wet = x >= threshold
            prediction_wet = y >= threshold
            counts["hits"] += int(np.count_nonzero(truth_wet & prediction_wet))
            counts["misses"] += int(np.count_nonzero(truth_wet & ~prediction_wet))
            counts["false_alarms"] += int(
                np.count_nonzero(~truth_wet & prediction_wet)
            )

    def merge(self, other: "RunningFieldMetrics") -> None:
        if self.thresholds != other.thresholds:
            raise ValueError("cannot merge metrics with different thresholds")
        for name in (
            "count",
            "truth_sum",
            "prediction_sum",
            "truth_square_sum",
            "prediction_square_sum",
            "cross_sum",
            "squared_error_sum",
            "absolute_error_sum",
            "error_sum",
        ):
            setattr(self, name, getattr(self, name) + getattr(other, name))
        for threshold in self.thresholds:
            for name in ("hits", "misses", "false_alarms"):
                self.contingency[threshold][name] += other.contingency[threshold][name]

    def result(self, prefix: str, unit: str) -> dict[str, float | int]:
        suffix = f"_{unit}" if unit else ""
        if self.count == 0:
            result: dict[str, float | int] = {
                f"{prefix}_rmse{suffix}": math.nan,
                f"{prefix}_mae{suffix}": math.nan,
                f"{prefix}_bias{suffix}": math.nan,
                f"{prefix}_correlation": math.nan,
                f"{prefix}_nse": math.nan,
                f"{prefix}_kge": math.nan,
                f"{prefix}_evaluated_values": 0,
            }
        else:
            n = float(self.count)
            truth_mean = self.truth_sum / n
            prediction_mean = self.prediction_sum / n
            truth_variance_sum = (
                self.truth_square_sum - self.truth_sum * self.truth_sum / n
            )
            prediction_variance_sum = (
                self.prediction_square_sum
                - self.prediction_sum * self.prediction_sum / n
            )
            correlation = _finite_correlation(
                self.count,
                self.truth_sum,
                self.prediction_sum,
                self.truth_square_sum,
                self.prediction_square_sum,
                self.cross_sum,
            )
            nse = (
                1.0 - self.squared_error_sum / truth_variance_sum
                if truth_variance_sum > 0.0
                else math.nan
            )
            if (
                math.isfinite(correlation)
                and abs(truth_mean) > 1.0e-12
                and truth_variance_sum > 0.0
            ):
                truth_std = math.sqrt(truth_variance_sum / n)
                prediction_std = math.sqrt(max(prediction_variance_sum, 0.0) / n)
                kge = 1.0 - math.sqrt(
                    (correlation - 1.0) ** 2
                    + (prediction_std / truth_std - 1.0) ** 2
                    + (prediction_mean / truth_mean - 1.0) ** 2
                )
            else:
                kge = math.nan
            result = {
                f"{prefix}_rmse{suffix}": math.sqrt(self.squared_error_sum / n),
                f"{prefix}_mae{suffix}": self.absolute_error_sum / n,
                f"{prefix}_bias{suffix}": self.error_sum / n,
                f"{prefix}_correlation": correlation,
                f"{prefix}_nse": nse,
                f"{prefix}_kge": kge,
                f"{prefix}_evaluated_values": self.count,
            }
        for threshold, counts in self.contingency.items():
            label = _threshold_label(threshold)
            hits = counts["hits"]
            misses = counts["misses"]
            false_alarms = counts["false_alarms"]
            union = hits + misses + false_alarms
            predicted_positive = hits + false_alarms
            observed_positive = hits + misses
            f1_denominator = 2 * hits + misses + false_alarms
            result[f"{prefix}_csi_at_{label}m"] = (
                hits / union if union else math.nan
            )
            result[f"{prefix}_f1_at_{label}m"] = (
                2 * hits / f1_denominator if f1_denominator else math.nan
            )
            result[f"{prefix}_precision_at_{label}m"] = (
                hits / predicted_positive if predicted_positive else math.nan
            )
            result[f"{prefix}_recall_at_{label}m"] = (
                hits / observed_positive if observed_positive else math.nan
            )
        return result


def _first_crossing(values: np.ndarray, threshold: float) -> float:
    indices = np.flatnonzero(np.asarray(values) >= threshold)
    return float(indices[0] + 1) if indices.size else math.nan


def hydrograph_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    *,
    interval_hours: float = 1.0,
    thresholds: Iterable[float] = (0.05, 0.10, 0.30),
) -> dict[str, float | int]:
    """Metrics for one gauge hydrograph, excluding its initial state."""

    observed = np.asarray(truth, dtype=np.float64).reshape(-1)
    modeled = np.asarray(prediction, dtype=np.float64).reshape(-1)
    if observed.shape != modeled.shape:
        raise ValueError("truth and prediction hydrographs must have equal lengths")
    if not math.isfinite(interval_hours) or interval_hours <= 0.0:
        raise ValueError("interval_hours must be finite and positive")
    threshold_values = tuple(float(value) for value in thresholds)
    running = RunningFieldMetrics(threshold_values)
    running.update(observed, modeled)
    result = running.result("depth", "m")
    valid = np.isfinite(observed) & np.isfinite(modeled)
    result["n_hours"] = int(np.count_nonzero(valid))
    if not np.any(valid):
        result.update(
            {
                "truth_peak_depth_m": math.nan,
                "predicted_peak_depth_m": math.nan,
                "peak_depth_error_m": math.nan,
                "absolute_peak_depth_error_m": math.nan,
                "peak_timing_error_hours": math.nan,
                "absolute_peak_timing_error_hours": math.nan,
            }
        )
    else:
        safe_truth = np.where(valid, observed, -np.inf)
        safe_prediction = np.where(valid, modeled, -np.inf)
        truth_peak_index = int(np.argmax(safe_truth))
        prediction_peak_index = int(np.argmax(safe_prediction))
        truth_peak = float(safe_truth[truth_peak_index])
        predicted_peak = float(safe_prediction[prediction_peak_index])
        peak_error = predicted_peak - truth_peak
        timing_error = (prediction_peak_index - truth_peak_index) * interval_hours
        result.update(
            {
                "truth_peak_depth_m": truth_peak,
                "predicted_peak_depth_m": predicted_peak,
                "peak_depth_error_m": peak_error,
                "absolute_peak_depth_error_m": abs(peak_error),
                "peak_timing_error_hours": timing_error,
                "absolute_peak_timing_error_hours": abs(timing_error),
            }
        )
    for threshold in threshold_values:
        label = _threshold_label(threshold)
        truth_arrival = _first_crossing(observed, threshold) * interval_hours
        prediction_arrival = _first_crossing(modeled, threshold) * interval_hours
        arrival_error = prediction_arrival - truth_arrival
        result[f"truth_arrival_hour_at_{label}m"] = truth_arrival
        result[f"predicted_arrival_hour_at_{label}m"] = prediction_arrival
        result[f"arrival_timing_error_hours_at_{label}m"] = arrival_error
        result[f"absolute_arrival_timing_error_hours_at_{label}m"] = abs(
            arrival_error
        )
    return result


def summarize_events(
    frame: pd.DataFrame,
    metrics: Mapping[str, tuple[str, bool]],
    *,
    within_event_keys: Iterable[str] = (),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the established equal-event-weighting aggregation convention.

    Member (and optionally station) values are first reduced to an event
    median. The six event medians are then summarized by median and IQR.
    """

    if "event_id" not in frame:
        raise ValueError("metric table is missing event_id")
    available = [name for name in metrics if name in frame]
    if not available:
        raise ValueError("none of the requested metrics is present")
    keys = [str(key) for key in within_event_keys]
    intermediate = frame.copy()
    if keys:
        intermediate = (
            intermediate.groupby(["event_id", *keys], as_index=False)[available]
            .median(numeric_only=True)
        )
    event = (
        intermediate.groupby("event_id", as_index=False)[available]
        .median(numeric_only=True)
        .sort_values("event_id")
        .reset_index(drop=True)
    )
    rows: list[dict[str, object]] = []
    for metric in available:
        values = pd.to_numeric(event[metric], errors="coerce").dropna()
        rows.append(
            {
                "metric": metric,
                "metric_label": metrics[metric][0],
                "higher_is_better": bool(metrics[metric][1]),
                "test_events": int(event["event_id"].nunique()),
                "finite_event_values": int(values.size),
                "median": float(values.median()) if values.size else math.nan,
                "q25": float(values.quantile(0.25)) if values.size else math.nan,
                "q75": float(values.quantile(0.75)) if values.size else math.nan,
            }
        )
    return event, pd.DataFrame(rows)
