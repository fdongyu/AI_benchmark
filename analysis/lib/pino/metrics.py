"""Evaluation metrics for conservative flood-state rollouts."""

from __future__ import annotations

from typing import Dict, Iterable, Mapping

import torch

from .losses import masked_mean


def state_rmse(
    predicted: torch.Tensor, target: torch.Tensor, domain_mask: torch.Tensor
) -> torch.Tensor:
    return torch.stack(
        [
            masked_mean(
                (
                    predicted[:, index : index + 1] - target[:, index : index + 1]
                ).square(),
                domain_mask,
            ).sqrt()
            for index in range(3)
        ]
    )


def critical_success_index(
    predicted_depth: torch.Tensor,
    target_depth: torch.Tensor,
    domain_mask: torch.Tensor,
    threshold_m: float,
    eps: float = 1.0e-12,
) -> torch.Tensor:
    active = domain_mask.bool()
    predicted_wet = (predicted_depth >= threshold_m) & active
    target_wet = (target_depth >= threshold_m) & active
    hits = (predicted_wet & target_wet).sum(dtype=torch.float64)
    union = (predicted_wet | target_wet).sum(dtype=torch.float64)
    score = torch.where(
        union > eps,
        hits / union.clamp_min(eps),
        torch.ones_like(union),
    )
    return score.to(predicted_depth.dtype)


def summarize_step(
    predicted: torch.Tensor,
    target: torch.Tensor,
    domain_mask: torch.Tensor,
    thresholds_m: Iterable[float] = (0.05, 0.30),
) -> Dict[str, float]:
    rmse = state_rmse(predicted, target, domain_mask).detach().cpu()
    result = {
        "rmse_h_m": float(rmse[0]),
        "rmse_hu_m2_s": float(rmse[1]),
        "rmse_hv_m2_s": float(rmse[2]),
    }
    for threshold in thresholds_m:
        result[f"csi_{threshold:.2f}m"] = float(
            critical_success_index(
                predicted[:, :1], target[:, :1], domain_mask, threshold
            )
            .detach()
            .cpu()
        )
    return result


def metric_statistics(
    predicted: torch.Tensor,
    target: torch.Tensor,
    domain_mask: torch.Tensor,
    thresholds_m: Iterable[float] = (0.05, 0.30),
) -> list[Dict[str, float]]:
    """Return additive per-sample statistics for unbiased epoch metrics."""

    active = domain_mask.bool()
    active_values = active.to(predicted.dtype)
    squared_error = ((predicted - target).square() * active_values).sum(dim=(-2, -1))
    counts = active_values.sum(dim=(-2, -1)).squeeze(1)
    squared_error = squared_error.detach().cpu()
    counts = counts.detach().cpu()
    threshold_counts: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for threshold in thresholds_m:
        predicted_wet = (predicted[:, :1] >= threshold) & active
        target_wet = (target[:, :1] >= threshold) & active
        hits = (predicted_wet & target_wet).sum(dim=(-3, -2, -1))
        union = (predicted_wet | target_wet).sum(dim=(-3, -2, -1))
        threshold_counts[f"{threshold:.2f}m"] = (
            hits.detach().cpu(),
            union.detach().cpu(),
        )
    records: list[Dict[str, float]] = []
    for sample in range(predicted.shape[0]):
        record = {
            "count": float(counts[sample]),
            "sse_h": float(squared_error[sample, 0]),
            "sse_hu": float(squared_error[sample, 1]),
            "sse_hv": float(squared_error[sample, 2]),
        }
        for suffix, (hits, union) in threshold_counts.items():
            record[f"hits_{suffix}"] = float(hits[sample])
            record[f"union_{suffix}"] = float(union[sample])
        records.append(record)
    return records


def merge_metric_statistics(
    destination: Dict[str, float], source: Mapping[str, float]
) -> None:
    """Add one sufficient-statistics record into an accumulator."""

    for key, value in source.items():
        destination[key] = destination.get(key, 0.0) + float(value)


def finalize_metric_statistics(statistics: Mapping[str, float]) -> Dict[str, float]:
    """Convert pooled sufficient statistics to RMSE and CSI values."""

    count = float(statistics.get("count", 0.0))
    if count <= 0.0:
        raise ValueError("cannot compute metrics without active evaluation cells")
    result = {
        "rmse_h_m": (float(statistics.get("sse_h", 0.0)) / count) ** 0.5,
        "rmse_hu_m2_s": (float(statistics.get("sse_hu", 0.0)) / count) ** 0.5,
        "rmse_hv_m2_s": (float(statistics.get("sse_hv", 0.0)) / count) ** 0.5,
    }
    for key in sorted(statistics):
        if not key.startswith("hits_"):
            continue
        suffix = key[len("hits_") :]
        union = float(statistics.get(f"union_{suffix}", 0.0))
        result[f"csi_{suffix}"] = float(statistics[key]) / union if union > 0.0 else 1.0
    return result
