"""Complete, atomic checkpoints for training and event adaptation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional
import os

import torch


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[Any],
    epoch: int,
    config: Mapping[str, Any],
    normalization: Mapping[str, Any],
    metrics: Mapping[str, Any],
    provenance: Mapping[str, Any],
    loss_balancer: Optional[torch.nn.Module] = None,
    scaler: Optional[Any] = None,
    training_state: Optional[Mapping[str, Any]] = None,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    payload: Dict[str, Any] = {
        "format_version": 2,
        "epoch": int(epoch),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "config": dict(config),
        "normalization": dict(normalization),
        "metrics": dict(metrics),
        "provenance": dict(provenance),
        "loss_balancer": (
            loss_balancer.state_dict() if loss_balancer is not None else None
        ),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "training_state": dict(training_state or {}),
    }
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def load_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    loss_balancer: Optional[torch.nn.Module] = None,
    scaler: Optional[Any] = None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> Dict[str, Any]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(payload["model"], strict=strict)
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler") is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if loss_balancer is not None and payload.get("loss_balancer") is not None:
        loss_balancer.load_state_dict(payload["loss_balancer"])
    if scaler is not None and payload.get("scaler") is not None:
        scaler.load_state_dict(payload["scaler"])
    return payload
