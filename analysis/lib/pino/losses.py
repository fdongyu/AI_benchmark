"""Mask-aware supervised losses and live multi-objective balancing."""

from __future__ import annotations

from typing import Dict, Mapping

import torch
from torch import nn
import torch.nn.functional as F


def masked_mean(
    values: torch.Tensor, mask: torch.Tensor, eps: float = 1.0e-12
) -> torch.Tensor:
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(1)
    if mask.shape[1] == 1 and values.shape[1] != 1:
        mask = mask.expand(-1, values.shape[1], -1, -1)
    weights = mask.to(dtype=values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(eps)


def supervised_state_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    domain_mask: torch.Tensor,
    depth_weight: float = 2.0,
    momentum_weight: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """Return separately inspectable depth and momentum MSE terms."""

    if predicted.shape != target.shape or predicted.shape[1] != 3:
        raise ValueError("predicted and target must both have shape [B,3,H,W]")
    squared = (predicted - target).square()
    depth = masked_mean(squared[:, :1], domain_mask)
    momentum = masked_mean(squared[:, 1:], domain_mask)
    return {
        "depth": depth_weight * depth,
        "momentum_data": momentum_weight * momentum,
    }


def soft_inundation_loss(
    predicted_depth: torch.Tensor,
    target_depth: torch.Tensor,
    domain_mask: torch.Tensor,
    threshold_m: float = 0.05,
    transition_m: float = 0.02,
) -> torch.Tensor:
    """Differentiable wet/dry classification loss around a physical threshold."""

    target_probability = torch.sigmoid((target_depth - threshold_m) / transition_m)
    logits = (predicted_depth - threshold_m) / transition_m
    pointwise = F.binary_cross_entropy_with_logits(
        logits, target_probability, reduction="none"
    )
    return masked_mean(pointwise, domain_mask)


def ocean_stage_loss(
    predicted_state: torch.Tensor,
    bed_elevation: torch.Tensor,
    ocean_height_m: torch.Tensor,
    ocean_boundary_mask: torch.Tensor,
) -> torch.Tensor:
    """Penalize free-surface mismatch on cells marked as open ocean boundary.

    The conservative state stores water depth, whereas the forcing stores a
    water-surface elevation.  Comparing ``h + z_b`` avoids confusing those two
    quantities.  Batches/tiles with no ocean-boundary cells contribute zero.
    """

    if predicted_state.ndim != 4 or predicted_state.shape[1] != 3:
        raise ValueError("predicted_state must have shape [B,3,H,W]")
    batch = predicted_state.shape[0]
    stage = predicted_state[:, :1] + bed_elevation
    target = ocean_height_m.reshape(batch, 1, 1, 1)
    return masked_mean((stage - target).square(), ocean_boundary_mask)


class AdaptiveLossBalancer(nn.Module):
    """Balance named losses using registered, in-place-updated EMA buffers."""

    def __init__(
        self,
        names: list[str],
        decay: float = 0.95,
        minimum: float = 1.0e-4,
        maximum: float = 10.0,
        eps: float = 1.0e-8,
    ) -> None:
        super().__init__()
        if not names or len(set(names)) != len(names):
            raise ValueError("loss names must be unique and non-empty")
        self.names = tuple(names)
        self.decay = float(decay)
        self.minimum = float(minimum)
        self.maximum = float(maximum)
        self.eps = float(eps)
        self.register_buffer("ema", torch.ones(len(names)))
        self.register_buffer("updates", torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def update(self, losses: Mapping[str, torch.Tensor]) -> None:
        values = torch.stack(
            [losses[name].detach().abs().float() for name in self.names]
        )
        if self.updates.item() == 0:
            self.ema.copy_(values.clamp_min(self.eps))
        else:
            self.ema.mul_(self.decay).add_(values, alpha=1.0 - self.decay)
        self.updates.add_(1)

    def current_weights(self) -> Dict[str, torch.Tensor]:
        inverse = self.ema.clamp_min(self.eps).reciprocal()
        normalized = inverse / inverse.mean().clamp_min(self.eps)
        # Clamp after normalization.  Clamping the unnormalized inverse first
        # makes a very large loss retain far too much weight, and renormalizing
        # after this clamp can violate the advertised bounds.
        bounded = normalized.clamp(self.minimum, self.maximum)
        return {name: bounded[index] for index, name in enumerate(self.names)}

    def forward(
        self, losses: Mapping[str, torch.Tensor], update: bool = True
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        missing = set(self.names).difference(losses)
        if missing:
            raise KeyError(f"missing loss terms: {sorted(missing)}")
        if update:
            self.update(losses)
        weights = self.current_weights()
        total = sum(weights[name] * losses[name] for name in self.names)
        return total, weights
