"""Small analytical acceptance benchmarks for the 2-D SWE implementation."""

from __future__ import annotations

import argparse
import json
from typing import Any, Optional, Sequence

import torch

from .physics import global_mass_balance, project_wet_dry, swe_residual


def _state(depth: torch.Tensor) -> torch.Tensor:
    zeros = torch.zeros_like(depth)
    return torch.stack((depth, zeros, zeros), dim=1)


def run_benchmarks(device: str | torch.device = "cpu") -> dict[str, Any]:
    target = torch.device(device)
    # Match the FP32 physics path used in training; the unit tests separately
    # exercise tighter float64 identities.
    dtype = torch.float32
    results: dict[str, Any] = {}

    # A spatially uniform rain pulse has an exact one-step storage solution.
    height, width = 9, 13
    rain = torch.full((1, 1, height, width), 18.0, dtype=dtype, device=target)
    initial = _state(torch.full((1, height, width), 0.2, dtype=dtype, device=target))
    final = initial.clone()
    final[:, 0] += 0.018
    rain_residual = swe_residual(
        initial, final, rain, 0.0, 0.03, None, 3600.0, 100.0, 120.0
    )
    balance = global_mass_balance(initial, final, rain, None, 3600.0, 100.0, 120.0)
    results["rainfall_storage"] = {
        "max_local_residual": float(rain_residual.abs().max().cpu()),
        "max_global_closure_m3": float(balance["closure_error_m3"].abs().max().cpu()),
        "max_relative_closure": float(
            balance["relative_closure_error"].abs().max().cpu()
        ),
    }

    # A constant free surface over variable topography must remain stationary.
    y = torch.linspace(-1.0, 1.0, height, dtype=dtype, device=target)[:, None]
    x = torch.linspace(-1.0, 1.0, width, dtype=dtype, device=target)[None, :]
    bed = 0.25 * x + 0.15 * y + 0.03 * torch.sin(4.0 * x)
    lake = _state((2.0 - bed).unsqueeze(0)).requires_grad_()
    lake_residual = swe_residual(lake, lake, 0.0, bed, 0.035, None, 60.0, 50.0, 50.0)
    lake_residual.square().sum().backward()
    results["lake_at_rest"] = {
        "max_residual": float(lake_residual.abs().max().detach().cpu()),
        "finite_gradient": bool(torch.isfinite(lake.grad).all().cpu()),
    }

    # Projection is the final guard against negative water and dry momentum.
    raw = torch.tensor(
        [[[[-0.1, 1.0e-8, 0.4]], [[2.0, 3.0, 1.0]], [[-2.0, -3.0, -1.0]]]],
        dtype=dtype,
        device=target,
    )
    projected = project_wet_dry(raw, dry_tolerance=1.0e-6)
    dry = projected[:, :1] <= 1.0e-6
    dry_momentum = projected[:, 1:].abs() * dry.expand(-1, 2, -1, -1)
    results["wet_dry_projection"] = {
        "minimum_depth": float(projected[:, 0].min().cpu()),
        "max_dry_momentum": float(dry_momentum.max().cpu()),
    }

    passed = (
        results["rainfall_storage"]["max_local_residual"] < 1.0e-8
        and results["rainfall_storage"]["max_relative_closure"] < 1.0e-5
        and results["lake_at_rest"]["max_residual"] < 1.0e-5
        and results["lake_at_rest"]["finite_gradient"]
        and results["wet_dry_projection"]["minimum_depth"] >= 0.0
        and results["wet_dry_projection"]["max_dry_momentum"] == 0.0
    )
    results["passed"] = bool(passed)
    return results


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cpu", help="cpu, cuda, or a concrete torch device"
    )
    args = parser.parse_args(argv)
    results = run_benchmarks(args.device)
    print(json.dumps(results, indent=2, sort_keys=True))
    if not results["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
