"""Command-line interfaces for base training and leakage-safe event adaptation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from .config import ExperimentConfig, load_config
from .inference import forecast_file
from .trainer import finetune_experiment, train_experiment


def _runtime_overrides(
    config: ExperimentConfig,
    *,
    device: Optional[str],
    allow_cpu: bool,
) -> None:
    if device:
        config.training.device = device
        if device == "cpu":
            config.training.require_cuda = False
    if allow_cpu:
        config.training.require_cuda = False


def _train_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the multi-event rainfall-driven Houston PINO"
    )
    parser.add_argument("--config", required=True, help="experiment YAML")
    parser.add_argument("--device", help="override configured torch device")
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="allow a CPU fallback when CUDA was requested (debugging only)",
    )
    parser.add_argument("--resume", help="override training.resume checkpoint")
    parser.add_argument("--epochs", type=int, help="override training.epochs")
    return parser


def train_main(argv: Optional[Sequence[str]] = None) -> None:
    args = _train_parser().parse_args(argv)
    config = load_config(args.config)
    _runtime_overrides(config, device=args.device, allow_cpu=args.allow_cpu)
    if args.resume:
        config.training.resume = str(Path(args.resume).expanduser().resolve())
    if args.epochs is not None:
        if args.epochs < 1:
            raise ValueError("--epochs must be positive")
        config.training.epochs = args.epochs
    checkpoint = train_experiment(config)
    print(f"best checkpoint: {checkpoint}", flush=True)


def _finetune_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Adapt a trained Houston PINO to an observed event prefix"
    )
    parser.add_argument("--config", required=True, help="experiment YAML")
    parser.add_argument(
        "--event-id", required=True, help="one held-out event identifier"
    )
    parser.add_argument("--checkpoint", required=True, help="trained base checkpoint")
    parser.add_argument("--device", help="override configured torch device")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--split", help="override finetune.split")
    parser.add_argument("--mode", choices=("adapter", "head", "full"))
    parser.add_argument("--observation-hours", type=float)
    parser.add_argument("--epochs", type=int, help="override finetune.epochs")
    return parser


def finetune_main(argv: Optional[Sequence[str]] = None) -> None:
    args = _finetune_parser().parse_args(argv)
    config = load_config(args.config)
    _runtime_overrides(config, device=args.device, allow_cpu=args.allow_cpu)
    if args.split:
        config.finetune.split = args.split
    if args.mode:
        config.finetune.mode = args.mode
    if args.observation_hours is not None:
        if args.observation_hours < 0.0:
            raise ValueError("--observation-hours must be nonnegative")
        config.finetune.observation_hours = args.observation_hours
    if args.epochs is not None:
        if args.epochs < 1:
            raise ValueError("--epochs must be positive")
        config.finetune.epochs = args.epochs
    checkpoint = finetune_experiment(
        config,
        event_id=args.event_id,
        checkpoint=args.checkpoint,
    )
    print(f"best adapted checkpoint: {checkpoint}", flush=True)


def _forecast_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a target-free autoregressive Houston PINO forecast"
    )
    parser.add_argument("--config", required=True, help="experiment YAML")
    parser.add_argument(
        "--checkpoint", required=True, help="base or adapted checkpoint"
    )
    parser.add_argument("--input", required=True, help="forcing HDF5 input")
    parser.add_argument("--output", required=True, help="new forecast HDF5 output")
    parser.add_argument("--device", help="override configured torch device")
    parser.add_argument("--allow-cpu", action="store_true")
    return parser


def forecast_main(argv: Optional[Sequence[str]] = None) -> None:
    args = _forecast_parser().parse_args(argv)
    config = load_config(args.config)
    _runtime_overrides(config, device=args.device, allow_cpu=args.allow_cpu)
    destination = forecast_file(
        config,
        checkpoint=args.checkpoint,
        input_path=args.input,
        output_path=args.output,
    )
    print(f"forecast: {destination}", flush=True)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Houston PINO workflows")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("train", add_help=False)
    subparsers.add_parser("finetune", add_help=False)
    subparsers.add_parser("forecast", add_help=False)
    known, remainder = parser.parse_known_args(argv)
    if known.command == "train":
        train_main(remainder)
    elif known.command == "finetune":
        finetune_main(remainder)
    else:
        forecast_main(remainder)


if __name__ == "__main__":
    main()
