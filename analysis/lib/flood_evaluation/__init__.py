"""Post-processing tools for Houston flood-emulator rollout evaluation.

This package intentionally lives outside :mod:`pino`: adding or revising
analysis code must not invalidate the source hash used for exact training
restarts.
"""

from .metrics import RunningFieldMetrics, hydrograph_metrics, summarize_events

__all__ = ["RunningFieldMetrics", "hydrograph_metrics", "summarize_events"]
