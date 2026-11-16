# Locked protocol

- Inputs: the ordered common13 feature tensor.
- Target: normalized conservative-state increment with the common external
  residual update and wet/dry projector.
- Curriculum: AR4 → AR8 → AR16 → AR32.
- Exposure per epoch: 8,192 training and 2,048 validation forecast steps.
- State loss: `2*MSE(h) + 0.5*MSE(qx) + 0.5*MSE(qy)`.
- Auxiliary wetness/inundation loss: none.
- Optimizer: AdamW, weight decay `1e-5`, gradient clipping `1`.
- Scheduler: `ReduceLROnPlateau(mode=min, factor=0.5, patience=2,
  threshold=1e-4, threshold_mode=rel, cooldown=0, min_lr=0, eps=1e-8)`.
- Convergence: at least 12 epochs, then stop after 8 epochs without a relative
  validation improvement of at least `0.001`; 240 epochs is a fail-closed
  ceiling, not a successful fixed budget.
- Precision: FP32; AMP and TF32 disabled; seeded cuDNN deterministic mode;
  global deterministic algorithms disabled because this CUDA interpolation
  path lacks a deterministic backward kernel.
- Stage handoff: validation-best model weights; fresh optimizer and scheduler.
- Checkpoint selection: validation supervised state loss only; physics is
  excluded from the selection metric.

The SWE term is enabled only for `pino_resunet_small` and the canonical
`pino_resunet_large` reference. This preserves the intended
data-only/PINO ablations. Plateau convergence gives models the same stopping
rule, not identical total epochs or GPU cost; reports must include completed
epochs and forecast-step exposure.
