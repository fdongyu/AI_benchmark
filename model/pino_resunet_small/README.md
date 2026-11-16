# PINO-ResU-Net-small (`pino_resunet_small`)

The same small depth-4 residual U-Net used by `resunet_small`: 14 base
channels, no event-adapter head, and no dropout. This treatment adds the
declared shallow-water-equation consistency loss in the common trainer.

The physics loss is not implemented in this connector. Keeping it in the
training recipe makes the small PINO/data-only comparison explicit. The exact
shared network source is `src/vendor/pino/model.py`.
