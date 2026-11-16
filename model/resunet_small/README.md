# ResU-Net-small (`resunet_small`)

Small depth-4 residual U-Net trained with the data-matching loss only. It uses
14 base channels, no event-adapter head, and no dropout. It is the matched
data-only counterpart of `pino_resunet_small`.

Both folders deliberately build the same network. The exact shared network
source is `src/vendor/pino/model.py`.
