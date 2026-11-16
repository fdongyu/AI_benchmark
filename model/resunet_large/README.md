# ResU-Net-large (`resunet_large`)

Large depth-4 residual U-Net trained with the data-matching loss only. It uses
24 base channels, no dropout, and an eight-channel event-adapter head that is
kept frozen and neutral during shared training. It is the matched data-only
counterpart of `pino_resunet_large`.

The retained data-only run failed during AR4; its failure record is part of the
benchmark. The exact shared network source is `src/vendor/pino/model.py`.
