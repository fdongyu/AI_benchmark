# PINO-ResU-Net-large (`pino_resunet_large`)

The same large depth-4 residual U-Net used by `resunet_large`: 24 base
channels, no dropout, and an eight-channel event-adapter head kept frozen and
neutral during shared training. This treatment adds the declared
shallow-water-equation consistency loss in the training recipe.

The retained result has a separately recorded continuation history; the
connector describes only its network architecture. The exact shared network
source is `src/vendor/pino/model.py`.

This folder also owns the model-specific reproducibility code:

- `evaluate.py` is the single public evaluation entry point.
- `_evaluation_backend.py` adapts its 13-feature residual forecast to the
  shared flood metrics and is not intended to be run directly.
- `load_historical_checkpoint.py` reconstructs the retained checkpoint from
  the immutable trainer snapshot.

Training data handling, curriculum, losses, and checkpoint writing remain in
the common `src/train_benchmark_model.py` framework. The byte-exact historical
trainer and protocol manifest remain under `provenance/training_sources/`
because the retained run predates the common public trainer and resumed an
earlier AR4 checkpoint.
