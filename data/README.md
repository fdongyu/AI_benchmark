# Data inventory

- `metadata/m032/shared/`: frozen event/trajectory split and train-only
  normalization used by the common13 experiments.
- `metadata/m032/channels/`: derived channel, stream, and boundary masks.
- `metadata/validation/`: static grid and 14-station USGS mapping used by the
  held-out evaluator.
- `evaluations/pino_resunet_large_m032_ar32/`: retained
  PINO-ResU-Net-large metrics, hydrographs, snapshots, and spatial products for
  192 free rollouts.

The full 138-GiB HDF5 trajectory store is external and is not replaced by a
synthetic or downsampled copy. Its canonical NERSC location, environment
override, directory layout, 864/192/192 split, and content hashes are defined
in [`../TRAINING_DATA.md`](../TRAINING_DATA.md).

`trajectory_split.csv` records the original absolute NERSC paths. The
publication trainer relocates the `trajectories/...` suffix beneath
`HOUSTON_DATASET_ROOT`, so the frozen table need not be edited. Do not change
event/member split assignments. Absolute `/compyfs` paths in preparation
metadata are historical provenance, not current runtime locations.
