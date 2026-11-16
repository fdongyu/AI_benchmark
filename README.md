# Houston `m032` emulator benchmark

This repository compares six flood emulators on the same Houston dataset
split, the same 13 input maps, a common staged training recipe, and one planned
evaluation procedure. The physical evaluation is not yet complete for all
models. `m032` is the experiment label for 32 ensemble members per event;
`common13` means the fixed 13-map input set.

For a plain-language explanation of the data, models, one-hour state update,
training stages, losses, current results, and how to add another emulator, see
the [benchmark guide](docs/BENCHMARK_GUIDE.md).

| Model | Why it is included | Final 32-step stage |
|---|---|---|
| U-Net | Reference convolutional design | Complete |
| FNO | Reference Fourier Neural Operator design | Complete |
| ResU-Net-small | Small data-only model for a matched comparison | Complete |
| PINO-ResU-Net-small | Same small network plus shallow-water-equation loss | Complete |
| ResU-Net-large | Large data-only model for a matched comparison | **Failed at AR4; no reportable AR32 result** |
| PINO-ResU-Net-large | Same large network plus shallow-water-equation loss | Complete; selected main result |

U-Net and FNO test architecture choice. They are not controls for the physics
loss. The effect of that loss can be isolated only by comparing a ResU-Net and
PINO-ResU-Net with the same network and model size. The small pair finished;
the large comparison remains incomplete until ResU-Net-large reaches AR32 and
both models receive the same free-running test forecasts.

Machine-readable status and output-file checksums are in
[`publication_manifest.json`](publication_manifest.json). The repository is
currently marked `release_ready: false`; the failed model is shown rather than
silently omitted or replaced.

## Repository contents

- `model/`: one readable folder per emulator, plus `model_new0` as the plug-in
  template for a future published model.
- `src/`: common training code and checksum-bound historical implementations.
- `workflows/`: checked Slurm submission and evaluation entry points.
- `data/metadata/`: fixed `m032` split, normalization, grid, masks, and gauge
  metadata.
- `data/evaluations/`: retained PINO-ResU-Net-large flood forecasts, metrics,
  and figures.
- `results/`: selected saved models and their training histories.
- `provenance/`: preserved original code and settings used by completed runs.
- `analysis/`: reusable physical-domain and USGS-gauge evaluation and plotting.
- `docs/`: benchmark guide, protocol, reproducibility, and evaluation records.
- `environment/`: recorded software environment.

All six model records are presented together under `results/m032_benchmark/`.
The later shared campaign trained five of them; PINO-ResU-Net-large had already
been trained with the same comparison settings but a separately recorded
continuation history. Its connector, evaluator, and checkpoint loader now live
together under `model/pino_resunet_large/`. Original source snapshots remain
under `provenance/` because completed records bind their hashes. The completed
workflow is read-only; a future emulator must use a new experiment and output
directory, as described in the benchmark guide.
Large trajectory HDF5 files are referenced rather than duplicated;
their exact path, layout, split counts, and hashes are in
[`TRAINING_DATA.md`](TRAINING_DATA.md).

## Shared publication protocol

The shared training recipe uses the ordered common13 inputs: `h`, `qx`, `qy`,
rainfall, ocean stage, bed elevation, domain mask, ocean-boundary mask,
normalized `x`, normalized `y`, stream-core mask, channel-corridor mask, and
stream proximity. Each model predicts the scaled one-hour change in water
depth and x- and y-direction unit discharge (`qx`, `qy`; m²/s). Shared code
adds that change to the current state and applies the same domain and wet/dry
rules.

Training proceeds through 4-, 8-, 16-, and 32-hour forecasts, with 8,192
one-hour training updates per epoch and 2,048 fixed validation updates. Each
stage starts from the best saved model from the preceding stage. Training uses
32-bit calculations and the data-matching loss `2 × depth error + 0.5 × qx
error + 0.5 × qy error`. No separate
wetness, inundation, channel, rising-limb, or peak loss is used. PINO models
add the stated SWE consistency penalty. Report completed epochs, total hourly
updates, and GPU cost because models may stop after different epoch counts.

The scaled validation loss is not a physical error in metres. Manuscript
ranking must use the same free-running test forecasts and the same full-domain,
gauge, hydrograph, and peak-flood-map analysis.

Local comparison figures are regenerated with
`python analysis/generate_benchmark_figures.py --allow-incomplete --overwrite`. Products
are written under the Git-ignored `analysis/figures/` directory; the source
script and underlying evaluation tables remain version controlled.

## Selected main result

`houston-m032-common13-pino-depth4-plateau-v2` is the historical protocol ID
of the retained PINO-ResU-Net-large result. Its validation-best AR32 checkpoint (epoch 52) is
identified by checksum in [`ARTIFACTS.sha256`](ARTIFACTS.sha256). The
192-trajectory
evaluation and headline metrics are recorded in
[`docs/PINO_RESUNET_LARGE_V2_EVALUATION.md`](docs/PINO_RESUNET_LARGE_V2_EVALUATION.md).

The retained PINO-ResU-Net-large run continued from a checksum-recorded earlier AR4
epoch-120 saved model; the v2 workflow did not start from random weights. The
old v1 result tree and parent checkpoint were removed from publication scope;
only the recorded identity and checksum remain. Exact original v2 code remains
under `provenance/`. The unused prospective cold-start replay was moved to the
external cleanup archive; it was not an exact historical rerun and produced no
retained result. The PINO-ResU-Net-large evaluation entry point is
`workflows/evaluate_pino_resunet_large.sbatch` and refuses to overwrite the
retained export.

No other model currently has an equivalent retained physical forecast export.
Five models have completed AR32 saved models, but ResU-Net-large failed at
AR4. Therefore a six-model physical ranking and an estimate of the physics
loss effect for the large pair are not yet complete.

Do not move or modify a workflow, checkpoint lineage, lock, receipt, or source
snapshot while a dependent Slurm task is active or requeued.

## Integrity and release status

Run these commands after activating the Python 3.9 environment described in
[`environment/README.md`](environment/README.md); Perlmutter's base Python 3.6
is too old for this repository.

Verify selected model/evaluation artifacts with:

```bash
python3 scripts/check_repository.py
python3 -m pytest -q
sha256sum -c ARTIFACTS.sha256
python3 scripts/write_repository_checksums.py --check
```

`MANIFEST.sha256` covers the stable repository release surface and should be
regenerated only after all coordinated code and documentation edits finish.
The public license and dataset/archive DOI are release TODOs; none is claimed
here.

## Before the first GitHub upload

Saved models and scientific arrays are assigned to Git LFS by
`.gitattributes`. Initialize LFS before the first `git add`, then confirm that
the binary artifacts are represented by LFS pointers:

```bash
git init
git lfs install
git add .
git lfs ls-files
```

This workspace does not include the 148 GB trajectory store; GitHub users
resolve it through `HOUSTON_DATASET_ROOT` as documented in
[`TRAINING_DATA.md`](TRAINING_DATA.md). Select an explicit repository license
before making the remote public.

See [`docs/PROTOCOLS.md`](docs/PROTOCOLS.md) for the detailed comparison rules,
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for rerun guidance, and
[`AGENTS.md`](AGENTS.md) for the Codex/Claude maintainer handoff.
