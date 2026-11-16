# Reproducibility guide

## Runtime and data boundary

Training and analysis run on NERSC Perlmutter A100 GPUs under account
`m3780_g`. The observed Python environment is recorded in
`environment/pip-freeze.txt`, with key versions in `environment/README.md`.
The training contract disables AMP and TF32. Submission receipts bind
executable source and configuration by SHA-256.

The repository includes the fixed split, train-only normalization, channel
masks, static grid, and USGS-gauge mapping under `data/metadata/`. The full
hourly trajectory store is external. Set:

```bash
export HOUSTON_DATASET_ROOT=/global/cfs/cdirs/m4267/dongyu/AI_foundation/DATA/rdycore_ml_dataset_core
```

For a direct historical trainer invocation, pass
`--dataset-root "$HOUSTON_DATASET_ROOT"` (or set its historical shell variable
`DATASET_ROOT="$HOUSTON_DATASET_ROOT"`). Receipt-bound historical workflows
retain their recorded absolute path and must not be edited. The HDF5
trajectories are not duplicated. Layout, split semantics, sizes, and hashes
are specified in [`../TRAINING_DATA.md`](../TRAINING_DATA.md).

## Publication cohort

The exact six definitions are U-Net, FNO, ResU-Net-small,
PINO-ResU-Net-small, ResU-Net-large, and PINO-ResU-Net-large. U-Net/FNO are
architecture baselines; only equal-capacity ResU-Net/PINO pairs provide
physics attribution. Consult
[`../publication_manifest.json`](../publication_manifest.json) before using a
result.

Five models have validation-selected AR32 checkpoints. ResU-Net-large failed
at AR4 with a non-finite gradient norm and has no reportable AR32 artifact. Do
not substitute its AR4 best checkpoint, omit the treatment, or infer a
large-pair physics effect. Before reporting any checkpoint, verify its stage
summary, protocol identifier, feature order, initialization, parent
checkpoint, realized exposure, convergence status, and SHA-256.

While a Slurm task is running or requeued, do not edit or move any source,
workflow, metadata, lock, receipt, or checkpoint path named by that task. The
active shared-campaign registry contains five trainable definitions;
PINO-ResU-Net-large uses its separately retained historical lineage. Compact original
trainer, campaign-manifest, and registry snapshots under
`provenance/training_sources/` are immutable evidence retained only because
completed stage records bind their SHA-256 values. Extra historical entries
inside those snapshots are not active executable or publication scope.

## Retained PINO-ResU-Net-large

The completed reference result is:

```text
results/m032_benchmark/pino_resunet_large/stage_ar32/
```

Its selected artifact is `checkpoint_best_validation_supervised.pt` at epoch
52. The campaign manifest verifies both the checkpoint and stage summary
before dependent work proceeds.

The retained PINO-ResU-Net-large lineage historically continued a hash-bound
earlier AR4 epoch-120 parent. The earlier result tree was deliberately removed from this
publication workspace. Exact original v2 source identities remain under
`provenance/training_sources/pino_depth4_plateau_v2_original/`. The unused
prospective cold-start replay was moved to the external cleanup archive; it
produced no retained scientific result and was not a bitwise reproduction of
the retained lineage.

The retained 192-trajectory physical evaluation is at:

```text
data/evaluations/pino_resunet_large_m032_ar32/
```

It contains full-domain, lead-time, event, trajectory, and 14-gauge metrics,
plus event/station hydrographs, peak-flood maps, and a spatial-error map. The
canonical wrapper is `workflows/evaluate_pino_resunet_large.sbatch`.
It intentionally refuses to overwrite an existing export. To reproduce rather
than verify the retained copy, set `PINO_RESUNET_LARGE_EVAL_OUTPUT_DIR` to a
new empty directory; never overwrite the retained export. The wrapper loads the
byte-exact historical trainer through
`model/pino_resunet_large/load_historical_checkpoint.py`, which supplies the
stable vendored network path without modifying the source snapshot.

The concise scientific record is
`docs/PINO_RESUNET_LARGE_V2_EVALUATION.md`.
`model/pino_resunet_large/evaluate.py` is the strict model-specific adapter,
`analysis/plot_flood_evaluation.py` renders its products, and the byte-exact
historical evaluator remains under `provenance/` for audit. The generalized
six-model evaluation path remains a release blocker until every model can be
evaluated through the same adapter contract and written to separate outputs.

## Comparable reporting

For every converged AR32 model, run the same free-rollout evaluator over the
same 192 held-out trajectories. Report physical metrics and uncertainty by
event/member, as well as parameter count, completed epochs, forecast-step
exposure, GPU-hours, peak memory, initialization, and checkpoint hash. Do not
compare normalized training losses across protocols or present them as
physical RMSE.

Only PINO-ResU-Net-large currently has a retained physical rollout
export. U-Net and FNO support an architecture comparison only after the same
evaluation. The small pair supports physics attribution only after both
receive that evaluation. ResU-Net-large must first produce a valid AR32
checkpoint.

## Integrity

`ARTIFACTS.sha256` binds selected AR32 checkpoints, stage records, and the
canonical evaluation. The exact shared rollout source used by that evaluation
is retained under `provenance/evaluation_sources/`. `MANIFEST.sha256` covers stable source, workflow,
metadata, environment, evaluation, report, and documentation files. Check it
without mutation using
`python3 scripts/write_repository_checksums.py --check`.

Within a completed stage, `checkpoint_last.pt` and `checkpoint_final.pt` may
be hard links to the same bytes, and `slice_status.json` may match
`stage_summary.json`. Keep both names: the workflow assigns them different
restart, handoff, and audit roles, and hard-linked checkpoints do not consume
duplicate physical storage on this filesystem.

Regenerate the manifest only from a quiescent, audited tree after all
coordinated edits finish; do not freeze a release while a campaign task is
active or requeued. The public license and dataset/archive DOI remain explicit
release TODOs; none is claimed here.
