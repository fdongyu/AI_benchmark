# Maintainer handoff for Codex and Claude

This file is the authoritative agent handoff for the Houston `m032` benchmark.
Read it before editing code, workflows, metadata, results, or release files.

## Mission and publication scope

Maintain a reproducible six-model comparison:

1. U-Net — convolutional architecture baseline;
2. FNO — Fourier-operator architecture baseline;
3. ResU-Net-small — small data-only control;
4. PINO-ResU-Net-small — the same small architecture plus SWE consistency;
5. ResU-Net-large — large data-only control; currently failed at AR4; and
6. PINO-ResU-Net-large — large SWE treatment and selected reference result.

U-Net/FNO test architectures. Only a matched ResU-Net/PINO pair can attribute
an effect to the physics loss. Never describe the incomplete large pair as a
completed physics ablation.

## Read order

1. `README.md`
2. `docs/BENCHMARK_GUIDE.md`
3. `publication_manifest.json`
4. `docs/PROTOCOLS.md`
5. `TRAINING_DATA.md`
6. `docs/REPRODUCIBILITY.md`
7. the relevant workflow receipt, stage `training_config.json`,
   `stage_summary.json`, and `provenance.json`

## Current immutable facts

- Five models have completed validation-selected AR32 artifacts.
- ResU-Net-large job `60991759_8` failed during AR4 after epoch 9 because the
  gradient norm became non-finite; best AR4 validation supervised objective
  was 8.103914388056737 at epoch 7. There is no reportable AR32 artifact.
- `houston-m032-common13-pino-depth4-plateau-v2` is the historical protocol ID
  of PINO-ResU-Net-large. Its
  selected AR32 checkpoint is epoch 52 with SHA-256
  `189f6e4a7b5a5493d82ba17f7f563d50d5890660f8c390c0ab74cc6ed59e90eb`.
- That retained result historically continued a hash-bound earlier AR4
  epoch-120 parent. The publication cleanup removed the v1 result tree. The
  unused prospective `coldstart-v1` replay was archived outside this release;
  it produced no retained result and was not an exact replay of the retained
  lineage.
- Only PINO-ResU-Net-large currently has a retained 192-trajectory physical
  rollout evaluation.
- `release_ready` must remain false until the blockers in
  `publication_manifest.json` are resolved.

## Safety and provenance rules

- Inspect `squeue`, `sacct`, locks, and receipts before touching a campaign.
- Never edit, move, or replace a source snapshot, configuration, metadata
  file, workflow, receipt, lock, checkpoint, or output used by an active or
  requeued job.
- The completed shared-campaign registry has five trainable definitions, while
  PINO-ResU-Net-large has an earlier retained training lineage. Compact original
  trainer/manifest/registry snapshots under `provenance/training_sources/`
  exist only to satisfy SHA-bound completed-stage provenance. Their extra
  historical entries do not expand paper scope; never use or rewrite those
  snapshots as the active workflow.
- `workflows/m032_benchmark_training/` is a completed, read-only record. Its
  submission and runtime scripts refuse to launch under the old identity.
  Copy it to a new versioned directory and assign new experiment and
  training-recipe IDs before adding a model.
- Do not overwrite retained evaluation outputs. Write reruns to a new empty
  directory, validate them, then promote deliberately.
- Keep generated comparison products under `analysis/figures/`. This directory
  is intentionally ignored by Git; commit the generator and source evaluation
  tables, not locally rendered figures.
- The retained PINO-ResU-Net-large evaluator uses immutable source snapshots
  through `model/pino_resunet_large/load_historical_checkpoint.py`. Never overwrite or relabel the
  historical continuation-based PINO-ResU-Net-large result.
- Do not rank models using normalized validation objective values. Use
  identical held-out physical rollouts.
- Preserve failed treatments transparently. Do not substitute an intermediate
  checkpoint for AR32 or silently remove a failed model from the design.
- Do not invent a license, DOI, pretrained-weight source, or scientific
  result.

## Data and model contract

Resolve external data through `HOUSTON_DATASET_ROOT`; the canonical path and
hashes are in `TRAINING_DATA.md`. Preserve the 864/192/192 event-disjoint split
and train-only normalization.

A publication-compatible emulator adapter must:

- consume common13 in the documented fixed order;
- emit three normalized conservative-state increments for one hour;
- use the shared denormalization, residual update, domain mask, and wet/dry
  projector;
- train through AR4→AR8→AR16→AR32 with exposure counted in forecast-step
  transitions;
- use the common supervised selector and no hidden auxiliary loss; and
- expose parameter count, initialization, checkpoint parent, and hashes.

PINO differs from its paired ResU-Net only by the declared SWE consistency
objective. Any other treatment change invalidates a direct physics-loss
attribution and must be labeled as a model-system comparison.

## Adding a newly published emulator

Do not append a model to the completed `m032_benchmark_training` model list.
Copy that workflow to a newly named, versioned experiment and copy
`model/model_new0/` to `model/<new_model_id>/`. The common trainer can load a
connector named in the new model list, and the official submission script
derives its array size from that list.
Creating a folder does not schedule it: only an explicit entry in the copied
versioned `model_registry.json` becomes a Slurm task.

1. Add a minimal `model/<new_model_id>/connector.py` without changing frozen
   data or evaluation semantics.
2. Record upstream repository, commit, license, pretrained checkpoint source,
   and SHA-256. If any is unknown, leave an explicit blocker; never guess.
3. Run `scripts/check_model_connector.py`, then add shape, finite-value, and
   small multi-step tests.
4. Run a preflight on frozen metadata and a small new output path.
5. Submit a receipt-bound curriculum; record every parent checkpoint and
   realized transition exposure.
6. Require a converged, validation-selected AR32 checkpoint before promotion.
7. Run the exact 192-trajectory physical evaluation into a unique empty
   directory and generate the same gauge/spatial products.
8. Update `publication_manifest.json`, `ARTIFACTS.sha256`, docs, tests, and only
   then the repository manifest.

## Validation and release sequence

From the repository root:

Use the recorded Python 3.9 environment; do not use Perlmutter's base Python
3.6 interpreter.

```bash
python3 scripts/check_repository.py
python3 -m pytest -q
sha256sum -c ARTIFACTS.sha256
python3 scripts/write_repository_checksums.py --check
```

During coordinated edits, `--check` is expected to report a stale
`MANIFEST.sha256`; do not regenerate it until all agents finish. At release:

1. confirm the tree and Slurm state are quiescent;
2. validate JSON and Markdown links;
3. run tests and artifact checksum verification;
4. run `python3 scripts/write_repository_checksums.py` once;
5. run `python3 scripts/write_repository_checksums.py --check`; and
6. independently run `sha256sum -c MANIFEST.sha256`.

`ARTIFACTS.sha256` intentionally selects scientific artifacts, whereas
`MANIFEST.sha256` inventories the stable release surface. Do not create a
circular checksum by placing `MANIFEST.sha256` inside itself.

## Open release blockers

- obtain a stable AR32 ResU-Net-large result or formally revise the scientific
  design before release;
- produce identical physical rollout exports for all reportable models;
- audit/extend the common evaluator so all six model adapters share one metric
  contract;
- reconstruct the deleted hash-bound v1 parent if an exact end-to-end replay
  of the historical PINO-ResU-Net-large training lineage is required;
- select an explicit license; and
- publish the external training data (or approved access instructions) and
  record its real DOI/archive identifier.
