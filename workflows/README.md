# Workflows

`m032_benchmark_training/` is the checked, read-only record and copy template
for the shared AR4→AR8→AR16→AR32 campaign. Never add a model to that completed
experiment; copy it to a new versioned directory and change both experiment
identifiers and the output directory.

`evaluate_pino_resunet_large.sbatch` reproduces the PINO-ResU-Net-large
physical evaluation with the retained source snapshots. Its default output is
the existing publication export, so it fails closed instead of overwriting it.
Set `PINO_RESUNET_LARGE_EVAL_OUTPUT_DIR` to a new empty directory for a rerun.

Runtime `logs/`, `cache/`, `locks/`, and `receipts/` are intentionally ignored
by Git. Model folders alone never launch jobs: only entries explicitly added
to a versioned `model_registry.json` become Slurm tasks.
