# Completed m032 depth-emphasis benchmark

This workflow records the five-model campaign that applied the common
comparison contract after PINO-ResU-Net-large had already been trained. The
`pino_resunet_large` result is not copied or retrained here;
`campaign_manifest.json` binds its AR32 validation-best checkpoint and stage
summary by SHA-256, and all six models appear together under
`results/m032_benchmark/`.

The official submission script reads the model count from
`model_registry.json` and sets the Slurm array range. In this completed
experiment, tasks `0..4` map to the five listed models. Every AR4 model uses
seeded random initialization. Every later stage starts from the saved model
with the lowest validation error in the preceding stage, with a fresh
optimizer and learning-rate controller. U-Net and FNO definitions are kept
under `src/vendor/baselines/`.

This workflow directory is now a read-only record: its submission script
refuses to launch another job under the completed v1 identity. The original
training source bound to the retained results is preserved under
`provenance/training_sources/m032_depth4_plateau_v1_original/`.

Do not add a model to this completed experiment. For a future model, copy the
workflow under a new versioned name, change the experiment/training-recipe IDs
in `runtime_env.sh` and `campaign_manifest.json`, copy `model/model_new0/` to a
new model folder, add its connector to the model-list entry, and write to the
matching new result directory. See the
[plain-language benchmark guide](../../docs/BENCHMARK_GUIDE.md).

After copying and assigning new IDs, validate the new workflow without
submitting:

```bash
cd /global/cfs/cdirs/m4267/dongyu/AI_benchmark
./workflows/<new_experiment>/submit_campaign.sh --dry-run
```

Submit only after reviewing the two printed `sbatch` commands:

```bash
./workflows/<new_experiment>/submit_campaign.sh
```

The retained outputs are under `results/m032_benchmark/`. Completed result
records and original source snapshots are immutable inputs.
