# Analysis scripts

The publication cohort is U-Net, FNO, ResU-Net-small,
PINO-ResU-Net-small, ResU-Net-large, and PINO-ResU-Net-large. U-Net/FNO are
architecture baselines; only capacity-matched ResU-Net/PINO comparisons can
attribute an effect to the SWE loss.

- `plot_flood_evaluation.py`: hydrographs, peak-flood maps, and spatial-error maps from
  an evaluation export.
- `generate_benchmark_figures.py`: publication-style comparisons across every
  compatible evaluation export currently available.
- `lib/flood_evaluation/`: shared free-rollout and physical-metric functions.

Generate the comparison tables and figures with:

```bash
python analysis/generate_benchmark_figures.py --allow-incomplete --overwrite
```

The default output is `analysis/figures/`. That directory is deliberately
ignored by Git because every product is reproducible from the tracked script
and evaluation exports. The generator writes an availability table and refuses
to imply a complete comparison unless all requested exports have compatible
test populations, gauge mappings, truth series, and snapshots.
It refuses a non-empty output directory unless `--overwrite` is given, then
renders into a temporary directory before replacing the old generated output.
“USGS-location” metrics compare emulator depth with RDycore depth at mapped
gauge cells; they are not comparisons with observed USGS records.

Model-specific evaluation entry points live with their model connectors. For
PINO-ResU-Net-large, use `model/pino_resunet_large/evaluate.py`; its private
backend and historical checkpoint loader are documented in that model folder.
The shared evaluator still needs to be adapted and audited for the other five
models before claiming a complete six-model physical comparison.

Only PINO-ResU-Net-large currently has a retained physical rollout
export, under
`data/evaluations/pino_resunet_large_m032_ar32/`. Keep every future
model export in its own empty directory and use the same 192 test trajectories,
autoregressive update/projector, gauges, lead times, and metric definitions.
The canonical export is immutable and its provided Slurm wrapper will not
overwrite it.

`analysis/lib/pino/model.py` and `physics.py` intentionally mirror the
checksum-bound training implementations under `src/vendor/pino/`. The mature
evaluator imports them as a self-contained package, while completed training
provenance resolves the vendor paths. They are small compatibility copies, not
competing model definitions.

A normalized training/validation objective is not a physical-unit test
metric. ResU-Net-large has no valid AR32 checkpoint, so a six-model physical
ranking and large-capacity physics ablation remain incomplete.
