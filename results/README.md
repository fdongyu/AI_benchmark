# Benchmark results

`m032_benchmark/` is the single public comparison tree. It contains one
subfolder for each model:

- `unet_v2`
- `fno_v2`
- `resunet_small`
- `pino_resunet_small`
- `resunet_large`
- `pino_resunet_large`

All six models use the same Houston data split, 13 ordered inputs, normalized
one-hour state-change target, AR4 -> AR8 -> AR16 -> AR32 curriculum, 8,192
training forecast steps per epoch, depth/momentum weighting, and plateau
convergence rule. They can therefore be presented in one comparison tree.

The folders do not imply identical initialization histories. In particular,
PINO-ResU-Net-large resumed a separately recorded AR4 parent, whereas the five
models in the later shared campaign started from their declared campaign
initializations. Its historical protocol ID and original source snapshots are
preserved under `provenance/`; the files inside its result folder are not
rewritten. Consequently, absolute paths embedded in completed-run JSON files
may show the former NERSC execution locations. Those strings are historical
records, not current repository paths.

Use `publication_manifest.json` for current relative artifact locations and
checksums. Do not rank models using their normalized training objectives;
physical comparison requires the same retained rollout evaluation for every
model.
