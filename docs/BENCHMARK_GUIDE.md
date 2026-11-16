# Guide to the Houston flood-emulator benchmark

## 1. Purpose

This repository compares neural-network emulators of two-dimensional flooding
in the Houston region. Hourly RDycore simulations provide the reference water
depth and flow fields.

The main question is:

> When models use the same flood events, input maps, state update, training
> schedule, and test data, how much does model design affect forecast accuracy?

The retained PINO-ResU-Net-large result has a different recorded initialization history
from the random-start shared models. Comparisons involving it are therefore
not pure architecture-only or physics-loss-only tests, even though the data,
inputs, state update, and staged forecast lengths are aligned.

A second question compares otherwise identical ResU-Net models trained with
and without a shallow-water-equation (SWE) loss. This matched comparison is
needed to separate the effect of that loss from differences in network design
or model size.

The repository contains six model records. Five reached the final 32-step
training stage. ResU-Net-large failed during the first stage and is retained so
the failure remains visible.

Two short labels occur throughout the files:

- `m032` is the retained data configuration with 32 ensemble members per
  event.
- `common13` is the fixed set of 13 input map layers used by every model.

## 2. Training data

The dataset contains hourly RDycore simulations:

- 39 flood events;
- 32 ensemble members per event;
- 1,248 trajectories in total; and
- 109–365 hourly states per trajectory.

The fixed split is made by event:

| Use | Events | Trajectories |
|---|---:|---:|
| Training | 27 | 864 |
| Validation | 6 | 192 |
| Testing | 6 | 192 |

An event belongs to only one group. Members of the same event therefore cannot
appear in both training and testing.

The 137.430-GiB trajectory collection is too large for GitHub. Its NERSC
location, file layout, relocation variable, and checksums are in
[`../TRAINING_DATA.md`](../TRAINING_DATA.md).

## 3. Information given to every model

Every model receives the same 13 map layers at each forecast step.

| Group | Map layer | Meaning |
|---|---|---|
| Current flood state | `h` | Water depth |
| Current flood state | `qx`, `qy` | Unit discharge in the two horizontal directions (m²/s) |
| Weather forcing | rainfall | Rainfall for the forecast hour; `log(1+x)` is applied before scaling |
| Coastal forcing | ocean stage | Water level at the ocean boundary |
| Terrain | bed elevation | Ground elevation |
| Model domain | domain mask | Cells included in the simulation |
| Boundary | ocean-boundary mask | Cells belonging to the ocean boundary |
| Location | normalized `x`, `y` | Position within the model grid |
| River network | stream-core mask | Main stream cells |
| River network | channel-corridor mask | Wider area surrounding the channels |
| River network | stream proximity | Continuous measure that decreases away from a stream |

The current state has three layers and location has two layers, so the table
contains 13 input channels. Scaling statistics are calculated from the
training group only.

## 4. What every model predicts

All six models predict the change during the next hour, not the complete next
state. The three output maps are the scaled one-hour changes in `h`, `qx`, and
`qy`.

In simplified form:

```text
next state = wet/dry check(current state + predicted one-hour change)
```

The shared state-update code:

1. converts the predicted scaled change back to model units;
2. adds it to the current state;
3. prevents negative water depth;
4. sets discharge to zero in dry cells;
5. limits unrealistically large velocity; and
6. removes values outside the model domain.

The model's own prediction becomes the input for the following hour. RDycore
truth is not inserted between forecast steps. This is also called a
free-running or autoregressive forecast.

The word *residual* has two meanings here:

- **Residual state update:** every model predicts a change and adds it to the
  current flood state.
- **Residual neural block:** ResU-Net also has short internal connections that
  add a block's input to its output.

These are separate ideas.

## 5. Model descriptions

### U-Net

U-Net is a convolutional network that first reduces map resolution to learn
large spatial patterns and then restores full resolution. Connections between
matching resolution levels preserve local flood-boundary information.

The retained U-Net has three downsampling and three upsampling steps, starts
with 16 feature maps, has 489,475 trainable parameters, and does not use the
SWE loss.

### FNO

The Fourier Neural Operator (FNO) learns spatial patterns with Fourier modes.
This gives it a direct way to represent broad, full-domain interactions in
addition to local detail.

The retained FNO has four Fourier layers, keeps 12 by 20 Fourier modes, has
613,851 trainable parameters, and does not use the SWE loss.

### ResU-Net-small

ResU-Net has the same general downsample–upsample layout as U-Net, but each
convolutional block has an internal shortcut connection. These shortcuts can
make a deeper network easier to train.

“Depth 4” means four map-resolution levels; it does not mean only four neural
layers. The small model starts with 14 feature maps, has 535,825 trainable
parameters, and does not use the SWE loss.

### PINO-ResU-Net-small

This model uses exactly the same network and parameter count as
ResU-Net-small. Its declared training difference is the additional
two-dimensional SWE consistency loss. This completed pair can therefore be
examined to study the physics loss at the small model size, once both models
have the same physical evaluation.

Here, PINO means Physics-Informed Neural Operator. It describes the training
setup; it is not a different network architecture from the paired ResU-Net.

### ResU-Net-large

This is the data-only counterpart of PINO-ResU-Net-large. It starts with 24
feature maps, has four resolution levels, and has 1,570,934 total parameters
(1,570,875 trainable in the matched setup).

Its retained run failed during AR4 when training became numerically unstable
(the gradient norm was non-finite).
The best observed AR4 validation score occurred at epoch 7. No checkpoint was
retained from the failed run, and no valid AR32 checkpoint exists. It must not
be presented as a completed result.

### PINO-ResU-Net-large

This model uses the same large ResU-Net design and adds the SWE consistency
loss. It has 1,570,934 total parameters and 1,570,875 trainable parameters. A
small output-adjustment component is kept neutral and frozen to match the
earlier PINO-ResU-Net-large design.

This is the selected PINO-ResU-Net-large result. It completed all training stages and is
the only model with a retained evaluation over all 192 test trajectories; that
does not yet prove that it is physically best among all six models.

The retained v2 run continued from a recorded AR4 checkpoint from an earlier
PINO-ResU-Net-large run. That parent checkpoint file has since been removed; its identity and
checksum remain in the record, so an exact end-to-end historical replay is not
currently possible. An unused seeded cold-start replay was archived during
repository cleanup; it produced no retained result and must not be confused
with the continuation history. Because the
large data-only model failed and PINO-ResU-Net-large has the continuation history, the
large pair cannot currently isolate the SWE-loss contribution.

## 6. Common staged training schedule

Training increases the forecast length in four stages:

```text
AR4 → AR8 → AR16 → AR32
```

`AR4` means four consecutive one-hour predictions. `AR32` means 32
consecutive one-hour predictions. Starting with a short forecast and then
increasing its length helps a model learn to recover from errors that build up
when it uses its own previous predictions.

At each stage:

- one training epoch contains exactly 8,192 one-hour forecast updates;
- validation uses the same fixed 2,048 one-hour forecast updates;
- sampling balances events, members, and forecast windows;
- the next stage starts from the saved model with the lowest validation error
  in the preceding stage;
- the optimizer and learning-rate controller restart;
- at least 12 epochs are required;
- training stops after eight epochs without a validation improvement of at
  least 0.1%; and
- 240 epochs is the safety limit. Reaching it without convergence is reported
  as a failure.

The stopping rule is common, but models may finish after different numbers of
epochs. Reports should therefore include the total number of epochs and the
total number of one-hour updates used in training.

All retained runs use 32-bit calculations. Reduced-precision training is
disabled, and the random seed is fixed. Deterministic PyTorch algorithms are
not enforced, however, so an exact numerical replay is not guaranteed. AdamW
is the optimizer. Starting learning rates may differ by model and are recorded
in the model list and result files.

## 7. Training losses

### Data-matching loss

The prediction is compared with the RDycore one-hour change inside the valid
domain. The retained weighting is:

```text
2 × depth error + 0.5 × qx error + 0.5 × qy error
```

This gives water depth four times the relative emphasis of either discharge
component.

There is no separate wetness, inundation, channel, rising-limb, or flood-peak
loss. Adding such a term to only one model would change the comparison. The
saved “best” model is selected with the validation data-matching loss only.

### Shallow-water-equation loss

The two PINO models add checks based on the two-dimensional shallow-water
equations:

- conservation of water mass, including rainfall;
- consistency of momentum in the two horizontal directions; and
- effects of bed slope, gravity, and Manning resistance.

The ocean-boundary neighborhood and unsuitable wet/dry shoreline stencils are
excluded from this check. Manning's coefficient is used only in the momentum
consistency calculation. It is not an input map or a separate prediction
target.

The physics contribution is introduced gradually during the first two epochs
of each stage. The best checkpoint is still selected with the data-matching
validation score, not the physics term.

## 8. Current training status

The AR32 validation score below is a scaled training diagnostic. Lower is
better, but the value is not an error in metres and is not sufficient for the
manuscript ranking.

| Model | Parameters | SWE loss | Final 32-step stage | Best AR32 validation score |
|---|---:|---:|---|---:|
| U-Net | 489,475 | No | Complete | 33.4581 |
| FNO | 613,851 | No | Complete | 24.2027 |
| ResU-Net-small | 535,825 | No | Complete | 5.3277 |
| PINO-ResU-Net-small | 535,825 | Yes | Complete | 5.6153 |
| ResU-Net-large | 1,570,934 total | No | Failed during AR4 | — |
| PINO-ResU-Net-large | 1,570,934 total | Yes | Complete | 3.0332 |

The small PINO does not have a lower AR32 data-matching score than its matched
data-only model. Any claim that the SWE loss improves flood forecasts must use
the common physical evaluation, not this table alone.

## 9. Flood-forecast evaluation

The final comparison should use a full-event free-running forecast for every
one of the same 192 test trajectories. The retained test events require
109–160 consecutive one-hour predictions; AR32 is the training-window length,
not the physical-evaluation horizon. The comparison should report:

- full-domain depth RMSE, MAE, bias, NSE, and KGE;
- agreement of flooded area at stated depth thresholds;
- spatial error in maximum water depth;
- flood-peak timing error;
- hydrograph and peak errors at the same USGS gauges;
- event-by-event and ensemble-member distributions;
- number of model parameters;
- one-hour updates used in training; and
- GPU training and forecast cost.

Plain-language metric definitions:

- **RMSE:** typical error size, with extra penalty for large errors.
- **MAE:** average absolute error size.
- **NSE:** hydrologic agreement; 1 is ideal.
- **KGE:** combined agreement in correlation, variability, and mean; 1 is
  ideal.
- **CSI:** overlap between predicted and reference flooded cells; 1 is ideal.
- **Peak-time error:** difference between predicted and reference flood-peak
  time.

Only PINO-ResU-Net-large currently has a retained physical evaluation over all
192 test trajectories and 14 mapped USGS locations. The gauge metrics compare
PINO with RDycore at those locations; they are not comparisons with observed
USGS gauge records.

For each metric, the evaluator first takes the median within an event and then
the median across the six test events. The table is not a pooled-grid or
pooled-trajectory summary.

| Metric | Median |
|---|---:|
| Full-domain depth RMSE | 0.024002 m |
| Full-domain depth MAE | 0.007070 m |
| Full-domain depth NSE | 0.989550 |
| Full-domain CSI at 0.10 m | 0.755732 |
| Peak-depth RMSE | 0.050546 m |
| Peak-time MAE | 1.845623 h |
| Gauge depth RMSE | 0.045705 m |
| Gauge depth NSE | 0.980046 |
| Gauge absolute peak-depth error | 0.109662 m |
| Gauge absolute peak-time error | 0 h |

See
[`PINO_RESUNET_LARGE_V2_EVALUATION.md`](PINO_RESUNET_LARGE_V2_EVALUATION.md)
for the complete table and paths to hydrographs and maps.

Equivalent physical exports do not yet exist for the other models, so the
repository does not yet support a complete six-model physical ranking.

## 10. One comparison result directory

`results/m032_benchmark/` contains all six model records: U-Net, FNO,
ResU-Net-small, PINO-ResU-Net-small, ResU-Net-large, and
PINO-ResU-Net-large. This is a presentation layout, not a claim of identical
initialization. PINO-ResU-Net-large resumed an earlier AR4 run, while the later
five-model campaign used its recorded campaign initializations. The original
protocol IDs, paths, and source hashes remain unchanged inside immutable
result and provenance records.

## 11. Repository map

| Location | Purpose |
|---|---|
| `model/` | One readable plug-in folder per emulator, plus the new-model template |
| `src/` | Shared training and connector-loading code |
| `src/vendor/` | Exact published or previously developed model code needed for reproduction |
| `workflows/` | Checks and Slurm submission scripts for each versioned experiment |
| `results/` | Selected checkpoints and training-stage records |
| `data/metadata/` | Fixed split, scaling, grid, masks, and gauge locations |
| `data/evaluations/` | Physical-unit metrics, hydrographs, and maps |
| `analysis/` | Common forecast evaluation and plotting |
| `docs/` | Human-readable explanations and result summaries |
| `scripts/` | Connector, experiment-manifest, and repository checksum checks |
| `provenance/` | Preserved copies of code and settings tied to completed runs |
| `environment/` | Python environment record |

A reproducibility record says exactly which code, data, settings, and saved
model produced a result. A SHA-256 checksum is a file fingerprint used to
detect accidental changes.

## 12. Adding a new emulator

A new model should use a small **model connector** rather than copying the
data loading, loss, or staged-training code. The connector has one simple
input/output rule:

```text
input:  [batch, 13, 256, 512]
output: [batch,  3, 256, 512]
```

The three output maps are the scaled one-hour changes in `h`, `qx`, and `qy`.
The shared trainer remains responsible for sampling, state updates, wet/dry
handling, losses, training stages, and saved models. Connector support is not
yet wired into the official physical evaluator; that evaluator must be
generalized and audited before a new model can receive the same official
full-domain, gauge, hydrograph, and map products.

Use this five-step workflow:

1. **Create one model folder.** Copy `model/model_new0/` to
   `model/<model_id>/`. Put the exact published code in that folder (for
   example, under `upstream/`) or record the pinned package dependency. Record
   its source URL, commit or release, license, and pretrained-weight checksum.
2. **Connect the input and output.** Edit the copied `connector.py` so
   `create_model()` returns the published network, and make `source_files()`
   list the exact implementation files that must be checksummed.
3. **Run a small compatibility check.** For example:

   ```bash
   /global/common/software/m3780/feng779/conda/rdycore-ai-gpu-py39/bin/python \
     scripts/check_model_connector.py \
     --model <model_id> \
     --settings-json '{}'
   ```

   This small check uses a reduced map by default. If the published model
   only accepts the fixed Houston grid, add `--height 256 --width 512`, or let
   the required GPU preflight perform the full-size check before training.

4. **Create a new versioned experiment.** Copy the current workflow to a new
   directory. Give it a new experiment ID, training-recipe ID, model list, and
   result directory. Add one model-list entry with
   `"connector": "model/<model_id>/connector.py"`, settings,
   source files, starting method, learning rate, and SWE-loss choice. Then run:

   ```bash
   /global/common/software/m3780/feng779/conda/rdycore-ai-gpu-py39/bin/python \
     scripts/sync_experiment_manifest.py \
     --experiment-dir workflows/<new_experiment> --write
   ```

   This synchronizes `model_keys`, `physics_treatments`, the model-list
   checksum, and the recorded Slurm array. The official submission script also
   calculates its live array from the model count. It rejects a copied
   workflow that still uses either completed v1 ID.
5. **Train, then extend the evaluator.** Run the preflight and the common
   AR4→AR8→AR16→AR32 schedule. Add and audit a connector-aware loader in the
   physical evaluator before evaluating the selected AR32 model on the fixed
   192-trajectory test group. Then add its checkpoint, metrics, figures,
   source identity, and checksums to the publication record.

Do not append a model to the completed `m032_benchmark_training` experiment.
Its original exact training sources are preserved under `provenance/`. A new
model gets a new versioned experiment so old results keep their meaning.

A model belongs in the direct-comparison table only if it uses the common
inputs, one-hour change target, state update, data loss, staged schedule, and
test evaluation. If a published method requires different inputs,
absolute-state prediction, special losses, pretraining, or a different time
step, keep those scientifically important choices but label the run as an
**adapted training recipe**. It is informative, but it is not a change of
architecture alone.

Pretrained foundation models should keep their published weights and should be
reported as a pretrained-model comparison rather than a random-start control.
The connector interface does not yet provide centrally checked hot-start
loading. Before an official foundation-model run, add and test core handling
that verifies the starting checkpoint checksum and records its source,
release/commit, and license; do not hide checkpoint loading inside
`create_model()`.

More connector details are in [`../model/README.md`](../model/README.md), and
the editable starting instructions are in
[`../model/model_new0/README.txt`](../model/model_new0/README.txt).

## 13. Short glossary

- **AR4 / AR8 / AR16 / AR32:** 4, 8, 16, or 32 consecutive one-hour
  predictions.
- **Checkpoint / saved model:** model weights saved at a point in training.
- **Cold start:** training begins from seeded random weights.
- **Warm or hot start:** training begins from an existing trained checkpoint.
- **Curriculum / staged schedule:** forecast length increases during training.
- **Epoch:** one training cycle with the fixed sampling budget.
- **Free-running forecast:** each prediction is used to make the next one,
  without correction from RDycore truth.
- **Model connector:** small file that makes a model accept the benchmark
  inputs and return the required outputs.
- **Model size:** number of trainable parameters and the scale of the network.
- **Physics loss:** penalty for inconsistency with the shallow-water
  equations.
- **Residual state update:** predict the change and add it to the current
  state.
- **Validation group:** simulations used to decide when to stop and which
  saved model to select.
- **Test group:** simulations used only for final reported evaluation.
- **Validation plateau:** validation performance has stopped improving by the
  stated amount for the stated number of epochs.
