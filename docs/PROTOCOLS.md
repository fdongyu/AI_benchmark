# Publication protocol

## Six-model scientific design

The publication cohort has two architecture baselines and two matched
data-only/physics pairs:

| Comparison class | Data-only | Physics-informed | Permitted inference |
|---|---|---|---|
| Architecture baselines | U-Net, FNO | — | Model-system performance under the common training contract |
| Small matched pair | ResU-Net-small | PINO-ResU-Net-small | Effect of adding the SWE loss at small capacity |
| Large matched pair | ResU-Net-large | PINO-ResU-Net-large | Effect of adding the SWE loss at large capacity, once both are complete |

U-Net and FNO do not share the ResU-Net architecture and cannot isolate the
physics loss. Their shared contract covers data, features, target/update,
exposure curriculum, supervised criterion, precision, and evaluation;
architecture-specific optimization settings remain hash-recorded. Within a
completed matched pair, architecture, parameterization,
initialization, features, sampling, optimizer, curriculum, state update, and
supervised loss must be identical. The PINO treatment adds the declared 2-D
shallow-water-equation consistency term. Only such a pair supports a direct
statement about the contribution of that term. A comparison between small and
large models also changes capacity and is not a physics ablation.

The small pair reached AR32. PINO-ResU-Net-large reached AR32 and is the
canonical v2 result. ResU-Net-large failed during AR4 after epoch 9 when
gradient clipping detected a non-finite norm; its best observed AR4 validation
objective was 8.103914388056737 at epoch 7. This incomplete run is not an AR32
result and cannot support a large-pair physics claim.

## Common data and features

All treatments use the fixed event-level `m032` assignment: 864 training, 192
validation, and 192 test trajectories, with train-only normalization. Inputs
have this exact order:

1. water depth `h`;
2. unit discharges `qx`, `qy`;
3. rainfall;
4. ocean stage;
5. bed elevation;
6. domain mask;
7. ocean-boundary mask;
8. normalized `x`, `y`;
9. stream-core mask;
10. channel-corridor mask; and
11. stream proximity.

These are 13 channels because the state and coordinate groups contain three
and two channels, respectively.

The external data location, relocatable-path rule, layout, and hashes are in
[`../TRAINING_DATA.md`](../TRAINING_DATA.md).

## State update and supervised criterion

The common six-model protocol predicts the normalized one-hour change
in conservative state `(h, qx, qy)`. The shared external update denormalizes
the increment, adds it to the current state, and applies the same domain and
wet/dry projector. The literal supervised quantity used for optimization and
validation selection is

\[
L_{\mathrm{state}} = 2D + 0.5Q_x + 0.5Q_y,
\]

where each term is a domain-masked MSE in normalized increment space. This is
equivalent to a 4:1:1 relative emphasis on depth. The publication protocol
does not add a wetness, inundation, channel, rising-limb, or peak auxiliary
loss.

For PINO treatments, the objective additionally includes configured mass and
momentum consistency terms derived from the 2-D shallow-water equations. The
configuration records their weights, scales, warm-up, gravity, Manning
coefficient, grid spacing, and one-hour time step. Manning resistance acts
only through the momentum-residual consistency calculation; it is not an
extra input feature or separate supervised target.

## Curriculum and convergence

Training advances through AR4→AR8→AR16→AR32. Each epoch receives 8,192
training forecast-step transitions; validation uses a fixed 2,048-transition
sample. Each stage begins from the preceding stage's validation-best weights,
restarts AdamW and `ReduceLROnPlateau`, and uses the locked convergence rule:
minimum 12 epochs, patience 8 after a minimum relative improvement of 0.1%,
with a 240-epoch safety ceiling. Reaching the ceiling without convergence is
reported as a failure, not silently accepted.

Strict FP32 is used throughout: AMP and TF32 are disabled, gradients are
clipped at 1, and stage handoff is validation-best rather than final epoch.
Plateau stopping can produce different realized epoch counts, so every table
must disclose completed epochs, total forecast-step exposure, and GPU-hours.

## Retained PINO-ResU-Net-large

The completed `houston-m032-common13-pino-depth4-plateau-v2` run is the
retained PINO-ResU-Net-large model. It restores that model's depth emphasis and
learning-rate convergence while using common13 features. The completed stages
converged at 144, 84, 73, and 60 epochs; the selected AR32 checkpoint is epoch
52. Its checkpoint and summary are identified by SHA-256 in the retained
campaign provenance.

This is a protocol reproduction on the common13 schema, not a byte-identical
rerun of the earlier 11-feature experiment.

The v2 lineage continued a hash-bound AR4 parent from the v1 lineage before
completing the v2 curriculum. That continuation is recorded in the stage
configuration and provenance and must not be described as a byte-identical
cold start.

Publication cleanup removed the v1 result tree. The exact historical v2
trainer and campaign manifest remain under `provenance/training_sources/`.
The later seeded AR4 cold-start replay had a distinct identity and
initialization, produced no retained result, and was moved to the external
cleanup archive. It cannot be relabeled as historical v2.

## Reporting contract

Training/validation objective values can diagnose convergence only. Final
model ranking uses identical held-out free rollouts and reports, at minimum,
full-domain depth RMSE/MAE/bias, USGS-gauge hydrograph and peak/timing error,
peak-depth-map error, event/member distributions, parameter count, exposure,
and compute cost.

At present, only PINO-ResU-Net-large has a retained physical
evaluation over all 192 test trajectories. Consequently no six-model physical
ranking, and no completed large-pair physics attribution, is yet justified.
