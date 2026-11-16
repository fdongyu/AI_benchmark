# PINO-ResU-Net-large m032 evaluation

This report records the full-event, free-running evaluation of the retained
AR32-trained PINO-ResU-Net-large model with historical protocol ID
`houston-m032-common13-pino-depth4-plateau-v2`. The evaluated checkpoint
is `checkpoint_best_validation_supervised.pt` from epoch 52, with validation
supervised score `3.0332421664857083` and SHA-256
`189f6e4a7b5a5493d82ba17f7f563d50d5890660f8c390c0ab74cc6ed59e90eb`.

The evaluation contains all 192 test trajectories (6 independent events x 32
members), 14 mapped USGS locations, and full-event free-running forecasts of
109–160 one-hour steps. AR32 is the training-window length, not the evaluation
horizon. Gauge metrics compare PINO-ResU-Net-large with RDycore at the USGS
locations; they are not comparisons with observed USGS records. The evaluator takes the
median within each event and then the median across the six events; values are
not pooled-grid statistics.

| Metric | Median |
|---|---:|
| Full-domain depth RMSE | 0.024002 m |
| Full-domain depth MAE | 0.007070 m |
| Full-domain depth NSE | 0.989550 |
| Full-domain depth KGE | 0.951322 |
| Full-domain CSI at 0.10 m | 0.755732 |
| Peak-depth RMSE | 0.050546 m |
| Peak-depth correlation | 0.995854 |
| Peak-time MAE | 1.845623 h |
| USGS-gauge depth RMSE | 0.045705 m |
| USGS-gauge depth MAE | 0.023378 m |
| USGS-gauge depth NSE | 0.980046 |
| USGS-gauge depth KGE | 0.888153 |
| USGS-gauge absolute peak-depth error | 0.109662 m |
| USGS-gauge absolute peak-time error | 0 h |

Relative to the earlier PINO-ResU-Net-large evaluation, this retained model reduces
full-domain depth RMSE by 3.44%, gauge depth RMSE by 4.91%, peak-depth RMSE by
1.81%, and gauge absolute peak-depth error by 1.93%. It is not better on every
metric: full-domain and gauge MAE, gauge NSE, and peak-time MAE are modestly
worse. The canonical choice therefore reflects the requested RMSE/peak-depth
priority rather than uniform dominance.

## Reproducible products

- Machine-readable full-domain metrics:
  `data/evaluations/pino_resunet_large_m032_ar32/domain_summary_metrics.csv`
- Machine-readable gauge metrics:
  `data/evaluations/pino_resunet_large_m032_ar32/gauge_summary_metrics.csv`
- Event hydrographs:
  `data/evaluations/pino_resunet_large_m032_ar32/figures/event_hydrographs/`
- Peak-depth maps:
  `data/evaluations/pino_resunet_large_m032_ar32/figures/peak_flood_maps/`
- Full-domain spatial-error map:
  `data/evaluations/pino_resunet_large_m032_ar32/figures/spatial_error_maps/full_domain_spatial_errors.png`
- Evaluator:
  `model/pino_resunet_large/evaluate.py`
- Perlmutter analysis workflow:
  `workflows/evaluate_pino_resunet_large.sbatch`

The canonical `figures/` directory is the sole retained rendering of these
metrics and rollouts.
