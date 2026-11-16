# Fourier Neural Operator (`fno_v2`)

Four-layer, full-domain Fourier Neural Operator used as an architecture
baseline. It maps the common 13 input maps to three normalized one-hour
state-change maps and uses no shallow-water-equation loss.

The retained settings are 24 hidden channels, 12 by 20 Fourier modes, 48
lifting channels, 48 projection channels, and one-sided domain padding of
0.0625. The exact network source is `src/vendor/baselines/fno_model.py`.
