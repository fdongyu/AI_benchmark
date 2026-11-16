# ResU-Net-large training failure

The data-only large ResU-Net did not produce a reportable AR32 result. Slurm
array task `60991759_8` reached nine logged AR4 epochs and failed during the
next training epoch when gradient clipping detected a non-finite L2 gradient
norm. No checkpoint from this failed run is retained.

- Protocol: `houston-m032-common13-depth4-plateau-v1`
- Stage: AR4
- Best logged epoch: 7
- Best normalized validation supervised loss: `8.103914388056737`
- Logged training exposure: 73,728 forecast-step transitions
- Failure: `RuntimeError` from `torch.nn.utils.clip_grad_norm_` with
  `error_if_nonfinite=True`
- Publication status: failed/incomplete; no AR32 metric

`history.csv` is the exact nine-row metric history extracted from the Slurm
standard output. `failure_summary.json` records the source-log hashes so the
compact record remains auditable after runtime logs are removed.
