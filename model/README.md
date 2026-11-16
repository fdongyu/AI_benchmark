# Model folders

Each subfolder describes how the benchmark builds one emulator. The common
trainer remains responsible for the Houston data, 13 input maps, one-hour
state-change target, AR4 -> AR8 -> AR16 -> AR32 curriculum, losses, and saved
checkpoints.

The required file in an active model folder is `connector.py`. It provides:

```python
def create_model(*, in_channels, out_channels, settings):
    # Return torch.nn.Module.
    ...
```

The optional `model_details()` function records a short description in the
training provenance. The optional `source_files()` function lists the exact
repository files that implement the model so a future workflow can checksum
them.

Current folders use the stable model IDs already recorded in the benchmark:

- `unet_v2`
- `fno_v2`
- `resunet_small`
- `pino_resunet_small`
- `resunet_large`
- `pino_resunet_large`

The exact historical network implementations remain under `src/vendor/`
because completed checkpoints bind those paths and file hashes. The connector
files import those implementations; they do not copy the network code.

`pino_resunet_small` and `resunet_small` use the same small network.
`pino_resunet_large` and `resunet_large` use the same large network. “PINO” in
these names means that the experiment adds the shallow-water-equation loss;
that loss is controlled by the experiment model list, not hidden inside the
network connector.

To add a published emulator, copy `model_new0` to a new, descriptive model ID,
replace its example network, record the source and license in its README, and
add the new model to a new versioned experiment. Do not modify the completed
`m032_benchmark_training` experiment.

From the repository root, check the copied model with:

```bash
python scripts/check_model_connector.py --model <new_model_id>
```

Then add an explicit entry to the copied experiment's `model_registry.json`:

```json
{
  "model": "<new_model_id>",
  "architecture_key": "<new_model_id>",
  "connector": "model/<new_model_id>/connector.py",
  "settings": {},
  "source_files": [],
  "physics_regularizer": false,
  "initialization_policy": "describe_the_starting_weights",
  "learning_rate": 0.0001,
  "endpoint_learning_rate": 0.0,
  "slice_epochs": 12,
  "description": "Plain-language description of the emulator."
}
```

`source_files()` in the connector and `source_files` in this entry are both
included in the submission fingerprint. A folder is never auto-submitted;
only models listed in a versioned experiment become training jobs.
