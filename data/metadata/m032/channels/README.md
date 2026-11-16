# Channel metadata

`channel_masks_fullres.npz` contains the stream-core, channel-corridor, and
stream-proximity maps used by the common13 inputs. `core_gauges.csv` records
the 14-gauge source list used when those maps were prepared, and
`metadata.json` records the derivation settings.

The old high-flow sampling-priority table and its QA plot were not read by the
retained depth4/plateau training protocol, which uses fixed hierarchical
transition exposure. They were moved to the dated external cleanup archive
instead of being published as active benchmark inputs.
