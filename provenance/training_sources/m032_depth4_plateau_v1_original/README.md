# Original shared-campaign provenance snapshot

The completed U-Net, FNO, ResU-Net-small, and PINO-ResU-Net-small stages were
created before the publication repository removed out-of-scope PINO-FNO,
Poseidon, and MORPH branches. Their stage provenance binds the exact original
trainer and campaign files by SHA-256.

These files are immutable evidence, not the current executable workflow:

| File | SHA-256 |
|---|---|
| `train_m032_depth4_plateau.py` | `5ae66372f94f22659df9bd7884d815d45693638493f18551b091ed629cb85ff1` |
| `campaign_manifest.json` | `6e5e9e90896377677a37e14a8e53bcfb4da58ef43f30072cfce3a998dc8227e8` |
| `model_registry.json` | `b4578a55aadc43c5a3921a083825f57a16bdf27535790c261607a254c1db2db4` |

The one-time cleanup migration is complete. Its helper script was moved to the
external `AI_archive/AI_benchmark_cleanup_2026-11-16/` bundle rather than kept
as an active repository command. Expected content hashes remain unchanged,
and each migrated stage record carries the explicit audit note needed to
verify the relocation without rerunning the migration.
