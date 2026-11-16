# Original PINO depth4 plateau v2 source snapshot

The completed canonical PINO-ResU-Net-large result was trained with these
exact files before the deleted v1 continuation parent was removed from the
publication workspace. They are immutable provenance evidence, not the active
cold-start workflow.

| File | SHA-256 |
|---|---|
| `train_pino_depth4_plateau_v2.py` | `0c1b01460c2cd0cf7d8a1a375eec5a14f64471254b8157448229cd362cc16ea9` |
| `campaign_manifest.json` | `e079d0b62d9bc3e1b05d965c93a1cea2c827266a07ce4746ca041249556edd6c` |

Only the `files.trainer.path` and `files.campaign_manifest.path` identity
locations in the four retained stage provenance records are migrated to this
snapshot. Their SHA-256 values remain unchanged, and the historical
`continuation_import.json`, history, configuration, and continuation blocks
remain unchanged.

The two historical submission receipts are also left byte-for-byte unchanged.
Their original trainer and manifest paths no longer match the repaired active
workflow by design. `receipt_cleanup_migration.json` records the receipt hashes,
the exact snapshot relocation, and the active orchestration paths changed or
removed by the cold-start repair.
