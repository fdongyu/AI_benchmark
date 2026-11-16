# Houston training-data contract

## Canonical NERSC location

The benchmark uses the RDycore hourly trajectory store at exactly:

```text
/global/cfs/cdirs/m4267/dongyu/AI_foundation/DATA/rdycore_ml_dataset_core
```

Use one relocatable environment variable in new wrappers:

```bash
export HOUSTON_DATASET_ROOT=/global/cfs/cdirs/m4267/dongyu/AI_foundation/DATA/rdycore_ml_dataset_core
```

Pass `--dataset-root "$HOUSTON_DATASET_ROOT"` to a direct trainer invocation.
Historical receipt-bound workflows retain the recorded absolute path and must
not be rewritten after submission.

## Layout and volume

The canonical tree has this layout:

```text
rdycore_ml_dataset_core/
├── static_grid.h5
├── dataset_manifest.csv
├── dataset_summary.json
├── verification.json
├── seed_summary.json
└── trajectories/
    └── <event_id>/
        └── member_NNN.h5
```

The audited store contains 39 events × 32 members = 1,248 trajectory files,
plus `static_grid.h5` (1,249 HDF5 files total). Trajectories contain 109–365
hourly states. The HDF5 payload is 147,564,086,109 bytes (137.430 GiB); `du
-sh` reports 138G on NERSC CFS.

## Frozen split

The event-disjoint publication split is:

| Partition | Events | Members/event | Trajectories |
|---|---:|---:|---:|
| Train | 27 | 32 | 864 |
| Validation | 6 | 32 | 192 |
| Test | 6 | 32 | 192 |
| Total | 39 | 32 | 1,248 |

`data/metadata/m032/shared/trajectory_split.csv` stores the canonical absolute
site paths. The publication trainer automatically preserves the portion from
`trajectories/` downward and resolves it beneath `HOUSTON_DATASET_ROOT` when
the store is relocated; the split file does not need to be rewritten. Do not
regenerate or reshuffle event/member assignments. Normalization in
`data/metadata/m032/shared/normalization.json` is train-only.

Absolute `/compyfs` locations found in `preparation_metadata.json` are legacy
provenance from data preparation, not current runtime paths.

## Integrity hashes

Canonical dataset control files:

```text
e06154cedeb399dd61e21cbcd92afd51a6f0f3b09e90461f754c63dd96489a99  dataset_manifest.csv
9fdff14d42e6f9145ec2c29ac57be410dd4c9460818141cf81dc212b2e3f73cd  dataset_summary.json
6e79dd9e22be6afec5f26f39d77db5f53e0de17a4369661f6a0eefc740af9fdc  verification.json
6fe691205dcdb96633a9b7a71c37cc38be5a97d2ef24cfcb7416a2e02f9d9a81  seed_summary.json
4545bc8660af5a42579936a5b39614f5de5281a4b3e6e634b08db60af987b3a4  static_grid.h5
```

Publication split/normalization files in this repository:

```text
823b9810e95a7c4e7ebdf09c7161122b7be87a0b177a879cf6d6cd0e8ebbfa55  data/metadata/m032/shared/trajectory_split.csv
c51556ec145bef17a1e3c83a1c20a9e4c3024b64cd6750accbd699c69486bfc8  data/metadata/m032/shared/event_split.csv
2dc069fc41f9f3699133e2b4a18eb9d67ee36aaef7fe32d2b3f77bdd42387e9c  data/metadata/m032/shared/normalization.json
```

The listed control hashes identify the audited store; they are not a Merkle
digest of every 137-GiB trajectory. A public release must preserve the
numerical HDF5 content and publish an independently generated complete file
inventory/checksum list with the data archive.

## Public-release TODOs

- Select and add a repository/data license. No license is asserted yet.
- Deposit or otherwise publish the external trajectory store and assign its
  real DOI/archive identifier. No DOI is asserted yet.
- Add a complete per-HDF5 checksum inventory to that archive.
- Replace site paths only through the relocation rule above; never alter the
  frozen split or normalization statistics.
