# Canonical PINO-v2 evaluation source

This directory preserves the exact shared rollout adapter and canonical PINO
wrapper used to produce the retained PINO-ResU-Net-large evaluation. The live
adapters later changed, so evaluation metadata points here to keep the recorded
SHA-256 identities resolvable.

The source was recovered byte-for-byte from the NERSC CFS daily snapshot dated
2026-11-08. It is immutable historical provenance, not the active adapter.

The historical filenames contain the old `common13` and protocol-v2 labels.
They are preserved deliberately because the retained export metadata, hashes,
and wrapper import record those exact names. For clear public entry points, use
`analysis/evaluate_flood_forecasts.py` or
`workflows/evaluate_selected_pino.sbatch`; do not run files in this directory
as the general benchmark interface.
