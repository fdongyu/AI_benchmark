#!/bin/bash
# Locked runtime for the m032 multi-model depth4/plateau campaign.

set -euo pipefail

if type module >/dev/null 2>&1; then
  module unload pytorch >/dev/null 2>&1 || true
  module unload cudatoolkit >/dev/null 2>&1 || true
fi

unset PYTHONPATH
unset PYTHONUSERBASE
export PYTHONNOUSERSITE=1

export M032_DEPTH_WORKFLOW_ROOT=$(
  cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P
)
if [[ "$(basename -- "$M032_DEPTH_WORKFLOW_ROOT")" == m032_benchmark_training ]]; then
  echo "m032_benchmark_training is a completed, read-only experiment." >&2
  return 78 2>/dev/null || exit 78
fi
export NERSC_PROJECT=${NERSC_PROJECT:-m3780_g}
export PYTHON_EXE=${PYTHON_EXE:-/global/common/software/m3780/feng779/conda/rdycore-ai-gpu-py39/bin/python}
export AI_BENCHMARK_ROOT=${AI_BENCHMARK_ROOT:-$(readlink -f -- "$M032_DEPTH_WORKFLOW_ROOT/../..")}
export AI_FOUNDATION_ROOT=${AI_FOUNDATION_ROOT:-$AI_BENCHMARK_ROOT/../AI_foundation}
export M032_DEPTH_TRAINER="$AI_BENCHMARK_ROOT/src/train_benchmark_model.py"
export M032_DEPTH_IO_RUNNER="$M032_DEPTH_WORKFLOW_ROOT/run_trainer_with_io_status.py"
export M032_DEPTH_OUTPUT_ROOT="$AI_BENCHMARK_ROOT/results/m032_benchmark"
# These generic names make a copied, versioned experiment explicit. A future
# workflow changes these values instead of editing the common trainer.
export AI_BENCHMARK_EXPERIMENT_DIR="$M032_DEPTH_WORKFLOW_ROOT"
export AI_BENCHMARK_EXPERIMENT_MANIFEST="$M032_DEPTH_WORKFLOW_ROOT/campaign_manifest.json"
export AI_BENCHMARK_MODEL_LIST="$M032_DEPTH_WORKFLOW_ROOT/model_registry.json"
export AI_BENCHMARK_OUTPUT_ROOT="$M032_DEPTH_OUTPUT_ROOT"
export AI_BENCHMARK_EXPERIMENT_ID=houston-common13-m032-depth4-plateau-v1
export AI_BENCHMARK_PROTOCOL_ID=houston-m032-common13-depth4-plateau-v1
if [[ "$AI_BENCHMARK_EXPERIMENT_ID" == houston-common13-m032-depth4-plateau-v1 \
   || "$AI_BENCHMARK_PROTOCOL_ID" == houston-m032-common13-depth4-plateau-v1 ]]; then
  echo "A copied workflow must replace both completed v1 IDs before it can run." >&2
  return 78 2>/dev/null || exit 78
fi
export HOUSTON_DATASET_ROOT="${HOUSTON_DATASET_ROOT:-$AI_FOUNDATION_ROOT/DATA/rdycore_ml_dataset_core}"
export DATASET_ROOT="$HOUSTON_DATASET_ROOT"
export SHARED_ARTIFACT_DIR="$AI_BENCHMARK_ROOT/data/metadata/m032/shared"
export CHANNEL_ARTIFACT_DIR="$AI_BENCHMARK_ROOT/data/metadata/m032/channels"

export NVIDIA_TF32_OVERRIDE=0
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONUNBUFFERED=1
export WANDB_DISABLED=true
export WANDB_MODE=disabled

export XDG_CACHE_HOME="$M032_DEPTH_WORKFLOW_ROOT/cache"
export TORCH_HOME="$XDG_CACHE_HOME/torch"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
mkdir -p -- "$TORCH_HOME" "$MPLCONFIGDIR" \
  "$M032_DEPTH_WORKFLOW_ROOT/logs" "$M032_DEPTH_OUTPUT_ROOT"

for required in \
  "$PYTHON_EXE" \
  "$M032_DEPTH_TRAINER" \
  "$AI_BENCHMARK_ROOT/src/model_connector.py" \
  "$AI_BENCHMARK_ROOT/src/model_list.py" \
  "$M032_DEPTH_IO_RUNNER" \
  "$DATASET_ROOT/static_grid.h5" \
  "$SHARED_ARTIFACT_DIR/trajectory_split.csv" \
  "$SHARED_ARTIFACT_DIR/normalization.json" \
  "$CHANNEL_ARTIFACT_DIR/channel_masks_fullres.npz" \
  "$AI_BENCHMARK_ROOT/src/vendor/baselines/unet_model.py" \
  "$AI_BENCHMARK_ROOT/src/vendor/baselines/fno_model.py" \
  "$M032_DEPTH_WORKFLOW_ROOT/campaign_manifest.json" \
  "$M032_DEPTH_WORKFLOW_ROOT/model_registry.json"
do
  [[ -e "$required" ]] || { echo "Missing campaign input: $required" >&2; exit 2; }
done
