#!/bin/bash
#SBATCH --job-name=pino-resunet-large-eval
#SBATCH --account=m3780_g
#SBATCH --constraint=gpu&hbm40g
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --chdir=/global/cfs/cdirs/m4267/dongyu/AI_benchmark
#SBATCH --output=/global/cfs/cdirs/m4267/dongyu/AI_benchmark/workflows/logs/pino-resunet-large-evaluation.%j.out
#SBATCH --error=/global/cfs/cdirs/m4267/dongyu/AI_benchmark/workflows/logs/pino-resunet-large-evaluation.%j.err

set -euo pipefail

if type module >/dev/null 2>&1; then
  module unload pytorch >/dev/null 2>&1 || true
  module unload cudatoolkit >/dev/null 2>&1 || true
fi

unset PYTHONPATH
unset PYTHONUSERBASE
export PYTHONNOUSERSITE=1

readonly AI_BENCHMARK_ROOT=${AI_BENCHMARK_ROOT:-/global/cfs/cdirs/m4267/dongyu/AI_benchmark}
readonly PYTHON_EXE=${PYTHON_EXE:-/global/common/software/m3780/feng779/conda/rdycore-ai-gpu-py39/bin/python}
readonly STAGE="${AI_BENCHMARK_ROOT}/results/m032_benchmark/pino_resunet_large/stage_ar32"
readonly BEST_CHECKPOINT="${STAGE}/checkpoint_best_validation_supervised.pt"
readonly EXPORT_DIR=${PINO_RESUNET_LARGE_EVAL_OUTPUT_DIR:-"${AI_BENCHMARK_ROOT}/data/evaluations/pino_resunet_large_m032_ar32"}
readonly EVALUATOR="${AI_BENCHMARK_ROOT}/model/pino_resunet_large/evaluate.py"
readonly MODEL_LOADER="${AI_BENCHMARK_ROOT}/model/pino_resunet_large/load_historical_checkpoint.py"
readonly PLOTTER="${AI_BENCHMARK_ROOT}/analysis/plot_flood_evaluation.py"
readonly GAUGES="${AI_BENCHMARK_ROOT}/data/metadata/validation/rdycore_usgs_gauge_cells.csv"
readonly RESTART_COUNT=${SLURM_RESTART_COUNT:-0}
readonly EXPORT_STAGE="${EXPORT_DIR}.${SLURM_JOB_ID:-manual}.${RESTART_COUNT}.tmp"

export PYTHONPATH="${AI_BENCHMARK_ROOT}/src:${AI_BENCHMARK_ROOT}/analysis:${AI_BENCHMARK_ROOT}/analysis/lib"
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=1
export NVIDIA_TF32_OVERRIDE=0
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONUNBUFFERED=1
export WANDB_DISABLED=true
export WANDB_MODE=disabled
export XDG_CACHE_HOME="${AI_BENCHMARK_ROOT}/workflows/cache"
export TORCH_HOME="${XDG_CACHE_HOME}/torch"
export MPLCONFIGDIR="${XDG_CACHE_HOME}/matplotlib"

mkdir -p -- \
  "${AI_BENCHMARK_ROOT}/workflows/logs" \
  "${TORCH_HOME}" \
  "${MPLCONFIGDIR}" \
  "$(dirname -- "${EXPORT_DIR}")"

for required in \
  "${PYTHON_EXE}" \
  "${BEST_CHECKPOINT}" \
  "${EVALUATOR}" \
  "${MODEL_LOADER}" \
  "${PLOTTER}" \
  "${GAUGES}"
do
  [[ -s "${required}" ]] || { echo "Missing required input: ${required}" >&2; exit 2; }
done

if [[ -e "${EXPORT_DIR}" ]]; then
  echo "Refusing to overwrite completed export: ${EXPORT_DIR}" >&2
  echo "Set PINO_RESUNET_LARGE_EVAL_OUTPUT_DIR to a new empty directory for a rerun." >&2
  exit 73
fi
if [[ -e "${EXPORT_STAGE}" ]]; then
  echo "Export staging path already exists: ${EXPORT_STAGE}" >&2
  exit 77
fi

srun --ntasks=1 --gpus-per-task=1 --cpus-per-task=32 --cpu-bind=cores \
  "${PYTHON_EXE}" -B "${EVALUATOR}" \
  --model-loader "${MODEL_LOADER}" \
  --checkpoint "${BEST_CHECKPOINT}" \
  --checkpoint-role best \
  --campaign-primary-checkpoint "${BEST_CHECKPOINT}" \
  --gauge-file "${GAUGES}" \
  --output-dir "${EXPORT_STAGE}" \
  --split test \
  --evaluation-members 32 \
  --snapshot-member 1 \
  --device cuda \
  --threads 16

srun --ntasks=1 --cpus-per-task=32 --cpu-bind=cores \
  "${PYTHON_EXE}" -B "${PLOTTER}" \
  --input-dir "${EXPORT_STAGE}" \
  --output-dir "${EXPORT_STAGE}/figures" \
  --model-label "PINO-ResU-Net-large (depth4 plateau v2)"

for product in \
  "${EXPORT_STAGE}/export_metadata.json" \
  "${EXPORT_STAGE}/domain_summary_metrics.csv" \
  "${EXPORT_STAGE}/gauge_summary_metrics.csv" \
  "${EXPORT_STAGE}/domain_metrics_by_trajectory.csv" \
  "${EXPORT_STAGE}/gauge_metrics_by_trajectory.csv" \
  "${EXPORT_STAGE}/spatial_error_summary.npz" \
  "${EXPORT_STAGE}/gauge_timeseries.csv.gz" \
  "${EXPORT_STAGE}/figures/event_hydrographs/AORC_2003111519_01422.hydrographs.png" \
  "${EXPORT_STAGE}/figures/peak_flood_maps/AORC_2003111519_01422__m001.peak_flood.png" \
  "${EXPORT_STAGE}/figures/spatial_error_maps/full_domain_spatial_errors.png"
do
  [[ -s "${product}" ]] || { echo "Missing completed export product: ${product}" >&2; exit 78; }
done

"${PYTHON_EXE}" - "${EXPORT_STAGE}/export_metadata.json" <<'PY'
import json
import sys
from pathlib import Path

metadata = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = {
    "complete_rollouts": 192,
    "test_events": 6,
    "evaluation_members_per_event": 32,
    "core_gauges": 14,
    "checkpoint_epoch": 52,
    "checkpoint_role": "best",
    "protocol_id": "houston-m032-common13-pino-depth4-plateau-v2",
}
mismatches = {
    key: (metadata.get(key), value)
    for key, value in expected.items()
    if metadata.get(key) != value
}
if mismatches:
    raise RuntimeError(f"evaluation metadata mismatch: {mismatches}")
PY

if [[ -e "${EXPORT_DIR}" ]]; then
  echo "Export destination appeared during execution: ${EXPORT_DIR}" >&2
  exit 78
fi
mv -- "${EXPORT_STAGE}" "${EXPORT_DIR}"
echo "Published PINO-ResU-Net-large rollout export: ${EXPORT_DIR}"
