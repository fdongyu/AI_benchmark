#!/bin/bash
#SBATCH --job-name=aib-d4m32-pre
#SBATCH --account=m3780_g
#SBATCH --constraint=gpu&hbm40g
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --array=0-4%4
#SBATCH --open-mode=append
#SBATCH --output=/global/cfs/cdirs/m4267/dongyu/AI_benchmark/workflows/m032_benchmark_training/logs/preflight.%A_%a.out
#SBATCH --error=/global/cfs/cdirs/m4267/dongyu/AI_benchmark/workflows/m032_benchmark_training/logs/preflight.%A_%a.err

set -euo pipefail
source "${M032_DEPTH_ENV:?M032_DEPTH_ENV is required}"

TASK_INDEX=${SLURM_ARRAY_TASK_ID:?array task is required}
MODEL=$(
  "$PYTHON_EXE" -B "$M032_DEPTH_WORKFLOW_ROOT/resolve_model.py" \
    --task-index "$TASK_INDEX" --field model
)

srun --cpu-bind=cores "$PYTHON_EXE" -B \
  "$M032_DEPTH_WORKFLOW_ROOT/validate_setup.py" \
  --model "$MODEL" \
  --device cuda
