#!/bin/bash
#SBATCH --job-name=aib-d4m32-train
#SBATCH --account=m3780_g
#SBATCH --constraint=gpu&hbm40g
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --requeue
# The six-hour warning allows the current epoch to finish and commit atomically.
#SBATCH --signal=USR1@21600
#SBATCH --array=0-4%4
#SBATCH --open-mode=append
#SBATCH --output=/global/cfs/cdirs/m4267/dongyu/AI_benchmark/workflows/m032_benchmark_training/logs/train.%A_%a.out
#SBATCH --error=/global/cfs/cdirs/m4267/dongyu/AI_benchmark/workflows/m032_benchmark_training/logs/train.%A_%a.err

set -euo pipefail
source "${M032_DEPTH_ENV:?M032_DEPTH_ENV is required}"

SUBSET=m032
TRANSITIONS_PER_EPOCH=8192
VALIDATION_TRANSITIONS=2048
TASK_INDEX=${SLURM_ARRAY_TASK_ID:-${M032_DEPTH_TASK_INDEX:?array task is required}}
MODEL_RECORD=$(
  "$PYTHON_EXE" -B "$M032_DEPTH_WORKFLOW_ROOT/resolve_model.py" \
    --task-index "$TASK_INDEX" --field tsv
)
IFS=$'\t' read -r MODEL LEARNING_RATE ENDPOINT_LEARNING_RATE EPOCHS_PER_INVOCATION \
  <<< "$MODEL_RECORD"
if [[ -z "$MODEL" || -z "$LEARNING_RATE" || -z "$ENDPOINT_LEARNING_RATE" \
      || -z "$EPOCHS_PER_INVOCATION" ]]; then
  echo "Invalid model registry record for task $TASK_INDEX: $MODEL_RECORD" >&2
  exit 2
fi

if [[ "$NERSC_PROJECT" != m3780_g || "${NVIDIA_TF32_OVERRIDE:-}" != 0 ]]; then
  echo "The reproduction requires m3780_g and NVIDIA_TF32_OVERRIDE=0" >&2
  exit 2
fi
MAX_RESTARTS=$(( 4 * ((240 + EPOCHS_PER_INVOCATION - 1) / EPOCHS_PER_INVOCATION) + 40 ))
if (( MAX_RESTARTS < 520 )); then
  MAX_RESTARTS=520
fi
if (( ${SLURM_RESTART_COUNT:-0} > MAX_RESTARTS )); then
  echo "Refusing restart count ${SLURM_RESTART_COUNT}" >&2
  exit 87
fi

HOURS=$(
  "$PYTHON_EXE" -B "$M032_DEPTH_WORKFLOW_ROOT/next_stage.py" \
    --model "$MODEL" \
    --output-root "$M032_DEPTH_OUTPUT_ROOT"
)
if [[ "$HOURS" == complete ]]; then
  echo "$SUBSET $MODEL curriculum is already complete"
  exit 0
fi
case "$HOURS" in
  4|8|16|32) ;;
  *) echo "Invalid next-stage response: $HOURS" >&2; exit 2 ;;
esac

LOCK_PARENT="$M032_DEPTH_WORKFLOW_ROOT/locks/stages/$MODEL"
LOCK_DIR="$LOCK_PARENT/ar$(printf '%02d' "$HOURS")"
LOCK_OWNER="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}_${TASK_INDEX}"
mkdir -p -- "$LOCK_PARENT"
if mkdir -- "$LOCK_DIR" 2>/dev/null; then
  printf '%s\n' "$LOCK_OWNER" > "$LOCK_DIR/owner"
else
  EXISTING_OWNER=""
  [[ -r "$LOCK_DIR/owner" ]] && IFS= read -r EXISTING_OWNER < "$LOCK_DIR/owner"
  if [[ "$EXISTING_OWNER" != "$LOCK_OWNER" ]]; then
    if [[ ! "$EXISTING_OWNER" =~ ^[0-9]+_[0-9]+$ ]]; then
      echo "$SUBSET $MODEL AR$HOURS has an invalid lock owner: ${EXISTING_OWNER:-unknown}" >&2
      exit 74
    fi
    if ! owner_queue_state=$(squeue -h -j "$EXISTING_OWNER" -o '%i'); then
      echo "Cannot verify whether lock owner $EXISTING_OWNER is still active" >&2
      exit 74
    fi
    lock_age=$(( $(date +%s) - $(stat -c %Y "$LOCK_DIR/owner") ))
    if [[ -n "$owner_queue_state" || "$lock_age" -lt 300 ]]; then
      echo "$SUBSET $MODEL AR$HOURS is locked by active/recent job $EXISTING_OWNER" >&2
      exit 74
    fi
    echo "Reclaiming ${lock_age}s-old lock from inactive job $EXISTING_OWNER"
    rm -f -- "$LOCK_DIR/owner"
    rmdir -- "$LOCK_DIR"
    mkdir -- "$LOCK_DIR"
    printf '%s\n' "$LOCK_OWNER" > "$LOCK_DIR/owner"
  else
    echo "$SUBSET $MODEL AR$HOURS recovering lock owned by this requeued task"
  fi
fi

release_stage_lock() {
  local owner=""
  [[ -r "$LOCK_DIR/owner" ]] && IFS= read -r owner < "$LOCK_DIR/owner"
  if [[ "$owner" == "$LOCK_OWNER" ]]; then
    rm -f -- "$LOCK_DIR/owner"
    rmdir -- "$LOCK_DIR" 2>/dev/null || true
  fi
}
trap release_stage_lock EXIT TERM INT

requeue_current_task() {
  local reason=$1
  if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "$reason, but no Slurm job exists to requeue" >&2
    return 85
  fi
  local target=$SLURM_JOB_ID
  if [[ -n "${SLURM_ARRAY_JOB_ID:-}" && -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    target="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
  fi
  echo "$reason; requeueing $target"
  release_stage_lock
  trap - EXIT TERM INT
  if ! scontrol requeue "$target"; then
    echo "$reason; failed to requeue $target" >&2
    return 86
  fi
  echo "$reason; requeue accepted"
  sleep 10
}

INVOCATION_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}.${TASK_INDEX}.${SLURM_RESTART_COUNT:-0}.$(date +%s).${RANDOM}"
args=(
  "$M032_DEPTH_TRAINER"
  --subset "$SUBSET"
  --model "$MODEL"
  --rollout-hours "$HOURS"
  --dataset-root "$DATASET_ROOT"
  --shared-artifact-dir "$SHARED_ARTIFACT_DIR"
  --channel-artifact-dir "$CHANNEL_ARTIFACT_DIR"
  --output-root "$M032_DEPTH_OUTPUT_ROOT"
  --campaign-manifest "$M032_DEPTH_WORKFLOW_ROOT/campaign_manifest.json"
  --epochs-per-invocation "$EPOCHS_PER_INVOCATION"
  --transitions-per-epoch "$TRANSITIONS_PER_EPOCH"
  --validation-transitions "$VALIDATION_TRANSITIONS"
  --batch-size 1
  --num-workers 4
  --threads 16
  --detach-interval 4
  --seed 20260917
  --device cuda
  --mixed-precision off
  --no-deterministic
  --pin-memory
  --depth-weight 4
  --momentum-weight 1
  --supervised-scale 3
  --weight-decay 1e-5
  --learning-rate "$LEARNING_RATE"
  --endpoint-learning-rate "$ENDPOINT_LEARNING_RATE"
  --dry-tolerance-m 1e-4
  --max-velocity-m-s 10
  --invocation-id "$INVOCATION_ID"
)

CFS_RETRY_DELAYS=(5 20 60 120)
CFS_MAX_ATTEMPTS=5
cfs_attempt=1
while true; do
  set +e
  srun --ntasks=1 --gpus-per-task=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" \
    --cpu-bind=cores "$PYTHON_EXE" -B "$M032_DEPTH_IO_RUNNER" "${args[@]}"
  status=$?
  set -e
  if (( status != 89 || cfs_attempt >= CFS_MAX_ATTEMPTS )); then
    break
  fi
  cfs_delay=${CFS_RETRY_DELAYS[$((cfs_attempt - 1))]}
  echo "Transient CFS/HDF5 failure on attempt $cfs_attempt/$CFS_MAX_ATTEMPTS; retrying from the atomic checkpoint in ${cfs_delay}s" >&2
  sleep "$cfs_delay"
  ((cfs_attempt += 1))
done

if (( status == 85 )); then
  "$PYTHON_EXE" -B "$M032_DEPTH_WORKFLOW_ROOT/validate_stage.py" \
    --mode progress \
    --model "$MODEL" \
    --hours "$HOURS" \
    --output-root "$M032_DEPTH_OUTPUT_ROOT" \
    --invocation-id "$INVOCATION_ID"
elif (( status == 89 )); then
  requeue_current_task "$SUBSET $MODEL AR$HOURS transient CFS/HDF5 recovery"
  exit $?
elif (( status != 0 )); then
  exit "$status"
else
  "$PYTHON_EXE" -B "$M032_DEPTH_WORKFLOW_ROOT/validate_stage.py" \
    --mode complete \
    --model "$MODEL" \
    --hours "$HOURS" \
    --output-root "$M032_DEPTH_OUTPUT_ROOT"
fi

NEXT_HOURS=$(
  "$PYTHON_EXE" -B "$M032_DEPTH_WORKFLOW_ROOT/next_stage.py" \
    --model "$MODEL" \
    --output-root "$M032_DEPTH_OUTPUT_ROOT"
)
if [[ "$NEXT_HOURS" == complete ]]; then
  echo "$SUBSET $MODEL completed AR4 -> AR8 -> AR16 -> AR32"
  release_stage_lock
  trap - EXIT TERM INT
  exit 0
fi
requeue_current_task "$SUBSET $MODEL next work unit AR$NEXT_HOURS"
exit $?
