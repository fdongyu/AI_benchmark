#!/bin/bash
# Submit the guarded m032 depth4/plateau preflight and training arrays.

set -euo pipefail

DRY_RUN=false
if [[ "${1:-}" == --dry-run ]]; then
  DRY_RUN=true
  shift
fi
if (( $# != 0 )); then
  echo "Usage: $0 [--dry-run]" >&2
  exit 2
fi

WORKFLOW=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
ROOT=$(readlink -f -- "$WORKFLOW/../..")
ENV_FILE="$WORKFLOW/runtime_env.sh"
PREFLIGHT_SCRIPT="$WORKFLOW/preflight.sbatch"
TRAIN_SCRIPT="$WORKFLOW/train.sbatch"
PYTHON=/global/common/software/m3780/feng779/conda/rdycore-ai-gpu-py39/bin/python

# This directory records a completed experiment. Copy the workflow to a new,
# versioned directory before submitting changed code or another model. The
# name-based guard remains false in that copy.
if [[ "$(basename -- "$WORKFLOW")" == m032_benchmark_training ]]; then
  echo "m032_benchmark_training is a completed, read-only experiment." >&2
  echo "Copy this workflow to a newly named version and assign new IDs." >&2
  exit 78
fi

if [[ "$(readlink -f -- "$ROOT")" != "$ROOT" ]]; then
  echo "Benchmark root must be canonical: $ROOT" >&2
  exit 2
fi
for required in \
  "$ENV_FILE" \
  "$PREFLIGHT_SCRIPT" \
  "$TRAIN_SCRIPT" \
  "$WORKFLOW/campaign_manifest.json" \
  "$WORKFLOW/model_registry.json" \
  "$WORKFLOW/resolve_model.py" \
  "$WORKFLOW/next_stage.py" \
  "$WORKFLOW/run_trainer_with_io_status.py" \
  "$WORKFLOW/validate_stage.py" \
  "$WORKFLOW/write_submission_receipt.py" \
  "$ROOT/scripts/sync_experiment_manifest.py" \
  "$ROOT/src/train_benchmark_model.py" \
  "$ROOT/src/model_connector.py" \
  "$ROOT/src/model_list.py"
do
  [[ -f "$required" ]] || { echo "Missing campaign file: $required" >&2; exit 2; }
done

# These directories must exist before sbatch because Slurm opens stdout/stderr
# before the batch script can source runtime_env.sh.
mkdir -p -- \
  "$WORKFLOW/logs" \
  "$WORKFLOW/cache" \
  "$WORKFLOW/locks" \
  "$WORKFLOW/receipts" \
  "$ROOT/results/$(basename -- "$WORKFLOW")"

exec 9> "$WORKFLOW/locks/submission.lock"
if ! flock -n 9; then
  echo "Another m032 depth4 submitter holds the submission lock" >&2
  exit 73
fi

bash -n "$ENV_FILE" "$PREFLIGHT_SCRIPT" "$TRAIN_SCRIPT"
source "$ENV_FILE"
if [[ "$M032_DEPTH_WORKFLOW_ROOT" != "$WORKFLOW" ]]; then
  echo "runtime_env.sh resolved a different experiment directory" >&2
  exit 2
fi
if [[ "$(readlink -f -- "$AI_BENCHMARK_ROOT")" != "$ROOT" ]]; then
  echo "runtime_env.sh resolved a different benchmark root" >&2
  exit 2
fi
PYTHON="$PYTHON_EXE"
PYTHONPYCACHEPREFIX="$WORKFLOW/cache/pycache" "$PYTHON" -m py_compile \
  "$ROOT/src/train_benchmark_model.py" \
  "$ROOT/src/model_connector.py" \
  "$ROOT/src/model_list.py" \
  "$ROOT/model/model_new0/connector.py" \
  "$ROOT/scripts/check_model_connector.py" \
  "$ROOT/scripts/sync_experiment_manifest.py" \
  "$ROOT/src/vendor/baselines/unet_model.py" \
  "$ROOT/src/vendor/baselines/fno_model.py" \
  "$WORKFLOW/validate_setup.py" \
  "$WORKFLOW/resolve_model.py" \
  "$WORKFLOW/validate_stage.py" \
  "$WORKFLOW/next_stage.py" \
  "$WORKFLOW/run_trainer_with_io_status.py" \
  "$WORKFLOW/write_submission_receipt.py"
"$PYTHON" -B "$ROOT/scripts/sync_experiment_manifest.py" \
  --experiment-dir "$WORKFLOW" \
  --experiment-id "$AI_BENCHMARK_EXPERIMENT_ID" \
  --protocol-id "$AI_BENCHMARK_PROTOCOL_ID"
env \
  -u AI_BENCHMARK_EXPERIMENT_DIR \
  -u AI_BENCHMARK_EXPERIMENT_MANIFEST \
  -u AI_BENCHMARK_MODEL_LIST \
  -u AI_BENCHMARK_OUTPUT_ROOT \
  -u AI_BENCHMARK_EXPERIMENT_ID \
  -u AI_BENCHMARK_PROTOCOL_ID \
  "$PYTHON_EXE" -B -m pytest -q -p no:cacheprovider \
  "$ROOT/tests/test_benchmark_training.py"
MODEL_COUNT=$("$PYTHON" -B "$WORKFLOW/resolve_model.py" --field count)
if [[ ! "$MODEL_COUNT" =~ ^[1-9][0-9]*$ ]]; then
  echo "Invalid model count: $MODEL_COUNT" >&2
  exit 2
fi
LAST_TASK=$((MODEL_COUNT - 1))
ARRAY_SPEC="0-${LAST_TASK}%4"
for ((task_index = 0; task_index < MODEL_COUNT; task_index += 1)); do
  model=$(
    "$PYTHON" -B "$WORKFLOW/resolve_model.py" \
      --task-index "$task_index" --field model
  )
  has_connector=$(
    "$PYTHON" -B "$WORKFLOW/resolve_model.py" \
      --task-index "$task_index" --field has_connector
  )
  if [[ "$has_connector" == true ]]; then
    # A published model may require the fixed 256x512 Houston grid. The
    # dependent GPU preflight exercises that real shape before training.
    continue
  fi
  "$PYTHON_EXE" -B "$ROOT/src/train_benchmark_model.py" \
    --subset m032 --model "$model" --rollout-hours 4 --smoke >/dev/null
done

WORKFLOW_NAME=$(basename -- "$WORKFLOW")
PREFLIGHT_JOB_NAME="${WORKFLOW_NAME:0:72}-pre"
TRAIN_JOB_NAME="${WORKFLOW_NAME:0:70}-train"

if ! $DRY_RUN; then
  while IFS= read -r job_name; do
    case "$job_name" in
      "$PREFLIGHT_JOB_NAME"|"$TRAIN_JOB_NAME")
        echo "Refusing duplicate active campaign job: $job_name" >&2
        exit 75
        ;;
    esac
  done < <(squeue -h -u "${USER:?USER is required}" -o '%j')
fi

preflight_command=(
  sbatch
  --parsable
  --job-name="$PREFLIGHT_JOB_NAME"
  --array="$ARRAY_SPEC"
  --output="$WORKFLOW/logs/preflight.%A_%a.out"
  --error="$WORKFLOW/logs/preflight.%A_%a.err"
  --export="ALL,M032_DEPTH_ENV=$ENV_FILE"
  "$PREFLIGHT_SCRIPT"
)
printf -v preflight_text '%q ' "${preflight_command[@]}"

if $DRY_RUN; then
  printf '%s\n' "${preflight_text% }"
  printf '%s\n' \
    "sbatch --parsable --job-name=$TRAIN_JOB_NAME --array=$ARRAY_SPEC --output=$WORKFLOW/logs/train.%A_%a.out --error=$WORKFLOW/logs/train.%A_%a.err --dependency=afterok:<preflight_job_id> --kill-on-invalid-dep=yes --export=ALL,M032_DEPTH_ENV=$ENV_FILE $TRAIN_SCRIPT"
  exit 0
fi

preflight_raw=$("${preflight_command[@]}")
preflight_id=${preflight_raw%%;*}
if [[ ! "$preflight_id" =~ ^[0-9]+$ ]]; then
  echo "Could not parse preflight job ID: $preflight_raw" >&2
  exit 76
fi

training_command=(
  sbatch
  --parsable
  --job-name="$TRAIN_JOB_NAME"
  --array="$ARRAY_SPEC"
  --output="$WORKFLOW/logs/train.%A_%a.out"
  --error="$WORKFLOW/logs/train.%A_%a.err"
  --dependency="afterok:$preflight_id"
  --kill-on-invalid-dep=yes
  --export="ALL,M032_DEPTH_ENV=$ENV_FILE"
  "$TRAIN_SCRIPT"
)
printf -v training_text '%q ' "${training_command[@]}"
if ! training_raw=$("${training_command[@]}"); then
  echo "Preflight $preflight_id was accepted, but training submission failed" >&2
  exit 77
fi
training_id=${training_raw%%;*}
if [[ ! "$training_id" =~ ^[0-9]+$ ]]; then
  echo "Could not parse training job ID: $training_raw" >&2
  exit 76
fi

receipt=$(
  "$PYTHON" -B "$WORKFLOW/write_submission_receipt.py" \
    --preflight-job "$preflight_id" \
    --training-job "$training_id" \
    --preflight-command "${preflight_text% }" \
    --training-command "${training_text% }" \
    --array-spec "$ARRAY_SPEC" \
    --receipt-dir "$WORKFLOW/receipts"
)
printf 'preflight_job_id=%s\ntraining_job_id=%s\nreceipt=%s\n' \
  "$preflight_id" "$training_id" "$receipt"
squeue -j "$preflight_id,$training_id" -r \
  -o '%.18i %.2t %.10M %.12q %.20a %.32R'
