#!/usr/bin/env bash
# run_parallel.sh — Split datasets across multiple GPUs for faster experiments.
#
# Usage:
#   bash scripts/run_parallel.sh llm                              # LLM mode, all datasets, auto-detect free GPUs
#   bash scripts/run_parallel.sh llm --gpus 3,7                   # LLM mode, pin to GPU 3 and 7
#   bash scripts/run_parallel.sh baseline --gpus 3,7              # Deterministic mode
#   bash scripts/run_parallel.sh llm --gpus 3,7 EPF_BE ETTh1 ...  # Custom P1 datasets (rest go to P2)
#
#   CONFIG=other.yaml bash scripts/run_parallel.sh llm --gpus 3,7
#
# Each process gets CUDA_VISIBLE_DEVICES pinned to one GPU. DL models use cuda:0
# which maps to the assigned GPU, preventing cross-GPU memory contention.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
MODE=""
GPUS=""
CUSTOM_DATASETS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    llm|agent|baseline|deterministic)
      MODE="$1"
      shift
      ;;
    --gpus)
      GPUS="$2"
      shift 2
      ;;
    --gpus=*)
      GPUS="${1#*=}"
      shift
      ;;
    *)
      CUSTOM_DATASETS+=("$1")
      shift
      ;;
  esac
done

MODE="${MODE:-llm}"
CONFIG="${CONFIG:-config.yaml}"

# ---------------------------------------------------------------------------
# Validate mode
# ---------------------------------------------------------------------------
case "$MODE" in
  baseline|deterministic)
    MODE="deterministic"
    MODE_LABEL="Baseline (statistical + DL models)"
    ;;
  llm|agent)
    MODE="llm"
    MODE_LABEL="LLM orchestration"
    ;;
  *)
    echo "Usage: $0 [llm|baseline] [--gpus 3,7] [dataset1 dataset2 ...]"
    echo ""
    echo "  llm           LLM orchestration mode"
    echo "  baseline      Deterministic mode"
    echo "  --gpus 3,7    Comma-separated GPU IDs (default: auto-detect free GPUs)"
    echo ""
    echo "  Extra args become Process 1's datasets; the rest go to other GPUs."
    echo "  CONFIG        Path to YAML config (default: config.yaml)"
    exit 1
    ;;
esac

# ---------------------------------------------------------------------------
# Auto-detect free GPUs if not specified
# ---------------------------------------------------------------------------
if [ -z "$GPUS" ]; then
  if command -v nvidia-smi &>/dev/null; then
    # Find GPUs with < 500 MiB used
    FREE_GPUS=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null \
      | awk -F',' '{gsub(/ /,""); if ($2<500) print $1}' | paste -sd, - 2>/dev/null || echo "")
    if [ -z "$FREE_GPUS" ]; then
      # Fallback: use GPUs 0 and 1
      echo "[warn] No completely free GPUs found. Using GPU 0,1."
      GPUS="0,1"
    else
      # Take at most 2 free GPUs
      GPUS=$(echo "$FREE_GPUS" | cut -d, -f1,2)
      echo "[info] Auto-detected free GPUs: $GPUS"
    fi
  else
    GPUS="0,1"
    echo "[warn] nvidia-smi not found. Defaulting to GPU 0,1."
  fi
fi

# Convert GPU list to array
IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
NUM_GPUS=${#GPU_ARRAY[@]}

# ---------------------------------------------------------------------------
# Warn about existing processes
# ---------------------------------------------------------------------------
BASELINE_PROCS=$(ps aux | grep "run_experiment.py" | grep -v grep | grep "baseline_config" || true)
if [ -n "$BASELINE_PROCS" ]; then
  echo "[info] Baseline process detected (will NOT be disturbed):"
  echo "$BASELINE_PROCS" | awk '{print "       PID " $2 " (started " $9 ")"}' | head -3
fi

EXISTING_LLM=$(ps aux | grep "run_experiment.py.*config.yaml" | grep -v grep | grep -v baseline_config | grep -v "$$" || true)
if [ -n "$EXISTING_LLM" ]; then
  LLM_COUNT=$(echo "$EXISTING_LLM" | wc -l)
  echo "[warn] $LLM_COUNT other LLM process(es) already running:"
  echo "$EXISTING_LLM" | awk '{print "       PID " $2 " args: " $11, $12, $13, $14, $15}' | head -10
  echo "[warn] Running another parallel batch may cause dataset overlap and file corruption."
  echo "       Press Ctrl-C to cancel, or wait 5s to continue anyway..."
  sleep 5
fi

# ---------------------------------------------------------------------------
# Determine dataset split
# ---------------------------------------------------------------------------
ALL_DATASETS=(
  EPF_BE
  EPF_DE
  EPF_FR
  EPF_NP
  EPF_PJM
  ETTh1
  ETTm1
  WP
  SP
  MOPEX
)

# Build per-GPU dataset groups
declare -a GPU_DATASETS=()
for ((i=0; i<NUM_GPUS; i++)); do
  GPU_DATASETS[$i]=""
done

if [ ${#CUSTOM_DATASETS[@]} -gt 0 ]; then
  # User specified datasets — ONLY run those, split round-robin across all GPUs
  for ((i=0; i<${#CUSTOM_DATASETS[@]}; i++)); do
    gpu_idx=$((i % NUM_GPUS))
    if [ -n "${GPU_DATASETS[$gpu_idx]}" ]; then
      GPU_DATASETS[$gpu_idx]="${GPU_DATASETS[$gpu_idx]} ${CUSTOM_DATASETS[$i]}"
    else
      GPU_DATASETS[$gpu_idx]="${CUSTOM_DATASETS[$i]}"
    fi
  done
else
  # Default: distribute round-robin across all GPUs
  for ((i=0; i<${#ALL_DATASETS[@]}; i++)); do
    gpu_idx=$((i % NUM_GPUS))
    if [ -n "${GPU_DATASETS[$gpu_idx]}" ]; then
      GPU_DATASETS[$gpu_idx]="${GPU_DATASETS[$gpu_idx]} ${ALL_DATASETS[$i]}"
    else
      GPU_DATASETS[$gpu_idx]="${ALL_DATASETS[$i]}"
    fi
  done
fi

# ---------------------------------------------------------------------------
# Print header
# ---------------------------------------------------------------------------
echo "============================================"
echo " AlphaCast Parallel Runner"
echo "============================================"
echo " Mode:   $MODE ($MODE_LABEL)"
echo " Config: $CONFIG"
echo " GPUs:   $GPUS ($NUM_GPUS device(s))"
echo ""
for ((i=0; i<NUM_GPUS; i++)); do
  read -ra DS_ARRAY <<< "${GPU_DATASETS[$i]}"
  echo " GPU ${GPU_ARRAY[$i]} (${#DS_ARRAY[@]} datasets): ${DS_ARRAY[*]}"
done
echo ""
echo " Output dir: (from $CONFIG)"
echo "============================================"
echo ""

# ---------------------------------------------------------------------------
# Launch processes
# ---------------------------------------------------------------------------
declare -a PIDS=()
declare -a LOGS=()
declare -a GPU_LABELS=()

for ((i=0; i<NUM_GPUS; i++)); do
  LOG_FILE="/tmp/alphacast_gpu${GPU_ARRAY[$i]}_$$.log"
  LOGS+=("$LOG_FILE")
  GPU_LABELS+=("GPU${GPU_ARRAY[$i]}")
done

cleanup() {
  for log in "${LOGS[@]}"; do
    rm -f "$log"
  done
}
trap cleanup EXIT

for ((i=0; i<NUM_GPUS; i++)); do
  gpu_id="${GPU_ARRAY[$i]}"
  ds_list="${GPU_DATASETS[$i]}"
  log="${LOGS[$i]}"
  label="${GPU_LABELS[$i]}"

  if [ -z "$ds_list" ]; then
    echo "[info] GPU $gpu_id: no datasets assigned, skipping."
    PIDS+=(0)
    continue
  fi

  # Build --dataset flags
  DS_FLAGS=""
  for ds in $ds_list; do
    DS_FLAGS="$DS_FLAGS --dataset $ds"
  done

  echo "[info] GPU $gpu_id: starting process for: $ds_list"

  env CUDA_VISIBLE_DEVICES="$gpu_id" \
    conda run -n AlphaCast --no-capture-output \
      env ORCHESTRATION_MODE="$MODE" \
        python -u run_experiment.py --config "$CONFIG" $DS_FLAGS \
    > "$log" 2>&1 &

  PIDS+=($!)
  echo "       PID: ${PIDS[$i]} | CUDA_VISIBLE_DEVICES=$gpu_id | Log: $log"
done

echo ""
echo "[info] All processes launched. Waiting for completion..."
echo ""

# ---------------------------------------------------------------------------
# Live tail with GPU labels
# ---------------------------------------------------------------------------
{
  declare -a TAIL_PIDS=()
  for ((i=0; i<NUM_GPUS; i++)); do
    if [ "${PIDS[$i]}" -eq 0 ]; then
      TAIL_PIDS+=(0)
      continue
    fi
    tail -n +0 -f "${LOGS[$i]}" 2>/dev/null | sed "s/^/[${GPU_LABELS[$i]}] /" &
    TAIL_PIDS+=($!)
  done

  # Wait for all main processes
  declare -a EXIT_CODES=()
  all_ok=true
  for ((i=0; i<NUM_GPUS; i++)); do
    if [ "${PIDS[$i]}" -eq 0 ]; then
      EXIT_CODES+=(0)
      continue
    fi
    ec=0
    wait "${PIDS[$i]}" || ec=$?
    EXIT_CODES+=($ec)
    if [ $ec -ne 0 ]; then
      all_ok=false
    fi
  done

  # Give tails a moment to flush
  sleep 1
  for tp in "${TAIL_PIDS[@]}"; do
    [ "$tp" -ne 0 ] && kill "$tp" 2>/dev/null || true
  done
  for tp in "${TAIL_PIDS[@]}"; do
    [ "$tp" -ne 0 ] && wait "$tp" 2>/dev/null || true
  done
}

echo ""
echo "============================================"
echo " All processes finished."
echo "============================================"

# ---------------------------------------------------------------------------
# Print per-GPU summaries
# ---------------------------------------------------------------------------
for ((i=0; i<NUM_GPUS; i++)); do
  echo ""
  echo "--- ${GPU_LABELS[$i]} Summary (exit=${EXIT_CODES[$i]}) ---"
  grep -A 20 "=== Experiment Summary ===" "${LOGS[$i]}" 2>/dev/null || echo "(no summary found)"
done

if ! $all_ok; then
  echo ""
  echo "[warn] One or more processes exited with errors."
  for ((i=0; i<NUM_GPUS; i++)); do
    if [ "${EXIT_CODES[$i]}" -ne 0 ]; then
      echo "       ${GPU_LABELS[$i]} exit code: ${EXIT_CODES[$i]}  (log: ${LOGS[$i]})"
    fi
  done
  exit 1
fi

echo ""
echo "[done] All datasets complete."
