#!/usr/bin/env bash
# AlphaCast experiment runner — baseline (deterministic) and LLM modes.
#
# Usage:
#   bash scripts/run_experiments.sh                           # Baseline mode, all datasets
#   bash scripts/run_experiments.sh baseline                  # Same as above
#   bash scripts/run_experiments.sh baseline EPF_BE           # Baseline, single dataset
#   bash scripts/run_experiments.sh llm                       # LLM mode, all datasets
#   bash scripts/run_experiments.sh llm ETTh1                 # LLM mode, single dataset
#
# Configuration is read from config.yaml by default.
# Override with CONFIG=config.yaml (or another YAML path).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

MODE="${1:-baseline}"
DATASET_ARG="${2:-}"
CONFIG="${CONFIG:-config.yaml}"

# ---------------------------------------------------------------------------
# Validate mode
# ---------------------------------------------------------------------------
case "$MODE" in
  baseline|deterministic)
    MODE="deterministic"
    MODE_LABEL="Baseline (statistical + DL models, no LLM)"
    ;;
  llm|agent)
    MODE="llm"
    MODE_LABEL="LLM orchestration (requires API key in .env)"
    ;;
  *)
    echo "Usage: $0 [baseline|llm] [dataset_name_or_alias]"
    echo ""
    echo "  baseline    Deterministic mode: sliding-window evaluation using statistical + DL models"
    echo "  llm         LLM orchestration mode: LLM agent calls tools and emits predictions"
    echo ""
    echo "Environment variables:"
    echo "  CONFIG      Path to YAML config (default: config.yaml)"
    echo ""
    exit 1
    ;;
esac

# ---------------------------------------------------------------------------
# Build command
# ---------------------------------------------------------------------------
cd "$ROOT_DIR"

CMD=(
  conda run -n AlphaCast --no-capture-output
  env ORCHESTRATION_MODE="$MODE"
  python -u run_experiment.py
  --config "$CONFIG"
)

if [ -n "$DATASET_ARG" ]; then
  CMD+=(--dataset "$DATASET_ARG")
fi

echo "============================================"
echo " AlphaCast Experiment Runner"
echo "============================================"
echo " Mode:      $MODE ($MODE_LABEL)"
echo " Config:    $CONFIG"
echo " Dataset:   ${DATASET_ARG:-all}"
echo " Output:    (see config YAML 'output_dir' field)"
echo "============================================"
echo ""

if [ "$MODE" = "llm" ]; then
  if [ ! -f .env ]; then
    echo "[warn] .env file not found. LLM mode requires API keys."
    echo "       Create .env with your API key, e.g.:"
    echo "         DEEPSEEK_API_KEY=sk-..."
    echo ""
  fi
  echo "[info] This mode calls the LLM API per sliding window. Cost may be significant."
  echo "[info] Progress is automatically saved for resume on re-run."
  echo ""
fi

echo "[info] Running: ${CMD[*]}"
echo ""

"${CMD[@]}"
