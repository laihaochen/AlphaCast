#!/usr/bin/env bash
# run_baseline.sh — Run deterministic (baseline) experiments with results saved to
# baseline_outputs/ so they can be compared side-by-side with LLM results in outputs/.
#
# Usage:
#   bash scripts/run_baseline.sh                  # All active datasets
#   bash scripts/run_baseline.sh EPF_BE           # Single dataset
#   bash scripts/run_baseline.sh EPF_BE ETTh1      # Multiple datasets (one at a time)
#
# This script creates a temporary copy of config.yaml with output_dir set to
# baseline_outputs/, then invokes the main experiment runner.  The original
# config.yaml is never touched, so it is safe to interrupt (Ctrl-C) at any time.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

CONFIG_ORIG="$ROOT_DIR/config.yaml"
CONFIG_TMP="/tmp/baseline_config_$$.yaml"

# ---------------------------------------------------------------------------
# Cleanup helper — always remove the temporary config on exit
# ---------------------------------------------------------------------------
cleanup() {
  rm -f "$CONFIG_TMP"
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# Build temporary config with output_dir → baseline_outputs
# ---------------------------------------------------------------------------
sed 's/^output_dir:.*/output_dir: baseline_outputs/' "$CONFIG_ORIG" > "$CONFIG_TMP"

echo "============================================"
echo " Baseline Experiment Runner"
echo "============================================"
echo " Original config:  $CONFIG_ORIG"
echo " Temporary config: $CONFIG_TMP"
echo " Output dir:       $ROOT_DIR/baseline_outputs/"
echo "============================================"
echo ""

# ---------------------------------------------------------------------------
# Run baseline for each requested dataset (or all if none given)
# ---------------------------------------------------------------------------
if [ $# -eq 0 ]; then
  echo "[info] Running baseline for ALL active datasets..."
  echo ""
  CONFIG="$CONFIG_TMP" bash "$SCRIPT_DIR/run_experiments.sh" baseline
else
  for DATASET in "$@"; do
    echo "[info] Running baseline for: $DATASET"
    echo ""
    CONFIG="$CONFIG_TMP" bash "$SCRIPT_DIR/run_experiments.sh" baseline "$DATASET"
    echo ""
    echo "[done] Finished $DATASET"
    echo "------------------------------------------------------------"
    echo ""
  done
fi

echo ""
echo "============================================"
echo " All baseline runs complete."
echo " Results are in: $ROOT_DIR/baseline_outputs/"
echo " Compare against: $ROOT_DIR/outputs/  (LLM results)"
echo "============================================"
