#!/usr/bin/env bash
# Retrain all 5 deep learning checkpoints from scratch, parallelized across 2 GPUs.
#
# Usage:
#   bash retrain_dl_checkpoints.sh
#
# Configure the two GPU IDs below before running.

set -euo pipefail

# =============================================================================
# TODO: fill in the two GPU IDs you want to use
# =============================================================================
GPU0=4
GPU1=7

# Remove all existing DL checkpoints
echo "=== Removing old checkpoints ==="
rm -f checkpoints/*/Autoformer_checkpoint.pth
rm -f checkpoints/*/DLinear_checkpoint.pth
rm -f checkpoints/*/PatchTST_checkpoint.pth
rm -f checkpoints/*/TimesNet_checkpoint.pth
rm -f checkpoints/*/iTransformer_checkpoint.pth

# =============================================================================
# Model distribution (balanced by memory footprint):
#   GPU $GPU0: Autoformer (heavy, d_model=512), TimesNet (tiny, d_model=16), iTransformer (medium, d_model=128)
#   GPU $GPU1: PatchTST  (heavy, d_model=512), DLinear (light, linear-only)
# =============================================================================

echo "=== Launching GPU $GPU0 jobs ==="
CUDA_VISIBLE_DEVICES=$GPU0 bash scripts/train_dl_checkpoints.sh "" Autoformer &
PID0_1=$!
CUDA_VISIBLE_DEVICES=$GPU0 bash scripts/train_dl_checkpoints.sh "" TimesNet &
PID0_2=$!
CUDA_VISIBLE_DEVICES=$GPU0 bash scripts/train_dl_checkpoints.sh "" iTransformer &
PID0_3=$!

echo "=== Launching GPU $GPU1 jobs ==="
CUDA_VISIBLE_DEVICES=$GPU1 bash scripts/train_dl_checkpoints.sh "" PatchTST &
PID1_1=$!
CUDA_VISIBLE_DEVICES=$GPU1 bash scripts/train_dl_checkpoints.sh "" DLinear &
PID1_2=$!

echo ""
echo "=== All jobs launched. Waiting for completion... ==="
echo "GPU $GPU0: Autoformer (PID $PID0_1) | TimesNet (PID $PID0_2) | iTransformer (PID $PID0_3)"
echo "GPU $GPU1: PatchTST  (PID $PID1_1) | DLinear  (PID $PID1_2)"

wait $PID0_1 $PID0_2 $PID0_3 $PID1_1 $PID1_2

echo ""
echo "=== All 5 models retrained successfully ==="
