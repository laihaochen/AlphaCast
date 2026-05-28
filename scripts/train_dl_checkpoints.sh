#!/usr/bin/env bash
# Train univariate deep learning checkpoints for AlphaCast baseline prediction.
#
# This script clones the THUML/Time-Series-Library and trains each DL model
# (Autoformer, DLinear, PatchTST, TimesNet, iTransformer) on every configured
# dataset with univariate settings (features=S, enc_in=1).
#
# Requirements: conda, CUDA-capable GPU, ~20 GB disk per dataset-model pair.
#
# Usage:
#   bash scripts/train_dl_checkpoints.sh          # Train all models on all datasets
#   bash scripts/train_dl_checkpoints.sh ETTh1    # Train all models on ETTh1 only
#   bash scripts/train_dl_checkpoints.sh ETTh1 Autoformer  # Single model+dataset

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
THUML_DIR="$ROOT_DIR/.thuml/Time-Series-Library"
CHECKPOINTS_DIR="$ROOT_DIR/checkpoints"

# ---------------------------------------------------------------------------
# Dataset → THUML dataset mapping
# ---------------------------------------------------------------------------
# Each entry: "dataset_name|root_path|data_path|data_name|freq|target_col|source_csv"
# - dataset_name: AlphaCast dataset name (must match config.yaml)
# - root_path:    THUML dataset root dir (relative to THUML repo root)
# - data_path:    CSV filename within root_path
# - data_name:    THUML --data argument ('custom' for generic CSV loader)
# - freq:         time feature frequency ('h'=hourly, 't'=minutely, 'd'=daily)
# - target_col:   target column name (required when data_name=custom)
# - source_csv:   path to *_train.csv relative to AlphaCast ROOT_DIR
# ---------------------------------------------------------------------------
DATASETS=(
  # ETT — built-in ETT data loader
  "ETTh1|./dataset/ETT-small/|ETTh1.csv|ETTh1|h||dataset/ETT/ETTh1/ETT_ETTh1_train.csv"
  "ETTh2|./dataset/ETT-small/|ETTh2.csv|custom|h|OT|dataset/ETT/ETTh2/ETT_ETTh2_train.csv"
  "ETTm1|./dataset/ETT-small/|ETTm1.csv|ETTm1|t||dataset/ETT/ETTm1/ETT_ETTm1_train.csv"
  "ETTm2|./dataset/ETT-small/|ETTm2.csv|custom|t|OT|dataset/ETT/ETTm2/ETT_ETTm2_train.csv"
  # EPF — custom CSV loader
  "EPF_BE|./dataset/EPF/|EPF_BE.csv|custom|h|real_power|dataset/EPF/BE/EPF_BE_train.csv"
  "EPF_DE|./dataset/EPF/|EPF_DE.csv|custom|h|real_power|dataset/EPF/DE/EPF_DE_train.csv"
  "EPF_FR|./dataset/EPF/|EPF_FR.csv|custom|h|real_power|dataset/EPF/FR/EPF_FR_train.csv"
  "EPF_NP|./dataset/EPF/|EPF_NP.csv|custom|h|real_power|dataset/EPF/NP/EPF_NP_train.csv"
  "EPF_PJM|./dataset/EPF/|EPF_PJM.csv|custom|h|real_power|dataset/EPF/PJM/EPF_PJM_train.csv"
  # BS — bike sharing, hourly
  "BS|./dataset/BS/|BS.csv|custom|h|cnt|dataset/BS/BS_train.csv"
  # MOPEX — daily streamflow
  "MOPEX|./dataset/MOPEX/|mopex.csv|custom|d|daily streamflow discharge|dataset/MOPEX/mopex_train.csv"
  # POWER — wind & solar, 15-minute
  "POWER_Windy|./dataset/POWER/|windy_power.csv|custom|t|real_power|dataset/POWER_NEW/windy_power/windy_power_train.csv"
  "POWER_Sunny|./dataset/POWER/|sunny_power.csv|custom|t|real_power|dataset/POWER_NEW/sunny_power/sunny_power_train.csv"
)

# All 5 DL models used by AlphaCast
MODELS=("Autoformer" "DLinear" "PatchTST" "TimesNet" "iTransformer")

# Shared training hyperparameters
SEQ_LEN=96
LABEL_LEN=48
PRED_LEN=96

# ---------------------------------------------------------------------------
# Model-specific hyperparameters
# TimesNet's Inception_Block_V1 uses 2D convolutions that scale quadratically
# with d_model*d_ff*kernel^2, so it needs much smaller values than other models.
# Using d_model=512/d_ff=2048 on TimesNet produces ~1.2B params → immediate OOM.
# ---------------------------------------------------------------------------
get_model_params() {
  local model="$1"
  case "$model" in
    TimesNet)
      echo "16|32|8|2"      # d_model=16, d_ff=32, n_heads=8, e_layers=2
      ;;
    PatchTST)
      echo "512|2048|2|1"   # d_model=512, d_ff=2048, n_heads=2, e_layers=1
      ;;
    iTransformer)
      echo "128|128|8|2"    # d_model=128, d_ff=128, n_heads=8, e_layers=2
      ;;
    *)
      echo "512|2048|8|2"   # Default for Autoformer/DLinear
      ;;
  esac
}

# ---------------------------------------------------------------------------
# Parse optional filters
# ---------------------------------------------------------------------------
FILTER_DATASET="${1:-}"
FILTER_MODEL="${2:-}"

# ---------------------------------------------------------------------------
# Clone THUML library if not present
# ---------------------------------------------------------------------------
setup_thuml() {
  if [ ! -d "$THUML_DIR" ]; then
    echo "=== Cloning THUML/Time-Series-Library ==="
    mkdir -p "$(dirname "$THUML_DIR")"
    git clone --depth 1 https://github.com/thuml/Time-Series-Library.git "$THUML_DIR"
  fi

  echo "=== Installing THUML dependencies (AlphaCast conda env) ==="
  conda run -n AlphaCast pip install numpy scipy scikit-learn pandas matplotlib sktime sympy \
    PyWavelets datasets einops tqdm patool 2>&1 | tail -3
}

# ---------------------------------------------------------------------------
# Prepare dataset symlinks
# ---------------------------------------------------------------------------
prepare_datasets() {
  echo "=== Preparing datasets ==="
  local thuml_data="$THUML_DIR/dataset"

  for entry in "${DATASETS[@]}"; do
    IFS='|' read -r ds_name root_path data_path _data_name _freq _target_col source_csv <<<"$entry"

    # Strip leading "./dataset/" from root_path to get the subdir inside THUML's dataset/
    local subdir="${root_path#./dataset/}"
    local dst_dir="$thuml_data/$subdir"
    mkdir -p "$dst_dir"
    local dst="$dst_dir/$data_path"

    if [ -f "$dst" ]; then
      echo "  Dataset $ds_name already exists at $dst"
      continue
    fi

    if [ -n "${source_csv:-}" ] && [ -f "$ROOT_DIR/$source_csv" ]; then
      # Use *_train.csv as the training data file for THUML.
      # THUML handles internal train/val split; AlphaCast evaluates on *_test.csv separately.
      python3 - "$ROOT_DIR/$source_csv" "$dst" "$ds_name" <<'PYEOF'
import sys
import pandas as pd

src_path, dst_path, ds_name = sys.argv[1], sys.argv[2], sys.argv[3]
df = pd.read_csv(src_path)

# Ensure a 'date' column exists (THUML requires it)
time_cols = ["date", "time_stamp", "timestamp", "time"]
time_col = next((c for c in time_cols if c in df.columns), None)
if time_col and time_col != "date":
    df.rename(columns={time_col: "date"}, inplace=True)

df.to_csv(dst_path, index=False)
print(f"  Prepared {ds_name}: {src_path} → {dst_path} ({len(df)} rows)")
PYEOF
    else
      echo "  [warn] Source data not found for $ds_name (source_csv=$source_csv). Skipping."
    fi
  done
}

# ---------------------------------------------------------------------------
# Train a single model on a single dataset
# ---------------------------------------------------------------------------
train_one() {
  local ds_name="$1"
  local model="$2"
  local root_path data_path data_name freq target_col

  for entry in "${DATASETS[@]}"; do
    IFS='|' read -r _ds _root _path _data _freq _target _src <<<"$entry"
    if [ "$_ds" = "$ds_name" ]; then
      root_path="$_root"
      data_path="$_path"
      data_name="$_data"
      freq="$_freq"
      target_col="${_target:-OT}"
      break
    fi
  done

  if [ -z "${root_path:-}" ]; then
    echo "[warn] Unknown dataset '$ds_name'. Skipping."
    return
  fi

  local model_id="${data_name}_${SEQ_LEN}_${PRED_LEN}"
  local out_dir="$CHECKPOINTS_DIR/$ds_name"
  mkdir -p "$out_dir"

  local ckpt_path="$out_dir/${model}_checkpoint.pth"

  if [ -f "$ckpt_path" ]; then
    echo "  [$ds_name/$model] Checkpoint already exists at $ckpt_path. Skipping training."
    return
  fi

  # Model-specific hyperparameters
  local d_model d_ff n_heads e_layers
  IFS='|' read -r d_model d_ff n_heads e_layers <<<"$(get_model_params "$model")"

  echo "  [$ds_name/$model] Training univariate checkpoint (model_id=$model_id, d_model=$d_model, d_ff=$d_ff, n_heads=$n_heads, e_layers=$e_layers) ..."

  pushd "$THUML_DIR" > /dev/null

  conda run -n AlphaCast python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path "$root_path" \
    --data_path "$data_path" \
    --model_id "$model_id" \
    --model "$model" \
    --data "$data_name" \
    --features S \
    --target "$target_col" \
    --seq_len "$SEQ_LEN" \
    --label_len "$LABEL_LEN" \
    --pred_len "$PRED_LEN" \
    --enc_in 1 \
    --dec_in 1 \
    --c_out 1 \
    --e_layers "$e_layers" \
    --d_layers 1 \
    --d_model "$d_model" \
    --d_ff "$d_ff" \
    --factor 3 \
    --n_heads "$n_heads" \
    --des 'AlphaCast' \
    --itr 1 \
    --train_epochs 100 \
    --batch_size 32 \
    --patience 10 \
    --learning_rate 0.0001 \
    --freq "$freq"

  popd > /dev/null

  # Copy checkpoint from THUML output to AlphaCast checkpoints directory.
  # THUML generates a directory name with many details; use find to locate it.
  local thuml_ckpt_file
  thuml_ckpt_file=$(find "$THUML_DIR/checkpoints" -path "*${model_id}*${model}*AlphaCast*" -name "checkpoint.pth" -print -quit 2>/dev/null)

  if [ -n "$thuml_ckpt_file" ] && [ -f "$thuml_ckpt_file" ]; then
    cp "$thuml_ckpt_file" "$ckpt_path"
    echo "  [$ds_name/$model] Checkpoint saved to $ckpt_path"
  else
    echo "  [warn] [$ds_name/$model] Training completed but checkpoint not found."
    echo "         Please locate it under $THUML_DIR/checkpoints/ and copy to: $ckpt_path"
  fi
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
main() {
  echo "=== AlphaCast DL Checkpoint Training ==="
  echo "Models: ${MODELS[*]}"
  echo "Datasets:"
  for entry in "${DATASETS[@]}"; do
    IFS='|' read -r ds_name _ <<<"$entry"
    echo "  - $ds_name"
  done
  echo ""

  setup_thuml
  prepare_datasets

  local total=0
  local skipped=0
  local failed=0

  for entry in "${DATASETS[@]}"; do
    IFS='|' read -r ds_name _ <<<"$entry"

    if [ -n "$FILTER_DATASET" ] && [ "$ds_name" != "$FILTER_DATASET" ]; then
      continue
    fi

    for model in "${MODELS[@]}"; do
      if [ -n "$FILTER_MODEL" ] && [ "$model" != "$FILTER_MODEL" ]; then
        continue
      fi

      total=$((total + 1))

      local ckpt_path="$CHECKPOINTS_DIR/$ds_name/${model}_checkpoint.pth"
      if [ -f "$ckpt_path" ]; then
        skipped=$((skipped + 1))
        echo "[$ds_name/$model] Checkpoint exists. Skipped."
        continue
      fi

      if train_one "$ds_name" "$model"; then
        echo "[$ds_name/$model] Done."
      else
        failed=$((failed + 1))
        echo "[warn] [$ds_name/$model] Training failed."
      fi
    done
  done

  echo ""
  echo "=== Summary ==="
  echo "Total: $total | Skipped (already exist): $skipped | Failed: $failed"
  if [ $failed -gt 0 ]; then
    echo "[warn] Some trainings failed. Check logs above for details."
  fi
}

main
