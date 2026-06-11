#!/usr/bin/env bash
# Train deep learning checkpoints for EPF datasets with 168-lookback / 24-prediction.
#
# The existing checkpoints/ directory holds models trained with seq_len=96/pred_len=96
# (for long-term datasets: ETT, POWER, MOPEX). EPF needs seq_len=168/pred_len=24.
# This script trains to a SEPARATE directory (checkpoints_168_24/) so the 96-lookback
# checkpoints remain untouched for long-term datasets.
#
# Usage:
#   bash scripts/train_epf_checkpoints.sh                    # All 5 EPF datasets, all 5 models
#   bash scripts/train_epf_checkpoints.sh EPF_BE             # Single dataset, all models
#   bash scripts/train_epf_checkpoints.sh EPF_BE PatchTST    # Single dataset+model
#
# After training, update config.yaml checkpoints for each EPF dataset, e.g.:
#   checkpoints:
#     DLinear: checkpoints_168_24/EPF_BE/DLinear_checkpoint.pth
#     iTransformer: checkpoints_168_24/EPF_BE/iTransformer_checkpoint.pth
#     PatchTST: checkpoints_168_24/EPF_BE/PatchTST_checkpoint.pth
#     TimesNet: checkpoints_168_24/EPF_BE/TimesNet_checkpoint.pth
#     Autoformer: checkpoints_168_24/EPF_BE/Autoformer_checkpoint.pth

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
THUML_DIR="$ROOT_DIR/.thuml/Time-Series-Library"
# NEW output directory — separate from checkpoints/ (96-lookback)
CHECKPOINTS_DIR="$ROOT_DIR/checkpoints_168_24"

# ---------------------------------------------------------------------------
# EPF datasets only (168 lookback, 24 prediction)
# ---------------------------------------------------------------------------
DATASETS=(
  "EPF_BE|./dataset/EPF/|EPF_BE.csv|custom|h|real_power|dataset/EPF/BE/EPF_BE_train.csv"
  "EPF_DE|./dataset/EPF/|EPF_DE.csv|custom|h|real_power|dataset/EPF/DE/EPF_DE_train.csv"
  "EPF_FR|./dataset/EPF/|EPF_FR.csv|custom|h|real_power|dataset/EPF/FR/EPF_FR_train.csv"
  "EPF_NP|./dataset/EPF/|EPF_NP.csv|custom|h|real_power|dataset/EPF/NP/EPF_NP_train.csv"
  "EPF_PJM|./dataset/EPF/|EPF_PJM.csv|custom|h|real_power|dataset/EPF/PJM/EPF_PJM_train.csv"
)

# All 5 DL models
MODELS=("Autoformer" "DLinear" "PatchTST" "TimesNet" "iTransformer")

# EPF-specific window sizes
SEQ_LEN=168
LABEL_LEN=84
PRED_LEN=24

# ---------------------------------------------------------------------------
# Model-specific hyperparameters (same as original, shared across window sizes)
# ---------------------------------------------------------------------------
get_model_params() {
  local model="$1"
  case "$model" in
    TimesNet)
      echo "16|32|8|2"      # d_model=16, d_ff=32, n_heads=8, e_layers=2
      ;;
    PatchTST)
      echo "512|2048|4|2"   # d_model=512, d_ff=2048, n_heads=4, e_layers=2 (bigger seq needs more heads)
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
# Clone THUML if not present
# ---------------------------------------------------------------------------
setup_thuml() {
  if [ ! -d "$THUML_DIR" ]; then
    echo "=== Cloning THUML/Time-Series-Library ==="
    mkdir -p "$(dirname "$THUML_DIR")"
    git clone --depth 1 https://github.com/thuml/Time-Series-Library.git "$THUML_DIR"
  fi

  echo "=== Installing THUML dependencies ==="
  conda run -n AlphaCast pip install numpy scipy scikit-learn pandas matplotlib sktime sympy \
    PyWavelets datasets einops tqdm patool 2>&1 | tail -3
}

# ---------------------------------------------------------------------------
# Prepare dataset symlinks into THUML's expected layout
# ---------------------------------------------------------------------------
prepare_datasets() {
  echo "=== Preparing datasets ==="
  local thuml_data="$THUML_DIR/dataset"

  for entry in "${DATASETS[@]}"; do
    IFS='|' read -r ds_name root_path data_path _data_name _freq _target_col source_csv <<<"$entry"
    local subdir="${root_path#./dataset/}"
    local dst_dir="$thuml_data/$subdir"
    mkdir -p "$dst_dir"
    local dst="$dst_dir/$data_path"

    if [ -f "$dst" ]; then
      echo "  Dataset $ds_name already exists at $dst"
      continue
    fi

    if [ -n "${source_csv:-}" ] && [ -f "$ROOT_DIR/$source_csv" ]; then
      python3 - "$ROOT_DIR/$source_csv" "$dst" "$ds_name" <<'PYEOF'
import sys, pandas as pd
src_path, dst_path, ds_name = sys.argv[1], sys.argv[2], sys.argv[3]
df = pd.read_csv(src_path)
time_cols = ["date", "time_stamp", "timestamp", "time"]
time_col = next((c for c in time_cols if c in df.columns), None)
if time_col and time_col != "date":
    df.rename(columns={time_col: "date"}, inplace=True)
df.to_csv(dst_path, index=False)
print(f"  Prepared {ds_name}: {src_path} → {dst_path} ({len(df)} rows)")
PYEOF
    else
      echo "  [warn] Source data not found for $ds_name. Skipping."
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

  local d_model d_ff n_heads e_layers
  IFS='|' read -r d_model d_ff n_heads e_layers <<<"$(get_model_params "$model")"

  echo "  [$ds_name/$model] Training (seq_len=$SEQ_LEN, pred_len=$PRED_LEN, d_model=$d_model, n_heads=$n_heads) ..."

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
    --des 'AlphaCast_EPF' \
    --itr 1 \
    --train_epochs 100 \
    --batch_size 32 \
    --patience 10 \
    --learning_rate 0.0001 \
    --freq "$freq"

  popd > /dev/null

  # Copy checkpoint from THUML output to AlphaCast checkpoints_168_24 directory
  local thuml_ckpt_file
  thuml_ckpt_file=$(find "$THUML_DIR/checkpoints" -path "*${model_id}*${model}*AlphaCast_EPF*" -name "checkpoint.pth" -print -quit 2>/dev/null)

  if [ -n "$thuml_ckpt_file" ] && [ -f "$thuml_ckpt_file" ]; then
    cp "$thuml_ckpt_file" "$ckpt_path"
    echo "  [$ds_name/$model] Checkpoint saved → $ckpt_path"
  else
    echo "  [warn] [$ds_name/$model] Training completed but checkpoint not found under THUML."
    echo "         Locate manually and copy to: $ckpt_path"
  fi
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
main() {
  echo "============================================"
  echo " EPF Checkpoint Training (168→24)"
  echo "============================================"
  echo " Output dir: $CHECKPOINTS_DIR"
  echo " Models: ${MODELS[*]}"
  echo " Datasets:"
  for entry in "${DATASETS[@]}"; do
    IFS='|' read -r ds_name _ <<<"$entry"
    echo "  - $ds_name"
  done
  echo ""

  setup_thuml
  prepare_datasets

  local total=0 skipped=0

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
      ckpt_path="$CHECKPOINTS_DIR/$ds_name/${model}_checkpoint.pth"
      if [ -f "$ckpt_path" ]; then
        skipped=$((skipped + 1))
        echo "[$ds_name/$model] Checkpoint exists. Skipped."
        continue
      fi
      train_one "$ds_name" "$model" || echo "[warn] [$ds_name/$model] Training failed."
    done
  done

  echo ""
  echo "=== Summary ==="
  echo "Total: $total | Skipped: $skipped"
  echo ""
  echo "Next step: update config.yaml checkpoints for EPF datasets:"
  for entry in "${DATASETS[@]}"; do
    IFS='|' read -r ds_name _ <<<"$entry"
    echo ""
    echo "  # config.yaml → $ds_name"
    echo "  checkpoints:"
    for model in "${MODELS[@]}"; do
      echo "    $model: checkpoints_168_24/$ds_name/${model}_checkpoint.pth"
    done
  done
}

main
