#!/usr/bin/env python3
"""Quick evaluation: row counts + MAE/MSE/sMAPE for all datasets."""
from __future__ import annotations
import numpy as np, pandas as pd, yaml, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config.yaml"

with open(CONFIG) as f:
    cfg = yaml.safe_load(f)

print(f"{'Dataset':<14} {'Pred':>6} {'Test':>6} {'Expect':>6} {'MAE':>10} {'MSE':>10} {'sMAPE':>8}")
print("-" * 72)

for ds in cfg["datasets"]:
    name = ds["name"]
    pred_path = ROOT / "outputs" / name / "predictions.csv"
    test_path = ROOT / ds["test_csv"]
    if not pred_path.exists() or not test_path.exists():
        print(f"{name:<14} {'MISSING':>6}")
        continue

    pred_df = pd.read_csv(pred_path)
    test_df = pd.read_csv(test_path)

    # target column
    exclude = {"date", "time_stamp", "predicted_ans", "features_used"}
    target_col = [c for c in test_df.columns if c not in exclude][-1]
    actual = test_df[target_col].to_numpy(dtype=float)

    pred_col = "predicted_ans" if "predicted_ans" in pred_df.columns else "prediction"
    predicted = pred_df[pred_col].to_numpy(dtype=float)

    expect = len(test_df) - 96  # look_back
    n = min(len(predicted), len(actual))

    if n == 0:
        print(f"{name:<14} {len(predicted):>6} {len(actual):>6} {expect:>6}")
        continue

    a = actual[-n:]
    p = predicted[:n]
    mask = ~(np.isnan(a) | np.isnan(p))
    a, p = a[mask], p[mask]

    mae = float(np.mean(np.abs(a - p)))
    mse = float(np.mean((a - p) ** 2))
    denom = np.abs(a) + np.abs(p)
    smape = float(np.mean(200 * np.abs(a - p) / np.where(denom == 0, np.inf, denom)))

    print(f"{name:<14} {len(predicted):>6} {len(actual):>6} {expect:>6} {mae:>10.4f} {mse:>10.4f} {smape:>7.2f}%")

print("-" * 72)
