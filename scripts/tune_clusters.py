#!/usr/bin/env python3
"""
Cluster number hyperparameter sweep for AlphaCast.

For each (num_clusters, dataset) combination:
  1. Creates a temporary config with the target num_clusters value.
  2. Cleans stale output files (cluster, predictions, resume state).
  3. Runs the full LLM pipeline via run_experiment.py.
  4. Computes MSE / MAE / sMAPE from predictions.csv vs test.csv.
  5. Prints a comparison table at the end.

Usage:
  python scripts/tune_clusters.py --clusters 6,8,10,12 --datasets EPF_BE,EPF_FR
  python scripts/tune_clusters.py --clusters 8,10,12 --datasets all_epf
  python scripts/tune_clusters.py --clusters 6,10 --datasets EPF_BE --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.yaml"

# Files to clean before each run so the pipeline starts fresh
CLEAN_FILES = [
    "cluster_base.json",
    "predictions.csv",
    "residuals.csv",
    "prophet_components.csv",
    "llm_resume_state.json",
    "chain_of_thought.log",
    "basemodel_results.json",
]

# Shorthand aliases for convenience
DATASET_GROUPS: Dict[str, List[str]] = {
    "all_epf": ["EPF_BE", "EPF_DE", "EPF_FR", "EPF_NP", "EPF_PJM"],
    "all_long": ["ETTh1", "ETTm1", "WP", "SP", "MOPEX"],
    "all": [],  # populated at runtime from config
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def smape(a: np.ndarray, p: np.ndarray) -> float:
    denom = np.abs(a) + np.abs(p)
    return float(np.nanmean(200 * np.abs(a - p) / np.where(denom < 1e-9, np.inf, denom)))


def get_target_col(df: pd.DataFrame) -> str:
    skip = {"date", "time_stamp", "predicted_ans", "features_used"}
    candidates = [c for c in df.columns if c not in skip]
    return candidates[-1] if candidates else df.columns[-1]


def compute_metrics(
    output_dir: str, dataset_name: str, test_csv: str
) -> Tuple[float, float, float] | None:
    """Compute MSE / MAE / sMAPE from predictions.csv vs test.csv."""
    pred_path = Path(output_dir) / dataset_name / "predictions.csv"
    test_path = ROOT / test_csv

    if not pred_path.exists() or not test_path.exists():
        return None

    try:
        pred_df = pd.read_csv(pred_path)
        test_df = pd.read_csv(test_path)
    except Exception:
        return None

    target_col = get_target_col(test_df)
    col_p = "predicted_ans" if "predicted_ans" in pred_df.columns else "prediction"
    if col_p not in pred_df.columns:
        col_p = [c for c in pred_df.columns if c != "time_stamp"][0]

    actual = test_df[target_col].to_numpy(dtype=float)
    preds = pred_df[col_p].to_numpy(dtype=float)

    n = min(len(preds), len(actual))
    actual, preds = actual[:n], preds[:n]

    mask = ~(np.isnan(actual) | np.isnan(preds))
    actual, preds = actual[mask], preds[mask]

    if len(actual) < 10:
        return None

    mse = float(np.mean((actual - preds) ** 2))
    mae = float(np.mean(np.abs(actual - preds)))
    sm = smape(actual, preds)
    return mse, mae, sm


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Sweep num_clusters across datasets and compare results."
    )
    parser.add_argument(
        "--clusters", required=True,
        help="Comma-separated cluster numbers, e.g. 6,8,10,12",
    )
    parser.add_argument(
        "--datasets", required=True,
        help=(
            "Comma-separated dataset names or group alias. "
            "Aliases: all_epf, all_long, all"
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print configs that would be used but don't run anything.",
    )
    parser.add_argument(
        "--no-clean", action="store_true",
        help="Skip cleaning output files (use for resume).",
    )
    args = parser.parse_args()

    # ---- Resolve cluster numbers ----
    cluster_nums = [int(x.strip()) for x in args.clusters.split(",") if x.strip()]

    # ---- Resolve datasets ----
    with open(CONFIG_PATH) as f:
        base_cfg = yaml.safe_load(f)

    all_names = [d["name"] for d in base_cfg["datasets"]]
    all_test_csv = {d["name"]: d["test_csv"] for d in base_cfg["datasets"]}
    DATASET_GROUPS["all"] = all_names

    dataset_tokens = [x.strip() for x in args.datasets.split(",") if x.strip()]
    datasets: List[str] = []
    for token in dataset_tokens:
        if token in DATASET_GROUPS:
            datasets.extend(DATASET_GROUPS[token])
        elif token in all_names:
            datasets.append(token)
        else:
            print(f"[warn] Unknown dataset/group '{token}' — skipping")
    datasets = list(dict.fromkeys(datasets))  # dedup, preserve order

    if not datasets:
        print("[error] No valid datasets specified.")
        sys.exit(1)

    output_dir = base_cfg.get("output_dir", "outputs")

    print(f"\n{'='*70}")
    print(f"Cluster Sweep: {cluster_nums}")
    print(f"Datasets:       {datasets}")
    print(f"Output dir:     {output_dir}/")
    print(f"{'='*70}")

    if args.dry_run:
        print("\n[Dry-run] Would create temp configs with:")
        for nc in cluster_nums:
            print(f"  num_clusters={nc} on {datasets}")
        print("\n[Dry-run] Done — no experiments executed.")
        return

    # ---- Run sweep ----
    all_results: List[dict] = []

    for nc in cluster_nums:
        # Build temp config
        with open(CONFIG_PATH) as f:
            run_cfg = yaml.safe_load(f)
        for d in run_cfg["datasets"]:
            if d["name"] in datasets:
                d["num_clusters"] = nc

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, dir=ROOT
        ) as tf:
            yaml.dump(run_cfg, tf, default_flow_style=False)
            temp_config = tf.name

        print(f"\n{'─'*60}")
        print(f"[run] num_clusters={nc}")
        print(f"[run] temp config: {temp_config}")
        print(f"{'─'*60}")

        # Clean stale files
        if not args.no_clean:
            for ds in datasets:
                ds_dir = Path(output_dir) / ds
                for fname in CLEAN_FILES:
                    p = ds_dir / fname
                    if p.exists():
                        p.unlink()
                        print(f"  [clean] {ds}/{fname}")
                # Also clean case_base / case_neighbor (they embed cluster info)
                for extra in ["case_base.json", "case_neighbor.json"]:
                    ep = ds_dir / extra
                    if ep.exists():
                        ep.unlink()

        # Build command
        ds_flags: List[str] = []
        for ds in datasets:
            ds_flags.extend(["--dataset", ds])

        cmd = [
            sys.executable, "-u",
            str(ROOT / "run_experiment.py"),
            "--config", temp_config,
        ] + ds_flags

        env = os.environ.copy()
        env["ORCHESTRATION_MODE"] = "llm"

        print(f"\n[cmd] ORCHESTRATION_MODE=llm python -u run_experiment.py ...")
        print(f"[cmd] {' '.join(ds_flags)}")

        try:
            proc = subprocess.run(
                cmd, env=env, cwd=str(ROOT),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True,
            )
        except KeyboardInterrupt:
            print(f"\n[abort] Interrupted during num_clusters={nc}. Cleaning up...")
            os.unlink(temp_config)
            sys.exit(1)
        finally:
            if os.path.exists(temp_config):
                os.unlink(temp_config)

        # Collect per-dataset metrics
        for ds in datasets:
            metrics = compute_metrics(output_dir, ds, all_test_csv[ds])
            if metrics:
                mse, mae, sm = metrics
                print(f"  [{ds}] MSE={mse:.6f}  MAE={mae:.6f}  sMAPE={sm:.6f}")
                all_results.append({
                    "num_clusters": nc,
                    "dataset": ds,
                    "mse": mse,
                    "mae": mae,
                    "smape": sm,
                })
            else:
                print(f"  [{ds}] no valid predictions — skipped")

    # ---- Summary tables ----
    if not all_results:
        print("\n[info] No results collected.")
        return

    df = pd.DataFrame(all_results)

    # Per-dataset comparison: rows=dataset, cols=cluster_num
    print(f"\n{'='*80}")
    print("Cluster Sweep Summary — MAE")
    print(f"{'='*80}")

    for ds in datasets:
        ds_rows = df[df["dataset"] == ds].sort_values("num_clusters")
        if ds_rows.empty:
            continue
        print(f"\n  {ds}:")
        print(f"  {'num_clusters':>14} {'MSE':>12} {'MAE':>10} {'sMAPE':>8}")
        print(f"  {'─'*48}")
        best_mae = ds_rows["mae"].min()
        for _, r in ds_rows.iterrows():
            marker = " ← best" if r["mae"] == best_mae else ""
            print(
                f"  {int(r['num_clusters']):>14} "
                f"{r['mse']:>12.6f} {r['mae']:>10.6f} "
                f"{r['smape']:>8.6f}{marker}"
            )

    # Overall: best cluster per dataset
    print(f"\n{'='*80}")
    print("Best num_clusters per Dataset")
    print(f"{'='*80}")
    print(f"  {'Dataset':<14} {'Best k':>8} {'MAE':>10} {'MSE':>12} {'sMAPE':>8}")
    print(f"  {'─'*56}")
    for ds in datasets:
        ds_rows = df[df["dataset"] == ds]
        if ds_rows.empty:
            continue
        best = ds_rows.loc[ds_rows["mae"].idxmin()]
        print(
            f"  {ds:<14} {int(best['num_clusters']):>8} "
            f"{best['mae']:>10.6f} {best['mse']:>12.6f} "
            f"{best['smape']:>8.6f}"
        )

    # Unified summary block (matching result.txt format)
    print(f"\n{'='*80}")
    print("=== Experiment Summary ===")
    print(f"{'='*80}")
    for nc in cluster_nums:
        nc_rows = df[df["num_clusters"] == nc].sort_values("dataset")
        if nc_rows.empty:
            continue
        print(f"\n  num_clusters={nc}")
        print(f"  {'dataset':>14} {'MSE':>12} {'MAE':>10} {'sMAPE':>8}  model")
        print(f"  {'─'*52}")
        for _, r in nc_rows.iterrows():
            print(
                f"  {r['dataset']:>14} "
                f"{r['mse']:>12.6f} {r['mae']:>10.6f} "
                f"{r['smape']:>8.6f}   LLM"
            )

    print(f"\nDone.")


if __name__ == "__main__":
    main()
