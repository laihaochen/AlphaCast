#!/usr/bin/env python3
"""
Evaluate Prophet decomposition accuracy on the test set (no LLM involved).

Prophet prediction = trend_global(t) + trend_local(t) + seasonality_global(t)
where:
  - trend_global + seasonality: fitted once on full training set
  - trend_local: per-window weighted linear regression on detrended lookback

Usage:
  python scripts/eval_prophet.py EPF_BE                     # single dataset
  python scripts/eval_prophet.py EPF_BE EPF_DE EPF_FR       # multiple
  python scripts/eval_prophet.py                            # all in config.yaml
  python scripts/eval_prophet.py EPF_BE --decay 0.05        # adjust local trend reactivity
  python scripts/eval_prophet.py EPF_BE --method linear     # use equal-weight linear

Output:
  - Terminal: per-dataset metrics table (global vs combined), per-window breakdown
  - evaluation_plots/<Dataset>_prophet_eval.png  (if --plot)
"""

from __future__ import annotations
import argparse, json, os, sys, warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from alphacast.models.base import ProphetModel
from alphacast.tools.forecast import _forecast_regularized_trend

CONFIG_PATH = ROOT / "config.yaml"

# Matplotlib setup
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def smape(a: np.ndarray, p: np.ndarray) -> float:
    denom = np.abs(a) + np.abs(p)
    return float(np.nanmean(200 * np.abs(a - p) / np.where(denom < 1e-9, np.inf, denom)))


def safe_load_csv(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def get_target_col(df: pd.DataFrame) -> str:
    skip = {"date", "time_stamp", "predicted_ans"}
    candidates = [c for c in df.columns if c not in skip]
    return candidates[-1] if candidates else df.columns[-1]


# ---------------------------------------------------------------------------
# Prophet prediction pipeline
# ---------------------------------------------------------------------------
def build_prophet_predictions(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    look_back: int,
    forecast_horizon: int,
    stride: int,
    alpha: float = 0.05,
    cp_threshold: float = 0.0,
    decay: float = 0.03,
    continuity_scale: float = 1.0,
) -> dict:
    """
    Sliding-window Prophet decomposition over test set.

    Returns dict with arrays: global_only, combined (global+local), actual.
    """
    train_y = train_df[target_col].to_numpy(dtype=float)
    train_ts = pd.to_datetime(train_df["date"])
    test_y = test_df[target_col].to_numpy(dtype=float)
    test_ts = pd.to_datetime(test_df["date"])

    # ---- Global Prophet (fit once on full training set) ----
    gmodel = ProphetModel()
    gmodel.fit(train_y, timestamps=train_ts)

    # Pre-compute decomposition for all timestamps (train + test)
    train_decomp = gmodel.decompose(timestamps=train_ts)
    test_decomp = gmodel.decompose(timestamps=test_ts)

    global_trend_train = train_decomp["trend"]
    global_seas_train = train_decomp["seasonality"]
    global_trend_test = test_decomp["trend"]
    global_seas_test = test_decomp["seasonality"]

    # ---- Sliding window over test set ----
    target_len = len(test_y) - look_back
    n_windows = 0
    pred_global: list[float] = []        # global trend + seasonality
    pred_combined: list[float] = []      # global + local trend + seasonality
    trend_global_only: list[float] = []   # global trend only (no seasonality)
    trend_combined_only: list[float] = [] # combined trend only (no seasonality)
    actual_vals: list[float] = []
    window_metrics: list[dict] = []

    prev_slope = None
    prev_boundary = None

    offset = 0
    while offset + forecast_horizon <= target_len:
        lb_start = offset
        lb_end = offset + look_back
        fc_start = lb_end
        fc_end = fc_start + forecast_horizon

        # Lookback window
        y_lb = test_y[lb_start:lb_end]
        t_g_lb = global_trend_test[lb_start:lb_end]
        s_g_lb = global_seas_test[lb_start:lb_end]

        # Forecast window
        t_g_fc = global_trend_test[fc_start:fc_end]
        s_g_fc = global_seas_test[fc_start:fc_end]
        y_actual = test_y[fc_start:fc_end]

        # Global-only prediction: trend + seasonality
        y_global = t_g_fc + s_g_fc

        # Local trend correction on detrended lookback
        detrended_lb = y_lb - t_g_lb - s_g_lb
        has_cp, cp_pos = False, None
        try:
            t_local_fc, prev_slope, prev_boundary, has_cp, cp_pos = _forecast_regularized_trend(
                detrended_lb, forecast_horizon,
                prev_slope=prev_slope, prev_boundary=prev_boundary,
                ema_alpha=alpha, cp_threshold=cp_threshold, decay=decay,
                continuity_scale=continuity_scale,
            )
        except Exception:
            t_local_fc = np.zeros(forecast_horizon)

        # Combined prediction: (global trend + local correction) + seasonality
        y_combined = t_g_fc + t_local_fc + s_g_fc

        pred_global.extend(y_global.tolist())
        pred_combined.extend(y_combined.tolist())
        trend_global_only.extend(t_g_fc.tolist())
        trend_combined_only.extend((t_g_fc + t_local_fc).tolist())
        actual_vals.extend(y_actual.tolist())

        # Per-window metrics
        n_windows += 1
        mae_g = float(np.mean(np.abs(y_actual - y_global)))
        mae_c = float(np.mean(np.abs(y_actual - y_combined)))
        local_mean = float(np.mean(np.abs(t_local_fc)))

        # Track changepoint activation
        cp_improv = 0.0  # not separately tracked for regularized

        window_metrics.append({
            "window": n_windows,
            "offset": offset,
            "global_mae": mae_g,
            "combined_mae": mae_c,
            "delta_mae": mae_c - mae_g,
            "local_correction_mean_abs": local_mean,
            "has_changepoint": has_cp,
            "changepoint_pos": cp_pos,
            "changepoint_improvement": cp_improv,
        })

        offset += stride

    # Tail: remaining points if stride doesn't divide evenly
    remaining = target_len - offset
    if remaining > 0:
        lb_start = offset
        lb_end = offset + look_back
        fc_start = lb_end
        fc_end = min(fc_start + remaining, len(test_y))

        y_lb = test_y[lb_start:lb_end]
        t_g_lb = global_trend_test[lb_start:lb_end]
        s_g_lb = global_seas_test[lb_start:lb_end]
        t_g_fc = global_trend_test[fc_start:fc_end]
        s_g_fc = global_seas_test[fc_start:fc_end]
        y_actual = test_y[fc_start:fc_end]

        detrended_lb = y_lb - t_g_lb - s_g_lb
        try:
            t_local_fc, _, _, _, _ = _forecast_regularized_trend(
                detrended_lb, remaining,
                prev_slope=prev_slope, prev_boundary=prev_boundary,
                ema_alpha=alpha, cp_threshold=cp_threshold, decay=decay,
                continuity_scale=continuity_scale,
            )
        except Exception:
            t_local_fc = np.zeros(remaining)

        pred_global.extend((t_g_fc + s_g_fc).tolist())
        pred_combined.extend((t_g_fc + t_local_fc + s_g_fc).tolist())
        trend_global_only.extend(t_g_fc.tolist())
        trend_combined_only.extend((t_g_fc + t_local_fc).tolist())
        actual_vals.extend(y_actual.tolist())

    return {
        "pred_global": np.asarray(pred_global),
        "pred_combined": np.asarray(pred_combined),
        "trend_global_only": np.asarray(trend_global_only),
        "trend_combined_only": np.asarray(trend_combined_only),
        "actual": np.asarray(actual_vals),
        "n_windows": n_windows,
        "window_metrics": window_metrics,
        "test_timestamps": [ts.isoformat() for ts in test_ts[look_back: look_back + len(actual_vals)]],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate Prophet decomposition accuracy")
    parser.add_argument("datasets", nargs="*", help="Dataset names to evaluate (default: all in config)")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="EMA blending rate (0=frozen, 1=no smoothing) (default: 0.05)")
    parser.add_argument("--cp-threshold", type=float, default=0.0,
                        help="Changepoint detection threshold as relative slope change. "
                             "Lower = more sensitive; 0 = always trigger CP. NOT an L1 reg coeff. (default: 0.0)")
    parser.add_argument("--l1-lambda", type=float, dest="cp_threshold",
                        help="Deprecated alias for --cp-threshold")
    parser.add_argument("--decay", type=float, default=0.03,
                        help="OLS weight decay within window (default: 0.03)")
    parser.add_argument("--continuity-scale", type=float, default=1.0,
                        help="Multiplier on std(y) for cross-window continuity clamp. "
                             "0 = no continuity, higher = more smoothing (default: 1.0)")
    parser.add_argument("--plot", action="store_true", help="Generate per-dataset evaluation plots")
    parser.add_argument("--output-dir", default=None, help="Override output directory for plots")
    args = parser.parse_args()

    # Load config
    with open(CONFIG_PATH) as f:
        import yaml
        cfg = yaml.safe_load(f)

    # Resolve datasets
    all_names = [ds["name"] for ds in cfg["datasets"]]
    alias_map: dict[str, str] = {}
    for ds in cfg["datasets"]:
        for a in ds.get("aliases", []):
            alias_map[a.lower()] = ds["name"]
        alias_map[ds["name"].lower()] = ds["name"]

    if args.datasets:
        names = []
        for token in args.datasets:
            resolved = alias_map.get(token.lower(), token)
            names.append(resolved)
    else:
        names = all_names

    plots_dir = Path(args.output_dir) if args.output_dir else ROOT / "evaluation_plots"
    os.makedirs(plots_dir, exist_ok=True)

    header = f"Prophet Decomposition Evaluation — α={args.alpha}, cp_threshold={args.cp_threshold}, cont_scale={args.continuity_scale}"
    print(f"\n{'='*90}")
    print(header)
    print(f"{'='*90}")
    print(f"{'Dataset':<14} {'Windows':>8} {'Global MAE':>12} {'Global sMAPE':>13} "
          f"{'Combined MAE':>13} {'Combined sMAPE':>14} {'Δ MAE':>10} {'Win rate':>10} {'CP act.':>9}")
    print("-" * 108)

    all_results = []

    for name in names:
        ds_cfg = None
        for d in cfg["datasets"]:
            if d["name"] == name:
                ds_cfg = d
                break
        if ds_cfg is None:
            print(f"  [{name}] not in config — skip")
            continue

        train_path = ROOT / ds_cfg["training_csv"]
        test_path = ROOT / ds_cfg["test_csv"]
        train_df = safe_load_csv(train_path)
        test_df = safe_load_csv(test_path)

        if train_df is None or test_df is None:
            print(f"  [{name}] missing data — skip")
            continue

        train_df["date"] = pd.to_datetime(train_df["date"])
        test_df["date"] = pd.to_datetime(test_df["date"])
        target_col = get_target_col(train_df)
        look_back = int(ds_cfg.get("look_back", 96))
        forecast_horizon = int(ds_cfg.get("predicted_window", 96))
        stride = int(ds_cfg.get("sliding_window", forecast_horizon))

        result = build_prophet_predictions(
            train_df, test_df, target_col,
            look_back, forecast_horizon, stride,
            alpha=args.alpha, cp_threshold=args.cp_threshold, decay=args.decay,
            continuity_scale=args.continuity_scale,
        )

        pred_g = result["pred_global"]
        pred_c = result["pred_combined"]
        actual = result["actual"]

        mae_g = float(np.mean(np.abs(actual - pred_g)))
        mse_g = float(np.mean((actual - pred_g) ** 2))
        smape_g = smape(actual, pred_g)

        mae_c = float(np.mean(np.abs(actual - pred_c)))
        mse_c = float(np.mean((actual - pred_c) ** 2))
        smape_c = smape(actual, pred_c)

        delta = mae_c - mae_g

        # Per-window win rate: how many windows does combined beat global?
        wm = result["window_metrics"]
        wins = sum(1 for m in wm if m["combined_mae"] < m["global_mae"])
        win_rate = f"{wins}/{len(wm)} ({wins/len(wm)*100:.0f}%)" if wm else "N/A"

        sign = "+" if delta > 0 else ""
        total_cp = sum(1 for m in wm if m.get("has_changepoint"))
        cp_str = f"{total_cp}/{len(wm)}" if wm else "N/A"
        print(f"{name:<14} {result['n_windows']:>8} {mae_g:>12.4f} {smape_g:>12.2f}% "
              f"{mae_c:>12.4f} {smape_c:>13.2f}% {sign}{delta:>9.4f} {win_rate:>10} {cp_str:>9}")

        all_results.append({
            "name": name,
            "n_windows": result["n_windows"],
            "mae_global": mae_g, "mse_global": mse_g, "smape_global": smape_g,
            "mae_combined": mae_c, "mse_combined": mse_c, "smape_combined": smape_c,
            "delta_mae": delta,
            "result": result,
        })

        # ---- Plot ----
        if args.plot and len(actual) > 0:
            n_pts = len(actual)
            fig, axes = plt.subplots(2, 2, figsize=(18, 9))
            fig.suptitle(f"{name} — Prophet Decomposition (α={args.alpha}, cp_threshold={args.cp_threshold}, cs={args.continuity_scale})",
                         fontweight="bold")

            x_all = np.arange(n_pts)
            trend_g = result.get("trend_global_only", pred_g)
            trend_c_raw = result.get("trend_combined_only", pred_c)

            trend_c = trend_c_raw

            # (A) Trend only — Global vs Combined Trend vs Actual
            disp_a = min(n_pts, 2000)
            step_a = max(1, n_pts // disp_a)
            idx_a = np.arange(0, n_pts, step_a)
            ax = axes[0, 0]
            ax.plot(x_all[idx_a], actual[idx_a], "k", alpha=0.35, lw=0.4, label="Actual")
            ax.plot(x_all[idx_a], trend_g[idx_a], "#64B5F6", alpha=0.8, lw=1.3, label="Global trend")
            ax.plot(x_all[idx_a], trend_c[idx_a], "#E65100", alpha=0.85, lw=1.0, label="Combined trend (per-window)")

            cp_windows = [m for m in wm if m.get("has_changepoint")]
            for m in cp_windows:
                ax.axvspan(m["offset"] + look_back,
                           min(m["offset"] + look_back + forecast_horizon, n_pts),
                           alpha=0.12, color="#FF5722")
            ax.set_title(f"Trend Only — {len(wm)} windows, {len(cp_windows)} CPs (cp={args.cp_threshold}, cs={args.continuity_scale})")
            ax.legend(fontsize=6.5, loc="upper right")
            ax.grid(True, alpha=0.3)

            # (B) Prophet residuals — what remains after removing trend + seasonality
            ax = axes[0, 1]
            residual_global = actual - pred_g  # after trend_global + seasonality
            residual_combined = actual - pred_c  # after trend_global + trend_local + seasonality
            disp_b = min(n_pts, 3000)
            step_b = max(1, n_pts // disp_b)
            idx_b = np.arange(0, n_pts, step_b)
            ax.plot(x_all[idx_b], residual_global[idx_b], "#90CAF9", alpha=0.6, lw=0.5,
                    label=f"Global (σ={np.std(residual_global):.2f})")
            ax.plot(x_all[idx_b], residual_combined[idx_b], "#7B1FA2", alpha=0.8, lw=0.7,
                    label=f"+ local trend (σ={np.std(residual_combined):.2f})")
            ax.axhline(0, color="k", lw=0.5, ls="--")
            ax.set_title(f"Prophet Residuals (y − trend − seasonality, n={n_pts})")
            ax.set_xlabel("Test time step")
            ax.set_ylabel("Residual value")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

            # (C) Per-window MAE comparison
            ax = axes[1, 0]
            xw = np.arange(len(wm))
            ww = 0.35
            ax.bar(xw - ww/2, [m["global_mae"] for m in wm], ww, color="#90CAF9", alpha=0.8, label="Global")
            ax.bar(xw + ww/2, [m["combined_mae"] for m in wm], ww, color="#E65100", alpha=0.8, label="Combined")
            ax.set_title(f"Per-Window MAE ({len(wm)} windows)")
            ax.set_xlabel("Window")
            ax.set_ylabel("MAE")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3, axis="y")

            # (D) Trend + Seasonality — Global vs Combined vs Actual
            ax = axes[1, 1]
            disp_d = min(n_pts, 2000)
            step_d = max(1, n_pts // disp_d)
            idx_d = np.arange(0, n_pts, step_d)
            ax.plot(x_all[idx_d], actual[idx_d], "k", alpha=0.35, lw=0.4, label="Actual")
            ax.plot(x_all[idx_d], pred_g[idx_d], "#64B5F6", alpha=0.8, lw=1.3, label="Global trend + seasonality")
            ax.plot(x_all[idx_d], pred_c[idx_d], "#E65100", alpha=0.85, lw=1.0, label="Combined trend + seasonality")

            # Window boundaries
            # Changepoint windows highlighted
            for m in cp_windows:
                ax.axvspan(m["offset"] + look_back,
                           min(m["offset"] + look_back + forecast_horizon, n_pts),
                           alpha=0.12, color="#FF5722")

            ax.set_title(f"Trend + Seasonality — {total_cp}/{len(wm)} windows with internal changepoint")
            ax.set_xlabel("Test time step")
            ax.set_ylabel("Value")
            ax.legend(fontsize=6.5, loc="upper right")
            ax.grid(True, alpha=0.3)

            fig.tight_layout(rect=[0, 0, 1, 0.95])
            fig.savefig(plots_dir / f"{name}_prophet_eval.png")
            plt.close(fig)
            print(f"  → saved {name}_prophet_eval.png")

    # ---- Summary ----
    if len(all_results) >= 2:
        print(f"\n{'='*90}")
        print("Summary")
        print(f"{'='*90}")
        print(f"{'Dataset':<14} {'Global MAE':>12} {'Combined MAE':>13} {'Δ MAE':>10} {'Local helps?':>12}")
        print("-" * 65)
        for r in all_results:
            delta = r["delta_mae"]
            sign = "+" if delta > 0 else ""
            better = "YES ✓" if delta < -0.001 else ("no ✗" if delta > 0.001 else "tie")
            print(f"{r['name']:<14} {r['mae_global']:>12.4f} {r['mae_combined']:>12.4f} {sign}{delta:>9.4f} {better:>12}")

    print(f"\nDone. Plots saved to: {plots_dir}/" if args.plot else "\nDone.")


if __name__ == "__main__":
    main()
