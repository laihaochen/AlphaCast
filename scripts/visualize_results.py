#!/usr/bin/env python3
"""
Comprehensive evaluation & visualization: baseline vs LLM predictions.

Usage:
    python scripts/visualize_results.py

Prerequisites:
    - config.yaml 中列出的每个数据集，需要在以下两个目录都有 predictions.csv：
        outputs/<dataset>/predictions.csv        （LLM 实验产出）
        baseline_outputs/<dataset>/predictions.csv （确定性 baseline 产出）
    - 对应的 test_csv 存在于 dataset/ 下

注意：
    - 只有 outputs/ 和 baseline_outputs/ 中 predictions 值不同的数据集
      才会生成 LLM 分析图（*_llm_analysis.png）
    - 目前只有 output_dir 为 outputs/ 和 baseline_outputs/，若 config 中
      output_dir 改过，需同步修改脚本顶部的 OUTPUTS_DIR / BASELINE_DIR

输出文件（evaluation_plots/ 目录下）:
    <Dataset>_comparison.png      每个数据集：实际值 vs baseline vs LLM，误差分布，逐窗口 MAE
    <Dataset>_llm_analysis.png    仅 LLM 有效的数据集：调整幅度、误差改善分析
    00_summary_comparison.png     全局对比柱状图（MAE / MSE / sMAPE）
    00_delta_summary.png          LLM 对每个数据集 MAE 的净影响

依赖:
    pip install numpy pandas matplotlib pyyaml
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import yaml
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.yaml"
OUTPUTS_DIR = ROOT / "outputs"
BASELINE_DIR = ROOT / "baseline_outputs"
PLOTS_DIR = ROOT / "evaluation_plots"

# ---------------------------------------------------------------------------
# Matplotlib setup
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 9,
})

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
    """Find the target (value) column — skip metadata columns."""
    skip = {"date", "time_stamp", "predicted_ans", "features_used"}
    candidates = [c for c in df.columns if c not in skip]
    return candidates[-1] if candidates else df.columns[-1]

# ---------------------------------------------------------------------------
# Load config
# ---------------------------------------------------------------------------
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

os.makedirs(PLOTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Progress summary — show expected vs actual prediction counts
# ---------------------------------------------------------------------------
def _read_first_last_cot_ts(cot_path: Path):
    """Return (first_timestamp, last_window_offset, total_entries) from CoT log."""
    if not cot_path.exists():
        return None, None, 0
    first_ts = None
    last_wo = None
    total = 0
    try:
        with open(cot_path) as f:
            for line in f:
                obj = json.loads(line)
                if first_ts is None:
                    first_ts = obj.get("timestamp", "")[:10]
                last_wo = obj.get("window_offset", last_wo)
                total += 1
    except Exception:
        pass
    return first_ts, last_wo, total

print(f"\n{'Dataset':<14} {'Expected':>8} {'Outputs':>8} {'Baseline':>8} {'Progress':>10} {'CoT last_w':>12} {'Model source':<22}")
print("-" * 100)

progress_info: dict[str, dict] = {}

for ds in cfg["datasets"]:
    name = ds["name"]
    look_back = ds.get("look_back", 96)
    pred_window = ds.get("predicted_window", 96)
    test_path = ROOT / ds["test_csv"]

    # Expected total predictions
    if test_path.exists():
        test_lines = sum(1 for _ in open(test_path)) - 1
        expected = test_lines - look_back
    else:
        expected = 0

    o_path = OUTPUTS_DIR / name / "predictions.csv"
    b_path = BASELINE_DIR / name / "predictions.csv"
    o_lines = sum(1 for _ in open(o_path)) - 1 if o_path.exists() else 0
    b_lines = sum(1 for _ in open(b_path)) - 1 if b_path.exists() else 0

    cot_path = OUTPUTS_DIR / name / "chain_of_thought.log"
    first_ts, last_wo, cot_entries = _read_first_last_cot_ts(cot_path)

    # Model source detection
    if first_ts is None:
        model_src = "baseline only"
    elif first_ts >= "2026-06-03":
        model_src = "qwen 3.6-reasoner"
    elif first_ts >= "2026-05-28":
        model_src = "mixed (v4-flash + qwen)"
    else:
        model_src = "mixed (v4-flash + qwen)"

    # Progress
    if expected > 0:
        total_windows = expected // pred_window if pred_window else 1
        done_windows = (last_wo or 0) // pred_window + 1 if last_wo is not None else 0
        pct = min(o_lines / expected * 100, 100)
        if o_lines >= expected:
            progress = "✅ done"
        else:
            progress = f"⏳ {pct:.0f}% ({done_windows}/{total_windows}w)"
    else:
        progress = "❓"

    print(f"{name:<14} {expected:>8} {o_lines:>8} {b_lines:>8} {progress:>10} "
          f"{str(last_wo or 'N/A'):>12} {model_src:<22}")

    progress_info[name] = {
        "expected": expected, "outputs_n": o_lines, "baseline_n": b_lines,
        "progress": progress, "model_src": model_src, "cot_last_w": last_wo,
        "cot_entries": cot_entries,
    }

# ---------------------------------------------------------------------------
# Per-dataset evaluation
# ---------------------------------------------------------------------------
records: list[dict] = []

for ds in cfg["datasets"]:
    name = ds["name"]
    look_back = ds.get("look_back", 96)
    pred_window = ds.get("predicted_window", 96)

    pred_o = safe_load_csv(OUTPUTS_DIR / name / "predictions.csv")
    pred_b = safe_load_csv(BASELINE_DIR / name / "predictions.csv")
    test_path = ROOT / ds["test_csv"]
    test_df = safe_load_csv(test_path)

    if pred_o is None or pred_b is None or test_df is None:
        print(f"[SKIP] {name}: missing predictions or test data")
        continue

    # --- Align data ---
    target_col = get_target_col(test_df)
    actual = test_df[target_col].to_numpy(dtype=float)

    col_o = "predicted_ans" if "predicted_ans" in pred_o.columns else "prediction"
    if col_o not in pred_o.columns:
        col_o = [c for c in pred_o.columns if c != "time_stamp"][0]
    col_b = "predicted_ans" if "predicted_ans" in pred_b.columns else "prediction"
    if col_b not in pred_b.columns:
        col_b = [c for c in pred_b.columns if c != "time_stamp"][0]

    pred_o_vals = pred_o[col_o].to_numpy(dtype=float)
    pred_b_vals = pred_b[col_b].to_numpy(dtype=float)

    # Align to same length — use [:n] (not [-n:]) because predictions are
    # aligned from the start of the test series, not the end.
    n = min(len(pred_o_vals), len(pred_b_vals), len(actual))
    actual = actual[:n]
    pred_o_vals = pred_o_vals[:n]
    pred_b_vals = pred_b_vals[:n]

    # Clean NaNs
    mask = ~(np.isnan(actual) | np.isnan(pred_o_vals) | np.isnan(pred_b_vals))
    actual_c, p_o, p_b = actual[mask], pred_o_vals[mask], pred_b_vals[mask]

    if len(actual_c) < 10:
        print(f"[SKIP] {name}: too few valid points ({len(actual_c)})")
        continue

    # --- Metrics ---
    mae_o, mse_o, smape_o = (
        float(np.mean(np.abs(actual_c - p_o))),
        float(np.mean((actual_c - p_o) ** 2)),
        smape(actual_c, p_o),
    )
    mae_b, mse_b, smape_b = (
        float(np.mean(np.abs(actual_c - p_b))),
        float(np.mean((actual_c - p_b) ** 2)),
        smape(actual_c, p_b),
    )

    llm_active = not np.allclose(p_o, p_b, rtol=1e-5, atol=1e-5)

    records.append({
        "name": name, "n": len(actual_c),
        "mae_o": mae_o, "mse_o": mse_o, "smape_o": smape_o,
        "mae_b": mae_b, "mse_b": mse_b, "smape_b": smape_b,
        "llm_active": llm_active,
    })

    delta_mae = mae_o - mae_b
    winner = "LLM ✅" if delta_mae < -0.001 else ("Baseline" if delta_mae > 0.001 else "Tie")
    print(f"{name:<14} n={len(actual_c):>5}  "
          f"MAE_b={mae_b:>10.4f}  MAE_o={mae_o:>10.4f}  "
          f"Δ={delta_mae:>+8.4f}  sMAPE_b={smape_b:>7.2f}%  sMAPE_o={smape_o:>7.2f}%  "
          f"LLM={'active' if llm_active else 'inactive':<8}  {winner}")

    # =====================================================================
    # Figure 1: Time series — ALL windows (actual vs baseline vs LLM)
    # =====================================================================
    plot_len = len(actual_c)
    n_windows = max(1, plot_len // pred_window)
    x = np.arange(plot_len)

    # Scale figure width for long series
    fig_w = max(16, min(48, n_windows * 1.6))
    fig, axes = plt.subplots(2, 2, figsize=(fig_w, 10))
    fig.suptitle(f"{name} — Baseline vs LLM Predictions", fontweight="bold", y=0.98)

    # --- (A) Full time series ---
    ax = axes[0, 0]
    lw_ts = 0.6 if plot_len > 2000 else 1.2
    ax.plot(x, actual_c, "k-", alpha=0.35, lw=lw_ts, label="Actual")
    ax.plot(x, p_b, "#2196F3", alpha=0.7, lw=max(0.4, lw_ts*0.8), label="Baseline (deterministic)")
    if llm_active:
        ax.plot(x, p_o, "#FF5722", alpha=0.7, lw=max(0.4, lw_ts*0.8), label="LLM")
        # Shade windows where LLM differs
        for w in range(n_windows):
            start, end = w * pred_window, min((w + 1) * pred_window, plot_len)
            w_diff = np.mean(np.abs(p_o[start:end] - p_b[start:end]))
            if w_diff > 1e-5:
                ax.axvspan(start, end, alpha=0.08, color="#FF5722")
    ax.set_title(f"Predicted vs Actual (all {n_windows} windows, {plot_len} steps)")
    ax.set_xlabel("Time step")
    ax.set_ylabel(target_col)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)

    # --- (B) LLM residuals over time ---
    ax = axes[0, 1]
    # Load residual emissions from LLM (residuals.csv has time_stamp, prediction, window_offset, horizon_index)
    res_csv = OUTPUTS_DIR / name / "residuals.csv"
    if res_csv.exists():
        try:
            res_df = pd.read_csv(res_csv)
            if "time_stamp" in res_df.columns and "prediction" in res_df.columns:
                res_df["time_stamp"] = pd.to_datetime(res_df["time_stamp"])
                # Dedup by time_stamp (keep last emission for each timestamp)
                res_unique = res_df.sort_values("time_stamp").drop_duplicates(subset=["time_stamp"], keep="last")
                res_vals = res_unique["prediction"].to_numpy(dtype=float)
                # Align to test data: use first min(len(res), len(actual_c)) points
                n_res = min(len(res_vals), len(actual_c))
                res_aligned = res_vals[:n_res]
                x_res = np.arange(n_res)
                lw_res = 0.5 if n_res > 2000 else 1.0
                ax.plot(x_res, res_aligned, "#7B1FA2", alpha=0.7, lw=lw_res, label="LLM residual")
                ax.axhline(0, color="k", lw=0.5, ls="--")
                ax.set_title(f"LLM Residual Emissions ({n_res} steps)")
                ax.set_xlabel("Time step")
                ax.set_ylabel("Residual value")
                ax.legend(fontsize=7)
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, "No residual data", ha="center", va="center", transform=ax.transAxes)
                ax.set_title("LLM Residuals (unavailable)")
        except Exception:
            ax.text(0.5, 0.5, "Failed to load residuals", ha="center", va="center", transform=ax.transAxes)
            ax.set_title("LLM Residuals (load error)")
    else:
        ax.text(0.5, 0.5, "residuals.csv not found", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("LLM Residuals (not available)")

    # --- (C) Error distribution ---
    ax = axes[1, 0]
    err_o = p_o - actual_c
    err_b = p_b - actual_c
    bins = min(100, max(30, len(actual_c) // 30))
    ax.hist(err_b, bins=bins, alpha=0.5, color="#2196F3", density=True, label="Baseline")
    if llm_active:
        ax.hist(err_o, bins=bins, alpha=0.5, color="#FF5722", density=True, label="LLM")
    ax.axvline(0, color="k", lw=0.5, ls="--")
    ax.set_title(f"Error Distribution (n={len(actual_c)})")
    ax.set_xlabel("Error")
    ax.set_ylabel("Density")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # --- (D) Per-window MAE comparison ---
    ax = axes[1, 1]
    window_mae_b, window_mae_o = [], []
    window_labels = []
    for w in range(n_windows):
        s, e = w * pred_window, min((w + 1) * pred_window, plot_len)
        window_mae_b.append(float(np.mean(np.abs(p_b[s:e] - actual_c[s:e]))))
        window_mae_o.append(float(np.mean(np.abs(p_o[s:e] - actual_c[s:e]))))
        window_labels.append(f"W{w}")
    xw = np.arange(n_windows)
    ww = max(0.2, min(0.35, 0.8 / n_windows * 10))  # narrower bars for many windows
    ax.bar(xw - ww/2, window_mae_b, ww, color="#2196F3", alpha=0.8, label="Baseline MAE")
    if llm_active:
        ax.bar(xw + ww/2, window_mae_o, ww, color="#FF5722", alpha=0.8, label="LLM MAE")
    # Sparse x-ticks when many windows to avoid label overlap
    if n_windows <= 24:
        ax.set_xticks(xw)
        ax.set_xticklabels(window_labels, fontsize=6, rotation=45)
    else:
        tick_step = max(1, n_windows // 20)
        tick_positions = xw[::tick_step]
        tick_labels = [window_labels[i] for i in range(0, n_windows, tick_step)]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, fontsize=5, rotation=60)
    ax.set_title(f"Per-Window MAE (all {n_windows} windows, size={pred_window})")
    ax.set_ylabel("MAE")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(PLOTS_DIR / f"{name}_comparison.png")
    plt.close(fig)
    print(f"  → saved {name}_comparison.png")

    # =====================================================================
    # Figure 2 (LLM only): LLM adjustment analysis
    # =====================================================================
    if llm_active:
        fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5))
        fig2.suptitle(f"{name} — LLM Adjustment Analysis", fontweight="bold")

        adjustment = p_o - p_b
        abs_adj = np.abs(adjustment)
        err_reduction = np.abs(err_b) - np.abs(err_o)  # + = LLM improved

        # Adjustment magnitude distribution
        ax = axes2[0]
        ax.hist(adjustment, bins=min(80, len(adjustment)//20 + 20), color="#FF5722", alpha=0.7, edgecolor="white", lw=0.3)
        ax.axvline(0, color="k", lw=0.8, ls="--")
        ax.set_title(f"LLM Adjustment Magnitude (n={len(adjustment)})")
        ax.set_xlabel("Δ (LLM − Baseline)")
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.3)

        # Adjustment vs baseline error
        ax = axes2[1]
        ax.scatter(err_b, adjustment, c="#FF5722", alpha=0.15, s=4, edgecolors="none")
        ax.axhline(0, color="k", lw=0.5, ls="--")
        ax.axvline(0, color="k", lw=0.5, ls="--")
        ax.set_xlabel("Baseline Error")
        ax.set_ylabel("LLM Adjustment")
        ax.set_title(f"LLM Adjustment vs Baseline Error (n={len(adjustment)})")
        ax.grid(True, alpha=0.3)

        # Improvement histogram — downsample for display when too many points
        ax = axes2[2]
        if len(err_reduction) > 2000:
            # Aggregate by window for readability
            win_improve = []
            for w in range(n_windows):
                s, e = w * pred_window, min((w + 1) * pred_window, plot_len)
                win_improve.append(float(np.mean(err_reduction[s:e])))
            colors = ["#4CAF50" if v > 0 else "#F44336" for v in win_improve]
            ax.bar(range(len(win_improve)), win_improve, color=colors, width=0.8, alpha=0.7)
            ax.set_title(f"Per-Window Error Reduction (green=LLM better)\nMean: {np.mean(err_reduction):+.4f}")
            ax.set_xlabel("Window index")
        else:
            colors = ["#4CAF50" if v > 0 else "#F44336" for v in err_reduction]
            ax.bar(range(len(err_reduction)), err_reduction, color=colors, width=1.0, alpha=0.7)
            ax.set_title(f"Error Reduction (green=LLM better, red=worse)\nMean: {np.mean(err_reduction):+.4f}")
            ax.set_xlabel("Time step")
        ax.set_ylabel("|Baseline Err| − |LLM Err|")
        ax.grid(True, alpha=0.3)

        fig2.tight_layout(rect=[0, 0, 1, 0.93])
        fig2.savefig(PLOTS_DIR / f"{name}_llm_analysis.png")
        plt.close(fig2)
        print(f"  → saved {name}_llm_analysis.png")

# =========================================================================
# Summary figure: bar chart comparing all datasets
# =========================================================================
if records:
    names = [r["name"] for r in records]
    n_ds = len(names)

    fig3, axes3 = plt.subplots(1, 3, figsize=(max(14, n_ds * 0.8), 6))
    fig3.suptitle("Baseline vs LLM — Summary Comparison", fontweight="bold", y=1.01)

    x_idx = np.arange(n_ds)
    w = 0.35

    for ax_idx, (metric, label) in enumerate([
        ("mae", "MAE"), ("mse", "MSE"), ("smape", "sMAPE (%)")
    ]):
        ax = axes3[ax_idx]
        vals_b = [r[f"{metric}_b"] for r in records]
        vals_o = [r[f"{metric}_o"] for r in records]
        ax.bar(x_idx - w/2, vals_b, w, color="#2196F3", alpha=0.85, label="Baseline")
        ax.bar(x_idx + w/2, vals_o, w, color="#FF5722", alpha=0.85, label="LLM")
        ax.set_xticks(x_idx)
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
        ax.set_title(label)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, axis="y")

        # Annotate LLM-active markers
        for i, r in enumerate(records):
            if r["llm_active"]:
                better = vals_o[i] < vals_b[i]
                color = "#4CAF50" if better else "#F44336"
                ax.annotate("●" if better else "○", (x_idx[i] + w/2, vals_o[i]),
                            textcoords="offset points", xytext=(0, 5),
                            fontsize=12, color=color, ha="center")

    fig3.tight_layout()
    fig3.savefig(PLOTS_DIR / "00_summary_comparison.png")
    print(f"\n→ saved 00_summary_comparison.png")

    # =====================================================================
    # Summary table
    # =====================================================================
    print(f"\n{'Dataset':<14} {'Exp':>6} {'Now':>6} {'MAE_b':>10} {'MAE_o':>10} {'Δ_MAE':>10} "
          f"{'MSE_b':>10} {'MSE_o':>10} {'sMAPE_b':>8} {'sMAPE_o':>8} {'Winner':>10} {'Progress':>12}")
    print("-" * 130)
    for r in records:
        name = r["name"]
        delta = r["mae_o"] - r["mae_b"]
        w = "LLM ✅" if delta < -0.001 else ("Baseline" if delta > 0.001 else "Tie")
        if not r["llm_active"]:
            w = "(same)"
        sign = "+" if delta > 0 else ""
        pinfo = progress_info.get(name, {})
        exp = pinfo.get("expected", "?")
        now = pinfo.get("outputs_n", r["n"])
        prog = pinfo.get("progress", "")
        print(f"{name:<14} {str(exp):>6} {str(now):>6} {r['mae_b']:>10.4f} {r['mae_o']:>10.4f} {sign}{delta:>9.4f} "
              f"{r['mse_b']:>10.4f} {r['mse_o']:>10.4f} {r['smape_b']:>7.2f}% {r['smape_o']:>7.2f}% {w:>10} {prog:>12}")

    # =====================================================================
    # Delta summary figure
    # =====================================================================
    fig4, ax4 = plt.subplots(figsize=(max(12, n_ds * 0.7), 5))
    deltas = [r["mae_o"] - r["mae_b"] for r in records]
    colors = ["#4CAF50" if d < 0 else ("#F44336" if d > 0 else "#9E9E9E") for d in deltas]
    active = [r["llm_active"] for r in records]
    edge_colors = ["#FF5722" if a else "#BDBDBD" for a in active]
    linewidths = [2 if a else 1 for a in active]

    bars = ax4.bar(range(n_ds), deltas, color=colors, edgecolor=edge_colors,
                   linewidth=linewidths, alpha=0.85)
    ax4.axhline(0, color="k", lw=0.8)
    ax4.set_xticks(range(n_ds))
    ax4.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax4.set_ylabel("Δ MAE (LLM − Baseline)")
    ax4.set_title("LLM Impact on MAE (negative = LLM better, orange border = LLM active)")
    ax4.grid(True, alpha=0.3, axis="y")

    # Add value labels
    for i, (d, a) in enumerate(zip(deltas, active)):
        label = f"{d:+.3f}" if a else "N/A"
        va = "bottom" if d >= 0 else "top"
        ax4.text(i, d, label, ha="center", va=va, fontsize=7,
                 fontweight="bold" if a else "normal")

    fig4.tight_layout()
    fig4.savefig(PLOTS_DIR / "00_delta_summary.png")
    print("→ saved 00_delta_summary.png")

# =========================================================================
# Prophet Decomposition Visualization
# =========================================================================
print(f"\n{'='*70}")
print("Prophet decomposition visualization")
print(f"{'='*70}")

for ds in cfg["datasets"]:
    name = ds["name"]
    decomp_path = OUTPUTS_DIR / name / "prophet_decomposition.json"

    if not decomp_path.exists():
        print(f"  [{name}] No prophet_decomposition.json — skip")
        continue

    try:
        with open(decomp_path) as f:
            decomp = json.load(f)
    except Exception as exc:
        print(f"  [{name}] Failed to read: {exc} — skip")
        continue

    if not decomp.get("decomposition_enabled"):
        print(f"  [{name}] Decomposition not enabled — skip")
        continue

    train_trend = decomp.get("train_trend") or []
    train_seas = decomp.get("train_seasonality") or []
    train_resid = decomp.get("train_residual") or []
    train_ts_strs = decomp.get("train_timestamps") or []

    test_trend = decomp.get("test_trend") or []
    test_seas = decomp.get("test_seasonality") or []
    test_ts_strs = decomp.get("test_timestamps") or []

    if not train_trend:
        print(f"  [{name}] Empty decomposition — skip")
        continue

    n_train = len(train_trend)
    train_y = [train_trend[i] + train_seas[i] + train_resid[i] for i in range(n_train)]
    train_trend_plus_seas = [train_trend[i] + train_seas[i] for i in range(n_train)]

    # ---- Load test set ground truth ----
    test_y = None
    if test_ts_strs:
        test_path = ROOT / ds["test_csv"]
        test_df = safe_load_csv(test_path)
        if test_df is not None and "date" in test_df.columns:
            test_df["date"] = pd.to_datetime(test_df["date"])
            target_col = get_target_col(test_df)
            # Align by timestamp
            ts_to_val = dict(zip(
                test_df["date"].dt.strftime("%Y-%m-%dT%H:%M:%S"),
                test_df[target_col].to_numpy(dtype=float),
            ))
            # Also try without seconds
            ts_to_val2 = dict(zip(
                test_df["date"].dt.strftime("%Y-%m-%dT%H:%M"),
                test_df[target_col].to_numpy(dtype=float),
            ))
            test_y = []
            for ts_str in test_ts_strs:
                val = ts_to_val.get(ts_str[:19])  # truncate to seconds
                if val is None:
                    val = ts_to_val2.get(ts_str[:16])  # try minute precision
                if val is None:
                    val = ts_to_val.get(ts_str[:10])  # try date only
                test_y.append(val)
            test_y = [v for v in test_y if v is not None]
        elif test_df is not None:
            # No date column: align by position
            target_col = get_target_col(test_df)
            raw = test_df[target_col].to_numpy(dtype=float)
            raw_clean = raw[~np.isnan(raw)]
            n_match = min(len(test_trend), len(raw_clean))
            test_y = raw_clean[:n_match].tolist()

    has_test = test_y is not None and len(test_y) > 0
    n_test = len(test_y) if has_test else 0
    n_test_decomp = len(test_trend)

    # Align test decomposition with test_y
    if has_test:
        n_align = min(n_test, n_test_decomp)
        test_y = test_y[:n_align]
        test_trend_aligned = test_trend[:n_align]
        test_seas_aligned = test_seas[:n_align]
        test_trend_plus_seas = [test_trend_aligned[i] + test_seas_aligned[i] for i in range(n_align)]
        n_test = n_align
    else:
        # No ground truth; still show forecast components if available
        n_test = n_test_decomp

    # -------------------------------------------------------------------
    # Figure layout: always 2×2
    #   Top-left:  train — trend vs actual
    #   Top-right: train — trend+seasonality vs actual
    #   Bot-left:  test  — trend vs actual
    #   Bot-right: test  — trend+seasonality vs actual
    # -------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(20, 10))
    (ax_tr_t, ax_tr_ts), (ax_te_t, ax_te_ts) = axes
    fig.suptitle(f"{name} — Prophet Decomposition vs Actual", fontweight="bold", y=0.99)

    # ---- Helper: downsample index for readability ----
    def _ds_idx(n, limit=3000):
        if n <= limit:
            return np.arange(n)
        step = max(1, n // limit)
        return np.arange(0, n, step)

    # =================================================================
    # (A) Training: Trend vs Actual
    # =================================================================
    idx_tr = _ds_idx(n_train)
    x_tr = np.arange(n_train)

    ax_tr_t.plot(x_tr[idx_tr], [train_y[i] for i in idx_tr], "#90A4AE", alpha=0.5, lw=0.5, label="Actual y(t)")
    ax_tr_t.plot(x_tr, train_trend, "#E65100", alpha=0.9, lw=1.3, label="Prophet Trend(t)")
    ax_tr_t.set_title(f"Training Set: Trend vs Actual (n={n_train})")
    ax_tr_t.set_xlabel("Training time step")
    ax_tr_t.set_ylabel("Value")
    ax_tr_t.legend(fontsize=8, loc="upper left")
    ax_tr_t.grid(True, alpha=0.3)

    # Annotate fit quality
    trend_mae = np.mean(np.abs(np.array(train_trend) - np.array(train_y)))
    trend_corr = np.corrcoef(train_trend, train_y)[0, 1]
    ax_tr_t.text(0.02, 0.97,
                  f"Trend vs Actual: MAE={trend_mae:.4f}   corr={trend_corr:.4f}",
                  transform=ax_tr_t.transAxes, fontsize=7, va="top", ha="left",
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.7))

    # =================================================================
    # (B) Training: Trend + Seasonality vs Actual
    # =================================================================
    ax_tr_ts.plot(x_tr[idx_tr], [train_y[i] for i in idx_tr], "#90A4AE", alpha=0.5, lw=0.5, label="Actual y(t)")
    ax_tr_ts.plot(x_tr, train_trend_plus_seas, "#1565C0", alpha=0.85, lw=1.0, label="Prophet Trend(t) + Seasonality(t)")
    ax_tr_ts.set_title(f"Training Set: Trend + Seasonality vs Actual (n={n_train})")
    ax_tr_ts.set_xlabel("Training time step")
    ax_tr_ts.set_ylabel("Value")
    ax_tr_ts.legend(fontsize=8, loc="upper left")
    ax_tr_ts.grid(True, alpha=0.3)

    ts_mae = np.mean(np.abs(np.array(train_trend_plus_seas) - np.array(train_y)))
    ts_corr = np.corrcoef(train_trend_plus_seas, train_y)[0, 1]
    ax_tr_ts.text(0.02, 0.97,
                   f"Trend+Seas vs Actual: MAE={ts_mae:.4f}   corr={ts_corr:.4f}",
                   transform=ax_tr_ts.transAxes, fontsize=7, va="top", ha="left",
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="#E3F2FD", alpha=0.7))

    # ---- Load per-window combined trend (global + local) if available ----
    combined_trend_aligned = None
    combined_seas_aligned = None
    components_path = OUTPUTS_DIR / name / "prophet_components.csv"
    if components_path.exists():
        try:
            comp_df = pd.read_csv(components_path)
            if "time_stamp" in comp_df.columns and "trend_combined" in comp_df.columns:
                comp_df["time_stamp"] = pd.to_datetime(comp_df["time_stamp"])
                # Deduplicate by timestamp (keep last emission per timestamp)
                comp_df = comp_df.sort_values("time_stamp", kind="mergesort")
                comp_df = comp_df.drop_duplicates(subset=["time_stamp"], keep="last")
                comp_df = comp_df.sort_values("time_stamp").reset_index(drop=True)
                # Align with test_y timestamps
                if test_ts_strs:
                    ts_to_trend = dict(zip(
                        comp_df["time_stamp"].dt.strftime("%Y-%m-%dT%H:%M:%S"),
                        comp_df["trend_combined"].to_numpy(dtype=float),
                    ))
                    ts_to_seas = dict(zip(
                        comp_df["time_stamp"].dt.strftime("%Y-%m-%dT%H:%M:%S"),
                        comp_df["seasonality"].to_numpy(dtype=float),
                    ))
                    combined_trend_aligned = []
                    combined_seas_aligned = []
                    for ts_str in test_ts_strs[:n_test]:
                        val_t = ts_to_trend.get(ts_str[:19])
                        val_s = ts_to_seas.get(ts_str[:19])
                        combined_trend_aligned.append(float(val_t) if val_t is not None else None)
                        combined_seas_aligned.append(float(val_s) if val_s is not None else None)
                    # Fill gaps with global trend as fallback
                    for i in range(len(combined_trend_aligned)):
                        if combined_trend_aligned[i] is None:
                            combined_trend_aligned[i] = test_trend_aligned[i] if i < len(test_trend_aligned) else 0.0
                        if combined_seas_aligned[i] is None:
                            combined_seas_aligned[i] = test_seas_aligned[i] if i < len(test_seas_aligned) else 0.0
                    has_combined = True
                else:
                    has_combined = False
            else:
                has_combined = False
        except Exception:
            has_combined = False
    else:
        has_combined = False

    print(f"  [{name}] train={n_train} points, test={n_test} points, combined_trend={'yes' if has_combined else 'no'} → generating plots...")

    # -------------------------------------------------------------------
    # Figure layout: always 2×2
    # -------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(20, 10))
    (ax_tr_t, ax_tr_ts), (ax_te_t, ax_te_ts) = axes
    fig.suptitle(f"{name} — Prophet Decomposition vs Actual", fontweight="bold", y=0.99)

    def _ds_idx(n, limit=3000):
        if n <= limit: return np.arange(n)
        step = max(1, n // limit)
        return np.arange(0, n, step)

    # =================================================================
    # (A) Training: Trend vs Actual
    # =================================================================
    idx_tr = _ds_idx(n_train)
    x_tr = np.arange(n_train)

    ax_tr_t.plot(x_tr[idx_tr], [train_y[i] for i in idx_tr], "#90A4AE", alpha=0.5, lw=0.5, label="Actual y(t)")
    ax_tr_t.plot(x_tr, train_trend, "#E65100", alpha=0.9, lw=1.3, label="Prophet Trend(t)")
    ax_tr_t.set_title(f"Training Set: Trend vs Actual (n={n_train})")
    ax_tr_t.set_xlabel("Training time step")
    ax_tr_t.set_ylabel("Value")
    ax_tr_t.legend(fontsize=8, loc="upper left")
    ax_tr_t.grid(True, alpha=0.3)
    trend_mae = np.mean(np.abs(np.array(train_trend) - np.array(train_y)))
    trend_corr = np.corrcoef(train_trend, train_y)[0, 1]
    ax_tr_t.text(0.02, 0.97, f"Trend vs Actual: MAE={trend_mae:.4f}   corr={trend_corr:.4f}",
                  transform=ax_tr_t.transAxes, fontsize=7, va="top", ha="left",
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.7))

    # =================================================================
    # (B) Training: Trend + Seasonality vs Actual
    # =================================================================
    ax_tr_ts.plot(x_tr[idx_tr], [train_y[i] for i in idx_tr], "#90A4AE", alpha=0.5, lw=0.5, label="Actual y(t)")
    ax_tr_ts.plot(x_tr, train_trend_plus_seas, "#1565C0", alpha=0.85, lw=1.0, label="Prophet Trend(t) + Seasonality(t)")
    ax_tr_ts.set_title(f"Training Set: Trend + Seasonality vs Actual (n={n_train})")
    ax_tr_ts.set_xlabel("Training time step")
    ax_tr_ts.set_ylabel("Value")
    ax_tr_ts.legend(fontsize=8, loc="upper left")
    ax_tr_ts.grid(True, alpha=0.3)
    ts_mae = np.mean(np.abs(np.array(train_trend_plus_seas) - np.array(train_y)))
    ts_corr = np.corrcoef(train_trend_plus_seas, train_y)[0, 1]
    ax_tr_ts.text(0.02, 0.97, f"Trend+Seas vs Actual: MAE={ts_mae:.4f}   corr={ts_corr:.4f}",
                   transform=ax_tr_ts.transAxes, fontsize=7, va="top", ha="left",
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="#E3F2FD", alpha=0.7))

    # =================================================================
    # (C) Test: Trend vs Actual
    #   Shows global trend (dashed, from full train.csv) +
    #   combined trend (solid, global + per-window local correction)
    # =================================================================
    if has_test:
        idx_te = _ds_idx(n_test)
        x_te = np.arange(n_test)

        # Pre-compute all metrics before plotting
        trend_test_mae = np.mean(np.abs(np.array(test_trend_aligned) - np.array(test_y)))
        trend_test_corr = np.corrcoef(test_trend_aligned, test_y)[0, 1] if n_test >= 2 else 0.0
        ts_test_mae = np.mean(np.abs(np.array(test_trend_plus_seas) - np.array(test_y)))
        ts_test_corr = np.corrcoef(test_trend_plus_seas, test_y)[0, 1] if n_test >= 2 else 0.0

        ax_te_t.plot(x_te[idx_te], [test_y[i] for i in idx_te], "#424242", alpha=0.6, lw=0.7, label="Actual y(t)")
        ax_te_t.plot(x_te, test_trend_aligned, "#78909C", alpha=0.8, lw=1.2, label="Global trend only")
        if has_combined and combined_trend_aligned:
            ax_te_t.plot(x_te, combined_trend_aligned, "#E65100", alpha=0.9, lw=1.4, label="Combined trend (global + local)")
            comb_t_mae = np.mean(np.abs(np.array(combined_trend_aligned) - np.array(test_y)))
            comb_t_corr = np.corrcoef(combined_trend_aligned, test_y)[0, 1] if n_test >= 2 else 0.0
            label_t = (f"Global Trend MAE={trend_test_mae:.4f}   corr={trend_test_corr:.4f}\n"
                       f"Combined Trend MAE={comb_t_mae:.4f}   corr={comb_t_corr:.4f}")
        ax_te_t.set_title(f"Test Set: Trend Forecast vs Actual (n={n_test})")
        ax_te_t.set_xlabel("Test time step")
        ax_te_t.set_ylabel("Value")
        ax_te_t.legend(fontsize=7.5, loc="upper left")
        ax_te_t.grid(True, alpha=0.3)

        if has_combined and combined_trend_aligned:
            ax_te_t.text(0.02, 0.97, label_t,
                          transform=ax_te_t.transAxes, fontsize=7, va="top", ha="left",
                          bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.85))
        else:
            ax_te_t.text(0.02, 0.97, f"Global Trend MAE={trend_test_mae:.4f}   corr={trend_test_corr:.4f}",
                          transform=ax_te_t.transAxes, fontsize=7, va="top", ha="left",
                          bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.7))

        # =================================================================
        # (D) Test: Trend + Seasonality vs Actual
        # =================================================================
        ax_te_ts.plot(x_te[idx_te], [test_y[i] for i in idx_te], "#424242", alpha=0.6, lw=0.7, label="Actual y(t)")
        ax_te_ts.plot(x_te, test_trend_plus_seas, "#64B5F6", alpha=0.75, lw=1.2, label="Global Trend + Seasonality")
        if has_combined and combined_trend_aligned and combined_seas_aligned:
            comb_ts = [combined_trend_aligned[i] + combined_seas_aligned[i] for i in range(n_test)]
            ax_te_ts.plot(x_te, comb_ts, "#1565C0", alpha=0.85, lw=1.2, label="Combined Trend + Seasonality")
            comb_ts_mae = np.mean(np.abs(np.array(comb_ts) - np.array(test_y)))
            comb_ts_corr = np.corrcoef(comb_ts, test_y)[0, 1] if n_test >= 2 else 0.0
            label_ts = (f"Global T+S MAE={ts_test_mae:.4f}   corr={ts_test_corr:.4f}\n"
                        f"Combined T+S MAE={comb_ts_mae:.4f}   corr={comb_ts_corr:.4f}")
        ax_te_ts.set_title(f"Test Set: Trend + Seasonality Forecast vs Actual (n={n_test})")
        ax_te_ts.set_xlabel("Test time step")
        ax_te_ts.set_ylabel("Value")
        ax_te_ts.legend(fontsize=7.5, loc="upper left")
        ax_te_ts.grid(True, alpha=0.3)

        if has_combined and combined_trend_aligned:
            ax_te_ts.text(0.02, 0.97, label_ts,
                           transform=ax_te_ts.transAxes, fontsize=7, va="top", ha="left",
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="#E3F2FD", alpha=0.85))
        else:
            ax_te_ts.text(0.02, 0.97, f"Global T+S vs Actual: MAE={ts_test_mae:.4f}   corr={ts_test_corr:.4f}",
                           transform=ax_te_ts.transAxes, fontsize=7, va="top", ha="left",
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="#E3F2FD", alpha=0.7))
    else:
        # No test ground truth: just show forecast components
        idx_te = _ds_idx(n_test_decomp)
        x_te = np.arange(n_test_decomp)

        ax_te_t.plot(x_te[idx_te], [test_trend[i] for i in idx_te], "#E65100", alpha=0.9, lw=1.2, label="Trend forecast")
        ax_te_t.set_title(f"Test Set: Trend Forecast (no ground truth) (n={n_test_decomp})")
        ax_te_t.set_xlabel("Test time step")
        ax_te_t.set_ylabel("Value")
        ax_te_t.legend(fontsize=8)
        ax_te_t.grid(True, alpha=0.3)

        test_composite = [test_trend[i] + test_seas[i] for i in range(n_test_decomp)]
        ax_te_ts.plot(x_te[idx_te], [test_trend[i] for i in idx_te], "#E65100", alpha=0.6, lw=0.8, label="Trend")
        ax_te_ts.plot(x_te[idx_te], [test_seas[i] for i in idx_te], "#2E7D32", alpha=0.6, lw=0.8, label="Seasonality")
        ax_te_ts.plot(x_te[idx_te], [test_composite[i] for i in idx_te], "#1565C0", alpha=0.9, lw=1.2, label="Trend + Seasonality")
        ax_te_ts.set_title(f"Test Set: Forecast Components (no ground truth) (n={n_test_decomp})")
        ax_te_ts.set_xlabel("Test time step")
        ax_te_ts.set_ylabel("Value")
        ax_te_ts.legend(fontsize=8)
        ax_te_ts.grid(True, alpha=0.3)

    # -------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = PLOTS_DIR / f"{name}_prophet_decomposition.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  → saved {name}_prophet_decomposition.png")

print(f"\n✅ All plots saved to: {PLOTS_DIR}/")

# =========================================================================
# Final summary table (same format as result.txt)
# =========================================================================
if records:
    print(f"\n=== Experiment Summary ===")
    print(f"{'dataset':>14} {'MSE':>12} {'MAE':>10} {'sMAPE':>8}  model")
    print("-" * 52)
    for r in records:
        print(f"{r['name']:>14} {r['mse_o']:>12.6f} {r['mae_o']:>10.6f} {r['smape_o']:>8.6f}   LLM")
    print()
