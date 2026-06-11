from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

from ..data_loader import TIME_COL, TARGET_COL
from ..models.base import (
    ForecastModel,
    ProphetModel,
    configure_deep_learning_runtime,
    get_default_models,
)
if TYPE_CHECKING:
    from ..config import DatasetConfig
from ..utils.time import generate_future_timestamps


def forecast_with_model(
    model_name: str,
    last_window: np.ndarray,
    h: int,
    season_length: Optional[int],
    dataset: Optional["DatasetConfig"] = None,
    **kwargs
) -> np.ndarray:
    if dataset is not None:
        configure_deep_learning_runtime(
            dataset.checkpoints,
            dataset.predicted_window,
            dataset_name=getattr(dataset, "name", None),
        )

    models = {m.alias: m for m in get_default_models()}
    if model_name not in models:
        model_name = "SeasonalNaive"
    model = models[model_name]

    timestamps = kwargs.get("timestamps", None)
    future_timestamps = generate_future_timestamps(timestamps.iloc[-1], h, pd.infer_freq(pd.to_datetime(timestamps)))

    # Pass timestamps through to fit; keep predict unchanged
    model.fit(last_window, season_length=season_length, timestamps=timestamps)
    return model.predict(h, future_timestamps=future_timestamps)


def save_predictions_csv(out_path: str, timestamps: List[pd.Timestamp], preds: np.ndarray) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df = pd.DataFrame({"time_stamp": timestamps, "predicted_ans": preds})
    df.to_csv(out_path, index=False)


def prophet_decompose_series(
    train_df: pd.DataFrame,
    target_col: str,
    ds_name: str,
    output_dir: str,
    season_length: int,
    frequency: str,
    test_df: Optional[pd.DataFrame] = None,
) -> Optional[dict]:
    """
    Fit Prophet on the full training set and decompose into trend + seasonality.

    1. Creates a ProphetModel, fits on the full training set.
    2. Computes trend, seasonality, and residual for every training point.
    3. If test_df is provided, also computes Prophet forecast components for all test timestamps.
    4. Saves decomposition to outputs/<ds>/prophet_decomposition.json

    Returns:
        dict with keys: train_trend, train_seasonality, train_residual, train_timestamps,
        test_trend, test_seasonality, test_timestamps (if test_df provided), model_params.
        Returns None if Prophet fitting fails (fallback to raw mode).
    """
    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

    ds_out_dir = os.path.join(output_dir, ds_name)
    os.makedirs(ds_out_dir, exist_ok=True)

    train_y = train_df[target_col].to_numpy(dtype=float)
    train_ts = pd.to_datetime(train_df[TIME_COL])

    # Create and fit Prophet model
    model = ProphetModel()
    try:
        model.fit(train_y, season_length=season_length, timestamps=train_ts)
    except Exception as exc:
        print(f"[warn] Prophet decomposition failed for dataset '{ds_name}' during fit: {exc}. Falling back to raw mode.")
        return None

    # Decompose training set
    try:
        decomp = model.decompose(timestamps=train_ts)
    except Exception as exc:
        print(f"[warn] Prophet decomposition failed for dataset '{ds_name}' during decompose: {exc}. Falling back to raw mode.")
        return None

    train_trend = decomp["trend"].tolist()
    train_seasonality = decomp["seasonality"].tolist()
    train_residual = (train_y - np.asarray(train_trend) - np.asarray(train_seasonality)).tolist()

    result = {
        "train_trend": train_trend,
        "train_seasonality": train_seasonality,
        "train_residual": train_residual,
        "train_timestamps": [ts.isoformat() for ts in train_ts],
        "model_params": {
            "yearly_seasonality": model.yearly_seasonality,
            "weekly_seasonality": model.weekly_seasonality,
            "daily_seasonality": model.daily_seasonality,
        },
        "decomposition_enabled": True,
    }

    # If test_df provided, pre-compute Prophet components for entire test set
    if test_df is not None:
        try:
            test_ts = pd.to_datetime(test_df[TIME_COL])
            test_decomp = model.decompose(timestamps=test_ts)
            result["test_trend"] = test_decomp["trend"].tolist()
            result["test_seasonality"] = test_decomp["seasonality"].tolist()
            result["test_timestamps"] = [ts.isoformat() for ts in test_ts]
        except Exception as exc:
            print(f"[warn] Prophet test-set decomposition failed for dataset '{ds_name}': {exc}. Test components will be unavailable.")

    # Save to disk
    out_path = os.path.join(ds_out_dir, "prophet_decomposition.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"[info] Prophet decomposition saved for dataset '{ds_name}' -> {out_path}")
    return result


def get_prophet_forecast_components(
    ds_out_dir: str,
    timestamps: list,
) -> dict:
    """
    Look up Prophet trend + seasonality for given timestamps from pre-computed decomposition.

    Loads prophet_decomposition.json from ds_out_dir, matches the given timestamps
    against test_timestamps (or train_timestamps as fallback).

    Args:
        ds_out_dir: Path to outputs/<dataset>/
        timestamps: List of timestamp strings or pd.Timestamp to look up

    Returns:
        {"trend": [float, ...], "seasonality": [float, ...]}
        Empty lists if decomposition is unavailable.
    """
    decomp_path = os.path.join(ds_out_dir, "prophet_decomposition.json")
    if not os.path.exists(decomp_path):
        return {"trend": [], "seasonality": []}

    try:
        with open(decomp_path, "r", encoding="utf-8") as f:
            decomp = json.load(f)
    except Exception:
        return {"trend": [], "seasonality": []}

    if not decomp or not decomp.get("decomposition_enabled"):
        return {"trend": [], "seasonality": []}

    # Normalize timestamps to ISO strings for lookup
    ts_strs = []
    for ts in timestamps:
        if isinstance(ts, pd.Timestamp):
            ts_strs.append(ts.isoformat())
        elif isinstance(ts, str):
            ts_strs.append(ts)
        else:
            ts_strs.append(str(ts))

    # Build lookup maps from test decomposition
    test_ts_list = decomp.get("test_timestamps") or []
    test_trend = decomp.get("test_trend") or []
    test_seasonality = decomp.get("test_seasonality") or []

    ts_to_trend = dict(zip(test_ts_list, test_trend)) if len(test_ts_list) == len(test_trend) else {}
    ts_to_seas = dict(zip(test_ts_list, test_seasonality)) if len(test_ts_list) == len(test_seasonality) else {}

    # Also build train lookup as fallback
    train_ts_list = decomp.get("train_timestamps") or []
    train_trend = decomp.get("train_trend") or []
    train_seasonality = decomp.get("train_seasonality") or []
    train_to_trend = dict(zip(train_ts_list, train_trend)) if len(train_ts_list) == len(train_trend) else {}
    train_to_seas = dict(zip(train_ts_list, train_seasonality)) if len(train_ts_list) == len(train_seasonality) else {}

    trend_out = []
    seasonality_out = []
    for ts_str in ts_strs:
        # Try test lookup first, then train lookup
        if ts_str in ts_to_trend:
            trend_out.append(float(ts_to_trend[ts_str]))
            seasonality_out.append(float(ts_to_seas.get(ts_str, 0.0)))
        elif ts_str in train_to_trend:
            trend_out.append(float(train_to_trend[ts_str]))
            seasonality_out.append(float(train_to_seas.get(ts_str, 0.0)))
        else:
            # Timestamp not found — use 0 as fallback (shouldn't happen in normal flow)
            trend_out.append(0.0)
            seasonality_out.append(0.0)

    return {"trend": trend_out, "seasonality": seasonality_out}


def forecast_local_trend(
    detrended_lookback: np.ndarray,
    forecast_horizon: int,
    prev_slope: Optional[float] = None,
    prev_boundary: Optional[float] = None,
    ema_alpha: float = 0.05,
    cp_threshold: float = 0.0,
    decay: float = 0.03,
    continuity_scale: float = 1.0,
) -> np.ndarray:
    """Thin wrapper around _forecast_regularized_trend."""
    fc, _, boundary, _, _ = _forecast_regularized_trend(
        detrended_lookback, forecast_horizon,
        prev_slope=prev_slope, prev_boundary=prev_boundary,
        ema_alpha=ema_alpha, cp_threshold=cp_threshold, decay=decay,
        continuity_scale=continuity_scale,
    )
    return fc, boundary


def _forecast_regularized_trend(
    detrended_lookback: np.ndarray,
    forecast_horizon: int,
    prev_slope: Optional[float] = None,
    prev_boundary: Optional[float] = None,
    ema_alpha: float = 0.05,
    cp_threshold: float = 0.0,
    decay: float = 0.03,
    continuity_scale: float = 1.0,
) -> tuple:
    """
    Slow EMA-smoothed local trend with cross-window slope inheritance.

    Args:
        detrended_lookback: 1-D array (e.g. 96 points).
        forecast_horizon: Steps to forecast.
        prev_slope: Previous EMA slope (None on first window).
        prev_boundary: Forecast value at x=0 from previous window (for continuity).
        ema_alpha: Base EMA rate (default 0.05).
        cp_threshold: Changepoint detection threshold as relative slope change.
            Lower = more sensitive; 0 = always trigger changepoint (alpha_eff=0.5).
            NOT an L1 regularization coefficient.
        decay: OLS weight decay within window.
        continuity_scale: Multiplier on std(y) for cross-window continuity clamp.
            0 = no continuity, higher = more smoothing (default 1.0).

    Returns:
        (forecast, smooth_slope, boundary_value, has_cp, cp_pos)
            — boundary_value = forecast[0], pass as prev_boundary to next window.
    """
    y = np.asarray(detrended_lookback, dtype=float)
    L = len(y)
    x = np.arange(L, dtype=float)

    # Weighted OLS (toward recent points)
    weights = np.exp(-decay * (L - 1 - x))
    w_sum = np.sum(weights)
    wx_mean = np.sum(weights * x) / w_sum
    wy_mean = np.sum(weights * y) / w_sum
    raw_slope = (np.sum(weights * (x - wx_mean) * (y - wy_mean)) /
                 np.sum(weights * (x - wx_mean) ** 2))

    # Changepoint: relative disagreement between raw slope and EMA-smoothed trend.
    # The slow EMA (α=0.05) is the primary noise filter — it naturally resists
    # single-window noise. A CP fires when new evidence strongly contradicts
    # the inherited trend, temporarily accelerating adaptation to α=0.5.
    has_cp = False
    cp_pos = None
    if prev_slope is not None and np.isfinite(prev_slope):
        rel_change = abs(raw_slope - prev_slope) / (abs(prev_slope) + 0.0001)
        if rel_change > cp_threshold:
            has_cp = True
            cp_pos = -1
            alpha_eff = 0.5
        else:
            alpha_eff = ema_alpha
    else:
        alpha_eff = 1.0  # first window

    if prev_slope is not None and np.isfinite(prev_slope):
        smooth_slope = alpha_eff * raw_slope + (1.0 - alpha_eff) * prev_slope
    else:
        smooth_slope = raw_slope

    # Intercept: anchor to weighted tail of lookback (data-driven)
    k = max(8, L // 3)
    tail_w = weights[-k:] / np.sum(weights[-k:])
    anchor_x = float(np.sum(tail_w * x[-k:]))
    anchor_y = float(np.sum(tail_w * y[-k:]))
    intercept_data = anchor_y - smooth_slope * anchor_x

    # Cross-window continuity: blend data-driven intercept with previous window's
    # boundary value. Clamp the correction to 1 std of lookback residuals so a
    # regime change doesn't drag the forecast away from the new window's data.
    if prev_boundary is not None and np.isfinite(prev_boundary):
        intercept_cont = prev_boundary - smooth_slope * L
        max_dev = max(float(np.std(y)) * continuity_scale, 1e-6)
        deviation = intercept_cont - intercept_data
        deviation = float(np.clip(deviation, -max_dev, max_dev))
        intercept = intercept_data + deviation
    else:
        intercept = intercept_data

    x_fc = np.arange(L, L + forecast_horizon, dtype=float)
    forecast = intercept + smooth_slope * x_fc
    boundary_value = float(forecast[0])

    return (forecast, smooth_slope, boundary_value, has_cp, cp_pos)


