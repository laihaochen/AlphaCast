from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import DatasetConfig, ExperimentConfig
from ..data_loader import TIME_COL, infer_target_column
from ..eval import align_predictions, mae, mse, smape
from ..features import extract_target_features, extract_exogenous_features
from ..features.extract_exogenous import (
    EXOGENOUS_BASES,
    EXOGENOUS_DESCRIPTIONS,
    _find_station_columns,
)
from ..tools.analysis import (
    analyze_training,
    choose_cluster_by_similarity,
    choose_model_by_similarity,
    choose_neighbor_by_similarity,
)
from ..tools.forecast import _forecast_regularized_trend, forecast_with_model, get_prophet_forecast_components, save_predictions_csv
from ..utils.time import (
    CaseEntry,
    CaseNeighbor,
    ClusterEntry,
    generate_future_timestamps,
)


def json_default(value: Any) -> Any:
    """Helper to JSON-serialize numpy/pandas objects."""
    if isinstance(value, (np.integer, np.floating)):
        return float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.ndarray, list, tuple)):
        return [json_default(v) for v in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, pd.Series):
        return [json_default(v) for v in value.tolist()]
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    return value


def _find_window_position_in_train(
    window_vals: list,
    train_y: np.ndarray,
    look_back: int,
) -> Optional[int]:
    """
    Find the starting index in train_y where window_vals best matches.

    First attempts exact-match with floating tolerance, then falls back to
    correlation-based matching. Returns None if no reasonable match is found.
    """
    if train_y is None or len(train_y) < look_back or not window_vals:
        return None

    window_arr = np.asarray(window_vals, dtype=float)
    train_arr = np.asarray(train_y, dtype=float)

    if len(window_arr) != look_back:
        return None

    # Try exact match first
    best_dist = float("inf")
    best_idx = None
    for i in range(len(train_arr) - look_back + 1):
        dist = np.sum((train_arr[i : i + look_back] - window_arr) ** 2)
        if dist < best_dist:
            best_dist = dist
            best_idx = i
            if dist < 1e-10:
                break  # Exact match found

    return best_idx


def prepare_investor_packet(
    cfg: ExperimentConfig | None,
    ds_cfg: DatasetConfig,
    briefing_lookup: dict[str, str],
    window_offset: int = 0,
    forecast_horizon: Optional[int] = None,
) -> dict:
    """Generate the InvestigatorAgent research packet for a dataset window."""

    dataset_name = ds_cfg.name
    ds_out_dir = os.path.join(cfg.output_dir, dataset_name) if cfg else os.path.join("outputs", dataset_name)

    def _read_json(filename: str):
        path = os.path.join(ds_out_dir, filename)
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    basemodel_results = _read_json("basemodel_results.json") or []

    memory = _read_json("memory.json") or {}
    features = _read_json("features.json") or {}
    exo_features = _read_json("exogenous_features.json") or {}
    exo_corr = _read_json("exogenous_correlations.json") or {}
    exo_top3 = _read_json("exogenous_top3.json") or []
    exo_columns_meta = _read_json("exogenous_columns.json") or {}
    case_base_raw = _read_json("case_base.json") or []
    case_neighbor_raw = _read_json("case_neighbor.json") or []
    cluster_base_raw = _read_json("cluster_base.json") or []

    prophet_decomp = _read_json("prophet_decomposition.json") or {}
    decomposition_enabled = bool(prophet_decomp.get("decomposition_enabled", False))

    train_df = pd.read_csv(ds_cfg.training_csv)
    train_df[TIME_COL] = pd.to_datetime(train_df[TIME_COL])
    train_df = train_df.sort_values(TIME_COL).reset_index(drop=True)

    test_df = pd.read_csv(ds_cfg.test_csv)
    test_df[TIME_COL] = pd.to_datetime(test_df[TIME_COL])
    test_df = test_df.sort_values(TIME_COL).reset_index(drop=True)

    target_col = infer_target_column(train_df, dataset_name)

    look_back = int(ds_cfg.look_back)
    offset = max(0, int(window_offset) if window_offset else 0)

    test_len = len(test_df)
    lookback_end = offset + look_back
    test_target = test_df[target_col].to_numpy(dtype=float)

    if test_len >= lookback_end:
        window_source = "test"
        window_vals = test_target[offset:lookback_end]
        window_ts = test_df[TIME_COL].iloc[offset:lookback_end]
    else:
        if len(train_df) < look_back:
            raise ValueError(f"Insufficient data to build look-back window for dataset '{dataset_name}'")
        window_source = "training"
        window_vals = train_df[target_col].to_numpy(dtype=float)[-look_back:]
        window_ts = train_df[TIME_COL].iloc[-look_back:]
        offset = max(0, test_len - look_back)
        lookback_end = offset + look_back

    target_start_ts = None
    if test_len > lookback_end:
        target_start_ts = pd.Timestamp(test_df[TIME_COL].iloc[lookback_end]).isoformat()

    remaining_points = max(0, test_len - lookback_end)
    forecast_hint = int(forecast_horizon) if forecast_horizon else None
    ref_horizon = forecast_hint or int(ds_cfg.predicted_window)
    if ref_horizon <= 0:
        ref_horizon = int(ds_cfg.predicted_window)

    recommended_model = None
    best_model = None
    reference_prediction = None
    configured_model = None
    season_length = int(memory.get("periodicity_lag", 1)) if isinstance(memory, dict) else 1

    config_sel_model = getattr(cfg, "sel_model", None) if cfg else None
    if config_sel_model:
        try:
            configured_model = config_sel_model
            pred = forecast_with_model(
                config_sel_model,
                np.asarray(window_vals, dtype=float),
                int(ref_horizon),
                season_length=season_length,
                dataset=ds_cfg,
                timestamps=window_ts,
            )
            reference_prediction = pred.tolist()
        except Exception:
            pass

    if reference_prediction is None and cluster_base_raw:
        try:
            clusters = [
                ClusterEntry(
                    window=c.get("window", []),
                    best_model=c.get("best_model", {}),
                    total_weight=c.get("total_weight", 1),
                )
                for c in cluster_base_raw
                if isinstance(c, dict)
            ]
            clusters = [c for c in clusters if c.window and c.best_model]
            if clusters:
                best_cluster = choose_cluster_by_similarity(clusters, np.asarray(window_vals, dtype=float))
                best_model, total_weight = best_cluster.best_model, best_cluster.total_weight
                for model_name in best_model:
                    weight = 1.0 * best_model[model_name] / total_weight
                    model_pred = forecast_with_model(
                        model_name,
                        np.asarray(window_vals, dtype=float),
                        int(ref_horizon),
                        season_length=season_length,
                        dataset=ds_cfg,
                        timestamps=window_ts,
                    ).tolist()
                    if reference_prediction is None:
                        reference_prediction = [0.0] * ref_horizon
                    for i in range(ref_horizon):
                        reference_prediction[i] = reference_prediction[i] + weight * model_pred[i]
                if reference_prediction is not None:
                    recommended_model = "cluster-weighted"
        except Exception:
            pass

    if not reference_prediction:
        if case_base_raw:
            try:
                cases = [
                    CaseEntry(window=c.get("window", []), best_model=c.get("best_model"))
                    for c in case_base_raw
                    if isinstance(c, dict)
                ]
                cases = [c for c in cases if c.window and c.best_model]
                if cases:
                    best_case = choose_model_by_similarity(cases, np.asarray(window_vals, dtype=float))
                    if best_case:
                        recommended_model = best_case.best_model
            except Exception:
                pass

        primary_model = recommended_model or (memory.get("best_model") if isinstance(memory, dict) else None)
        fallback_models = []
        if isinstance(memory, dict):
            fallback_models = memory.get("model_rankings") or []
        candidate_models = []
        if primary_model:
            candidate_models.append(primary_model)
        for model in fallback_models:
            if isinstance(model, str) and model not in candidate_models:
                candidate_models.append(model)
        if not candidate_models:
            candidate_models = [
                "auto_arima",
                "theta",
                "ets",
                "sarimax",
                "lgbm",
                "prophet",
            ]
        best_model = None
        primary_exc: Exception | None = None
        for model in candidate_models:
            try:
                pred = forecast_with_model(
                    model,
                    np.asarray(window_vals, dtype=float),
                    int(ref_horizon),
                    season_length=season_length,
                    dataset=ds_cfg,
                    timestamps=window_ts,
                )
                best_model = model
                reference_prediction = pred.tolist()
                break
            except Exception as exc:
                primary_exc = exc
                continue
        if reference_prediction is None and primary_exc is not None:
            raise primary_exc

    neighbor_lookback = None
    neighbor_pred = None
    if case_neighbor_raw:
        try:
            cases_neighbor = [
                CaseNeighbor(look_back_window=c.get("look_back_window", []), pred_window=c.get("pred_window", []))
                for c in case_neighbor_raw
                if isinstance(c, dict)
            ]
            cases_neighbor = [c for c in cases_neighbor if c.look_back_window and c.pred_window]
            if cases_neighbor:
                neighbor_lookback, neighbor_pred = choose_neighbor_by_similarity(
                    cases_neighbor,
                    np.asarray(window_vals, dtype=float),
                )
                neighbor_lookback = neighbor_lookback.tolist()
                neighbor_pred = neighbor_pred.tolist()
        except Exception:
            neighbor_lookback = None
            neighbor_pred = None

    def _sanitize_exo_key(name: str) -> str:
        cleaned = re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower())
        return cleaned.strip("_")

    def _series_to_float_list(series: pd.Series) -> List[Optional[float]]:
        values: List[Optional[float]] = []
        for val in series.tolist():
            if pd.isna(val):
                values.append(None)
            else:
                try:
                    values.append(float(val))
                except Exception:
                    values.append(None)
        return values

    def _collect_exogenous_slice(
        df: pd.DataFrame,
        start_idx: int,
        end_idx: int,
        details: List[dict],
    ) -> Dict[str, Dict[str, Any]]:
        if start_idx >= end_idx:
            return {}

        collected: Dict[str, Dict[str, Any]] = {}
        for info in details:
            canonical = info.get("canonical_name")
            columns = info.get("source_columns") or []
            entry = {
                "display_name": info.get("display_name"),
                "description": info.get("description"),
                "source_columns": columns,
                "values_by_column": {},
                "missing_columns": [],
            }

            for col in columns:
                if col in df.columns:
                    slice_vals = df[col].iloc[start_idx:end_idx]
                    entry["values_by_column"][col] = _series_to_float_list(slice_vals)
                else:
                    entry["missing_columns"].append(col)

            if canonical:
                collected[canonical] = entry

        return collected

    exogenous_column_map: dict[str, List[str]] = {}
    if isinstance(exo_columns_meta, dict) and exo_columns_meta:
        for raw_key, cols in exo_columns_meta.items():
            if not isinstance(cols, list):
                continue
            sanitized_key = _sanitize_exo_key(raw_key)
            cleaned_cols = [str(c) for c in cols if isinstance(c, str)]
            if sanitized_key and cleaned_cols:
                exogenous_column_map[sanitized_key] = cleaned_cols

    if not exogenous_column_map:
        try:
            for canonical, base_names in EXOGENOUS_BASES.items():
                cols = _find_station_columns(train_df, base_names)
                filtered = [
                    str(c)
                    for c in cols
                    if c not in {TIME_COL, target_col}
                ]
                if filtered:
                    exogenous_column_map[_sanitize_exo_key(canonical)] = filtered
        except Exception:
            exogenous_column_map = {}

    if not exogenous_column_map:
        fallback_cols = [
            str(c)
            for c in train_df.columns
            if c not in {TIME_COL, target_col}
        ]
        for col in fallback_cols:
            sanitized_key = _sanitize_exo_key(col)
            if sanitized_key:
                exogenous_column_map.setdefault(sanitized_key, [])
                if col not in exogenous_column_map[sanitized_key]:
                    exogenous_column_map[sanitized_key].append(col)

    exogenous_columns: List[str] = []
    for cols in exogenous_column_map.values():
        for col in cols:
            if col not in exogenous_columns:
                exogenous_columns.append(col)

    exogenous_column_details = []
    for canonical_key in sorted(exogenous_column_map.keys()):
        cols = exogenous_column_map[canonical_key]
        if not cols:
            continue
        display_name = canonical_key.replace("_", " ").title()
        description = EXOGENOUS_DESCRIPTIONS.get(canonical_key)
        if description is None:
            match_found = False
            for base, text in EXOGENOUS_DESCRIPTIONS.items():
                if base in canonical_key:
                    description = text
                    match_found = True
                    break
            if not match_found:
                description = ""
        entry = {
            "canonical_name": canonical_key,
            "display_name": display_name,
            "description": description,
            "source_columns": cols,
        }
        exogenous_column_details.append(entry)

    look_back_exogenous_values: Dict[str, Dict[str, Any]] | None = None
    forecast_window_exogenous_values: Dict[str, Dict[str, Any]] | None = None
    forecast_window_coverage: Dict[str, Any] | None = None
    forecast_window_timestamps: List[str] = []

    if exogenous_column_details:
        forecast_end_idx = lookback_end + ref_horizon
        if forecast_end_idx > len(test_df):
            forecast_end_idx = len(test_df)

        if target_start_ts:
            forecast_window_timestamps = [
                pd.Timestamp(test_df[TIME_COL].iloc[idx]).isoformat()
                for idx in range(lookback_end, forecast_end_idx)
            ]

        coverage: Dict[str, Any] = {
            "total_steps": ref_horizon,
            "available_steps": forecast_end_idx - lookback_end,
            "missing_steps": max(0, ref_horizon - (forecast_end_idx - lookback_end)),
        }
        forecast_window_coverage = coverage

        if lookback_end <= len(test_df):
            look_back_exogenous_values = _collect_exogenous_slice(
                test_df,
                max(0, lookback_end - look_back),
                lookback_end,
                exogenous_column_details,
            )
        else:
            lb_start = max(0, len(train_df) - look_back)
            look_back_exogenous_values = _collect_exogenous_slice(
                train_df,
                lb_start,
                len(train_df),
                exogenous_column_details,
            )

        if forecast_end_idx > lookback_end:
            forecast_window_exogenous_values = _collect_exogenous_slice(
                test_df,
                lookback_end,
                forecast_end_idx,
                exogenous_column_details,
            )

    basemodel_result = {
        "step_index": len(basemodel_results) + 1,
        "best_model": best_model,
        "recommended_model": recommended_model,
        "configured_model": configured_model,
        "reference_prediction": reference_prediction,
    }
    basemodel_results.append(basemodel_result)

    os.makedirs(ds_out_dir, exist_ok=True)
    with open(os.path.join(ds_out_dir, "basemodel_results.json"), "w", encoding="utf-8") as f:
        json.dump(basemodel_results, f, indent=2)

    # === Prophet 双组件趋势分解 + 残差空间转换 ===
    # trend_final(t) = trend_global(t) + trend_local(t)
    # seasonality(t)   = seasonality_global(t)   (global only, needs lots of data)
    look_back_residuals = [float(v) for v in window_vals]
    reference_residual = reference_prediction
    neighbor_residual_lookback = neighbor_lookback
    neighbor_residual_pred = neighbor_pred
    prophet_trend_forecast: List[float] = []       # combined trend = global + local
    prophet_trend_global: List[float] = []
    prophet_trend_local: List[float] = []
    prophet_seasonality_forecast: List[float] = []

    if decomposition_enabled and prophet_decomp:
        # ---- Step 1: Look up global trend + seasonality for lookback window ----
        look_back_ts_strs = [pd.Timestamp(ts).isoformat() for ts in window_ts]
        lb_components = get_prophet_forecast_components(ds_out_dir, look_back_ts_strs)
        lb_trend_global = lb_components.get("trend") or []
        lb_seas_global = lb_components.get("seasonality") or []

        if not lb_trend_global or len(lb_trend_global) != len(window_vals):
            # Fallback: use training decomposition directly
            train_trend_arr = prophet_decomp.get("train_trend") or []
            train_seas_arr = prophet_decomp.get("train_seasonality") or []
            train_ts_arr = prophet_decomp.get("train_timestamps") or []
            ts_to_idx = {ts: i for i, ts in enumerate(train_ts_arr)}
            lb_trend_global = []
            lb_seas_global = []
            for ts in window_ts:
                ts_str = pd.Timestamp(ts).isoformat()
                idx = ts_to_idx.get(ts_str)
                if idx is not None and idx < len(train_trend_arr):
                    lb_trend_global.append(float(train_trend_arr[idx]))
                    lb_seas_global.append(float(train_seas_arr[idx]))
                else:
                    lb_trend_global.append(0.0)
                    lb_seas_global.append(0.0)

        # ---- Step 2: Get global trend + seasonality for forecast window ----
        prophet_trend_global = []
        if forecast_window_timestamps and len(forecast_window_timestamps) > 0:
            fw_components = get_prophet_forecast_components(ds_out_dir, forecast_window_timestamps)
            prophet_trend_global = fw_components.get("trend") or []
            prophet_seasonality_forecast = fw_components.get("seasonality") or []

        # ---- Step 3: Compute detrended lookback residuals ----
        # r_lb = y_lb - trend_global_lb - seasonality_global_lb
        detrended_lb = np.asarray([
            float(window_vals[i]) - float(lb_trend_global[i]) - float(lb_seas_global[i])
            for i in range(len(window_vals))
        ], dtype=float)

        # ---- Step 4: Fit local trend on detrended lookback ----
        fc_horizon = int(ref_horizon) if ref_horizon > 0 else int(ds_cfg.predicted_window)
        # Only fit local trend if both lookback and forecast windows are populated
        if len(detrended_lb) >= 2 and fc_horizon > 0 and prophet_trend_global:
            try:
                _ema_slope = getattr(prepare_investor_packet, "_ema_slope", {})
                _ema_boundary = getattr(prepare_investor_packet, "_ema_boundary", {})
                prev_s = _ema_slope.get(dataset_name)
                prev_b = _ema_boundary.get(dataset_name)
                fc, new_s, boundary, has_cp, cp_pos = _forecast_regularized_trend(
                    detrended_lb, fc_horizon,
                    prev_slope=prev_s, prev_boundary=prev_b,
                    ema_alpha=0.05,
                    cp_threshold=ds_cfg.cp_threshold,
                    continuity_scale=ds_cfg.continuity_scale,
                )
                _ema_slope[dataset_name] = new_s
                _ema_boundary[dataset_name] = boundary
                prepare_investor_packet._ema_slope = _ema_slope
                prepare_investor_packet._ema_boundary = _ema_boundary
                prophet_trend_local = fc.tolist()
            except Exception:
                # Fallback: no local correction
                prophet_trend_local = [0.0] * fc_horizon
        else:
            prophet_trend_local = ([0.0] * fc_horizon) if fc_horizon > 0 else []

        # ---- Step 5: Combine trend = global + local ----
        if prophet_trend_global and prophet_trend_local and len(prophet_trend_global) == len(prophet_trend_local):
            prophet_trend_forecast = [
                float(prophet_trend_global[i]) + float(prophet_trend_local[i])
                for i in range(len(prophet_trend_global))
            ]
        elif prophet_trend_global:
            prophet_trend_forecast = [float(v) for v in prophet_trend_global]
        else:
            prophet_trend_forecast = []

        # ---- Step 6: Convert look_back to residual space ----
        # residual = y - trend_global - seasonality_global (lookback uses global only)
        look_back_residuals = detrended_lb.tolist()

        # ---- Step 7: Convert reference_prediction to residual space ----
        # reference_residual = reference_raw - trend_combined_fc - seasonality_fc
        if reference_prediction and prophet_trend_forecast:
            n_ref = min(len(reference_prediction), len(prophet_trend_forecast))
            reference_residual = [
                float(reference_prediction[i])
                - float(prophet_trend_forecast[i])
                - float(prophet_seasonality_forecast[i])
                for i in range(n_ref)
            ]
            if len(reference_residual) < len(reference_prediction):
                reference_residual = list(reference_residual) + reference_prediction[len(reference_residual):]

        # ---- Step 8: Convert neighbor windows to residual space ----
        train_y_full = train_df[target_col].to_numpy(dtype=float)
        train_trend_arr = prophet_decomp.get("train_trend") or []
        train_seas_arr = prophet_decomp.get("train_seasonality") or []

        if neighbor_lookback and train_trend_arr and len(train_trend_arr) == len(train_y_full):
            nlb_pos = _find_window_position_in_train(neighbor_lookback, train_y_full, look_back)
            if nlb_pos is not None:
                neighbor_residual_lookback = [
                    float(neighbor_lookback[i])
                    - float(train_trend_arr[nlb_pos + i])
                    - float(train_seas_arr[nlb_pos + i])
                    for i in range(look_back)
                ]

        if neighbor_pred and train_trend_arr and len(train_trend_arr) == len(train_y_full):
            if neighbor_residual_lookback is not neighbor_lookback and neighbor_lookback:
                nlb_pos = _find_window_position_in_train(neighbor_lookback, train_y_full, look_back)
                if nlb_pos is not None:
                    pred_start = nlb_pos + look_back
                    n_pred = min(len(neighbor_pred), len(train_trend_arr) - pred_start)
                    if n_pred > 0:
                        neighbor_residual_pred = [
                            float(neighbor_pred[i])
                            - float(train_trend_arr[pred_start + i])
                            - float(train_seas_arr[pred_start + i])
                            for i in range(n_pred)
                        ]
            elif neighbor_lookback:
                nlb_pos = _find_window_position_in_train(neighbor_lookback, train_y_full, look_back)
                if nlb_pos is not None:
                    pred_start = nlb_pos + look_back
                    n_pred = min(len(neighbor_pred), len(train_trend_arr) - pred_start)
                    if n_pred > 0:
                        neighbor_residual_pred = [
                            float(neighbor_pred[i])
                            - float(train_trend_arr[pred_start + i])
                            - float(train_seas_arr[pred_start + i])
                            for i in range(n_pred)
                        ]

    return {
        "dataset": dataset_name,
        "look_back_length": look_back,
        "look_back_window": look_back_residuals,
        "look_back_timestamps": [ts.isoformat() for ts in window_ts],
        "look_back_source": window_source,
        "window_offset": offset,
        "look_back_end_index": lookback_end,
        "prediction_start_timestamp": target_start_ts,
        "remaining_test_points": remaining_points,
        "forecast_horizon_hint": forecast_hint,
        "predicted_window": int(ds_cfg.predicted_window),
        "frequency": memory.get("frequency") if isinstance(memory, dict) else None,
        "memory": memory,
        "features": features,
        "exogenous_features": exo_features,
        "exogenous_correlations": exo_corr,
        "exogenous_top3": exo_top3,
        "exogenous_columns": exogenous_columns,
        "exogenous_column_map": exogenous_column_map,
        "exogenous_column_details": exogenous_column_details,
        "look_back_exogenous_values": look_back_exogenous_values,
        "forecast_window_timestamps": forecast_window_timestamps,
        "forecast_window_exogenous_values": forecast_window_exogenous_values,
        "forecast_window_coverage": forecast_window_coverage,
        "dataset_briefing": briefing_lookup.get(dataset_name, ""),
        "case_base_size": len(case_base_raw),
        "case_neighbor_size": len(case_neighbor_raw),
        "best_model_name": best_model,
        "configured_model_name": configured_model,
        "recommended_model_name": recommended_model,
        "reference_prediction": reference_residual,
        "neighbor_lookback": neighbor_residual_lookback,
        "neighbor_pred": neighbor_residual_pred,
        # === Prophet 双组件趋势字段 ===
        "prophet_trend_forecast": prophet_trend_forecast,        # combined = global + local
        "prophet_trend_global": prophet_trend_global,            # long-term trend from full train.csv
        "prophet_trend_local": prophet_trend_local,              # per-window local correction
        "prophet_seasonality_forecast": prophet_seasonality_forecast,
        "decomposition_enabled": decomposition_enabled,
    }


def assess_forecast(
    predictions: List[float],
    predicted_window: int,
    investor_packet: dict[str, Any],
    chain_of_thought: str,
) -> dict[str, Any]:
    issues: List[str] = []
    notes: List[str] = []

    if len(predictions) != predicted_window:
        issues.append(
            f"Prediction length {len(predictions)} does not match expected window {predicted_window}."
        )

    reference = investor_packet.get("reference_prediction") or []
    if reference and len(reference) == len(predictions):
        try:
            ref_arr = np.asarray(reference, dtype=float)
            pred_arr = np.asarray(predictions, dtype=float)
            diff = np.abs(ref_arr - pred_arr)
            notes.append(f"Mean abs deviation vs. baseline: {diff.mean():.4f}")
        except Exception as exc:
            notes.append(f"Skipped baseline comparison due to error: {exc}")
    elif reference:
        issues.append("Reference prediction length mismatch; cannot compare to baseline.")

    coverage = investor_packet.get("forecast_window_coverage") or {}
    if isinstance(coverage, dict):
        missing = int(coverage.get("missing_steps", 0) or 0)
        if missing > 0:
            notes.append(f"Coverage warning: {missing} horizon steps lack exogenous data.")

    if not chain_of_thought.strip():
        issues.append("Chain-of-thought log is empty; reasoning not persisted.")

    approved = not issues
    summary = {
        "approved": approved,
        "issues": issues,
        "notes": "; ".join(notes) if notes else "",
    }
    return summary


def deterministic_run_for_dataset(cfg: ExperimentConfig, ds) -> dict:
    # Load training + test data
    data = pd.read_csv(ds.training_csv)
    data[TIME_COL] = pd.to_datetime(data[TIME_COL])
    data = data.sort_values(TIME_COL).reset_index(drop=True)

    test_df = pd.read_csv(ds.test_csv)
    test_df[TIME_COL] = pd.to_datetime(test_df[TIME_COL])
    test_df = test_df.sort_values(TIME_COL).reset_index(drop=True)

    _ = analyze_training(
        data,
        ds.look_back,
        ds.predicted_window,
        cfg.output_dir,
        ds.name,
        ds.sliding_window,
        dataset_cfg=ds,
    )
    ds_out_dir = os.path.join(cfg.output_dir, ds.name)
    memory_path = os.path.join(ds_out_dir, "memory.json")
    case_base_path = os.path.join(ds_out_dir, "case_base.json")
    case_neighbor_path = os.path.join(ds_out_dir, "case_neighbor.json")
    with open(memory_path, "r", encoding="utf-8") as f:
        memory = json.load(f)

    target_col = infer_target_column(data, ds.name)
    y = data[target_col].to_numpy(dtype=float)

    with open(case_base_path, "r", encoding="utf-8") as f:
        raw_cases = json.load(f)
    cases = [CaseEntry(window=c["window"], best_model=c["best_model"]) for c in raw_cases]

    with open(case_neighbor_path, "r", encoding="utf-8") as f:
        raw_neighbors = json.load(f)
    cases_neighbors = [CaseNeighbor(look_back_window=c["look_back_window"], pred_window=c["pred_window"]) for c in raw_neighbors]

    feats = {}
    sel_cfg = getattr(cfg, "feature_selection_override", None)
    if getattr(cfg, "use_features", True):
        try:
            feats = extract_target_features(y, memory.get("frequency"))
            with open(os.path.join(ds_out_dir, "features.json"), "w", encoding="utf-8") as f:
                json.dump(feats, f, indent=2)
            if isinstance(sel_cfg, dict):
                with open(os.path.join(ds_out_dir, "selected_features.json"), "w", encoding="utf-8") as f:
                    json.dump(sel_cfg, f, indent=2)
        except Exception:
            pass

    if getattr(cfg, "use_exogenous", False):
        try:
            (
                exo_features_by_var,
                exo_corr_by_var,
                exo_top3,
                exo_columns_by_var,
            ) = extract_exogenous_features(
                data, target_col, ds.name, memory.get("frequency")
            )
            with open(os.path.join(ds_out_dir, "exogenous_features.json"), "w", encoding="utf-8") as f:
                json.dump(exo_features_by_var, f, indent=2)
            with open(os.path.join(ds_out_dir, "exogenous_correlations.json"), "w", encoding="utf-8") as f:
                json.dump(exo_corr_by_var, f, indent=2)
            with open(os.path.join(ds_out_dir, "exogenous_top3.json"), "w", encoding="utf-8") as f:
                json.dump(exo_top3, f, indent=2)
            with open(os.path.join(ds_out_dir, "exogenous_columns.json"), "w", encoding="utf-8") as f:
                json.dump(exo_columns_by_var, f, indent=2)
        except Exception:
            pass

    # Predict over the TEST period using sliding windows, matching the test-set
    # timeline.  Each window uses similarity-based model selection as in the
    # LLM-orchestrated path.
    L = int(ds.look_back)
    H = int(ds.predicted_window)
    stride = int(ds.sliding_window) if ds.sliding_window and int(ds.sliding_window) > 0 else H
    test_y = test_df[target_col].to_numpy(dtype=float)
    test_ts = test_df[TIME_COL]

    target_len = len(test_df) - L
    all_predictions: list[float] = []
    all_timestamps: list[pd.Timestamp] = []

    offset = 0
    while offset + H <= target_len:
        idx = offset + L
        # Current lookback window from the test set
        cur_win = test_y[offset:idx]
        cur_ts = test_ts.iloc[offset:idx]

        pre_model = choose_model_by_similarity(cases, cur_win)
        config_sel_model = getattr(cfg, "sel_model", None)
        model_name = config_sel_model or pre_model
        override_reason = None
        seas_val = None
        if config_sel_model:
            override_reason = "sel_model"
        elif getattr(cfg, "use_features", True):
            if (
                isinstance(sel_cfg, dict)
                and isinstance(sel_cfg.get("force_model"), str)
                and sel_cfg.get("force_model")
            ):
                model_name = str(sel_cfg["force_model"]) or model_name
                override_reason = "force_model"
            else:
                try:
                    seas_val = float(
                        feats.get("seasonal_strength", feats.get("seas_acf1", 0.0))
                    )
                except Exception:
                    seas_val = None
                prefer_seasonal = (
                    bool(sel_cfg.get("prefer_seasonal_naive_if_seasonal", False))
                    if isinstance(sel_cfg, dict)
                    else False
                )
                if prefer_seasonal and seas_val is not None and seas_val >= 0.6:
                    model_name = "SeasonalNaive"
                    override_reason = "prefer_seasonal_naive_if_seasonal"

        preds = forecast_with_model(
            model_name,
            cur_win,
            H,
            season_length=int(memory.get("periodicity_lag", 1)),
            dataset=ds,
            timestamps=cur_ts,
        )

        _mem_freq = memory.get("frequency")
        if isinstance(_mem_freq, str):
            _mem_freq_clean = _mem_freq.strip()
            if _mem_freq_clean.lower() in {"h", "d", "w", "m", "s"}:
                _mem_freq = _mem_freq_clean.upper()
            else:
                _mem_freq = _mem_freq_clean
        last_known_ts = test_ts.iloc[idx - 1]  # last timestamp in lookback window
        future_ts = generate_future_timestamps(pd.Timestamp(last_known_ts), H, _mem_freq)

        # Take only as many predictions as we have timestamps for
        n = min(H, len(future_ts), len(preds))
        all_predictions.extend(preds[:n].tolist())
        all_timestamps.extend(future_ts[:n])

        offset += stride

    # Tail: if stride doesn't divide target_len evenly, predict the remaining points
    remaining = len(test_df) - (offset + L)
    if remaining > 0:
        tail_len = min(remaining, H)
        cur_win = test_y[offset : offset + L]
        cur_ts = test_ts.iloc[offset : offset + L]
        preds = forecast_with_model(
            model_name,
            cur_win,
            tail_len,
            season_length=int(memory.get("periodicity_lag", 1)),
            dataset=ds,
            timestamps=cur_ts,
        )
        last_known_ts = test_ts.iloc[offset + L - 1]
        future_ts = generate_future_timestamps(pd.Timestamp(last_known_ts), tail_len, _mem_freq)
        n = min(tail_len, len(future_ts), len(preds))
        all_predictions.extend(preds[:n].tolist())
        all_timestamps.extend(future_ts[:n])

    out_csv = os.path.join(ds_out_dir, "predictions.csv")
    save_predictions_csv(out_csv, all_timestamps, np.asarray(all_predictions, dtype=float))

    # Evaluate against test set ground truth
    pred_df = pd.read_csv(out_csv)
    pred_df["time_stamp"] = pd.to_datetime(pred_df["time_stamp"])
    y_true, y_pred = align_predictions(test_df, pred_df, ds.name)

    meta = {
        "dataset": ds.name,
        "memory_path": memory_path,
        "case_base_path": case_base_path,
        "frequency": memory.get("frequency"),
        "periodicity_lag": memory.get("periodicity_lag"),
        "look_back": ds.look_back,
        "predicted_window": ds.predicted_window,
    }
    with open(os.path.join(ds_out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return {
        "dataset": ds.name,
        "MSE": mse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "sMAPE": smape(y_true, y_pred),
        "model": "deterministic",
    }


__all__ = [
    "json_default",
    "prepare_investor_packet",
    "assess_forecast",
    "deterministic_run_for_dataset",
]
