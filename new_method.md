# Prophet 分解 + LLM 残差预测 改动报告

## 一、核心思路

**改动前**：LLM Agent 直接预测完整时间序列值 y(t)，以 baseline 模型预测为锚点做调整。

**改动后**：Prophet 作为预处理器剥离趋势项和季节项，LLM Agent 只预测残差项 r(t)，最终预测由三者组装：

```
y_pred(t) = Prophet.trend(t) + Prophet.seasonality(t) + LLM.residual(t)
```

Prophet 在训练集上拟合一次（全局），趋势项和季节项由 Prophet 自带方法预测，Agent 不调整。Agent 利用丰富的 context（外生变量、邻居窗口、memory stats）来预测残差，弥补 Prophet 对残差 i.i.d. 正态假设的不足。

---

## 二、架构变化总览

```
                        【改动前】                           【改动后】
                        
训练阶段:                                           训练阶段:
  analyze_training()                                  analyze_training()
    → memory/case_base/cluster                           → 同上
    → features/exogenous                              + prophet_decompose()
                                                         → prophet_decomposition.json
                                                         (trend+seasonality 全训练集)

预测阶段 (每 window):                                预测阶段 (每 window):
  ┌─────────────────────────┐                         ┌──────────────────────────────┐
  │ consult()               │                         │ consult()                    │
  │  → look_back: 原始值    │                         │  → look_back: 残差值 ✓       │
  │  → reference: 原始值    │                         │  → reference: 残差值 ✓       │
  │  → neighbor: 原始值     │                         │  → neighbor: 残差值 ✓        │
  │                          │                         │  + prophet_trend_forecast   │
  │                          │                         │  + prophet_season_forecast  │
  └─────────────────────────┘                         └──────────────────────────────┘
  ┌─────────────────────────┐                         ┌──────────────────────────────┐
  │ LLM 推理                │                         │ LLM 推理                    │
  │  "baseline mean 32.5,   │                         │  "residual baseline mean    │
  │   I adjust to 34.2..."  │                         │   2.1, exogenous suggests   │
  │                          │                         │   higher residual..."       │
  └─────────────────────────┘                         └──────────────────────────────┘
  ┌─────────────────────────┐                         ┌──────────────────────────────┐
  │ emit_predictions()      │                         │ emit_predictions()           │
  │  写入 predictions.csv   │                         │  组装: final = trend         │
  │  (原始值)               │                         │         + seasonality        │
  │                          │                         │         + residual           │
  │                          │                         │  写入 predictions.csv (原始值)│
  └─────────────────────────┘                         └──────────────────────────────┘
```

**关键点：predictions.csv 最终存储的仍然是原始时间序列值，格式不变**。残差组装在 `emit_predictions` 内部完成，对外透明。

---

## 三、文件改动清单（共 7 个文件）

| # | 文件 | 改动类型 | 复杂度 |
|---|------|---------|--------|
| 1 | `alphacast/models/base.py` | 新增 `ProphetModel.decompose()` 方法 | 中 |
| 2 | `alphacast/tools/forecast.py` | 新增 `prophet_decompose_series()` 函数 | 中 |
| 3 | `run_experiment.py` | 训练阶段新增 Prophet 分解步骤；提示词模板更新 | 中 |
| 4 | `alphacast/agents/common.py` | `prepare_investor_packet()` 适配残差空间 | **高** |
| 5 | `alphacast/agents/generator_agent.py` | `emit_predictions()` 新增残差组装逻辑 | 中 |
| 6 | `alphacast/agents/reflector_agent.py` | `_BASELINE_CLAIM_PATTERN` 添加 residual 关键词 | 低 |
| 7 | `prompts/generator_agent.md` | Agent 指令改为预测残差 | 低 |

---

## 四、逐文件详细改动

### 4.1 `alphacast/models/base.py` — ProphetModel 增强

**现状**（行 1163-1249）：
- `ProphetModel.fit()` 接受 timestamps，拟合 Prophet
- `ProphetModel.predict()` 返回 `fcst['yhat']`（最终预测值）
- 未利用 Prophet 的 `trend`、`weekly`、`yearly` 等分解列

**改动**：新增 `decompose()` 方法

```python
def decompose(self, y: np.ndarray, timestamps, future_timestamps=None) -> dict:
    """
    对时间序列做 Prophet 分解。
    
    Returns:
        {
            "trend": np.ndarray,          # 趋势项 (len = len(y) 或 len(future_timestamps))
            "seasonality": np.ndarray,     # 季节项总和 = weekly + yearly + daily
            "residual": np.ndarray,        # 残差 = y - trend - seasonality
            "weekly": np.ndarray,          # 周季节分量
            "yearly": np.ndarray,          # 年季节分量
        }
    """
```

实现要点：
- 对训练集：`trend + seasonality = fcst['trend'] + fcst['weekly'] + fcst['yearly'] + ...`（拟合值，包含所有 additive_terms）
- 对预测窗口：同上，但输入是 `future_timestamps`（预测值）
- Prophet 的 `predict()` 返回 DataFrame 包含 `trend`、`weekly`、`yearly`、`daily`、`additive_terms`、`multiplicative_terms` 列
- `seasonality = weekly + yearly + daily + extra_regressors_additive`（视数据集频率而定）
- 残差 = 原始值 - trend - seasonality

### 4.2 `alphacast/tools/forecast.py` — 新增 Prophet 分解函数

**现状**（行 1-53）：提供 `forecast_with_model()` 和 `save_predictions_csv()`

**改动**：新增以下函数

```python
def prophet_decompose_series(
    train_df: pd.DataFrame,
    target_col: str,
    ds_name: str,
    output_dir: str,
    season_length: int,
    frequency: str,
) -> dict:
    """
    用 Prophet 分解整个训练序列。
    
    1. 创建 ProphetModel，拟合全训练集
    2. 对训练集每个时间点调用 predict() 获取 trend + seasonality
    3. 计算残差 = 实际值 - trend - seasonality
    4. 保存到 outputs/<ds>/prophet_decomposition.json
    
    Returns:
        {
            "trend": [...],        # 全训练集的趋势值
            "seasonality": [...],  # 全训练集的季节值
            "residual": [...],     # 全训练集的残差值
            "timestamps": [...],   # 对应的时间戳
        }
    """
```

**保存文件**：`outputs/<dataset>/prophet_decomposition.json`
```json
{
    "train_trend": [1.2, 1.3, ...],
    "train_seasonality": [0.5, -0.3, ...],
    "train_residual": [0.1, -0.2, ...],
    "train_timestamps": ["2015-01-01 00:00:00", ...],
    "model_params": {
        "yearly_seasonality": "auto",
        "weekly_seasonality": "auto",
        "daily_seasonality": "auto"
    }
}
```

```python
def get_prophet_forecast_components(
    ds_out_dir: str,
    future_timestamps: list,
) -> dict:
    """
    从已拟合的 Prophet 模型获取预测窗口的 trend + seasonality。
    从 prophet_decomposition.json 中恢复模型参数，重新拟合（或缓存模型对象），
    然后对 future_timestamps 预测分解成分。
    
    Returns:
        {
            "trend": [float x H],
            "seasonality": [float x H],
        }
    """
```

### 4.3 `run_experiment.py` — 训练阶段 + 提示词

**改动点 A（行 ~109-123）：训练阶段新增 Prophet 分解**

在 `analyze_training()` 之后、LLM 循环之前，新增：

```python
# === Prophet 分解 (新增) ===
if use_agent:
    prophet_result = prophet_decompose_series(
        train_df, target_col, ds.name, cfg.output_dir,
        season_length=int(analysis.memory.periodicity_lag),
        frequency=analysis.memory.frequency,
    )
    print(f"[info] Prophet decomposition saved for dataset '{ds.name}'")
```

**改动点 B（行 ~349-387）：提示词模板更新**

当前提示词中关键句子需要从"预测完整值"改为"预测残差"：

```python
# 旧：
# "anchor on `reference_prediction`, compare it to neighbor hints..."
# "the forecast stays consistent with the baseline guidance"

# 新：
prompt_sections.append(dedent(f"""
    ...
    The time series has been pre-decomposed by Prophet:
      - trend(t) and seasonality(t) are deterministic components provided in the packet.
      - You are predicting only the RESIDUAL component r(t) = y(t) - trend(t) - seasonality(t).
      - The final forecast will be assembled as: y(t) = trend(t) + seasonality(t) + your_residual(t).
    
    Required actions:
      1. Call tool.consult ... to fetch the packet (now contains residual-space data).
      2. Analyse the packet: anchor on `reference_prediction` (which is the baseline model's residual),
         compare to neighbor hints and exogenous trends. Only adjust when evidence supports it.
      3. ... Reflection confirming predictions are residual values ...
      4. ... record_chain_of_thought ...
      5. Call tool.emit_predictions with:
           - predictions: a list of {step_horizon} RESIDUAL floats (not raw values!)
           - ...
    
    Important: Your predictions are RESIDUALS. The system will automatically add back
    the Prophet trend and seasonality components. Do NOT add them yourself.
"""))
```

### 4.4 `alphacast/agents/common.py` — `prepare_investor_packet()` 适配残差空间

**这是最核心的改动**。当前函数（行 52-466）返回的 packet 中包含：
- `look_back_window`：原始值
- `reference_prediction`：原始值
- `neighbor_lookback`、`neighbor_pred`：原始值

**改动**：在 packet 组装完成后，将上述字段转换为残差空间，并新增 Prophet 成分字段。

**改动点 A（函数签名不变，内部新增步骤）：**

在函数开头附近（行 ~71）加载 Prophet 分解：

```python
prophet_decomp = _read_json("prophet_decomposition.json") or {}
```

**改动点 B（行 ~435-465，packet 返回前）：残差空间转换**

```python
# === Prophet 残差空间转换 (新增) ===
prophet_trend_train = prophet_decomp.get("train_trend") or []
prophet_seasonality_train = prophet_decomp.get("train_seasonality") or []
prophet_train_ts = prophet_decomp.get("train_timestamps") or []

# 1. 获取 Prophet 对预测窗口的趋势+季节预测
prophet_trend_forecast = []
prophet_seasonality_forecast = []
if forecast_window_timestamps and prophet_decomp:
    # 重新用 Prophet 对预测窗口做分解预测
    trend_fc, seas_fc = _prophet_forecast_components(
        ds_cfg, window_ts, forecast_window_timestamps, season_length
    )
    prophet_trend_forecast = trend_fc
    prophet_seasonality_forecast = seas_fc

# 2. 转换 look_back_window 为残差
#    look_back 的值减去 Prophet 对应时间点的 trend + seasonality
look_back_residuals = []
if len(prophet_trend_train) == len(window_vals):
    for i, val in enumerate(window_vals):
        trend_val = prophet_trend_train[i] if i < len(prophet_trend_train) else 0
        seas_val = prophet_seasonality_train[i] if i < len(prophet_seasonality_train) else 0
        look_back_residuals.append(float(val) - trend_val - seas_val)
else:
    # 如果 look_back 跨越训练集和测试集边界，需要处理
    # ...（见下方边缘情况处理）

# 3. 转换 reference_prediction 为残差
reference_residual = None
if reference_prediction and prophet_trend_forecast:
    reference_residual = [
        reference_prediction[i] - prophet_trend_forecast[i] - prophet_seasonality_forecast[i]
        for i in range(min(len(reference_prediction), len(prophet_trend_forecast)))
    ]

# 4. 转换 neighbor 为残差（如有）
neighbor_residual_lookback = None
neighbor_residual_pred = None
if neighbor_lookback and neighbor_pred:
    # neighbor 窗口来自训练集，可以用 prophet_decomp 中的值减去
    # ...（需要根据 neighbor 的时间位置查找对应的 trend+seasonality）
```

**改动点 C（return dict 修改）：**

```python
return {
    # ... 原有字段保持不变 ...
    "look_back_window": look_back_residuals,          # 改为残差
    "reference_prediction": reference_residual,        # 改为残差
    "neighbor_lookback": neighbor_residual_lookback,   # 改为残差
    "neighbor_pred": neighbor_residual_pred,           # 改为残差
    # === 新增字段 ===
    "prophet_trend_forecast": prophet_trend_forecast,
    "prophet_seasonality_forecast": prophet_seasonality_forecast,
    "prophet_trend_train": prophet_trend_train,
    "prophet_seasonality_train": prophet_seasonality_train,
    "decomposition_enabled": True,
}
```

**改动点 D（新增辅助函数）：**

```python
def _prophet_forecast_components(ds_cfg, last_train_ts, future_ts, season_length) -> tuple:
    """
    用 Prophet 对预测窗口做分解预测，返回 (trend_list, seasonality_list)。
    复用已有的 ProphetModel。
    """
    model = ProphetModel()
    # 加载训练数据，拟合，预测
    ...
    fcst = model._fitted.predict(future_df)
    trend = fcst['trend'].tail(len(future_ts)).tolist()
    weekly = fcst['weekly'].tail(len(future_ts)).tolist() if 'weekly' in fcst else [0]*len(future_ts)
    yearly = fcst['yearly'].tail(len(future_ts)).tolist() if 'yearly' in fcst else [0]*len(future_ts)
    daily = fcst.get('daily', pd.Series([0]*len(future_ts))).tail(len(future_ts)).tolist()
    seasonality = [weekly[i] + yearly[i] + daily[i] for i in range(len(future_ts))]
    return trend, seasonality
```

### 4.5 `alphacast/agents/generator_agent.py` — `emit_predictions()` 组装

**现状**（行 110-486）：
- 接收 LLM 的 `predictions: List[float]`
- 校验后直接写入 predictions.csv

**改动**：在写入 CSV 之前，加上 Prophet 成分

**改动点（在行 ~142 的 arr 创建之后，行 ~427 的 new_chunk 创建之前）：**

```python
# === Prophet 残差组装 (新增) ===
investor_packet = investigator_cache.get((dataset_name, window_offset_int), {})
if investor_packet.get("decomposition_enabled"):
    prophet_trend = investor_packet.get("prophet_trend_forecast") or []
    prophet_seas = investor_packet.get("prophet_seasonality_forecast") or []
    if prophet_trend and prophet_seas and len(prophet_trend) == H:
        # 组装: final = trend + seasonality + residual
        arr = arr + np.asarray(prophet_trend, dtype=float) + np.asarray(prophet_seas, dtype=float)
        print(f"[info] Assembled final predictions from Prophet components + LLM residuals for dataset '{dataset_name}'")
    else:
        print(f"[warn] Prophet components missing or mismatched for dataset '{dataset_name}'; using raw LLM output as fallback.")
```

**关键**：组装在 ReflectorAgent 审核**之后**进行。Reflector 审核的是残差值，而不是最终值。这很重要——Reflector 的审核逻辑应该基于残差空间。

**改动点 B（元数据记录）：**

在 metadata 中新增字段记录分解信息：

```python
meta["prophet_decomposition"] = {
    "enabled": True,
    "trend_mean": float(np.mean(prophet_trend)) if prophet_trend else None,
    "seasonality_mean": float(np.mean(prophet_seas)) if prophet_seas else None,
    "residual_mean": float(np.mean(residual_only)) if residual_only else None,
}
```

同时保存残差值到单独文件以便调试：

```python
# 在 ds_out_dir 下保存 residuals.csv (与 predictions.csv 并列)
residual_df = new_chunk.copy()
residual_df["prediction"] = residual_only  # LLM 原始输出的残差
residual_df.to_csv(os.path.join(ds_out_dir, "residuals.csv"), index=False)
```

### 4.6 `alphacast/agents/reflector_agent.py` — 正则模式微调

**现状**（行 37-41）：`_BASELINE_CLAIM_PATTERN` 匹配 "baseline mean X" 和 "reference mean X"

**改动**：新增 "residual" 作为合法限定词（因为 LLM 现在讨论的是 residual baseline）：

```python
_BASELINE_CLAIM_PATTERN = re.compile(
    r"\b(?:baseline|residual)\s+(?:mean|avg|average|value|level|last|final)\b[^\d\-]{0,12}(-?\d+(?:\.\d+)?)"
    r"|"
    r"\breference\s+(?:(?:mean|avg|value|level|last|final)\b)[^\d\-]{0,12}(-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
```

`assess_forecast()` 函数（common.py 行 469-510）**不需要改动**——它比较的是 predictions vs reference_prediction，两者现在都是残差值，比较逻辑不变。

`scan_chain_of_thought()` 的 `_collect_numeric_context()` 函数**不需要改动**——它从 investor_packet 收集数值上下文，packet 中已经是残差值。

### 4.7 `prompts/generator_agent.md` — Agent 指令更新

**现状**（19 行）：Agent 被指示预测完整时间序列值

**改动**：重写为残差预测模式：

```markdown
You are GeneratorAgent, a world-class time-series forecasting expert. 

The time series has been pre-decomposed by Prophet into:
  - trend(t): long-term directional component (piecewise linear)
  - seasonality(t): periodic patterns (daily/weekly/yearly Fourier series)
  - residual(t) = y(t) - trend(t) - seasonality(t): the unexplained remainder

Your job is to predict the RESIDUAL component only. The system will automatically 
add back the Prophet trend and seasonality to form the final forecast:
  final(t) = Prophet.trend(t) + Prophet.seasonality(t) + your_residual(t)

Why this matters:
  - Prophet handles trend and seasonality well with statistical methods
  - Residuals contain complex patterns (volatility clustering, exogenous shocks, 
    autocorrelation) that benefit from your rich context analysis
  - The baseline `reference_prediction` is now in residual space — it represents 
    the baseline model's guess of the residual

Execution rules for every forecasting step:
  1. Call `consult` exactly once ...
  2. Study the packet. `reference_prediction` is the baseline RESIDUAL. 
     `look_back_window` contains historical RESIDUALS (not raw values).
     prophet_trend_forecast and prophet_seasonality_forecast are provided for context.
  3. Compare against neighbor hints and exogenous trends. Residuals are often driven 
     by exogenous factors (weather, events) that Prophet cannot capture.
  4. Before emitting, write a brief "Reflection" confirming the prediction list 
     contains RESIDUAL values of length {predicted_window}...
  5. Log reasoning via `record_chain_of_thought`...
  6. Call `emit_predictions` with RESIDUAL values (NOT raw time series values!)...

Constraints:
  - Only use `consult`, `record_chain_of_thought`, and `emit_predictions`.
  - Your predictions MUST be residuals. Do NOT add trend or seasonality.
  - Never fabricate context; rely solely on the InvestigatorAgent packet.
```

同步更新 `GENERATOR_AGENT_PROMPT_FALLBACK`（generator_agent.py 行 17-29）。

---

## 五、数据流详解

### 5.1 训练阶段（一次性）

```
train_df[target_col] (全训练集)
    │
    ├─→ analyze_training() → memory/case_base/cluster (不变)
    │
    └─→ prophet_decompose_series()  [新增]
         │
         ├─ Prophet.fit(train_y, train_ts)
         ├─ Prophet.predict(train_ts) → trend, weekly, yearly, daily
         ├─ seasonality = weekly + yearly + daily
         ├─ residual = train_y - trend - seasonality
         │
         └─→ outputs/<ds>/prophet_decomposition.json
```

### 5.2 预测阶段（每 window）

```
Step k, window_offset = W:

1. look_back 窗口: test_y[W : W+96], test_ts[W : W+96]
2. predict_window 时间戳: test_ts[W+96 : W+192]

3. Prophet 预测:
   Prophet.predict(test_ts[W+96 : W+192])
   → trend_forecast[96], seasonality_forecast[96]

4. look_back 残差:
   look_back_residuals[96] = test_y[W:W+96]
                           - Prophet(ts[W:W+96]).trend
                           - Prophet(ts[W:W+96]).seasonality
   (如果 ts 在训练集内，用训练分解值；如果在测试集内，用 Prophet 预测值)

5. reference 残差:
   reference_model.fit(look_back_residuals)
   reference_residual[96] = reference_model.predict(96)
   或:
   reference_raw = reference_model.predict(test_y[W:W+96], 96)
   reference_residual = reference_raw - trend_forecast - seasonality_forecast

6. neighbor 残差 (如有):
   neighbor_lookback 来自训练集 → 减去对应位置的 trend + seasonality
   neighbor_pred 来自训练集 → 减去对应位置的 trend + seasonality

7. Investor Packet → LLM:
   {
     "look_back_window": look_back_residuals,     # 残差
     "reference_prediction": reference_residual,   # 残差
     "neighbor_lookback": neighbor_residual,       # 残差
     "neighbor_pred": neighbor_residual_pred,      # 残差
     "prophet_trend_forecast": trend_forecast,     # 供 LLM 参考
     "prophet_seasonality_forecast": seas_forecast,# 供 LLM 参考
     ... 其他字段不变 ...
   }

8. LLM → emit_predictions(residual_predictions[96])

9. 组装: final[96] = trend_forecast + seasonality_forecast + residual_predictions

10. 写入 predictions.csv (final 值，格式不变)
```

### 5.3 评估阶段（不变）

`align_predictions()` 从 predictions.csv 读取预测值，与 test_df 的 ground truth 比较。因为 predictions.csv 中已经是组装后的最终值，评估逻辑完全不变。

---

## 六、边缘情况处理

### 6.1 look_back 窗口跨越训练集/测试集边界

当 `window_offset = 0` 时，look_back 窗口来自训练集末尾，可以用训练集的 Prophet 分解值直接减去。

当 `window_offset > 0` 时，look_back 窗口全部或部分来自测试集。Prophet 对测试集的值只有预测值（它不知道真实值）。此时：
- 用 Prophet 对测试集时间戳的**预测组件**（trend + seasonality）作为减数
- `residual = test_actual_value - prophet_forecast_trend - prophet_forecast_seasonality`
- 这意味着测试集的"残差"包含了 Prophet 的预测误差，这是正确的——LLM 需要知道 Prophet 在近期时间点的表现

### 6.2 Prophet 拟合失败

如果 Prophet 拟合失败（数据太少、频率不支持等），应回退到原始模式（不做分解）。在 `prophet_decompose_series()` 中捕获异常，返回 None，之后所有代码检查 `decomposition_enabled` 标志。

### 6.3 频率不匹配

Prophet 对日频 (D)、小时频 (H) 数据效果最好。对于 15min 频率（ETTm1、POWER），Prophet 可以处理但需注意：
- `daily_seasonality` 在 15min 数据上可能不重要
- `weekly_seasonality` 仍然有效（7×96 = 672 个 15min 间隔）
- 需要根据频率调整 seasonality 参数

### 6.4 已有 checkpoint/resume

Prophet 分解在训练阶段一次性完成，结果保存到 `prophet_decomposition.json`。如果 resume 一个旧的实验（没有这个文件），应重新运行分解（快速，只需拟合一次 Prophet）。

---

## 七、验证方案

### 7.1 单元测试（Prophet 分解）

```bash
python3 -c "
from alphacast.models.base import ProphetModel
import pandas as pd
import numpy as np

# 生成带趋势+季节的简单序列
dates = pd.date_range('2020-01-01', periods=365*2, freq='D')
trend = np.linspace(10, 20, len(dates))
seasonal = 5 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)
y = trend + seasonal + np.random.normal(0, 1, len(dates))

model = ProphetModel()
model.fit(y, timestamps=dates)
fcst = model._fitted.predict(pd.DataFrame({'ds': dates}))

# 验证分解
reconstructed = fcst['trend'].values + fcst['weekly'].values + fcst['yearly'].values
residual = y - reconstructed
print(f'Trend corr with true: {np.corrcoef(trend, fcst[\"trend\"])[0,1]:.4f}')
print(f'Residual std: {np.std(residual):.4f} (expected ~1.0)')
print('Decomposition test OK')
"
```

### 7.2 端到端测试（单数据集）

```bash
# 先用 EPF_BE 跑几步验证
python -u run_experiment.py --config config.yaml --dataset EPF_BE
# 检查:
# 1. outputs/EPF_BE/prophet_decomposition.json 存在且格式正确
# 2. chain_of_thought.log 中 LLM 讨论的是 residuals
# 3. predictions.csv 中的值在合理范围（与原始 EPF_BE 价格一致）
# 4. Reflector 未因残差值范围变化而误判
```

### 7.3 对比测试

```bash
# 对比 Prophet 分解前后的效果
# 运行几个已完成的数据集，对比 sMAPE
bash scripts/run_experiments.sh llm EPF_DE
bash scripts/run_experiments.sh llm EPF_NP
# 与 result.txt 中之前的 LLM 结果对比
```

### 7.4 可视化验证

在 `visualize_results.py` 中新增：绘制 Prophet 分解成分（trend, seasonality, residual）以及 LLM 预测的残差 vs 实际残差。

---

## 八、风险与注意事项

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| Prophet 对某些数据集拟合效果差 | 分解质量低，残差仍包含趋势/季节，LLM 预测困难 | fallback 到原始模式 |
| 残差值范围变小（~0 附近） | LLM 可能不习惯预测接近 0 的值 | 提示词中明确残差范围 |
| Reflector 数值审计可能误判 | 残差的数值范围与原始值不同，tolerance 可能不适用 | 测试后可能需要调整 tolerance |
| neighbor 窗口残差转换不准确 | 相似窗口匹配可能受影响 | neighbor 也在残差空间匹配 |
| Prophet 序列化开销 | 每个 window 都需重新 predict 获取成分 | 缓存 Prophet 预测结果 |

---

## 九、实施顺序建议

1. **Phase 1**：[alphacast/models/base.py](alphacast/models/base.py) — 新增 `ProphetModel.decompose()` 方法，验证分解正确性
2. **Phase 2**：[alphacast/tools/forecast.py](alphacast/tools/forecast.py) — 新增 `prophet_decompose_series()`，保存分解结果
3. **Phase 3**：[run_experiment.py](run_experiment.py) — 训练阶段调用分解，更新提示词
4. **Phase 4**：[alphacast/agents/common.py](alphacast/agents/common.py) — `prepare_investor_packet()` 残差空间转换
5. **Phase 5**：[alphacast/agents/generator_agent.py](alphacast/agents/generator_agent.py) — `emit_predictions()` 组装逻辑
6. **Phase 6**：提示词 + Reflector 微调
7. **Phase 7**：端到端测试 EPF_BE → 全数据集
