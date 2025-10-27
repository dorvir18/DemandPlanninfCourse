# pipelines/build_artifacts.py
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing, ETSModel

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data"
PUBLIC = ROOT / "docs" / "data"
PUBLIC.mkdir(parents=True, exist_ok=True)

H = 6               # горизонт прогноза (месяцев)
SEASON_M = 12       # сезонность по умолчанию для месячных рядов
HOLDOUT = 3         # длина holdout для метрик

# ----------------------- Метрики -----------------------
def mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((y_true[mask]-y_pred[mask]) / y_true[mask]))*100)

def smape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denom != 0
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs(y_true[mask]-y_pred[mask]) / denom[mask]) * 100)

def mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))

def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred)**2)))

def mase(y_true, y_pred, y_insample, seasonality=1):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_insample = np.asarray(y_insample, dtype=float)
    mae_model = np.mean(np.abs(y_true - y_pred))
    if len(y_insample) <= seasonality:
        return np.nan
    naive = np.abs(y_insample[seasonality:] - y_insample[:-seasonality])
    mae_naive = np.mean(naive) if len(naive) else np.nan
    return float(mae_model / mae_naive) if mae_naive and mae_naive != 0 else np.nan

# -------------------- Простейшие модели --------------------
def fc_naive(s: pd.Series, h: int) -> np.ndarray:
    last = s.iloc[-1]
    return np.repeat(last, h)

def fc_snaive(s: pd.Series, h: int, m: int) -> np.ndarray:
    if len(s) < m:
        # fallback к Naive
        return fc_naive(s, h)
    # повторяем последний сезонный паттерн
    pattern = s.iloc[-m:].values
    reps = int(np.ceil(h / m))
    res = np.tile(pattern, reps)[:h]
    return res

def fc_ma(s: pd.Series, h: int, k_list=(3,6,12)) -> Tuple[np.ndarray, int]:
    # выбор k по минимальному MAE на rolling one-step (простой критерий)
    best_k, best_mae = k_list[0], np.inf
    for k in k_list:
        if len(s) <= k:
            continue
        preds = s.rolling(k).mean().shift(1).dropna()
        common = s.loc[preds.index]
        cur_mae = mae(common.values, preds.values)
        if cur_mae < best_mae:
            best_mae, best_k = cur_mae, k
    k = best_k
    mean_last = s.iloc[-k:].mean() if len(s) >= k else s.mean()
    return np.repeat(mean_last, h), k

# -------------------- Экспоненциальное сглаживание --------------------
def fc_ses(s: pd.Series, h: int) -> np.ndarray:
    fit = ExponentialSmoothing(s, trend=None, seasonal=None).fit(optimized=True, use_brute=True)
    return fit.forecast(h).values

def fc_hw(s: pd.Series, h: int, m: int) -> np.ndarray:
    # Holt–Winters additive сезонность, без тренда (как базовый вариант)
    fit = ExponentialSmoothing(s, trend="add", seasonal="add", seasonal_periods=m).fit(optimized=True, use_brute=True)
    return fit.forecast(h).values

def fc_ets_auto(s: pd.Series, h: int, m: int) -> np.ndarray:
    # Простой автоподбор по сетке: (trend ∈ {None,'add'}, seasonal ∈ {None,'add'})
    # Берём модель с минимальным AICc
    candidates = [
        dict(trend=None, seasonal=None),
        dict(trend="add", seasonal=None),
        dict(trend=None, seasonal="add"),
        dict(trend="add", seasonal="add"),
    ]
    best_fit, best_ic = None, np.inf
    for c in candidates:
        try:
            if c["seasonal"] is None:
                mod = ETSModel(s, error="add", trend=c["trend"], seasonal=None)
            else:
                mod = ETSModel(s, error="add", trend=c["trend"], seasonal=c["seasonal"], seasonal_periods=m)
            fit = mod.fit()
            ic = fit.aicc if hasattr(fit, "aicc") else fit.aic
            if ic < best_ic:
                best_ic, best_fit = ic, fit
        except Exception:
            continue
    if best_fit is None:
        return fc_ses(s, h)  # fallback
    return best_fit.forecast(h).values

# -------------------- ARIMA / SARIMA / ARIMAX --------------------
def fc_arima(s: pd.Series, h: int) -> np.ndarray:
    fit = SARIMAX(s, order=(1,1,1), seasonal_order=(0,0,0,0),
                  enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    return fit.forecast(h).values

def fc_sarima(s: pd.Series, h: int, m: int) -> np.ndarray:
    # простая сезонная SARIMA
    fit = SARIMAX(s, order=(1,1,1), seasonal_order=(1,1,1,m),
                  enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    return fit.forecast(h).values

def month_exog(index: pd.DatetimeIndex) -> pd.DataFrame:
    # 11 дамми-месяцев (один — базовый), чтобы иметь валидный future exog
    months = index.month
    df = pd.get_dummies(months)
    # переименуем 1..12 → m1..m12
    df.columns = [f"m{int(c)}" for c in df.columns]
    # удалим одну колонку для избежания мультиколлинеарности
    if "m12" in df.columns:
        df = df.drop(columns=["m12"])
    return df.astype(float)

def fc_arimax(s: pd.Series, h: int) -> np.ndarray:
    # ARIMAX c экзогенными month-dummies (без внешних источников — пригодно для Pages)
    exog_train = month_exog(s.index)
    future_idx = pd.date_range(s.index[-1] + pd.offsets.MonthBegin(), periods=h, freq="MS")
    exog_future = month_exog(future_idx)
    fit = SARIMAX(s, order=(1,1,1), seasonal_order=(0,0,0,0),
                  exog=exog_train, enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    return fit.forecast(h, exog=exog_future).values

# -------------------- Intermittent demand: Croston / SBA / TSB --------------------
def _split_demand_intervals(y: np.ndarray):
    # возвращает последовательности ненулевых величин и интервалов между ними
    sizes, intervals = [], []
    gap = 0
    for val in y:
        if val > 0:
            sizes.append(val)
            intervals.append(gap+1)
            gap = 0
        else:
            gap += 1
    if len(sizes) == 0:
        # все нули
        sizes = [0.0]; intervals = [max(1, gap)]
    return np.array(sizes, dtype=float), np.array(intervals, dtype=float)

def croston_classic(y: np.ndarray, alpha=0.1, h=6):
    sizes, intervals = _split_demand_intervals(y)
    z = sizes[0] if len(sizes) else 0.0
    p = intervals[0] if len(intervals) else 1.0
    for i in range(1, len(sizes)):
        z = z + alpha * (sizes[i] - z)
        p = p + alpha * (intervals[i] - p)
    rate = z / p if p != 0 else 0.0
    return np.repeat(rate, h)

def croston_sba(y: np.ndarray, alpha=0.1, h=6):
    fc = croston_classic(y, alpha=alpha, h=h)
    # SBA (Syntetos–Boylan Approximation) — поправка на смещение
    return fc * (1 - alpha/2)

def tsb(y: np.ndarray, alpha=0.1, beta=0.1, h=6):
    # Teunter–Syntetos–Babai: сглаживаем отдельно размер спроса и вероятность появления
    z = 0.0   # размер спроса при наличии
    p = 0.0   # вероятность ненулевого спроса
    first = True
    for val in y:
        occ = 1.0 if val > 0 else 0.0
        if first:
            z = val if val>0 else 0.0
            p = occ
            first = False
        else:
            z = z + alpha * ((val if val>0 else z) - z)
            p = p + beta * (occ - p)
    return np.repeat(z * p, h)

def fc_croston_family(s: pd.Series, h: int, variant="classic") -> np.ndarray:
    y = s.values.astype(float)
    zero_share = (y==0).mean()
    alpha = 0.2
    if variant == "classic":
        return croston_classic(y, alpha=alpha, h=h)
    elif variant == "sba":
        return croston_sba(y, alpha=alpha, h=h)
    elif variant == "tsb":
        return tsb(y, alpha=0.2, beta=0.2, h=h)
    else:
        raise ValueError("unknown variant")

# -------------------- Вспомогательные процедуры --------------------
def horizon_index(last_idx: pd.DatetimeIndex, h: int) -> pd.DatetimeIndex:
    return pd.date_range(last_idx[-1] + pd.offsets.MonthBegin(), periods=h, freq="MS")

def compute_all_models(s: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Возвращает:
      - fc_df: прогноз на горизонт H по всем моделям (колонки yhat_*)
      - hold_df: holdout-таблица: ds, y (факт), и предсказания моделей на holdout
    """
    s = s.asfreq("MS")
    s = s.fillna(method="ffill")

    # Holdout сплит
    split = -HOLDOUT if len(s) > (HOLDOUT+SEASON_M) else -max(1, HOLDOUT)
    train, test = s[:split], s[split:]
    h_idx = horizon_index(s.index, H)

    # ---- Прогнозы на горизонт
    yhat_naive  = fc_naive(s, H)
    yhat_snaive = fc_snaive(s, H, SEASON_M)
    yhat_ma, _k = fc_ma(s, H)
    yhat_ses    = fc_ses(s, H)
    yhat_hw     = fc_hw(s, H, SEASON_M)
    yhat_ets    = fc_ets_auto(s, H, SEASON_M)
    yhat_arima  = fc_arima(s, H)
    yhat_sarima = fc_sarima(s, H, SEASON_M)
    yhat_arimax = fc_arimax(s, H)
    yhat_crost  = fc_croston_family(s, H, "classic")
    yhat_sba    = fc_croston_family(s, H, "sba")
    yhat_tsb    = fc_croston_family(s, H, "tsb")

    fc_df = pd.DataFrame({
        "ds": h_idx,
        "yhat_naive":  yhat_naive,
        "yhat_snaive": yhat_snaive,
        "yhat_ma":     yhat_ma,
        "yhat_ses":    yhat_ses,
        "yhat_hw":     yhat_hw,
        "yhat_ets":    yhat_ets,
        "yhat_arima":  yhat_arima,
        "yhat_sarima": yhat_sarima,
        "yhat_arimax": yhat_arimax,
        "yhat_croston":yhat_crost,
        "yhat_sba":    yhat_sba,
        "yhat_tsb":    yhat_tsb,
    })

    # ---- Прогнозы на holdout (пересчитываем модели на train)
    def one_step_all(train: pd.Series, steps: int) -> Dict[str, np.ndarray]:
        res = {}
        res["naive"]   = fc_naive(train, steps)
        res["snaive"]  = fc_snaive(train, steps, SEASON_M)
        res["ma"], _k  = fc_ma(train, steps)
        res["ses"]     = fc_ses(train, steps)
        res["hw"]      = fc_hw(train, steps, SEASON_M)
        res["ets"]     = fc_ets_auto(train, steps, SEASON_M)
        res["arima"]   = fc_arima(train, steps)
        res["sarima"]  = fc_sarima(train, steps, SEASON_M)
        res["arimax"]  = fc_arimax(train, steps)
        res["croston"] = fc_croston_family(train, steps, "classic")
        res["sba"]     = fc_croston_family(train, steps, "sba")
        res["tsb"]     = fc_croston_family(train, steps, "tsb")
        return res

    hold_preds = one_step_all(train, len(test))
    hold_df = pd.DataFrame({"ds": test.index, "y": test.values})
    for k, v in hold_preds.items():
        hold_df[f"yhat_{k}"] = v

    return fc_df, hold_df, train

def metrics_table(hold_df: pd.DataFrame, train: pd.Series) -> pd.DataFrame:
    models = [c for c in hold_df.columns if c.startswith("yhat_")]
    rows = []
    for metric_name in ["MAPE","sMAPE","MAE","RMSE","MASE"]:
        rec = {"metric": metric_name}
        for m in models:
            y_true = hold_df["y"].values
            y_pred = hold_df[m].values
            if metric_name == "MAPE":
                val = mape(y_true, y_pred)
            elif metric_name == "sMAPE":
                val = smape(y_true, y_pred)
            elif metric_name == "MAE":
                val = mae(y_true, y_pred)
            elif metric_name == "RMSE":
                val = rmse(y_true, y_pred)
            elif metric_name == "MASE":
                val = mase(y_true, y_pred, train.values, seasonality=1)
            else:
                val = np.nan
            rec[m.replace("yhat_","").upper()] = float(val) if val is not None else None
        rows.append(rec)
    return pd.DataFrame(rows)

# ------------------------- Основной цикл -------------------------
def forecast_for_pair(df_pair: pd.DataFrame):
    # Берём sell-out как предсказываемую величину
    s = df_pair.set_index("date")["shipments_from_client"].asfreq("MS").astype(float)
    s = s.fillna(method="ffill")
    fc_df, hold_df, train = compute_all_models(s)
    met_df = metrics_table(hold_df, train)
    return fc_df, met_df, hold_df

def main():
    ts = pd.read_csv(SRC / "time_series.csv", parse_dates=["date"])
    agg = ts.groupby(["client_id","product_id"])["shipments_from_client"].sum().sort_values(ascending=False)
    top_pairs = agg.head(5).index.tolist()

    catalog = []
    for cid, pid in top_pairs:
        dfp = ts[(ts.client_id==cid) & (ts.product_id==pid)].sort_values("date").copy()
        key = f"{cid}_{pid}"
        fc, met, hold = forecast_for_pair(dfp)

        # ---- save artifacts
        fc.to_json(PUBLIC / f"forecast_{key}.json", orient="records", date_format="iso")
        met.to_json(PUBLIC / f"metrics_{key}.json", orient="records")
        hold.to_json(PUBLIC / f"holdout_{key}.json", orient="records", date_format="iso")

        catalog.append({"client_id":cid, "product_id":pid, "key":key})

    pd.DataFrame(catalog).to_json(PUBLIC / "catalog.json", orient="records")

if __name__ == "__main__":
    main()

