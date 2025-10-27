import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data"
PUBLIC = ROOT / "docs" / "data"
PUBLIC.mkdir(parents=True, exist_ok=True)

def mape(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask]-y_pred[mask]) / y_true[mask]))*100)

def smape(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred))/2
    mask = denom != 0
    return float(np.mean(np.abs(y_true[mask]-y_pred[mask]) / denom[mask]) * 100)

def mase(y_true, y_pred, y_insample, seasonality=1):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mae_model = np.mean(np.abs(y_true - y_pred))
    naive = np.abs(y_insample[seasonality:] - y_insample[:-seasonality])
    mae_naive = np.mean(naive)
    return float(mae_model / mae_naive) if mae_naive != 0 else np.nan

def forecast_for_pair(df_pair):
    # df_pair: rows for one client_id & product_id, sorted by date (monthly)
    s = df_pair.set_index("date")["shipments_from_client"].asfreq("MS")
    s = s.fillna(method="ffill")

    # SES baseline
    ses_fit = ExponentialSmoothing(s, trend=None, seasonal=None).fit(optimized=True, use_brute=True)
    ses_fc = ses_fit.forecast(6)

    # ARIMA demo
    arima_fit = SARIMAX(s, order=(1,1,1), seasonal_order=(0,0,0,0),
                        enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    arima_fc = arima_fit.forecast(6)

    horizon = pd.date_range(s.index[-1] + pd.offsets.MonthBegin(), periods=6, freq="MS")
    out = pd.DataFrame({
        "ds": horizon,
        "yhat_ses": ses_fc.values,
        "yhat_arima": arima_fc.values
    })
    # metrics on last 3 months
    split = -3 if len(s) > 6 else -1
    train, test = s[:split], s[split:]
    ses2 = ExponentialSmoothing(train, trend=None, seasonal=None).fit(optimized=True, use_brute=True)
    ses_pred = ses2.forecast(len(test))
    arima2 = SARIMAX(train, order=(1,1,1)).fit(disp=False)
    arima_pred = arima2.forecast(len(test))

    metrics = {
        "MAPE": {"SES": mape(test.values, ses_pred.values), "ARIMA": mape(test.values, arima_pred.values)},
        "sMAPE": {"SES": smape(test.values, ses_pred.values), "ARIMA": smape(test.values, arima_pred.values)},
        "MASE": {"SES": mase(test.values, ses_pred.values, train.values, 1), "ARIMA": mase(test.values, arima_pred.values, train.values, 1)}
    }
    return out, metrics

def main():
    ts = pd.read_csv(SRC / "time_series.csv", parse_dates=["date"])
    # choose a default pair for the homepage demo: the largest by sum of sell-out
    agg = ts.groupby(["client_id","product_id"])["shipments_from_client"].sum().sort_values(ascending=False)
    default_cid, default_pid = agg.index[0]

    # produce artifacts for the top 5 pairs
    top_pairs = agg.head(5).index.tolist()

    catalog = []
    for cid, pid in top_pairs:
        dfp = ts[(ts.client_id==cid)&(ts.product_id==pid)].sort_values("date").copy()
        fc, met = forecast_for_pair(dfp)
        # save artifacts
        key = f"{cid}_{pid}"
        fc.to_json(PUBLIC / f"forecast_{key}.json", orient="records", date_format="iso")
        pd.DataFrame([
            {"metric":"MAPE", "SES":met["MAPE"]["SES"], "ARIMA":met["MAPE"]["ARIMA"]},
            {"metric":"sMAPE", "SES":met["sMAPE"]["SES"], "ARIMA":met["sMAPE"]["ARIMA"]},
            {"metric":"MASE", "SES":met["MASE"]["SES"], "ARIMA":met["MASE"]["ARIMA"]},
        ]).to_json(PUBLIC / f"metrics_{key}.json", orient="records")
        catalog.append({"client_id":cid, "product_id":pid, "key":key})

    # write a small catalog for the front-end to list available pairs
    pd.DataFrame(catalog).to_json(PUBLIC / "catalog.json", orient="records")

if __name__ == "__main__":
    main()
