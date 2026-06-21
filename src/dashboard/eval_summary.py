import streamlit as st
from eval_utils import long_eval
from utils import fmt

LOWER_IS_BETTER = {"RMSE", "MAE", "MAPE"}
METRIC_ORDER = ["RMSE", "MAE", "MAPE", "R²", "Directional Accuracy"]

def best_row(df, metric):
    data = df[df["Metric"].eq(metric)].copy()
    if data.empty:
        return None
    idx = data["Value"].idxmin() if metric in LOWER_IS_BETTER else data["Value"].idxmax()
    return data.loc[idx]

def metric_note(metric):
    return "lebih kecil lebih baik" if metric in LOWER_IS_BETTER else "lebih besar lebih baik"

def show_summary_cards(eval_global):
    data = long_eval(eval_global, "global")
    st.subheader("Ringkasan Performa")
    if data.empty:
        st.info("Ringkasan evaluasi belum tersedia.")
        return
    metrics = [m for m in METRIC_ORDER if m in data["Metric"].unique().tolist()]
    if not metrics:
        st.info("Metrik utama belum ditemukan pada file evaluasi global.")
        return
    cols = st.columns(min(len(metrics), 5))
    for col, metric in zip(cols, metrics):
        row = best_row(data, metric)
        if row is None:
            continue
        decimals = 3 if metric in {"MAPE", "R²", "Directional Accuracy"} else 0
        suffix = "%" if metric in {"MAPE", "Directional Accuracy"} else ""
        value = f"{fmt(row['Value'], decimals)}{suffix}"
        col.metric(metric, value, f"{row['Model']} · {metric_note(metric)}")

def show_best_model(eval_global):
    data = long_eval(eval_global, "global")
    if data.empty:
        return
    priority = [m for m in ["RMSE", "MAE", "MAPE"] if m in data["Metric"].unique().tolist()]
    if not priority:
        return
    row = best_row(data, priority[0])
    if row is not None:
        st.success(f"Model terbaik berdasarkan {priority[0]} adalah {row['Model']}.")
