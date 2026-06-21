import pandas as pd
import plotly.express as px
import streamlit as st
from config import MODEL_COLORS
from ui import layout
from utils import norm, find_col

MODEL_CANDIDATES = ["model", "scenario", "skenario", "method", "metode", "variant"]
HORIZON_CANDIDATES = ["horizon", "h", "step", "forecast_horizon"]
TICKER_CANDIDATES = ["ticker", "symbol", "emiten", "kode_saham"]
METRIC_NAME = ["metric", "metrics", "metrik", "measure"]
VALUE_COLS = ["value", "nilai", "score", "hasil"]
ALIASES = {
    "RMSE": ["rmse", "rootmeansquarederror"],
    "MAE": ["mae", "meanabsoluteerror"],
    "MAPE": ["mape", "meanabsolutepercentageerror"],
    "R²": ["r2", "rsquared", "rsquare"],
    "Directional Accuracy": ["directionalaccuracy", "diracc", "dir_acc", "da"],
}
ORDER = ["RMSE", "MAE", "MAPE", "R²", "Directional Accuracy"]

def normalize_model(value):
    low = str(value).lower()
    if "llm" in low or "hybrid" in low or "sent" in low or "s1" in low:
        return "LLM-TFT"
    if "tft" in low or "base" in low or "s5" in low:
        return "TFT"
    return str(value)

def metric_name(col):
    low = norm(col)
    for name, aliases in ALIASES.items():
        if any(norm(alias) in low for alias in aliases):
            return name
    return str(col).replace("_", " ").title()

def metric_cols(df):
    numeric = list(df.select_dtypes(include="number").columns)
    return [c for c in numeric if norm(c) not in {"n", "count", "jumlah", "index"}]

def long_eval(df, scope):
    if df is None or df.empty:
        return pd.DataFrame()
    work = df.copy()
    mcol = find_col(work, MODEL_CANDIDATES)
    hcol = find_col(work, HORIZON_CANDIDATES)
    tcol = find_col(work, TICKER_CANDIDATES)
    ncol = find_col(work, METRIC_NAME)
    vcol = find_col(work, VALUE_COLS)
    if not mcol:
        work["Model"] = ["TFT", "LLM-TFT"][:len(work)] if len(work) <= 2 else "Model"
        mcol = "Model"
    rows = []
    if ncol and vcol:
        for _, row in work.iterrows():
            rows.append(row_dict(row, mcol, ncol, vcol, hcol, tcol))
    else:
        for idx, row in work.iterrows():
            horizon = row.get(hcol) if hcol else (f"H+{(idx % 3) + 1}" if scope == "horizon" else None)
            for col in metric_cols(work):
                rows.append({"Model": normalize_model(row[mcol]), "Metric": metric_name(col),
                             "Value": pd.to_numeric(row[col], errors="coerce"),
                             "Horizon": horizon, "Ticker": row.get(tcol) if tcol else None})
    return pd.DataFrame(rows).dropna(subset=["Value"])

def row_dict(row, mcol, ncol, vcol, hcol, tcol):
    return {"Model": normalize_model(row[mcol]), "Metric": metric_name(row[ncol]),
            "Value": pd.to_numeric(row[vcol], errors="coerce"),
            "Horizon": row.get(hcol), "Ticker": row.get(tcol)}

def show_eval(title, df, scope, xfield=None):
    data = long_eval(df, scope)
    if data.empty:
        st.info(f"{title} belum dapat ditampilkan.")
        return
    xaxis = xfield if xfield and data[xfield].notna().any() else "Model"
    metrics = [m for m in ORDER if m in data["Metric"].unique().tolist()]
    metrics += [m for m in data["Metric"].drop_duplicates().tolist() if m not in metrics]
    cols = st.columns(2)
    for i, metric in enumerate(metrics):
        subset = data[data["Metric"].eq(metric)]
        with cols[i % 2]:
            draw_chart(subset, metric, scope, xaxis)

def draw_chart(data, metric, scope, xaxis):
    if scope == "horizon":
        fig = px.line(data, x=xaxis, y="Value", color="Model", markers=True,
                      title=metric, color_discrete_map=MODEL_COLORS)
        fig.update_traces(line=dict(width=3), marker=dict(size=9))
    else:
        fig = px.bar(data, x=xaxis, y="Value", color="Model", barmode="group",
                     title=metric, color_discrete_map=MODEL_COLORS, text_auto=True)
    st.plotly_chart(layout(fig, 380), use_container_width=True)
