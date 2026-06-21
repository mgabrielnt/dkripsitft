import pandas as pd
import plotly.express as px
import streamlit as st
from combined_chart import combined_chart
from config import COLORS
from forecast import predict_checkpoints, extract_predictions, prediction_rows
from style import header
from ui import action_button, layout
from utils import date_col, filter_df, fmt


def selected_predictions(pred_df):
    if pred_df is None or pred_df.empty:
        return {"H+1": None, "H+2": None, "H+3": None}
    selected = pred_df[pred_df["Series"].eq("LLM-TFT")]
    selected = selected if not selected.empty else pred_df
    return {f"H+{int(row.Step)}": row.Harga for row in selected.itertuples()}


def sorted_data(df):
    if df is None or df.empty:
        return df
    sort_col = date_col(df) or ("time_idx" if "time_idx" in df.columns else None)
    return df.sort_values(sort_col) if sort_col else df


def latest_close(master):
    if master is None or master.empty or "close" not in master.columns:
        return None
    close = sorted_data(master)["close"].dropna()
    return float(close.tail(1).iloc[-1]) if not close.empty else None


def render(data, ticker, dates):
    header("Model dan Prediksi")
    with st.expander("Jalankan Proses"):
        c1, c2, c3, c4 = st.columns(4)
        with c1: action_button("Dataset", "Bangun dataset master")
        with c2: action_button("Latih TFT", "Latih TFT")
        with c3: action_button("Latih LLM-TFT", "Latih LLM-TFT")
        with c4: action_button("Backtest", "Backtest model")
    master = filter_df(data["master"], ticker, dates)
    pred_file = filter_df(data["predictions"], ticker, dates)
    dc = date_col(master)
    cutoff = pd.to_datetime(master[dc]).max() if master is not None and not master.empty and dc else None
    pred_df = prediction_rows(predict_checkpoints(data["master"], ticker, cutoff), extract_predictions(pred_file))
    chosen = selected_predictions(pred_df)
    close = latest_close(master)
    cards = st.columns(4)
    cards[0].metric("Dataset", fmt(len(master) if master is not None else 0))
    for card, horizon in zip(cards[1:], ["H+1", "H+2", "H+3"]):
        value = chosen.get(horizon)
        delta = f"{((value - close) / close) * 100:+.2f}%" if value and close else None
        card.metric(horizon, f"Rp {fmt(value)}" if value else "-", delta)
    combined_chart(master, ticker, dates, pred_df)
    show_split_chart(master)


def show_split_chart(master):
    if master is None or master.empty or "split" not in master.columns:
        return
    split = master["split"].value_counts().reset_index()
    split.columns = ["Split", "Jumlah"]
    fig = px.bar(split, x="Split", y="Jumlah", title="Distribusi Data Model",
                 color="Split", color_discrete_sequence=COLORS, text_auto=True)
    st.plotly_chart(layout(fig), use_container_width=True)
