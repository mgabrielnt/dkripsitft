import pandas as pd
import plotly.express as px
import streamlit as st
from config import MODEL_COLORS
from ui import layout
from utils import date_col, filter_df

def encoder_df(master, ticker, dates, n=15):
    df = filter_df(master, ticker, dates)
    if df is None or df.empty or "close" not in df.columns:
        return pd.DataFrame()
    work = df.copy()
    if "ticker" in work.columns and ticker == "Semua":
        sort_col = date_col(work) or ("time_idx" if "time_idx" in work.columns else None)
        tick = work.sort_values(sort_col).tail(1)["ticker"].iloc[0] if sort_col else work["ticker"].iloc[0]
        work = work[work["ticker"].eq(tick)]
    dc = date_col(work)
    work = work.sort_values(dc) if dc else work.sort_values("time_idx")
    enc = work.tail(n).copy()
    enc["Step"] = list(range(-len(enc) + 1, 1))
    enc["Harga"] = enc["close"]
    enc["Series"] = "Encoder 15 Hari"
    return enc[["Step", "Harga", "Series"]]

def combined_chart(master, ticker, dates, pred_df):
    rows = []
    enc = encoder_df(master, ticker, dates)
    if not enc.empty:
        rows.extend(enc.to_dict("records"))
    latest = rows[-1]["Harga"] if rows else None
    if pred_df is not None and not pred_df.empty:
        for series, group in pred_df.groupby("Series"):
            if latest is not None:
                rows.append({"Step": 0, "Harga": latest, "Series": series})
            rows.extend(group.to_dict("records"))
    chart = pd.DataFrame(rows).dropna(subset=["Harga"]) if rows else pd.DataFrame()
    st.subheader("Encoder 15 Hari dan Prediksi Multi-Horizon")
    if chart.empty:
        st.info("Data encoder dan prediksi belum tersedia.")
        return
    fig = px.line(chart.sort_values(["Series", "Step"]), x="Step", y="Harga",
                  color="Series", markers=True,
                  title="Gabungan Encoder dan Prediksi H+1 sampai H+3",
                  color_discrete_map=MODEL_COLORS)
    fig.update_traces(line=dict(width=4), marker=dict(size=10))
    ticks = list(range(-14, 4))
    labels = [f"T{x}" if x < 0 else ("T" if x == 0 else f"H+{x}") for x in ticks]
    fig.update_xaxes(tickmode="array", tickvals=ticks, ticktext=labels, title="Langkah Waktu")
    fig.update_yaxes(title="Harga Close / Prediksi")
    fig.add_vline(x=0, line_dash="dash", line_color="rgba(255,255,255,0.45)")
    st.plotly_chart(layout(fig, 460), use_container_width=True)
