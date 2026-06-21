import pandas as pd
import plotly.express as px
import streamlit as st
from config import COLORS, TECH_FEATURES
from style import header
from ui import action_button, layout
from utils import date_col, filter_df, fmt

def render(data, ticker, dates):
    header("Pengolahan Data Harga Saham")
    with st.expander("Jalankan Proses"):
        c1, c2, c3 = st.columns(3)
        with c1: action_button("Ambil Harga", "Ambil harga Yahoo Finance")
        with c2: action_button("Hitung Indikator", "Hitung indikator teknikal")
        with c3: action_button("Audit Kalender", "Audit kalender harga")
    df = filter_df(data["prices"], ticker, dates)
    if df is None or df.empty:
        st.warning("Data harga belum tersedia.")
        return
    dc = date_col(df) or "date"
    latest = df.sort_values(dc).tail(1).iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Jumlah Baris", fmt(len(df)))
    c2.metric("Jumlah Emiten", fmt(df["ticker"].nunique() if "ticker" in df.columns else 0))
    c3.metric("Close Terakhir", f"Rp {fmt(latest['close'])}" if "close" in df.columns else "-")
    if "close" in df.columns:
        fig = px.line(df.sort_values(dc), x=dc, y="close",
                      color="ticker" if ticker == "Semua" and "ticker" in df.columns else None,
                      title="Harga Penutupan", color_discrete_sequence=COLORS)
        fig.update_traces(line=dict(width=3))
        st.plotly_chart(layout(fig), use_container_width=True)
    available = [col for col in TECH_FEATURES if col in df.columns and col != "volume"]
    if available:
        selected = st.selectbox("Indikator", available)
        fig = px.line(df.sort_values(dc), x=dc, y=selected,
                      color="ticker" if ticker == "Semua" and "ticker" in df.columns else None,
                      title=f"Tren {selected}", color_discrete_sequence=COLORS)
        fig.update_traces(line=dict(width=3))
        st.plotly_chart(layout(fig), use_container_width=True)
    corr_cols = [col for col in available if pd.api.types.is_numeric_dtype(df[col])]
    if len(corr_cols) >= 2:
        fig = px.imshow(df[corr_cols].corr(numeric_only=True), text_auto=".2f",
                        title="Korelasi Indikator Teknikal", color_continuous_scale="Turbo")
        st.plotly_chart(layout(fig, 520), use_container_width=True)
