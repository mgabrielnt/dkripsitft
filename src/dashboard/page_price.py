import pandas as pd
import plotly.express as px
import streamlit as st
from config import COLORS, TECH_FEATURES
from style import header
from ui import action_button, layout
from utils import date_col, filter_df, fmt


def render(data, ticker, dates):
    header("Data Harga Saham")
    with st.expander("Jalankan Proses"):
        c1, c2, c3 = st.columns(3)
        with c1: action_button("Ambil Harga", "Ambil harga Yahoo Finance")
        with c2: action_button("Hitung Indikator", "Hitung indikator teknikal")
        with c3: action_button("Audit Kalender", "Audit kalender harga")
    df = filter_df(data["prices"], ticker, dates)
    if df is None or df.empty:
        st.warning("Data harga belum tersedia.")
        return
    show_price_section(df)
    show_indicator_section(df)


def show_price_section(df):
    st.subheader("Harga Saham")
    dc = date_col(df) or "date"
    latest = df.sort_values(dc).tail(1).iloc[0] if dc in df.columns else df.tail(1).iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Jumlah Baris", fmt(len(df)))
    c2.metric("Jumlah Emiten", fmt(df["ticker"].nunique() if "ticker" in df.columns else 0))
    c3.metric("Close Terakhir", f"Rp {fmt(latest['close'])}" if "close" in df.columns else "-")
    if "close" in df.columns and dc in df.columns:
        fig = px.line(df.sort_values(dc), x=dc, y="close", title="Harga Penutupan",
                      color_discrete_sequence=COLORS)
        fig.update_traces(line=dict(width=3))
        st.plotly_chart(layout(fig), use_container_width=True)


def show_indicator_section(df):
    st.subheader("Indikator Teknikal")
    dc = date_col(df) or "date"
    indicators = [c for c in TECH_FEATURES if c in df.columns and c not in {"close", "volume"}]
    if not indicators:
        st.info("Indikator teknikal belum tersedia.")
        return
    selected = st.selectbox("Pilih Indikator", indicators)
    if dc in df.columns:
        fig = px.line(df.sort_values(dc), x=dc, y=selected, title=f"Tren {selected}",
                      color_discrete_sequence=COLORS)
        fig.update_traces(line=dict(width=3))
        st.plotly_chart(layout(fig), use_container_width=True)
    corr_cols = [c for c in indicators if pd.api.types.is_numeric_dtype(df[c])]
    if len(corr_cols) >= 2:
        fig = px.imshow(df[corr_cols].corr(numeric_only=True), text_auto=".2f",
                        title="Korelasi Indikator Teknikal", color_continuous_scale="Turbo")
        st.plotly_chart(layout(fig, 520), use_container_width=True)
