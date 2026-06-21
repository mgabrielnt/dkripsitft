import pandas as pd
import streamlit as st
from data_loader import load_data
from style import apply_style
from utils import date_col
import page_model
import page_eval
import page_price
import page_news

PAGES = ["Model dan Prediksi", "Evaluasi Model", "Data Harga Saham", "Berita Keuangan dan Sentimen"]

st.set_page_config(page_title="Dashboard Prediksi Saham Indonesia", page_icon="📈", layout="wide")
apply_style()
data = load_data()


def ticker_options(data):
    source = data["master"] if data["master"] is not None and "ticker" in data["master"].columns else data["prices"]
    if source is None or "ticker" not in source.columns:
        return []
    return sorted(source["ticker"].dropna().astype(str).unique().tolist())


def date_filter(data):
    source = data["master"] if data["master"] is not None and date_col(data["master"]) else data["prices"]
    dc = date_col(source)
    if source is None or not dc:
        return None
    valid = source[dc].dropna()
    if valid.empty:
        return None
    start, end = valid.min(), valid.max()
    if not (pd.notna(start) and pd.notna(end)):
        return None
    return st.date_input("Rentang Tanggal", value=(start.date(), end.date()), min_value=start.date(), max_value=end.date())


with st.sidebar:
    st.markdown("## 📈 Prediksi Saham")
    st.caption("Dashboard TFT dan LLM-TFT")
    st.divider()
    page = st.radio("Menu Dashboard", PAGES)
    st.divider()
    st.markdown("### Filter Data")
    tickers = ticker_options(data)
    selected_ticker = st.selectbox("Emiten", tickers) if tickers else None
    selected_dates = date_filter(data)

if page == "Model dan Prediksi":
    page_model.render(data, selected_ticker, selected_dates)
elif page == "Evaluasi Model":
    page_eval.render(data, selected_ticker, selected_dates)
elif page == "Data Harga Saham":
    page_price.render(data, selected_ticker, selected_dates)
else:
    page_news.render(data, selected_ticker, selected_dates)
