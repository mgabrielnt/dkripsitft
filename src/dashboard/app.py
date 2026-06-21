import streamlit as st
from data_loader import load_data
from style import apply_style
from utils import date_col
import page_model
import page_eval
import page_price
import page_news

st.set_page_config(
    page_title="Dashboard Prediksi Saham Indonesia",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_style()
data = load_data()

with st.sidebar:
    st.markdown("## 📈 Prediksi Saham")
    st.caption("Dashboard TFT dan LLM-TFT")
    st.divider()
    page = st.radio(
        "Menu Dashboard",
        ["Model dan Prediksi", "Evaluasi Model", "Data Harga Saham", "Berita Keuangan dan Sentimen"],
    )
    st.divider()
    st.markdown("### Filter Data")
    source = data["master"] if data["master"] is not None and "ticker" in data["master"].columns else data["prices"]
    tickers = ["Semua"]
    if source is not None and "ticker" in source.columns:
        tickers += sorted(source["ticker"].dropna().astype(str).unique().tolist())
    selected_ticker = st.selectbox("Emiten", tickers)
    date_source = data["master"] if data["master"] is not None and date_col(data["master"]) else data["prices"]
    selected_dates = None
    dc = date_col(date_source)
    if date_source is not None and dc:
        min_date = date_source[dc].min()
        max_date = date_source[dc].max()
        if min_date is not None and max_date is not None:
            selected_dates = st.date_input(
                "Rentang Tanggal",
                value=(min_date.date(), max_date.date()),
                min_value=min_date.date(),
                max_value=max_date.date(),
            )

if page == "Model dan Prediksi":
    page_model.render(data, selected_ticker, selected_dates)
elif page == "Evaluasi Model":
    page_eval.render(data, selected_ticker, selected_dates)
elif page == "Data Harga Saham":
    page_price.render(data, selected_ticker, selected_dates)
else:
    page_news.render(data, selected_ticker, selected_dates)
