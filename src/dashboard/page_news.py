import plotly.express as px
import streamlit as st
from config import COLORS, FINAL_LABELS, SENT_FEATURES
from style import header
from ui import action_button, layout
from utils import date_col, filter_df, find_col, fmt

def render(data, ticker, dates):
    header("Berita Keuangan dan Sentimen")
    with st.expander("Jalankan Proses"):
        cols = st.columns(6)
        keys = ["Ambil berita RSS dan Google News", "Ambil berita Yahoo Finance",
                "Gabung sumber berita", "Bersihkan teks berita",
                "Label sentimen artikel", "Agregasi sentimen harian"]
        labels = ["RSS/Google", "Yahoo", "Gabung", "Bersihkan", "Label", "Agregasi"]
        for col, label, key in zip(cols, labels, keys):
            with col: action_button(label, key)
    news = filter_df(data["news"], ticker, dates)
    articles = filter_df(data["articles"], ticker, dates)
    daily = filter_df(data["daily"], ticker, dates)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Jumlah Berita", fmt(len(news) if news is not None else 0))
    c2.metric("Sentimen Artikel", fmt(len(articles) if articles is not None else 0))
    c3.metric("Sentimen Harian", fmt(len(daily) if daily is not None else 0))
    c4.metric("Jumlah Emiten", fmt(daily["ticker"].nunique() if daily is not None and "ticker" in daily else 0))
    left, right = st.columns(2)
    if news is not None and not news.empty:
        if "source" in news.columns:
            count = news["source"].fillna("Tidak diketahui").value_counts().reset_index()
            count.columns = ["Sumber", "Jumlah"]
            fig = px.bar(count.head(15), x="Sumber", y="Jumlah",
                         title="Jumlah Berita per Sumber", color="Jumlah",
                         color_continuous_scale="Bluered")
            left.plotly_chart(layout(fig), use_container_width=True)
        dc = date_col(news)
        if dc:
            daily_news = news.dropna(subset=[dc]).assign(day=lambda x: x[dc].dt.date)
            daily_news = daily_news.groupby("day").size().reset_index(name="Jumlah")
            fig = px.area(daily_news, x="day", y="Jumlah",
                          title="Tren Jumlah Berita Harian", color_discrete_sequence=["#38BDF8"])
            fig.update_traces(line=dict(width=3))
            right.plotly_chart(layout(fig), use_container_width=True)
    s_left, s_right = st.columns(2)
    final_col = find_col(articles, FINAL_LABELS)
    if articles is not None and not articles.empty and final_col:
        label_map = {-1: "Negatif", 0: "Netral", 1: "Positif", "-1": "Negatif", "0": "Netral", "1": "Positif"}
        count = articles[final_col].map(label_map).fillna(articles[final_col].astype(str)).value_counts().reset_index()
        count.columns = ["Sentimen", "Jumlah"]
        fig = px.pie(count, names="Sentimen", values="Jumlah", title="Distribusi Sentimen Artikel",
                     hole=0.5, color_discrete_sequence=["#22C55E", "#FACC15", "#F43F5E"])
        s_left.plotly_chart(layout(fig, 430), use_container_width=True)
    if daily is not None and not daily.empty:
        dc = date_col(daily) or "date"
        available = [col for col in SENT_FEATURES if col in daily.columns]
        if available:
            selected = st.selectbox("Fitur Sentimen", available)
            fig = px.line(daily.sort_values(dc), x=dc, y=selected,
                          color="ticker" if ticker == "Semua" and "ticker" in daily.columns else None,
                          title=f"Tren {selected}", color_discrete_sequence=COLORS)
            fig.update_traces(line=dict(width=3))
            s_right.plotly_chart(layout(fig), use_container_width=True)
