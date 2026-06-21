import plotly.express as px
import streamlit as st
from config import COLORS, FINAL_LABELS, SENT_FEATURES
from ui import layout
from utils import date_col, find_col, fmt


def show_news(news):
    st.subheader("Data Berita")
    c1, c2 = st.columns(2)
    c1.metric("Jumlah Berita", fmt(len(news) if news is not None else 0))
    c2.metric("Sumber Berita", fmt(news["source"].nunique() if news is not None and "source" in news else 0))
    if news is None or news.empty:
        st.info("Data berita belum tersedia.")
        return
    left, right = st.columns(2)
    show_source_chart(news, left)
    show_news_timeline(news, right)


def show_source_chart(news, container):
    if "source" not in news.columns:
        return
    count = news["source"].fillna("Tidak diketahui").value_counts().reset_index()
    count.columns = ["Sumber", "Jumlah"]
    fig = px.bar(count.head(15), x="Sumber", y="Jumlah", title="Jumlah Berita per Sumber",
                 color="Jumlah", color_continuous_scale="Bluered")
    container.plotly_chart(layout(fig), use_container_width=True)


def show_news_timeline(news, container):
    dc = date_col(news)
    if not dc:
        return
    timeline = news.dropna(subset=[dc]).assign(day=lambda x: x[dc].dt.date)
    timeline = timeline.groupby("day").size().reset_index(name="Jumlah")
    fig = px.area(timeline, x="day", y="Jumlah", title="Tren Jumlah Berita Harian",
                  color_discrete_sequence=["#38BDF8"])
    fig.update_traces(line=dict(width=3))
    container.plotly_chart(layout(fig), use_container_width=True)


def show_labeling(articles):
    st.subheader("Pelabelan Berita")
    final_col = find_col(articles, FINAL_LABELS)
    c1, c2 = st.columns(2)
    c1.metric("Artikel Berlabel", fmt(len(articles) if articles is not None else 0))
    c2.metric("Kolom Label", final_col if final_col else "-")
    if articles is None or articles.empty or not final_col:
        st.info("Data pelabelan belum tersedia.")
        return
    label_map = {-1: "Negatif", 0: "Netral", 1: "Positif", "-1": "Negatif", "0": "Netral", "1": "Positif"}
    count = articles[final_col].map(label_map).fillna(articles[final_col].astype(str)).value_counts().reset_index()
    count.columns = ["Sentimen", "Jumlah"]
    fig = px.pie(count, names="Sentimen", values="Jumlah", title="Distribusi Label Sentimen",
                 hole=0.5, color_discrete_sequence=["#22C55E", "#FACC15", "#F43F5E"])
    st.plotly_chart(layout(fig, 430), use_container_width=True)


def show_daily_sentiment(daily):
    st.subheader("Sentimen Harian")
    if daily is None or daily.empty:
        st.info("Data sentimen harian belum tersedia.")
        return
    c1, c2 = st.columns(2)
    c1.metric("Jumlah Baris", fmt(len(daily)))
    c2.metric("Jumlah Emiten", fmt(daily["ticker"].nunique() if "ticker" in daily else 0))
    dc = date_col(daily) or "date"
    available = [col for col in SENT_FEATURES if col in daily.columns]
    if available and dc in daily.columns:
        selected = st.selectbox("Fitur Sentimen", available)
        fig = px.line(daily.sort_values(dc), x=dc, y=selected, title=f"Tren {selected}", color_discrete_sequence=COLORS)
        fig.update_traces(line=dict(width=3))
        st.plotly_chart(layout(fig), use_container_width=True)
