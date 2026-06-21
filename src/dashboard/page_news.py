import streamlit as st
from news_sections import show_daily_sentiment, show_labeling, show_news
from style import header
from ui import action_button
from utils import filter_df

ACTIONS = [
    ("RSS/Google", "Ambil berita RSS dan Google News"),
    ("Yahoo", "Ambil berita Yahoo Finance"),
    ("Gabung", "Gabung sumber berita"),
    ("Bersihkan", "Bersihkan teks berita"),
    ("Label", "Label sentimen artikel"),
    ("Agregasi", "Agregasi sentimen harian"),
]


def render(data, ticker, dates):
    header("Berita Keuangan dan Sentimen")
    with st.expander("Jalankan Proses"):
        for col, (label, key) in zip(st.columns(6), ACTIONS):
            with col:
                action_button(label, key)
    news = filter_df(data["news"], ticker, dates)
    articles = filter_df(data["articles"], ticker, dates)
    daily = filter_df(data["daily"], ticker, dates)
    show_news(news)
    show_labeling(articles)
    show_daily_sentiment(daily)
