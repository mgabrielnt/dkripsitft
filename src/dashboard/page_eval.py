import streamlit as st
from attention_chart import show_attention
from eval_summary import show_best_model, show_summary_cards
from eval_utils import show_eval
from style import header
from ui import action_button


def render(data, ticker, dates):
    header("Evaluasi Model")
    with st.expander("Jalankan Proses"):
        c1, c2 = st.columns(2)
        with c1: action_button("Evaluasi", "Evaluasi model")
        with c2: action_button("Interpretasi", "Interpretasi model")
    tabs = st.tabs(["Ringkasan", "Global", "Horizon", "Emiten", "Attention"])
    with tabs[0]:
        show_summary_cards(data["eval_global"])
        show_best_model(data["eval_global"])
    with tabs[1]:
        st.subheader("Evaluasi Global")
        show_eval("Evaluasi Global", data["eval_global"], "global")
    with tabs[2]:
        st.subheader("Evaluasi per Horizon")
        show_eval("Evaluasi per Horizon", data["eval_horizon"], "horizon", "Horizon")
    with tabs[3]:
        st.subheader("Evaluasi per Emiten")
        show_eval("Evaluasi per Emiten", data["eval_ticker"], "ticker", "Ticker")
    with tabs[4]:
        st.subheader("Attention / Interpretabilitas")
        show_attention(data["attention"])
