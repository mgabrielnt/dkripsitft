import plotly.graph_objects as go
import streamlit as st
from attention_chart import show_attention
from eval_utils import show_eval
from style import header
from ui import action_button, layout
from utils import date_col, filter_df, find_col, find_contains_col

def render(data, ticker, dates):
    header("Evaluasi Model")
    with st.expander("Jalankan Proses"):
        c1, c2 = st.columns(2)
        with c1: action_button("Evaluasi", "Evaluasi model")
        with c2: action_button("Interpretasi", "Interpretasi model")
    tabs = st.tabs(["Global", "Horizon", "Emiten", "Attention"])
    with tabs[0]: show_eval("Evaluasi Global", data["eval_global"], "global")
    with tabs[1]: show_eval("Evaluasi per Horizon", data["eval_horizon"], "horizon", "Horizon")
    with tabs[2]: show_eval("Evaluasi per Emiten", data["eval_ticker"], "ticker", "Ticker")
    with tabs[3]: show_attention(data["attention"])
    pred = filter_df(data["predictions"], ticker, dates)
    if pred is None or pred.empty:
        return
    dc = date_col(pred)
    actual = find_col(pred, ["actual", "y_true", "close", "target"])
    predicted = find_col(pred, ["prediction", "predicted", "y_pred", "pred", "forecast"])
    predicted = predicted or find_contains_col(pred, ["pred"], ["error"])
    if dc and actual and predicted:
        st.subheader("Actual vs Predicted")
        plot_df = pred.sort_values(dc).tail(250)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=plot_df[dc], y=plot_df[actual], mode="lines",
                                 name="Aktual", line=dict(color="#E5E7EB", width=3)))
        fig.add_trace(go.Scatter(x=plot_df[dc], y=plot_df[predicted], mode="lines",
                                 name="Prediksi", line=dict(color="#22C55E", width=3)))
        fig.update_layout(title="Actual vs Predicted")
        st.plotly_chart(layout(fig), use_container_width=True)
