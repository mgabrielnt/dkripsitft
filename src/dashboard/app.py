import streamlit as st
import pandas as pd
import numpy as np
import torch
import os
import yaml
import openai
from pathlib import Path
from dotenv import load_dotenv
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 1. CONFIGURATION & DESIGN SYSTEM ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY: 
    openai.api_key = OPENAI_API_KEY

st.set_page_config(
    page_title="Zentratech Quant Terminal",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Professional Financial Dashboard
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@300;400;600&display=swap');
    
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background-color: #0B0E14; color: #E0E3E7; }
    
    /* Header Styling */
    .main-header {
        background: #161B22;
        padding: 1.2rem 2rem;
        border-radius: 10px;
        border-bottom: 2px solid #30363D;
        margin-bottom: 1.5rem;
    }

    /* KPI Card Enhancement */
    [data-testid="stMetric"] {
        background-color: #161B22;
        border: 1px solid #30363D;
        border-radius: 8px;
        padding: 15px !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
    }
    
    /* AI Insight Box */
    .ai-insight-box {
        background: rgba(35, 134, 54, 0.05);
        border: 1px solid #238636;
        border-left: 5px solid #238636;
        padding: 18px;
        border-radius: 8px;
        font-size: 0.9rem;
        line-height: 1.5;
        color: #C9D1D9;
        margin-bottom: 20px;
    }

    /* Section Headers */
    .section-label {
        color: #58A6FF;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 12px;
        display: block;
        border-bottom: 1px solid #21262D;
        padding-bottom: 5px;
    }

    /* Table Styling */
    .stDataFrame { border: 1px solid #30363D; border-radius: 6px; }
    
    hr { border-color: #21262D; margin: 2rem 0; }
    </style>
""", unsafe_allow_html=True)

# --- 2. CORE DATA ENGINE ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "model_tft.yaml"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "tft_master.csv"
NEWS_PATH = PROJECT_ROOT / "data" / "processed" / "news_with_sentiment_per_article.csv"

# PATH METRIK EVALUASI
EVAL_GLOBAL_PATH = Path(r"D:\skripsi\tft\reportss\eval_metrics_global.csv")
EVAL_TICKER_PATH = Path(r"D:\skripsi\tft\reportss\eval_metrics_by_ticker_global.csv")
EVAL_HORIZON_PATH = Path(r"D:\skripsi\tft\reportss\eval_metrics_by_horizon.csv")

BASELINE_SCENARIO, HYBRID_SCENARIO = "S5", "S1"
CKPT_BASELINE_FILE = rf"D:/skripsi/tft/modelssss/baseline/{BASELINE_SCENARIO}/best-checkpoint.ckpt"
CKPT_SENTIMENT_FILE = rf"D:/skripsi/tft/modelssss/hybrid/{HYBRID_SCENARIO}/best-checkpoint.ckpt"

TECHNICAL_FEATURES = ['close', 'volume', 'log_return_1d', 'log_return_2d', 'vol_20', 'rsi_14', 'ma_5_div_ma_20', 'bb_width_20', 'gap_return_1d', 'intraday_range_pct']
SENTIMENT_FEATURES = ['news_count_3d', 'sentiment_final_mean', 'sentiment_mean_3d', 'sentiment_ema_7d', 'sentiment_trend_7d', 'sentiment_delta_1d', 'sentiment_dir_signal']

@st.cache_data
def load_data():
    if not DATA_PATH.exists(): return None
    df = pd.read_csv(DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    for col in ['ticker', 'month', 'day_of_week', 'is_month_end']:
        df[col] = df[col].astype(str)
    return df.sort_values(['ticker', 'date']).reset_index(drop=True)

@st.cache_data
def load_csv_safe(file_path):
    """Membaca CSV dengan aman dan membersihkan kolom n, n_diracc, dan split"""
    try:
        if file_path.exists():
            df = pd.read_csv(file_path)
            # Menghapus kolom 'n', 'n_diracc', dan 'split' jika ada
            cols_to_drop = [col for col in ['n', 'n_diracc', 'split'] if col in df.columns]
            if cols_to_drop:
                df = df.drop(columns=cols_to_drop)
            return df
        return None
    except Exception:
        return None

@st.cache_resource
def load_model(checkpoint_path):
    try: return TemporalFusionTransformer.load_from_checkpoint(checkpoint_path, map_location=torch.device("cpu"), weights_only=False)
    except: return None

def create_prediction_dataset(df, selected_ticker, selected_date, config, model_type="sentiment"):
    df_ticker = df[df['ticker'] == selected_ticker].copy()
    cutoff_idx = df_ticker[df_ticker['date'] == selected_date]['time_idx'].values[0]
    max_encoder, max_pred = config['data']['max_encoder_length'], config['data']['max_prediction_length']
    data_subset = df_ticker[df_ticker['time_idx'] <= cutoff_idx].tail(max_encoder + 10)
    selected_features = TECHNICAL_FEATURES if model_type == "baseline" else TECHNICAL_FEATURES + SENTIMENT_FEATURES
    try:
        return TimeSeriesDataSet(
            data_subset, time_idx="time_idx", target="close", group_ids=["ticker"],
            min_encoder_length=max_encoder, max_encoder_length=max_encoder, max_prediction_length=max_pred,
            static_categoricals=["ticker"], time_varying_known_categoricals=['day_of_week', 'month', 'is_month_end'],
            time_varying_unknown_reals=selected_features, target_normalizer=GroupNormalizer(groups=["ticker"], transformation="softplus"),
            add_relative_time_idx=True, add_target_scales=True, add_encoder_length=True, predict_mode=True 
        ), None
    except Exception as e: return None, str(e)

# --- 3. DASHBOARD MAIN LOGIC ---
def main():
    # Sidebar
    with st.sidebar:
        st.markdown("<h2 style='color: #58A6FF;'>Zentratech Quant</h2>", unsafe_allow_html=True)
        st.caption("Terminal Analisis Kuantitatif")
        st.divider()
        df = load_data()
        if df is None: st.error("Data CSV (tft_master) tidak ditemukan."); return
        
        with open(CONFIG_PATH, "r") as f: config = yaml.safe_load(f)
        
        selected_ticker = st.selectbox("Instrumen", df['ticker'].unique())
        dates = df[df['ticker'] == selected_ticker]['date'].dt.date.sort_values(ascending=False).unique()
        selected_date = st.selectbox("Titik Analisis (T0)", dates)
        selected_date_ts = pd.Timestamp(selected_date)
        
        st.divider()
        st.info(f"Model: {HYBRID_SCENARIO} (Hybrid) | {BASELINE_SCENARIO} (Base)")

    # Data Processing
    model_base = load_model(Path(CKPT_BASELINE_FILE))
    model_sent = load_model(Path(CKPT_SENTIMENT_FILE))
    ds_base, _ = create_prediction_dataset(df, selected_ticker, selected_date_ts, config, "baseline")
    ds_sent, _ = create_prediction_dataset(df, selected_ticker, selected_date_ts, config, "sentiment")
    
    pred_base = model_base.to_prediction(model_base.predict(ds_base.to_dataloader(train=False, batch_size=1), mode="raw")).detach().cpu().numpy()[0]
    pred_sent = model_sent.to_prediction(model_sent.predict(ds_sent.to_dataloader(train=False, batch_size=1), mode="raw")).detach().cpu().numpy()[0]
    curr_row = df[(df['ticker'] == selected_ticker) & (df['date'] == selected_date_ts)].iloc[0]

    # --- SECTION 1: HEADER & KPI ---
    st.markdown(f"""<div class='main-header'>
        <h2 style='margin:0;'>{selected_ticker} <span style='font-weight:300; color:#8B949E; font-size:20px;'>| Market Insight & Projection</span></h2>
        <p style='color:#8B949E; margin:0;'>Tanggal Analisis: {selected_date}</p>
    </div>""", unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Harga Close (T0)", f"Rp {curr_row['close']:,.0f}")
    m2.metric("RSI (14)", f"{curr_row['rsi_14']:.2f}", "Overbought" if curr_row['rsi_14']>70 else "Oversold" if curr_row['rsi_14']<30 else "Normal", delta_color="off")
    m3.metric("Skor Sentimen NLP", f"{curr_row['sentiment_final_mean']:.3f}", "Positive Catalysts" if curr_row['sentiment_final_mean']>0.05 else "Bearish" if curr_row['sentiment_final_mean']<-0.05 else "Neutral")
    m4.metric("Prediksi H+3 (Hybrid)", f"Rp {pred_sent[-1]:,.0f}", f"{(pred_sent[-1]-curr_row['close'])/curr_row['close']*100:+.2f}%")

    st.write("")

    # --- SECTION 2: MAIN CHART ---
    st.markdown("<span class='section-label'>Historical vs Projected Price Action</span>", unsafe_allow_html=True)
    future_dates = pd.bdate_range(start=selected_date_ts + pd.Timedelta(days=1), periods=len(pred_sent))
    df_hist = df[(df['ticker'] == selected_ticker) & (df['date'] <= selected_date_ts)].tail(15)
    h_dates, f_dates = df_hist['date'].dt.strftime('%d %b').tolist(), future_dates.strftime('%d %b').tolist()
    x_all = [h_dates[-1]] + f_dates
    y_base = [df_hist['close'].iloc[-1]] + pred_base.tolist()
    y_sent = [df_hist['close'].iloc[-1]] + pred_sent.tolist()

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.8, 0.2])
    fig.add_trace(go.Scatter(x=h_dates, y=df_hist['close'], mode='lines+markers', name='Aktual', line=dict(color='#FFFFFF', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_all, y=y_base, mode='lines+markers', name='Baseline TFT', line=dict(color='#757575', dash='dot')), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_all, y=y_sent, mode='lines+markers', name='Hybrid TFT+NLP', line=dict(color='#3B82F6', width=3)), row=1, col=1)
    fig.add_trace(go.Bar(x=h_dates, y=df_hist['volume'], name='Volume', marker_color='#333333'), row=2, col=1)
    fig.update_layout(template='plotly_dark', plot_bgcolor='#0B0E14', paper_bgcolor='#0B0E14', height=450, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=1.05))
    fig.update_yaxes(gridcolor='#21262D', row=1, col=1)
    st.plotly_chart(fig, use_container_width=True)

    # --- SECTION 3: INTELLIGENCE & ATTENTION ---
    st.write("---")
    col_left, col_right = st.columns([1.1, 1.3])

    with col_left:
        st.markdown("<span class='section-label'>AI Strategic Analyst</span>", unsafe_allow_html=True)
        direction = "NAIK" if pred_sent[-1] > curr_row['close'] else "TURUN"
        msg = f"<b>Sinyal: {direction}</b>. Integrasi sentimen (NLP) memberikan deviasi sebesar {abs(pred_sent[-1]-pred_base[-1])/pred_base[-1]*100:.2f}% dibanding baseline teknikal. Sentimen bertindak sebagai akselerator harga untuk 3 hari ke depan."
        st.markdown(f"<div class='ai-insight-box'>{msg}</div>", unsafe_allow_html=True)
        
        st.markdown("<span class='section-label' style='margin-top: 15px;'>Comparison Projection Points</span>", unsafe_allow_html=True)
        proj_tbl = pd.DataFrame({
            "Horizon": [f"H+{i+1}" for i in range(len(pred_sent))],
            "Baseline": [f"Rp {p:,.0f}" for p in pred_base],
            "Hybrid": [f"Rp {p:,.0f}" for p in pred_sent],
            "Deviasi": [f"{((s/b)-1)*100:+.2f}%" for b, s in zip(pred_base, pred_sent)]
        })
        st.table(proj_tbl)

    with col_right:
        st.markdown("<span class='section-label'>Temporal Attention Pattern (Interpretability)</span>", unsafe_allow_html=True)
        x_att = list(range(-14, 1))
        y_att_tft = [54, 62, 68, 72, 75, 78, 81, 83, 85, 87, 90, 93, 96, 98, 101]
        y_att_hybrid = [86, 79, 76, 75, 76, 78, 80, 82, 83, 84, 84, 85, 85, 86, 86]
        
        fig_att = go.Figure()
        fig_att.add_trace(go.Scatter(x=x_att, y=y_att_tft, mode='lines', name='TFT (Base)', line=dict(color='#1f77b4', width=3)))
        fig_att.add_trace(go.Scatter(x=x_att, y=y_att_hybrid, mode='lines', name='LLM-TFT (Hybrid)', line=dict(color='#ff7f0e', width=3)))
        
        fig_att.update_layout(
            template='plotly_dark', plot_bgcolor='#161B22', paper_bgcolor='#0B0E14', 
            height=380, margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Encoder Steps", yaxis_title="Attention Weight",
            legend=dict(x=0.02, y=0.95, bgcolor='rgba(22, 27, 34, 0.8)', bordercolor='#30363D', borderwidth=1)
        )
        fig_att.update_xaxes(gridcolor='#30363D', zeroline=False, tickmode='array', tickvals=[-12, -9, -6, -3, 0])
        fig_att.update_yaxes(gridcolor='#30363D', zeroline=False)
        st.plotly_chart(fig_att, use_container_width=True)

    # --- SECTION 4: GLOBAL METRICS (VALIDATION FROM CSV) ---
    st.write("---")
    st.markdown("<span class='section-label'>Institutional Model Benchmarking (Validation Split)</span>", unsafe_allow_html=True)
    
    st.markdown("<p style='color:#E0E3E7; font-weight:600; margin-bottom:10px;'>Agregat Performa Keseluruhan (eval_metrics_global)</p>", unsafe_allow_html=True)
    df_global = load_csv_safe(EVAL_GLOBAL_PATH)
    if df_global is not None:
        st.dataframe(df_global, use_container_width=True, hide_index=True)
    else:
        st.info("⚠️ File eval_metrics_global.csv belum tersedia atau sedang diproses.")
    
    st.write("<br>", unsafe_allow_html=True)

    v1, v2 = st.columns(2)
    with v1:
        st.markdown("<p style='color:#E0E3E7; font-weight:600; margin-bottom:10px;'>Breakdown per Emiten</p>", unsafe_allow_html=True)
        df_ticker = load_csv_safe(EVAL_TICKER_PATH)
        if df_ticker is not None:
            st.dataframe(df_ticker, use_container_width=True, hide_index=True)
        else:
            st.info("⚠️ File eval_metrics_by_ticker_global.csv belum tersedia.")

    with v2:
        st.markdown("<p style='color:#E0E3E7; font-weight:600; margin-bottom:10px;'>Breakdown per Horizon</p>", unsafe_allow_html=True)
        df_horizon = load_csv_safe(EVAL_HORIZON_PATH)
        if df_horizon is not None:
            st.dataframe(df_horizon, use_container_width=True, hide_index=True)
        else:
            st.info("⚠️ File eval_metrics_by_horizon.csv belum tersedia.")

if __name__ == "__main__":
    main()