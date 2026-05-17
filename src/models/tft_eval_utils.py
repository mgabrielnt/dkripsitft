import streamlit as st
import pandas as pd
import numpy as np
import torch
import os
import yaml
import openai
from pathlib import Path
from dotenv import load_dotenv
import lightning.pytorch as pl
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 1. SETUP & KONFIGURASI ---

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY: openai.api_key = OPENAI_API_KEY

st.set_page_config(
    page_title="Zentratech | Quant Terminal",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed" 
)

# Custom CSS: Single-Pane Institutional Grid
st.markdown("""
    <style>
    /* Global Dark Theme */
    .stApp { background-color: #0E1117; color: #E0E0E0; font-family: 'Inter', sans-serif; }
    
    /* Panel/Card Containers */
    .dashboard-panel {
        background-color: #161A25;
        border: 1px solid #2D303E;
        border-radius: 6px;
        padding: 15px;
        height: 100%;
    }

    /* KPI Cards Minimalist - Diperbaiki agar tidak kepotong */
    [data-testid="stMetric"] {
        background-color: #161A25;
        border-radius: 6px;
        padding: 15px 20px;
        border: 1px solid #2D303E;
        border-left: 4px solid #3B82F6;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    [data-testid="stMetricValue"] { font-size: 26px !important; font-weight: 700; color: #FFFFFF; }
    [data-testid="stMetricLabel"] { font-size: 13px !important; color: #8A94A6; text-transform: uppercase; letter-spacing: 0.5px; }

    /* Custom AI Box */
    .ai-box {
        background-color: #12141C;
        border-left: 3px solid #10B981;
        padding: 15px;
        border-radius: 4px;
        color: #D1D5DB;
        font-size: 0.95em;
        line-height: 1.6;
        border: 1px solid #2D303E;
        margin-top: 10px;
    }
    
    .block-container { padding-top: 1.5rem; padding-bottom: 1.5rem; }
    h1, h2, h3, h4 { color: #F5F5F5; font-weight: 600; margin-bottom: 0.5rem; }
    hr { border-color: #2D303E; margin: 20px 0; }
    </style>
""", unsafe_allow_html=True)

# Setup Paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "model_tft.yaml"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "tft_master.csv"
NEWS_PATH = PROJECT_ROOT / "data" / "processed" / "news_with_sentiment_per_article.csv"

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
    df['ticker'], df['month'], df['day_of_week'], df['is_month_end'] = df['ticker'].astype(str), df['month'].astype(str), df['day_of_week'].astype(str), df['is_month_end'].astype(str)
    return df.sort_values(['ticker', 'date']).reset_index(drop=True)

@st.cache_data
def load_news_raw():
    if not NEWS_PATH.exists(): return None
    df = pd.read_csv(NEWS_PATH)
    df['date'] = pd.to_datetime(df['date'])
    return df

@st.cache_resource
def load_model(checkpoint_path):
    try: return TemporalFusionTransformer.load_from_checkpoint(checkpoint_path, map_location=torch.device("cpu"), weights_only=False)
    except: return None

def create_prediction_dataset(df, selected_ticker, selected_date, config, model_type="sentiment"):
    df_ticker = df[df['ticker'] == selected_ticker].copy()
    if selected_date not in df_ticker['date'].values: return None, "Tanggal tidak tersedia."
    
    cutoff_idx = df_ticker[df_ticker['date'] == selected_date]['time_idx'].values[0]
    max_encoder, max_pred = config['data']['max_encoder_length'], config['data']['max_prediction_length']
    if cutoff_idx < max_encoder: return None, "Data historis kurang."
    
    data_subset = df_ticker[df_ticker['time_idx'] <= cutoff_idx].tail(max_encoder + 10)
    selected_features = TECHNICAL_FEATURES if model_type == "baseline" else TECHNICAL_FEATURES + SENTIMENT_FEATURES

    try:
        dataset = TimeSeriesDataSet(
            data_subset, time_idx="time_idx", target="close", group_ids=["ticker"],
            min_encoder_length=max_encoder, max_encoder_length=max_encoder, max_prediction_length=max_pred,
            static_categoricals=["ticker"], time_varying_known_categoricals=['day_of_week', 'month', 'is_month_end'],
            time_varying_unknown_reals=selected_features, target_normalizer=GroupNormalizer(groups=["ticker"], transformation="softplus"),
            add_relative_time_idx=True, add_target_scales=True, add_encoder_length=True, predict_mode=True 
        )
        return dataset, None
    except Exception as e: return None, str(e)

def generate_expert_analysis(ticker, date, tech_data, sent_data, news_df, preds_base, preds_sent):
    if not OPENAI_API_KEY: return "⚠️ API Key OpenAI belum diatur."
    start_date = date - pd.Timedelta(days=5)
    
    if news_df is None or news_df.empty: headlines_list = "- (Data file berita tidak ditemukan)"
    else:
        relevant_news = news_df[(news_df['ticker'] == ticker) & (news_df['date'] >= start_date) & (news_df['date'] <= date)]
        if not relevant_news.empty:
            col_name = next((col for col in ['title', 'headline', 'news_title'] if col in relevant_news.columns), None)
            headlines_list = "\n".join([f"- '{h}'" for h in relevant_news[col_name].dropna().unique()[:3]]) if col_name else "- (Kolom judul tidak valid)"
        else: headlines_list = "- (Tidak ada berita material)"

    rsi, ma_status = tech_data['rsi_14'], "Uptrend" if tech_data['ma_5_div_ma_20'] > 1 else "Downtrend"
    impact = sent_data['sentiment_final_mean'] 
    sent_mood = "POSITIF" if impact > 0.1 else "NEGATIF" if impact < -0.1 else "NETRAL"
    sent_model_direction = "NAIK" if (preds_sent[-1] - preds_sent[0]) > 0 else "TURUN"
    
    prompt = f"""
    Chief Quant Analyst. EMITEN: {ticker} ({date.strftime('%Y-%m-%d')}). 
    DATA: RSI={rsi:.2f}, Tren={ma_status}. Skor NLP: {impact:.2f} ({sent_mood}). Berita: {headlines_list}. 
    TFT PREDICT: {sent_model_direction}. 
    TUGAS: Tulis 1 paragraf sangat ringkas (max 4 kalimat) analisis teknikal & sentimen. Jika ada divergence, sebutkan secara profesional.
    """
    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        return client.chat.completions.create(model="gpt-3.5-turbo", messages=[{"role": "system", "content": "You are a concise financial analyst."}, {"role": "user", "content": prompt}], temperature=0.3, max_tokens=250).choices[0].message.content.strip()
    except Exception as e: return f"Error API: {str(e)}"

# --- MAIN UI ---
def main():
    # SIDEBAR MINIMALIS
    with st.sidebar:
        st.markdown("<h3 style='color: #3B82F6;'>Zentratech Quant</h3>", unsafe_allow_html=True)
        st.caption("Institutional Forecasting")
        st.divider()
        try:
            with open(CONFIG_PATH, "r") as f: config = yaml.safe_load(f)
        except Exception: st.error("Config YAML tidak ditemukan."); return

        df, news_df = load_data(), load_news_raw()
        if df is None: st.error("Database CSV tidak ditemukan."); return

        selected_ticker = st.selectbox("Emiten Target", df['ticker'].unique(), index=0)
        selected_date = st.selectbox("Titik Waktu (T0)", df[df['ticker'] == selected_ticker]['date'].dt.date.sort_values(ascending=False).unique())
        selected_date_ts = pd.Timestamp(selected_date)
        st.caption(f"Engine: NLP({HYBRID_SCENARIO}) / Base({BASELINE_SCENARIO})")

    # LOAD MODEL
    ckpt_base, ckpt_sent = Path(CKPT_BASELINE_FILE), Path(CKPT_SENTIMENT_FILE)
    if not ckpt_base.exists() or not ckpt_sent.exists(): st.error("Model file (.ckpt) missing."); return

    model_base, model_sent = load_model(ckpt_base), load_model(ckpt_sent)
    if not model_base or not model_sent: return
    
    ds_base, _ = create_prediction_dataset(df, selected_ticker, selected_date_ts, config, "baseline")
    ds_sent, _ = create_prediction_dataset(df, selected_ticker, selected_date_ts, config, "sentiment")
    if ds_base is None or ds_sent is None: st.warning("Error Dataset"); return

    # PREDIKSI
    raw_pred_base = model_base.predict(ds_base.to_dataloader(train=False, batch_size=1, num_workers=0), mode="raw", return_x=True)
    raw_pred_sent = model_sent.predict(ds_sent.to_dataloader(train=False, batch_size=1, num_workers=0), mode="raw", return_x=True)
    pred_base = model_base.to_prediction(raw_pred_base.output).detach().cpu().numpy()[0]
    pred_sent = model_sent.to_prediction(raw_pred_sent.output).detach().cpu().numpy()[0]
    curr_row = df[(df['ticker'] == selected_ticker) & (df['date'] == selected_date_ts)].iloc[0]

    # ================== LAYOUT SINGLE PAGE GRID ==================

    # 1. HEADER (Memakai Lebar Penuh agar tidak kepotong)
    st.markdown(f"<h2 style='margin-bottom: 0px;'>{selected_ticker} <span style='font-weight:300; color:#8A94A6; font-size:24px;'>| Market Insight & Projection</span></h2>", unsafe_allow_html=True)
    st.markdown(f"<div style='color:#8A94A6; font-size:15px; margin-bottom: 25px;'>T0: <b>{selected_date.strftime('%d %b %Y')}</b> &nbsp;|&nbsp; Horizon: <b>H+3</b></div>", unsafe_allow_html=True)
    
    # 2. KPI CARDS (Ditaruh di baris baru dengan 4 kolom proporsional)
    col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
    with col_kpi1: st.metric("Harga (T0)", f"Rp {curr_row['close']:,.0f}")
    with col_kpi2: st.metric("RSI Momentum", f"{curr_row['rsi_14']:.1f}")
    with col_kpi3: st.metric("Sentimen AI", f"{curr_row['sentiment_final_mean']:.2f}")
    with col_kpi4: st.metric("Proyeksi H+3", f"Rp {pred_sent[-1]:,.0f}", f"{(pred_sent[-1] - curr_row['close']) / curr_row['close'] * 100:+.2f}%")

    st.write("---")

    # 3. MAIN WORKSPACE (Baris Tengah: Chart 70% | AI Insight 30%)
    col_chart, col_ai = st.columns([7, 3])

    # KIRI: GRAFIK
    with col_chart:
        st.markdown("<div class='dashboard-panel'>", unsafe_allow_html=True)
        st.markdown("#### 📈 Proyeksi Harga & Volume")
        future_dates = pd.bdate_range(start=selected_date_ts + pd.Timedelta(days=1), periods=len(pred_sent))
        df_hist = df[(df['ticker'] == selected_ticker) & (df['date'] <= selected_date_ts)].tail(15) 
        
        h_dates, f_dates = df_hist['date'].dt.strftime('%d %b').tolist(), future_dates.strftime('%d %b').tolist()
        h_prices, h_vols = df_hist['close'].tolist(), df_hist['volume'].tolist()
        last_h_date, last_h_price = h_dates[-1], h_prices[-1]
        
        x_pred, y_base_pred, y_sent_pred = [last_h_date] + f_dates, [last_h_price] + pred_base.tolist(), [last_h_price] + pred_sent.tolist()

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.8, 0.2])
        fig.add_trace(go.Scatter(x=h_dates, y=h_prices, mode='lines+markers', name='Aktual', line=dict(color='#FFFFFF', width=2), marker=dict(size=5, color='#FFFFFF')), row=1, col=1)
        fig.add_trace(go.Scatter(x=x_pred, y=y_base_pred, mode='lines+markers', name='TFT Base', line=dict(color='#757575', width=2, dash='dot'), marker=dict(symbol='square', size=4)), row=1, col=1)
        fig.add_trace(go.Scatter(x=x_pred, y=y_sent_pred, mode='lines+markers', name='TFT+NLP', line=dict(color='#3B82F6', width=2.5), marker=dict(symbol='circle', size=5)), row=1, col=1)
        fig.add_trace(go.Bar(x=h_dates, y=h_vols, name='Volume', marker=dict(color='#333333')), row=2, col=1)

        fig.update_layout(
            template='plotly_dark', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=10, b=0), hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#BDBDBD", size=10)),
            height=400
        )
        fig.update_yaxes(title_text="Harga", tickformat=",d", gridcolor='#212121', zerolinecolor='#212121', row=1, col=1)
        fig.update_yaxes(showticklabels=False, gridcolor='#212121', zerolinecolor='#212121', row=2, col=1) 
        fig.update_xaxes(gridcolor='#212121', showgrid=True, row=2, col=1)
        fig.add_vline(x=last_h_date, line_width=1, line_dash="dash", line_color="#757575", row='all', col='all')

        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # KANAN: AI INSIGHT
    with col_ai:
        st.markdown("<div class='dashboard-panel'>", unsafe_allow_html=True)
        st.markdown("#### 🧠 AI Strategist Insight")
        st.caption("Auto-generated text using Live Data")
        
        if st.button("Generate Insight Terbaru", use_container_width=True):
            st.session_state['run_analysis'] = True

        if st.session_state.get('run_analysis', False):
            with st.spinner("Processing..."):
                analysis = generate_expert_analysis(selected_ticker, selected_date_ts, {'rsi_14': curr_row['rsi_14'], 'ma_5_div_ma_20': curr_row['ma_5_div_ma_20']}, {'sentiment_final_mean': curr_row['sentiment_final_mean']}, news_df, pred_base, pred_sent)
                st.markdown(f"<div class='ai-box'>{analysis}</div>", unsafe_allow_html=True)
        else:
            st.info("Klik tombol di atas untuk mensintesis data sentimen dan teknikal menjadi laporan strategi singkat.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("---")

    # 4. DATA & VALIDASI (Baris Bawah: Berdampingan 50/50)
    col_data, col_val = st.columns(2)

    with col_data:
        st.markdown("<div class='dashboard-panel'>", unsafe_allow_html=True)
        st.markdown("#### 📋 Matriks Data Proyeksi (H+3)")
        table_data = [{"Horizon": f"H+{i+1} ({date.strftime('%d %b')})", "Prediksi TFT Base": f"Rp {pred_base[i]:,.0f}", "Prediksi TFT+NLP": f"Rp {pred_sent[i]:,.0f}", "Tren": "📈 Naik" if pred_sent[i] > prev_p_sent else "📉 Turun"} for i, date, prev_p_sent in zip(range(len(future_dates)), future_dates, [curr_row['close']] + pred_sent[:-1].tolist())]
        st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_val:
        st.markdown("<div class='dashboard-panel'>", unsafe_allow_html=True)
        st.markdown("#### 🎯 Validasi Model (Testing Split)")
        st.caption("Berdasarkan evaluasi 1.896 Sliding Windows")
        eval_df = pd.DataFrame({"Indikator": ["Akurasi Arah", "RMSE", "MAE", "MAPE"], "TFT Base": ["52.11%", "150.09", "110.62", "2.71%"], "TFT+NLP (Hybrid)": ["61.82%", "116.58", "84.20", "1.99%"], "Selisih Signifikansi": ["+9.71%", "-33.5", "-26.4", "-0.72%"]})
        st.table(eval_df)
        st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()