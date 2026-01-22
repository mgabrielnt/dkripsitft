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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --- 1. SETUP & KONFIGURASI ---

load_dotenv()

# Konfigurasi OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

st.set_page_config(
    page_title="TFT Expert Forecaster",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Setup Paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "model_tft.yaml"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "tft_master.csv"
NEWS_PATH = PROJECT_ROOT / "data" / "processed" / "news_with_sentiment_per_article.csv"

# Checkpoint Folders
CKPT_BASELINE_DIR = PROJECT_ROOT / "checkpoints" / "baseline"
CKPT_SENTIMENT_DIR = PROJECT_ROOT / "checkpoints" / "sentiment"

# --- DEFINISI FITUR ---
TECHNICAL_FEATURES = [
    "close", "volume", "log_return_1d", "vol_20", "rsi_14", 
    "ma_5_div_ma_20", "bb_width_20", "gap_return_1d", "intraday_range_pct"
]

SENTIMENT_FEATURES = [
    "has_news", "news_count_3d", 
    "sentiment_mean_3d", "sentiment_ema_7d", "sentiment_ema_14d",
    "sentiment_trend_7d", "sentiment_intraday_std", 
    "sentiment_vol_impact", "high_news_day",
    "sentiment_dir_signal" 
]

# --- 2. FUNGSI LOAD DATA & MODEL ---

@st.cache_data
def load_data():
    """Load data master dan fix tipe data."""
    if not DATA_PATH.exists():
        st.error(f"Data Master tidak ditemukan di {DATA_PATH}")
        return None
    
    df = pd.read_csv(DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    df['ticker'] = df['ticker'].astype(str)
    
    # Fix Categorical Types
    df['month'] = df['month'].astype(str)
    df['day_of_week'] = df['day_of_week'].astype(str)

    df = df.sort_values(['ticker', 'date']).reset_index(drop=True)
    return df

@st.cache_data
def load_news_raw():
    if not NEWS_PATH.exists(): return None
    df = pd.read_csv(NEWS_PATH)
    df['date'] = pd.to_datetime(df['date'])
    return df

@st.cache_resource
def load_model(checkpoint_path):
    try:
        # Tambahkan weights_only=False untuk kompatibilitas PyTorch 2.6
        model = TemporalFusionTransformer.load_from_checkpoint(
            checkpoint_path,
            map_location=torch.device("cpu"), 
            weights_only=False 
        )
        return model
    except Exception as e:
        st.error(f"Gagal load model: {e}")
        return None

def get_best_checkpoint(ckpt_dir):
    """Cari file .ckpt terbaik di folder tertentu."""
    if not ckpt_dir.exists(): return None
    ckpts = list(ckpt_dir.glob("*.ckpt"))
    if not ckpts: return None
    return max(ckpts, key=os.path.getmtime)

def create_prediction_dataset(df, selected_ticker, selected_date, config, model_type="sentiment"):
    """Membuat dataset prediksi."""
    df_ticker = df[df['ticker'] == selected_ticker].copy()
    
    if selected_date not in df_ticker['date'].values:
        return None, "Tanggal tidak tersedia."
    
    cutoff_idx = df_ticker[df_ticker['date'] == selected_date]['time_idx'].values[0]
    max_encoder = config['data']['max_encoder_length']
    max_pred = config['data']['max_prediction_length']
    
    if cutoff_idx < max_encoder:
        return None, "Data historis kurang."
        
    data_subset = df_ticker[df_ticker['time_idx'] <= cutoff_idx].tail(max_encoder + 10)
    
    if model_type == "baseline":
        selected_features = TECHNICAL_FEATURES
    else:
        selected_features = TECHNICAL_FEATURES + SENTIMENT_FEATURES

    try:
        dataset = TimeSeriesDataSet(
            data_subset,
            time_idx="time_idx",
            target="close",
            group_ids=["ticker"],
            min_encoder_length=max_encoder,
            max_encoder_length=max_encoder,
            max_prediction_length=max_pred,
            static_categoricals=["ticker"],
            time_varying_known_categoricals=["month", "day_of_week"],
            time_varying_known_reals=["time_idx", "is_month_end"],
            time_varying_unknown_reals=selected_features, 
            target_normalizer=GroupNormalizer(groups=["ticker"], transformation="softplus"),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
            predict_mode=True 
        )
        return dataset, None
    except Exception as e:
        return None, str(e)

# --- 3. FUNGSI AI ANALYST ---

def generate_expert_analysis(ticker, date, tech_data, sent_data, news_df, preds_base, preds_sent):
    """
    Prompt Expert yang WAJIB mengutip Judul Berita dan Angka Teknikal Spesifik.
    """
    if not OPENAI_API_KEY:
        return "⚠️ API Key OpenAI belum disetting."

    # 1. Siapkan Data Berita
    start_date = date - pd.Timedelta(days=5)
    relevant_news = news_df[
        (news_df['ticker'] == ticker) & 
        (news_df['date'] >= start_date) & 
        (news_df['date'] <= date)
    ]
    
    if not relevant_news.empty:
        headlines = relevant_news['title'].unique()[:5]
        headlines_list = "\n".join([f"- '{h}'" for h in headlines])
    else:
        headlines_list = "- (Tidak ada berita spesifik dalam 5 hari terakhir)"

    # 2. Interpretasi Teknikal
    rsi = tech_data['rsi_14']
    ma_div = tech_data['ma_5_div_ma_20']
    ma_status = "Uptrend (MA5 > MA20)" if ma_div > 1 else "Downtrend (MA5 < MA20)"
    
    # 3. Interpretasi Sentimen
    impact = sent_data['sentiment_vol_impact']
    if impact > 0.5: sent_mood = "POSITIF"
    elif impact < -0.5: sent_mood = "NEGATIF"
    else: sent_mood = "NETRAL"

    # 4. Interpretasi OUTPUT MODEL
    sent_trend_val = preds_sent[-1] - preds_sent[0]
    sent_model_direction = "NAIK" if sent_trend_val > 0 else "TURUN"
    
    # 5. PROMPT ENGINEERING
    prompt = f"""
    Anda adalah Senior Investment Strategist.
    
    EMITEN: {ticker} (Tanggal: {date.strftime('%Y-%m-%d')})
    
    TUGAS:
    Jelaskan mengapa Model Hybrid memprediksi harga akan **{sent_model_direction}**?
    
    DATA FAKTA (WAJIB DISEBUTKAN):
    1. **Data Teknikal:** RSI={rsi:.2f}, Tren={ma_status}.
    2. **Data Berita (Katalis):**
    {headlines_list}
    3. **Sentimen AI:** Skor {impact:.2f} ({sent_mood}).
    
    INSTRUKSI:
    - KUTIP LANGSUNG JUDUL BERITA PENTING.
    - SEBUTKAN ANGKA RSI ATAU MA.
    - Hubungkan berita spesifik dengan prediksi model.
    - Tulis dalam 1 paragraf profesional.
    """
    
    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a financial analyst who quotes specific data points."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=350
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error GPT: {str(e)}"

# --- 4. MAIN UI ---

def main():
    st.sidebar.title("🎛️ Expert Control")
    
    # Load Resources
    with open(CONFIG_PATH, "r") as f: config = yaml.safe_load(f)
    df = load_data()
    news_df = load_news_raw()
    
    if df is None: return

    # Sidebar Inputs
    tickers = df['ticker'].unique()
    selected_ticker = st.sidebar.selectbox("Pilih Saham", tickers, index=0)
    
    available_dates = df[df['ticker'] == selected_ticker]['date'].dt.date.sort_values(ascending=False).unique()
    selected_date = st.sidebar.selectbox("Tanggal Origin", available_dates)
    selected_date_ts = pd.Timestamp(selected_date)

    # Load Models
    ckpt_base = get_best_checkpoint(CKPT_BASELINE_DIR)
    ckpt_sent = get_best_checkpoint(CKPT_SENTIMENT_DIR)
    
    if not ckpt_base or not ckpt_sent:
        st.error("Checkpoint model Baseline atau Sentiment tidak lengkap.")
        return

    model_base = load_model(ckpt_base)
    model_sent = load_model(ckpt_sent)
    
    if not model_base or not model_sent: return
    
    # --- HEADER ---
    st.title(f"💹 Techno-Fundamental Analysis: {selected_ticker}")
    st.caption(f"Origin Date: {selected_date.strftime('%d %B %Y')} | Horizon: 3 Days")
    
    # Data Prep & Prediction
    ds_base, err1 = create_prediction_dataset(df, selected_ticker, selected_date_ts, config, "baseline")
    ds_sent, err2 = create_prediction_dataset(df, selected_ticker, selected_date_ts, config, "sentiment")
    
    if ds_base is None or ds_sent is None:
        st.error(f"Error Data: {err1 or err2}")
        return

    dl_base = ds_base.to_dataloader(train=False, batch_size=1, num_workers=0)
    dl_sent = ds_sent.to_dataloader(train=False, batch_size=1, num_workers=0)
    
    # Gunakan to_prediction untuk memastikan output Rupiah
    raw_pred_base = model_base.predict(dl_base, mode="raw", return_x=True)
    raw_pred_sent = model_sent.predict(dl_sent, mode="raw", return_x=True)
    
    pred_base = model_base.to_prediction(raw_pred_base.output).detach().cpu().numpy()[0]
    pred_sent = model_sent.to_prediction(raw_pred_sent.output).detach().cpu().numpy()[0]
    
    curr_row = df[(df['ticker'] == selected_ticker) & (df['date'] == selected_date_ts)].iloc[0]

    # --- SECTION 1: KEY METRICS ---
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Closing Price", f"Rp {curr_row['close']:,.0f}")
    with col2:
        rsi = curr_row['rsi_14']
        st.metric("RSI (14)", f"{rsi:.1f}", "Overbought" if rsi>70 else "Oversold" if rsi<30 else "Netral", delta_color="off")
    with col3:
        impact = curr_row['sentiment_vol_impact']
        st.metric("AI Sentiment Score", f"{impact:.2f}", "Bullish" if impact>0.5 else "Bearish" if impact<-0.5 else "Neutral")
    with col4:
        diff_models = pred_sent[-1] - pred_base[-1]
        st.metric("Sentiment Premium", f"Rp {diff_models:+,.0f}", help="Selisih harga prediksi Sentimen vs Baseline")

    # --- SECTION 2: GLOBAL DIAGNOSIS (ANGKA VALID & FINAL) ---
    st.markdown("---")
    st.subheader("📊 Hasil Diagnosis Model (Global Benchmark)")
    st.info("Berikut adalah performa model pada **1.896 Sliding Windows** (Test Set). Hasil ini telah divalidasi dengan metode Raw Data Lookup.")
    
    # Data Global (UPDATED DARI EVALUATE_RESULTS TERAKHIR)
    diag_data = {
        "Indikator Kinerja": [
            "Directional Accuracy (Akurasi Arah)",
            "RMSE (Root Mean Squared Error)",
            "MAE (Mean Absolute Error)", 
            "MAPE (Mean Absolute % Error)",
            "Improvement (Hybrid vs Base)"
        ],
        "Baseline (Teknikal Only)": [
            "52.11% (Acak/Coin Flip)",
            "150.09", 
            "110.62",
            "2.71%",
            "-"
        ],
        "Hybrid (Sentiment + Gating)": [
            "61.82% 👑",
            "116.58 👑",
            "84.20 👑",
            "1.99% 👑",
            "Signifikan ✅"
        ],
        "Kesimpulan": [
            "Sentimen Berita meningkatkan akurasi arah sebesar +9.70%",
            "Sentimen Berita menurunkan Error (RMSE) sebesar 33 poin",
            "Sentimen Berita membuat prediksi lebih stabil mendekati harga asli",
            "Tingkat kesalahan rata-rata hanya 1.99% (Sangat Presisi)",
            "Hipotesis Skripsi: DITERIMA (Berita Berpengaruh Positif)"
        ]
    }
    st.table(pd.DataFrame(diag_data))

    # --- SECTION 3: EXPERT AI ANALYSIS ---
    st.markdown("---")
    st.subheader("🧠 Expert Analyst Insight")
    
    if st.button("Generate Expert Analysis (Techno-Fundamental)"):
        with st.spinner("Menganalisis Chart Pattern & Berita..."):
            tech_data = {
                'rsi_14': curr_row['rsi_14'],
                'ma_5_div_ma_20': curr_row['ma_5_div_ma_20'],
            }
            sent_data = {
                'sentiment_vol_impact': curr_row['sentiment_vol_impact']
            }
            analysis = generate_expert_analysis(
                selected_ticker, selected_date_ts, 
                tech_data, sent_data, news_df,
                pred_base, pred_sent 
            )
            st.markdown(f"""
            <div style="border: 1px solid #444; padding: 20px; border-radius: 5px; background-color: #1E1E1E; color: #EEE;">
                <h4 style="color: #00CC96; margin-top:0;">🤖 Komentar Analis AI</h4>
                <p style="font-size: 1.1em; line-height: 1.6; font-family: sans-serif;">{analysis}</p>
            </div>
            """, unsafe_allow_html=True)

    # --- SECTION 4: PREDIKSI HARGA ---
    st.markdown("---")
    st.subheader("📋 Prediksi Harga (3 Hari ke Depan)")
    
    future_dates = pd.bdate_range(start=selected_date_ts + pd.Timedelta(days=1), periods=len(pred_sent))
    table_data = []
    prev_p_sent = curr_row['close']

    for i, date in enumerate(future_dates):
        row = {
            "Tanggal": date.strftime('%Y-%m-%d'),
            "Baseline (Rp)": f"{pred_base[i]:,.0f}",
            "Sentiment (Rp)": f"{pred_sent[i]:,.0f}",
            "Delta (Sent - Base)": f"{pred_sent[i] - pred_base[i]:+,.0f}",
            "Arah Sentiment": "⬆️" if pred_sent[i] > prev_p_sent else "⬇️"
        }
        table_data.append(row)
        prev_p_sent = pred_sent[i]

    st.table(pd.DataFrame(table_data))

if __name__ == "__main__":
    main()