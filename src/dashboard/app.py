import os
import sys
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# 1. KONFIGURASI DASAR
# ============================================================
st.set_page_config(
    page_title="Dashboard Prediksi Saham Indonesia",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reportss"
MODELS_DIR = PROJECT_ROOT / "modelssss"

PRICE_PATHS = [
    INTERIM_DIR / "prices_with_indicators.csv",
    PROCESSED_DIR / "prices_with_indicators.csv",
    DATA_DIR / "prices_with_indicators.csv",
]

NEWS_PATHS = [
    INTERIM_DIR / "news_clean.csv",
    PROCESSED_DIR / "news_clean.csv",
    PROCESSED_DIR / "news_with_sentiment_per_article.csv",
    DATA_DIR / "news_clean.csv",
]

ARTICLE_SENTIMENT_PATHS = [
    PROCESSED_DIR / "news_with_sentiment_per_article.csv",
    INTERIM_DIR / "news_with_sentiment_per_article.csv",
]

DAILY_SENTIMENT_PATHS = [
    PROCESSED_DIR / "daily_sentiment.csv",
    INTERIM_DIR / "daily_sentiment.csv",
]

MASTER_PATHS = [
    PROCESSED_DIR / "tft_master.csv",
    INTERIM_DIR / "tft_master.csv",
]

EVAL_GLOBAL_PATHS = [
    REPORTS_DIR / "eval_metrics_global.csv",
    REPORTS_DIR / "evaluation_metrics_global.csv",
    PROCESSED_DIR / "eval_metrics_global.csv",
]

EVAL_TICKER_PATHS = [
    REPORTS_DIR / "eval_metrics_by_ticker_global.csv",
    REPORTS_DIR / "eval_metrics_by_ticker.csv",
    PROCESSED_DIR / "eval_metrics_by_ticker_global.csv",
]

EVAL_HORIZON_PATHS = [
    REPORTS_DIR / "eval_metrics_by_horizon.csv",
    PROCESSED_DIR / "eval_metrics_by_horizon.csv",
]

BACKTEST_PATHS = [
    REPORTS_DIR / "backtest_predictions.csv",
    REPORTS_DIR / "tft_backtest.csv",
    PROCESSED_DIR / "backtest_predictions.csv",
]

PIPELINE_COMMANDS = {
    "Ambil berita RSS dan Google News": "python -m src.data.fetch_news_rss_google",
    "Ambil berita Yahoo Finance": "python -m src.data.fetch_news_yahoo",
    "Gabung sumber berita": "python -m src.data.merge_news_sources",
    "Bersihkan teks berita": "python -m src.data.preprocess_news_text",
    "Ambil harga Yahoo Finance": "python -m src.data.download_prices_yahoo",
    "Hitung indikator teknikal": "python -m src.data.compute_technical_indicators",
    "Audit kalender harga": "python -m src.data.check_price_calendar",
    "Label sentimen artikel": "python -m src.data.gpt_sentiment_labeling",
    "Agregasi sentimen harian": "python -m src.data.aggregate_daily_sentiment",
    "Bangun dataset master": "python -m src.data.build_tft_master_dataset",
    "Latih TFT": "python -m src.models.train_tft_baseline",
    "Latih LLM-TFT": "python -m src.models.train_tft_with_sentiment",
    "Evaluasi model": "python -m src.models.evaluate_tft_models",
    "Backtest model": "python -m src.models.evaluate_tft_backtest",
    "Interpretasi model": "python -m src.models.interpret_tft_models",
}

TECHNICAL_FEATURES = [
    "close",
    "volume",
    "log_return_1d",
    "log_return_2d",
    "vol_20",
    "rsi_14",
    "ma_5_div_ma_20",
    "bb_width_20",
    "gap_return_1d",
    "intraday_range_pct",
]

SENTIMENT_FEATURES = [
    "news_count_3d",
    "sentiment_mean_3d",
    "sentiment_ema_7d",
    "sentiment_trend_7d",
    "sentiment_delta_1d",
    "sentiment_dir_signal",
]


# ============================================================
# 2. STYLE DASHBOARD
# ============================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #07111f 0%, #0d1323 45%, #101827 100%);
        color: #e5e7eb;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #07111f 0%, #101827 100%);
        border-right: 1px solid rgba(148, 163, 184, 0.18);
    }

    .main-title {
        padding: 22px 26px;
        border-radius: 24px;
        background:
            radial-gradient(circle at top left, rgba(59, 130, 246, 0.36), transparent 35%),
            linear-gradient(135deg, rgba(15, 23, 42, 0.95), rgba(30, 41, 59, 0.75));
        border: 1px solid rgba(148, 163, 184, 0.24);
        box-shadow: 0 20px 55px rgba(0, 0, 0, 0.28);
        margin-bottom: 22px;
    }

    .main-title h1 {
        margin: 0;
        font-size: 2.05rem;
        letter-spacing: -0.04em;
        color: #f8fafc;
    }

    .main-title p {
        margin: 8px 0 0 0;
        color: #cbd5e1;
        font-size: 0.98rem;
    }

    .section-card {
        padding: 18px 20px;
        border-radius: 20px;
        background: rgba(15, 23, 42, 0.78);
        border: 1px solid rgba(148, 163, 184, 0.18);
        box-shadow: 0 14px 35px rgba(0, 0, 0, 0.22);
        margin-bottom: 18px;
    }

    .mini-card {
        padding: 16px 18px;
        border-radius: 18px;
        background: rgba(15, 23, 42, 0.72);
        border: 1px solid rgba(148, 163, 184, 0.16);
        min-height: 106px;
    }

    .mini-card h4 {
        margin: 0;
        color: #94a3b8;
        font-size: 0.83rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    .mini-card h2 {
        margin: 8px 0 0 0;
        color: #f8fafc;
        font-size: 1.72rem;
        letter-spacing: -0.03em;
    }

    .mini-card p {
        margin: 6px 0 0 0;
        color: #cbd5e1;
        font-size: 0.86rem;
    }

    .step-card {
        padding: 14px 16px;
        border-radius: 16px;
        background: rgba(30, 41, 59, 0.74);
        border: 1px solid rgba(148, 163, 184, 0.16);
        margin-bottom: 10px;
    }

    .step-card b {
        color: #f8fafc;
    }

    .step-card span {
        color: #cbd5e1;
        font-size: 0.9rem;
    }

    .status-ok {
        color: #34d399;
        font-weight: 700;
    }

    .status-warn {
        color: #fbbf24;
        font-weight: 700;
    }

    div[data-testid="stMetric"] {
        background: rgba(15, 23, 42, 0.78);
        border: 1px solid rgba(148, 163, 184, 0.18);
        padding: 15px 16px;
        border-radius: 18px;
        box-shadow: 0 10px 24px rgba(0, 0, 0, 0.22);
    }

    .stDataFrame {
        border-radius: 18px;
        overflow: hidden;
        border: 1px solid rgba(148, 163, 184, 0.16);
    }

    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 3. UTILITAS DATA
# ============================================================
def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


@st.cache_data(show_spinner=False)
def read_csv(path: str | Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    file_path = Path(path)
    if not file_path.exists():
        return None

    try:
        df = pd.read_csv(file_path)
    except Exception:
        return None

    for col in df.columns:
        if col.lower() in {"date", "published_at", "publish_date", "datetime", "timestamp"}:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    if "ticker" in df.columns:
        df["ticker"] = df["ticker"].astype(str)

    return df


def format_number(value, decimals: int = 0) -> str:
    if value is None or pd.isna(value):
        return "-"
    try:
        if decimals == 0:
            return f"{float(value):,.0f}".replace(",", ".")
        return f"{float(value):,.{decimals}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return str(value)


def get_date_col(df: pd.DataFrame | None) -> str | None:
    if df is None:
        return None
    candidates = ["date", "published_at", "publish_date", "datetime", "timestamp"]
    for col in candidates:
        if col in df.columns:
            return col
    return None


def filter_by_sidebar(df: pd.DataFrame | None, ticker: str | None, date_range: tuple | None) -> pd.DataFrame | None:
    if df is None:
        return None

    out = df.copy()
    if ticker and ticker != "Semua" and "ticker" in out.columns:
        out = out[out["ticker"] == ticker]

    date_col = get_date_col(out)
    if date_col and date_range and len(date_range) == 2:
        start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        out = out[(out[date_col] >= start) & (out[date_col] <= end)]

    return out


def plot_layout(fig, height: int = 420):
    fig.update_layout(
        template="plotly_dark",
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.35)",
        margin=dict(l=18, r=18, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font=dict(color="#e5e7eb"),
    )
    fig.update_xaxes(gridcolor="rgba(148,163,184,0.16)", zerolinecolor="rgba(148,163,184,0.16)")
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.16)", zerolinecolor="rgba(148,163,184,0.16)")
    return fig


def run_command(command: str):
    with st.spinner(f"Menjalankan: {command}"):
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=None,
            )
        except Exception as exc:
            st.error(f"Gagal menjalankan perintah: {exc}")
            return

    if result.returncode == 0:
        st.success("Proses selesai.")
    else:
        st.error("Proses gagal. Periksa pesan error di bawah.")

    with st.expander("Lihat log terminal"):
        if result.stdout:
            st.code(result.stdout[-8000:])
        if result.stderr:
            st.code(result.stderr[-8000:])


def action_button(label: str, command_key: str):
    command = PIPELINE_COMMANDS[command_key]
    if st.button(label, use_container_width=True):
        run_command(command)


def data_status(path: Path | None) -> str:
    if path and path.exists():
        return f"<span class='status-ok'>Tersedia</span><br><small>{path.relative_to(PROJECT_ROOT)}</small>"
    return "<span class='status-warn'>Belum tersedia</span><br><small>Jalankan pipeline terkait</small>"


def render_header(title: str, subtitle: str):
    st.markdown(
        f"""
        <div class="main-title">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(title: str, value: str, note: str):
    st.markdown(
        f"""
        <div class="mini-card">
            <h4>{title}</h4>
            <h2>{value}</h2>
            <p>{note}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_process_step(number: int, title: str, desc: str):
    st.markdown(
        f"""
        <div class="step-card">
            <b>{number}. {title}</b><br>
            <span>{desc}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# 4. LOAD DATA GLOBAL
# ============================================================
price_path = first_existing(PRICE_PATHS)
news_path = first_existing(NEWS_PATHS)
article_sentiment_path = first_existing(ARTICLE_SENTIMENT_PATHS)
daily_sentiment_path = first_existing(DAILY_SENTIMENT_PATHS)
master_path = first_existing(MASTER_PATHS)
eval_global_path = first_existing(EVAL_GLOBAL_PATHS)
eval_ticker_path = first_existing(EVAL_TICKER_PATHS)
eval_horizon_path = first_existing(EVAL_HORIZON_PATHS)
backtest_path = first_existing(BACKTEST_PATHS)

prices = read_csv(price_path)
news = read_csv(news_path)
article_sentiment = read_csv(article_sentiment_path)
daily_sentiment = read_csv(daily_sentiment_path)
master = read_csv(master_path)
eval_global = read_csv(eval_global_path)
eval_ticker = read_csv(eval_ticker_path)
eval_horizon = read_csv(eval_horizon_path)
backtest = read_csv(backtest_path)


# ============================================================
# 5. SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## 📈 Prediksi Saham")
    st.caption("Dashboard TFT dan LLM-TFT")
    st.divider()

    page = st.radio(
        "Menu Dashboard",
        [
            "Ringkasan Sistem",
            "Data Harga Saham",
            "Data Berita Keuangan",
            "Sentimen Harian",
            "Dataset Model",
            "Pelatihan dan Prediksi",
            "Evaluasi dan Backtest",
        ],
    )

    st.divider()
    st.markdown("### Filter Data")

    ticker_source = master if master is not None and "ticker" in master.columns else prices
    if ticker_source is not None and "ticker" in ticker_source.columns:
        tickers = ["Semua"] + sorted(ticker_source["ticker"].dropna().astype(str).unique().tolist())
    else:
        tickers = ["Semua"]

    selected_ticker = st.selectbox("Emiten", tickers)

    date_source = master if master is not None and get_date_col(master) else prices
    date_col_sidebar = get_date_col(date_source)
    date_range = None

    if date_source is not None and date_col_sidebar:
        min_date = pd.to_datetime(date_source[date_col_sidebar]).min()
        max_date = pd.to_datetime(date_source[date_col_sidebar]).max()
        if pd.notna(min_date) and pd.notna(max_date):
            date_range = st.date_input(
                "Rentang Tanggal",
                value=(min_date.date(), max_date.date()),
                min_value=min_date.date(),
                max_value=max_date.date(),
            )

    st.divider()
    st.markdown("### Status File")
    st.markdown(f"**Harga:** {data_status(price_path)}", unsafe_allow_html=True)
    st.markdown(f"**Berita:** {data_status(news_path)}", unsafe_allow_html=True)
    st.markdown(f"**Sentimen:** {data_status(daily_sentiment_path)}", unsafe_allow_html=True)
    st.markdown(f"**Dataset:** {data_status(master_path)}", unsafe_allow_html=True)
    st.markdown(f"**Evaluasi:** {data_status(eval_global_path)}", unsafe_allow_html=True)


# ============================================================
# 6. HALAMAN RINGKASAN
# ============================================================
if page == "Ringkasan Sistem":
    render_header(
        "Dashboard Prediksi Harga Saham Indonesia",
        "Monitoring profesional untuk data harga, berita keuangan, sentimen, dataset model, pelatihan, prediksi, evaluasi, dan backtest.",
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card("Data Harga", format_number(len(prices) if prices is not None else 0), "baris harga dan indikator")
    with c2:
        render_metric_card("Data Berita", format_number(len(news) if news is not None else 0), "baris berita keuangan")
    with c3:
        render_metric_card("Sentimen Harian", format_number(len(daily_sentiment) if daily_sentiment is not None else 0), "baris agregasi sentimen")
    with c4:
        render_metric_card("Dataset Model", format_number(len(master) if master is not None else 0), "baris tft_master")

    st.write("")

    left, right = st.columns([1.05, 1.2])
    with left:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Alur Sistem")
        steps = [
            ("Pengolahan Data Harga Saham", "Mengambil harga harian, menghitung indikator teknikal, dan melakukan audit kalender perdagangan."),
            ("Pengolahan Data Berita Keuangan", "Mengumpulkan, menggabungkan, dan membersihkan berita dari beberapa sumber."),
            ("Pelabelan dan Agregasi Sentimen", "Memberi label artikel dengan LLM, leksikon, dan respons pasar, lalu membentuk fitur harian."),
            ("Pembentukan Dataset Model", "Menggabungkan harga, indikator teknikal, kalender, dan sentimen ke dataset master."),
            ("Pelatihan Model Prediksi", "Melatih TFT dan LLM-TFT untuk horizon H+1, H+2, dan H+3."),
            ("Evaluasi dan Penyajian Hasil", "Membandingkan model dengan MAE, RMSE, MAPE, R², Directional Accuracy, dan backtest."),
        ]
        for i, (title, desc) in enumerate(steps, start=1):
            render_process_step(i, title, desc)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Kesiapan Komponen Sistem")
        status_df = pd.DataFrame(
            {
                "Komponen": ["Harga", "Berita", "Sentimen Artikel", "Sentimen Harian", "Dataset Master", "Evaluasi"],
                "Status": [
                    "Tersedia" if price_path else "Belum tersedia",
                    "Tersedia" if news_path else "Belum tersedia",
                    "Tersedia" if article_sentiment_path else "Belum tersedia",
                    "Tersedia" if daily_sentiment_path else "Belum tersedia",
                    "Tersedia" if master_path else "Belum tersedia",
                    "Tersedia" if eval_global_path else "Belum tersedia",
                ],
            }
        )
        fig = px.pie(status_df, names="Status", title="Status File Utama", hole=0.55)
        st.plotly_chart(plot_layout(fig, 350), use_container_width=True)

        if eval_global is not None and not eval_global.empty:
            metric_cols = [c for c in ["MAE", "RMSE", "MAPE", "R2", "R²", "Directional Accuracy", "DirAcc"] if c in eval_global.columns]
            model_col = "model" if "model" in eval_global.columns else "Model" if "Model" in eval_global.columns else None
            if model_col and metric_cols:
                chosen_metric = "RMSE" if "RMSE" in metric_cols else metric_cols[0]
                fig_metric = px.bar(eval_global, x=model_col, y=chosen_metric, title=f"Perbandingan Model Berdasarkan {chosen_metric}")
                st.plotly_chart(plot_layout(fig_metric, 330), use_container_width=True)
            else:
                st.dataframe(eval_global, use_container_width=True, hide_index=True)
        else:
            st.info("File evaluasi belum ditemukan. Jalankan evaluasi model untuk menampilkan perbandingan TFT dan LLM-TFT.")
        st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# 7. HALAMAN DATA HARGA
# ============================================================
elif page == "Data Harga Saham":
    render_header(
        "Pengolahan Data Harga Saham",
        "Halaman ini menampilkan harga harian, volume, indikator teknikal, audit kalender, dan tombol proses untuk membangun input teknikal model.",
    )

    with st.expander("Jalankan proses data harga", expanded=False):
        b1, b2, b3 = st.columns(3)
        with b1:
            action_button("Ambil harga Yahoo Finance", "Ambil harga Yahoo Finance")
        with b2:
            action_button("Hitung indikator teknikal", "Hitung indikator teknikal")
        with b3:
            action_button("Audit kalender harga", "Audit kalender harga")

    df_price = filter_by_sidebar(prices, selected_ticker, date_range)

    if df_price is None or df_price.empty:
        st.warning("Data harga belum tersedia. Jalankan proses harga terlebih dahulu.")
    else:
        date_col = get_date_col(df_price) or "date"
        latest = df_price.sort_values(date_col).tail(1).iloc[0]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Jumlah Baris", format_number(len(df_price)))
        c2.metric("Jumlah Emiten", format_number(df_price["ticker"].nunique() if "ticker" in df_price.columns else 0))
        c3.metric("Close Terakhir", f"Rp {format_number(latest['close'])}" if "close" in df_price.columns else "-")
        c4.metric("Volume Terakhir", format_number(latest["volume"]) if "volume" in df_price.columns else "-")

        tab1, tab2, tab3 = st.tabs(["Grafik Harga", "Indikator Teknikal", "Tabel Data"])

        with tab1:
            chart_df = df_price.sort_values(date_col)
            if "close" in chart_df.columns:
                fig = px.line(chart_df, x=date_col, y="close", color="ticker" if selected_ticker == "Semua" and "ticker" in chart_df.columns else None, title="Pergerakan Harga Penutupan")
                st.plotly_chart(plot_layout(fig), use_container_width=True)

            if "volume" in chart_df.columns:
                fig_vol = px.bar(chart_df.tail(250), x=date_col, y="volume", color="ticker" if selected_ticker == "Semua" and "ticker" in chart_df.columns else None, title="Volume Transaksi")
                st.plotly_chart(plot_layout(fig_vol, 360), use_container_width=True)

        with tab2:
            available_tech = [col for col in TECHNICAL_FEATURES if col in df_price.columns]
            if available_tech:
                selected_feature = st.selectbox("Pilih indikator", available_tech, index=0)
                fig_feature = px.line(
                    df_price.sort_values(date_col),
                    x=date_col,
                    y=selected_feature,
                    color="ticker" if selected_ticker == "Semua" and "ticker" in df_price.columns else None,
                    title=f"Tren {selected_feature}",
                )
                st.plotly_chart(plot_layout(fig_feature), use_container_width=True)

                corr_cols = [col for col in available_tech if pd.api.types.is_numeric_dtype(df_price[col])]
                if len(corr_cols) >= 2:
                    corr = df_price[corr_cols].corr(numeric_only=True)
                    fig_corr = px.imshow(corr, text_auto=".2f", title="Korelasi Indikator Teknikal")
                    st.plotly_chart(plot_layout(fig_corr, 520), use_container_width=True)
            else:
                st.info("Kolom indikator teknikal belum ditemukan pada data harga.")

        with tab3:
            st.dataframe(df_price.tail(500), use_container_width=True, hide_index=True)


# ============================================================
# 8. HALAMAN DATA BERITA
# ============================================================
elif page == "Data Berita Keuangan":
    render_header(
        "Pengolahan Data Berita Keuangan",
        "Halaman ini digunakan untuk memantau pengumpulan, penggabungan, dan pembersihan berita keuangan sebagai sumber informasi eksternal.",
    )

    with st.expander("Jalankan proses berita", expanded=False):
        b1, b2, b3, b4 = st.columns(4)
        with b1:
            action_button("Ambil RSS/Google", "Ambil berita RSS dan Google News")
        with b2:
            action_button("Ambil Yahoo", "Ambil berita Yahoo Finance")
        with b3:
            action_button("Gabung berita", "Gabung sumber berita")
        with b4:
            action_button("Bersihkan berita", "Bersihkan teks berita")

    df_news = filter_by_sidebar(news, selected_ticker, date_range)

    if df_news is None or df_news.empty:
        st.warning("Data berita belum tersedia. Jalankan proses berita terlebih dahulu.")
    else:
        date_col = get_date_col(df_news)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Jumlah Berita", format_number(len(df_news)))
        c2.metric("Jumlah Emiten", format_number(df_news["ticker"].nunique() if "ticker" in df_news.columns else 0))
        c3.metric("Jumlah Sumber", format_number(df_news["source"].nunique() if "source" in df_news.columns else 0))
        c4.metric("Kolom Data", format_number(len(df_news.columns)))

        left, right = st.columns(2)

        with left:
            if "source" in df_news.columns:
                source_count = df_news["source"].fillna("Tidak diketahui").value_counts().reset_index()
                source_count.columns = ["Sumber", "Jumlah"]
                fig = px.bar(source_count.head(15), x="Sumber", y="Jumlah", title="Jumlah Berita per Sumber")
                st.plotly_chart(plot_layout(fig), use_container_width=True)
            elif "ticker" in df_news.columns:
                ticker_count = df_news["ticker"].value_counts().reset_index()
                ticker_count.columns = ["Ticker", "Jumlah"]
                fig = px.pie(ticker_count, names="Ticker", values="Jumlah", title="Proporsi Berita per Emiten", hole=0.45)
                st.plotly_chart(plot_layout(fig), use_container_width=True)

        with right:
            if date_col:
                daily_news = (
                    df_news.dropna(subset=[date_col])
                    .assign(day=lambda x: pd.to_datetime(x[date_col]).dt.date)
                    .groupby("day")
                    .size()
                    .reset_index(name="Jumlah")
                )
                fig = px.area(daily_news, x="day", y="Jumlah", title="Tren Jumlah Berita Harian")
                st.plotly_chart(plot_layout(fig), use_container_width=True)
            elif "ticker" in df_news.columns:
                ticker_count = df_news["ticker"].value_counts().reset_index()
                ticker_count.columns = ["Ticker", "Jumlah"]
                fig = px.bar(ticker_count, x="Ticker", y="Jumlah", title="Jumlah Berita per Emiten")
                st.plotly_chart(plot_layout(fig), use_container_width=True)

        st.subheader("Preview Data Berita")
        preview_cols = [c for c in ["date", "published_at", "ticker", "source", "title", "description", "clean_text", "text_for_label"] if c in df_news.columns]
        st.dataframe(df_news[preview_cols].tail(500) if preview_cols else df_news.tail(500), use_container_width=True, hide_index=True)


# ============================================================
# 9. HALAMAN SENTIMEN
# ============================================================
elif page == "Sentimen Harian":
    render_header(
        "Pelabelan dan Agregasi Sentimen",
        "Halaman ini menampilkan hasil label sentimen per artikel dan fitur sentimen harian yang dipakai pada model LLM-TFT.",
    )

    with st.expander("Jalankan proses sentimen", expanded=False):
        b1, b2 = st.columns(2)
        with b1:
            action_button("Label sentimen artikel", "Label sentimen artikel")
        with b2:
            action_button("Agregasi sentimen harian", "Agregasi sentimen harian")

    df_article = filter_by_sidebar(article_sentiment, selected_ticker, date_range)
    df_daily = filter_by_sidebar(daily_sentiment, selected_ticker, date_range)

    t1, t2 = st.tabs(["Sentimen Artikel", "Sentimen Harian"])

    with t1:
        if df_article is None or df_article.empty:
            st.warning("Data sentimen artikel belum tersedia.")
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Jumlah Artikel", format_number(len(df_article)))
            c2.metric("LLM Label", "Ada" if "l_text" in df_article.columns else "Tidak ada")
            c3.metric("Leksikon", "Ada" if "l_lex" in df_article.columns else "Tidak ada")
            c4.metric("Respons Pasar", "Ada" if "l_market" in df_article.columns else "Tidak ada")

            label_col = "l_final" if "l_final" in df_article.columns else "sentiment" if "sentiment" in df_article.columns else None
            if label_col:
                label_map = {-1: "Negatif", 0: "Netral", 1: "Positif", "-1": "Negatif", "0": "Netral", "1": "Positif"}
                sentiment_count = df_article[label_col].map(label_map).fillna(df_article[label_col].astype(str)).value_counts().reset_index()
                sentiment_count.columns = ["Sentimen", "Jumlah"]
                fig = px.pie(sentiment_count, names="Sentimen", values="Jumlah", title="Distribusi Label Sentimen Artikel", hole=0.5)
                st.plotly_chart(plot_layout(fig, 430), use_container_width=True)

            preview_cols = [c for c in ["date", "published_at", "ticker", "title", "l_text", "l_lex", "l_market", "l_final", "sentiment_conf"] if c in df_article.columns]
            st.dataframe(df_article[preview_cols].tail(500) if preview_cols else df_article.tail(500), use_container_width=True, hide_index=True)

    with t2:
        if df_daily is None or df_daily.empty:
            st.warning("Data sentimen harian belum tersedia.")
        else:
            date_col = get_date_col(df_daily) or "date"
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Baris Sentimen Harian", format_number(len(df_daily)))
            c2.metric("Jumlah Emiten", format_number(df_daily["ticker"].nunique() if "ticker" in df_daily.columns else 0))
            c3.metric("Rata-rata News Count", format_number(df_daily["news_count_3d"].mean(), 2) if "news_count_3d" in df_daily.columns else "-")
            c4.metric("Rata-rata Sentimen", format_number(df_daily["sentiment_mean_3d"].mean(), 3) if "sentiment_mean_3d" in df_daily.columns else "-")

            available_sent = [col for col in SENTIMENT_FEATURES if col in df_daily.columns]
            if available_sent:
                selected_sent_feature = st.selectbox("Pilih fitur sentimen", available_sent)
                fig = px.line(
                    df_daily.sort_values(date_col),
                    x=date_col,
                    y=selected_sent_feature,
                    color="ticker" if selected_ticker == "Semua" and "ticker" in df_daily.columns else None,
                    title=f"Tren {selected_sent_feature}",
                )
                st.plotly_chart(plot_layout(fig), use_container_width=True)

                if "ticker" in df_daily.columns:
                    agg = df_daily.groupby("ticker")[selected_sent_feature].mean().reset_index()
                    fig_bar = px.bar(agg, x="ticker", y=selected_sent_feature, title=f"Rata-rata {selected_sent_feature} per Emiten")
                    st.plotly_chart(plot_layout(fig_bar, 360), use_container_width=True)

            st.dataframe(df_daily.tail(500), use_container_width=True, hide_index=True)


# ============================================================
# 10. HALAMAN DATASET MODEL
# ============================================================
elif page == "Dataset Model":
    render_header(
        "Pembentukan Dataset Model",
        "Halaman ini memeriksa dataset master yang menggabungkan data harga, indikator teknikal, fitur kalender, dan fitur sentimen harian.",
    )

    with st.expander("Jalankan pembentukan dataset", expanded=False):
        action_button("Bangun dataset master TFT", "Bangun dataset master")

    df_master = filter_by_sidebar(master, selected_ticker, date_range)

    if df_master is None or df_master.empty:
        st.warning("Dataset master belum tersedia. Jalankan pembentukan dataset terlebih dahulu.")
    else:
        date_col = get_date_col(df_master) or "date"
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Jumlah Baris", format_number(len(df_master)))
        c2.metric("Jumlah Kolom", format_number(len(df_master.columns)))
        c3.metric("Jumlah Emiten", format_number(df_master["ticker"].nunique() if "ticker" in df_master.columns else 0))
        c4.metric("Missing Value", format_number(int(df_master.isna().sum().sum())))

        left, right = st.columns(2)

        with left:
            if "ticker" in df_master.columns:
                count_ticker = df_master["ticker"].value_counts().reset_index()
                count_ticker.columns = ["Ticker", "Jumlah"]
                fig = px.pie(count_ticker, names="Ticker", values="Jumlah", title="Proporsi Data per Emiten", hole=0.5)
                st.plotly_chart(plot_layout(fig), use_container_width=True)

        with right:
            feature_status = pd.DataFrame(
                {
                    "Fitur": TECHNICAL_FEATURES + SENTIMENT_FEATURES,
                    "Status": ["Tersedia" if col in df_master.columns else "Belum ada" for col in TECHNICAL_FEATURES + SENTIMENT_FEATURES],
                    "Jenis": ["Teknikal"] * len(TECHNICAL_FEATURES) + ["Sentimen"] * len(SENTIMENT_FEATURES),
                }
            )
            status_count = feature_status.groupby(["Jenis", "Status"]).size().reset_index(name="Jumlah")
            fig = px.bar(status_count, x="Jenis", y="Jumlah", color="Status", barmode="group", title="Kelengkapan Fitur Model")
            st.plotly_chart(plot_layout(fig), use_container_width=True)

        if "split" in df_master.columns:
            split_count = df_master["split"].value_counts().reset_index()
            split_count.columns = ["Split", "Jumlah"]
            fig_split = px.bar(split_count, x="Split", y="Jumlah", title="Distribusi Train, Validation, dan Test")
            st.plotly_chart(plot_layout(fig_split, 360), use_container_width=True)
        elif date_col:
            timeline_count = (
                df_master.dropna(subset=[date_col])
                .assign(month=lambda x: pd.to_datetime(x[date_col]).dt.to_period("M").astype(str))
                .groupby("month")
                .size()
                .reset_index(name="Jumlah")
            )
            fig_time = px.area(timeline_count, x="month", y="Jumlah", title="Distribusi Data Berdasarkan Waktu")
            st.plotly_chart(plot_layout(fig_time, 360), use_container_width=True)

        st.subheader("Preview Dataset Master")
        st.dataframe(df_master.tail(500), use_container_width=True, hide_index=True)


# ============================================================
# 11. HALAMAN PELATIHAN DAN PREDIKSI
# ============================================================
elif page == "Pelatihan dan Prediksi":
    render_header(
        "Pelatihan Model Prediksi",
        "Halaman ini menyediakan tombol pelatihan TFT dan LLM-TFT serta ringkasan prediksi jika file backtest atau hasil prediksi tersedia.",
    )

    with st.expander("Jalankan pelatihan dan interpretasi model", expanded=False):
        b1, b2, b3 = st.columns(3)
        with b1:
            action_button("Latih TFT", "Latih TFT")
        with b2:
            action_button("Latih LLM-TFT", "Latih LLM-TFT")
        with b3:
            action_button("Interpretasi model", "Interpretasi model")

    st.subheader("Status Model")
    baseline_ckpt = list((MODELS_DIR / "baseline").glob("**/*.ckpt")) if (MODELS_DIR / "baseline").exists() else []
    hybrid_ckpt = list((MODELS_DIR / "hybrid").glob("**/*.ckpt")) if (MODELS_DIR / "hybrid").exists() else []

    c1, c2, c3 = st.columns(3)
    c1.metric("Checkpoint TFT", format_number(len(baseline_ckpt)))
    c2.metric("Checkpoint LLM-TFT", format_number(len(hybrid_ckpt)))
    c3.metric("Dataset Master", "Tersedia" if master_path else "Belum tersedia")

    df_backtest = filter_by_sidebar(backtest, selected_ticker, date_range)

    if df_backtest is not None and not df_backtest.empty:
        st.subheader("Visualisasi Prediksi dan Backtest")
        date_col = get_date_col(df_backtest)
        y_actual = next((c for c in ["actual", "Actual", "y_true", "close", "target"] if c in df_backtest.columns), None)
        y_pred = next((c for c in ["prediction", "predicted", "y_pred", "pred", "LLM-TFT", "llm_tft"] if c in df_backtest.columns), None)

        if date_col and y_actual and y_pred:
            plot_df = df_backtest.sort_values(date_col).tail(250)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=plot_df[date_col], y=plot_df[y_actual], mode="lines", name="Aktual"))
            fig.add_trace(go.Scatter(x=plot_df[date_col], y=plot_df[y_pred], mode="lines", name="Prediksi"))
            fig.update_layout(title="Actual vs Predicted")
            st.plotly_chart(plot_layout(fig), use_container_width=True)
        else:
            st.dataframe(df_backtest.tail(500), use_container_width=True, hide_index=True)
    else:
        st.info("File backtest/prediksi belum ditemukan. Jalankan evaluasi backtest untuk menampilkan grafik Actual vs Predicted.")

    st.subheader("Konfigurasi Model yang Ditampilkan")
    st.markdown(
        """
        - **TFT** memakai data harga, indikator teknikal, dan fitur kalender.
        - **LLM-TFT** memakai data harga, indikator teknikal, fitur kalender, dan fitur sentimen harian.
        - Horizon prediksi diarahkan untuk **H+1, H+2, dan H+3**.
        - Model terbaik dapat dibandingkan melalui halaman evaluasi.
        """
    )


# ============================================================
# 12. HALAMAN EVALUASI DAN BACKTEST
# ============================================================
elif page == "Evaluasi dan Backtest":
    render_header(
        "Evaluasi dan Penyajian Hasil",
        "Halaman ini membandingkan model TFT dan LLM-TFT menggunakan MAE, RMSE, MAPE, R², Directional Accuracy, serta hasil backtest.",
    )

    with st.expander("Jalankan evaluasi", expanded=False):
        b1, b2 = st.columns(2)
        with b1:
            action_button("Evaluasi model", "Evaluasi model")
        with b2:
            action_button("Backtest model", "Backtest model")

    if eval_global is None or eval_global.empty:
        st.warning("File evaluasi global belum tersedia.")
    else:
        st.subheader("Evaluasi Global")
        clean_eval = eval_global.drop(columns=[c for c in ["n", "n_diracc", "split"] if c in eval_global.columns], errors="ignore")
        st.dataframe(clean_eval, use_container_width=True, hide_index=True)

        model_col = "model" if "model" in clean_eval.columns else "Model" if "Model" in clean_eval.columns else None
        metrics = [m for m in ["MAE", "RMSE", "MAPE", "R2", "R²", "Directional Accuracy", "DirAcc"] if m in clean_eval.columns]

        if model_col and metrics:
            metric_choice = st.selectbox("Pilih metrik evaluasi", metrics, index=metrics.index("RMSE") if "RMSE" in metrics else 0)
            fig = px.bar(clean_eval, x=model_col, y=metric_choice, title=f"Perbandingan Model Berdasarkan {metric_choice}", text_auto=True)
            st.plotly_chart(plot_layout(fig), use_container_width=True)

    v1, v2 = st.columns(2)
    with v1:
        st.subheader("Evaluasi per Emiten")
        if eval_ticker is not None and not eval_ticker.empty:
            ticker_eval = eval_ticker.drop(columns=[c for c in ["n", "n_diracc", "split"] if c in eval_ticker.columns], errors="ignore")
            st.dataframe(ticker_eval, use_container_width=True, hide_index=True)

            ticker_col = "ticker" if "ticker" in ticker_eval.columns else "Ticker" if "Ticker" in ticker_eval.columns else None
            model_col = "model" if "model" in ticker_eval.columns else "Model" if "Model" in ticker_eval.columns else None
            if ticker_col and model_col and "RMSE" in ticker_eval.columns:
                fig = px.bar(ticker_eval, x=ticker_col, y="RMSE", color=model_col, barmode="group", title="RMSE per Emiten")
                st.plotly_chart(plot_layout(fig, 420), use_container_width=True)
        else:
            st.info("File evaluasi per emiten belum tersedia.")

    with v2:
        st.subheader("Evaluasi per Horizon")
        if eval_horizon is not None and not eval_horizon.empty:
            horizon_eval = eval_horizon.drop(columns=[c for c in ["n", "n_diracc", "split"] if c in eval_horizon.columns], errors="ignore")
            st.dataframe(horizon_eval, use_container_width=True, hide_index=True)

            horizon_col = "horizon" if "horizon" in horizon_eval.columns else "Horizon" if "Horizon" in horizon_eval.columns else None
            model_col = "model" if "model" in horizon_eval.columns else "Model" if "Model" in horizon_eval.columns else None
            if horizon_col and model_col and "RMSE" in horizon_eval.columns:
                fig = px.line(horizon_eval, x=horizon_col, y="RMSE", color=model_col, markers=True, title="RMSE Berdasarkan Horizon")
                st.plotly_chart(plot_layout(fig, 420), use_container_width=True)
        else:
            st.info("File evaluasi per horizon belum tersedia.")

    st.subheader("Catatan Interpretasi")
    st.markdown(
        """
        Evaluasi utama menggunakan **MAE**, **RMSE**, **MAPE**, **R²**, dan **Directional Accuracy**.
        Nilai error yang lebih rendah menunjukkan prediksi harga yang lebih dekat dengan data aktual.
        Nilai **R²** yang lebih tinggi menunjukkan kemampuan model menjelaskan variasi harga.
        Nilai **Directional Accuracy** yang lebih tinggi menunjukkan kemampuan model membaca arah pergerakan harga.
        """
    )
