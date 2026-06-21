import subprocess
from pathlib import Path

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
    DATA_DIR / "news_clean.csv",
]

ARTICLE_SENTIMENT_PATHS = [
    PROCESSED_DIR / "news_with_sentiment_per_article.csv",
    INTERIM_DIR / "news_with_sentiment_per_article.csv",
    PROCESSED_DIR / "article_sentiment.csv",
    INTERIM_DIR / "article_sentiment.csv",
]

DAILY_SENTIMENT_PATHS = [
    PROCESSED_DIR / "daily_sentiment.csv",
    INTERIM_DIR / "daily_sentiment.csv",
    PROCESSED_DIR / "sentiment_daily.csv",
    INTERIM_DIR / "sentiment_daily.csv",
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

LLM_LABEL_COLUMNS = [
    "l_text",
    "l_llm",
    "llm_label",
    "label_llm",
    "sentiment_llm",
    "llm_sentiment",
    "gpt_label",
    "gpt_sentiment",
]

LEXICON_LABEL_COLUMNS = [
    "l_lex",
    "l_lexicon",
    "lexicon_label",
    "label_lexicon",
    "sentiment_lexicon",
    "lex_label",
    "lex_sentiment",
]

MARKET_LABEL_COLUMNS = [
    "l_market",
    "l_response",
    "l_resp",
    "market_label",
    "label_market",
    "sentiment_market",
    "market_response_label",
    "response_label",
]

FINAL_LABEL_COLUMNS = [
    "l_final",
    "final_label",
    "label_final",
    "sentiment_final",
    "sentiment",
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

    div[data-testid="stMetric"] {
        background: rgba(15, 23, 42, 0.78);
        border: 1px solid rgba(148, 163, 184, 0.18);
        padding: 15px 16px;
        border-radius: 18px;
        box-shadow: 0 10px 24px rgba(0, 0, 0, 0.22);
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
# 3. UTILITAS
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
    for col in ["date", "published_at", "publish_date", "datetime", "timestamp"]:
        if col in df.columns:
            return col
    return None


def find_col(df: pd.DataFrame | None, candidates: list[str]) -> str | None:
    if df is None:
        return None
    lower_map = {col.lower(): col for col in df.columns}
    for col in candidates:
        if col.lower() in lower_map:
            return lower_map[col.lower()]
    return None


def filter_data(df: pd.DataFrame | None, ticker: str | None, date_range: tuple | None) -> pd.DataFrame | None:
    if df is None:
        return None
    out = df.copy()

    if ticker and ticker != "Semua" and "ticker" in out.columns:
        out = out[out["ticker"] == ticker]

    date_col = get_date_col(out)
    if date_col and date_range and len(date_range) == 2:
        start = pd.to_datetime(date_range[0])
        end = pd.to_datetime(date_range[1])
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
        st.error("Proses gagal.")

    with st.expander("Log terminal"):
        if result.stdout:
            st.code(result.stdout[-8000:])
        if result.stderr:
            st.code(result.stderr[-8000:])


def action_button(label: str, command_key: str):
    if st.button(label, use_container_width=True):
        run_command(PIPELINE_COMMANDS[command_key])


def render_header(title: str):
    st.markdown(
        f"""
        <div class="main-title">
            <h1>{title}</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(title: str, value: str, note: str = ""):
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


def render_step(number: int, title: str):
    st.markdown(
        f"""
        <div class="step-card">
            <b>{number}. {title}</b>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# 4. LOAD DATA
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
            "Model, Prediksi, dan Evaluasi",
        ],
    )

    st.divider()
    st.markdown("### Filter Data")

    ticker_source = master if master is not None and "ticker" in master.columns else prices
    if ticker_source is not None and "ticker" in ticker_source.columns:
        ticker_options = ["Semua"] + sorted(ticker_source["ticker"].dropna().astype(str).unique().tolist())
    else:
        ticker_options = ["Semua"]

    selected_ticker = st.selectbox("Emiten", ticker_options)

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


# ============================================================
# 6. RINGKASAN SISTEM
# ============================================================
if page == "Ringkasan Sistem":
    render_header("Dashboard Prediksi Harga Saham Indonesia")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card("Data Harga", format_number(len(prices) if prices is not None else 0), "baris")
    with c2:
        render_metric_card("Data Berita", format_number(len(news) if news is not None else 0), "baris")
    with c3:
        render_metric_card("Sentimen Harian", format_number(len(daily_sentiment) if daily_sentiment is not None else 0), "baris")
    with c4:
        render_metric_card("Dataset Model", format_number(len(master) if master is not None else 0), "baris")

    st.subheader("Alur Sistem")
    s1, s2 = st.columns(2)
    with s1:
        render_step(1, "Pengolahan Data Harga Saham")
        render_step(2, "Pengolahan Data Berita Keuangan")
        render_step(3, "Pelabelan dan Agregasi Sentimen")
    with s2:
        render_step(4, "Pembentukan Dataset Model")
        render_step(5, "Pelatihan Model Prediksi")
        render_step(6, "Evaluasi dan Penyajian Hasil")

    component_status = pd.DataFrame(
        {
            "Komponen": ["Harga", "Berita", "Sentimen Artikel", "Sentimen Harian", "Dataset", "Evaluasi"],
            "Status": [
                "Tersedia" if prices is not None else "Belum tersedia",
                "Tersedia" if news is not None else "Belum tersedia",
                "Tersedia" if article_sentiment is not None else "Belum tersedia",
                "Tersedia" if daily_sentiment is not None else "Belum tersedia",
                "Tersedia" if master is not None else "Belum tersedia",
                "Tersedia" if eval_global is not None else "Belum tersedia",
            ],
        }
    )
    fig_status = px.pie(component_status, names="Status", title="Status Komponen Sistem", hole=0.55)
    st.plotly_chart(plot_layout(fig_status, 360), use_container_width=True)

    if eval_global is not None and not eval_global.empty:
        model_col = find_col(eval_global, ["model", "Model"])
        metric_col = find_col(eval_global, ["RMSE", "MAE", "MAPE", "Directional Accuracy", "DirAcc"])
        if model_col and metric_col:
            fig_metric = px.bar(eval_global, x=model_col, y=metric_col, title=f"Perbandingan Model Berdasarkan {metric_col}")
            st.plotly_chart(plot_layout(fig_metric, 380), use_container_width=True)


# ============================================================
# 7. DATA HARGA SAHAM
# ============================================================
elif page == "Data Harga Saham":
    render_header("Pengolahan Data Harga Saham")

    with st.expander("Jalankan Proses"):
        b1, b2, b3 = st.columns(3)
        with b1:
            action_button("Ambil Harga", "Ambil harga Yahoo Finance")
        with b2:
            action_button("Hitung Indikator", "Hitung indikator teknikal")
        with b3:
            action_button("Audit Kalender", "Audit kalender harga")

    df_price = filter_data(prices, selected_ticker, date_range)

    if df_price is None or df_price.empty:
        st.warning("Data harga belum tersedia.")
    else:
        date_col = get_date_col(df_price) or "date"
        latest = df_price.sort_values(date_col).tail(1).iloc[0]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Jumlah Baris", format_number(len(df_price)))
        c2.metric("Jumlah Emiten", format_number(df_price["ticker"].nunique() if "ticker" in df_price.columns else 0))
        c3.metric("Close Terakhir", f"Rp {format_number(latest['close'])}" if "close" in df_price.columns else "-")
        c4.metric("Volume Terakhir", format_number(latest["volume"]) if "volume" in df_price.columns else "-")

        tab1, tab2 = st.tabs(["Grafik Harga", "Indikator Teknikal"])

        with tab1:
            chart_df = df_price.sort_values(date_col)
            if "close" in chart_df.columns:
                fig_close = px.line(
                    chart_df,
                    x=date_col,
                    y="close",
                    color="ticker" if selected_ticker == "Semua" and "ticker" in chart_df.columns else None,
                    title="Harga Penutupan",
                )
                st.plotly_chart(plot_layout(fig_close), use_container_width=True)

            if "volume" in chart_df.columns:
                fig_volume = px.bar(
                    chart_df.tail(250),
                    x=date_col,
                    y="volume",
                    color="ticker" if selected_ticker == "Semua" and "ticker" in chart_df.columns else None,
                    title="Volume Transaksi",
                )
                st.plotly_chart(plot_layout(fig_volume, 360), use_container_width=True)

        with tab2:
            available_tech = [col for col in TECHNICAL_FEATURES if col in df_price.columns]
            if available_tech:
                selected_feature = st.selectbox("Indikator", available_tech, index=0)
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


# ============================================================
# 8. DATA BERITA KEUANGAN
# ============================================================
elif page == "Data Berita Keuangan":
    render_header("Pengolahan Data Berita Keuangan")

    with st.expander("Jalankan Proses"):
        b1, b2, b3, b4 = st.columns(4)
        with b1:
            action_button("Ambil RSS/Google", "Ambil berita RSS dan Google News")
        with b2:
            action_button("Ambil Yahoo", "Ambil berita Yahoo Finance")
        with b3:
            action_button("Gabung Berita", "Gabung sumber berita")
        with b4:
            action_button("Bersihkan Berita", "Bersihkan teks berita")

    df_news = filter_data(news, selected_ticker, date_range)

    if df_news is None or df_news.empty:
        st.warning("Data berita belum tersedia.")
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
                fig_source = px.bar(source_count.head(15), x="Sumber", y="Jumlah", title="Jumlah Berita per Sumber")
                st.plotly_chart(plot_layout(fig_source), use_container_width=True)
            elif "ticker" in df_news.columns:
                ticker_count = df_news["ticker"].value_counts().reset_index()
                ticker_count.columns = ["Ticker", "Jumlah"]
                fig_ticker = px.pie(ticker_count, names="Ticker", values="Jumlah", title="Proporsi Berita per Emiten", hole=0.45)
                st.plotly_chart(plot_layout(fig_ticker), use_container_width=True)

        with right:
            if date_col:
                daily_news = (
                    df_news.dropna(subset=[date_col])
                    .assign(day=lambda x: pd.to_datetime(x[date_col]).dt.date)
                    .groupby("day")
                    .size()
                    .reset_index(name="Jumlah")
                )
                fig_daily = px.area(daily_news, x="day", y="Jumlah", title="Tren Jumlah Berita Harian")
                st.plotly_chart(plot_layout(fig_daily), use_container_width=True)


# ============================================================
# 9. SENTIMEN HARIAN
# ============================================================
elif page == "Sentimen Harian":
    render_header("Pelabelan dan Agregasi Sentimen")

    with st.expander("Jalankan Proses"):
        b1, b2 = st.columns(2)
        with b1:
            action_button("Label Sentimen", "Label sentimen artikel")
        with b2:
            action_button("Agregasi Harian", "Agregasi sentimen harian")

    df_article = filter_data(article_sentiment, selected_ticker, date_range)
    df_daily = filter_data(daily_sentiment, selected_ticker, date_range)

    tab1, tab2 = st.tabs(["Sentimen Artikel", "Sentimen Harian"])

    with tab1:
        if df_article is None or df_article.empty:
            st.warning("Data sentimen artikel belum tersedia.")
        else:
            llm_col = find_col(df_article, LLM_LABEL_COLUMNS)
            lex_col = find_col(df_article, LEXICON_LABEL_COLUMNS)
            market_col = find_col(df_article, MARKET_LABEL_COLUMNS)
            final_col = find_col(df_article, FINAL_LABEL_COLUMNS)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Jumlah Artikel", format_number(len(df_article)))
            c2.metric("LLM Label", "Ada" if llm_col else "Belum terdeteksi")
            c3.metric("Leksikon", "Ada" if lex_col else "Belum terdeteksi")
            c4.metric("Respons Pasar", "Ada" if market_col else "Belum terdeteksi")

            if final_col:
                label_map = {-1: "Negatif", 0: "Netral", 1: "Positif", "-1": "Negatif", "0": "Netral", "1": "Positif"}
                sentiment_count = df_article[final_col].map(label_map).fillna(df_article[final_col].astype(str)).value_counts().reset_index()
                sentiment_count.columns = ["Sentimen", "Jumlah"]
                fig_label = px.pie(sentiment_count, names="Sentimen", values="Jumlah", title="Distribusi Sentimen Artikel", hole=0.5)
                st.plotly_chart(plot_layout(fig_label, 430), use_container_width=True)

            label_sources = []
            if llm_col:
                label_sources.append("LLM")
            if lex_col:
                label_sources.append("Leksikon")
            if market_col:
                label_sources.append("Respons Pasar")
            if label_sources:
                source_df = pd.DataFrame({"Metode": label_sources, "Jumlah": [1] * len(label_sources)})
                fig_sources = px.pie(source_df, names="Metode", values="Jumlah", title="Metode Pelabelan Terdeteksi", hole=0.45)
                st.plotly_chart(plot_layout(fig_sources, 360), use_container_width=True)

    with tab2:
        if df_daily is None or df_daily.empty:
            st.warning("Data sentimen harian belum tersedia.")
        else:
            date_col = get_date_col(df_daily) or "date"

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Baris Sentimen", format_number(len(df_daily)))
            c2.metric("Jumlah Emiten", format_number(df_daily["ticker"].nunique() if "ticker" in df_daily.columns else 0))
            c3.metric("News Count", format_number(df_daily["news_count_3d"].mean(), 2) if "news_count_3d" in df_daily.columns else "-")
            c4.metric("Rata-rata Sentimen", format_number(df_daily["sentiment_mean_3d"].mean(), 3) if "sentiment_mean_3d" in df_daily.columns else "-")

            available_sent = [col for col in SENTIMENT_FEATURES if col in df_daily.columns]
            if available_sent:
                selected_sent_feature = st.selectbox("Fitur Sentimen", available_sent)
                fig_sent = px.line(
                    df_daily.sort_values(date_col),
                    x=date_col,
                    y=selected_sent_feature,
                    color="ticker" if selected_ticker == "Semua" and "ticker" in df_daily.columns else None,
                    title=f"Tren {selected_sent_feature}",
                )
                st.plotly_chart(plot_layout(fig_sent), use_container_width=True)

                if "ticker" in df_daily.columns:
                    agg = df_daily.groupby("ticker")[selected_sent_feature].mean().reset_index()
                    fig_agg = px.bar(agg, x="ticker", y=selected_sent_feature, title=f"Rata-rata {selected_sent_feature} per Emiten")
                    st.plotly_chart(plot_layout(fig_agg, 360), use_container_width=True)


# ============================================================
# 10. MODEL, PREDIKSI, DAN EVALUASI
# ============================================================
elif page == "Model, Prediksi, dan Evaluasi":
    render_header("Model, Prediksi, dan Evaluasi")

    with st.expander("Jalankan Proses"):
        b1, b2, b3, b4, b5, b6 = st.columns(6)
        with b1:
            action_button("Dataset", "Bangun dataset master")
        with b2:
            action_button("Latih TFT", "Latih TFT")
        with b3:
            action_button("Latih LLM-TFT", "Latih LLM-TFT")
        with b4:
            action_button("Evaluasi", "Evaluasi model")
        with b5:
            action_button("Backtest", "Backtest model")
        with b6:
            action_button("Interpretasi", "Interpretasi model")

    df_master = filter_data(master, selected_ticker, date_range)
    df_backtest = filter_data(backtest, selected_ticker, date_range)

    baseline_ckpt = list((MODELS_DIR / "baseline").glob("**/*.ckpt")) if (MODELS_DIR / "baseline").exists() else []
    hybrid_ckpt = list((MODELS_DIR / "hybrid").glob("**/*.ckpt")) if (MODELS_DIR / "hybrid").exists() else []

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dataset", format_number(len(df_master) if df_master is not None else 0))
    c2.metric("Checkpoint TFT", format_number(len(baseline_ckpt)))
    c3.metric("Checkpoint LLM-TFT", format_number(len(hybrid_ckpt)))
    c4.metric("Missing Value", format_number(int(df_master.isna().sum().sum())) if df_master is not None and not df_master.empty else "-")

    if df_master is not None and not df_master.empty:
        date_col = get_date_col(df_master) or "date"
        left, right = st.columns(2)

        with left:
            if "ticker" in df_master.columns:
                count_ticker = df_master["ticker"].value_counts().reset_index()
                count_ticker.columns = ["Ticker", "Jumlah"]
                fig_ticker = px.pie(count_ticker, names="Ticker", values="Jumlah", title="Proporsi Dataset per Emiten", hole=0.5)
                st.plotly_chart(plot_layout(fig_ticker), use_container_width=True)

        with right:
            if "split" in df_master.columns:
                split_count = df_master["split"].value_counts().reset_index()
                split_count.columns = ["Split", "Jumlah"]
                fig_split = px.bar(split_count, x="Split", y="Jumlah", title="Distribusi Data Model")
                st.plotly_chart(plot_layout(fig_split), use_container_width=True)
            elif date_col:
                timeline = (
                    df_master.dropna(subset=[date_col])
                    .assign(month=lambda x: pd.to_datetime(x[date_col]).dt.to_period("M").astype(str))
                    .groupby("month")
                    .size()
                    .reset_index(name="Jumlah")
                )
                fig_timeline = px.area(timeline, x="month", y="Jumlah", title="Distribusi Dataset Berdasarkan Waktu")
                st.plotly_chart(plot_layout(fig_timeline), use_container_width=True)

    st.subheader("Evaluasi Model")
    if eval_global is not None and not eval_global.empty:
        model_col = find_col(eval_global, ["model", "Model"])
        available_metrics = [m for m in ["MAE", "RMSE", "MAPE", "R2", "R²", "Directional Accuracy", "DirAcc"] if m in eval_global.columns]

        if model_col and available_metrics:
            metric_choice = st.selectbox("Metrik", available_metrics, index=available_metrics.index("RMSE") if "RMSE" in available_metrics else 0)
            fig_eval = px.bar(eval_global, x=model_col, y=metric_choice, title=f"Perbandingan Model Berdasarkan {metric_choice}", text_auto=True)
            st.plotly_chart(plot_layout(fig_eval), use_container_width=True)

    left, right = st.columns(2)
    with left:
        if eval_ticker is not None and not eval_ticker.empty:
            ticker_col = find_col(eval_ticker, ["ticker", "Ticker"])
            model_col = find_col(eval_ticker, ["model", "Model"])
            metric_col = find_col(eval_ticker, ["RMSE", "MAE", "MAPE"])
            if ticker_col and model_col and metric_col:
                fig_ticker_eval = px.bar(
                    eval_ticker,
                    x=ticker_col,
                    y=metric_col,
                    color=model_col,
                    barmode="group",
                    title=f"{metric_col} per Emiten",
                )
                st.plotly_chart(plot_layout(fig_ticker_eval, 420), use_container_width=True)

    with right:
        if eval_horizon is not None and not eval_horizon.empty:
            horizon_col = find_col(eval_horizon, ["horizon", "Horizon"])
            model_col = find_col(eval_horizon, ["model", "Model"])
            metric_col = find_col(eval_horizon, ["RMSE", "MAE", "MAPE"])
            if horizon_col and model_col and metric_col:
                fig_horizon = px.line(
                    eval_horizon,
                    x=horizon_col,
                    y=metric_col,
                    color=model_col,
                    markers=True,
                    title=f"{metric_col} Berdasarkan Horizon",
                )
                st.plotly_chart(plot_layout(fig_horizon, 420), use_container_width=True)

    if df_backtest is not None and not df_backtest.empty:
        date_col = get_date_col(df_backtest)
        actual_col = find_col(df_backtest, ["actual", "Actual", "y_true", "close", "target"])
        pred_col = find_col(df_backtest, ["prediction", "predicted", "y_pred", "pred", "LLM-TFT", "llm_tft"])

        if date_col and actual_col and pred_col:
            plot_df = df_backtest.sort_values(date_col).tail(250)
            fig_backtest = go.Figure()
            fig_backtest.add_trace(go.Scatter(x=plot_df[date_col], y=plot_df[actual_col], mode="lines", name="Aktual"))
            fig_backtest.add_trace(go.Scatter(x=plot_df[date_col], y=plot_df[pred_col], mode="lines", name="Prediksi"))
            fig_backtest.update_layout(title="Actual vs Predicted")
            st.plotly_chart(plot_layout(fig_backtest), use_container_width=True)
