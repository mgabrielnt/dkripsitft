import subprocess
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="Dashboard Prediksi Saham Indonesia",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
REPORTSS = ROOT / "reportss"
REPORTS = ROOT / "reports"
MODELSSS = ROOT / "modelssss"
MODELS = ROOT / "models"

COLORS = ["#38BDF8", "#F97316", "#22C55E", "#A855F7", "#F43F5E", "#14B8A6", "#FACC15"]
MODEL_COLORS = {"TFT": "#38BDF8", "LLM-TFT": "#F97316", "Baseline": "#38BDF8", "Hybrid": "#F97316"}

PRICE_PATHS = [INTERIM / "prices_with_indicators.csv", PROCESSED / "prices_with_indicators.csv", DATA / "prices_with_indicators.csv"]
NEWS_PATHS = [INTERIM / "news_clean.csv", PROCESSED / "news_clean.csv", DATA / "news_clean.csv"]
ARTICLE_PATHS = [
    PROCESSED / "news_with_sentiment_per_article.csv",
    INTERIM / "news_with_sentiment_per_article.csv",
    PROCESSED / "article_sentiment.csv",
    INTERIM / "article_sentiment.csv",
    PROCESSED / "sentiment_articles.csv",
    INTERIM / "sentiment_articles.csv",
]
DAILY_SENTIMENT_PATHS = [
    PROCESSED / "daily_sentiment.csv",
    INTERIM / "daily_sentiment.csv",
    PROCESSED / "sentiment_daily.csv",
    INTERIM / "sentiment_daily.csv",
]
MASTER_PATHS = [PROCESSED / "tft_master.csv", INTERIM / "tft_master.csv"]
EVAL_GLOBAL_PATHS = [
    REPORTSS / "eval_metrics_global.csv",
    REPORTS / "eval_metrics_global.csv",
    REPORTSS / "evaluation_metrics_global.csv",
    PROCESSED / "eval_metrics_global.csv",
]
EVAL_TICKER_PATHS = [
    REPORTSS / "eval_metrics_by_ticker_global.csv",
    REPORTS / "eval_metrics_by_ticker_global.csv",
    REPORTSS / "eval_metrics_by_ticker.csv",
    REPORTS / "eval_metrics_by_ticker.csv",
    PROCESSED / "eval_metrics_by_ticker_global.csv",
]
EVAL_HORIZON_PATHS = [REPORTSS / "eval_metrics_by_horizon.csv", REPORTS / "eval_metrics_by_horizon.csv", PROCESSED / "eval_metrics_by_horizon.csv"]
PREDICTION_PATHS = [
    REPORTSS / "backtest_predictions.csv",
    REPORTS / "backtest_predictions.csv",
    REPORTSS / "test_predictions.csv",
    REPORTS / "test_predictions.csv",
    REPORTSS / "predictions.csv",
    REPORTS / "predictions.csv",
    REPORTSS / "backtest.csv",
    REPORTS / "backtest.csv",
    REPORTSS / "backtest_results.csv",
    REPORTS / "backtest_results.csv",
    REPORTSS / "sample_forecast.csv",
    REPORTS / "sample_forecast.csv",
    REPORTSS / "sample_forecast_llm_tft.csv",
    REPORTS / "sample_forecast_llm_tft.csv",
    PROCESSED / "backtest_predictions.csv",
    PROCESSED / "predictions.csv",
]

PIPELINE = {
    "Ambil harga": "python -m src.data.download_prices_yahoo",
    "Hitung indikator": "python -m src.data.compute_technical_indicators",
    "Audit kalender": "python -m src.data.check_price_calendar",
    "RSS Google": "python -m src.data.fetch_news_rss_google",
    "Yahoo News": "python -m src.data.fetch_news_yahoo",
    "Gabung berita": "python -m src.data.merge_news_sources",
    "Bersihkan berita": "python -m src.data.preprocess_news_text",
    "Label sentimen": "python -m src.data.gpt_sentiment_labeling",
    "Agregasi sentimen": "python -m src.data.aggregate_daily_sentiment",
    "Dataset": "python -m src.data.build_tft_master_dataset",
    "Latih TFT": "python -m src.models.train_tft_baseline",
    "Latih LLM-TFT": "python -m src.models.train_tft_with_sentiment",
    "Evaluasi": "python -m src.models.evaluate_tft_models",
    "Backtest": "python -m src.models.evaluate_tft_backtest",
    "Interpretasi": "python -m src.models.interpret_tft_models",
}

TECHNICAL_FEATURES = [
    "close", "volume", "log_return_1d", "log_return_2d", "vol_20", "rsi_14",
    "ma_5_div_ma_20", "bb_width_20", "gap_return_1d", "intraday_range_pct",
]
SENTIMENT_FEATURES = ["news_count_3d", "sentiment_mean_3d", "sentiment_ema_7d", "sentiment_trend_7d", "sentiment_delta_1d", "sentiment_dir_signal"]
FINAL_LABELS = ["l_final", "final_label", "label_final", "sentiment_final", "sentiment", "sentiment_label"]


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp {
        background:
        radial-gradient(circle at top left, rgba(56,189,248,0.18), transparent 34%),
        radial-gradient(circle at top right, rgba(249,115,22,0.16), transparent 30%),
        linear-gradient(135deg, #050B16 0%, #0F172A 48%, #111827 100%);
        color: #e5e7eb;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #050B16 0%, #0B1220 100%);
        border-right: 1px solid rgba(148,163,184,0.18);
    }
    .main-title {
        padding: 24px 28px; border-radius: 26px;
        background:
        radial-gradient(circle at 10% 0%, rgba(56,189,248,0.34), transparent 34%),
        radial-gradient(circle at 95% 20%, rgba(249,115,22,0.22), transparent 30%),
        linear-gradient(135deg, rgba(15,23,42,0.98), rgba(30,41,59,0.72));
        border: 1px solid rgba(148,163,184,0.25);
        box-shadow: 0 22px 60px rgba(0,0,0,0.30);
        margin-bottom: 22px;
    }
    .main-title h1 { margin: 0; font-size: 2.05rem; letter-spacing: -0.04em; color: #f8fafc; }
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, rgba(15,23,42,0.90), rgba(30,41,59,0.72));
        border: 1px solid rgba(148,163,184,0.18);
        padding: 15px 16px; border-radius: 18px; box-shadow: 0 10px 24px rgba(0,0,0,0.22);
    }
    .block-container { padding-top: 1.4rem; padding-bottom: 2.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def first_existing(paths):
    for path in paths:
        if path.exists():
            return path
    return None


def first_prediction_file():
    direct = first_existing(PREDICTION_PATHS)
    if direct:
        return direct
    folders = [REPORTSS, REPORTS, PROCESSED, INTERIM]
    keywords = ["pred", "forecast", "backtest"]
    for folder in folders:
        if folder.exists():
            for file in folder.glob("*.csv"):
                if any(key in file.name.lower() for key in keywords):
                    return file
    return None


@st.cache_data(show_spinner=False)
def read_csv(path):
    if path is None or not Path(path).exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    for col in df.columns:
        if col.lower() in {"date", "target_date", "prediction_date", "published_at", "publish_date", "datetime", "timestamp"}:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    if "ticker" in df.columns:
        df["ticker"] = df["ticker"].astype(str)
    return df


def fmt(value, decimals=0):
    if value is None or pd.isna(value):
        return "-"
    try:
        if decimals == 0:
            return f"{float(value):,.0f}".replace(",", ".")
        return f"{float(value):,.{decimals}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return str(value)


def date_col(df):
    if df is None:
        return None
    for col in ["date", "target_date", "prediction_date", "published_at", "publish_date", "datetime", "timestamp"]:
        if col in df.columns:
            return col
    return None


def find_col(df, candidates):
    if df is None:
        return None
    lower_map = {col.lower(): col for col in df.columns}
    for col in candidates:
        if col.lower() in lower_map:
            return lower_map[col.lower()]
    return None


def find_contains_col(df, includes, excludes=None):
    if df is None:
        return None
    excludes = excludes or []
    for col in df.columns:
        low = col.lower()
        if all(x in low for x in includes) and not any(x in low for x in excludes):
            return col
    return None


def filter_df(df, ticker, dates):
    if df is None:
        return None
    out = df.copy()
    if ticker != "Semua" and "ticker" in out.columns:
        out = out[out["ticker"] == ticker]
    dc = date_col(out)
    if dc and dates and len(dates) == 2:
        start, end = pd.to_datetime(dates[0]), pd.to_datetime(dates[1])
        out = out[(out[dc] >= start) & (out[dc] <= end)]
    return out


def normalize_model(value):
    low = str(value).lower()
    if "llm" in low or "hybrid" in low or "sent" in low:
        return "LLM-TFT"
    if "tft" in low or "base" in low:
        return "TFT"
    return str(value)


def layout(fig, height=420):
    fig.update_layout(
        template="plotly_dark",
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.38)",
        margin=dict(l=18, r=18, t=48, b=24),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font=dict(color="#e5e7eb"),
        colorway=COLORS,
    )
    fig.update_xaxes(gridcolor="rgba(148,163,184,0.14)", zerolinecolor="rgba(148,163,184,0.14)")
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.14)", zerolinecolor="rgba(148,163,184,0.14)")
    return fig


def run_command(command):
    with st.spinner(f"Menjalankan: {command}"):
        try:
            result = subprocess.run(command, shell=True, cwd=ROOT, capture_output=True, text=True)
        except Exception as exc:
            st.error(f"Gagal menjalankan perintah: {exc}")
            return
    st.success("Proses selesai.") if result.returncode == 0 else st.error("Proses gagal.")
    with st.expander("Log terminal"):
        if result.stdout:
            st.code(result.stdout[-8000:])
        if result.stderr:
            st.code(result.stderr[-8000:])


def action_button(label, key):
    if st.button(label, use_container_width=True):
        run_command(PIPELINE[key])


def header(title):
    st.markdown(f"<div class='main-title'><h1>{title}</h1></div>", unsafe_allow_html=True)


def extract_predictions(pred_df, master_df):
    result = {"H+1": None, "H+2": None, "H+3": None}
    source = "file"

    if pred_df is not None and not pred_df.empty:
        work = pred_df.copy()
        dc = date_col(work)
        if dc:
            work = work.sort_values(dc)

        model_col = find_col(work, ["model", "Model", "model_name"])
        if model_col:
            work["_model_norm"] = work[model_col].apply(normalize_model)
            llm = work[work["_model_norm"] == "LLM-TFT"]
            if not llm.empty:
                work = llm

        wide_cols = {
            "H+1": ["pred_h1", "prediction_h1", "h1_pred", "horizon_1", "pred_1", "y_pred_h1", "H+1", "h+1"],
            "H+2": ["pred_h2", "prediction_h2", "h2_pred", "horizon_2", "pred_2", "y_pred_h2", "H+2", "h+2"],
            "H+3": ["pred_h3", "prediction_h3", "h3_pred", "horizon_3", "pred_3", "y_pred_h3", "H+3", "h+3"],
        }
        for horizon, cols in wide_cols.items():
            col = find_col(work, cols)
            if col and pd.api.types.is_numeric_dtype(work[col]):
                val = work[col].dropna().tail(1)
                if not val.empty:
                    result[horizon] = float(val.iloc[-1])

        if any(v is not None for v in result.values()):
            return result, source

        horizon_col = find_col(work, ["horizon", "Horizon", "step", "forecast_horizon"])
        pred_col = find_col(work, ["prediction", "predicted", "y_pred", "pred", "forecast", "forecast_value", "pred_close", "prediction_close"])
        if pred_col is None:
            pred_col = find_contains_col(work, ["pred"], ["error", "direction"])

        if horizon_col and pred_col:
            tmp = work[[horizon_col, pred_col]].dropna().copy()
            tmp[horizon_col] = tmp[horizon_col].astype(str).str.replace("H+", "", regex=False).str.replace("h+", "", regex=False)
            tmp[horizon_col] = pd.to_numeric(tmp[horizon_col], errors="coerce")
            tmp = tmp.dropna(subset=[horizon_col])
            for i in [1, 2, 3]:
                val = tmp[tmp[horizon_col] == i][pred_col].dropna().tail(1)
                if not val.empty:
                    result[f"H+{i}"] = float(val.iloc[-1])
            if any(v is not None for v in result.values()):
                return result, source

    if master_df is not None and not master_df.empty and "close" in master_df.columns:
        work = master_df.copy()
        dc = date_col(work)
        if dc:
            work = work.sort_values(dc)
        close = work["close"].dropna()
        if len(close) > 0:
            last_close = float(close.iloc[-1])
            if "log_return_1d" in work.columns:
                recent_return = pd.to_numeric(work["log_return_1d"], errors="coerce").dropna().tail(15).mean()
            else:
                recent_return = close.pct_change().dropna().tail(15).mean()
            if pd.isna(recent_return):
                recent_return = 0
            for i in [1, 2, 3]:
                result[f"H+{i}"] = last_close * ((1 + float(recent_return)) ** i)
            source = "estimasi"

    return result, source


price_path = first_existing(PRICE_PATHS)
news_path = first_existing(NEWS_PATHS)
article_path = first_existing(ARTICLE_PATHS)
daily_path = first_existing(DAILY_SENTIMENT_PATHS)
master_path = first_existing(MASTER_PATHS)
eval_global_path = first_existing(EVAL_GLOBAL_PATHS)
eval_ticker_path = first_existing(EVAL_TICKER_PATHS)
eval_horizon_path = first_existing(EVAL_HORIZON_PATHS)
prediction_path = first_prediction_file()

prices = read_csv(price_path)
news = read_csv(news_path)
articles = read_csv(article_path)
daily_sentiment = read_csv(daily_path)
master = read_csv(master_path)
eval_global = read_csv(eval_global_path)
eval_ticker = read_csv(eval_ticker_path)
eval_horizon = read_csv(eval_horizon_path)
predictions = read_csv(prediction_path)


with st.sidebar:
    st.markdown("## 📈 Prediksi Saham")
    st.caption("Dashboard TFT dan LLM-TFT")
    st.divider()
    page = st.radio("Menu Dashboard", ["Data Harga Saham", "Berita Keuangan Sentimen", "Model, Prediksi, dan Evaluasi"])
    st.divider()
    st.markdown("### Filter Data")

    ticker_source = master if master is not None and "ticker" in master.columns else prices
    ticker_options = ["Semua"]
    if ticker_source is not None and "ticker" in ticker_source.columns:
        ticker_options += sorted(ticker_source["ticker"].dropna().astype(str).unique().tolist())
    selected_ticker = st.selectbox("Emiten", ticker_options)

    date_source = master if master is not None and date_col(master) else prices
    selected_dates = None
    dc_sidebar = date_col(date_source)
    if date_source is not None and dc_sidebar:
        min_date = pd.to_datetime(date_source[dc_sidebar]).min()
        max_date = pd.to_datetime(date_source[dc_sidebar]).max()
        if pd.notna(min_date) and pd.notna(max_date):
            selected_dates = st.date_input(
                "Rentang Tanggal",
                value=(min_date.date(), max_date.date()),
                min_value=min_date.date(),
                max_value=max_date.date(),
            )


if page == "Data Harga Saham":
    header("Pengolahan Data Harga Saham")

    with st.expander("Jalankan Proses"):
        c1, c2, c3 = st.columns(3)
        with c1:
            action_button("Ambil Harga", "Ambil harga")
        with c2:
            action_button("Hitung Indikator", "Hitung indikator")
        with c3:
            action_button("Audit Kalender", "Audit kalender")

    df_price = filter_df(prices, selected_ticker, selected_dates)
    if df_price is None or df_price.empty:
        st.warning("Data harga belum tersedia.")
    else:
        dc = date_col(df_price) or "date"
        latest = df_price.sort_values(dc).tail(1).iloc[0]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Jumlah Baris", fmt(len(df_price)))
        k2.metric("Jumlah Emiten", fmt(df_price["ticker"].nunique() if "ticker" in df_price.columns else 0))
        k3.metric("Close Terakhir", f"Rp {fmt(latest['close'])}" if "close" in df_price.columns else "-")
        k4.metric("Volume Terakhir", fmt(latest["volume"]) if "volume" in df_price.columns else "-")

        left, right = st.columns([1.25, 1])
        with left:
            if "close" in df_price.columns:
                fig = px.line(df_price.sort_values(dc), x=dc, y="close", color="ticker" if selected_ticker == "Semua" and "ticker" in df_price.columns else None, title="Harga Penutupan", color_discrete_sequence=COLORS)
                fig.update_traces(line=dict(width=3))
                st.plotly_chart(layout(fig, 430), use_container_width=True)
        with right:
            if "volume" in df_price.columns:
                fig = px.bar(df_price.sort_values(dc).tail(250), x=dc, y="volume", color="ticker" if selected_ticker == "Semua" and "ticker" in df_price.columns else None, title="Volume Transaksi", color_discrete_sequence=COLORS)
                st.plotly_chart(layout(fig, 430), use_container_width=True)

        available = [col for col in TECHNICAL_FEATURES if col in df_price.columns]
        if available:
            selected_feature = st.selectbox("Indikator Teknikal", available)
            fig = px.line(df_price.sort_values(dc), x=dc, y=selected_feature, color="ticker" if selected_ticker == "Semua" and "ticker" in df_price.columns else None, title=f"Tren {selected_feature}", color_discrete_sequence=COLORS)
            fig.update_traces(line=dict(width=3))
            st.plotly_chart(layout(fig, 390), use_container_width=True)

            corr_cols = [col for col in available if pd.api.types.is_numeric_dtype(df_price[col])]
            if len(corr_cols) >= 2:
                fig = px.imshow(df_price[corr_cols].corr(numeric_only=True), text_auto=".2f", title="Korelasi Indikator Teknikal", color_continuous_scale="Turbo")
                st.plotly_chart(layout(fig, 520), use_container_width=True)


elif page == "Berita Keuangan Sentimen":
    header("Berita Keuangan Sentimen")

    with st.expander("Jalankan Proses"):
        b1, b2, b3, b4, b5, b6 = st.columns(6)
        with b1:
            action_button("RSS/Google", "RSS Google")
        with b2:
            action_button("Yahoo", "Yahoo News")
        with b3:
            action_button("Gabung", "Gabung berita")
        with b4:
            action_button("Bersihkan", "Bersihkan berita")
        with b5:
            action_button("Label", "Label sentimen")
        with b6:
            action_button("Agregasi", "Agregasi sentimen")

    df_news = filter_df(news, selected_ticker, selected_dates)
    df_articles = filter_df(articles, selected_ticker, selected_dates)
    df_daily = filter_df(daily_sentiment, selected_ticker, selected_dates)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Jumlah Berita", fmt(len(df_news) if df_news is not None else 0))
    k2.metric("Sentimen Artikel", fmt(len(df_articles) if df_articles is not None else 0))
    k3.metric("Sentimen Harian", fmt(len(df_daily) if df_daily is not None else 0))
    k4.metric("Jumlah Emiten", fmt(df_daily["ticker"].nunique() if df_daily is not None and "ticker" in df_daily.columns else 0))

    left, right = st.columns(2)
    with left:
        if df_news is not None and not df_news.empty:
            if "source" in df_news.columns:
                source_count = df_news["source"].fillna("Tidak diketahui").value_counts().reset_index()
                source_count.columns = ["Sumber", "Jumlah"]
                fig = px.bar(source_count.head(15), x="Sumber", y="Jumlah", title="Jumlah Berita per Sumber", color="Jumlah", color_continuous_scale="Bluered")
                st.plotly_chart(layout(fig), use_container_width=True)
            elif "ticker" in df_news.columns:
                ticker_count = df_news["ticker"].value_counts().reset_index()
                ticker_count.columns = ["Ticker", "Jumlah"]
                fig = px.pie(ticker_count, names="Ticker", values="Jumlah", title="Proporsi Berita per Emiten", hole=0.45, color_discrete_sequence=COLORS)
                st.plotly_chart(layout(fig), use_container_width=True)

    with right:
        if df_news is not None and not df_news.empty:
            dc = date_col(df_news)
            if dc:
                daily_count = df_news.dropna(subset=[dc]).assign(day=lambda x: pd.to_datetime(x[dc]).dt.date).groupby("day").size().reset_index(name="Jumlah")
                fig = px.area(daily_count, x="day", y="Jumlah", title="Tren Jumlah Berita Harian", color_discrete_sequence=["#38BDF8"])
                fig.update_traces(line=dict(width=3))
                st.plotly_chart(layout(fig), use_container_width=True)

    s1, s2, s3 = st.columns(3)
    has_sentiment = (df_articles is not None and not df_articles.empty) or (df_daily is not None and not df_daily.empty)
    s1.metric("LLM Label", "Ada" if has_sentiment else "-")
    s2.metric("Leksikon", "Ada" if has_sentiment else "-")
    s3.metric("Respons Pasar", "Ada" if has_sentiment else "-")

    left, right = st.columns(2)
    with left:
        final_col = find_col(df_articles, FINAL_LABELS)
        if df_articles is not None and not df_articles.empty and final_col:
            label_map = {-1: "Negatif", 0: "Netral", 1: "Positif", "-1": "Negatif", "0": "Netral", "1": "Positif"}
            count = df_articles[final_col].map(label_map).fillna(df_articles[final_col].astype(str)).value_counts().reset_index()
            count.columns = ["Sentimen", "Jumlah"]
            fig = px.pie(count, names="Sentimen", values="Jumlah", title="Distribusi Sentimen Artikel", hole=0.5, color_discrete_sequence=["#22C55E", "#FACC15", "#F43F5E", "#38BDF8"])
        else:
            method_df = pd.DataFrame({"Metode": ["LLM", "Leksikon", "Respons Pasar"], "Jumlah": [1, 1, 1]})
            fig = px.pie(method_df, names="Metode", values="Jumlah", title="Metode Pelabelan Sentimen", hole=0.45, color_discrete_sequence=["#38BDF8", "#A855F7", "#F97316"])
        st.plotly_chart(layout(fig, 430), use_container_width=True)

    with right:
        if df_daily is not None and not df_daily.empty:
            dc = date_col(df_daily) or "date"
            available = [col for col in SENTIMENT_FEATURES if col in df_daily.columns]
            if available:
                selected_feature = st.selectbox("Fitur Sentimen", available)
                fig = px.line(df_daily.sort_values(dc), x=dc, y=selected_feature, color="ticker" if selected_ticker == "Semua" and "ticker" in df_daily.columns else None, title=f"Tren {selected_feature}", color_discrete_sequence=COLORS)
                fig.update_traces(line=dict(width=3))
                st.plotly_chart(layout(fig), use_container_width=True)
                if "ticker" in df_daily.columns:
                    avg = df_daily.groupby("ticker")[selected_feature].mean().reset_index()
                    fig = px.bar(avg, x="ticker", y=selected_feature, title=f"Rata-rata {selected_feature} per Emiten", color=selected_feature, color_continuous_scale="Sunsetdark")
                    st.plotly_chart(layout(fig, 360), use_container_width=True)


elif page == "Model, Prediksi, dan Evaluasi":
    header("Model, Prediksi, dan Evaluasi")

    with st.expander("Jalankan Proses"):
        b1, b2, b3, b4, b5, b6 = st.columns(6)
        with b1:
            action_button("Dataset", "Dataset")
        with b2:
            action_button("Latih TFT", "Latih TFT")
        with b3:
            action_button("Latih LLM-TFT", "Latih LLM-TFT")
        with b4:
            action_button("Evaluasi", "Evaluasi")
        with b5:
            action_button("Backtest", "Backtest")
        with b6:
            action_button("Interpretasi", "Interpretasi")

    df_master = filter_df(master, selected_ticker, selected_dates)
    df_pred = filter_df(predictions, selected_ticker, selected_dates)

    baseline_ckpt = []
    hybrid_ckpt = []
    for base in [MODELSSS, MODELS]:
        if (base / "baseline").exists():
            baseline_ckpt.extend((base / "baseline").glob("**/*.ckpt"))
        for folder in ["hybrid", "llm_tft", "sentiment"]:
            if (base / folder).exists():
                hybrid_ckpt.extend((base / folder).glob("**/*.ckpt"))

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Dataset", fmt(len(df_master) if df_master is not None else 0))
    k2.metric("Checkpoint TFT", fmt(len(baseline_ckpt)))
    k3.metric("Checkpoint LLM-TFT", fmt(len(hybrid_ckpt)))
    k4.metric("Missing Value", fmt(int(df_master.isna().sum().sum())) if df_master is not None and not df_master.empty else "-")

    st.subheader("Hasil Prediksi H+1 sampai H+3")
    pred_values, pred_source = extract_predictions(df_pred, df_master)

    latest_close = None
    if df_master is not None and not df_master.empty and "close" in df_master.columns:
        dc = date_col(df_master) or "date"
        close_series = df_master.sort_values(dc)["close"].dropna()
        if not close_series.empty:
            latest_close = float(close_series.iloc[-1])

    c1, c2, c3 = st.columns(3)
    for box, horizon in zip([c1, c2, c3], ["H+1", "H+2", "H+3"]):
        value = pred_values.get(horizon)
        delta = f"{((value - latest_close) / latest_close) * 100:+.2f}%" if value is not None and latest_close else None
        box.metric(horizon, f"Rp {fmt(value)}" if value is not None else "-", delta)

    if pred_source == "estimasi":
        st.caption("Nilai H+1 sampai H+3 memakai estimasi sementara dari tren harga terakhir karena file prediksi resmi belum ditemukan.")

    if df_master is not None and not df_master.empty:
        dc = date_col(df_master) or "date"
        left, right = st.columns(2)
        with left:
            if "ticker" in df_master.columns:
                count = df_master["ticker"].value_counts().reset_index()
                count.columns = ["Ticker", "Jumlah"]
                fig = px.pie(count, names="Ticker", values="Jumlah", title="Proporsi Dataset per Emiten", hole=0.5, color_discrete_sequence=COLORS)
                st.plotly_chart(layout(fig), use_container_width=True)
        with right:
            if "split" in df_master.columns:
                count = df_master["split"].value_counts().reset_index()
                count.columns = ["Split", "Jumlah"]
                fig = px.bar(count, x="Split", y="Jumlah", title="Distribusi Data Model", color="Split", color_discrete_sequence=COLORS, text_auto=True)
                st.plotly_chart(layout(fig), use_container_width=True)
            else:
                timeline = df_master.dropna(subset=[dc]).assign(month=lambda x: pd.to_datetime(x[dc]).dt.to_period("M").astype(str)).groupby("month").size().reset_index(name="Jumlah")
                fig = px.area(timeline, x="month", y="Jumlah", title="Distribusi Dataset Berdasarkan Waktu", color_discrete_sequence=["#22C55E"])
                fig.update_traces(line=dict(width=3))
                st.plotly_chart(layout(fig), use_container_width=True)

    st.subheader("Evaluasi Model")
    if eval_global is not None and not eval_global.empty:
        plot_df = eval_global.copy()
        model_col = find_col(plot_df, ["model", "Model"])
        metrics = [m for m in ["MAE", "RMSE", "MAPE", "R2", "R²", "Directional Accuracy", "DirAcc"] if m in plot_df.columns]
        if model_col and metrics:
            plot_df[model_col] = plot_df[model_col].apply(normalize_model)
            metric = st.selectbox("Metrik", metrics, index=metrics.index("RMSE") if "RMSE" in metrics else 0)
            fig = px.bar(plot_df, x=model_col, y=metric, color=model_col, title=f"Perbandingan Model Berdasarkan {metric}", color_discrete_map=MODEL_COLORS, text_auto=True)
            st.plotly_chart(layout(fig), use_container_width=True)

    left, right = st.columns(2)
    with left:
        if eval_ticker is not None and not eval_ticker.empty:
            plot_df = eval_ticker.copy()
            ticker_col = find_col(plot_df, ["ticker", "Ticker"])
            model_col = find_col(plot_df, ["model", "Model"])
            metric_col = find_col(plot_df, ["RMSE", "MAE", "MAPE"])
            if ticker_col and model_col and metric_col:
                plot_df[model_col] = plot_df[model_col].apply(normalize_model)
                fig = px.bar(plot_df, x=ticker_col, y=metric_col, color=model_col, barmode="group", title=f"{metric_col} per Emiten", color_discrete_map=MODEL_COLORS, text_auto=True)
                st.plotly_chart(layout(fig, 420), use_container_width=True)

    with right:
        if eval_horizon is not None and not eval_horizon.empty:
            plot_df = eval_horizon.copy()
            horizon_col = find_col(plot_df, ["horizon", "Horizon"])
            model_col = find_col(plot_df, ["model", "Model"])
            metric_col = find_col(plot_df, ["RMSE", "MAE", "MAPE"])
            if horizon_col and model_col and metric_col:
                plot_df[model_col] = plot_df[model_col].apply(normalize_model)
                fig = px.line(plot_df, x=horizon_col, y=metric_col, color=model_col, markers=True, title=f"{metric_col} Berdasarkan Horizon", color_discrete_map=MODEL_COLORS)
                fig.update_traces(line=dict(width=3), marker=dict(size=9))
                st.plotly_chart(layout(fig, 420), use_container_width=True)

    if df_pred is not None and not df_pred.empty:
        dc = date_col(df_pred)
        actual_col = find_col(df_pred, ["actual", "Actual", "y_true", "close", "target"])
        pred_col = find_col(df_pred, ["prediction", "predicted", "y_pred", "pred", "forecast", "pred_close"])
        if pred_col is None:
            pred_col = find_contains_col(df_pred, ["pred"], ["error", "direction"])
        if dc and actual_col and pred_col:
            plot_df = df_pred.sort_values(dc).tail(250)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=plot_df[dc], y=plot_df[actual_col], mode="lines", name="Aktual", line=dict(color="#E5E7EB", width=3)))
            fig.add_trace(go.Scatter(x=plot_df[dc], y=plot_df[pred_col], mode="lines", name="Prediksi", line=dict(color="#22C55E", width=3)))
            fig.update_layout(title="Actual vs Predicted")
            st.plotly_chart(layout(fig), use_container_width=True)
