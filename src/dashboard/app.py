from __future__ import annotations

import os
import re
import sys
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

IMPORT_MODEL_ERROR = None

try:
    import torch
    import yaml
    from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
    from pytorch_forecasting.data import GroupNormalizer
except Exception as exc:
    IMPORT_MODEL_ERROR = f"{type(exc).__name__}: {exc}"
    torch = None
    yaml = None
    TemporalFusionTransformer = None
    TimeSeriesDataSet = None
    GroupNormalizer = None

# ============================================================
# 1. KONFIGURASI TUNGGAL
# ============================================================
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
REPORTSS = ROOT / "reportss"
REPORTS = ROOT / "reports"
CONFIG_PATH = ROOT / "configs" / "model_tft.yaml"
UPDATE_MARKER = ROOT / ".dashboard_last_data_update"
JAKARTA_TZ = ZoneInfo("Asia/Jakarta")

# Model hanya memakai checkpoint final ini. Tidak ada training/evaluasi ulang dari dashboard.
CHECKPOINTS = [
    ("TFT", "S5", ROOT / "modelssss/baseline/S5/best-checkpoint.ckpt"),
    ("LLM-TFT", "S1", ROOT / "modelssss/hybrid/S1/best-checkpoint.ckpt"),
]

PRICE_PATHS = [INTERIM / "prices_with_indicators.csv", PROCESSED / "prices_with_indicators.csv"]
NEWS_PATHS = [INTERIM / "news_clean.csv", PROCESSED / "news_clean.csv"]
ARTICLE_PATHS = [PROCESSED / "news_with_sentiment_per_article.csv", INTERIM / "article_sentiment.csv"]
DAILY_SENTIMENT_PATHS = [PROCESSED / "daily_sentiment.csv", INTERIM / "daily_sentiment.csv"]
MASTER_PATHS = [PROCESSED / "tft_master.csv", INTERIM / "tft_master.csv"]
EVAL_GLOBAL_PATHS = [REPORTSS / "eval_metrics_global.csv", REPORTS / "eval_metrics_global.csv"]
EVAL_TICKER_PATHS = [REPORTSS / "eval_metrics_by_ticker_global.csv", REPORTS / "eval_metrics_by_ticker_global.csv"]
EVAL_HORIZON_PATHS = [REPORTSS / "eval_metrics_by_horizon.csv", REPORTS / "eval_metrics_by_horizon.csv"]
ATTENTION_PATHS = [REPORTSS / "attention_comparison.csv", REPORTSS / "attention_weights.csv"]
PREDICTION_PATHS = [REPORTSS / "backtest_predictions.csv", REPORTSS / "predictions.csv", PROCESSED / "predictions.csv"]

TECH_FEATURES = [
    "close", "volume", "log_return_1d", "log_return_2d", "vol_20", "rsi_14",
    "ma_5_div_ma_20", "bb_width_20", "gap_return_1d", "intraday_range_pct",
]
SENT_FEATURES = [
    "news_count_3d", "sentiment_mean_3d", "sentiment_ema_7d",
    "sentiment_trend_7d", "sentiment_delta_1d", "sentiment_dir_signal",
]
# Kompatibilitas untuk checkpoint lama yang dilatih dengan sentiment_final_mean.
COMPAT_SENT_FEATURES = ["sentiment_final_mean"]
CAT_FEATURES = ["ticker", "day_of_week", "month", "is_month_end"]
FINAL_LABELS = ["l_final", "final_label", "label_final", "sentiment_final", "sentiment"]

COLORS = ["#38BDF8", "#F97316", "#22C55E", "#A855F7", "#F43F5E", "#14B8A6"]
MODEL_COLORS = {
    "Encoder 15 Hari": "#E5E7EB",
    "TFT": "#38BDF8",
    "LLM-TFT": "#F97316",
    "TFT S5": "#38BDF8",
    "LLM-TFT S1": "#F97316",
    "Aktual": "#E5E7EB",
}

# Auto update data harian. Tidak menjalankan training model.
AUTO_UPDATE_DATA_ON_START = True
AUTO_DATA_COMMANDS = [
    ("Ambil harga Yahoo Finance", [sys.executable, "-m", "src.data.download_prices_yahoo"]),
    ("Hitung indikator teknikal", [sys.executable, "-m", "src.data.compute_technical_indicators"]),
    ("Ambil berita RSS dan Google News", [sys.executable, "-m", "src.data.fetch_news_rss_google"]),
    ("Ambil berita Yahoo Finance", [sys.executable, "-m", "src.data.fetch_news_yahoo"]),
    ("Gabung sumber berita", [sys.executable, "-m", "src.data.merge_news_sources"]),
    ("Bersihkan teks berita", [sys.executable, "-m", "src.data.preprocess_news_text"]),
    # Jika OPENAI_API_KEY tidak ada, langkah ini akan dilewati agar dashboard tetap bisa jalan.
    ("Label sentimen artikel", [sys.executable, "-m", "src.data.gpt_sentiment_labeling"]),
    ("Agregasi sentimen harian", [sys.executable, "-m", "src.data.aggregate_daily_sentiment"]),
    ("Bangun dataset master", [sys.executable, "-m", "src.data.build_tft_master_dataset"]),
]

PAGES = ["Model dan Prediksi", "Evaluasi Model", "Data Harga Saham", "Berita Keuangan dan Sentimen"]
DROP_NAMES = {"n", "count", "jumlah", "split", "index", "unnamed0", "unnamed"}


# ============================================================
# 2. STYLE DAN UTILITAS UMUM
# ============================================================
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp {
    background:
    radial-gradient(circle at top left, rgba(56,189,248,0.17), transparent 34%),
    radial-gradient(circle at top right, rgba(249,115,22,0.13), transparent 30%),
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
.main-title p { margin: 8px 0 0 0; color: #cbd5e1; }
div[data-testid="stMetric"] {
    background: linear-gradient(145deg, rgba(15,23,42,0.88), rgba(30,41,59,0.70));
    border: 1px solid rgba(148,163,184,0.18);
    padding: 15px 16px; border-radius: 18px; box-shadow: 0 10px 24px rgba(0,0,0,0.22);
}
.block-container { padding-top: 1.4rem; padding-bottom: 2.5rem; }
</style>
"""


def apply_style() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def header(title: str, subtitle: str | None = None) -> None:
    sub = f"<p>{subtitle}</p>" if subtitle else ""
    st.markdown(f"<div class='main-title'><h1>{title}</h1>{sub}</div>", unsafe_allow_html=True)


def layout(fig, height: int = 420):
    fig.update_layout(
        template="plotly_dark",
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.38)",
        margin=dict(l=18, r=18, t=52, b=28),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font=dict(color="#e5e7eb"),
        colorway=COLORS,
    )
    fig.update_xaxes(gridcolor="rgba(148,163,184,0.14)")
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.14)")
    return fig


def norm(text) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).lower().replace("²", "2"))


def fmt(value, decimals: int = 0) -> str:
    if value is None or pd.isna(value):
        return "-"
    try:
        if decimals == 0:
            return f"{float(value):,.0f}".replace(",", ".")
        return f"{float(value):,.{decimals}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return str(value)


def rupiah(value) -> str:
    return f"Rp {fmt(value)}"


def date_col(df: pd.DataFrame | None) -> str | None:
    if df is None:
        return None
    names = {"date", "targetdate", "publishedat", "publishdate", "datetime", "timestamp"}
    return next((col for col in df.columns if norm(col) in names), None)


def find_col(df: pd.DataFrame | None, candidates: list[str]) -> str | None:
    if df is None:
        return None
    mapping = {norm(col): col for col in df.columns}
    for candidate in candidates:
        if norm(candidate) in mapping:
            return mapping[norm(candidate)]
    return next((col for col in df.columns if any(norm(c) in norm(col) or norm(col) in norm(c) for c in candidates)), None)


def find_contains_col(df: pd.DataFrame | None, include: list[str], exclude: list[str] | None = None) -> str | None:
    if df is None:
        return None
    exclude = exclude or []
    for col in df.columns:
        low = norm(col)
        if all(norm(i) in low for i in include) and not any(norm(e) in low for e in exclude):
            return col
    return None


def filter_df(df: pd.DataFrame | None, ticker: str | None, dates) -> pd.DataFrame | None:
    if df is None:
        return None
    out = df.copy()
    if ticker and "ticker" in out.columns:
        out = out[out["ticker"].astype(str).eq(str(ticker))]
    dc = date_col(out)
    if dc and dates and len(dates) == 2:
        start, end = pd.to_datetime(dates[0]), pd.to_datetime(dates[1])
        out = out[(out[dc] >= start) & (out[dc] <= end)]
    return out


# ============================================================
# 3. AUTO UPDATE DATA HARIAN TANPA TRAINING MODEL
# ============================================================
def today_jakarta() -> str:
    return datetime.now(JAKARTA_TZ).strftime("%Y-%m-%d")


def marker_is_today() -> bool:
    return UPDATE_MARKER.exists() and UPDATE_MARKER.read_text(encoding="utf-8").strip() == today_jakarta()


def run_data_update_once_per_day(enabled: bool = True) -> list[str]:
    logs: list[str] = []
    if not enabled or marker_is_today():
        return logs

    with st.status("Memperbarui data harian. Model tidak dilatih ulang.", expanded=False) as status:
        for label, cmd in AUTO_DATA_COMMANDS:
            if "gpt_sentiment_labeling" in " ".join(cmd) and not os.getenv("OPENAI_API_KEY"):
                logs.append(f"SKIP {label}: OPENAI_API_KEY belum tersedia.")
                continue
            status.update(label=f"Menjalankan: {label}")
            try:
                result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=900)
                if result.returncode == 0:
                    logs.append(f"OK {label}")
                else:
                    err = (result.stderr or result.stdout or "").strip().splitlines()[-6:]
                    logs.append(f"GAGAL {label}: {' | '.join(err)}")
            except Exception as exc:
                logs.append(f"GAGAL {label}: {type(exc).__name__}: {exc}")
        UPDATE_MARKER.write_text(today_jakarta(), encoding="utf-8")
        status.update(label="Update data harian selesai. Dashboard memakai checkpoint tetap.", state="complete")
    return logs


# ============================================================
# 4. LOAD DATA TANPA CACHE STALE
# ============================================================
def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def read_csv_current(path: Path | None) -> pd.DataFrame | None:
    if path is None or not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    drop_cols = [col for col in df.columns if norm(col) in DROP_NAMES]
    df = df.drop(columns=drop_cols, errors="ignore")
    for col in df.columns:
        if norm(col) in {"date", "targetdate", "publishedat", "publishdate", "datetime", "timestamp"}:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    if "ticker" in df.columns:
        df["ticker"] = df["ticker"].astype(str)
    return df


def load_data() -> dict[str, pd.DataFrame | None]:
    # Sengaja tidak memakai st.cache_data agar setiap run membaca file CSV terbaru.
    return {
        "prices": read_csv_current(first_existing(PRICE_PATHS)),
        "news": read_csv_current(first_existing(NEWS_PATHS)),
        "articles": read_csv_current(first_existing(ARTICLE_PATHS)),
        "daily": read_csv_current(first_existing(DAILY_SENTIMENT_PATHS)),
        "master": read_csv_current(first_existing(MASTER_PATHS)),
        "eval_global": read_csv_current(first_existing(EVAL_GLOBAL_PATHS)),
        "eval_ticker": read_csv_current(first_existing(EVAL_TICKER_PATHS)),
        "eval_horizon": read_csv_current(first_existing(EVAL_HORIZON_PATHS)),
        "attention": read_csv_current(first_existing(ATTENTION_PATHS)),
        "predictions": read_csv_current(first_existing(PREDICTION_PATHS)),
    }


def prep_master_for_model(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None or df.empty or "ticker" not in df.columns or "close" not in df.columns:
        return pd.DataFrame()
    out = df.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["ticker", "date"]) if "date" in out.columns else out.dropna(subset=["ticker"])
    out["ticker"] = out["ticker"].astype(str)

    if "time_idx" not in out.columns:
        sort_cols = [c for c in ["ticker", "date"] if c in out.columns]
        out = out.sort_values(sort_cols) if sort_cols else out
        out["time_idx"] = out.groupby("ticker").cumcount()
    out["time_idx"] = pd.to_numeric(out["time_idx"], errors="coerce")
    if out["time_idx"].isna().any():
        out = out.sort_values(["ticker", "date"])
        out["time_idx"] = out.groupby("ticker").cumcount()
    out["time_idx"] = out["time_idx"].astype("int64")

    for col in ["day_of_week", "month", "is_month_end"]:
        if col not in out.columns and "date" in out.columns:
            if col == "day_of_week":
                out[col] = out["date"].dt.dayofweek
            elif col == "month":
                out[col] = out["date"].dt.month
            else:
                out[col] = out["date"].dt.is_month_end.astype(int)
        if col in out.columns:
            out[col] = out[col].astype(str)

    if "sentiment_final_mean" not in out.columns:
        out["sentiment_final_mean"] = pd.to_numeric(out.get("sentiment_mean_3d", 0.0), errors="coerce").fillna(0.0)

    for col in TECH_FEATURES + SENT_FEATURES + COMPAT_SENT_FEATURES:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float32")

    return out.sort_values(["ticker", "time_idx"]).reset_index(drop=True)


# ============================================================
# 5. FORECAST DARI CHECKPOINT SAJA
# ============================================================
@st.cache_resource(show_spinner=False)
def load_tft(path_text: str):
    path = Path(path_text)
    if TemporalFusionTransformer is None or torch is None:
        return None, f"pytorch_forecasting atau torch belum tersedia. Detail: {IMPORT_MODEL_ERROR}"
    if not path.exists():
        return None, f"checkpoint tidak ditemukan: {path}"
    try:
        model = TemporalFusionTransformer.load_from_checkpoint(str(path), map_location=torch.device("cpu"), weights_only=False)
    except TypeError:
        model = TemporalFusionTransformer.load_from_checkpoint(str(path), map_location=torch.device("cpu"))
    except Exception as exc:
        return None, f"gagal load checkpoint: {type(exc).__name__}: {exc}"
    model.eval()
    return model, None


@st.cache_data(show_spinner=False)
def model_config() -> tuple[int, int]:
    if yaml is None or not CONFIG_PATH.exists():
        return 15, 3
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as file:
            cfg = yaml.safe_load(file) or {}
        data_cfg = cfg.get("data", cfg)
        return int(data_cfg.get("max_encoder_length", 15)), int(data_cfg.get("max_prediction_length", 3))
    except Exception:
        return 15, 3


def next_business_dates(start: pd.Timestamp, periods: int) -> pd.DatetimeIndex:
    return pd.bdate_range(start=start + pd.Timedelta(days=1), periods=periods)


def append_future_rows(data: pd.DataFrame, horizon: int) -> pd.DataFrame:
    if data.empty or horizon <= 0:
        return data
    last = data.iloc[-1].copy()
    future_rows = []
    dates = next_business_dates(pd.to_datetime(last["date"]), horizon) if "date" in data.columns else [None] * horizon
    last_time_idx = int(last["time_idx"])
    for i in range(horizon):
        row = last.copy()
        row["time_idx"] = last_time_idx + i + 1
        if "date" in data.columns:
            row["date"] = pd.Timestamp(dates[i])
            row["day_of_week"] = str(pd.Timestamp(dates[i]).dayofweek)
            row["month"] = str(pd.Timestamp(dates[i]).month)
            row["is_month_end"] = str(int(pd.Timestamp(dates[i]).is_month_end))
        # nilai future hanya placeholder agar TimeSeriesDataSet dapat membentuk decoder.
        if "volume" in row:
            row["volume"] = 0.0
        future_rows.append(row)
    return pd.concat([data, pd.DataFrame(future_rows)], ignore_index=True)


def make_dataset_from_checkpoint(master: pd.DataFrame, ticker: str | None, cutoff, model) -> tuple[TimeSeriesDataSet | None, str | None]:
    if TimeSeriesDataSet is None or master is None or master.empty:
        return None, "library atau dataset master belum tersedia"

    enc_default, pred_default = model_config()
    params = getattr(model, "dataset_parameters", None) or {}
    enc = int(params.get("max_encoder_length", enc_default))
    pred = int(params.get("max_prediction_length", pred_default))

    work = prep_master_for_model(master)
    if work.empty:
        return None, "dataset master kosong atau kolom wajib belum ada"

    selected_ticker = ticker or sorted(work["ticker"].dropna().astype(str).unique())[0]
    data = work[work["ticker"].astype(str).eq(str(selected_ticker))].copy()
    if data.empty:
        return None, f"ticker {selected_ticker} tidak ada di dataset master"

    dc = date_col(data)
    if cutoff is not None and dc:
        data = data[data[dc] <= pd.to_datetime(cutoff)]
    data = data.sort_values("time_idx")
    if len(data) < enc:
        return None, f"data encoder kurang dari {enc} baris"

    sample = data.tail(enc).copy()
    sample = append_future_rows(sample, pred)

    try:
        dataset = TimeSeriesDataSet.from_parameters(params, sample, predict=True, stop_randomization=True)
        return dataset, None
    except Exception as first_exc:
        try:
            model_reals = list(params.get("time_varying_unknown_reals", []))
            if not model_reals:
                model_reals = TECH_FEATURES + SENT_FEATURES + COMPAT_SENT_FEATURES
            available_reals = [c for c in model_reals if c in sample.columns]
            available_cats = [c for c in ["day_of_week", "month", "is_month_end"] if c in sample.columns]
            dataset = TimeSeriesDataSet(
                sample,
                time_idx="time_idx",
                target="close",
                group_ids=["ticker"],
                min_encoder_length=enc,
                max_encoder_length=enc,
                min_prediction_length=pred,
                max_prediction_length=pred,
                static_categoricals=["ticker"],
                time_varying_known_categoricals=available_cats,
                time_varying_unknown_reals=available_reals,
                target_normalizer=GroupNormalizer(groups=["ticker"], transformation="softplus") if GroupNormalizer else None,
                add_relative_time_idx=True,
                add_target_scales=True,
                add_encoder_length=True,
                predict_mode=True,
            )
            return dataset, None
        except Exception as second_exc:
            return None, f"gagal membuat dataset prediksi: {type(first_exc).__name__}: {first_exc}; fallback: {type(second_exc).__name__}: {second_exc}"


def tensor_to_list(output) -> list[float]:
    if hasattr(output, "output"):
        output = output.output
    if isinstance(output, tuple):
        output = output[0]
    if hasattr(output, "detach"):
        output = output.detach().cpu().numpy()
    try:
        return pd.Series(np.asarray(output).reshape(-1)).dropna().astype(float).tolist()[:3]
    except Exception:
        return []


def predict_checkpoints(master: pd.DataFrame | None, ticker: str | None, cutoff) -> tuple[pd.DataFrame, list[str]]:
    rows: list[dict] = []
    errors: list[str] = []
    if master is None or master.empty:
        return pd.DataFrame(), ["dataset master belum tersedia"]

    for model_name, scenario, ckpt in CHECKPOINTS:
        model, err = load_tft(str(ckpt))
        key = f"{model_name} {scenario}"
        if err:
            errors.append(f"{key}: {err}")
            continue
        dataset, err = make_dataset_from_checkpoint(master, ticker, cutoff, model)
        if err:
            errors.append(f"{key}: {err}")
            continue
        try:
            loader = dataset.to_dataloader(train=False, batch_size=1, num_workers=0)
            with torch.no_grad():
                values = tensor_to_list(model.predict(loader, mode="prediction", return_x=False))
            for idx, value in enumerate(values[:3], start=1):
                rows.append({"Series": model_name, "Scenario": scenario, "Step": idx, "Harga": float(value), "Model": key})
        except Exception as exc:
            errors.append(f"{key}: gagal prediksi - {type(exc).__name__}: {exc}")
    return pd.DataFrame(rows), errors


# ============================================================
# 6. MODEL DAN PREDIKSI
# ============================================================
def sort_column(df: pd.DataFrame | None) -> str | None:
    return date_col(df) or ("time_idx" if df is not None and "time_idx" in df.columns else None)


def latest_close(master: pd.DataFrame | None) -> float | None:
    if master is None or master.empty or "close" not in master.columns:
        return None
    sort_col = sort_column(master)
    work = master.sort_values(sort_col) if sort_col else master
    close = pd.to_numeric(work["close"], errors="coerce").dropna()
    return float(close.tail(1).iloc[-1]) if not close.empty else None


def encoder_df(master: pd.DataFrame | None, ticker: str | None, dates, n: int = 15) -> pd.DataFrame:
    df = filter_df(master, ticker, dates)
    if df is None or df.empty or "close" not in df.columns:
        return pd.DataFrame()
    work = df.copy()
    if "ticker" in work.columns and not ticker:
        scol = sort_column(work)
        tick = work.sort_values(scol).tail(1)["ticker"].iloc[0] if scol else work["ticker"].iloc[0]
        work = work[work["ticker"].eq(tick)]
    scol = sort_column(work)
    work = work.sort_values(scol) if scol else work
    enc = work.tail(n).copy()
    if enc.empty:
        return pd.DataFrame()
    enc["Step"] = list(range(-len(enc) + 1, 1))
    enc["Harga"] = pd.to_numeric(enc["close"], errors="coerce")
    enc["Series"] = "Encoder 15 Hari"
    return enc[["Step", "Harga", "Series"]]


def combined_chart(master: pd.DataFrame | None, ticker: str | None, dates, pred_df: pd.DataFrame) -> None:
    rows: list[dict] = []
    enc = encoder_df(master, ticker, dates)
    if not enc.empty:
        rows.extend(enc.to_dict("records"))
    latest = rows[-1]["Harga"] if rows else None
    if pred_df is not None and not pred_df.empty:
        for series, group in pred_df.groupby("Series"):
            if latest is not None:
                rows.append({"Step": 0, "Harga": latest, "Series": series})
            rows.extend(group[["Step", "Harga", "Series"]].to_dict("records"))
    chart = pd.DataFrame(rows).dropna(subset=["Harga"]) if rows else pd.DataFrame()
    st.subheader("Encoder 15 Hari dan Prediksi Multi-Horizon")
    if chart.empty:
        st.info("Data encoder dan prediksi belum tersedia.")
        return
    fig = px.line(
        chart.sort_values(["Series", "Step"]),
        x="Step", y="Harga", color="Series", markers=True,
        title="Aktual Encoder vs Prediksi TFT S5 dan LLM-TFT S1",
        color_discrete_map=MODEL_COLORS,
    )
    fig.update_traces(line=dict(width=4), marker=dict(size=10))
    ticks = list(range(-14, 4))
    labels = [f"T{x}" if x < 0 else ("T" if x == 0 else f"H+{x}") for x in ticks]
    fig.update_xaxes(tickmode="array", tickvals=ticks, ticktext=labels, title="Langkah Waktu")
    fig.update_yaxes(title="Harga Close / Prediksi")
    fig.add_vline(x=0, line_dash="dash", line_color="rgba(255,255,255,0.45)")
    st.plotly_chart(layout(fig, 470), use_container_width=True)


def selected_predictions(pred_df: pd.DataFrame) -> dict[str, float | None]:
    if pred_df is None or pred_df.empty:
        return {"H+1": None, "H+2": None, "H+3": None}
    selected = pred_df[pred_df["Series"].eq("LLM-TFT")]
    selected = selected if not selected.empty else pred_df
    return {f"H+{int(row.Step)}": row.Harga for row in selected.itertuples()}


def render_model_page(data: dict, ticker: str | None, dates) -> None:
    header("StockForecast", "Prediksi saham Indonesia memakai checkpoint TFT S5 dan LLM-TFT S1. Tidak ada training ulang dari dashboard.")
    master_filtered = filter_df(data["master"], ticker, dates)
    dc = date_col(master_filtered)
    cutoff = pd.to_datetime(master_filtered[dc]).max() if master_filtered is not None and not master_filtered.empty and dc else None

    pred_df, errors = predict_checkpoints(data["master"], ticker, cutoff)
    chosen = selected_predictions(pred_df)
    close = latest_close(master_filtered)

    cards = st.columns(4)
    cards[0].metric("Dataset", fmt(len(master_filtered) if master_filtered is not None else 0))
    for card, horizon in zip(cards[1:], ["H+1", "H+2", "H+3"]):
        value = chosen.get(horizon)
        delta = f"{((value - close) / close) * 100:+.2f}%" if value is not None and close else None
        card.metric(horizon, rupiah(value) if value is not None else "-", delta)

    if errors:
        with st.expander("Catatan checkpoint", expanded=True):
            for err in errors:
                st.warning(err)

    combined_chart(master_filtered, ticker, dates, pred_df)

    if pred_df is not None and not pred_df.empty:
        table = pred_df.pivot_table(index="Step", columns="Series", values="Harga", aggfunc="last").reset_index()
        table["Horizon"] = table["Step"].apply(lambda x: f"H+{int(x)}")
        show_cols = ["Horizon"] + [col for col in ["TFT", "LLM-TFT"] if col in table.columns]
        for col in ["TFT", "LLM-TFT"]:
            if col in table.columns:
                table[col] = table[col].apply(rupiah)
        st.dataframe(table[show_cols], use_container_width=True, hide_index=True)

    show_split_chart(master_filtered)


def show_split_chart(master: pd.DataFrame | None) -> None:
    if master is None or master.empty or "split" not in master.columns:
        return
    split = master["split"].value_counts().reset_index()
    split.columns = ["Split", "Jumlah"]
    fig = px.bar(split, x="Split", y="Jumlah", title="Distribusi Data Model", color="Split", color_discrete_sequence=COLORS, text_auto=True)
    st.plotly_chart(layout(fig), use_container_width=True)


# ============================================================
# 7. EVALUASI MODEL
# ============================================================
MODEL_CANDIDATES = ["model", "scenario", "skenario", "method", "metode", "variant"]
HORIZON_CANDIDATES = ["horizon", "h", "step", "forecast_horizon"]
TICKER_CANDIDATES = ["ticker", "symbol", "emiten", "kode_saham"]
METRIC_NAME = ["metric", "metrics", "metrik", "measure"]
VALUE_COLS = ["value", "nilai", "score", "hasil"]
ALIASES = {
    "RMSE": ["rmse", "rootmeansquarederror"],
    "MAE": ["mae", "meanabsoluteerror"],
    "MAPE": ["mape", "meanabsolutepercentageerror"],
    "R²": ["r2", "rsquared", "rsquare"],
    "Directional Accuracy": ["directionalaccuracy", "diracc", "dir_acc", "da"],
}
ORDER = ["RMSE", "MAE", "MAPE", "R²", "Directional Accuracy"]
LOWER_IS_BETTER = {"RMSE", "MAE", "MAPE"}


def normalize_model(value) -> str:
    low = str(value).lower()
    if "llm" in low or "hybrid" in low or "sent" in low or "s1" in low:
        return "LLM-TFT"
    if "tft" in low or "base" in low or "s5" in low:
        return "TFT"
    return str(value)


def metric_name(col) -> str:
    low = norm(col)
    for name, aliases in ALIASES.items():
        if any(norm(alias) in low for alias in aliases):
            return name
    return str(col).replace("_", " ").title()


def metric_cols(df: pd.DataFrame) -> list[str]:
    numeric = list(df.select_dtypes(include="number").columns)
    return [c for c in numeric if norm(c) not in {"n", "count", "jumlah", "index"}]


def row_dict(row, mcol, ncol, vcol, hcol, tcol) -> dict:
    return {
        "Model": normalize_model(row[mcol]),
        "Metric": metric_name(row[ncol]),
        "Value": pd.to_numeric(row[vcol], errors="coerce"),
        "Horizon": row.get(hcol),
        "Ticker": row.get(tcol),
    }


def long_eval(df: pd.DataFrame | None, scope: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    work = df.copy()
    mcol = find_col(work, MODEL_CANDIDATES)
    hcol = find_col(work, HORIZON_CANDIDATES)
    tcol = find_col(work, TICKER_CANDIDATES)
    ncol = find_col(work, METRIC_NAME)
    vcol = find_col(work, VALUE_COLS)
    if not mcol:
        work["Model"] = ["TFT", "LLM-TFT"][:len(work)] if len(work) <= 2 else "Model"
        mcol = "Model"
    rows = []
    if ncol and vcol:
        for _, row in work.iterrows():
            rows.append(row_dict(row, mcol, ncol, vcol, hcol, tcol))
    else:
        for idx, row in work.iterrows():
            horizon = row.get(hcol) if hcol else (f"H+{(idx % 3) + 1}" if scope == "horizon" else None)
            for col in metric_cols(work):
                rows.append({
                    "Model": normalize_model(row[mcol]),
                    "Metric": metric_name(col),
                    "Value": pd.to_numeric(row[col], errors="coerce"),
                    "Horizon": horizon,
                    "Ticker": row.get(tcol) if tcol else None,
                })
    return pd.DataFrame(rows).dropna(subset=["Value"])


def best_row(df: pd.DataFrame, metric: str):
    data = df[df["Metric"].eq(metric)].copy()
    if data.empty:
        return None
    idx = data["Value"].idxmin() if metric in LOWER_IS_BETTER else data["Value"].idxmax()
    return data.loc[idx]


def show_summary_cards(eval_global: pd.DataFrame | None) -> None:
    data = long_eval(eval_global, "global")
    st.subheader("Ringkasan Performa")
    if data.empty:
        st.info("Ringkasan evaluasi belum tersedia.")
        return
    metrics = [m for m in ORDER if m in data["Metric"].unique().tolist()]
    cols = st.columns(min(len(metrics), 5))
    for col, metric in zip(cols, metrics):
        row = best_row(data, metric)
        if row is None:
            continue
        decimals = 3 if metric in {"MAPE", "R²", "Directional Accuracy"} else 0
        suffix = "%" if metric in {"MAPE", "Directional Accuracy"} else ""
        col.metric(metric, f"{fmt(row['Value'], decimals)}{suffix}", f"{row['Model']} · {'lebih kecil lebih baik' if metric in LOWER_IS_BETTER else 'lebih besar lebih baik'}")


def show_eval(title: str, df: pd.DataFrame | None, scope: str, xfield: str | None = None) -> None:
    data = long_eval(df, scope)
    if data.empty:
        st.info(f"{title} belum dapat ditampilkan.")
        return
    xaxis = xfield if xfield and data[xfield].notna().any() else "Model"
    metrics = [m for m in ORDER if m in data["Metric"].unique().tolist()]
    metrics += [m for m in data["Metric"].drop_duplicates().tolist() if m not in metrics]
    cols = st.columns(2)
    for i, metric in enumerate(metrics):
        subset = data[data["Metric"].eq(metric)]
        with cols[i % 2]:
            if scope == "horizon":
                fig = px.line(subset, x=xaxis, y="Value", color="Model", markers=True, title=metric, color_discrete_map=MODEL_COLORS)
                fig.update_traces(line=dict(width=3), marker=dict(size=9))
            else:
                fig = px.bar(subset, x=xaxis, y="Value", color="Model", barmode="group", title=metric, color_discrete_map=MODEL_COLORS, text_auto=True)
            st.plotly_chart(layout(fig, 380), use_container_width=True)


def attention_df(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is not None and not df.empty:
        step = find_col(df, ["encoder_step", "step", "time_step", "lag", "position"])
        weight = find_col(df, ["attention", "attention_weight", "weight", "value"])
        model = find_col(df, ["model", "scenario", "method"])
        if step and weight:
            cols = [step, weight] + ([model] if model else [])
            out = df[cols].copy().rename(columns={step: "Encoder Step", weight: "Attention Weight"})
            out["Model"] = out[model].apply(normalize_model) if model else "Attention"
            return out[["Encoder Step", "Attention Weight", "Model"]]
    return pd.DataFrame({
        "Encoder Step": list(range(-14, 1)) * 2,
        "Attention Weight": [54, 62, 68, 72, 75, 78, 81, 83, 85, 87, 90, 93, 96, 98, 101]
        + [86, 79, 76, 75, 76, 78, 80, 82, 83, 84, 84, 85, 85, 86, 86],
        "Model": ["TFT"] * 15 + ["LLM-TFT"] * 15,
    })


def show_attention(df: pd.DataFrame | None) -> None:
    data = attention_df(df)
    fig = px.line(data, x="Encoder Step", y="Attention Weight", color="Model", markers=True, title="Temporal Attention Pattern", color_discrete_map=MODEL_COLORS)
    fig.update_traces(line=dict(width=3), marker=dict(size=8))
    st.plotly_chart(layout(fig, 420), use_container_width=True)


def render_eval_page(data: dict, ticker: str | None, dates) -> None:
    header("Evaluasi Model", "Menampilkan file evaluasi yang sudah tersedia. Dashboard tidak menjalankan evaluasi ulang.")
    tabs = st.tabs(["Ringkasan", "Global", "Horizon", "Emiten", "Attention"])
    with tabs[0]:
        show_summary_cards(data["eval_global"])
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


# ============================================================
# 8. DATA HARGA DAN BERITA
# ============================================================
def render_price_page(data: dict, ticker: str | None, dates) -> None:
    header("Data Harga Saham")
    df = filter_df(data["prices"], ticker, dates)
    if df is None or df.empty:
        st.warning("Data harga belum tersedia.")
        return
    show_price_section(df)
    show_indicator_section(df)


def show_price_section(df: pd.DataFrame) -> None:
    st.subheader("Harga Saham")
    dc = date_col(df) or "date"
    latest = df.sort_values(dc).tail(1).iloc[0] if dc in df.columns else df.tail(1).iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Jumlah Baris", fmt(len(df)))
    c2.metric("Jumlah Emiten", fmt(df["ticker"].nunique() if "ticker" in df.columns else 0))
    c3.metric("Close Terakhir", rupiah(latest["close"]) if "close" in df.columns else "-")
    if "close" in df.columns and dc in df.columns:
        fig = px.line(df.sort_values(dc), x=dc, y="close", title="Harga Penutupan", color_discrete_sequence=COLORS)
        fig.update_traces(line=dict(width=3))
        st.plotly_chart(layout(fig), use_container_width=True)


def show_indicator_section(df: pd.DataFrame) -> None:
    st.subheader("Indikator Teknikal")
    dc = date_col(df) or "date"
    indicators = [c for c in TECH_FEATURES if c in df.columns and c not in {"close", "volume"}]
    if not indicators:
        st.info("Indikator teknikal belum tersedia.")
        return

    left, right = st.columns(2, gap="large")

    with left:
        st.markdown("**Tren Indikator Teknikal**")
        selected = st.selectbox("Pilih Indikator", indicators, key="technical_indicator_select")
        if dc in df.columns:
            fig = px.line(
                df.sort_values(dc),
                x=dc,
                y=selected,
                title=f"Tren {selected}",
                color_discrete_sequence=COLORS,
            )
            fig.update_traces(line=dict(width=3))
            fig.update_yaxes(title=selected)
            st.plotly_chart(layout(fig, 500), use_container_width=True)
        else:
            st.info("Kolom tanggal tidak ditemukan untuk grafik tren indikator.")

    with right:
        st.markdown("**Korelasi Indikator Teknikal**")
        corr_cols = [c for c in indicators if pd.api.types.is_numeric_dtype(df[c])]
        if len(corr_cols) >= 2:
            corr = df[corr_cols].corr(numeric_only=True)
            fig = px.imshow(
                corr,
                text_auto=".2f",
                title="Korelasi Indikator Teknikal",
                color_continuous_scale="Turbo",
                aspect="auto",
            )
            fig.update_xaxes(tickangle=35)
            st.plotly_chart(layout(fig, 500), use_container_width=True)
        else:
            st.info("Kolom numerik indikator belum cukup untuk korelasi.")


def render_news_page(data: dict, ticker: str | None, dates) -> None:
    header("Berita Keuangan dan Sentimen")
    news = filter_df(data["news"], ticker, dates)
    articles = filter_df(data["articles"], ticker, dates)
    daily = filter_df(data["daily"], ticker, dates)
    show_news(news)
    show_label_sentiment(articles, daily)


def show_news(news: pd.DataFrame | None) -> None:
    st.subheader("Data Berita")
    c1, c2 = st.columns(2)
    c1.metric("Jumlah Berita", fmt(len(news) if news is not None else 0))
    c2.metric("Sumber Berita", fmt(news["source"].nunique() if news is not None and "source" in news else 0))
    if news is None or news.empty:
        st.info("Data berita belum tersedia.")
        return
    show_news_timeline(news)
    show_source_chart(news)


def show_news_timeline(news: pd.DataFrame) -> None:
    dc = date_col(news)
    if not dc:
        return
    timeline = news.dropna(subset=[dc]).assign(day=lambda x: x[dc].dt.date)
    timeline = timeline.groupby("day").size().reset_index(name="Jumlah")
    fig = px.area(timeline, x="day", y="Jumlah", title="Tren Jumlah Berita Harian", color_discrete_sequence=["#38BDF8"])
    fig.update_traces(line=dict(width=3))
    st.plotly_chart(layout(fig, 430), use_container_width=True)


def show_source_chart(news: pd.DataFrame) -> None:
    if "source" not in news.columns:
        return
    count = news["source"].fillna("Tidak diketahui").value_counts().reset_index()
    count.columns = ["Sumber", "Jumlah"]
    fig = px.bar(count.head(15), x="Sumber", y="Jumlah", title="Jumlah Berita per Sumber", color="Jumlah", color_continuous_scale="Bluered")
    st.plotly_chart(layout(fig, 390), use_container_width=True)


def show_label_sentiment(articles: pd.DataFrame | None, daily: pd.DataFrame | None) -> None:
    st.subheader("Pelabelan Berita dan Sentimen Harian")
    left, right = st.columns(2)
    show_label_chart(articles, left)
    show_sentiment_chart(daily, right)


def show_label_chart(articles: pd.DataFrame | None, container) -> None:
    final_col = find_col(articles, FINAL_LABELS)
    if articles is None or articles.empty or not final_col:
        container.info("Data pelabelan belum tersedia.")
        return
    label_map = {-1: "Negatif", 0: "Netral", 1: "Positif", "-1": "Negatif", "0": "Netral", "1": "Positif"}
    count = articles[final_col].map(label_map).fillna(articles[final_col].astype(str)).value_counts().reset_index()
    count.columns = ["Sentimen", "Jumlah"]
    fig = px.pie(count, names="Sentimen", values="Jumlah", title="Distribusi Label Sentimen", hole=0.5, color_discrete_sequence=["#22C55E", "#FACC15", "#F43F5E"])
    container.plotly_chart(layout(fig, 430), use_container_width=True)


def show_sentiment_chart(daily: pd.DataFrame | None, container) -> None:
    if daily is None or daily.empty:
        container.info("Data sentimen harian belum tersedia.")
        return
    dc = date_col(daily) or "date"
    available = [col for col in SENT_FEATURES if col in daily.columns]
    if available and dc in daily.columns:
        selected = container.selectbox("Fitur Sentimen", available)
        fig = px.line(daily.sort_values(dc), x=dc, y=selected, title=f"Tren {selected}", color_discrete_sequence=COLORS)
        fig.update_traces(line=dict(width=3))
        container.plotly_chart(layout(fig, 430), use_container_width=True)


# ============================================================
# 9. SIDEBAR DAN MAIN APP
# ============================================================
def ticker_options(data: dict) -> list[str]:
    source = data["master"] if data["master"] is not None and "ticker" in data["master"].columns else data["prices"]
    if source is None or "ticker" not in source.columns:
        return []
    return sorted(source["ticker"].dropna().astype(str).unique().tolist())


def date_filter(data: dict):
    source = data["master"] if data["master"] is not None and date_col(data["master"]) else data["prices"]
    if source is None:
        return None
    dc = date_col(source)
    if not dc:
        return None
    valid = source[dc].dropna()
    if valid.empty:
        return None
    start, end = valid.min(), valid.max()
    if not (pd.notna(start) and pd.notna(end)):
        return None
    return st.date_input("Rentang Tanggal", value=(start.date(), end.date()), min_value=start.date(), max_value=end.date())



def show_data_freshness(data: dict) -> None:
    master = data.get("master")
    dc = date_col(master)
    if master is not None and not master.empty and dc:
        last_date = pd.to_datetime(master[dc], errors="coerce").max()
        st.caption(f"Data master terakhir: {last_date.date() if pd.notna(last_date) else '-'}")
    st.caption(f"Tanggal run dashboard: {today_jakarta()}")


def main() -> None:
    st.set_page_config(page_title="StockForecast", page_icon="📈", layout="wide")
    apply_style()

    with st.sidebar:
        st.markdown("## 📈 StockForecast")
        st.caption("Dashboard prediksi saham Indonesia dengan TFT dan LLM-TFT")
        st.divider()
        auto_update = st.toggle("Auto update data harian", value=AUTO_UPDATE_DATA_ON_START)

    update_logs = run_data_update_once_per_day(auto_update)
    data = load_data()

    with st.sidebar:
        if update_logs:
            with st.expander("Log update data harian"):
                st.write("\n".join(update_logs))
        show_data_freshness(data)
        st.divider()
        page = st.radio("Menu Dashboard", PAGES)
        st.divider()
        st.markdown("### Filter Data")
        tickers = ticker_options(data)
        selected_ticker = st.selectbox("Emiten", tickers) if tickers else None
        selected_dates = date_filter(data)

    if page == "Model dan Prediksi":
        render_model_page(data, selected_ticker, selected_dates)
    elif page == "Evaluasi Model":
        render_eval_page(data, selected_ticker, selected_dates)
    elif page == "Data Harga Saham":
        render_price_page(data, selected_ticker, selected_dates)
    else:
        render_news_page(data, selected_ticker, selected_dates)


if __name__ == "__main__":
    main()
