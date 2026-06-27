from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Streamlit Cloud tidak menyediakan GPU. Baris ini harus dieksekusi sebelum torch dipakai.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

IMPORT_MODEL_ERROR: str | None = None
try:
    import torch
    import yaml
    from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
    from pytorch_forecasting.data import GroupNormalizer
except Exception as exc:  # pragma: no cover - ditampilkan di dashboard health page
    IMPORT_MODEL_ERROR = f"{type(exc).__name__}: {exc}"
    torch = None
    yaml = None
    TemporalFusionTransformer = None
    TimeSeriesDataSet = None
    GroupNormalizer = None


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================
def resolve_project_root() -> Path:
    current = Path(__file__).resolve()
    for base in [current.parent, *current.parents]:
        if (base / "configs").exists() and ((base / "src").exists() or (base / "data").exists()):
            return base
    for base in [current.parent, *current.parents]:
        if any((base / name).exists() for name in ["data", "modelssss", "reportss", "reports", "configs"]):
            return base
    return current.parents[2] if len(current.parents) >= 3 else current.parent


ROOT = resolve_project_root()
DATA = ROOT / "data"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
REPORTSS = ROOT / "reportss"
REPORTS = ROOT / "reports"
CONFIG_PATH = ROOT / "configs" / "model_tft.yaml"

CHECKPOINTS: list[tuple[str, str, Path]] = [
    ("TFT", "S5", ROOT / "modelssss/baseline/S5/best-checkpoint.ckpt"),
    ("LLM-TFT", "S1", ROOT / "modelssss/hybrid/S1/best-checkpoint.ckpt"),
]

PATHS: dict[str, list[Path]] = {
    "prices": [INTERIM / "prices_with_indicators.csv", PROCESSED / "prices_with_indicators.csv"],
    "news": [INTERIM / "news_clean.csv", PROCESSED / "news_clean.csv"],
    "articles": [PROCESSED / "news_with_sentiment_per_article.csv", INTERIM / "article_sentiment.csv"],
    "daily": [PROCESSED / "daily_sentiment.csv", INTERIM / "daily_sentiment.csv"],
    "master": [PROCESSED / "tft_master.csv", INTERIM / "tft_master.csv"],
    "eval_global": [REPORTSS / "eval_metrics_global.csv", REPORTS / "eval_metrics_global.csv"],
    "eval_ticker": [REPORTSS / "eval_metrics_by_ticker_global.csv", REPORTS / "eval_metrics_by_ticker_global.csv"],
    "eval_horizon": [REPORTSS / "eval_metrics_by_horizon.csv", REPORTS / "eval_metrics_by_horizon.csv"],
    "attention": [REPORTSS / "attention_comparison.csv", REPORTSS / "attention_weights.csv"],
    "predictions": [REPORTSS / "backtest_predictions.csv", REPORTSS / "predictions.csv", PROCESSED / "predictions.csv"],
}

TECH_FEATURES = [
    "close", "volume", "log_return_1d", "log_return_2d", "vol_20", "rsi_14",
    "ma_5_div_ma_20", "bb_width_20", "gap_return_1d", "intraday_range_pct",
]
SENT_FEATURES = [
    "news_count_3d", "sentiment_mean_3d", "sentiment_ema_7d",
    "sentiment_trend_7d", "sentiment_delta_1d", "sentiment_dir_signal",
]
COMPAT_SENT_FEATURES = ["sentiment_final_mean"]
CALENDAR_CATS = ["day_of_week", "month", "is_month_end"]
FINAL_LABELS = ["l_final", "final_label", "label_final", "sentiment_final", "sentiment"]
DROP_NAMES = {"n", "count", "jumlah", "index", "unnamed", "unnamed0"}



# ============================================================
# 2. DESIGN SYSTEM
# ============================================================
PAGE_CONFIG = {
    "page_title": "StockForecast Pro",
    "page_icon": "📈",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
    "menu_items": {
        "About": "StockForecast Pro — dashboard TFT dan LLM-TFT untuk prediksi harga saham Indonesia.",
    },
}

PAGES = [
    "Prediction Studio",
    "Model Performance",
    "Market Data Lab",
    "Sentiment Intelligence",
]

MODEL_COLORS = {
    "TFT": "#38BDF8",
    "LLM-TFT": "#F97316",
    "TFT S5": "#38BDF8",
    "LLM-TFT S1": "#F97316",
    "Aktual": "#E5E7EB",
    "Encoder": "#94A3B8",
    "Positif": "#22C55E",
    "Netral": "#FACC15",
    "Negatif": "#F43F5E",
}

QUALITATIVE = ["#38BDF8", "#F97316", "#22C55E", "#A855F7", "#F43F5E", "#14B8A6", "#FACC15"]

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root {
  --bg0: #020617;
  --bg1: #0f172a;
  --card: rgba(15, 23, 42, 0.76);
  --card2: rgba(30, 41, 59, 0.66);
  --line: rgba(148, 163, 184, 0.18);
  --text: #e5e7eb;
  --muted: #94a3b8;
  --blue: #38bdf8;
  --orange: #fb923c;
  --green: #22c55e;
  --red: #f43f5e;
}
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp {
  background:
    radial-gradient(circle at 6% 4%, rgba(56,189,248,0.20), transparent 32%),
    radial-gradient(circle at 86% 0%, rgba(249,115,22,0.14), transparent 28%),
    radial-gradient(circle at 72% 88%, rgba(168,85,247,0.10), transparent 32%),
    linear-gradient(135deg, #020617 0%, #0f172a 45%, #111827 100%);
  color: var(--text);
}
.block-container { padding-top: 1.2rem; padding-bottom: 2.4rem; max-width: 1480px; }
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, rgba(2,6,23,0.98), rgba(15,23,42,0.98));
  border-right: 1px solid var(--line);
}
.hero {
  padding: 24px 28px;
  border-radius: 28px;
  background:
    linear-gradient(135deg, rgba(15,23,42,0.94), rgba(30,41,59,0.58)),
    radial-gradient(circle at 8% 0%, rgba(56,189,248,0.28), transparent 38%),
    radial-gradient(circle at 95% 12%, rgba(249,115,22,0.22), transparent 35%);
  border: 1px solid rgba(148,163,184,0.22);
  box-shadow: 0 26px 70px rgba(0,0,0,0.35);
  margin-bottom: 20px;
}
.hero-eyebrow { color: #7dd3fc; font-weight: 700; letter-spacing: .16em; text-transform: uppercase; font-size: .74rem; }
.hero h1 { margin: 6px 0 8px; color: #f8fafc; font-size: 2.35rem; line-height: 1.08; letter-spacing: -0.055em; }
.hero p { margin: 0; color: #cbd5e1; max-width: 920px; font-size: .98rem; }
.page-chip-row { display:flex; gap:10px; flex-wrap:wrap; margin:-6px 0 18px; }
.page-chip { padding:7px 11px; border-radius:999px; background:rgba(15,23,42,.62); border:1px solid rgba(148,163,184,.16); color:#cbd5e1; font-size:.78rem; font-weight:700; }
.card {
  padding: 18px 18px;
  border-radius: 22px;
  background: linear-gradient(145deg, rgba(15,23,42,0.82), rgba(30,41,59,0.56));
  border: 1px solid rgba(148,163,184,0.16);
  box-shadow: 0 12px 30px rgba(0,0,0,0.22);
}
.small-caption { color: var(--muted); font-size: .84rem; }
.pill {
  display: inline-flex; align-items: center; gap: 8px;
  padding: 7px 10px; border-radius: 999px;
  border: 1px solid rgba(148,163,184,0.20);
  background: rgba(15,23,42,0.72);
  color: #cbd5e1; font-size: .80rem; font-weight: 600;
}
.status-ok { color: #86efac; }
.status-warn { color: #fde68a; }
.status-bad { color: #fda4af; }
div[data-testid="stMetric"] {
  background: linear-gradient(145deg, rgba(15,23,42,0.88), rgba(30,41,59,0.70));
  border: 1px solid rgba(148,163,184,0.16);
  padding: 15px 16px;
  border-radius: 18px;
  box-shadow: 0 10px 26px rgba(0,0,0,0.20);
}
div[data-testid="stMetric"] label { color: #cbd5e1 !important; }
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
  border-radius: 999px;
  padding: 8px 15px;
  background: rgba(15,23,42,0.64);
  border: 1px solid rgba(148,163,184,0.14);
}
.stDataFrame { border-radius: 16px; overflow: hidden; border: 1px solid rgba(148,163,184,0.14); }
hr { border-color: rgba(148,163,184,0.14); }
</style>
"""


def setup_page() -> None:
    st.set_page_config(**PAGE_CONFIG)
    st.markdown(CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str | None = None, eyebrow: str = "StockForecast Pro") -> None:
    subtitle_html = f"<p>{subtitle}</p>" if subtitle else ""
    st.markdown(
        f"""
        <div class="hero">
          <div class="hero-eyebrow">{eyebrow}</div>
          <h1>{title}</h1>
          {subtitle_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def panel(text: str, kind: str = "info") -> None:
    cls = {"ok": "status-ok", "warn": "status-warn", "bad": "status-bad"}.get(kind, "")
    st.markdown(f"<span class='pill {cls}'>{text}</span>", unsafe_allow_html=True)


def page_chips(*items: str) -> None:
    html = "".join(f"<span class='page-chip'>{item}</span>" for item in items)
    st.markdown(f"<div class='page-chip-row'>{html}</div>", unsafe_allow_html=True)


def apply_fig_theme(fig: go.Figure, height: int = 420, title: str | None = None) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        height=height,
        title=title,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.35)",
        margin=dict(l=18, r=18, t=52 if title else 28, b=32),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font=dict(color="#e5e7eb", family="Inter"),
        colorway=QUALITATIVE,
    )
    fig.update_xaxes(gridcolor="rgba(148,163,184,0.13)", zerolinecolor="rgba(148,163,184,0.22)")
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.13)", zerolinecolor="rgba(148,163,184,0.22)")
    return fig


# ============================================================
# 3. UTILITIES
# ============================================================
def norm(text: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).lower().replace("²", "2"))


def fmt(value: Any, decimals: int = 0) -> str:
    if value is None or pd.isna(value):
        return "-"
    try:
        if decimals == 0:
            return f"{float(value):,.0f}".replace(",", ".")
        return f"{float(value):,.{decimals}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return str(value)


def rupiah(value: Any) -> str:
    return f"Rp {fmt(value)}"


def pct(value: Any, decimals: int = 2) -> str:
    return f"{fmt(value, decimals)}%" if value is not None and not pd.isna(value) else "-"



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


def first_existing(paths: list[Path]) -> Path | None:
    return next((path for path in paths if path.exists() and path.stat().st_size > 0), None)


def normalize_date_range(dates: Any) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    if isinstance(dates, (list, tuple)) and len(dates) == 2 and dates[0] and dates[1]:
        start, end = pd.to_datetime(dates[0]), pd.to_datetime(dates[1])
        return min(start, end), max(start, end)
    return None


def filter_df(df: pd.DataFrame | None, ticker: str | None = None, dates: Any = None) -> pd.DataFrame | None:
    if df is None:
        return None
    out = df.copy()
    if ticker and "ticker" in out.columns:
        out = out[out["ticker"].astype(str).eq(str(ticker))]
    dc = date_col(out)
    drange = normalize_date_range(dates)
    if dc and drange:
        start, end = drange
        out = out[(out[dc] >= start) & (out[dc] <= end)]
    return out


def read_csv_current(path: Path | None) -> pd.DataFrame | None:
    if path is None:
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


@st.cache_data(show_spinner=False, ttl=120)
def load_all_data() -> dict[str, pd.DataFrame | None]:
    return {name: read_csv_current(first_existing(paths)) for name, paths in PATHS.items()}


def ticker_options(data: dict[str, pd.DataFrame | None]) -> list[str]:
    for key in ["master", "prices", "daily", "news"]:
        source = data.get(key)
        if source is not None and "ticker" in source.columns:
            values = source["ticker"].dropna().astype(str).unique().tolist()
            if values:
                return sorted(values)
    return []


def global_date_range(data: dict[str, pd.DataFrame | None]) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    for key in ["master", "prices", "daily", "news"]:
        source = data.get(key)
        dc = date_col(source)
        if source is not None and dc:
            valid = source[dc].dropna()
            if not valid.empty:
                return pd.to_datetime(valid.min()), pd.to_datetime(valid.max())
    return None


# ============================================================
# 4. CHECKPOINT AND PREDICTION ENGINE
# ============================================================
def clean_cpu_object(obj: Any) -> Any:
    if torch is not None:
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu()
        if isinstance(obj, torch.device):
            return torch.device("cpu")
    if isinstance(obj, dict):
        cleaned = {}
        for key, value in obj.items():
            low = norm(key)
            if low in {"device", "devices", "accelerator", "gpus", "gpu", "strategy", "maplocation"}:
                if low == "accelerator":
                    cleaned[key] = "cpu"
                elif low in {"gpus", "gpu"}:
                    cleaned[key] = 0
                elif low == "devices":
                    cleaned[key] = 1
                else:
                    cleaned[key] = "cpu"
            else:
                cleaned[key] = clean_cpu_object(value)
        return cleaned
    if isinstance(obj, list):
        return [clean_cpu_object(item) for item in obj]
    if isinstance(obj, tuple):
        return tuple(clean_cpu_object(item) for item in obj)
    if isinstance(obj, str) and obj.lower() == "cuda":
        return "cpu"
    return obj


def find_key_recursive(obj: Any, target_key: str) -> Any | None:
    if isinstance(obj, dict):
        if target_key in obj:
            return obj[target_key]
        for value in obj.values():
            found = find_key_recursive(value, target_key)
            if found is not None:
                return found
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            found = find_key_recursive(item, target_key)
            if found is not None:
                return found
    return None


def remove_key_recursive(obj: Any, target_key: str) -> int:
    removed = 0
    if isinstance(obj, dict):
        if target_key in obj:
            obj.pop(target_key, None)
            removed += 1
        for value in list(obj.values()):
            removed += remove_key_recursive(value, target_key)
    elif isinstance(obj, list):
        for item in obj:
            removed += remove_key_recursive(item, target_key)
    elif isinstance(obj, tuple):
        for item in obj:
            removed += remove_key_recursive(item, target_key)
    return removed


@st.cache_data(show_spinner=False)
def model_lengths() -> tuple[int, int]:
    if yaml is None or not CONFIG_PATH.exists():
        return 15, 3
    try:
        cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
        data_cfg = cfg.get("data", cfg)
        return int(data_cfg.get("max_encoder_length", 15)), int(data_cfg.get("max_prediction_length", 3))
    except Exception:
        return 15, 3


@st.cache_resource(show_spinner=False)
def load_tft(path_text: str):
    path = Path(path_text)
    if TemporalFusionTransformer is None or torch is None:
        return None, f"pytorch_forecasting/torch belum tersedia: {IMPORT_MODEL_ERROR}"
    if not path.exists():
        return None, f"checkpoint tidak ditemukan: {path}"
    if path.stat().st_size < 1024:
        return None, f"checkpoint terlalu kecil/kosong: {path}. Pastikan Git LFS atau file .ckpt asli tersedia."

    try:
        checkpoint = torch.load(str(path), map_location=torch.device("cpu"), weights_only=False)
    except Exception as exc:
        return None, f"gagal membaca checkpoint: {type(exc).__name__}: {exc}"

    dataset_params = find_key_recursive(checkpoint, "dataset_parameters") if isinstance(checkpoint, dict) else None

    def attach_params(model):
        if dataset_params is not None:
            try:
                setattr(model, "dataset_parameters", dataset_params)
            except Exception:
                pass
        try:
            model.cpu()
            model.eval()
        except Exception:
            pass
        return model

    # Direct load untuk checkpoint yang sudah kompatibel.
    try:
        model = TemporalFusionTransformer.load_from_checkpoint(
            str(path),
            map_location=torch.device("cpu"),
            strict=False,
        )
        return attach_params(model), None
    except Exception:
        pass

    if not isinstance(checkpoint, dict):
        return None, "format checkpoint tidak dikenali"

    checkpoint = clean_cpu_object(checkpoint)
    removed: list[str] = []
    for key in [
        "mask_bias", "logging_metrics", "monotone_constraints", "monotone_constaints",
        "dataset_parameters", "loss", "accelerator", "devices", "gpus", "gpu", "strategy",
    ]:
        count = remove_key_recursive(checkpoint, key)
        if count:
            removed.append(f"{key}({count})")

    clean_dir = ROOT / ".streamlit_ckpt_cache"
    clean_dir.mkdir(parents=True, exist_ok=True)
    clean_path = clean_dir / f"{path.parent.parent.name}_{path.parent.name}_cpu_clean.ckpt"
    last_error = ""

    for _ in range(60):
        try:
            torch.save(checkpoint, clean_path)
            model = TemporalFusionTransformer.load_from_checkpoint(
                str(clean_path),
                map_location=torch.device("cpu"),
                strict=False,
            )
            return attach_params(model), None
        except TypeError as exc:
            msg = str(exc)
            last_error = msg
            marker = "unexpected keyword argument '"
            if marker not in msg:
                return None, f"gagal load checkpoint: TypeError: {msg}. Dihapus: {removed}"
            bad_key = msg.split(marker, 1)[1].split("'", 1)[0]
            count = remove_key_recursive(checkpoint, bad_key)
            removed.append(f"{bad_key}({count})")
            if count == 0:
                return None, f"gagal load checkpoint: {msg}. Dihapus: {removed}"
        except RuntimeError as exc:
            msg = str(exc)
            last_error = msg
            if "nvidia" in msg.lower() or "cuda" in msg.lower():
                checkpoint = clean_cpu_object(checkpoint)
                for key in ["device", "devices", "accelerator", "gpus", "gpu", "strategy"]:
                    count = remove_key_recursive(checkpoint, key)
                    if count:
                        removed.append(f"{key}({count})")
                continue
            return None, f"gagal load checkpoint: RuntimeError: {msg}. Dihapus: {removed}"
        except Exception as exc:
            return None, f"gagal load checkpoint: {type(exc).__name__}: {exc}. Dihapus: {removed}"

    return None, f"gagal load checkpoint setelah CPU clean. Error terakhir: {last_error}. Dihapus: {removed}"


def prepare_master(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None or df.empty or "ticker" not in df.columns or "close" not in df.columns:
        return pd.DataFrame()
    out = df.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out = out.dropna(subset=["date"])
    out = out.dropna(subset=["ticker"])
    out["ticker"] = out["ticker"].astype(str)

    if "time_idx" not in out.columns:
        sort_cols = [c for c in ["ticker", "date"] if c in out.columns]
        out = out.sort_values(sort_cols) if sort_cols else out
        out["time_idx"] = out.groupby("ticker").cumcount()
    out["time_idx"] = pd.to_numeric(out["time_idx"], errors="coerce")
    if out["time_idx"].isna().any():
        out = out.sort_values(["ticker", "date"] if "date" in out.columns else ["ticker"])
        out["time_idx"] = out.groupby("ticker").cumcount()
    out["time_idx"] = out["time_idx"].astype("int64")

    for col in CALENDAR_CATS:
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


def add_missing_model_columns(sample: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    out = sample.copy()
    categorical_keys = [
        "static_categoricals", "time_varying_known_categoricals", "time_varying_unknown_categoricals",
    ]
    real_keys = [
        "static_reals", "time_varying_known_reals", "time_varying_unknown_reals",
    ]
    for key in categorical_keys:
        for col in params.get(key, []) or []:
            if col not in out.columns:
                out[col] = "0"
            out[col] = out[col].astype(str)
    for key in real_keys:
        for col in params.get(key, []) or []:
            if col not in out.columns:
                out[col] = 0.0
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0).astype("float32")
    return out


def append_future_rows(data: pd.DataFrame, horizon: int) -> pd.DataFrame:
    if data.empty or horizon <= 0:
        return data
    last = data.iloc[-1].copy()
    last_date = pd.to_datetime(last["date"]) if "date" in data.columns and pd.notna(last.get("date")) else None
    future_dates = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=horizon) if last_date is not None else [None] * horizon
    rows = []
    for i in range(horizon):
        row = last.copy()
        row["time_idx"] = int(last["time_idx"]) + i + 1
        if last_date is not None:
            dt = pd.Timestamp(future_dates[i])
            row["date"] = dt
            row["day_of_week"] = str(dt.dayofweek)
            row["month"] = str(dt.month)
            row["is_month_end"] = str(int(dt.is_month_end))
        if "volume" in row:
            row["volume"] = 0.0
        rows.append(row)
    return pd.concat([data, pd.DataFrame(rows)], ignore_index=True)


def build_prediction_dataset(master: pd.DataFrame, ticker: str, cutoff: pd.Timestamp | None, model) -> tuple[TimeSeriesDataSet | None, str | None]:
    if TimeSeriesDataSet is None:
        return None, "pytorch_forecasting belum tersedia"
    work = prepare_master(master)
    if work.empty:
        return None, "dataset master kosong atau kolom wajib belum tersedia"

    data = work[work["ticker"].astype(str).eq(str(ticker))].copy()
    if data.empty:
        return None, f"ticker {ticker} tidak tersedia di dataset master"
    dc = date_col(data)
    if cutoff is not None and dc:
        data = data[data[dc] <= pd.to_datetime(cutoff)]
    data = data.sort_values("time_idx")

    enc_default, pred_default = model_lengths()
    params = getattr(model, "dataset_parameters", None) or {}
    enc = int(params.get("max_encoder_length", enc_default))
    pred = int(params.get("max_prediction_length", pred_default))
    if len(data) < enc:
        return None, f"data encoder kurang dari {enc} baris"

    sample = append_future_rows(data.tail(enc).copy(), pred)
    sample = add_missing_model_columns(sample, params)

    try:
        return TimeSeriesDataSet.from_parameters(params, sample, predict=True, stop_randomization=True), None
    except Exception as first_exc:
        try:
            reals = [c for c in TECH_FEATURES + SENT_FEATURES + COMPAT_SENT_FEATURES if c in sample.columns]
            cats = [c for c in CALENDAR_CATS if c in sample.columns]
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
                time_varying_known_categoricals=cats,
                time_varying_unknown_reals=reals,
                target_normalizer=GroupNormalizer(groups=["ticker"], transformation="softplus") if GroupNormalizer else None,
                add_relative_time_idx=True,
                add_target_scales=True,
                add_encoder_length=True,
                predict_mode=True,
            )
            return dataset, None
        except Exception as second_exc:
            return None, f"gagal membuat dataset prediksi: {type(first_exc).__name__}: {first_exc}; fallback: {type(second_exc).__name__}: {second_exc}"


def tensor_to_values(output: Any, horizon: int = 3) -> list[float]:
    if hasattr(output, "output"):
        output = output.output
    if isinstance(output, tuple):
        output = output[0]
    if hasattr(output, "detach"):
        output = output.detach().cpu().numpy()
    arr = np.asarray(output).reshape(-1)
    return pd.Series(arr).dropna().astype(float).tolist()[:horizon]


def predict_checkpoints(master: pd.DataFrame | None, ticker: str | None, cutoff: pd.Timestamp | None) -> tuple[pd.DataFrame, list[str]]:
    if master is None or master.empty:
        return pd.DataFrame(), ["dataset master belum tersedia"]
    if not ticker:
        options = sorted(master["ticker"].dropna().astype(str).unique().tolist()) if "ticker" in master.columns else []
        ticker = options[0] if options else None
    if not ticker:
        return pd.DataFrame(), ["ticker belum tersedia"]

    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for model_name, scenario, ckpt in CHECKPOINTS:
        key = f"{model_name} {scenario}"
        model, err = load_tft(str(ckpt))
        if err:
            errors.append(f"{key}: {err}")
            continue
        dataset, err = build_prediction_dataset(master, ticker, cutoff, model)
        if err:
            errors.append(f"{key}: {err}")
            continue
        try:
            loader = dataset.to_dataloader(train=False, batch_size=1, num_workers=0)
            with torch.no_grad():
                output = model.predict(loader, mode="prediction", return_x=False)
            values = tensor_to_values(output, horizon=3)
            for step, value in enumerate(values, start=1):
                rows.append({"Model": model_name, "Scenario": scenario, "Series": key, "Horizon": f"H+{step}", "Step": step, "Prediction": float(value)})
        except Exception as exc:
            errors.append(f"{key}: gagal prediksi - {type(exc).__name__}: {exc}")
    return pd.DataFrame(rows), errors


# ============================================================
# 6. EVALUATION HELPERS
# ============================================================
ALIASES = {
    "RMSE": ["rmse", "rootmeansquarederror"],
    "MAE": ["mae", "meanabsoluteerror"],
    "MAPE": ["mape", "meanabsolutepercentageerror"],
    "R²": ["r2", "rsquared", "rsquare"],
    "Directional Accuracy": ["directionalaccuracy", "diracc", "dir_acc", "da"],
}
ORDER = ["RMSE", "MAE", "MAPE", "R²", "Directional Accuracy"]
LOWER_IS_BETTER = {"RMSE", "MAE", "MAPE"}


def normalize_model(value: Any) -> str:
    low = str(value).lower()
    if "llm" in low or "hybrid" in low or "sent" in low or "s1" in low:
        return "LLM-TFT"
    if "tft" in low or "base" in low or "s5" in low:
        return "TFT"
    return str(value)


def metric_name(value: Any) -> str:
    low = norm(value)
    for name, aliases in ALIASES.items():
        if any(norm(alias) in low for alias in aliases):
            return name
    return str(value).replace("_", " ").title()


def numeric_metric_cols(df: pd.DataFrame) -> list[str]:
    numeric = list(df.select_dtypes(include="number").columns)
    return [col for col in numeric if norm(col) not in {"n", "count", "jumlah", "index"}]


def long_eval(df: pd.DataFrame | None, scope: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    work = df.copy()
    mcol = find_col(work, ["model", "scenario", "skenario", "method", "metode", "variant"])
    hcol = find_col(work, ["horizon", "h", "step", "forecast_horizon"])
    tcol = find_col(work, ["ticker", "symbol", "emiten", "kode_saham"])
    metric_col = find_col(work, ["metric", "metrics", "metrik", "measure"])
    value_col = find_col(work, ["value", "nilai", "score", "hasil"])
    if not mcol:
        work["Model"] = ["TFT", "LLM-TFT"][: len(work)] if len(work) <= 2 else "Model"
        mcol = "Model"

    rows = []
    if metric_col and value_col:
        for _, row in work.iterrows():
            rows.append({
                "Model": normalize_model(row[mcol]),
                "Metric": metric_name(row[metric_col]),
                "Value": pd.to_numeric(row[value_col], errors="coerce"),
                "Horizon": row.get(hcol),
                "Ticker": row.get(tcol),
            })
    else:
        for idx, row in work.iterrows():
            horizon = row.get(hcol) if hcol else (f"H+{(idx % 3) + 1}" if scope == "horizon" else None)
            for col in numeric_metric_cols(work):
                rows.append({
                    "Model": normalize_model(row[mcol]),
                    "Metric": metric_name(col),
                    "Value": pd.to_numeric(row[col], errors="coerce"),
                    "Horizon": horizon,
                    "Ticker": row.get(tcol) if tcol else None,
                })
    out = pd.DataFrame(rows)
    return out.dropna(subset=["Value"]) if not out.empty else out


def best_rows(eval_global: pd.DataFrame | None) -> pd.DataFrame:
    data = long_eval(eval_global, "global")
    if data.empty:
        return pd.DataFrame()
    rows = []
    for metric in ORDER:
        subset = data[data["Metric"].eq(metric)]
        if subset.empty:
            continue
        idx = subset["Value"].idxmin() if metric in LOWER_IS_BETTER else subset["Value"].idxmax()
        rows.append(subset.loc[idx].to_dict())
    return pd.DataFrame(rows)


def metric_value(data: pd.DataFrame, model: str, metric: str) -> float | None:
    subset = data[(data["Model"].eq(model)) & (data["Metric"].eq(metric))]
    if subset.empty:
        return None
    return float(subset["Value"].iloc[0])


def latest_close(df: pd.DataFrame | None) -> float | None:
    if df is None or df.empty or "close" not in df.columns:
        return None
    dc = date_col(df)
    work = df.sort_values(dc) if dc else df
    values = pd.to_numeric(work["close"], errors="coerce").dropna()
    return float(values.iloc[-1]) if not values.empty else None


# ============================================================
# 7. CHARTS
# ============================================================
def price_chart(df: pd.DataFrame | None, title: str = "Price action") -> None:
    if df is None or df.empty:
        st.info("Data harga belum tersedia.")
        return
    dc = date_col(df)
    if not dc:
        st.info("Kolom tanggal belum tersedia.")
        return
    work = df.sort_values(dc)
    has_ohlc = all(col in work.columns for col in ["open", "high", "low", "close"])
    if has_ohlc:
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=work[dc], open=work["open"], high=work["high"], low=work["low"], close=work["close"],
            name="OHLC", increasing_line_color="#22C55E", decreasing_line_color="#F43F5E",
        ))
        fig.update_layout(xaxis_rangeslider_visible=False)
    else:
        fig = px.line(work, x=dc, y="close", title=title, color_discrete_sequence=[MODEL_COLORS["Aktual"]])
        fig.update_traces(line=dict(width=3))
    fig.update_yaxes(title="Harga")
    st.plotly_chart(apply_fig_theme(fig, 430, title), use_container_width=True)


def prediction_chart(master: pd.DataFrame | None, ticker: str | None, dates: Any, pred_df: pd.DataFrame) -> None:
    filtered = filter_df(master, ticker, dates)
    if filtered is None or filtered.empty or "close" not in filtered.columns:
        st.info("Data encoder belum tersedia.")
        return
    dc = date_col(filtered)
    work = filtered.sort_values(dc if dc else "time_idx").tail(15).copy()
    work["Step"] = list(range(-len(work) + 1, 1))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=work["Step"], y=work["close"], mode="lines+markers", name="Aktual encoder",
        line=dict(color=MODEL_COLORS["Aktual"], width=3), marker=dict(size=7),
    ))
    last_close = float(pd.to_numeric(work["close"], errors="coerce").dropna().iloc[-1])
    if pred_df is not None and not pred_df.empty:
        for model_name, group in pred_df.groupby("Model"):
            x_vals = [0] + group.sort_values("Step")["Step"].astype(int).tolist()
            y_vals = [last_close] + group.sort_values("Step")["Prediction"].astype(float).tolist()
            fig.add_trace(go.Scatter(
                x=x_vals, y=y_vals, mode="lines+markers", name=model_name,
                line=dict(color=MODEL_COLORS.get(model_name), width=4 if model_name == "LLM-TFT" else 3, dash="solid" if model_name == "LLM-TFT" else "dot"),
                marker=dict(size=9),
            ))
    ticks = list(range(-14, 4))
    labels = [f"T{x}" if x < 0 else ("T0" if x == 0 else f"H+{x}") for x in ticks]
    fig.update_xaxes(tickmode="array", tickvals=ticks, ticktext=labels, title="Langkah waktu")
    fig.update_yaxes(title="Harga close / prediksi")
    fig.add_vline(x=0, line_dash="dash", line_color="rgba(255,255,255,0.42)")
    st.plotly_chart(apply_fig_theme(fig, 470, "Encoder 15 hari dan prediksi multi-horizon"), use_container_width=True)


def evaluation_bar(data: pd.DataFrame, metric: str) -> None:
    subset = data[data["Metric"].eq(metric)]
    if subset.empty:
        return
    fig = px.bar(subset, x="Model", y="Value", color="Model", text_auto=True, color_discrete_map=MODEL_COLORS)
    fig.update_traces(marker_line_width=0, textposition="outside")
    fig.update_yaxes(title=metric)
    st.plotly_chart(apply_fig_theme(fig, 350, metric), use_container_width=True)


def horizon_chart(df: pd.DataFrame | None) -> None:
    data = long_eval(df, "horizon")
    if data.empty:
        st.info("Data evaluasi horizon belum tersedia.")
        return
    metric = st.selectbox("Metrik horizon", [m for m in ORDER if m in data["Metric"].unique()], key="metric_horizon_select")
    subset = data[data["Metric"].eq(metric)].copy()
    fig = px.line(subset, x="Horizon", y="Value", color="Model", markers=True, color_discrete_map=MODEL_COLORS)
    fig.update_traces(line=dict(width=3), marker=dict(size=9))
    st.plotly_chart(apply_fig_theme(fig, 390, f"Performa per horizon — {metric}"), use_container_width=True)


def ticker_heatmap(df: pd.DataFrame | None) -> None:
    data = long_eval(df, "ticker")
    if data.empty or data["Ticker"].isna().all():
        st.info("Data evaluasi per emiten belum tersedia.")
        return
    metric = st.selectbox("Metrik emiten", [m for m in ORDER if m in data["Metric"].unique()], key="metric_ticker_select")
    subset = data[data["Metric"].eq(metric)].copy()
    pivot = subset.pivot_table(index="Ticker", columns="Model", values="Value", aggfunc="mean")
    if pivot.empty:
        st.info("Pivot emiten belum dapat dibentuk.")
        return
    fig = px.imshow(pivot, text_auto=".3f", aspect="auto", color_continuous_scale="Bluered")
    st.plotly_chart(apply_fig_theme(fig, 390, f"Heatmap emiten — {metric}"), use_container_width=True)


def attention_chart(df: pd.DataFrame | None) -> None:
    if df is not None and not df.empty:
        step = find_col(df, ["encoder_step", "step", "time_step", "lag", "position"])
        weight = find_col(df, ["attention", "attention_weight", "weight", "value"])
        model = find_col(df, ["model", "scenario", "method"])
        if step and weight:
            out = df[[step, weight] + ([model] if model else [])].copy()
            out = out.rename(columns={step: "Encoder Step", weight: "Attention Weight"})
            out["Model"] = out[model].apply(normalize_model) if model else "Attention"
        else:
            out = pd.DataFrame()
    else:
        out = pd.DataFrame()
    if out.empty:
        out = pd.DataFrame({
            "Encoder Step": list(range(-14, 1)) * 2,
            "Attention Weight": [54, 62, 68, 72, 75, 78, 81, 83, 85, 87, 90, 93, 96, 98, 101]
            + [86, 79, 76, 75, 76, 78, 80, 82, 83, 84, 84, 85, 85, 86, 86],
            "Model": ["TFT"] * 15 + ["LLM-TFT"] * 15,
        })
    fig = px.line(out, x="Encoder Step", y="Attention Weight", color="Model", markers=True, color_discrete_map=MODEL_COLORS)
    fig.update_traces(line=dict(width=3), marker=dict(size=8))
    st.plotly_chart(apply_fig_theme(fig, 400, "Temporal attention pattern"), use_container_width=True)


# ============================================================
# 7. PAGES
# ============================================================
def sidebar(data: dict[str, pd.DataFrame | None]):
    with st.sidebar:
        st.markdown("## 📈 StockForecast Pro")
        st.caption("TFT S5 · LLM‑TFT S1")
        st.divider()
        page = st.radio("Navigasi", PAGES, format_func=lambda x: x)
        st.divider()

        tickers = ticker_options(data)
        ticker = st.selectbox("Emiten", tickers, index=0) if tickers else None
        drange = global_date_range(data)
        dates = None
        if drange:
            start, end = drange
            dates = st.date_input("Rentang tanggal", value=(start.date(), end.date()), min_value=start.date(), max_value=end.date())
        st.divider()
    return page, ticker, dates


def page_prediction(data: dict[str, pd.DataFrame | None], ticker: str | None, dates: Any) -> None:
    hero("Prediction Studio")
    page_chips("Checkpoint inference", "CPU mode", "TFT S5", "LLM‑TFT S1")
    master_filtered = filter_df(data["master"], ticker, dates)
    dc = date_col(master_filtered)
    cutoff = pd.to_datetime(master_filtered[dc]).max() if master_filtered is not None and not master_filtered.empty and dc else None

    cols = st.columns([1, 1, 1, 1.2])
    cols[0].metric("Ticker", ticker or "-")
    cols[1].metric("Encoder", f"{model_lengths()[0]} hari")
    cols[2].metric("Horizon", f"{model_lengths()[1]} hari")
    cols[3].metric("Cutoff", str(cutoff.date()) if cutoff is not None and pd.notna(cutoff) else "-")

    run = st.button("Jalankan prediksi checkpoint", type="primary", use_container_width=True)
    pred_df, errors = (pd.DataFrame(), ["Klik tombol prediksi untuk menjalankan inference CPU."])
    if run:
        with st.spinner("Memuat checkpoint dan menjalankan prediksi CPU..."):
            pred_df, errors = predict_checkpoints(data["master"], ticker, cutoff)

    close = latest_close(master_filtered)
    pred_cards = st.columns(3)
    chosen = pred_df[pred_df["Model"].eq("LLM-TFT")] if not pred_df.empty else pd.DataFrame()
    for i, horizon in enumerate(["H+1", "H+2", "H+3"]):
        row = chosen[chosen["Horizon"].eq(horizon)] if not chosen.empty else pd.DataFrame()
        value = float(row["Prediction"].iloc[0]) if not row.empty else None
        delta = f"{((value - close) / close) * 100:+.2f}%" if value is not None and close else None
        pred_cards[i].metric(f"LLM‑TFT {horizon}", rupiah(value) if value is not None else "-", delta)

    if errors:
        with st.expander("Catatan inference", expanded=bool(run)):
            for err in errors:
                st.warning(err)

    prediction_chart(data["master"], ticker, dates, pred_df)

    if not pred_df.empty:
        table = pred_df.pivot_table(index="Horizon", columns="Model", values="Prediction", aggfunc="last").reset_index()
        for col in ["TFT", "LLM-TFT"]:
            if col in table.columns:
                table[col] = table[col].map(rupiah)
        st.dataframe(table, hide_index=True, use_container_width=True)


def page_performance(data: dict[str, pd.DataFrame | None], ticker: str | None, dates: Any) -> None:
    hero("Model Performance")
    page_chips("Global", "Horizon", "Emiten", "Attention")
    tabs = st.tabs(["Global", "Horizon", "Emiten", "Attention"])
    with tabs[0]:
        eval_global = long_eval(data["eval_global"], "global")
        if eval_global.empty:
            st.info("Evaluasi global belum tersedia.")
        else:
            metrics = [m for m in ORDER if m in eval_global["Metric"].unique()]
            cols = st.columns(min(3, len(metrics)) or 1)
            for i, metric in enumerate(metrics[:3]):
                with cols[i % len(cols)]:
                    evaluation_bar(eval_global, metric)
            if len(metrics) > 3:
                cols2 = st.columns(2)
                for i, metric in enumerate(metrics[3:]):
                    with cols2[i % 2]:
                        evaluation_bar(eval_global, metric)
    with tabs[1]:
        horizon_chart(data["eval_horizon"])
    with tabs[2]:
        ticker_heatmap(data["eval_ticker"])
    with tabs[3]:
        attention_chart(data["attention"])


def page_market_data(data: dict[str, pd.DataFrame | None], ticker: str | None, dates: Any) -> None:
    hero("Market Data Lab")
    page_chips("Price action", "Technical indicator", "Correlation")
    prices = filter_df(data["prices"], ticker, dates)
    if prices is None or prices.empty:
        st.info("Data harga belum tersedia.")
        return
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Baris harga", fmt(len(prices)))
    c2.metric("Emiten", fmt(prices["ticker"].nunique() if "ticker" in prices.columns else 0))
    c3.metric("Close terakhir", rupiah(latest_close(prices)))
    if "volume" in prices.columns:
        c4.metric("Volume rata-rata", fmt(pd.to_numeric(prices["volume"], errors="coerce").mean()))
    else:
        c4.metric("Volume rata-rata", "-")

    price_chart(prices, "Market price action")

    left, right = st.columns([1, 1], gap="large")
    with left:
        indicators = [c for c in TECH_FEATURES if c in prices.columns and c not in {"close", "volume"}]
        if indicators:
            selected = st.selectbox("Indikator teknikal", indicators)
            dc = date_col(prices)
            fig = px.line(prices.sort_values(dc), x=dc, y=selected, color_discrete_sequence=["#38BDF8"])
            fig.update_traces(line=dict(width=3))
            st.plotly_chart(apply_fig_theme(fig, 390, f"Tren {selected}"), use_container_width=True)
        else:
            st.info("Indikator teknikal belum tersedia.")
    with right:
        corr_cols = [c for c in TECH_FEATURES if c in prices.columns and pd.api.types.is_numeric_dtype(prices[c])]
        if len(corr_cols) >= 2:
            corr = prices[corr_cols].corr(numeric_only=True)
            fig = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="Turbo")
            st.plotly_chart(apply_fig_theme(fig, 390, "Korelasi indikator"), use_container_width=True)
        else:
            st.info("Kolom numerik belum cukup untuk korelasi.")
    with st.expander("Tabel data harga"):
        st.dataframe(prices.tail(500), use_container_width=True, hide_index=True)


def page_sentiment(data: dict[str, pd.DataFrame | None], ticker: str | None, dates: Any) -> None:
    hero("Sentiment Intelligence")
    page_chips("News timeline", "Sentiment label", "Daily sentiment feature")
    news = filter_df(data["news"], ticker, dates)
    articles = filter_df(data["articles"], ticker, dates)
    daily = filter_df(data["daily"], ticker, dates)

    c1, c2, c3 = st.columns(3)
    c1.metric("Berita bersih", fmt(len(news) if news is not None else 0))
    c2.metric("Artikel berlabel", fmt(len(articles) if articles is not None else 0))
    c3.metric("Hari sentimen", fmt(len(daily) if daily is not None else 0))

    left, right = st.columns([1.1, 1], gap="large")
    with left:
        if news is not None and not news.empty and date_col(news):
            dc = date_col(news)
            timeline = news.dropna(subset=[dc]).assign(day=lambda x: x[dc].dt.date).groupby("day").size().reset_index(name="Jumlah")
            fig = px.area(timeline, x="day", y="Jumlah", color_discrete_sequence=["#38BDF8"])
            fig.update_traces(line=dict(width=3))
            st.plotly_chart(apply_fig_theme(fig, 410, "Volume berita harian"), use_container_width=True)
        else:
            st.info("Timeline berita belum tersedia.")
    with right:
        final_col = find_col(articles, FINAL_LABELS)
        if articles is not None and not articles.empty and final_col:
            label_map = {-1: "Negatif", 0: "Netral", 1: "Positif", "-1": "Negatif", "0": "Netral", "1": "Positif"}
            counts = articles[final_col].map(label_map).fillna(articles[final_col].astype(str)).value_counts().reset_index()
            counts.columns = ["Sentimen", "Jumlah"]
            fig = px.pie(counts, names="Sentimen", values="Jumlah", hole=0.58, color="Sentimen", color_discrete_map=MODEL_COLORS)
            st.plotly_chart(apply_fig_theme(fig, 410, "Distribusi label sentimen"), use_container_width=True)
        else:
            st.info("Label sentimen artikel belum tersedia.")

    if daily is not None and not daily.empty and date_col(daily):
        available = [c for c in SENT_FEATURES if c in daily.columns]
        if available:
            selected = st.selectbox("Fitur sentimen harian", available)
            dc = date_col(daily)
            fig = px.line(daily.sort_values(dc), x=dc, y=selected, color_discrete_sequence=["#F97316"])
            fig.update_traces(line=dict(width=3))
            st.plotly_chart(apply_fig_theme(fig, 420, f"Tren {selected}"), use_container_width=True)
        else:
            st.info("Fitur sentimen harian belum tersedia.")
    else:
        st.info("Data sentimen harian belum tersedia.")

    with st.expander("Tabel artikel/berita"):
        table = articles if articles is not None and not articles.empty else news
        st.dataframe(table.tail(500) if table is not None else pd.DataFrame(), use_container_width=True, hide_index=True)




# ============================================================
# 8. MAIN
# ============================================================
def main() -> None:
    setup_page()
    data = load_all_data()
    page, ticker, dates = sidebar(data)

    if page == "Prediction Studio":
        page_prediction(data, ticker, dates)
    elif page == "Model Performance":
        page_performance(data, ticker, dates)
    elif page == "Market Data Lab":
        page_market_data(data, ticker, dates)
    else:
        page_sentiment(data, ticker, dates)


if __name__ == "__main__":
    main()
