import pandas as pd
import streamlit as st
from config import CONFIG_PATH, TECH_FEATURES, SENT_FEATURES
from utils import date_col

try:
    import torch, yaml
    from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
except Exception:
    torch = yaml = TemporalFusionTransformer = TimeSeriesDataSet = None

@st.cache_resource(show_spinner=False)
def load_tft(path):
    if TemporalFusionTransformer is None or not path.exists():
        return None
    return TemporalFusionTransformer.load_from_checkpoint(str(path), map_location=torch.device("cpu"), weights_only=False)

@st.cache_data(show_spinner=False)
def model_config():
    if yaml is None or not CONFIG_PATH.exists():
        return 15, 3
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as file:
            cfg = yaml.safe_load(file) or {}
        data_cfg = cfg.get("data", cfg)
        return int(data_cfg.get("max_encoder_length", 15)), int(data_cfg.get("max_prediction_length", 3))
    except Exception:
        return 15, 3

def prep_master(df):
    if df is None or df.empty or "ticker" not in df.columns or "close" not in df.columns:
        return pd.DataFrame()
    out = df.copy()
    out["ticker"] = out["ticker"].astype(str).str.upper()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
    if "time_idx" not in out.columns:
        out = out.sort_values([c for c in ["ticker", "date"] if c in out.columns])
        out["time_idx"] = out.groupby("ticker").cumcount().astype("int64")
    for col in ["ticker", "day_of_week", "month", "is_month_end"]:
        if col in out.columns:
            out[col] = out[col].astype(str)
    return out

def make_dataset(df, ticker, cutoff, model_name):
    if TimeSeriesDataSet is None:
        return None
    enc, pred = model_config()
    work = prep_master(df)
    if work.empty:
        return None
    selected = str(ticker).upper() if ticker else sorted(work["ticker"].dropna().unique())[0]
    data = work[work["ticker"].eq(selected)].copy()
    dc = date_col(data)
    if cutoff is not None and dc:
        data = data[data[dc] <= pd.to_datetime(cutoff)]
    data = data.sort_values("time_idx")
    if len(data) < enc + pred:
        return None
    features = [c for c in TECH_FEATURES if c in data.columns]
    if model_name == "LLM-TFT":
        features += [c for c in SENT_FEATURES if c in data.columns]
    return build_dataset(data.tail(enc + pred + 10), enc, pred, features)

def build_dataset(data, enc, pred, features):
    cats = [c for c in ["day_of_week", "month", "is_month_end"] if c in data.columns]
    for col in features + ["close"]:
        data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0.0).astype("float32")
    return TimeSeriesDataSet(
        data, time_idx="time_idx", target="close", group_ids=["ticker"],
        min_encoder_length=enc, max_encoder_length=enc,
        min_prediction_length=pred, max_prediction_length=pred,
        static_categoricals=["ticker"], time_varying_known_categoricals=cats,
        time_varying_unknown_reals=features,
        add_relative_time_idx=True, add_target_scales=True, add_encoder_length=True,
        allow_missing_timesteps=False, predict_mode=True,
    )
