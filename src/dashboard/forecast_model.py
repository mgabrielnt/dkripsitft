import pandas as pd
import streamlit as st
from config import CHECKPOINTS, CONFIG_PATH, TECH_FEATURES, SENT_FEATURES
from utils import date_col

try:
    import torch, yaml
    from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
    from pytorch_forecasting.data import GroupNormalizer
except Exception:
    torch = yaml = TemporalFusionTransformer = TimeSeriesDataSet = GroupNormalizer = None

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
    out = df.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
    if "time_idx" not in out.columns:
        out = out.sort_values(["ticker", "date"] if "date" in out.columns else ["ticker"])
        out["time_idx"] = out.groupby("ticker").cumcount()
    for col in ["ticker", "month", "day_of_week", "is_month_end"]:
        if col in out.columns:
            out[col] = out[col].astype(str)
    return out

def make_dataset(df, ticker, cutoff, model_name):
    if TimeSeriesDataSet is None or GroupNormalizer is None:
        return None
    enc, pred = model_config()
    work = prep_master(df)
    ticker = sorted(work["ticker"].dropna().unique())[0] if ticker == "Semua" else ticker
    data = work[work["ticker"].eq(ticker)].copy()
    dc = date_col(data)
    if cutoff is not None and dc:
        data = data[data[dc] <= pd.to_datetime(cutoff)]
    data = data.sort_values("time_idx")
    if len(data) < enc + pred:
        return None
    reals = [c for c in TECH_FEATURES if c in data.columns]
    if model_name == "LLM-TFT":
        reals += [c for c in SENT_FEATURES if c in data.columns]
    cats = [c for c in ["day_of_week", "month", "is_month_end"] if c in data.columns]
    data = data.tail(enc + pred + 10)
    return TimeSeriesDataSet(
        data, time_idx="time_idx", target="close", group_ids=["ticker"],
        min_encoder_length=enc, max_encoder_length=enc, max_prediction_length=pred,
        static_categoricals=["ticker"], time_varying_known_categoricals=cats,
        time_varying_unknown_reals=reals,
        target_normalizer=GroupNormalizer(groups=["ticker"], transformation="softplus"),
        add_relative_time_idx=True, add_target_scales=True, add_encoder_length=True,
        predict_mode=True,
    )
