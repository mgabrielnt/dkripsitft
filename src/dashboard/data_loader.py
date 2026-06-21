import pandas as pd
from config import *
from utils import norm, DROP_NAMES

def first_existing(paths):
    for path in paths:
        if path.exists():
            return path
    return None

def read_csv(path):
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

def load_data():
    return {
        "prices": read_csv(first_existing(PRICE_PATHS)),
        "news": read_csv(first_existing(NEWS_PATHS)),
        "articles": read_csv(first_existing(ARTICLE_PATHS)),
        "daily": read_csv(first_existing(DAILY_SENTIMENT_PATHS)),
        "master": read_csv(first_existing(MASTER_PATHS)),
        "eval_global": read_csv(first_existing(EVAL_GLOBAL_PATHS)),
        "eval_ticker": read_csv(first_existing(EVAL_TICKER_PATHS)),
        "eval_horizon": read_csv(first_existing(EVAL_HORIZON_PATHS)),
        "attention": read_csv(first_existing(ATTENTION_PATHS)),
        "predictions": read_csv(first_existing(PREDICTION_PATHS)),
    }
