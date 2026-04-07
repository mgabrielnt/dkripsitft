import os
import numpy as np
import pandas as pd
import yaml

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_INTERIM_DIR = os.path.join(ROOT_DIR, "data", "interim")
DATA_PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")
CONFIG_DATA_PATH = os.path.join(ROOT_DIR, "configs", "data.yaml")
PRICES_PATH = os.path.join(DATA_INTERIM_DIR, "prices_with_indicators.csv")
SENTIMENT_PATH = os.path.join(DATA_PROCESSED_DIR, "daily_sentiment.csv")
OUT_PATH = os.path.join(DATA_PROCESSED_DIR, "tft_master.csv")
KEEP_COLS = [
    "time_idx", "date", "ticker", "day_of_week", "month", "is_month_end", "split", "close",
    "volume", "log_return_1d", "log_return_2d", "vol_20", "rsi_14", "ma_5_div_ma_20",
    "bb_width_20", "gap_return_1d", "intraday_range_pct", "news_count_3d",
    "sentiment_final_mean", "sentiment_delta_1d", "sentiment_mean_3d",
    "sentiment_ema_7d", "sentiment_trend_7d", "sentiment_dir_signal",
]
FILL_ZERO = [c for c in KEEP_COLS if c not in {"time_idx", "date", "ticker", "day_of_week", "month", "is_month_end", "split"}]


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_allowed_tickers(cfg: dict) -> set[str]:
    return {
        str(t).strip().upper()
        for t in cfg.get("tickers", [])
        if str(t).strip()
    }


def parse_end_date(raw):
    if raw is None:
        return None
    dt = pd.to_datetime(raw, errors='coerce')
    return None if pd.isna(dt) else dt.normalize()


def validate_cfg(cfg: dict):
    train_ratio = float(cfg.get("train_ratio", 0.7))
    val_ratio = float(cfg.get("val_ratio", 0.15))
    test_ratio = float(cfg.get("test_ratio", 0.15))
    freq = str(cfg.get("freq", "B")).upper()
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-8 or min(train_ratio, val_ratio, test_ratio) <= 0:
        raise ValueError("train_ratio, val_ratio, dan test_ratio harus > 0 dan total = 1.")
    if freq != "B":
        raise ValueError("Script ini hanya mendukung freq='B'.")
    return train_ratio, val_ratio, test_ratio, float(cfg.get("sentiment_gate_threshold", 0.5))


def add_minimal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["ticker", "date"]).copy()
    close = pd.to_numeric(df["close"], errors="coerce").replace(0, np.nan)
    sent = pd.to_numeric(df.get("sentiment_final_mean", 0), errors="coerce").fillna(0.0)
    if "log_return_1d" not in df.columns:
        df["log_return_1d"] = np.log(close / close.groupby(df["ticker"]).shift(1))
    df["log_return_2d"] = np.log(close / close.groupby(df["ticker"]).shift(2))
    df["sentiment_delta_1d"] = sent.groupby(df["ticker"]).diff()
    raw = pd.to_numeric(df["sentiment_mean_3d"], errors="coerce").fillna(0.0) * np.log1p(pd.to_numeric(df["volume"], errors="coerce").fillna(0.0))
    df["sentiment_dir_signal"] = 0.0
    df.loc[raw > 0, "sentiment_dir_signal"] = 1.0
    df.loc[raw < 0, "sentiment_dir_signal"] = -1.0
    return df


def add_split_and_calendar(df: pd.DataFrame, train_ratio: float, val_ratio: float):
    df = df.sort_values(["ticker", "date"]).copy()
    df["day_of_week"] = df["date"].dt.weekday.astype(str)
    df["month"] = df["date"].dt.month.astype(str)
    df["is_month_end"] = df["date"].dt.is_month_end.astype(int).astype(str)
    dates = sorted(df["date"].dropna().unique())
    if len(dates) < 10:
        raise ValueError("Jumlah tanggal terlalu sedikit untuk split yang stabil.")
    df["time_idx"] = df["date"].map({d: i for i, d in enumerate(dates)}).astype("int64")
    i1, i2 = int(len(dates) * train_ratio), int(len(dates) * (train_ratio + val_ratio))
    i1, i2 = min(max(i1, 1), len(dates) - 2), min(max(i2, i1 + 1), len(dates) - 1)
    d1, d2 = dates[i1 - 1], dates[i2 - 1]
    df["split"] = "test"
    df.loc[df["date"] <= d1, "split"] = "train"
    df.loc[(df["date"] > d1) & (df["date"] <= d2), "split"] = "val"
    return df


def main():
    cfg = load_yaml(CONFIG_DATA_PATH)
    train_ratio, val_ratio, _, gate = validate_cfg(cfg)
    allowed_tickers = get_allowed_tickers(cfg)

    prices = pd.read_csv(PRICES_PATH, parse_dates=["date"])
    senti = pd.read_csv(SENTIMENT_PATH, parse_dates=["date"])

    prices["ticker"] = prices["ticker"].astype(str).str.upper()
    senti["ticker"] = senti["ticker"].astype(str).str.upper()

    if allowed_tickers:
        prices = prices[prices["ticker"].isin(allowed_tickers)].copy()
        senti = senti[senti["ticker"].isin(allowed_tickers)].copy()

    end_date = parse_end_date(cfg.get('end_date'))
    if end_date is not None:
        prices = prices[prices['date'] <= end_date].copy()
        senti = senti[senti['date'] <= end_date].copy()

    prices["volume"] = pd.to_numeric(prices["volume"], errors="coerce").fillna(0)
    prices = prices[prices["volume"] > 0].copy()

    df = prices.merge(senti, on=["ticker", "date"], how="left").sort_values(["ticker", "date"])
    for c in [x for x in senti.columns if x not in {"ticker", "date"}]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    df = add_minimal_features(df)
    raw = pd.to_numeric(df["sentiment_mean_3d"], errors="coerce").fillna(0.0) * np.log1p(pd.to_numeric(df["volume"], errors="coerce").fillna(0.0))
    df["sentiment_dir_signal"] = np.where(raw.abs() > gate, np.sign(raw), 0.0)
    df = add_split_and_calendar(df, train_ratio, val_ratio)

    for c in FILL_ZERO:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

    result = df[[c for c in KEEP_COLS if c in df.columns]].copy()

    if allowed_tickers:
        result = result[result["ticker"].isin(allowed_tickers)].copy()

    forbidden = {"BBCA.JK", "UNVR.JK"} & set(result["ticker"].astype(str).str.upper().unique())
    if forbidden:
        raise ValueError(f"Ticker terlarang masih ada di tft_master.csv: {sorted(forbidden)}")

    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
    result.to_csv(OUT_PATH, index=False)
    print(f"saved -> {OUT_PATH}")
    print(result.columns.tolist())
    print(result["split"].value_counts().to_dict())


if __name__ == "__main__":
    main()
