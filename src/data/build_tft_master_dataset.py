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
    "time_idx", "date", "ticker", "day_of_week", "month", "is_month_end", "split",
    "close", "volume",
    "log_return_1d", "log_return_2d", "vol_20", "rsi_14", "ma_5_div_ma_20",
    "bb_width_20", "gap_return_1d", "intraday_range_pct",
    "news_count_3d", "sentiment_final_mean", "sentiment_mean_3d",
    "sentiment_ema_7d", "sentiment_trend_7d", "sentiment_delta_1d",
    "sentiment_dir_signal",
]

STRUCT_COLS = {"time_idx", "date", "ticker", "day_of_week", "month", "is_month_end", "split"}
FILL_ZERO = [c for c in KEEP_COLS if c not in STRUCT_COLS]
FORBIDDEN_TICKERS = {"BBCA.JK", "UNVR.JK"}


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_allowed_tickers(cfg: dict) -> set[str]:
    return {str(t).strip().upper() for t in cfg.get("tickers", []) if str(t).strip()}


def parse_end_date(raw):
    if raw is None:
        return None
    dt = pd.to_datetime(raw, errors="coerce")
    return None if pd.isna(dt) else dt.normalize()


def validate_cfg(cfg: dict):
    train_ratio = float(cfg.get("train_ratio", 0.7))
    val_ratio = float(cfg.get("val_ratio", 0.15))
    test_ratio = float(cfg.get("test_ratio", 0.15))
    freq = str(cfg.get("freq", "B")).upper()

    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-8:
        raise ValueError("train_ratio, val_ratio, dan test_ratio harus berjumlah 1.0.")
    if min(train_ratio, val_ratio, test_ratio) <= 0:
        raise ValueError("train_ratio, val_ratio, dan test_ratio harus > 0.")
    if freq != "B":
        raise ValueError("Script ini hanya mendukung freq='B'.")

    return train_ratio, val_ratio, test_ratio


def ensure_columns(df: pd.DataFrame, cols: list[str], ctx: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{ctx} kolom wajib hilang: {missing}")


def add_split_and_calendar(df: pd.DataFrame, train_ratio: float, val_ratio: float) -> pd.DataFrame:
    df = df.sort_values(["ticker", "date"]).copy()

    df["day_of_week"] = df["date"].dt.weekday.astype(str)
    df["month"] = df["date"].dt.month.astype(str)
    df["is_month_end"] = df["date"].dt.is_month_end.astype(int).astype(str)

    unique_dates = pd.Index(sorted(df["date"].dropna().unique()))
    if len(unique_dates) < 10:
        raise ValueError("Jumlah tanggal terlalu sedikit untuk split yang stabil.")

    i1 = int(len(unique_dates) * train_ratio)
    i2 = int(len(unique_dates) * (train_ratio + val_ratio))

    i1 = min(max(i1, 1), len(unique_dates) - 2)
    i2 = min(max(i2, i1 + 1), len(unique_dates) - 1)

    d1 = unique_dates[i1 - 1]
    d2 = unique_dates[i2 - 1]

    df["split"] = "test"
    df.loc[df["date"] <= d1, "split"] = "train"
    df.loc[(df["date"] > d1) & (df["date"] <= d2), "split"] = "val"

    # time_idx rapat per ticker
    df["time_idx"] = df.groupby("ticker").cumcount().astype("int64")
    return df


def audit_result(df: pd.DataFrame):
    problems = []
    for ticker, g in df.sort_values(["ticker", "date"]).groupby("ticker", sort=True):
        diff = g["time_idx"].diff().dropna()
        bad = diff[diff != 1]
        if not bad.empty:
            problems.append((ticker, int(len(bad)), float(bad.max())))
    if problems:
        raise ValueError(f"time_idx belum rapat per ticker: {problems}")


def main():
    cfg = load_yaml(CONFIG_DATA_PATH)
    train_ratio, val_ratio, _ = validate_cfg(cfg)
    allowed_tickers = get_allowed_tickers(cfg)
    end_date = parse_end_date(cfg.get("end_date"))

    prices = pd.read_csv(PRICES_PATH, parse_dates=["date"])
    senti = pd.read_csv(SENTIMENT_PATH, parse_dates=["date"])

    ensure_columns(
        prices,
        [
            "ticker", "date", "close", "volume",
            "log_return_1d", "log_return_2d",
            "vol_20", "rsi_14", "ma_5_div_ma_20",
            "bb_width_20", "gap_return_1d", "intraday_range_pct",
        ],
        "[PRICES]",
    )

    ensure_columns(
        senti,
        [
            "ticker", "date",
            "sentiment_final_mean", "news_count_3d", "sentiment_mean_3d",
            "sentiment_ema_7d", "sentiment_trend_7d",
            "sentiment_delta_1d", "sentiment_dir_signal",
        ],
        "[SENTIMENT]",
    )

    prices["ticker"] = prices["ticker"].astype(str).str.upper()
    senti["ticker"] = senti["ticker"].astype(str).str.upper()
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
    senti["date"] = pd.to_datetime(senti["date"], errors="coerce")

    prices = prices.dropna(subset=["ticker", "date"]).copy()
    senti = senti.dropna(subset=["ticker", "date"]).copy()

    if allowed_tickers:
        prices = prices[prices["ticker"].isin(allowed_tickers)].copy()
        senti = senti[senti["ticker"].isin(allowed_tickers)].copy()

    if end_date is not None:
        prices = prices[prices["date"] <= end_date].copy()
        senti = senti[senti["date"] <= end_date].copy()

    prices["volume"] = pd.to_numeric(prices["volume"], errors="coerce").fillna(0)
    prices = prices[prices["volume"] > 0].copy()

    prices = (
        prices.sort_values(["ticker", "date"])
        .drop_duplicates(["ticker", "date"], keep="last")
        .copy()
    )
    senti = (
        senti.sort_values(["ticker", "date"])
        .drop_duplicates(["ticker", "date"], keep="last")
        .copy()
    )

    df = prices.merge(senti, on=["ticker", "date"], how="left").sort_values(["ticker", "date"]).copy()

    # Di master, fitur sentimen tidak dihitung ulang.
    # Semua fitur yang berasal dari agregat hanya dipaksa numerik lalu diisi 0 jika tanggal itu tidak punya berita.
    for col in [c for c in senti.columns if c not in {"ticker", "date"}]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df = add_split_and_calendar(df, train_ratio, val_ratio)

    for col in FILL_ZERO:
        df[col] = (
            pd.to_numeric(df[col], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )

    result = df[KEEP_COLS].copy()

    if allowed_tickers:
        result = result[result["ticker"].isin(allowed_tickers)].copy()

    forbidden = FORBIDDEN_TICKERS & set(result["ticker"].astype(str).str.upper().unique())
    if forbidden:
        raise ValueError(f"Ticker terlarang masih ada di tft_master.csv: {sorted(forbidden)}")

    audit_result(result)

    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
    result.to_csv(OUT_PATH, index=False)

    print(f"saved -> {OUT_PATH}")
    print(result.columns.tolist())
    print(result["split"].value_counts().to_dict())


if __name__ == "__main__":
    main()