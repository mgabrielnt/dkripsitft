import os
from typing import Dict, Any

import pandas as pd
import yaml

# Path dasar
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_INTERIM_DIR = os.path.join(ROOT_DIR, "data", "interim")
DATA_PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")
CONFIG_DATA_PATH = os.path.join(ROOT_DIR, "configs", "data.yaml")

PRICES_PATH = os.path.join(DATA_INTERIM_DIR, "prices_with_indicators.csv")
SENTIMENT_PATH = os.path.join(DATA_PROCESSED_DIR, "daily_sentiment.csv")
OUT_PATH = os.path.join(DATA_PROCESSED_DIR, "tft_master.csv")


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    if not os.path.exists(PRICES_PATH):
        raise FileNotFoundError(f"Tidak ditemukan file harga: {PRICES_PATH}")

    if not os.path.exists(SENTIMENT_PATH):
        raise FileNotFoundError(f"Tidak ditemukan file sentimen harian: {SENTIMENT_PATH}")

    data_cfg = load_yaml(CONFIG_DATA_PATH)
    tickers_cfg = data_cfg.get("tickers", None)
    train_ratio = float(data_cfg.get("train_ratio", 0.6))
    val_ratio = float(data_cfg.get("val_ratio", 0.2))
    # sisanya otomatis jadi test

    print(f"[INFO] Loading prices from {PRICES_PATH}")
    df_p = pd.read_csv(PRICES_PATH, parse_dates=["date"])

    print(f"[INFO] Loading daily sentiment from {SENTIMENT_PATH}")
    df_s = pd.read_csv(SENTIMENT_PATH, parse_dates=["date"])

    # Filter ticker sesuai config (kalau ada)
    if tickers_cfg:
        df_p = df_p[df_p["ticker"].isin(tickers_cfg)].copy()
        df_s = df_s[df_s["ticker"].isin(tickers_cfg)].copy()

    # Fitur kalender dasar
    df_p["day_of_week"] = df_p["date"].dt.weekday
    df_p["month"] = df_p["date"].dt.month
    df_p["is_month_end"] = df_p["date"].dt.is_month_end.astype(int)

    # Merge left: semua hari harga tetap ada meski tidak ada berita
    df = pd.merge(
        df_p,
        df_s,
        on=["ticker", "date"],
        how="left",
        suffixes=("", "_sent"),
    )

    # ==== Isi NaN khusus fitur sentimen ====
    sentiment_cols = [
        "sentiment_mean",
        "sentiment_max",
        "sentiment_min",
        "sentiment_mean_3d",
        "sentiment_shock",
    ]
    count_cols = [
        "news_count",
        "pos_count",
        "neg_count",
        "neu_count",
        "news_count_3d",
    ]
    binary_cols = ["has_news", "extreme_news"]

    for col in sentiment_cols:
        if col in df.columns:
            df[col] = df[col].astype(float).fillna(0.0)

    for col in count_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)

    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)

    # time_idx global berbasis tanggal
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)
    unique_dates = df["date"].drop_duplicates().sort_values()
    date_to_idx = {d: i for i, d in enumerate(unique_dates)}
    df["time_idx"] = df["date"].map(date_to_idx).astype("int64")

    # Kolom urutan rapih (opsional)
    base_cols = [
        "time_idx",
        "date",
        "ticker",
        "day_of_week",
        "month",
        "is_month_end",
    ]
    # sisanya dibiarkan di belakang
    other_cols = [c for c in df.columns if c not in base_cols]
    df = df[base_cols + other_cols]

    # ==== Bikin split train / val / test ====
    unique_idx = sorted(df["time_idx"].unique())
    n = len(unique_idx)

    train_end = unique_idx[int(n * train_ratio) - 1]
    val_end = unique_idx[int(n * (train_ratio + val_ratio)) - 1]

    df["split"] = "test"
    df.loc[df["time_idx"] <= train_end, "split"] = "train"
    df.loc[
        (df["time_idx"] > train_end) & (df["time_idx"] <= val_end), "split"
    ] = "val"

    print("[INFO] Split counts:")
    print(df["split"].value_counts())

    print(f"[INFO] Saving TFT master dataset to {OUT_PATH}")
    df.to_csv(OUT_PATH, index=False)
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
