import os
from typing import List

import numpy as np
import pandas as pd

# Lokasi root project
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")

SRC_PATH = os.path.join(DATA_PROCESSED_DIR, "news_with_sentiment_per_article.csv")
OUT_PATH = os.path.join(DATA_PROCESSED_DIR, "daily_sentiment.csv")


def shift_to_next_monday(d: pd.Timestamp) -> pd.Timestamp:
    """
    Geser berita yang jatuh di weekend ke hari Senin berikutnya,
    supaya align dengan hari perdagangan.
    """
    if pd.isna(d):
        return d
    wd = d.weekday()  # 0=Mon ... 5=Sat, 6=Sun
    if wd >= 5:
        return d + pd.Timedelta(days=7 - wd)
    return d


def main():
    if not os.path.exists(SRC_PATH):
        raise FileNotFoundError(f"Tidak ditemukan: {SRC_PATH}")

    print(f"[INFO] Loading {SRC_PATH}")
    df = pd.read_csv(SRC_PATH, parse_dates=["date"])

    before = len(df)
    subset_cols: List[str] = []
    for c in ["date", "ticker", "title", "link"]:
        if c in df.columns:
            subset_cols.append(c)
    if not subset_cols:
        subset_cols = ["date", "ticker", "gpt_score"]

    df = df.drop_duplicates(subset=subset_cols).copy()
    after = len(df)
    print(f"[INFO] Drop duplikat: {before - after} baris dibuang, sisa {after}")

    # pastikan kolom yang diperlukan ada
    if "gpt_score" not in df.columns:
        raise ValueError("Kolom 'gpt_score' tidak ditemukan di news_with_sentiment_per_article.csv")

    df["date"] = pd.to_datetime(df["date"])
    df["date_shifted"] = df["date"].apply(shift_to_next_monday)

    # flag has_news per artikel (nantinya di-agg)
    df["has_news"] = 1

    all_daily = []

    for ticker, g in df.groupby("ticker"):
        g = g.copy()

        daily = (
            g.groupby("date_shifted")
            .agg(
                sentiment_mean=("gpt_score", "mean"),
                sentiment_max=("gpt_score", "max"),
                sentiment_min=("gpt_score", "min"),
                news_count=("gpt_score", "size"),
                pos_count=("gpt_score", lambda x: (x > 0).sum()),
                neg_count=("gpt_score", lambda x: (x < 0).sum()),
                neu_count=("gpt_score", lambda x: (x == 0).sum()),
                has_news=("has_news", "max"),
            )
            .reset_index()
        )

        if daily.empty:
            continue

        # Reindex ke semua hari kerja antara min & max tanggal (supaya align dengan harga)
        start_date = daily["date_shifted"].min()
        end_date = daily["date_shifted"].max()
        full_idx = pd.bdate_range(start=start_date, end=end_date, freq="B")

        daily = daily.set_index("date_shifted").reindex(full_idx)
        daily.index.name = "date"

        daily["ticker"] = ticker

        # isi kosong dengan 0 (hari tanpa berita)
        sentiment_cols = ["sentiment_mean", "sentiment_max", "sentiment_min"]
        count_cols = ["news_count", "pos_count", "neg_count", "neu_count"]
        binary_cols = ["has_news"]

        daily[sentiment_cols] = daily[sentiment_cols].fillna(0.0)
        daily[count_cols] = daily[count_cols].fillna(0)
        daily[binary_cols] = daily[binary_cols].fillna(0)

        # rolling 3 hari
        daily["sentiment_mean_3d"] = (
            daily["sentiment_mean"].rolling(window=3, min_periods=1).mean()
        )
        daily["news_count_3d"] = (
            daily["news_count"].rolling(window=3, min_periods=1).sum()
        )

        # Sentiment shock = deviasi dari rata-rata 7 hari
        base7 = daily["sentiment_mean"].rolling(window=7, min_periods=1).mean()
        daily["sentiment_shock"] = daily["sentiment_mean"] - base7

        # Extreme news: jika shock besar (threshold bisa kamu jelaskan di skripsi)
        threshold = 1.0
        daily["extreme_news"] = (daily["sentiment_shock"].abs() > threshold).astype(int)

        all_daily.append(daily.reset_index())

    if not all_daily:
        print("[WARN] Tidak ada data harian setelah agregasi.")
        # tetap simpan CSV kosong dengan schema lengkap
        empty_cols = [
            "date",
            "ticker",
            "sentiment_mean",
            "sentiment_max",
            "sentiment_min",
            "news_count",
            "pos_count",
            "neg_count",
            "neu_count",
            "sentiment_mean_3d",
            "news_count_3d",
            "has_news",
            "sentiment_shock",
            "extreme_news",
        ]
        pd.DataFrame(columns=empty_cols).to_csv(OUT_PATH, index=False)
        print(f"[INFO] Saving empty daily_sentiment to {OUT_PATH}")
        return

    df_daily = pd.concat(all_daily, ignore_index=True)

    # Final safety fill
    sentiment_cols_all = [
        "sentiment_mean",
        "sentiment_max",
        "sentiment_min",
        "sentiment_mean_3d",
        "sentiment_shock",
    ]
    count_cols_all = [
        "news_count",
        "pos_count",
        "neg_count",
        "neu_count",
        "news_count_3d",
    ]
    binary_cols_all = ["has_news", "extreme_news"]

    for col in sentiment_cols_all:
        if col in df_daily.columns:
            df_daily[col] = df_daily[col].astype(float).fillna(0.0)
    for col in count_cols_all:
        if col in df_daily.columns:
            df_daily[col] = df_daily[col].fillna(0).astype(int)
    for col in binary_cols_all:
        if col in df_daily.columns:
            df_daily[col] = df_daily[col].fillna(0).astype(int)

    df_daily = df_daily.sort_values(["ticker", "date"]).reset_index(drop=True)

    print(f"[INFO] Total baris harian (setelah shifting weekend): {len(df_daily)}")
    print(f"[INFO] Saving to {OUT_PATH}")
    df_daily.to_csv(OUT_PATH, index=False)
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
