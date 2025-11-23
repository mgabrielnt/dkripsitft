"""Agregasi sentimen harian berbasis label 3-class {-1,0,+1}.

Alur fitur (sentiment pipeline):
1) news_with_sentiment_per_article.csv (processed) ->
   python -m src.data.convert_sentiment_scale
   menghasilkan data/interim/news_with_sentiment_3class.csv
   dengan kolom: gpt_score (asli), sentiment_label_3, pos/neg/neu_count_3 per artikel.
2) File 3-class di atas di-shift weekend -> weekday, lalu diagregasi harian per ticker
   untuk membentuk daily_sentiment.csv.
3) daily_sentiment.csv dipakai di build_tft_master_dataset.py untuk join ke harga.

Setiap kolom utama berasal dari:
- sentiment_mean/min/max : rata-rata/min/max harian dari sentiment_label_3 (-1..1)
- news_count             : jumlah artikel pada hari tsb
- pos_count/neg_count/neu_count: jumlah artikel berdasarkan label_3
- has_news               : 1 jika ada artikel di hari tsb
- sentiment_mean_3d      : rolling mean 3 hari pada sentiment_mean (include hari ini)
- news_count_3d          : rolling sum 3 hari pada news_count
- sentiment_shock        : sentiment_mean - rata-rata 3 hari sebelumnya
- extreme_news           : 1 jika |shock| berada di atas quantile 0.9 per ticker
"""
import os
from typing import List

import pandas as pd

# Lokasi root project
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_INTERIM_DIR = os.path.join(ROOT_DIR, "data", "interim")
DATA_PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")

SRC_PATH = os.path.join(DATA_INTERIM_DIR, "news_with_sentiment_3class.csv")
OUT_PATH = os.path.join(DATA_PROCESSED_DIR, "daily_sentiment.csv")


def shift_to_next_monday(d: pd.Timestamp) -> pd.Timestamp:
    """Geser berita weekend ke hari Senin berikutnya agar selaras dengan hari bursa."""

    if pd.isna(d):
        return d
    wd = d.weekday()  # 0=Mon ... 5=Sat, 6=Sun
    if wd >= 5:
        return d + pd.Timedelta(days=7 - wd)
    return d


def compute_extreme_flag(series: pd.Series, q: float = 0.9) -> pd.Series:
    """Tandai nilai ekstrem berdasarkan quantile absolut per ticker."""

    threshold = series.abs().quantile(q)
    if pd.isna(threshold) or threshold == 0:
        threshold = series.abs().max()  # fallback minimal, bisa nol juga
    return (series.abs() > threshold).astype(int)


def main():
    if not os.path.exists(SRC_PATH):
        raise FileNotFoundError(
            f"Tidak ditemukan file input 3-class: {SRC_PATH}. Jalankan convert_sentiment_scale dulu."
        )

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
    if "sentiment_label_3" not in df.columns:
        raise ValueError(
            "Kolom 'sentiment_label_3' tidak ditemukan. Pastikan sudah menjalankan convert_sentiment_scale."
        )

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
                sentiment_mean=("sentiment_label_3", "mean"),
                sentiment_max=("sentiment_label_3", "max"),
                sentiment_min=("sentiment_label_3", "min"),
                news_count=("sentiment_label_3", "size"),
                pos_count=("sentiment_label_3", lambda x: (x > 0).sum()),
                neg_count=("sentiment_label_3", lambda x: (x < 0).sum()),
                neu_count=("sentiment_label_3", lambda x: (x == 0).sum()),
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

        # Sentiment shock = deviasi terhadap rata-rata 3 hari sebelumnya (stabil, tidak terlalu besar)
        prev3 = daily["sentiment_mean"].shift(1).rolling(window=3, min_periods=1).mean()
        daily["sentiment_shock"] = (daily["sentiment_mean"] - prev3).fillna(0.0)

        # Extreme news: threshold berdasarkan quantile 0.9 abs(shock) per ticker
        daily["extreme_news"] = compute_extreme_flag(daily["sentiment_shock"], q=0.9)

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
