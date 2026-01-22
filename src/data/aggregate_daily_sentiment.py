# file: src/data/aggregate_daily_sentiment.py
"""
Agregasi sentimen harian berbasis label multi-sumber {-1,0,+1} dengan
Advanced Feature Engineering untuk TFT.

Output utama per (ticker, date):
1. Fitur Agregat Dasar:
   - sentiment_final_mean, news_count
2. Fitur Lanjutan (Advanced):
   - sentiment_ema_7d, sentiment_ema_14d (Trend Halus)
   - sentiment_trend_7d (Momentum)
   - sentiment_intraday_std (Ketidakpastian/Divergensi berita harian)
   - high_news_volume_flag (Apakah hari ini banjir berita?)
"""

import os
import numpy as np
import pandas as pd

# Lokasi root project
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")

SRC_PATH = os.path.join(DATA_PROCESSED_DIR, "news_with_sentiment_per_article.csv")
OUT_PATH = os.path.join(DATA_PROCESSED_DIR, "daily_sentiment.csv")


def shift_to_next_monday(d: pd.Timestamp) -> pd.Timestamp:
    """Geser berita weekend ke hari Senin berikutnya."""
    if pd.isna(d):
        return d
    wd = d.weekday()  # 0=Mon ... 6=Sun
    if wd >= 5:
        return d + pd.Timedelta(days=7 - wd)
    return d

def calculate_advanced_features(df_daily: pd.DataFrame) -> pd.DataFrame:
    """Menghitung fitur turunan (Rolling, EMA, Trend) per ticker."""
    df_daily = df_daily.sort_values("date").copy()
    
    # 1. Rolling Mean Biasa (3 Hari) - Sinyal Jangka Pendek
    df_daily["sentiment_mean_3d"] = df_daily["sentiment_final_mean"].rolling(window=3, min_periods=1).mean()
    df_daily["news_count_3d"] = df_daily["news_count"].rolling(window=3, min_periods=1).sum()

    # 2. Exponential Moving Average (EMA) - Sinyal Trend Halus
    # EMA lebih responsif terhadap data baru dibanding rolling mean biasa
    df_daily["sentiment_ema_7d"] = df_daily["sentiment_final_mean"].ewm(span=7, adjust=False).mean()
    df_daily["sentiment_ema_14d"] = df_daily["sentiment_final_mean"].ewm(span=14, adjust=False).mean()

    # 3. Sentiment Momentum / Trend
    # Perubahan sentimen hari ini dibanding rata-rata 7 hari lalu (Are we getting better or worse?)
    # Shift 1 untuk menghindari look-ahead bias yang ketat, atau bandingkan dengan lag
    sent_lag_7d = df_daily["sentiment_final_mean"].shift(5).rolling(window=5, min_periods=1).mean()
    df_daily["sentiment_trend_7d"] = df_daily["sentiment_final_mean"] - sent_lag_7d

    # 4. High News Volume Flag (Top 90% percentile per ticker)
    threshold = df_daily["news_count"].quantile(0.9)
    if threshold == 0: threshold = 1
    df_daily["high_news_day"] = (df_daily["news_count"] >= threshold).astype(int)

    return df_daily

def main():
    if not os.path.exists(SRC_PATH):
        raise FileNotFoundError(f"File tidak ditemukan: {SRC_PATH}")

    print(f"[INFO] Loading {SRC_PATH}")
    df = pd.read_csv(SRC_PATH, parse_dates=["date", "event_date"])

    # Drop duplikat
    df = df.drop_duplicates(subset=["date", "ticker", "title"]).copy()
    
    # Geser tanggal weekend
    df["date"] = pd.to_datetime(df["date"])
    df["date_shifted"] = df["date"].apply(shift_to_next_monday)
    df["has_news"] = 1

    all_daily = []

    # Proses per Ticker
    for ticker, g in df.groupby("ticker"):
        g = g.copy()

        # Agregasi Harian
        agg_funcs = {
            "l_final": ["mean", "std", "count"], # Mean sentiment, Uncertainty (std), Volume
            "sentiment_conf": ["mean"],
            "has_news": ["max"]
        }
        
        daily = g.groupby("date_shifted").agg(agg_funcs).reset_index()
        
        # Ratakan MultiIndex columns
        daily.columns = [
            "date", 
            "sentiment_final_mean", "sentiment_intraday_std", "news_count", 
            "sentiment_conf_mean", 
            "has_news"
        ]
        
        daily["ticker"] = ticker
        
        # Fill NaN untuk std (jika cuma 1 berita, std=NaN -> jadi 0)
        daily["sentiment_intraday_std"] = daily["sentiment_intraday_std"].fillna(0.0)

        # Reindex ke full range date agar rolling window akurat (mengisi hari kosong dengan 0)
        min_date, max_date = daily["date"].min(), daily["date"].max()
        full_idx = pd.bdate_range(min_date, max_date)
        daily = daily.set_index("date").reindex(full_idx).reset_index().rename(columns={"index": "date"})
        
        daily["ticker"] = ticker
        
        # Fill 0 untuk hari tanpa berita
        cols_to_zero = ["sentiment_final_mean", "sentiment_intraday_std", "news_count", "sentiment_conf_mean", "has_news"]
        daily[cols_to_zero] = daily[cols_to_zero].fillna(0)

        # Hitung Advanced Features
        daily = calculate_advanced_features(daily)
        
        all_daily.append(daily)

    if not all_daily:
        print("[WARN] Data kosong setelah agregasi.")
        return

    df_daily = pd.concat(all_daily, ignore_index=True)
    
    # Sort dan Simpan
    df_daily = df_daily.sort_values(["ticker", "date"])
    
    print(f"[INFO] Saving aggregates with advanced features to {OUT_PATH}")
    print(f"[INFO] Columns: {df_daily.columns.tolist()}")
    df_daily.to_csv(OUT_PATH, index=False)
    print("[INFO] Done.")

if __name__ == "__main__":
    main()