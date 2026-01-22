"""
Bangun dataset TFT Master dengan fitur "Interaction" (Sentimen x Volume)
dan GATING MECHANISM (Filter Sinyal Lemah).
"""

import os
import numpy as np
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

def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def drop_non_trading_days(df_p):
    """Hapus baris dimana volume 0 atau NaN."""
    df_p["volume"] = pd.to_numeric(df_p["volume"], errors="coerce").fillna(0)
    return df_p[df_p["volume"] > 0].copy()

def add_interaction_features(df):
    """
    Membuat fitur interaksi dengan GATING MECHANISM.
    Hanya sinyal kuat yang diloloskan untuk mengurangi noise.
    """
    # 1. Sentiment x Volume Impact (Original)
    # Log volume agar skalanya tidak meledak
    log_vol = np.log1p(df["volume"])
    raw_impact = df["sentiment_mean_3d"] * log_vol
    
    # --- PERBAIKAN: GATING MECHANISM (THRESHOLDING) ---
    # Kita hanya ambil impact yang nilainya > 0.5 atau < -0.5
    # Nilai kecil dianggap NOISE (dibuat jadi 0)
    df["sentiment_vol_impact"] = raw_impact.apply(lambda x: x if abs(x) > 0.5 else 0.0)
    
    # 2. Sinyal Arah Eksplisit (Categorical Direction)
    # Membantu model menangkap arah secara tegas (-1, 0, 1)
    # Ini "contekan" langsung buat model agar DA naik
    df["sentiment_dir_signal"] = 0.0
    df.loc[df["sentiment_vol_impact"] > 0.5, "sentiment_dir_signal"] = 1.0
    df.loc[df["sentiment_vol_impact"] < -0.5, "sentiment_dir_signal"] = -1.0
    
    return df

def build_keep_columns(df):
    """Daftar kolom final untuk masuk ke TFT."""
    
    base = ["time_idx", "date", "ticker", "day_of_week", "month", "is_month_end", "split"]
    target = ["close"]
    
    # Indikator Teknikal
    tech = [
        "volume", "log_return_1d", "vol_20", "rsi_14", 
        "ma_5_div_ma_20", "bb_width_20", "gap_return_1d", "intraday_range_pct"
    ]
    
    # Indikator Sentimen (UPDATED WITH DIR SIGNAL)
    sent = [
        "has_news", 
        "news_count_3d", 
        "sentiment_final_mean",      
        "sentiment_mean_3d",         
        "sentiment_ema_7d",          
        "sentiment_ema_14d",         
        "sentiment_trend_7d",        
        "sentiment_intraday_std",    
        "sentiment_vol_impact",      
        "high_news_day",
        "sentiment_dir_signal"       # <--- FITUR BARU PENTING
    ]

    # Gabung dan pastikan kolomnya ada di DF
    all_cols = base + target + tech + sent
    return [c for c in all_cols if c in df.columns]

def main():
    # 1. Load Data
    print(f"[INFO] Loading prices: {PRICES_PATH}")
    df_p = pd.read_csv(PRICES_PATH, parse_dates=["date"])
    
    print(f"[INFO] Loading sentiment: {SENTIMENT_PATH}")
    df_s = pd.read_csv(SENTIMENT_PATH, parse_dates=["date"])

    # 2. Clean Prices
    df_p = drop_non_trading_days(df_p)
    
    # Config Split
    data_cfg = load_yaml(CONFIG_DATA_PATH)
    train_ratio = data_cfg.get("train_ratio", 0.7)
    val_ratio = data_cfg.get("val_ratio", 0.2)

    # 3. Merge Sentiment (Left Join ke Harga)
    df = pd.merge(df_p, df_s, on=["ticker", "date"], how="left")

    # 4. Fill NaN Sentimen (No News = 0)
    sentiment_cols = [c for c in df_s.columns if c not in ["ticker", "date"]]
    df[sentiment_cols] = df[sentiment_cols].fillna(0)

    # 5. Add Calendar Features
    df["day_of_week"] = df["date"].dt.weekday
    df["month"] = df["date"].dt.month
    df["is_month_end"] = df["date"].dt.is_month_end.astype(int)

    # 6. Add Time Index
    dates = sorted(df["date"].unique())
    date_map = {d: i for i, d in enumerate(dates)}
    df["time_idx"] = df["date"].map(date_map)

    # 7. Add Interaction Features (GATING APPLIED)
    print("[INFO] Adding interaction features & Gating Mechanism...")
    df = add_interaction_features(df)

    # 8. Split Data
    dates_list = sorted(df["date"].unique())
    n_dates = len(dates_list)
    train_idx = int(n_dates * train_ratio)
    val_idx = int(n_dates * (train_ratio + val_ratio))
    
    train_date_max = dates_list[train_idx]
    val_date_max = dates_list[val_idx]

    df["split"] = "test"
    df.loc[df["date"] <= train_date_max, "split"] = "train"
    df.loc[(df["date"] > train_date_max) & (df["date"] <= val_date_max), "split"] = "val"

    print(f"[INFO] Split Counts: {df['split'].value_counts().to_dict()}")

    # 9. Filter Final Columns
    final_cols = build_keep_columns(df)
    df_final = df[final_cols].copy()

    # 10. Save
    print(f"[INFO] Saving TFT Master to {OUT_PATH}")
    print(f"[INFO] Final Columns: {df_final.columns.tolist()}")
    df_final.to_csv(OUT_PATH, index=False)
    print("[INFO] Done.")

if __name__ == "__main__":
    main()