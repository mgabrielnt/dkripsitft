"""
download_prices_yahoo.py

Mengambil data harga harian (OHLCV) dari Yahoo Finance untuk daftar ticker
di configs/data.yaml dan menyimpannya ke:

    data/raw/prices/prices_all_raw.csv

Versi ROBUST:
- Selalu download ulang full (non-incremental).
- Ada mekanisme RETRY jika download gagal.
- Format long: date, ticker, Open, High, Low, Close, Adj Close, Volume
"""

import os
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List

import pandas as pd
import yaml

try:
    import yfinance as yf
except ImportError as e:
    raise ImportError(
        "Module 'yfinance' belum terinstall. Jalankan: pip install yfinance"
    ) from e

# Lokasi root project
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_RAW_PRICES_DIR = os.path.join(ROOT_DIR, "data", "raw", "prices")
os.makedirs(DATA_RAW_PRICES_DIR, exist_ok=True)

CONFIG_DATA_PATH = os.path.join(ROOT_DIR, "configs", "data.yaml")
OUT_PATH = os.path.join(DATA_RAW_PRICES_DIR, "prices_all_raw.csv")


# ====================== Helper config ======================

def load_data_config(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config data.yaml tidak ditemukan: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_start_date(raw: Any) -> datetime:
    default_dt = datetime(2017, 1, 1)
    if raw is None:
        return default_dt
    if isinstance(raw, str):
        try:
            return datetime.fromisoformat(raw)
        except Exception:
            return default_dt
    return default_dt


def get_end_date_utc_plus_one() -> datetime:
    today_utc = datetime.now(timezone.utc).date()
    return datetime.combine(today_utc + timedelta(days=1), datetime.min.time())


# ====================== Download per ticker ======================

def download_prices_for_ticker(
    ticker: str,
    start_dt: datetime,
    end_dt: datetime,
    max_retries: int = 3
) -> pd.DataFrame:
    """
    Download harga harian dengan mekanisme RETRY.
    """
    if start_dt >= end_dt:
        return pd.DataFrame()

    for attempt in range(max_retries):
        try:
            print(
                f"[INFO] Download {ticker} (Attempt {attempt+1}/{max_retries}) "
                f"from {start_dt.date()} to {end_dt.date()}..."
            )
            
            df = yf.download(
                ticker,
                start=start_dt,
                end=end_dt,
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=False # Single thread lebih stabil untuk loop
            )

            if df.empty:
                print(f"[WARN] Data kosong untuk {ticker}. Retrying..." if attempt < max_retries - 1 else "[ERR] Gagal total.")
                time.sleep(2)
                continue

            # Sukses download, proses data
            df = df.reset_index()
            
            # Standardisasi kolom Date
            if "Date" in df.columns:
                df.rename(columns={"Date": "date"}, inplace=True)
            elif "date" not in df.columns:
                df.rename_axis("date", inplace=True)
                df.reset_index(inplace=True)

            df["ticker"] = ticker
            
            # Flat column names (kadang yf kasih multi-index)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

            keep_cols = ["date", "ticker", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
            final_cols = [c for c in keep_cols if c in df.columns]
            
            return df[final_cols]

        except Exception as e:
            print(f"[ERROR] {ticker} attempt {attempt+1} failed: {e}")
            time.sleep(2)

    return pd.DataFrame() # Gagal setelah semua retries


# ====================== MAIN ======================

def main():
    cfg = load_data_config(CONFIG_DATA_PATH)
    tickers: List[str] = cfg.get("tickers", []) or []
    
    if not tickers:
        raise ValueError("Config 'tickers' kosong. Isi data.yaml dulu.")

    start_dt = parse_start_date(cfg.get("start_date", None))
    end_dt = get_end_date_utc_plus_one()

    print("==================================================")
    print("   STOCK PRICE DOWNLOADER (Yahoo Finance)")
    print("==================================================")
    print(f" Tickers    : {len(tickers)} saham")
    print(f" Start Date : {start_dt.date()}")
    print("==================================================\n")

    all_frames: List[pd.DataFrame] = []
    failed_tickers = []

    for ticker in tickers:
        df_t = download_prices_for_ticker(ticker, start_dt, end_dt)
        
        if not df_t.empty:
            df_t["date"] = pd.to_datetime(df_t["date"])
            df_t = df_t.sort_values("date")
            print(f"   -> OK: {len(df_t)} rows ({df_t['date'].min().date()} - {df_t['date'].max().date()})")
            all_frames.append(df_t)
        else:
            print(f"   -> FAILED: {ticker} (Data Empty)")
            failed_tickers.append(ticker)
        
        # Jeda sopan agar tidak kena rate limit
        time.sleep(0.5)

    print("\n==================================================")
    if failed_tickers:
        print(f"[WARN] {len(failed_tickers)} ticker gagal didownload: {failed_tickers}")
    
    if not all_frames:
        print("[ERROR] Tidak ada data yang berhasil diambil. Exit.")
        return

    print("Combining dataframes...")
    df_all = pd.concat(all_frames, ignore_index=True)
    df_all["date"] = pd.to_datetime(df_all["date"])
    df_all = df_all.sort_values(["ticker", "date"]).reset_index(drop=True)

    print(f"[INFO] Total Rows: {len(df_all)}")
    print(f"[INFO] Saving to: {OUT_PATH}")
    
    try:
        df_all.to_csv(OUT_PATH, index=False)
        print("[SUCCESS] Data saved successfully.")
    except Exception as e:
        print(f"[CRITICAL] Gagal menyimpan file CSV: {e}")
        print("Coba tutup file jika sedang dibuka di Excel!")

if __name__ == "__main__":
    main()