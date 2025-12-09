"""
download_prices_yahoo.py

Mengambil data harga harian (OHLCV) dari Yahoo Finance untuk daftar ticker
di configs/data.yaml dan menyimpannya ke:

    data/raw/prices/prices_all_raw.csv

Versi SIMPLE:
- Tidak incremental, selalu download ulang full dari start_date sampai hari ini.
- Format long:
    date, ticker, Open, High, Low, Close, Adj Close, Volume
"""

import os
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
    """Load configs/data.yaml sebagai dict."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config data.yaml tidak ditemukan: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg


def parse_start_date(raw: Any) -> datetime:
    """
    Parse start_date dari config menjadi datetime (naive).
    Default fallback: 2017-01-01 kalau tidak ada / invalid.
    """
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
    """
    Yahoo Finance: parameter end = exclusive.
    Jadi kalau mau sampai hari ini (berdasarkan UTC), pakai (UTC_today + 1 hari).
    """
    today_utc = datetime.now(timezone.utc).date()
    return datetime.combine(today_utc + timedelta(days=1), datetime.min.time())


# ====================== Download per ticker ======================

def download_prices_for_ticker(
    ticker: str,
    start_dt: datetime,
    end_dt: datetime,
) -> pd.DataFrame:
    """
    Download harga harian untuk satu ticker, dari start_dt (inclusive)
    sampai end_dt (exclusive).
    """
    if start_dt >= end_dt:
        print(f"[INFO] Rentang waktu kosong untuk {ticker}, skip.")
        return pd.DataFrame()

    print(
        f"[INFO] Download harga {ticker} "
        f"dari {start_dt.date()} s/d {end_dt.date()} (exclusive end)"
    )

    df = yf.download(
        ticker,
        start=start_dt,
        end=end_dt,
        interval="1d",
        auto_adjust=False,
        progress=False,
    )

    if df.empty:
        print(f"[WARN] Tidak ada data dari Yahoo untuk {ticker} dalam rentang ini.")
        return pd.DataFrame()

    df = df.reset_index()

    # yfinance biasanya pakai kolom 'Date'
    if "Date" in df.columns:
        df.rename(columns={"Date": "date"}, inplace=True)
    else:
        # fallback kalau index adalah date
        if "date" not in df.columns:
            df.rename_axis("date", inplace=True)
            df.reset_index(inplace=True)

    df["ticker"] = ticker

    keep_cols = [
        "date",
        "ticker",
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
        "Volume",
    ]
    # jaga-jaga kalau ada kolom yang tidak tersedia
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols]

    return df


# ====================== MAIN ======================

def main():
    cfg = load_data_config(CONFIG_DATA_PATH)

    tickers: List[str] = cfg.get("tickers", []) or []
    if not tickers:
        raise ValueError(
            "Daftar 'tickers' di configs/data.yaml kosong. "
            "Minimal isi 1 ticker, misal: ['BBCA.JK', 'BBRI.JK']"
        )

    start_dt = parse_start_date(cfg.get("start_date", None))
    end_dt = get_end_date_utc_plus_one()

    print("[INFO] Konfigurasi download harga:")
    print(f"       tickers    : {tickers}")
    print(f"       start_date : {start_dt.date()}")
    print(f"       end_date   : {end_dt.date()} (exclusive)")

    all_frames: List[pd.DataFrame] = []

    for ticker in tickers:
        df_t = download_prices_for_ticker(
            ticker=ticker,
            start_dt=start_dt,
            end_dt=end_dt,
        )
        if df_t.empty:
            print(f"[WARN] Data kosong untuk {ticker}, akan dilewati.")
        else:
            df_t["date"] = pd.to_datetime(df_t["date"])
            df_t = df_t.sort_values("date")
            print(
                f"[INFO] {ticker}: {len(df_t)} baris, "
                f"rentang {df_t['date'].min().date()} s/d {df_t['date'].max().date()}"
            )
            all_frames.append(df_t)

    if not all_frames:
        print("[ERROR] Tidak ada data harga untuk semua ticker. File tidak dibuat.")
        return

    df_all = pd.concat(all_frames, ignore_index=True)

    # Pastikan tanggal datetime & sort
    df_all["date"] = pd.to_datetime(df_all["date"])
    df_all = df_all.sort_values(["ticker", "date"]).reset_index(drop=True)

    print(f"[INFO] Total baris: {len(df_all)}")
    print(
        f"[INFO] Rentang tanggal global: "
        f"{df_all['date'].min().date()}  s/d  {df_all['date'].max().date()}"
    )

    print(f"[INFO] Menyimpan ke: {OUT_PATH}")
    df_all.to_csv(OUT_PATH, index=False)
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
