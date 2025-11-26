"""Download harga saham dari Yahoo Finance sesuai daftar ticker di configs/data.yaml.

Output: data/raw/prices/prices_all_raw.csv
Kolom: date, ticker, Open, High, Low, Close, Adj Close, Volume
"""

import os
from typing import Any, Dict, List

import pandas as pd
import yaml

try:
    import yfinance as yf
except ImportError as e:  # pragma: no cover - dependency error is explicit to users
    raise ImportError("Module 'yfinance' belum terinstall. Jalankan: pip install yfinance") from e

# Lokasi root project
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CONFIG_DATA_PATH = os.path.join(ROOT_DIR, "configs", "data.yaml")
DATA_RAW_PRICES_DIR = os.path.join(ROOT_DIR, "data", "raw", "prices")

os.makedirs(DATA_RAW_PRICES_DIR, exist_ok=True)

OUT_PATH = os.path.join(DATA_RAW_PRICES_DIR, "prices_all_raw.csv")


def load_config(path: str) -> Dict[str, Any]:
    """Load YAML configs/data.yaml dan pastikan field kunci tersedia."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Config data.yaml tidak ditemukan: {path}. Isi tickers, start_date, end_date."
        )

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    tickers = cfg.get("tickers") or []
    if not tickers:
        raise ValueError("Field 'tickers' di configs/data.yaml kosong. Isi minimal satu ticker.")

    start_date = cfg.get("start_date")
    end_date = cfg.get("end_date")
    if not start_date or not end_date:
        raise ValueError(
            "Field 'start_date' dan 'end_date' wajib di configs/data.yaml untuk download harga."
        )

    return {
        "tickers": tickers,
        "start_date": str(start_date),
        "end_date": str(end_date),
    }


def download_prices(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Ambil OHLCV untuk satu ticker dan kembalikan DataFrame panjang."""
    print(f"[INFO] Download harga {ticker} dari {start_date} sampai {end_date}")
    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False, progress=False)

    if df.empty:
        print(f"[WARN] Tidak ada data harga untuk {ticker} dalam rentang tersebut.")
        return pd.DataFrame()

    df = df.reset_index().rename(columns={"Date": "date"})
    df["ticker"] = ticker
    # Pastikan kolom urut supaya mudah dibaca downstream
    cols = ["date", "ticker", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
    df = df[cols]
    return df


def main() -> None:
    cfg = load_config(CONFIG_DATA_PATH)
    tickers: List[str] = cfg["tickers"]
    start_date: str = cfg["start_date"]
    end_date: str = cfg["end_date"]

    all_prices: List[pd.DataFrame] = []
    for ticker in tickers:
        df = download_prices(ticker, start_date=start_date, end_date=end_date)
        if not df.empty:
            all_prices.append(df)

    if not all_prices:
        print("[ERROR] Tidak ada data harga yang berhasil diunduh. Cek ticker atau rentang tanggal.")
        return

    df_all = pd.concat(all_prices, ignore_index=True)
    df_all = df_all.sort_values(["ticker", "date"]).reset_index(drop=True)

    print(f"[INFO] Total baris harga gabungan: {len(df_all)}")
    print(f"[INFO] Menyimpan ke: {OUT_PATH}")
    df_all.to_csv(OUT_PATH, index=False)
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
