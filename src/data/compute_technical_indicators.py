import os
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import yaml

# Lokasi root project
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_RAW_PRICES_DIR = os.path.join(ROOT_DIR, "data", "raw", "prices")
DATA_INTERIM_DIR = os.path.join(ROOT_DIR, "data", "interim")
CONFIG_DATA_PATH = os.path.join(ROOT_DIR, "configs", "data.yaml")

os.makedirs(DATA_INTERIM_DIR, exist_ok=True)

RAW_MERGED_PATH = os.path.join(DATA_RAW_PRICES_DIR, "prices_all_raw.csv")
OUT_PATH = os.path.join(DATA_INTERIM_DIR, "prices_with_indicators.csv")


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ============================================================
# HELPER: RSI
# ============================================================
def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    gain = pd.Series(gain, index=series.index)
    loss = pd.Series(loss, index=series.index)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


# ============================================================
# BERSIHKAN RAW PRICES
# ============================================================
def clean_raw_prices(df: pd.DataFrame) -> pd.DataFrame:
    if "date" not in df.columns:
        raise KeyError("Kolom 'date' tidak ditemukan di prices_all_raw.csv.")

    # buang baris aneh (date NaN)
    df = df[df["date"].notna()].copy()
    df["date"] = pd.to_datetime(df["date"])

    base_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]

    # deteksi format WIDE
    has_wide = any(any(col.startswith(base + ".") for base in base_cols) for col in df.columns)

    if not has_wide:
        # format long
        for col in base_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    # format wide -> gabung suffix
    suffixes = ["", ".1", ".2", ".3", ".4", ".5"]
    for col in base_cols:
        base_series = pd.Series(np.nan, index=df.index)
        for suf in suffixes:
            col_name = col if suf == "" else col + suf
            if col_name in df.columns:
                base_series = base_series.where(~base_series.isna(), df[col_name])
        df[col] = pd.to_numeric(base_series, errors="coerce")

    # drop kolom wide sisa
    drop_cols = []
    for c in df.columns:
        for base in base_cols:
            if c.startswith(base + "."):
                drop_cols.append(c)
                break
    if drop_cols:
        df = df.drop(columns=drop_cols)

    return df


# ============================================================
# FILTER TRADING DAYS + KALENDER INTERSECTION
# ============================================================
def drop_non_trading_rows(df: pd.DataFrame) -> pd.DataFrame:
    # definisi non-trading: volume <= 0 atau NaN
    df = df.copy()
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")
    df = df[df["Volume"].notna() & (df["Volume"] > 0)].copy()
    return df


def apply_intersection_calendar(df: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    # ambil tanggal yang ada di semua ticker
    sets = []
    for t in tickers:
        d = df.loc[df["ticker"] == t, "date"].dropna().drop_duplicates()
        sets.append(set(d.tolist()))
    common = set.intersection(*sets) if sets else set()
    df = df[df["date"].isin(common)].copy()
    return df


# ============================================================
# HITUNG INDIKATOR TEKNIKAL (MINIMAL FINAL)
# ============================================================
def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["ticker", "date"]).copy()

    rename_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    }
    df.rename(columns=rename_map, inplace=True)

    required = ["ticker", "date", "open", "high", "low", "close", "volume"]
    for c in required:
        if c not in df.columns:
            raise KeyError(f"Kolom '{c}' tidak ditemukan. Cek file raw merged Anda.")

    # pastikan numerik
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    eps = 1e-8
    out_list: List[pd.DataFrame] = []

    for ticker, g in df.groupby("ticker"):
        g = g.sort_values("date").copy()

        prev_close = g["close"].shift(1)

        # log return
        g["log_return_1d"] = np.log(g["close"] / prev_close)

        # vol 20 dari log_return
        g["vol_20"] = g["log_return_1d"].rolling(window=20, min_periods=20).std()

        # MA ratio
        ma_5 = g["close"].rolling(window=5, min_periods=5).mean()
        ma_20 = g["close"].rolling(window=20, min_periods=20).mean()
        g["ma_5_div_ma_20"] = ma_5 / (ma_20 + eps)

        # RSI 14
        g["rsi_14"] = compute_rsi(g["close"], period=14)

        # Bollinger width
        close_std_20 = g["close"].rolling(window=20, min_periods=20).std()
        bb_upper = ma_20 + 2 * close_std_20
        bb_lower = ma_20 - 2 * close_std_20
        g["bb_width_20"] = (bb_upper - bb_lower) / (ma_20 + eps)

        # Intraday range (%)
        g["intraday_range_pct"] = (g["high"] - g["low"]) / (g["close"] + eps)

        # Gap return (overnight)
        g["gap_return_1d"] = np.log(g["open"] / (prev_close + eps))

        out_list.append(g)

    df_ind = pd.concat(out_list, ignore_index=True)

    # keep minimal cols (sesuai dataset final)
    keep = [
        "ticker", "date",
        "close", "volume",
        "log_return_1d", "vol_20",
        "rsi_14", "ma_5_div_ma_20", "bb_width_20",
        "gap_return_1d", "intraday_range_pct",
    ]
    df_ind = df_ind[keep].copy()

    # drop warm-up rows yang masih NaN (penting untuk training stabil)
    df_ind = df_ind.dropna(subset=[
        "log_return_1d", "vol_20", "rsi_14", "ma_5_div_ma_20", "bb_width_20",
        "gap_return_1d", "intraday_range_pct"
    ])

    return df_ind


def main():
    if not os.path.exists(RAW_MERGED_PATH):
        raise FileNotFoundError(f"File harga gabungan tidak ditemukan: {RAW_MERGED_PATH}")

    cfg = load_yaml(CONFIG_DATA_PATH)
    tickers_cfg = cfg.get("tickers", None)

    print(f"[INFO] Loading raw prices from {RAW_MERGED_PATH}")
    df_raw = pd.read_csv(RAW_MERGED_PATH)

    df_clean = clean_raw_prices(df_raw)

    if "ticker" not in df_clean.columns:
        raise KeyError("Kolom 'ticker' tidak ditemukan pada raw merged. Pastikan pipeline download/merge menambahkannya.")

    # filter tickers jika ada di config
    if tickers_cfg:
        df_clean = df_clean[df_clean["ticker"].isin(tickers_cfg)].copy()

    tickers_final = sorted(df_clean["ticker"].dropna().unique().tolist())
    if not tickers_final:
        raise ValueError("Tidak ada ticker tersisa setelah filter.")

    # 1) drop non-trading/sintetis
    df_clean = drop_non_trading_rows(df_clean)

    # 2) intersection calendar supaya panel konsisten
    df_clean = apply_intersection_calendar(df_clean, tickers_final)

    # 3) hitung indikator pada trading-days only + panel konsisten
    df_ind = add_technical_indicators(df_clean)

    print("[INFO] Kolom yang disimpan di prices_with_indicators.csv:")
    print(df_ind.columns.tolist())

    print("\n[INFO] Shape akhir:", df_ind.shape)
    print(f"[INFO] Saving prices with indicators to {OUT_PATH}")
    df_ind.to_csv(OUT_PATH, index=False)
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
