import os
from typing import List

import numpy as np
import pandas as pd

# Lokasi root project (sesuaikan dengan strukturmu)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_RAW_PRICES_DIR = os.path.join(ROOT_DIR, "data", "raw", "prices")
DATA_INTERIM_DIR = os.path.join(ROOT_DIR, "data", "interim")

os.makedirs(DATA_INTERIM_DIR, exist_ok=True)

RAW_MERGED_PATH = os.path.join(DATA_RAW_PRICES_DIR, "prices_all_raw.csv")
OUT_PATH = os.path.join(DATA_INTERIM_DIR, "prices_with_indicators.csv")


# ============================================================
# HELPER: RSI
# ============================================================
def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Hitung RSI sederhana.

    Dipakai untuk menghasilkan fitur rsi_14 yang lulus seleksi VIF.
    """
    delta = series.diff()

    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

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
    """
    Bersihkan prices_all_raw.csv supaya:
    - buang baris header aneh (date NaN)
    - kalau format WIDE (Open, Open.1, Open.2, ...), semua digabung jadi 1 kolom:
        Open, High, Low, Close, Adj Close, Volume
      sehingga setiap baris hanya punya 1 set OHLCV sesuai 'ticker'-nya
    - kalau format sudah LONG (tanpa .1/.2/.3), tetap aman
    - pastikan tipe data numerik & date = datetime
    """
    if "date" not in df.columns:
        raise KeyError(
            "Kolom 'date' tidak ditemukan di prices_all_raw.csv. "
            "Pastikan download_prices_yahoo.py menyimpan kolom 'date'."
        )

    # 1) Buang baris yang tanggalnya kosong (biasanya baris header ticker)
    df = df[df["date"].notna()].copy()
    df["date"] = pd.to_datetime(df["date"])

    # 2) Definisi kolom harga/volume dasar
    base_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]

    # Cek apakah ada format WIDE (kolom dengan suffix .1, .2, ...)
    has_wide = any(
        any(col.startswith(base + ".") for base in base_cols)
        for col in df.columns
    )

    if not has_wide:
        # Format sudah long: cukup pastikan numerik
        for col in base_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    # 3) Format WIDE → gabungkan semua suffix .1 s/d .5 ke kolom utama
    suffixes = ["", ".1", ".2", ".3", ".4", ".5"]

    for col in base_cols:
        # mulai dengan semua NaN
        base_series = pd.Series(np.nan, index=df.index)

        # urutan prioritas: kolom tanpa suffix → .1 → .2 → ...
        for suf in suffixes:
            col_name = col if suf == "" else col + suf
            if col_name in df.columns:
                base_series = base_series.where(~base_series.isna(), df[col_name])

        # konversi ke numerik
        df[col] = pd.to_numeric(base_series, errors="coerce")

    # 4) Hapus semua kolom *.1, *.2, ... yang sudah tidak dipakai
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
# HITUNG INDIKATOR TEKNIKAL TERPILIH (+ FITUR TAMBAHAN)
# ============================================================
def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tambahkan indikator teknikal yang DIPAKAI di model akhir
    (berdasarkan hasil VIF & analisis korelasi) + beberapa fitur tambahan
    yang berpotensi membantu model TFT.

    ***Fitur utama:***
    - close
    - volume
    - log_return_1d
    - vol_20                 (std log_return_1d 20 hari)
    - rsi_14
    - ma_5_div_ma_20
    - bb_width_20
    - volume_ma_ratio_20
    - close_lag_2
    - close_lag_3
    - return_mean_5d
    - return_std_5d

    ***Fitur tambahan yang mungkin berguna:***
    - intraday_range_pct     : (high - low) / close
    - atr_14                 : average true range 14 hari
    - gap_return_1d          : log(open / close_prev)
    - price_zscore_20        : (close - ma_20) / std(close, 20)
    - volume_zscore_20       : (volume - volume_ma_20) / std(volume, 20)
    """
    # Pastikan urut per ticker & tanggal
    df = df.sort_values(["ticker", "date"]).copy()

    # Rename kolom OHLCV jadi lowercase biar konsisten
    rename_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    }
    df.rename(columns=rename_map, inplace=True)

    if "ticker" not in df.columns:
        raise KeyError(
            "Kolom 'ticker' tidak ditemukan. Pastikan download_prices_yahoo.py "
            "menambahkan kolom 'ticker' sebelum disimpan."
        )

    result_list: List[pd.DataFrame] = []
    eps = 1e-8

    for ticker, g in df.groupby("ticker"):
        g = g.sort_values("date").copy()

        # ==============================
        # 0) Prev close untuk beberapa indikator
        # ==============================
        prev_close = g["close"].shift(1)

        # ==============================
        # 1) Moving Average harga (MA5, MA20)
        # ==============================
        g["ma_5"] = g["close"].rolling(window=5, min_periods=5).mean()
        g["ma_20"] = g["close"].rolling(window=20, min_periods=20).mean()
        g["ma_5_div_ma_20"] = g["ma_5"] / g["ma_20"]

        # ==============================
        # 2) RSI 14
        # ==============================
        g["rsi_14"] = compute_rsi(g["close"], period=14)

        # ==============================
        # 3) Log return harian + volatilitas 20 hari
        # ==============================
        g["log_return_1d"] = np.log(g["close"] / prev_close)
        g["vol_20"] = g["log_return_1d"].rolling(window=20, min_periods=20).std()

        # ==============================
        # 4) Bollinger Band width 20 hari
        #    middle = MA20, upper/lower = MA20 ± 2*std(close, 20)
        # ==============================
        close_std_20 = g["close"].rolling(window=20, min_periods=20).std()
        g["bb_upper_20"] = g["ma_20"] + 2 * close_std_20
        g["bb_lower_20"] = g["ma_20"] - 2 * close_std_20
        g["bb_width_20"] = (g["bb_upper_20"] - g["bb_lower_20"]) / (g["ma_20"] + eps)

        # *** Tambahan: price_zscore_20 ***
        g["price_zscore_20"] = (g["close"] - g["ma_20"]) / (close_std_20 + eps)

        # ==============================
        # 5) Volume MA ratio 20 hari + z-score volume
        # ==============================
        g["volume_ma_20"] = g["volume"].rolling(window=20, min_periods=5).mean()
        g["volume_ma_ratio_20"] = g["volume"] / (g["volume_ma_20"] + eps)

        volume_std_20 = g["volume"].rolling(window=20, min_periods=5).std()
        g["volume_zscore_20"] = (g["volume"] - g["volume_ma_20"]) / (volume_std_20 + eps)

        # ==============================
        # 6) Lagged close (1, 2 & 3 hari)
        # ==============================
        g["close_lag_1"] = prev_close
        g["close_lag_2"] = g["close"].shift(2)
        g["close_lag_3"] = g["close"].shift(3)

        # ==============================
        # 7) Return window 5 hari: mean & std
        # ==============================
        g["return_mean_5d"] = g["log_return_1d"].rolling(window=5, min_periods=3).mean()
        g["return_std_5d"] = g["log_return_1d"].rolling(window=5, min_periods=3).std()

        # ==============================
        # 8) Intraday range & ATR 14 (true range)
        # ==============================
        # intraday range (% dari close)
        g["intraday_range_pct"] = (g["high"] - g["low"]) / (g["close"] + eps)

        # True Range
        tr1 = g["high"] - g["low"]
        tr2 = (g["high"] - prev_close).abs()
        tr3 = (g["low"] - prev_close).abs()
        g["true_range"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # ATR 14 (pakai simple rolling mean)
        g["atr_14"] = g["true_range"].rolling(window=14, min_periods=5).mean()

        # ==============================
        # 9) Gap return (overnight gap)
        # ==============================
        g["gap_return_1d"] = np.log(g["open"] / (prev_close + eps))

        result_list.append(g)

    df_ind = pd.concat(result_list, ignore_index=True)

    # Buang kolom intermediate yang tidak diperlukan oleh model
    drop_cols = [
        "open",
        "high",
        "low",
        "adj_close",
        "ma_5",
        "ma_20",
        "bb_upper_20",
        "bb_lower_20",
        "volume_ma_20",
        "true_range",
        "close_lag_1",
    ]
    df_ind = df_ind.drop(columns=[c for c in drop_cols if c in df_ind.columns])

    # Susun ulang kolom: identitas + fitur yang dipakai
    keep_cols = [
        "ticker",
        "date",
        # harga & volume dasar
        "close",
        "volume",
        # return & volatilitas
        "log_return_1d",
        "vol_20",
        "return_mean_5d",
        "return_std_5d",
        # indikator tren / level
        "rsi_14",
        "ma_5_div_ma_20",
        "bb_width_20",
        "price_zscore_20",
        # indikator volume
        "volume_ma_ratio_20",
        "volume_zscore_20",
        # lag harga
        "close_lag_2",
        "close_lag_3",
        # fitur volatilitas & range tambahan
        "intraday_range_pct",
        "atr_14",
        "gap_return_1d",
    ]
    keep_cols = [c for c in keep_cols if c in df_ind.columns]

    df_ind = df_ind[keep_cols]

    return df_ind


# ============================================================
# MAIN
# ============================================================
def main():
    if not os.path.exists(RAW_MERGED_PATH):
        raise FileNotFoundError(f"File harga gabungan tidak ditemukan: {RAW_MERGED_PATH}")

    print(f"[INFO] Loading raw prices from {RAW_MERGED_PATH}")
    # Baca mentah dulu (tanpa parse_dates, akan di-handle di clean_raw_prices)
    df_raw = pd.read_csv(RAW_MERGED_PATH)

    # 🔧 BERSIHKAN data mentah (buang baris aneh, gabung kolom *.1/.2/... → satu kolom)
    df_clean = clean_raw_prices(df_raw)

    # Tambah indikator teknikal yang sudah diseleksi via VIF & analisis korelasi
    # + beberapa fitur tambahan yang berpotensi membantu TFT
    df_ind = add_technical_indicators(df_clean)

    print("[INFO] Kolom yang disimpan di prices_with_indicators.csv:")
    print(df_ind.columns.tolist())

    print("\n[INFO] Jumlah NaN per kolom (setelah hitung indikator):")
    print(df_ind.isna().sum())

    print(f"\n[INFO] Saving prices with indicators to {OUT_PATH}")
    df_ind.to_csv(OUT_PATH, index=False)
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
