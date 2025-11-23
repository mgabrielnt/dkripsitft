import os
import numpy as np
import pandas as pd

# Lokasi root project (sesuaikan dengan strukturmu)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_RAW_PRICES_DIR = os.path.join(ROOT_DIR, "data", "raw", "prices")
DATA_INTERIM_DIR = os.path.join(ROOT_DIR, "data", "interim")

os.makedirs(DATA_INTERIM_DIR, exist_ok=True)

RAW_MERGED_PATH = os.path.join(DATA_RAW_PRICES_DIR, "prices_all_raw.csv")
OUT_PATH = os.path.join(DATA_INTERIM_DIR, "prices_with_indicators.csv")


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Hitung RSI sederhana."""
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


def clean_raw_prices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bersihkan prices_all_raw.csv supaya:
    - baris header duplikat hilang
    - kolom *.1 (BBRI) digabung ke kolom utama
    - kolom harga & volume jadi numerik
    """
    # 1) Buang baris yang tanggalnya kosong (itu baris 'BBCA.JK, BBRI.JK, ...')
    df = df[df["date"].notna()].copy()

    # 2) Gabungkan kolom Close.1 → Close, High.1 → High, dst (kalau ada)
    base_cols = ["Close", "High", "Low", "Open", "Volume"]
    for col in base_cols:
        other = col + ".1"
        if other in df.columns:
            # kalau kolom utama NaN (misal baris BBRI), isi dengan nilai dari kolom .1
            df[col] = df[col].fillna(df[other])

    # 3) Hapus kolom *.1 yang sudah tidak dipakai
    df = df.drop(columns=[c for c in df.columns if c.endswith(".1")])

    # 4) Pastikan kolom harga & volume benar-benar numerik (float)
    for col in base_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 5) Pastikan kolom tanggal bertipe datetime
    df["date"] = pd.to_datetime(df["date"])

    return df


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
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

    result_list = []

    for ticker, g in df.groupby("ticker"):
        g = g.sort_values("date").copy()

        # ========== MOVING AVERAGE (SMA) ==========
        g["ma_5"] = g["close"].rolling(window=5, min_periods=5).mean()
        g["ma_10"] = g["close"].rolling(window=10, min_periods=10).mean()
        g["ma_20"] = g["close"].rolling(window=20, min_periods=20).mean()
        # MA252 (kurang lebih 1 tahun perdagangan)
        g["ma_252"] = g["close"].rolling(window=252, min_periods=252).mean()

        # ========== EMA ==========
        g["ema_21"] = g["close"].ewm(span=21, min_periods=21, adjust=False).mean()
        # EMA untuk MACD (standar: 12 & 26)
        g["ema_12"] = g["close"].ewm(span=12, min_periods=12, adjust=False).mean()
        g["ema_26"] = g["close"].ewm(span=26, min_periods=26, adjust=False).mean()

        # ========== RSI ==========
        g["rsi_14"] = compute_rsi(g["close"], period=14)
        g["rsi_63"] = compute_rsi(g["close"], period=63)  # RSI63

        # ========== RETURN, ROC, MOMENTUM ==========
        # log return harian (sudah kamu pakai)
        g["log_return_1d"] = np.log(g["close"] / g["close"].shift(1))

        # ROC63: persen perubahan 63 hari
        g["roc_63"] = g["close"].pct_change(periods=63)

        # MOM63: momentum = selisih harga
        g["mom_63"] = g["close"] - g["close"].shift(63)

        # Volatilitas 20 hari (std log_return_1d)
        g["vol_20"] = g["log_return_1d"].rolling(window=20, min_periods=20).std()

        # Rasio MA: MA5 / MA20 (ini yang kita pakai, bukan ma_5 & ma_20 mentah)
        g["ma_5_div_ma_20"] = g["ma_5"] / g["ma_20"]

        # ========== MACD ==========
        g["macd"] = g["ema_12"] - g["ema_26"]
        g["macd_signal"] = g["macd"].ewm(span=9, min_periods=9, adjust=False).mean()
        g["macd_hist"] = g["macd"] - g["macd_signal"]

        # ========== BOLLINGER BANDS (20 hari, 2*std) ==========
        rolling_std_20 = g["close"].rolling(window=20, min_periods=20).std()
        g["bb_middle_20"] = g["ma_20"]                      # middle band = SMA20
        g["bb_upper_20"] = g["ma_20"] + 2 * rolling_std_20  # upper band
        g["bb_lower_20"] = g["ma_20"] - 2 * rolling_std_20  # lower band

        result_list.append(g)

    df_ind = pd.concat(result_list, ignore_index=True)

    # 🔥 HAPUS fitur yang sangat kolinear / tidak dipakai sebagai input TFT:
    # - open, high, low: sangat kolinear dengan close
    # - ma_5, ma_10, ma_20: sangat kolinear dengan close + satu sama lain
    drop_cols = [
        "open",
        "high",
        "low",
        "ma_5",
        "ma_10",
        "ma_20",
        # opsional kalau tidak dipakai sama sekali:
        # "adj_close",
    ]
    df_ind = df_ind.drop(columns=[c for c in drop_cols if c in df_ind.columns])

    return df_ind


def main():
    if not os.path.exists(RAW_MERGED_PATH):
        raise FileNotFoundError(f"File harga gabungan tidak ditemukan: {RAW_MERGED_PATH}")

    print(f"[INFO] Loading raw prices from {RAW_MERGED_PATH}")
    # Baca mentah dulu (tanpa parse_dates, akan di-handle di clean_raw_prices)
    df_raw = pd.read_csv(RAW_MERGED_PATH)

    # 🔧 BERSIHKAN data mentah (buang baris aneh, gabung kolom *.1, ubah ke numerik)
    df_clean = clean_raw_prices(df_raw)

    # Tambah indikator teknikal
    df_ind = add_technical_indicators(df_clean)

    print(f"[INFO] Saving prices with indicators to {OUT_PATH}")
    df_ind.to_csv(OUT_PATH, index=False)
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
