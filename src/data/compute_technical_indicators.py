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
    """
    Tambahkan hanya indikator teknikal yang DIPAKAI di model akhir
    (berdasarkan hasil VIF):

    - close
    - volume
    - log_return_1d
    - vol_20        (std log_return_1d 20 hari)
    - rsi_14
    - ma_5_div_ma_20

    Indikator lain (MACD, Bollinger, RSI63, ROC, MOM, MA panjang, dll)
    SENGAJA tidak dihitung untuk menghindari fitur berlebih / multikolinear.
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

    result_list: list[pd.DataFrame] = []

    for ticker, g in df.groupby("ticker"):
        g = g.sort_values("date").copy()

        # ========== MOVING AVERAGE (hanya yang perlu untuk rasio) ==========
        # MA5 & MA20 cuma dipakai sebagai intermediate untuk ma_5_div_ma_20
        g["ma_5"] = g["close"].rolling(window=5, min_periods=5).mean()
        g["ma_20"] = g["close"].rolling(window=20, min_periods=20).mean()
        g["ma_5_div_ma_20"] = g["ma_5"] / g["ma_20"]

        # ========== RSI 14 (sesuai fitur VIF) ==========
        g["rsi_14"] = compute_rsi(g["close"], period=14)

        # ========== LOG RETURN & VOLATILITAS ==========
        # log return harian
        g["log_return_1d"] = np.log(g["close"] / g["close"].shift(1))

        # Volatilitas 20 hari (std log_return_1d)
        g["vol_20"] = g["log_return_1d"].rolling(window=20, min_periods=20).std()

        # Simpan group
        result_list.append(g)

    df_ind = pd.concat(result_list, ignore_index=True)

    # Buang kolom intermediate & kolom harga yang tidak dipakai di model
    drop_cols = [
        "open",
        "high",
        "low",
        "adj_close",
        "ma_5",
        "ma_20",
    ]
    df_ind = df_ind.drop(columns=[c for c in drop_cols if c in df_ind.columns])

    # Susun ulang kolom: identitas + fitur yang dipakai
    # (kolom lain di-drop supaya konsisten dengan desain fitur final)
    keep_cols = [
        "ticker",
        "date",
        "close",
        "volume",
        "log_return_1d",
        "vol_20",
        "rsi_14",
        "ma_5_div_ma_20",
    ]
    # Pastikan hanya kolom yang ada
    keep_cols = [c for c in keep_cols if c in df_ind.columns]
    df_ind = df_ind[keep_cols]

    return df_ind


def main():
    if not os.path.exists(RAW_MERGED_PATH):
        raise FileNotFoundError(f"File harga gabungan tidak ditemukan: {RAW_MERGED_PATH}")

    print(f"[INFO] Loading raw prices from {RAW_MERGED_PATH}")
    # Baca mentah dulu (tanpa parse_dates, akan di-handle di clean_raw_prices)
    df_raw = pd.read_csv(RAW_MERGED_PATH)

    # 🔧 BERSIHKAN data mentah (buang baris aneh, gabung kolom *.1, ubah ke numerik)
    df_clean = clean_raw_prices(df_raw)

    # Tambah indikator teknikal yang sudah diseleksi via VIF
    df_ind = add_technical_indicators(df_clean)

    print("[INFO] Kolom yang disimpan di prices_with_indicators.csv:")
    print(df_ind.columns.tolist())

    print(f"[INFO] Saving prices with indicators to {OUT_PATH}")
    df_ind.to_csv(OUT_PATH, index=False)
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
