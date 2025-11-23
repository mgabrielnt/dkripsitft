import os
import numpy as np
import pandas as pd

from statsmodels.stats.outliers_influence import variance_inflation_factor

# Lokasi root project
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")

TFT_MASTER_PATH = os.path.join(DATA_PROCESSED_DIR, "tft_master.csv")
OUT_VIF_PATH = os.path.join(DATA_PROCESSED_DIR, "vif_features.csv")


def main():
    if not os.path.exists(TFT_MASTER_PATH):
        raise FileNotFoundError(f"Tidak ditemukan: {TFT_MASTER_PATH}")

    print(f"[INFO] Loading {TFT_MASTER_PATH}")
    df = pd.read_csv(TFT_MASTER_PATH, parse_dates=["date"])

    # Pakai hanya data TRAIN untuk hitung VIF
    df_train = df[df["split"] == "train"].copy()

    # Kandidat fitur kontinu (sesuaikan kalau mau)
    candidate_cols = [
        "open", "high", "low", "close",
        "volume",
        "ma_5", "ma_10", "ma_20",
        "rsi_14",
        "log_return_1d",
        "vol_20",
        "ma_5_div_ma_20",
        "sentiment_mean",
        "news_count",
        "sentiment_mean_3d",
        "news_count_3d",
    ]

    feature_cols = [c for c in candidate_cols if c in df_train.columns]
    if not feature_cols:
        raise ValueError("Tidak ada kolom kandidat fitur yang ditemukan di tft_master.")

    X = df_train[feature_cols].copy()

    # 1) Bersihkan NaN & inf
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.dropna(axis=0, how="any")

    # 2) Drop kolom yang konstan (std == 0) → kalau tidak, std=0 bikin inf
    std_series = X.std(ddof=0)
    zero_std_cols = std_series[std_series == 0].index.tolist()
    if zero_std_cols:
        print(f"[WARN] Kolom dengan std=0 (konstan) akan dibuang sebelum VIF: {zero_std_cols}")
        X = X.drop(columns=zero_std_cols)

    # Update list fitur setelah buang kolom konstan
    feature_cols = list(X.columns)

    # 3) Standardisasi dulu biar VIF lebih stabil
    X_std = (X - X.mean()) / X.std(ddof=0)

    # 4) Bersihkan lagi inf / NaN setelah standardisasi
    X_std = X_std.replace([np.inf, -np.inf], np.nan)
    X_std = X_std.dropna(axis=0, how="any")

    print(f"[INFO] Jumlah baris untuk VIF: {len(X_std)}, jumlah fitur: {len(feature_cols)}")

    vif_rows = []
    for i, col in enumerate(X_std.columns):
        vif_val = variance_inflation_factor(X_std.values, i)
        vif_rows.append({"feature": col, "VIF": float(vif_val)})

    vif_df = pd.DataFrame(vif_rows).sort_values("VIF", ascending=False)

    print("\n[INFO] Hasil VIF (dari terbesar):")
    print(vif_df)

    print(f"\n[INFO] Simpan VIF ke: {OUT_VIF_PATH}")
    vif_df.to_csv(OUT_VIF_PATH, index=False)


if __name__ == "__main__":
    main()
