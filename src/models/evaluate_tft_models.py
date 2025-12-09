# file: src/models/evaluate_tft_models.py

import os
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
import yaml

import lightning.pytorch as pl
from pytorch_forecasting import TemporalFusionTransformer

# ============================================================
# PATH & KONFIGURASI DASAR
# ============================================================

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")
CONFIG_DATA_PATH = os.path.join(ROOT_DIR, "configs", "data.yaml")
CONFIG_MODEL_PATH = os.path.join(ROOT_DIR, "configs", "model_tft.yaml")
CONFIG_EXPERIMENTS_PATH = os.path.join(ROOT_DIR, "configs", "experiments.yaml")

TFT_MASTER_PATH = os.path.join(DATA_PROCESSED_DIR, "tft_master.csv")

# Output standar (dipakai evaluate_tft_diagnostics)
OUT_PRED_BASELINE = os.path.join(DATA_PROCESSED_DIR, "predictions_tft_baseline_test.csv")
OUT_PRED_HYBRID = os.path.join(DATA_PROCESSED_DIR, "predictions_tft_with_sentiment_test.csv")

# Output tambahan untuk analisis & dashboard (forecast multi-horizon + bucket)
OUT_FORECAST_WITH_BUCKET = os.path.join(
    DATA_PROCESSED_DIR, "tft_forecasts_test_with_bucket.csv"
)

# ----------------------
# SET FITUR (MINIMAL UNTUK CEK & CLEANING)
# ----------------------

# Fitur waktu
TIME_FEATURES = ["time_idx", "day_of_week", "month", "is_month_end"]

# Fitur teknikal utama (baseline)
BASE_FEATURES = [
    "close",
    "volume",
    "log_return_1d",
    "vol_20",
    "rsi_14",
    "ma_5_div_ma_20",
    "bb_width_20",
    "volume_ma_ratio_20",
    "close_lag_2",
    "close_lag_3",
    "return_mean_5d",
    "return_std_5d",
]

# Fitur sentimen minimal yang dipakai untuk cek ketersediaan HYBRID
SENTIMENT_FEATURES = [
    "sentiment_mean",
    "news_count",
    "sentiment_mean_3d",
    "news_count_3d",
    "has_news",
    "sentiment_shock",
    "extreme_news",
]

REQUIRED_BASE_COLS = ["ticker", *TIME_FEATURES, *BASE_FEATURES, "split"]
REQUIRED_HYBRID_COLS = [*REQUIRED_BASE_COLS, *SENTIMENT_FEATURES]


# ============================================================
# UTIL: HEURISTIK DETEKSI FITUR SENTIMEN / NEWS
# (SAMA DENGAN TRAIN_TFT_WITH_SENTIMENT)
# ============================================================

def is_sentiment_col(name: str) -> bool:
    """
    Heuristik: semua kolom yang berkaitan sentiment / news / count berita.
    Dipakai untuk:
      - train_tft_with_sentiment (infer_feature_sets)
      - evaluate_tft_models (isi NaN sebelum prediksi)
    """
    name = name.lower()
    return (
        "sentiment" in name
        or "news" in name
        or name.startswith("pos_count")
        or name.startswith("neg_count")
        or name.startswith("neu_count")
        or "strong_market" in name
        or "strong_lex" in name
        or "has_news" in name
        or "extreme_news" in name
    )


# ============================================================
# UTIL LAIN
# ============================================================

def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def prepare_dataframe(df_all: pd.DataFrame, required_cols) -> pd.DataFrame:
    """
    - Cek kolom wajib ada semua.
    - Drop baris yang NA di kolom wajib.
    - Pastikan tipe time_idx & ticker benar.
    """
    missing = [c for c in required_cols if c not in df_all.columns]
    if missing:
        raise ValueError(
            "Kolom berikut tidak ditemukan di tft_master.csv: " + ", ".join(missing)
        )

    before = len(df_all)
    df_all = df_all.dropna(subset=required_cols).copy()
    after = len(df_all)
    print(f"[INFO] Drop baris dengan NaN di kolom wajib: {before} -> {after}")

    df_all["time_idx"] = df_all["time_idx"].astype("int64")
    df_all["ticker"] = df_all["ticker"].astype("category")

    return df_all


def run_model_on_df(
    model_ckpt: str,
    df_test: pd.DataFrame,
    batch_size: int = 64,
    label: str = "MODEL",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load model dari checkpoint dan jalankan .predict() langsung pada DataFrame test.

    Output:
        y_true_2d: (n_series, horizon)
        y_pred_2d: (n_series, horizon)

    n_series di sini biasanya = jumlah ticker,
    karena BaseModel.predict dengan mode="prediction" akan mengenerate
    satu deret horizon per seri.
    """
    if not os.path.exists(model_ckpt):
        raise FileNotFoundError(f"Checkpoint '{model_ckpt}' tidak ditemukan")

    print(f"[INFO] Load model {label} dari checkpoint: {model_ckpt}")
    pl.seed_everything(42)

    model = TemporalFusionTransformer.load_from_checkpoint(model_ckpt)

    preds_obj = model.predict(
        df_test,
        mode="prediction",
        return_y=True,
        batch_size=batch_size,
        num_workers=0,
        trainer_kwargs=dict(accelerator="cpu"),
    )

    # Beberapa versi pytorch-forecasting menyimpan di .prediction, beberapa di .output
    y_pred = getattr(preds_obj, "prediction", None)
    if y_pred is None:
        y_pred = preds_obj.output

    y_true_raw = preds_obj.y

    if isinstance(y_true_raw, (list, tuple)):
        y_true = y_true_raw[0]
    else:
        y_true = y_true_raw

    y_pred_np = y_pred.detach().cpu().numpy()
    y_true_np = y_true.detach().cpu().numpy()

    # shape -> (n_series, horizon)
    if y_pred_np.ndim == 3:
        y_pred_np = y_pred_np[..., 0]
    if y_true_np.ndim == 3:
        y_true_np = y_true_np[..., 0]

    print(
        f"[INFO] Bentuk prediksi {label}: {y_pred_np.shape} "
        f"(seharusnya (n_series, max_prediction_length))"
    )
    return y_true_np, y_pred_np


def build_detailed_pred_df(
    df_test: pd.DataFrame,
    tickers_order: List[str],
    y_true_2d: np.ndarray,
    y_pred_2d: np.ndarray,
) -> pd.DataFrame:
    """
    Bangun DataFrame prediksi multi-horizon dengan struktur:

        date, ticker, horizon, y_true, y_pred

    Asumsi:
        - y_true_2d, y_pred_2d memiliki shape (n_series, horizon)
        - n_series == len(tickers_order)
        - Setiap baris i merepresentasikan 1 ticker,
          dan tickers_order[i] adalah ticker-nya.
        - Untuk setiap ticker, kita anggap model memprediksi
          horizon ke depan dari tanggal terakhir di test set
          untuk ticker tersebut (last close).
    """
    n_series, horizon = y_pred_2d.shape

    if len(tickers_order) != n_series:
        raise ValueError(
            f"Mismatch: len(tickers_order)={len(tickers_order)} "
            f"berbeda dengan n_series={n_series} dari prediksi."
        )

    if "date" not in df_test.columns or "ticker" not in df_test.columns:
        raise ValueError("df_test harus memiliki kolom 'date' dan 'ticker'.")

    records = []

    for i, ticker in enumerate(tickers_order):
        df_t = df_test[df_test["ticker"] == ticker]
        if df_t.empty:
            continue

        last_date = df_t["date"].max()

        for h in range(horizon):
            records.append(
                {
                    "date": last_date,
                    "ticker": ticker,
                    "horizon": h + 1,  # H+1, H+2, ...
                    "y_true": float(y_true_2d[i, h]),
                    "y_pred": float(y_pred_2d[i, h]),
                }
            )

    df_out = pd.DataFrame.from_records(records)
    df_out = df_out.sort_values(["ticker", "date", "horizon"]).reset_index(drop=True)
    return df_out


def compute_metrics_per_horizon(
    y_true_2d: np.ndarray,
    y_pred_2d: np.ndarray,
    prefix: str = "",
) -> Tuple[float, float, float]:
    """
    Hitung metrik global (semua horizon) + print per horizon di console.
    """
    eps = 1e-8

    y_true_flat = y_true_2d.reshape(-1)
    y_pred_flat = y_pred_2d.reshape(-1)

    mae = float(np.mean(np.abs(y_pred_flat - y_true_flat)))
    rmse = float(np.sqrt(np.mean((y_pred_flat - y_true_flat) ** 2)))
    mape = float(
        np.mean(np.abs((y_pred_flat - y_true_flat) / (np.abs(y_true_flat) + eps))) * 100.0
    )

    print(f"[{prefix}] (GLOBAL semua horizon)")
    print(f"  MAE  (test)  = {mae:.4f}")
    print(f"  RMSE (test)  = {rmse:.4f}")
    print(f"  MAPE (test)  = {mape:.4f} %\n")

    horizon = y_true_2d.shape[1]
    print(f"[{prefix}] METRIK PER HORIZON:")
    for h in range(horizon):
        yt_h = y_true_2d[:, h]
        yp_h = y_pred_2d[:, h]

        mae_h = float(np.mean(np.abs(yp_h - yt_h)))
        rmse_h = float(np.sqrt(np.mean((yp_h - yt_h) ** 2)))
        mape_h = float(
            np.mean(np.abs((yp_h - yt_h) / (np.abs(yt_h) + eps))) * 100.0
        )
        print(
            f"  H+{h+1}: MAE={mae_h:.4f}, RMSE={rmse_h:.4f}, MAPE={mape_h:.4f} %"
        )

    return mae, rmse, mape


def safe_improvement(base: float, new: float) -> float:
    if base == 0:
        return 0.0
    return (base - new) / base * 100.0


def add_buckets_and_save_forecasts(
    df_base: pd.DataFrame,
    df_hybrid: pd.DataFrame,
    out_path: str,
):
    """
    Gabungkan forecast baseline + hybrid, tambahkan bucket_true & bucket_pred
    (berbasis kuartil y_true & y_pred global), lalu simpan ke CSV:

        tft_forecasts_test_with_bucket.csv

    File ini bisa dipakai dashboard / analisis bucket.
    """
    df_base = df_base.copy()
    if not df_base.empty:
        df_base["model"] = "baseline"

    df_hybrid = df_hybrid.copy()
    if not df_hybrid.empty:
        df_hybrid["model"] = "hybrid"

    df_all = pd.concat([df_base, df_hybrid], ignore_index=True)

    if df_all.empty:
        print("[WARN] Tidak ada data forecast untuk disimpan ke file with_bucket.")
        df_all.to_csv(out_path, index=False)
        print(f"[INFO] Menyimpan file kosong ke: {out_path}")
        return

    # ==== Bucket berbasis kuartil y_true & y_pred ====
    def make_bucket(series: pd.Series) -> pd.Series:
        s = series.astype(float)
        q = s.quantile([0.25, 0.5, 0.75])
        q1, q2, q3 = q.iloc[0], q.iloc[1], q.iloc[2]

        def _bucket(v: float) -> int:
            if np.isnan(v):
                return -1  # bucket khusus NaN (kalau ada)
            if v <= q1:
                return 0
            elif v <= q2:
                return 1
            elif v <= q3:
                return 2
            else:
                return 3

        return s.apply(_bucket)

    df_all["bucket_true"] = make_bucket(df_all["y_true"])
    df_all["bucket_pred"] = make_bucket(df_all["y_pred"])

    df_all = df_all.sort_values(["ticker", "date", "horizon", "model"]).reset_index(
        drop=True
    )

    df_all.to_csv(out_path, index=False)
    print(f"[INFO] Simpan forecast + bucket ke: {out_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    if not os.path.exists(TFT_MASTER_PATH):
        raise FileNotFoundError(f"Tidak ditemukan: {TFT_MASTER_PATH}")

    data_cfg = load_yaml(CONFIG_DATA_PATH)
    model_cfg = load_yaml(CONFIG_MODEL_PATH)
    exp_cfg = load_yaml(CONFIG_EXPERIMENTS_PATH)

    baseline_ckpts = exp_cfg["tft_baseline"]["checkpoint_paths"]
    hybrid_ckpts = exp_cfg["tft_with_sentiment"]["checkpoint_paths"]

    baseline_ckpt = baseline_ckpts[0] if baseline_ckpts else ""
    hybrid_ckpt = hybrid_ckpts[0] if hybrid_ckpts else ""

    print(f"[INFO] Loading {TFT_MASTER_PATH}")
    df_all_raw = pd.read_csv(TFT_MASTER_PATH, parse_dates=["date"])

    # ==== Isi NaN untuk SEMUA fitur sentimen/news ====
    # Menggunakan heuristik yang sama dengan train_tft_with_sentiment.py
    sentiment_cols_detected = [
        col for col in df_all_raw.columns if is_sentiment_col(col)
    ]
    print("[INFO] Detected sentiment/news cols:", sentiment_cols_detected)

    for col in sentiment_cols_detected:
        # fitur-fitur ini seharusnya numerik (indikator, count, skor sentimen)
        if pd.api.types.is_numeric_dtype(df_all_raw[col]):
            df_all_raw[col] = df_all_raw[col].fillna(0.0)

    # Pilih kolom yang wajib untuk evaluasi
    required_for_eval = REQUIRED_BASE_COLS.copy()
    sentiment_available = not [c for c in SENTIMENT_FEATURES if c not in df_all_raw.columns]
    if hybrid_ckpt and sentiment_available:
        required_for_eval = REQUIRED_HYBRID_COLS

    print("[INFO] Kolom tersedia di tft_master:", df_all_raw.columns.tolist())

    print("\n[INFO] NaN per kolom (sebelum cleaning) di df_all:")
    print(df_all_raw[required_for_eval].isna().sum())

    df_all = prepare_dataframe(df_all_raw, required_for_eval)

    test_rows = (df_all["split"] == "test").sum()
    print(f"\n[INFO] Test rows: {test_rows}")

    df_test = (
        df_all[df_all["split"] == "test"]
        .copy()
        .sort_values(["ticker", "time_idx"])
        .reset_index(drop=True)
    )
    tickers_order = list(df_test["ticker"].drop_duplicates())
    print("[INFO] Tickers di test set:", tickers_order)

    batch_size = model_cfg.get("batch_size", 64)

    # ================= BASELINE =================
    print("\n========== EVALUASI TFT BASELINE (tanpa sentimen) ==========")

    if not baseline_ckpt or not os.path.exists(baseline_ckpt):
        print(f"[ERROR] Checkpoint baseline tidak ditemukan ({baseline_ckpt}).")
        return

    y_true_base_2d, y_pred_base_2d = run_model_on_df(
        model_ckpt=baseline_ckpt,
        df_test=df_test,
        batch_size=batch_size,
        label="BASELINE",
    )

    mae_base, rmse_base, mape_base = compute_metrics_per_horizon(
        y_true_base_2d, y_pred_base_2d, prefix="BASELINE"
    )

    # Simpan format "flatten" untuk evaluate_tft_diagnostics
    df_pred_base_flat = pd.DataFrame(
        {
            "y_true": y_true_base_2d.reshape(-1),
            "y_pred": y_pred_base_2d.reshape(-1),
        }
    )
    df_pred_base_flat.to_csv(OUT_PRED_BASELINE, index=False)
    print(f"[INFO] Simpan prediksi baseline (flatten) ke: {OUT_PRED_BASELINE}")

    # Versi multi-horizon per ticker (dipakai file with_bucket)
    df_pred_base = build_detailed_pred_df(
        df_test=df_test,
        tickers_order=tickers_order,
        y_true_2d=y_true_base_2d,
        y_pred_2d=y_pred_base_2d,
    )

    # ================= HYBRID =================
    print("\n========== EVALUASI TFT HYBRID (dengan sentimen) ==========")

    if not hybrid_ckpt or not os.path.exists(hybrid_ckpt):
        print(f"[WARN] Checkpoint hybrid tidak ditemukan ({hybrid_ckpt}). Lewatkan evaluasi HYBRID dulu.")
        print("\n========== RINGKASAN (HANYA BASELINE) ==========")
        print(f"MAE  baseline = {mae_base:.4f}")
        print(f"RMSE baseline = {rmse_base:.4f}")
        print(f"MAPE baseline = {mape_base:.4f} %")
        # Walaupun hybrid tidak ada, simpan file with_bucket untuk baseline saja
        add_buckets_and_save_forecasts(df_pred_base, pd.DataFrame(), OUT_FORECAST_WITH_BUCKET)
        return

    missing_sent_cols = [c for c in SENTIMENT_FEATURES if c not in df_all.columns]
    if missing_sent_cols:
        print(f"[WARN] Kolom sentimen {missing_sent_cols} tidak ada di tft_master. Lewatkan evaluasi HYBRID.")
        print("\n========== RINGKASAN (HANYA BASELINE) ==========")
        print(f"MAE  baseline = {mae_base:.4f}")
        print(f"RMSE baseline = {rmse_base:.4f}")
        print(f"MAPE baseline = {mape_base:.4f} %")
        add_buckets_and_save_forecasts(df_pred_base, pd.DataFrame(), OUT_FORECAST_WITH_BUCKET)
        return

    y_true_h_2d, y_pred_h_2d = run_model_on_df(
        model_ckpt=hybrid_ckpt,
        df_test=df_test,
        batch_size=batch_size,
        label="HYBRID",
    )

    mae_h, rmse_h, mape_h = compute_metrics_per_horizon(
        y_true_h_2d, y_pred_h_2d, prefix="HYBRID"
    )

    # Simpan format "flatten" untuk evaluate_tft_diagnostics
    df_pred_h_flat = pd.DataFrame(
        {
            "y_true": y_true_h_2d.reshape(-1),
            "y_pred": y_pred_h_2d.reshape(-1),
        }
    )
    df_pred_h_flat.to_csv(OUT_PRED_HYBRID, index=False)
    print(f"[INFO] Simpan prediksi hybrid (flatten) ke: {OUT_PRED_HYBRID}")

    # Versi multi-horizon per ticker (dipakai dashboard / bucket)
    df_pred_h = build_detailed_pred_df(
        df_test=df_test,
        tickers_order=tickers_order,
        y_true_2d=y_true_h_2d,
        y_pred_2d=y_pred_h_2d,
    )

    # Gabung baseline + hybrid, tambahkan bucket, dan simpan file untuk dashboard
    add_buckets_and_save_forecasts(df_pred_base, df_pred_h, OUT_FORECAST_WITH_BUCKET)

    improv_mae = safe_improvement(mae_base, mae_h)
    improv_rmse = safe_improvement(rmse_base, rmse_h)
    improv_mape = safe_improvement(mape_base, mape_h)

    print("\n========== RINGKASAN GLOBAL (BASELINE vs HYBRID) ==========")
    print(f"MAE  baseline = {mae_base:.4f}, hybrid = {mae_h:.4f}, perbaikan = {improv_mae:.2f} %")
    print(f"RMSE baseline = {rmse_base:.4f}, hybrid = {rmse_h:.4f}, perbaikan = {improv_rmse:.2f} %")
    print(f"MAPE baseline = {mape_base:.4f} %, hybrid = {mape_h:.4f} %, perbaikan = {improv_mape:.2f} %")
    print("Catatan: nilai perbaikan positif berarti hybrid lebih baik (error lebih kecil).")


if __name__ == "__main__":
    main()
