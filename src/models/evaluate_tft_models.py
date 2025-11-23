import os
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import yaml

import lightning.pytorch as pl
from pytorch_forecasting import TemporalFusionTransformer

# ==== Path dasar ====
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")
CONFIG_DATA_PATH = os.path.join(ROOT_DIR, "configs", "data.yaml")
CONFIG_MODEL_PATH = os.path.join(ROOT_DIR, "configs", "model_tft.yaml")
CONFIG_EXPERIMENTS_PATH = os.path.join(ROOT_DIR, "configs", "experiments.yaml")

TFT_MASTER_PATH = os.path.join(DATA_PROCESSED_DIR, "tft_master.csv")
OUT_PRED_BASELINE = os.path.join(DATA_PROCESSED_DIR, "predictions_tft_baseline_test.csv")
OUT_PRED_HYBRID = os.path.join(DATA_PROCESSED_DIR, "predictions_tft_with_sentiment_test.csv")

TIME_FEATURES = ["time_idx", "day_of_week", "month", "is_month_end"]
BASE_FEATURES = [
    "close",
    "volume",
    "rsi_14",
    "log_return_1d",
    "vol_20",
    "ma_5_div_ma_20",
]
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


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def bucketize_sentiment(df: pd.DataFrame, threshold: float = 0.0) -> pd.DataFrame:
    """Ubah sentimen kontinu menjadi {-1,0,1} dengan ambang netral."""

    df = df.copy()
    for col in ["sentiment_mean", "sentiment_mean_3d"]:
        if col not in df.columns:
            continue

        values = df[col].astype(float)
        df[col] = values.apply(
            lambda v: 0.0 if abs(v) < threshold else (1.0 if v > 0 else (-1.0 if v < 0 else 0.0))
        )

    return df


def prepare_dataframe(df_all: pd.DataFrame, required_cols) -> pd.DataFrame:
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

    Kita tidak lagi bikin TimeSeriesDataSet manual.
    Pytorch Forecasting akan menggunakan dataset_parameters yang tersimpan
    di checkpoint, sehingga urutan & jumlah fitur selalu konsisten.
    """
    if not os.path.exists(model_ckpt):
        raise FileNotFoundError(f"Checkpoint '{model_ckpt}' tidak ditemukan")

    print(f"[INFO] Load model {label} dari checkpoint: {model_ckpt}")
    pl.seed_everything(42)

    model = TemporalFusionTransformer.load_from_checkpoint(model_ckpt)

    # Langsung kirim DataFrame.
    # Di dalam, BaseModel.predict akan:
    # - TimeSeriesDataSet.from_parameters(model.dataset_parameters, df_test, predict=True)
    # - bikin dataloader dan menjalankan trainer.predict
    preds_obj = model.predict(
        df_test,
        mode="prediction",
        return_y=True,
        batch_size=batch_size,
        num_workers=0,
        trainer_kwargs=dict(accelerator="cpu"),
    )

    # preds_obj.output: tensor prediksi (batch, horizon [,1])
    # preds_obj.y: target sebenarnya
    y_pred = preds_obj.prediction if hasattr(preds_obj, "prediction") else preds_obj.output
    y_true_raw = preds_obj.y

    # handle kalau y_true adalah tuple/list
    if isinstance(y_true_raw, (list, tuple)):
        y_true = y_true_raw[0]
    else:
        y_true = y_true_raw

    # ke numpy
    y_pred_np = y_pred.detach().cpu().numpy()
    y_true_np = y_true.detach().cpu().numpy()

    # Pastikan shape = (n_samples, horizon)
    if y_pred_np.ndim == 3:
        # misal (batch, horizon, 1) -> ambil dim terakhir
        y_pred_np = y_pred_np[..., 0]
    if y_true_np.ndim == 3:
        y_true_np = y_true_np[..., 0]

    return y_true_np, y_pred_np


def compute_metrics_per_horizon(
    y_true_2d: np.ndarray,
    y_pred_2d: np.ndarray,
    prefix: str = "",
):
    """
    Hitung MAE, RMSE, MAPE global dan per horizon.
    """
    eps = 1e-8

    # global (semua horizon digabung)
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

    # per horizon
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


def main():
    if not os.path.exists(TFT_MASTER_PATH):
        raise FileNotFoundError(f"Tidak ditemukan: {TFT_MASTER_PATH}")

    data_cfg = load_yaml(CONFIG_DATA_PATH)
    model_cfg = load_yaml(CONFIG_MODEL_PATH)
    exp_cfg = load_yaml(CONFIG_EXPERIMENTS_PATH)

    sentiment_repr = str(model_cfg.get("sentiment_representation", "raw")).lower()
    sentiment_threshold = float(model_cfg.get("sentiment_bucket_threshold", 0.0))

    baseline_ckpts = exp_cfg["tft_baseline"]["checkpoint_paths"]
    hybrid_ckpts = exp_cfg["tft_with_sentiment"]["checkpoint_paths"]

    baseline_ckpt = baseline_ckpts[0] if baseline_ckpts else ""
    hybrid_ckpt = hybrid_ckpts[0] if hybrid_ckpts else ""

    print(f"[INFO] Loading {TFT_MASTER_PATH}")
    df_all_raw = pd.read_csv(TFT_MASTER_PATH, parse_dates=["date"])

    if sentiment_repr == "sign":
        df_all_raw = bucketize_sentiment(df_all_raw, threshold=sentiment_threshold)
        print(
            f"[INFO] Representasi sentimen sign (-1/0/1) diaktifkan (threshold {sentiment_threshold})"
        )

    required_for_eval = REQUIRED_BASE_COLS.copy()
    sentiment_available = not [c for c in SENTIMENT_FEATURES if c not in df_all_raw.columns]
    if hybrid_ckpt and sentiment_available:
        required_for_eval = REQUIRED_HYBRID_COLS

    print("[INFO] Kolom tersedia:", df_all_raw.columns.tolist())

    # cek NaN di kolom wajib (baseline/hybrid)
    print("\n[INFO] NaN per kolom (sebelum cleaning) di df_all:")
    print(df_all_raw[required_for_eval].isna().sum())

    df_all = prepare_dataframe(df_all_raw, required_for_eval)

    # hanya info jumlah test
    test_rows = (df_all["split"] == "test").sum()
    print(f"\n[INFO] Test rows: {test_rows}")

    # DataFrame test (dipakai untuk baseline & hybrid supaya fair)
    df_test = df_all[df_all["split"] == "test"].copy()
    print("[INFO] Tickers di test set:", df_test["ticker"].unique().tolist())

    batch_size = model_cfg.get("batch_size", 64)

    # ================= BASELINE (tanpa sentimen) =================
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

    # simpan CSV sederhana (sekedar y_true / y_pred flatten)
    df_pred_base = pd.DataFrame(
        {
            "y_true": y_true_base_2d.reshape(-1),
            "y_pred": y_pred_base_2d.reshape(-1),
        }
    )
    df_pred_base.to_csv(OUT_PRED_BASELINE, index=False)
    print(f"[INFO] Simpan prediksi baseline ke: {OUT_PRED_BASELINE}")

    # ================= HYBRID (dengan sentimen) =================
    print("\n========== EVALUASI TFT HYBRID (dengan sentimen) ==========")

    if not hybrid_ckpt or not os.path.exists(hybrid_ckpt):
        print(f"[WARN] Checkpoint hybrid tidak ditemukan ({hybrid_ckpt}). Lewatkan evaluasi HYBRID dulu.")
        print("\n========== RINGKASAN (HANYA BASELINE) ==========")
        print(f"MAE  baseline = {mae_base:.4f}")
        print(f"RMSE baseline = {rmse_base:.4f}")
        print(f"MAPE baseline = {mape_base:.4f} %")
        return

    # Optional: cek apakah kolom sentimen ada
    missing_sent_cols = [c for c in SENTIMENT_FEATURES if c not in df_all.columns]
    if missing_sent_cols:
        print(f"[WARN] Kolom sentimen {missing_sent_cols} tidak ada di tft_master. Lewatkan evaluasi HYBRID.")
        print("\n========== RINGKASAN (HANYA BASELINE) ==========")
        print(f"MAE  baseline = {mae_base:.4f}")
        print(f"RMSE baseline = {rmse_base:.4f}")
        print(f"MAPE baseline = {mape_base:.4f} %")
        return

    # Hybrid: pakai df_test yang sama, tapi model HYBRID akan otomatis membaca
    # fitur sentimen + teknikal sesuai dataset_parameters di checkpoint.
    y_true_h_2d, y_pred_h_2d = run_model_on_df(
        model_ckpt=hybrid_ckpt,
        df_test=df_test,
        batch_size=batch_size,
        label="HYBRID",
    )

    mae_h, rmse_h, mape_h = compute_metrics_per_horizon(
        y_true_h_2d, y_pred_h_2d, prefix="HYBRID"
    )

    df_pred_h = pd.DataFrame(
        {
            "y_true": y_true_h_2d.reshape(-1),
            "y_pred": y_pred_h_2d.reshape(-1),
        }
    )
    df_pred_h.to_csv(OUT_PRED_HYBRID, index=False)
    print(f"[INFO] Simpan prediksi hybrid ke: {OUT_PRED_HYBRID}")

    # ========= RINGKASAN & PERSENTASE PERBAIKAN =========
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
