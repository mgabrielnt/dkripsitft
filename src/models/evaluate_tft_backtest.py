# D:\skripsi\tft\src\models\evaluate_tft_backtest.py

import os
from typing import Dict, Any, List, Tuple

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
FORECAST_TIMELINE_PATH = os.path.join(DATA_PROCESSED_DIR, "tft_forecasts_timeline.csv")

# ==== Kolom wajib ====
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


def prepare_dataframe(df_all_raw: pd.DataFrame, required_cols: List[str]) -> pd.DataFrame:
    """
    Bersihkan tft_master:
      - pastikan semua kolom wajib ada
      - drop baris NaN di kolom wajib
      - cast time_idx & ticker ke tipe yang konsisten
    """
    missing = [c for c in required_cols if c not in df_all_raw.columns]
    if missing:
        raise ValueError(
            "Kolom berikut tidak ditemukan di tft_master.csv: " + ", ".join(missing)
        )

    before = len(df_all_raw)
    df_all = df_all_raw.dropna(subset=required_cols).copy()
    after = len(df_all)
    print(f"[INFO] Drop baris NaN untuk {len(required_cols)} kolom wajib: {before} -> {after}")

    df_all["time_idx"] = df_all["time_idx"].astype("int64")
    df_all["ticker"] = df_all["ticker"].astype("category")

    return df_all


def run_future_forecast(
    model_ckpt: str,
    df_all: pd.DataFrame,
    required_cols: List[str],
    batch_size: int = 64,
    label: str = "MODEL",
) -> Tuple[List[str], np.ndarray]:
    """
    Jalankan .predict() TFT untuk menghasilkan forecast future per ticker.

    Return:
        tickers_order: urutan ticker yang dipakai model
        y_pred_2d    : array (n_ticker, horizon)
    """
    if not os.path.exists(model_ckpt):
        raise FileNotFoundError(f"Checkpoint '{model_ckpt}' tidak ditemukan")

    print(f"[INFO] Drop baris NaN & siapkan dataframe untuk {label}")
    df_eval = df_all.dropna(subset=required_cols).copy()
    df_eval["time_idx"] = df_eval["time_idx"].astype("int64")
    df_eval["ticker"] = df_eval["ticker"].astype("category")
    df_eval = df_eval.sort_values(["ticker", "time_idx"])

    tickers_order = sorted(df_eval["ticker"].unique())
    print(f"[INFO] Tickers (urutan tetap untuk prediksi {label}): {tickers_order}")

    if not tickers_order:
        raise ValueError("Tidak ada ticker yang valid di dataframe untuk forecasting.")

    print(f"[INFO] Load model {label} dari checkpoint: {model_ckpt}")
    pl.seed_everything(42)
    model = TemporalFusionTransformer.load_from_checkpoint(model_ckpt)

    # PENTING: di sini TIDAK di-unpack jadi dua nilai lagi.
    preds_obj = model.predict(
        df_eval,
        mode="prediction",
        batch_size=batch_size,
        num_workers=0,
        trainer_kwargs=dict(accelerator="cpu"),
    )

    # Ambil tensor prediksi dari objek keluaran
    if hasattr(preds_obj, "prediction"):
        y_pred = preds_obj.prediction
    else:
        y_pred = preds_obj

    # Konversi ke numpy 2D: (n_ticker, horizon)
    if hasattr(y_pred, "detach"):
        y_pred_np = y_pred.detach().cpu().numpy()
    else:
        y_pred_np = np.asarray(y_pred)

    if y_pred_np.ndim == 3:
        y_pred_np = y_pred_np[..., 0]

    n_series, horizon = y_pred_np.shape
    print(f"[INFO] Bentuk prediksi {label}: {y_pred_np.shape} (seharusnya (n_ticker, max_prediction_length))")

    if n_series != len(tickers_order):
        print(
            f"[WARN] n_series ({n_series}) != jumlah ticker unik ({len(tickers_order)}). "
            "Pastikan konfigurasi TimeSeriesDataSet sesuai (group_ids=['ticker'])."
        )

    return tickers_order, y_pred_np


def build_timeline_with_forecast(
    df_all_raw: pd.DataFrame,
    tickers_base: List[str],
    y_pred_base_2d: np.ndarray,
    tickers_hybrid: List[str],
    y_pred_hybrid_2d: np.ndarray,
) -> pd.DataFrame:
    """
    Bangun dataframe timeline:
        - Baris historis: close dari tft_master, pred_* = NaN
        - Baris forecast: tanggal future per ticker, close=NaN, pred diisi
    """
    df_hist = df_all_raw[["date", "ticker", "split", "close"]].copy()
    df_hist = df_hist.sort_values(["ticker", "date"]).reset_index(drop=True)
    df_hist["pred_baseline"] = np.nan
    df_hist["pred_hybrid"] = np.nan

    # Mapping ticker -> pred arrays
    pred_base_map = {}
    if y_pred_base_2d is not None and tickers_base:
        for i, t in enumerate(tickers_base):
            pred_base_map[str(t)] = y_pred_base_2d[i]

    pred_hybrid_map = {}
    if y_pred_hybrid_2d is not None and tickers_hybrid:
        for i, t in enumerate(tickers_hybrid):
            pred_hybrid_map[str(t)] = y_pred_hybrid_2d[i]

    forecast_rows = []

    # Pastikan kolom date sudah Timestamp
    if not np.issubdtype(df_hist["date"].dtype, np.datetime64):
        df_hist["date"] = pd.to_datetime(df_hist["date"])

    tickers_all = sorted(df_hist["ticker"].unique())
    for ticker in tickers_all:
        df_t = df_hist[df_hist["ticker"] == ticker]
        if df_t.empty:
            continue

        last_date = df_t["date"].max()

        base_preds = pred_base_map.get(str(ticker))
        hybrid_preds = pred_hybrid_map.get(str(ticker))

        # Kalau kedua model tidak punya forecast untuk ticker ini -> skip future
        if base_preds is None and hybrid_preds is None:
            continue

        # Tentukan horizon dari salah satu yang ada
        if base_preds is not None:
            horizon = len(base_preds)
        else:
            horizon = len(hybrid_preds)

        # Generate tanggal bursa (hari kerja) ke depan
        future_dates = pd.bdate_range(
            start=last_date + pd.Timedelta(days=1),
            periods=horizon,
        )

        for step, d in enumerate(future_dates):
            forecast_rows.append(
                {
                    "date": d,
                    "ticker": ticker,
                    "split": "forecast",
                    "close": np.nan,
                    "pred_baseline": float(base_preds[step]) if base_preds is not None else np.nan,
                    "pred_hybrid": float(hybrid_preds[step]) if hybrid_preds is not None else np.nan,
                }
            )

    df_future = pd.DataFrame(forecast_rows)
    if df_future.empty:
        print("[WARN] Tidak ada baris forecast future yang dihasilkan.")
        df_timeline = df_hist.copy()
    else:
        df_timeline = pd.concat([df_hist, df_future], ignore_index=True)
        df_timeline = df_timeline.sort_values(["ticker", "date"]).reset_index(drop=True)

    return df_timeline


def main() -> None:
    if not os.path.exists(TFT_MASTER_PATH):
        raise FileNotFoundError(f"Tidak ditemukan: {TFT_MASTER_PATH}")

    print(f"[INFO] Membaca master dataset dari: {TFT_MASTER_PATH}")
    df_all_raw = pd.read_csv(TFT_MASTER_PATH, parse_dates=["date"])

    data_cfg = load_yaml(CONFIG_DATA_PATH)
    model_cfg = load_yaml(CONFIG_MODEL_PATH)
    exp_cfg = load_yaml(CONFIG_EXPERIMENTS_PATH)

    baseline_ckpts = exp_cfg["tft_baseline"]["checkpoint_paths"]
    hybrid_ckpts = exp_cfg["tft_with_sentiment"]["checkpoint_paths"]

    baseline_ckpt = baseline_ckpts[0] if baseline_ckpts else ""
    hybrid_ckpt = hybrid_ckpts[0] if hybrid_ckpts else ""

    batch_size = model_cfg.get("batch_size", 64)

    # ================= BASELINE =================
    if not baseline_ckpt or not os.path.exists(baseline_ckpt):
        raise FileNotFoundError(
            f"Checkpoint baseline tidak ditemukan atau kosong: '{baseline_ckpt}'. "
            "Pastikan train_tft_baseline sudah dijalankan dan experiments.yaml terupdate."
        )

    print("\n[STEP] Siapkan dataframe untuk BASELINE (tanpa fitur sentimen)")
    df_all_base = prepare_dataframe(df_all_raw, REQUIRED_BASE_COLS)

    tickers_base, y_pred_base_2d = run_future_forecast(
        model_ckpt=baseline_ckpt,
        df_all=df_all_base,
        required_cols=REQUIRED_BASE_COLS,
        batch_size=batch_size,
        label="BASELINE",
    )

    # ================= HYBRID =================
    tickers_hybrid: List[str] = []
    y_pred_hybrid_2d: np.ndarray = None  # type: ignore

    if hybrid_ckpt and os.path.exists(hybrid_ckpt):
        missing_sent_cols = [c for c in SENTIMENT_FEATURES if c not in df_all_raw.columns]
        if missing_sent_cols:
            print(
                f"[WARN] Kolom sentimen {missing_sent_cols} tidak ada di tft_master. "
                "Forecast HYBRID tidak akan dihitung."
            )
        else:
            print("\n[STEP] Siapkan dataframe untuk HYBRID (dengan fitur sentimen)")
            df_all_hybrid = prepare_dataframe(df_all_raw, REQUIRED_HYBRID_COLS)

            tickers_hybrid, y_pred_hybrid_2d = run_future_forecast(
                model_ckpt=hybrid_ckpt,
                df_all=df_all_hybrid,
                required_cols=REQUIRED_HYBRID_COLS,
                batch_size=batch_size,
                label="HYBRID",
            )
    else:
        print(
            f"[WARN] Checkpoint HYBRID tidak ditemukan ({hybrid_ckpt}). "
            "Timeline hanya akan berisi prediksi BASELINE."
        )

    # ================= BUILD TIMELINE =================
    print("\n[STEP] Bangun timeline historis + forecast future")
    df_timeline = build_timeline_with_forecast(
        df_all_raw=df_all_raw,
        tickers_base=tickers_base,
        y_pred_base_2d=y_pred_base_2d,
        tickers_hybrid=tickers_hybrid,
        y_pred_hybrid_2d=y_pred_hybrid_2d,
    )

    print(f"[INFO] Timeline forecast disimpan ke: {FORECAST_TIMELINE_PATH}")
    df_timeline.to_csv(FORECAST_TIMELINE_PATH, index=False)
    print("[INFO] Selesai.")


if __name__ == "__main__":
    main()
