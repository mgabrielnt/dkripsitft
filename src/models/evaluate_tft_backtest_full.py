# D:\skripsi\tft\src\models\evaluate_tft_backtest_full.py

import os
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import yaml

import lightning.pytorch as pl
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
import torch

# ============================================================
# PATH DASAR & KONFIG
# ============================================================
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")
CONFIG_DATA_PATH = os.path.join(ROOT_DIR, "configs", "data.yaml")
CONFIG_MODEL_PATH = os.path.join(ROOT_DIR, "configs", "model_tft.yaml")
CONFIG_EXPERIMENTS_PATH = os.path.join(ROOT_DIR, "configs", "experiments.yaml")

TFT_MASTER_PATH = os.path.join(DATA_PROCESSED_DIR, "tft_master.csv")
TFT_BACKTEST_FULL_PATH = os.path.join(DATA_PROCESSED_DIR, "tft_backtest_full.csv")

# Dataloader config (aman untuk Windows & Linux)
if os.name == "nt":  # Windows
    NUM_WORKERS = 2
else:
    cpu_count = os.cpu_count() or 2
    NUM_WORKERS = min(4, max(1, cpu_count - 1))

# Kolom wajib (sinkron dengan script lain)
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

REQUIRED_BASE_COLS = ["ticker", *TIME_FEATURES, *BASE_FEATURES, "split", "date"]
REQUIRED_HYBRID_COLS = [*REQUIRED_BASE_COLS, *SENTIMENT_FEATURES]


# ============================================================
# HELPER
# ============================================================

def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def prepare_dataframe(df_all_raw: pd.DataFrame, required_cols: List[str]) -> pd.DataFrame:
    """
    - Cek kolom wajib
    - Drop NaN di kolom wajib
    - Normalize tipe time_idx, date, ticker
    """
    missing = [c for c in required_cols if c not in df_all_raw.columns]
    if missing:
        raise ValueError("Missing required columns in tft_master.csv: " + ", ".join(missing))

    before = len(df_all_raw)
    df_all = df_all_raw.dropna(subset=required_cols).copy()
    after = len(df_all)
    print(f"[INFO] Drop NaN rows on {len(required_cols)} required cols: {before} -> {after}")

    df_all["time_idx"] = df_all["time_idx"].astype("int64")

    if not np.issubdtype(df_all["date"].dtype, np.datetime64):
        df_all["date"] = pd.to_datetime(df_all["date"])

    # ticker sebagai string (biar aman buat merge/join)
    df_all["ticker"] = df_all["ticker"].astype(str)

    df_all = df_all.sort_values(["ticker", "time_idx"]).reset_index(drop=True)
    return df_all


def to_numpy(a):
    """Convert tensor / array-like -> numpy."""
    if hasattr(a, "detach"):
        return a.detach().cpu().numpy()
    return np.asarray(a)


# ============================================================
# ROLLING BACKTEST
# ============================================================

def run_rolling_backtest(
    model_ckpt: str,
    df_all: pd.DataFrame,
    required_cols: List[str],
    batch_size: int,
    max_prediction_length: int,
    label: str,
    target_col: str = "close",
) -> pd.DataFrame:
    """
    Full rolling backtest untuk 1 model (baseline / hybrid).

    - Model mungkin dilatih pada target "close" atau "log_return_1d".
    - Di sini, kita selalu mengembalikan y_true & y_pred dalam UNIT HARGA CLOSE,
      dengan cara:
        * kalau target_col == "close"    → pakai langsung
        * kalau target_col == "log_return_1d"
              (diasumsikan log_return_1d = ln(C_t / C_{t-1})):
              y_pred_close = close_{t-1} * exp(y_pred_return)
    """
    if not os.path.exists(model_ckpt):
        raise FileNotFoundError(f"Checkpoint '{model_ckpt}' not found")

    print(f"\n[STEP] Rolling backtest for model: {label}")
    print(f"[INFO] Load checkpoint: {model_ckpt}")

    pl.seed_everything(42)
    model = TemporalFusionTransformer.load_from_checkpoint(model_ckpt)

    if getattr(model, "dataset_parameters", None) is None:
        raise RuntimeError(
            "Checkpoint TFT has no dataset_parameters. "
            "Make sure model was trained via TemporalFusionTransformer.from_dataset(...)."
        )

    # Filter & sort dataframe evaluasi
    df_eval = df_all.dropna(subset=required_cols).copy()
    df_eval["ticker"] = df_eval["ticker"].astype(str)

    print("[INFO] Build TimeSeriesDataSet for rolling backtest (predict=False, full windows)")
    dataset = TimeSeriesDataSet.from_parameters(
        model.dataset_parameters,
        df_eval,
        predict=False,           # full sliding windows, bukan hanya last window
        stop_randomization=True,
    )

    print(f"[INFO] Building DataLoader with num_workers={NUM_WORKERS}")
    dataloader = dataset.to_dataloader(
        train=False,
        batch_size=batch_size,
        num_workers=NUM_WORKERS,
    )

    print(f"[INFO] Number of samples (windows) in dataset {label}: {len(dataset)}")

    # Pilih device otomatis: GPU kalau ada, kalau tidak CPU
    print("[INFO] Run model.predict(..., mode='raw', return_x=True)")
    if torch.cuda.is_available():
        print("[INFO] Using GPU for prediction")
        trainer_kwargs = dict(accelerator="gpu", devices=1)
    else:
        print("[INFO] Using CPU for prediction")
        trainer_kwargs = dict(accelerator="cpu", devices=1)

    preds_obj = model.predict(
        dataloader,
        mode="raw",
        return_x=True,
        trainer_kwargs=trainer_kwargs,
    )

    # ----------------------
    # Ambil raw & x dari output predict()
    # ----------------------
    raw = None
    x = None

    if isinstance(preds_obj, (list, tuple)):
        # Versi lama: [raw, x, (index), ...] -> kita pakai 2 yang pertama
        if len(preds_obj) < 2:
            raise RuntimeError("Unexpected structure from predict(): too few elements.")
        raw = preds_obj[0]
        x = preds_obj[1]
    else:
        # Versi baru: object punya attr .prediction & .x
        raw = getattr(preds_obj, "prediction", preds_obj)
        x = getattr(preds_obj, "x", None)

    if isinstance(raw, dict):
        pred_tensor = raw.get("prediction", None)
        if pred_tensor is None:
            # fallback: ambil elemen pertama yang ada
            pred_tensor = list(raw.values())[0]
    elif hasattr(raw, "prediction"):
        pred_tensor = raw.prediction
    else:
        pred_tensor = raw

    if x is None:
        raise RuntimeError("predict(..., return_x=True) did not return x metadata.")

    # Beberapa versi membungkus x sebagai list/tuple
    if isinstance(x, (list, tuple)):
        if not x:
            raise RuntimeError("Empty x returned from predict().")
        x = x[0]

    if not isinstance(x, dict):
        if hasattr(x, "__dict__"):
            x = x.__dict__
        else:
            raise RuntimeError("Could not interpret x as dict from predict().")

    decoder_time_idx = x.get("decoder_time_idx", None)
    decoder_target = x.get("decoder_target", None)

    if decoder_time_idx is None or decoder_target is None:
        raise RuntimeError(
            "x does not contain 'decoder_time_idx' and 'decoder_target'. "
            "Keys available: " + ", ".join(x.keys())
        )

    # ----------------------
    # Konversi ke numpy
    # ----------------------
    y_pred = to_numpy(pred_tensor)
    dec_ti = to_numpy(decoder_time_idx).astype("int64")
    y_true = to_numpy(decoder_target)

    # Bentuk umum:
    # - MAE / MAPE / SMAPE, dst: (n_samples, horizon)
    # - QuantileLoss: (n_samples, horizon, n_quantiles)
    if y_pred.ndim == 3:
        # Kalau multi-quantile, ambil quantile median (biasanya idx tengah)
        if y_pred.shape[-1] > 1:
            median_idx = y_pred.shape[-1] // 2
            y_pred = y_pred[..., median_idx]
        else:
            y_pred = y_pred[..., 0]

    if y_true.ndim == 3:
        # target biasanya cuma 1 channel
        y_true = y_true[..., 0]

    n_samples, horizon_pred = y_pred.shape
    print(f"[INFO] y_pred shape: {y_pred.shape} (samples x horizon)")
    print(f"[INFO] y_true shape: {y_true.shape} (samples x horizon)")

    if horizon_pred != max_prediction_length:
        print(
            f"[WARN] horizon from model ({horizon_pred}) != "
            f"max_prediction_length from config ({max_prediction_length})"
        )

    if dec_ti.shape != y_pred.shape:
        raise RuntimeError(
            f"decoder_time_idx shape {dec_ti.shape} does not match predictions shape {y_pred.shape}"
        )

    # ----------------------
    # Build lookup index: (time_idx, target_rounded) -> row meta
    # target_col bisa "close" atau "log_return_1d"
    # ----------------------
    if target_col not in df_eval.columns:
        raise ValueError(f"target_col '{target_col}' tidak ada di df_eval")

    df_lookup = df_eval[["time_idx", "close", target_col, "ticker", "date", "split"]].copy()
    df_lookup["target_rounded"] = df_lookup[target_col].astype(float).round(6)
    df_lookup.set_index(["time_idx", "target_rounded"], inplace=True)

    records: List[Dict[str, Any]] = []

    for i in range(n_samples):
        for h in range(horizon_pred):
            ti = int(dec_ti[i, h])
            yt = float(y_true[i, h])      # nilai target di space training (close / return)
            yp = float(y_pred[i, h])

            key = (ti, round(yt, 6))
            if key not in df_lookup.index:
                # Biasanya ini padding di ujung seri / horizon yang incomplete
                continue

            row = df_lookup.loc[key]
            # Kalau entah kenapa ada duplikat, ambil baris pertama
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]

            ticker_val = str(row["ticker"])
            true_close = float(row["close"])

            # Default: kalau target_col = "close", pakai langsung
            if target_col == "close":
                pred_close = yp
                true_val = true_close
            else:
                # butuh harga close_{t-1} untuk rekonstruksi dari return
                ti_prev = ti - 1
                prev_rows = df_eval[
                    (df_eval["ticker"].astype(str) == ticker_val)
                    & (df_eval["time_idx"].astype("int64") == ti_prev)
                ]
                if prev_rows.empty:
                    # tidak ada harga sebelumnya (awal seri), skip
                    continue

                close_prev = float(prev_rows["close"].iloc[0])

                if target_col == "log_return_1d":
                    # asumsi: log_return_1d = ln(C_t / C_{t-1})
                    pred_close = close_prev * np.exp(yp)
                else:
                    # fallback: target dianggap sebagai selisih harga (diff)
                    pred_close = close_prev + yp

                true_val = true_close

            records.append(
                dict(
                    model=label.lower(),         # "baseline" / "hybrid"
                    ticker=ticker_val,
                    time_idx_input=None,
                    time_idx_target=ti,
                    date_target=row["date"],
                    horizon=int(h + 1),          # 1..max_prediction_length
                    y_true=true_val,             # harga close sebenarnya
                    y_pred=pred_close,           # harga close prediksi
                    split=row["split"],
                )
            )

    df_bt = pd.DataFrame.from_records(records)
    if df_bt.empty:
        print(f"[WARN] No backtest rows created for model {label}.")
        return df_bt

    df_bt = df_bt.sort_values(
        ["model", "ticker", "date_target", "horizon"]
    ).reset_index(drop=True)

    print(
        f"[INFO] Backtest {label}: {len(df_bt)} rows, "
        f"{df_bt['ticker'].nunique()} tickers, "
        f"{df_bt['horizon'].nunique()} horizons"
    )
    return df_bt


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    if not os.path.exists(TFT_MASTER_PATH):
        raise FileNotFoundError(f"{TFT_MASTER_PATH} not found")

    print(f"[INFO] Read master dataset from: {TFT_MASTER_PATH}")
    df_all_raw = pd.read_csv(TFT_MASTER_PATH, parse_dates=["date"])

    data_cfg = load_yaml(CONFIG_DATA_PATH)
    model_cfg = load_yaml(CONFIG_MODEL_PATH)
    exp_cfg = load_yaml(CONFIG_EXPERIMENTS_PATH)

    max_prediction_length = int(model_cfg.get("max_prediction_length", 5))
    batch_size = int(model_cfg.get("batch_size", 64))
    target_col = model_cfg.get("target", "close")

    print("[INFO] Model config:")
    print(f"       max_prediction_length = {max_prediction_length}")
    print(f"       batch_size            = {batch_size}")
    print(f"       target                = {target_col}")

    baseline_ckpts = exp_cfg["tft_baseline"]["checkpoint_paths"]
    hybrid_ckpts = exp_cfg["tft_with_sentiment"]["checkpoint_paths"]

    baseline_ckpt = baseline_ckpts[0] if baseline_ckpts else ""
    hybrid_ckpt = hybrid_ckpts[0] if hybrid_ckpts else ""

    # ================= BASELINE =================
    if not baseline_ckpt or not os.path.exists(baseline_ckpt):
        raise FileNotFoundError(
            f"Baseline checkpoint not found or empty: '{baseline_ckpt}'. "
            "Run train_tft_baseline and update experiments.yaml."
        )

    print("\n[STEP] Prepare dataframe for BASELINE (technical only)")
    df_all_base = prepare_dataframe(df_all_raw, REQUIRED_BASE_COLS)

    df_bt_list: List[pd.DataFrame] = []

    df_bt_base = run_rolling_backtest(
        model_ckpt=baseline_ckpt,
        df_all=df_all_base,
        required_cols=REQUIRED_BASE_COLS,
        batch_size=batch_size,
        max_prediction_length=max_prediction_length,
        label="baseline",
        target_col=target_col,
    )
    if not df_bt_base.empty:
        df_bt_list.append(df_bt_base)

    # ================= HYBRID =================
    if hybrid_ckpt and os.path.exists(hybrid_ckpt):
        missing_sent = [c for c in SENTIMENT_FEATURES if c not in df_all_raw.columns]
        if missing_sent:
            print(
                f"[WARN] Sentiment columns {missing_sent} not found in tft_master. "
                "HYBRID rolling backtest will be skipped."
            )
        else:
            print("\n[STEP] Prepare dataframe for HYBRID (technical + sentiment)")
            df_all_hybrid = prepare_dataframe(df_all_raw, REQUIRED_HYBRID_COLS)

            df_bt_hyb = run_rolling_backtest(
                model_ckpt=hybrid_ckpt,
                df_all=df_all_hybrid,
                required_cols=REQUIRED_HYBRID_COLS,
                batch_size=batch_size,
                max_prediction_length=max_prediction_length,
                label="hybrid",
                target_col=target_col,
            )
            if not df_bt_hyb.empty:
                df_bt_list.append(df_bt_hyb)
    else:
        print(
            f"[WARN] HYBRID checkpoint not found ({hybrid_ckpt}). "
            "Only BASELINE backtest will be available."
        )

    if not df_bt_list:
        raise RuntimeError("No backtest results produced (baseline/hybrid).")

    df_bt_all = pd.concat(df_bt_list, ignore_index=True)
    df_bt_all = df_bt_all.sort_values(
        ["ticker", "date_target", "horizon", "model"]
    ).reset_index(drop=True)

    print(f"\n[STEP] Save full rolling backtest to: {TFT_BACKTEST_FULL_PATH}")
    df_bt_all.to_csv(TFT_BACKTEST_FULL_PATH, index=False)
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
