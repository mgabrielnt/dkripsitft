import os
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import yaml

import lightning.pytorch as pl
from torch.utils.data import DataLoader

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer


# ==== Path dasar ====
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")
CONFIG_DATA_PATH = os.path.join(ROOT_DIR, "configs", "data.yaml")
CONFIG_MODEL_PATH = os.path.join(ROOT_DIR, "configs", "model_tft.yaml")
CONFIG_EXPERIMENTS_PATH = os.path.join(ROOT_DIR, "configs", "experiments.yaml")

TFT_MASTER_PATH = os.path.join(DATA_PROCESSED_DIR, "tft_master.csv")

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


def prepare_dataframe(df_all: pd.DataFrame, required_cols: List[str], label: str) -> pd.DataFrame:
    missing = [c for c in required_cols if c not in df_all.columns]
    if missing:
        raise ValueError(
            f"Kolom {missing} tidak ditemukan di tft_master.csv (dibutuhkan untuk {label})."
        )

    before = len(df_all)
    df_clean = df_all.dropna(subset=required_cols).copy()
    after = len(df_clean)
    print(f"[INFO] Drop baris NaN untuk {label}: {before} -> {after}")

    df_clean["time_idx"] = df_clean["time_idx"].astype("int64")
    df_clean["ticker"] = df_clean["ticker"].astype("category")

    return df_clean


def make_training_dataset(
    df_all: pd.DataFrame,
    data_cfg: Dict[str, Any],
    model_cfg: Dict[str, Any],
    use_sentiment: bool,
) -> TimeSeriesDataSet:
    """
    Bangun TimeSeriesDataSet untuk TRAIN saja.

    PENTING:
    - Daftar fitur HARUS sama dengan saat training:
      * train_tft_baseline.py  -> use_sentiment=False
      * train_tft_with_sentiment.py -> use_sentiment=True
    """
    target = model_cfg.get("target", "close")
    max_encoder_length = model_cfg.get("max_encoder_length", 60)
    max_prediction_length = model_cfg.get(
        "max_prediction_length",
        data_cfg.get("horizon", 5),
    )

    df_train = df_all[df_all["split"] == "train"].copy()

    static_categoricals = ["ticker"]
    static_reals: List[str] = []

    time_varying_known_reals = TIME_FEATURES
    time_varying_known_categoricals: List[str] = []

    # === Fitur teknikal VIF sehat, sama seperti train_tft_baseline.py ===
    time_varying_unknown_reals = BASE_FEATURES.copy()

    if use_sentiment:
        # === Tambahan fitur sentimen, sama persis dengan train_tft_with_sentiment.py ===
        time_varying_unknown_reals += SENTIMENT_FEATURES

    time_varying_unknown_categoricals: List[str] = []

    training_ds = TimeSeriesDataSet(
        df_train,
        time_idx="time_idx",
        target=target,
        group_ids=["ticker"],
        min_encoder_length=max_encoder_length // 2,
        max_encoder_length=max_encoder_length,
        min_prediction_length=max_prediction_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=static_categoricals,
        static_reals=static_reals,
        time_varying_known_categoricals=time_varying_known_categoricals,
        time_varying_known_reals=time_varying_known_reals,
        time_varying_unknown_categoricals=time_varying_unknown_categoricals,
        time_varying_unknown_reals=time_varying_unknown_reals,
        target_normalizer=GroupNormalizer(
            groups=["ticker"], transformation="softplus"
        ),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    return training_ds


def run_model_on_dataset(
    model_ckpt: str,
    test_ds: TimeSeriesDataSet,
    batch_size: int = 64,
    label: str = "MODEL",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load model dari checkpoint dan jalankan .predict() pada test_ds.

    Return:
      y_true_2d, y_pred_2d dalam shape (n_samples, horizon)
    """
    if not os.path.exists(model_ckpt):
        raise FileNotFoundError(f"Checkpoint '{model_ckpt}' tidak ditemukan")

    print(f"[INFO] Load model {label} dari checkpoint: {model_ckpt}")
    pl.seed_everything(42)

    model = TemporalFusionTransformer.load_from_checkpoint(model_ckpt)

    test_loader: DataLoader = test_ds.to_dataloader(
        train=False,
        batch_size=batch_size,
        num_workers=0,
    )

    preds_obj = model.predict(
        test_loader,
        return_y=True,
        trainer_kwargs=dict(accelerator="cpu"),
    )

    y_pred = preds_obj.output
    y_true_raw = preds_obj.y

    if isinstance(y_true_raw, (list, tuple)):
        y_true = y_true_raw[0]
    else:
        y_true = y_true_raw

    y_pred_np = y_pred.detach().cpu().numpy()
    y_true_np = y_true.detach().cpu().numpy()

    # Pastikan (n_samples, horizon)
    if y_pred_np.ndim == 3:
        y_pred_np = y_pred_np[..., 0]
    if y_true_np.ndim == 3:
        y_true_np = y_true_np[..., 0]

    return y_true_np, y_pred_np


def compute_metrics_per_horizon(
    y_true_2d: np.ndarray,
    y_pred_2d: np.ndarray,
):
    """
    Hitung MAE, RMSE, MAPE global dan per horizon.
    Format output disesuaikan dengan log backtest sebelumnya.
    """
    eps = 1e-8

    y_true_flat = y_true_2d.reshape(-1)
    y_pred_flat = y_pred_2d.reshape(-1)

    mae = float(np.mean(np.abs(y_pred_flat - y_true_flat)))
    rmse = float(np.sqrt(np.mean((y_pred_flat - y_true_flat) ** 2)))
    mape = float(
        np.mean(np.abs((y_pred_flat - y_true_flat) / (np.abs(y_true_flat) + eps))) * 100.0
    )

    print(f"  GLOBAL: MAE={mae:.4f}, RMSE={rmse:.4f}, MAPE={mape:.4f} %")

    horizon = y_true_2d.shape[1]
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


def build_windows(df_test: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Bagi test set menjadi 3 window waktu: early, mid, late.
    Berdasarkan kuantil time_idx test.
    """
    df_test = df_test.sort_values(["time_idx", "ticker"]).copy()

    q1 = df_test["time_idx"].quantile(1 / 3)
    q2 = df_test["time_idx"].quantile(2 / 3)

    early = df_test[df_test["time_idx"] <= q1].copy()
    mid = df_test[(df_test["time_idx"] > q1) & (df_test["time_idx"] <= q2)].copy()
    late = df_test[df_test["time_idx"] > q2].copy()

    print("\n[INFO] Window sizes (test set):")
    print(f"  early: {len(early)} rows")
    print(f"  mid  : {len(mid)} rows")
    print(f"  late : {len(late)} rows")

    return {
        "early": early,
        "mid": mid,
        "late": late,
    }


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

    df_all_base = prepare_dataframe(df_all_raw, REQUIRED_BASE_COLS, "baseline")

    print("\n========== BACKTEST TFT BASELINE ==========")

    # kolom wajib (tanpa sentimen, sama seperti evaluate_tft_models baseline)
    print("\n[INFO] NaN per kolom (sebelum cleaning) di df_all (baseline):")
    print(df_all_base[REQUIRED_BASE_COLS].isna().sum())

    df_test = df_all_base[df_all_base["split"] == "test"].copy()
    print(f"[INFO] Total test rows: {len(df_test)}")

    # bangun window waktu dari test set
    windows = build_windows(df_test)

    # ===== BASELINE =====
    if not baseline_ckpt or not os.path.exists(baseline_ckpt):
        print(f"[WARN] Checkpoint baseline tidak ditemukan ({baseline_ckpt}). Lewatkan BACKTEST BASELINE.")
    else:
        # training dataset baseline
        training_base = make_training_dataset(
            df_all=df_all_base,
            data_cfg=data_cfg,
            model_cfg=model_cfg,
            use_sentiment=False,
        )

        print("\n==== BASELINE ====\n")
        for win_name, df_win in windows.items():
            if df_win.empty:
                print(f"[Window: {win_name}] (skip, kosong)")
                continue

            test_ds = TimeSeriesDataSet.from_dataset(
                training_base,
                df_win,
                stop_randomization=True,
                predict=False,
            )

            print(f"[Window: {win_name}]")
            y_true, y_pred = run_model_on_dataset(
                model_ckpt=baseline_ckpt,
                test_ds=test_ds,
                batch_size=model_cfg.get("batch_size", 64),
                label="BASELINE",
            )
            compute_metrics_per_horizon(y_true, y_pred)
            print("")

    # ===== HYBRID =====
    print("\n========== BACKTEST TFT HYBRID ==========")

    if not hybrid_ckpt or not os.path.exists(hybrid_ckpt):
        print(f"[WARN] Checkpoint hybrid tidak ditemukan ({hybrid_ckpt}). Lewatkan BACKTEST HYBRID.")
        return

    # Pastikan semua kolom sentimen tersedia
    missing_sent_cols = [c for c in SENTIMENT_FEATURES if c not in df_all_raw.columns]
    if missing_sent_cols:
        print(f"[WARN] Kolom sentimen {missing_sent_cols} tidak ada di tft_master. Lewatkan BACKTEST HYBRID.")
        return

    df_all_hybrid = prepare_dataframe(df_all_raw, REQUIRED_HYBRID_COLS, "hybrid")

    training_hybrid = make_training_dataset(
        df_all=df_all_hybrid,
        data_cfg=data_cfg,
        model_cfg=model_cfg,
        use_sentiment=True,
    )

    print("\n==== HYBRID ====\n")
    for win_name, df_win in build_windows(df_all_hybrid[df_all_hybrid["split"] == "test"]).items():
        if df_win.empty:
            print(f"[Window: {win_name}] (skip, kosong)")
            continue

        test_ds = TimeSeriesDataSet.from_dataset(
            training_hybrid,
            df_win,
            stop_randomization=True,
            predict=False,
        )

        print(f"[Window: {win_name}]")
        y_true_h, y_pred_h = run_model_on_dataset(
            model_ckpt=hybrid_ckpt,
            test_ds=test_ds,
            batch_size=model_cfg.get("batch_size", 64),
            label="HYBRID",
        )
        compute_metrics_per_horizon(y_true_h, y_pred_h)
        print("")


if __name__ == "__main__":
    main()
