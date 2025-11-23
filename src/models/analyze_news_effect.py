import os
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error

import lightning.pytorch as pl
from torch.utils.data import DataLoader

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer


# ==== Path dasar & config ====
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


def prepare_dataframe(df_all: pd.DataFrame) -> pd.DataFrame:
    """
    Pastikan kolom wajib tersedia & tidak ada NaN untuk baseline/hybrid.
    Untuk perbandingan yang adil, kita pakai subset data yang memiliki
    seluruh kolom (teknikal + sentimen).
    """

    missing = [c for c in REQUIRED_HYBRID_COLS if c not in df_all.columns]
    if missing:
        raise ValueError(
            "Kolom berikut tidak ditemukan di tft_master.csv: " + ", ".join(missing)
        )

    before = len(df_all)
    df_all = df_all.dropna(subset=REQUIRED_HYBRID_COLS).copy()
    after = len(df_all)
    print(f"[INFO] Drop baris dengan NaN di kolom wajib: {before} -> {after}")

    df_all["time_idx"] = df_all["time_idx"].astype("int64")
    df_all["ticker"] = df_all["ticker"].astype("category")

    return df_all


def make_datasets_for_mode(
    df_all: pd.DataFrame,
    data_cfg: Dict[str, Any],
    model_cfg: Dict[str, Any],
    use_sentiment: bool,
) -> Tuple[TimeSeriesDataSet, TimeSeriesDataSet, pd.DataFrame]:
    """
    Membuat TimeSeriesDataSet untuk train dan test
    dengan opsi menggunakan fitur sentimen atau tidak.
    Disamakan dengan evaluate_tft_backtest.py
    """
    target = model_cfg.get("target", "close")
    max_encoder_length = model_cfg.get("max_encoder_length", 60)
    max_prediction_length = model_cfg.get(
        "max_prediction_length",
        data_cfg.get("horizon", 5),
    )

    df_train = df_all[df_all["split"] == "train"].copy()
    df_test = df_all[df_all["split"] == "test"].copy()

    static_categoricals = ["ticker"]
    static_reals: list[str] = []

    time_varying_known_reals = TIME_FEATURES
    time_varying_known_categoricals: list[str] = []

    time_varying_unknown_reals = BASE_FEATURES.copy()
    if use_sentiment:
        time_varying_unknown_reals += SENTIMENT_FEATURES

    time_varying_unknown_categoricals: list[str] = []

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
            groups=["ticker"],
            transformation="softplus",
        ),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    test_ds = TimeSeriesDataSet.from_dataset(
        training_ds,
        df_test,
        stop_randomization=True,
        predict=False,
    )

    return training_ds, test_ds, df_test


def get_h1_predictions(
    model_ckpt: str,
    test_ds: TimeSeriesDataSet,
    batch_size: int,
    label: str = "MODEL",
) -> pd.DataFrame:
    """
    Mengambil prediksi H+1 (horizon pertama) untuk seluruh sample test.

    Return DataFrame:
      - time_idx
      - y_true (H+1)
      - y_pred (H+1)
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

    # Prediksi RAW + input X (kompatibel versi baru & lama pytorch-forecasting)
    pred_obj = model.predict(
        test_loader,
        mode="raw",
        return_x=True,
        trainer_kwargs=dict(accelerator="cpu"),
    )

    if hasattr(pred_obj, "output") and hasattr(pred_obj, "x"):
        raw_predictions = pred_obj.output
        x = pred_obj.x
    elif isinstance(pred_obj, (list, tuple)) and len(pred_obj) >= 2:
        raw_predictions, x = pred_obj[0], pred_obj[1]
    else:
        raise RuntimeError(
            f"Tidak bisa menginterpretasi hasil model.predict, tipe: {type(pred_obj)}"
        )

    # pred: [batch, horizon, 1]
    pred = raw_predictions["prediction"]              # tensor [B, H, 1]
    decoder_target = x["decoder_target"]              # tensor [B, H]
    decoder_time_idx = x["decoder_time_idx"]          # tensor [B, H]

    pred_np = pred.detach().cpu().numpy().squeeze(-1)        # [B, H]
    target_np = decoder_target.detach().cpu().numpy()        # [B, H]
    time_idx_np = decoder_time_idx.detach().cpu().numpy()    # [B, H]

    # Ambil hanya horizon pertama (H+1)
    y_pred_h1 = pred_np[:, 0]
    y_true_h1 = target_np[:, 0]
    time_idx_h1 = time_idx_np[:, 0].astype("int64")

    df_pred = pd.DataFrame(
        {
            "time_idx": time_idx_h1,
            "y_true": y_true_h1,
            "y_pred": y_pred_h1,
        }
    )

    return df_pred


def metrics(y_true: pd.Series, y_pred: pd.Series):
    mae = mean_absolute_error(y_true, y_pred)
    # sklearn < 0.22 tidak mendukung argumen squared; fallback manual untuk kompatibilitas
    try:
        rmse = mean_squared_error(y_true, y_pred, squared=False)
    except TypeError:
        rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mape = (
        (y_true - y_pred)
        .abs()
        / y_true.abs().clip(lower=1e-6)
    ).mean() * 100
    return mae, rmse, mape


def main():
    if not os.path.exists(TFT_MASTER_PATH):
        raise FileNotFoundError(f"Tidak ditemukan: {TFT_MASTER_PATH}")

    data_cfg = load_yaml(CONFIG_DATA_PATH)
    model_cfg = load_yaml(CONFIG_MODEL_PATH)
    exp_cfg = load_yaml(CONFIG_EXPERIMENTS_PATH)

    sentiment_repr = str(model_cfg.get("sentiment_representation", "raw")).lower()
    sentiment_threshold = float(model_cfg.get("sentiment_bucket_threshold", 0.0))

    print(f"[INFO] Loading {TFT_MASTER_PATH}")
    df_all_raw = pd.read_csv(TFT_MASTER_PATH, parse_dates=["date"])
    df_all = prepare_dataframe(df_all_raw)

    df_all = prepare_dataframe(df_all_raw)

    baseline_ckpt = exp_cfg["tft_baseline"]["checkpoint_paths"][0]
    hybrid_ckpt = exp_cfg["tft_with_sentiment"]["checkpoint_paths"][0]

    # ==== Siapkan dataset TEST untuk baseline & hybrid ====
    _, test_ds_base, df_test_base = make_datasets_for_mode(
        df_all=df_all,
        data_cfg=data_cfg,
        model_cfg=model_cfg,
        use_sentiment=False,
    )
    _, test_ds_hyb, df_test_hyb = make_datasets_for_mode(
        df_all=df_all,
        data_cfg=data_cfg,
        model_cfg=model_cfg,
        use_sentiment=True,
    )

    batch_size = model_cfg.get("batch_size", 64)

    # ==== Prediksi H+1 baseline & hybrid (per sample, sama-sama dari test set) ====
    df_pred_base = get_h1_predictions(
        model_ckpt=baseline_ckpt,
        test_ds=test_ds_base,
        batch_size=batch_size,
        label="BASELINE",
    )
    df_pred_hyb = get_h1_predictions(
        model_ckpt=hybrid_ckpt,
        test_ds=test_ds_hyb,
        batch_size=batch_size,
        label="HYBRID",
    )

    # ==== Align baseline & hybrid berdasarkan index (sample ke-n sama) ====
    if len(df_pred_base) != len(df_pred_hyb):
        raise ValueError(
            f"Jumlah sample prediksi baseline ({len(df_pred_base)}) "
            f"!= hybrid ({len(df_pred_hyb)})."
        )

    # Cek (opsional) konsistensi time_idx
    if not np.array_equal(df_pred_base["time_idx"].values, df_pred_hyb["time_idx"].values):
        print("[WARN] time_idx baseline vs hybrid tidak identik urutannya. "
              "Tetap lanjut dengan alignment berdasarkan index.")

    # Bangun dataframe prediksi gabungan per sample
    df_pred = pd.DataFrame(
        {
            "time_idx": df_pred_base["time_idx"].values,
            "y_true": df_pred_base["y_true"].values,
            "y_pred_base": df_pred_base["y_pred"].values,
            "y_pred_hyb": df_pred_hyb["y_pred"].values,
        }
    )

    # ==== Ambil info news_count per time_idx dari test split ====
    df_test_all = df_all[df_all["split"] == "test"].copy()

    news_info = (
        df_test_all[["time_idx", "date", "news_count"]]
        .groupby(["time_idx", "date"], as_index=False)
        .agg({"news_count": "max"})  # ada berita minimal satu ticker di tanggal tsb
    )

    df_merged = df_pred.merge(
        news_info,
        on="time_idx",
        how="left",
    )

    df_merged["news_count"] = df_merged["news_count"].fillna(0)

    # ==== Split subset: ALL, NEWS>0, NO_NEWS ====
    df_all_subset = df_merged
    df_news = df_merged[df_merged["news_count"] > 0].copy()
    df_nonews = df_merged[df_merged["news_count"] == 0].copy()

    print("[INFO] Jumlah sampel H+1 (per sample, sliding window):", len(df_all_subset))
    print("[INFO]  - Dengan berita (news_count > 0):", len(df_news))
    print("[INFO]  - Tanpa berita (news_count == 0):", len(df_nonews))

    # ==== Cetak metrik ====
    for subset_name, subset in [
        ("ALL", df_all_subset),
        ("NEWS>0", df_news),
        ("NO_NEWS", df_nonews),
    ]:
        print(f"\n==== {subset_name} (H+1) ====")
        if subset.empty:
            print("Tidak ada data pada subset ini.")
            continue

        for label, col in [
            ("BASELINE", "y_pred_base"),
            ("HYBRID", "y_pred_hyb"),
        ]:
            mae, rmse, mape = metrics(subset["y_true"], subset[col])
            print(
                f"{label} | MAE={mae:.4f}, RMSE={rmse:.4f}, MAPE={mape:.4f}%"
            )


if __name__ == "__main__":
    main()
