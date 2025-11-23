import os

import pandas as pd
import yaml

import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import MAE, QuantileLoss

# ==== Path dasar ====
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")
CONFIG_DATA_PATH = os.path.join(ROOT_DIR, "configs", "data.yaml")
CONFIG_MODEL_PATH = os.path.join(ROOT_DIR, "configs", "model_tft.yaml")

TFT_MASTER_PATH = os.path.join(DATA_PROCESSED_DIR, "tft_master.csv")
MODELS_DIR = os.path.join(ROOT_DIR, "models", "tft_baseline")
os.makedirs(MODELS_DIR, exist_ok=True)


def load_yaml(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    # ====== Load config ======
    data_cfg = load_yaml(CONFIG_DATA_PATH)
    model_cfg = load_yaml(CONFIG_MODEL_PATH)

    target = model_cfg.get("target", "close")

    max_encoder_length = model_cfg.get("max_encoder_length", 60)
    max_prediction_length = model_cfg.get(
        "max_prediction_length",
        data_cfg.get("horizon", 5),
    )

    batch_size = model_cfg.get("batch_size", 64)
    max_epochs = model_cfg.get("max_epochs", 50)

    # learning_rate dari YAML bisa string, paksa ke float
    learning_rate_raw = model_cfg.get("learning_rate", 5e-4)
    learning_rate = float(learning_rate_raw)

    accelerator = model_cfg.get("accelerator", "auto")
    loss_name = str(model_cfg.get("loss", "mae")).lower()

    # ====== Load data ======
    if not os.path.exists(TFT_MASTER_PATH):
        raise FileNotFoundError(f"Tidak ditemukan: {TFT_MASTER_PATH}")

    print(f"[INFO] Loading {TFT_MASTER_PATH}")
    df = pd.read_csv(TFT_MASTER_PATH, parse_dates=["date"])

    print("[INFO] Sample columns:", df.columns.tolist())
    print("[INFO] Split counts:")
    print(df["split"].value_counts())

    # Pastikan tipe data benar
    df["time_idx"] = df["time_idx"].astype("int64")
    df["ticker"] = df["ticker"].astype("category")

    # ====== Bersihkan NaN di kolom yang dipakai TFT baseline ======
    # hanya fitur dengan VIF yang sehat + target
    required_cols = [
        "time_idx",
        "ticker",
        "day_of_week",
        "month",
        "is_month_end",
        "close",            # target + lagged input
        "volume",
        "rsi_14",
        "log_return_1d",
        "vol_20",
        "ma_5_div_ma_20",
        "split",
    ]

    print("\n[INFO] NaN per kolom (sebelum cleaning):")
    print(df[required_cols].isna().sum())

    before_len = len(df)
    df = df.dropna(subset=required_cols).copy()
    after_len = len(df)
    print(f"[INFO] Drop baris dengan NaN di kolom wajib: {before_len} -> {after_len}")

    # ====== Bagi data train / val / test ======
    df_train = df[df["split"] == "train"].copy()
    df_val = df[df["split"] == "val"].copy()
    df_test = df[df["split"] == "test"].copy()

    print(f"[INFO] Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")

    # ====== Definisi fitur untuk TFT Baseline ======
    static_categoricals = ["ticker"]
    static_reals = []

    # Known future (bisa diketahui untuk masa depan)
    time_varying_known_reals = [
        "time_idx",
        "day_of_week",
        "month",
        "is_month_end",
    ]
    time_varying_known_categoricals = []

    # Observed unknown reals – fitur dengan VIF sehat
    time_varying_unknown_reals = [
        "close",
        "volume",
        "rsi_14",
        "log_return_1d",
        "vol_20",
        "ma_5_div_ma_20",
    ]
    time_varying_unknown_categoricals = []

    # ====== Buat TimeSeriesDataSet untuk TRAIN ======
    training = TimeSeriesDataSet(
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

    # ====== TimeSeriesDataSet untuk VALIDATION ======
    validation = TimeSeriesDataSet.from_dataset(
        training,
        df_val,
        stop_randomization=True,
    )

    print(f"[INFO] Len training dataset: {len(training)}, len validation: {len(validation)}")

    # ====== DataLoader ======
    train_dataloader = training.to_dataloader(
        train=True,
        batch_size=batch_size,
        num_workers=0,  # 0 untuk Windows
    )
    val_dataloader = validation.to_dataloader(
        train=False,
        batch_size=batch_size,
        num_workers=0,
    )

    # ====== Seed biar reproducible ======
    pl.seed_everything(42)

    # ====== Pilih loss (MAE untuk point forecast, QuantileLoss untuk probabilistic) ======
    if loss_name == "mae":
        loss = MAE()
        output_size = 1
        print("[INFO] Menggunakan loss MAE (point forecast, output_size=1)")
    else:
        loss = QuantileLoss()
        output_size = 7
        print("[INFO] Menggunakan QuantileLoss (probabilistic, output_size=7)")

    # ====== Buat model TFT dari dataset ======
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=learning_rate,
        hidden_size=model_cfg.get("hidden_size", 32),
        lstm_layers=model_cfg.get("lstm_layers", 2),
        dropout=model_cfg.get("dropout", 0.1),
        attention_head_size=model_cfg.get("attention_head_size", 4),
        hidden_continuous_size=model_cfg.get("hidden_continuous_size", 16),
        loss=loss,
        output_size=output_size,
        log_interval=10,
        reduce_on_plateau_patience=4,
    )

    try:
        print(f"[INFO] Model parameter count: {tft.size()}")
    except Exception:
        print("[INFO] Tidak bisa menghitung jumlah parameter dengan tft.size()")

    # ====== Callbacks ======
    lr_logger = LearningRateMonitor(logging_interval="epoch")
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=5,
        mode="min",
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=MODELS_DIR,
        filename="tft-baseline-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        save_top_k=1,
        mode="min",
    )

    # ====== Trainer ======
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        gradient_clip_val=0.1,
        callbacks=[lr_logger, early_stop_callback, checkpoint_callback],
        log_every_n_steps=10,
    )

    # ====== Train ======
    print("[INFO] Start training TFT baseline...")
    trainer.fit(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    print("[INFO] Training selesai.")
    print(f"[INFO] Model terbaik tersimpan di: {checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()
