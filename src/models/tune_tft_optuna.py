# file: src/models/tune_tft_optuna.py
import os
import argparse

import optuna
import pandas as pd
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping

from pytorch_forecasting.metrics import MAE
from pytorch_forecasting import TemporalFusionTransformer

from .train_tft_baseline import (
    load_yaml as load_yaml_cfg,
    prepare_baseline_datasets,
)
from .train_tft_with_sentiment import prepare_hybrid_datasets

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")
TFT_MASTER_PATH = os.path.join(DATA_PROCESSED_DIR, "tft_master.csv")
CONFIG_DATA_PATH = os.path.join(ROOT_DIR, "configs", "data.yaml")
CONFIG_MODEL_PATH = os.path.join(ROOT_DIR, "configs", "model_tft.yaml")


def objective(
    trial: optuna.Trial,
    model_type: str,
    df: pd.DataFrame,
    data_cfg: dict,
    base_model_cfg: dict,
):
    """
    Objective untuk Optuna.

    model_type: "baseline" atau "hybrid"
    """
    # jaga konsistensi random di setiap trial
    pl.seed_everything(42)

    # copy base config supaya tidak mengubah dict asli
    model_cfg = dict(base_model_cfg)

    # ====== sampling hyperparameter ======
    if model_type == "baseline":
        model_cfg["hidden_size"] = trial.suggest_categorical(
            "hidden_size", [32, 64, 96]
        )
        model_cfg["dropout"] = trial.suggest_float("dropout", 0.05, 0.3)
    else:
        model_cfg["hidden_size_hybrid"] = trial.suggest_categorical(
            "hidden_size_hybrid", [64, 96, 128, 160]
        )
        model_cfg["dropout_hybrid"] = trial.suggest_float("dropout_hybrid", 0.1, 0.4)

    model_cfg["learning_rate"] = trial.suggest_float(
        "learning_rate", 1e-4, 2e-3, log=True
    )
    # epochs khusus untuk tuning (lebih pendek)
    model_cfg["max_epochs"] = 40

    batch_size = model_cfg.get("batch_size", 64)

    # ====== buat dataset (pakai split train/val dari kolom 'split') ======
    if model_type == "baseline":
        training, validation, _, _, _, _ = prepare_baseline_datasets(
            df, data_cfg, model_cfg
        )
    else:
        training, validation, _, _, _, _ = prepare_hybrid_datasets(
            df, data_cfg, model_cfg
        )

    train_dataloader = training.to_dataloader(
        train=True,
        batch_size=batch_size,
        num_workers=3,
    )
    val_dataloader = validation.to_dataloader(
        train=False,
        batch_size=batch_size,
        num_workers=3,
    )

    # ====== definisi model TFT ======
    loss = MAE()
    output_size = 1

    if model_type == "baseline":
        hidden_size = model_cfg.get("hidden_size", 64)
        dropout = model_cfg.get("dropout", 0.1)
    else:
        hidden_size = model_cfg.get(
            "hidden_size_hybrid", model_cfg.get("hidden_size", 64)
        )
        dropout = model_cfg.get("dropout_hybrid", 0.2)

    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=float(model_cfg["learning_rate"]),
        hidden_size=hidden_size,
        lstm_layers=model_cfg.get("lstm_layers", 2),
        dropout=dropout,
        attention_head_size=model_cfg.get("attention_head_size", 4),
        hidden_continuous_size=model_cfg.get("hidden_continuous_size", 32),
        loss=loss,
        output_size=output_size,
        log_interval=50,
        reduce_on_plateau_patience=4,
    )

    # ====== Trainer + EarlyStopping ======
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        mode="min",
    )

    trainer = pl.Trainer(
        max_epochs=model_cfg["max_epochs"],
        accelerator="auto",
        gradient_clip_val=0.1,
        callbacks=[early_stop],
        enable_checkpointing=False,
        logger=False,
    )

    trainer.fit(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    # ambil best val_loss dari callback_metrics
    val_metrics = trainer.callback_metrics
    val_loss = float(val_metrics["val_loss"].item())

    return val_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", choices=["baseline", "hybrid"], required=True)
    parser.add_argument("--n-trials", type=int, default=20)
    args = parser.parse_args()

    if not os.path.exists(TFT_MASTER_PATH):
        raise FileNotFoundError(f"Tidak ditemukan: {TFT_MASTER_PATH}")

    df = pd.read_csv(TFT_MASTER_PATH, parse_dates=["date"])
    data_cfg = load_yaml_cfg(CONFIG_DATA_PATH)
    model_cfg = load_yaml_cfg(CONFIG_MODEL_PATH)

    # seed global
    pl.seed_everything(42)

    def _objective(trial: optuna.Trial):
        return objective(trial, args.model_type, df, data_cfg, model_cfg)

    study = optuna.create_study(direction="minimize")
    study.optimize(_objective, n_trials=args.n_trials)

    print(f"=== Selesai tuning {args.model_type} ===")
    print("Best trial:")
    print("  value (val_loss) :", study.best_value)
    print("  params           :", study.best_params)


if __name__ == "__main__":
    main()
