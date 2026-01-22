import os
import yaml
import pandas as pd
import torch
import warnings

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from pathlib import Path
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss

warnings.filterwarnings("ignore", category=UserWarning)

# --- OPTIMISASI GPU ---
torch.set_float32_matmul_precision('medium')

def train_baseline():
    # --- 1. SETUP PATH ---
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parent.parent
    CONFIG_PATH = PROJECT_ROOT / "configs" / "model_tft.yaml"
    
    print(f"Loading config from: {CONFIG_PATH}")
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config file not found at: {CONFIG_PATH}")

    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    DATA_PATH = PROJECT_ROOT / config['data']['dataset_path']
    print(f"Loading data from: {DATA_PATH}")

    # --- 2. Load Data ---
    df = pd.read_csv(DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    df['ticker'] = df['ticker'].astype(str)
    df['day_of_week'] = df['day_of_week'].astype(str)
    df['month'] = df['month'].astype(str)

    # --- 3. Definisikan Dataset (BASELINE: HANYA TEKNIKAL) ---
    training_cutoff = df["time_idx"].max() - config['data']['max_prediction_length']

    training_dataset = TimeSeriesDataSet(
        df[lambda x: x.time_idx <= training_cutoff],
        time_idx=config['data']['time_idx'],
        target=config['data']['target'],
        group_ids=config['data']['group_ids'],
        min_encoder_length=config['data']['min_encoder_length'],
        max_encoder_length=config['data']['max_encoder_length'],
        max_prediction_length=config['data']['max_prediction_length'],
        static_categoricals=["ticker"],
        time_varying_known_categoricals=["month", "day_of_week"],
        time_varying_known_reals=["time_idx", "is_month_end"],
        # HANYA TEKNIKAL
        time_varying_unknown_reals=[
            "close", "volume", "log_return_1d", "vol_20", "rsi_14", 
            "ma_5_div_ma_20", "bb_width_20", "gap_return_1d", "intraday_range_pct"
        ],
        target_normalizer=GroupNormalizer(groups=["ticker"], transformation="softplus"),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    validation_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset, df, predict=True, stop_randomization=True
    )

    train_dataloader = training_dataset.to_dataloader(
        train=True, batch_size=config['training']['batch_size'], num_workers=0
    )
    val_dataloader = validation_dataset.to_dataloader(
        train=False, batch_size=config['training']['batch_size'] * 2, num_workers=0
    )

    # --- 4. Inisialisasi Model ---
    tft = TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=config['model']['learning_rate'],
        hidden_size=config['model']['hidden_size'],
        attention_head_size=config['model']['attention_head_size'],
        dropout=config['model']['dropout'],
        hidden_continuous_size=config['model']['hidden_continuous_size'],
        loss=QuantileLoss(),
    )

    # --- 5. Setup Trainer ---
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=1e-4,
        patience=config['training']['patience'],
        verbose=False,
        mode="min"
    )
    
    CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "baseline"
    LOG_DIR = PROJECT_ROOT / "logs"

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=CHECKPOINT_DIR,
        filename="tft-baseline-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        mode="min"
    )

    logger = TensorBoardLogger(save_dir=LOG_DIR, name="baseline_model")

    trainer = pl.Trainer(
        max_epochs=config['training']['max_epochs'],
        accelerator="auto",
        gradient_clip_val=config['training']['gradient_clip_val'],
        callbacks=[early_stop_callback, checkpoint_callback],
        logger=logger,
        enable_model_summary=True
    )

    # --- 6. Mulai Training ---
    print("\n" + "="*50)
    print("   MEMULAI TRAINING BASELINE MODEL")
    print("="*50 + "\n")
    
    trainer.fit(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
    
    best_model_path = trainer.checkpoint_callback.best_model_path
    print(f"\nTraining Selesai. Model Tersimpan di: {best_model_path}")

if __name__ == "__main__":
    train_baseline()