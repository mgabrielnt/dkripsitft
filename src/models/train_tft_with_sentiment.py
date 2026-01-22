import os
import yaml
import pandas as pd
import torch
from pathlib import Path

# --- PERBAIKAN IMPORT (VITAL) ---
# Menggunakan namespace 'lightning.pytorch' agar kompatibel dengan pytorch-forecasting terbaru
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss

# --- SETUP PATH ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "model_tft.yaml"

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    data_path = PROJECT_ROOT / config['data']['dataset_path']
    
    # 1. Load Data
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    df['ticker'] = df['ticker'].astype(str)
    df['day_of_week'] = df['day_of_week'].astype(str)
    df['month'] = df['month'].astype(str)
    
    # Filter Data Training
    training_cutoff = df["time_idx"].max() - config['data']['max_prediction_length']
    
    # 2. Buat Dataset Training
    # Pastikan 'sentiment_dir_signal' ada di list fitur
    training = TimeSeriesDataSet(
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
        time_varying_unknown_reals=[
            "close", "volume", "log_return_1d", "vol_20", "rsi_14", 
            "ma_5_div_ma_20", "bb_width_20", "gap_return_1d", "intraday_range_pct",
            "has_news", "news_count_3d", 
            "sentiment_mean_3d", "sentiment_ema_7d", "sentiment_ema_14d",
            "sentiment_trend_7d", "sentiment_intraday_std", 
            "sentiment_vol_impact", "high_news_day",
            "sentiment_dir_signal" # <--- FITUR BARU WAJIB ADA
        ],
        target_normalizer=GroupNormalizer(groups=["ticker"], transformation="softplus"),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    # 3. Validation Set
    validation = TimeSeriesDataSet.from_dataset(training, df, predict=True, stop_randomization=True)
    
    # Dataloaders
    batch_size = config['training']['batch_size']
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

    # 4. Model Setup (Hyperparameters dari Config baru)
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=config['model']['learning_rate'],
        hidden_size=config['model']['hidden_size'],
        attention_head_size=config['model']['attention_head_size'],
        dropout=config['model']['dropout'],
        hidden_continuous_size=config['model']['hidden_continuous_size'],
        output_size=7,
        loss=QuantileLoss(),
        log_interval=10,
        reduce_on_plateau_patience=4,
    )

    # 5. Trainer Setup
    checkpoint_callback = ModelCheckpoint(
        dirpath=PROJECT_ROOT / "checkpoints" / "sentiment",
        filename="tft-sentiment-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        monitor="val_loss",
        mode="min"
    )
    
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=1e-4,
        patience=config['training']['patience'],
        verbose=False,
        mode="min"
    )

    trainer = pl.Trainer(
        max_epochs=config['training']['max_epochs'],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        enable_model_summary=True,
        gradient_clip_val=config['training']['gradient_clip_val'],
        callbacks=[checkpoint_callback, early_stop_callback],
    )

    # 6. Start Training
    print("\n" + "="*50)
    print("   MEMULAI TRAINING SENTIMENT MODEL (TUNED)")
    print("="*50)
    trainer.fit(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
    
    print(f"\nTraining Selesai. Model Tersimpan di: {checkpoint_callback.best_model_path}")

if __name__ == "__main__":
    main()