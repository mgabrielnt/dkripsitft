# file: src/models/train_tft_baseline.py
"""
Trainer TFT BASELINE (tanpa fitur sentimen).

VERSI INI:
- Menggunakan SUBSET fitur teknikal numerik yang SUDAH DISELEKSI
  dari tft_master.csv (bukan semua), berdasarkan:
    * korelasi / Mutual Information terhadap target,
    * pertimbangan konsep (price/volatility/volume/momentum).
- TIDAK menggunakan fitur sentimen / news sama sekali → pure price & technical.

Target diambil dari configs/model_tft.yaml:
    - bisa "close" atau "log_return_1d".

Konversi dari prediksi log_return_1d → harga (kalau diperlukan)
dilakukan di tahap evaluasi/analisis, bukan di trainer ini.
"""

import os
from typing import List, Tuple

import pandas as pd
import yaml
import torch

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

# ============================================================
# WHITELIST FITUR TEKNIKAL (KONSISTEN DENGAN HYBRID)
# ============================================================

TECHNICAL_FEATURES_WHITELIST: List[str] = [
    # volume & tekanan volume
    "volume",
    "volume_ma_ratio_20",
    # return & statistik harga
    "log_return_1d",
    "return_mean_5d",
    "return_std_5d",
    # volatilitas & band
    "vol_20",
    "bb_width_20",
    # momentum & trend
    "rsi_14",
    "ma_5_div_ma_20",
    # lag harga
    "close_lag_2",
    "close_lag_3",
    # volatilitas intraday & gap
    "intraday_range_pct",
    "atr_14",
    "gap_return_1d",
]


def load_yaml(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def infer_feature_sets(df: pd.DataFrame, target: str) -> Tuple[List[str], List[str]]:
    """
    Infer semua fitur numerik dari tft_master, lalu pisahkan:

    - technical_reals : SUBSET fitur teknikal numerik (dipakai baseline)
    - sentiment_reals : semua fitur numerik terkait sentimen/news (DILEWAT baseline)

    Proses:
      1. Cari semua kandidat fitur numerik (kecuali meta/calendar/target).
      2. Klasifikasikan jadi teknikal vs sentimen via nama kolom.
      3. Ambil hanya fitur teknikal yang ada di TECHNICAL_FEATURES_WHITELIST.
    """
    # kolom meta yang tidak jadi covariate
    meta_cols = {
        "date",   # tanggal asli
        "split",  # train/val/test
    }

    # kolom kalender yang jadi known_future_real
    calendar_cols = {"time_idx", "day_of_week", "month", "is_month_end"}

    # tidak boleh dipakai sebagai covariate real biasa
    forbidden = meta_cols | calendar_cols | {"ticker", target}

    numeric_covariate_candidates: List[str] = []
    for col in df.columns:
        if col in forbidden:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_covariate_candidates.append(col)

    sentiment_reals_all: List[str] = []
    technical_reals_all: List[str] = []

    for col in numeric_covariate_candidates:
        name = col.lower()
        # heuristik: semua kolom yang berkaitan sentimen / news / count berita
        if (
            "sentiment" in name
            or "news" in name
            or name.startswith("pos_count")
            or name.startswith("neg_count")
            or name.startswith("neu_count")
            or "strong_market" in name
            or "strong_lex" in name
            or "has_news" in name
            or "extreme_news" in name
            or "bullish" in name
            or "bearish" in name
            or "rel_sentiment" in name
            or "market_sentiment" in name
        ):
            sentiment_reals_all.append(col)
        else:
            technical_reals_all.append(col)

    # Terapkan whitelist teknikal
    technical_reals: List[str] = []
    missing_tech: List[str] = []
    for feat in TECHNICAL_FEATURES_WHITELIST:
        if feat in technical_reals_all:
            technical_reals.append(feat)
        else:
            missing_tech.append(feat)

    if missing_tech:
        print(
            "[BASELINE] WARNING: fitur teknikal di whitelist tetapi tidak ditemukan "
            "atau tidak terklasifikasi sebagai teknikal:",
            missing_tech,
        )

    if not technical_reals:
        print(
            "[BASELINE] WARNING: whitelist teknikal menghasilkan 0 fitur, "
            "fallback ke SEMUA fitur teknikal yang terdeteksi."
        )
        technical_reals = technical_reals_all

    sentiment_reals = sentiment_reals_all  # hanya untuk logging

    print("\n[BASELINE] Infer fitur dari tft_master.csv (SETELAH SELEKSI)")
    print(f"[BASELINE]  - total numeric covariates   : {len(numeric_covariate_candidates)}")
    print(f"[BASELINE]  - technical_reals (dipakai)  : {len(technical_reals)} kolom")
    for c in technical_reals:
        print(f"           * {c}")
    print(f"[BASELINE]  - sentiment_reals (IGNORED)  : {len(sentiment_reals)} kolom")
    for c in sentiment_reals:
        print(f"           * {c}")

    return technical_reals, sentiment_reals


def prepare_baseline_datasets(
    df: pd.DataFrame,
    data_cfg: dict,
    model_cfg: dict,
):
    """
    Siapkan TimeSeriesDataSet untuk train/val/test baseline.

    Dikembalikan:
        training_ds, val_ds, test_ds, train_df, val_df, test_df
    """
    target = model_cfg.get("target", "log_return_1d")

    # pastikan tipe data benar
    df = df.copy()
    df["time_idx"] = df["time_idx"].astype("int64")
    df["ticker"] = df["ticker"].astype("category")
    df["day_of_week"] = df["day_of_week"].astype("int64")
    df["month"] = df["month"].astype("int64")
    df["is_month_end"] = df["is_month_end"].astype("int64")

    # kalau nanti kamu punya kolom 'sector', ini bisa dipakai sebagai static categorical optional
    has_sector = "sector" in df.columns
    if has_sector:
        df["sector"] = df["sector"].astype("category")

    # validasi kolom wajib
    required_base = [
        "time_idx",
        "ticker",
        "day_of_week",
        "month",
        "is_month_end",
        "split",
        target,
    ]
    missing_base = [c for c in required_base if c not in df.columns]
    if missing_base:
        raise ValueError(f"[BASELINE] Kolom wajib hilang di tft_master.csv: {missing_base}")

    # infer fitur (teknikal vs sentimen) + terapkan whitelist teknikal
    technical_reals, sentiment_reals = infer_feature_sets(df, target=target)

    # log NaN
    print("\n[BASELINE] NaN per kolom base + teknikal (sebelum cleaning):")
    cols_for_nan_log = list(dict.fromkeys(required_base + technical_reals))
    print(df[cols_for_nan_log].isna().sum())

    before_len = len(df)
    df = df.dropna(subset=required_base).copy()
    after_len = len(df)
    print(f"[BASELINE] Drop baris dengan NaN di kolom wajib base: {before_len} -> {after_len}")

    # isi NaN fitur teknikal dengan 0.0
    for col in technical_reals:
        if col in df.columns:
            df[col] = df[col].astype(float).fillna(0.0)

    # split
    df_train = df[df["split"] == "train"].copy()
    df_val = df[df["split"] == "val"].copy()
    df_test = df[df["split"] == "test"].copy()

    print(f"[BASELINE] Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")

    # log kolom yang tidak dipakai baseline
    used_cols = set(required_base + technical_reals)
    base_exclude = {"date"}
    unused_cols = sorted(c for c in df.columns if c not in used_cols and c not in base_exclude)
    print("\n[BASELINE] Kolom lain di tft_master yang TIDAK dipakai oleh TFT baseline:")
    if unused_cols:
        for c in unused_cols:
            print(f"  - {c}")
    else:
        print("  (semua kolom numerik utama sudah dipakai baseline)")

    # definisi fitur untuk TFT
    static_categoricals = ["ticker"]
    if has_sector:
        static_categoricals.append("sector")

    static_reals: List[str] = []

    time_varying_known_reals = [
        "time_idx",
        "day_of_week",
        "month",
        "is_month_end",
    ]
    time_varying_known_categoricals: List[str] = []

    time_varying_unknown_categoricals: List[str] = []
    time_varying_unknown_reals = technical_reals

    max_encoder_length = model_cfg.get("max_encoder_length", 60)
    max_prediction_length = model_cfg.get(
        "max_prediction_length",
        data_cfg.get("horizon", 5),
    )

    # dataset training
    training = TimeSeriesDataSet(
        df_train,
        time_idx="time_idx",
        target=target,
        group_ids=["ticker"],
        allow_missing_timesteps=True,
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

    # dataset val & test dari template training
    validation = TimeSeriesDataSet.from_dataset(
        training,
        df_val,
        stop_randomization=True,
    )
    test = TimeSeriesDataSet.from_dataset(
        training,
        df_test,
        stop_randomization=True,
    )

    print(
        f"[BASELINE] Len training dataset: {len(training)}, "
        f"len validation: {len(validation)}, len test: {len(test)}"
    )

    return training, validation, test, df_train, df_val, df_test


def main():
    # ====== Load config ======
    data_cfg = load_yaml(CONFIG_DATA_PATH)
    model_cfg = load_yaml(CONFIG_MODEL_PATH)

    target = model_cfg.get("target", "log_return_1d")
    max_epochs = model_cfg.get("max_epochs", 100)
    batch_size = model_cfg.get("batch_size", 64)

    learning_rate_raw = model_cfg.get("learning_rate", 5e-4)
    learning_rate = float(learning_rate_raw)

    accelerator = model_cfg.get("accelerator", "auto")
    loss_name = str(model_cfg.get("loss", "mae")).lower()

    # precision dari config (default 32-true untuk hindari error half overflow)
    precision = model_cfg.get("precision", "32-true")

    # ====== Load data ======
    if not os.path.exists(TFT_MASTER_PATH):
        raise FileNotFoundError(f"Tidak ditemukan: {TFT_MASTER_PATH}")

    print(f"[BASELINE] Loading {TFT_MASTER_PATH}")
    df = pd.read_csv(TFT_MASTER_PATH, parse_dates=["date"])

    print("[BASELINE] Sample columns:", df.columns.tolist())
    print("[BASELINE] Split counts:")
    print(df["split"].value_counts())

    # ====== Siapkan dataset ======
    training, validation, _, _, _, _ = prepare_baseline_datasets(df, data_cfg, model_cfg)

    # ====== DataLoader ======
    # Untuk Windows: num_workers kecil dulu (0 atau 2).
    train_dataloader = training.to_dataloader(
        train=True,
        batch_size=batch_size,
        num_workers=2,
    )
    val_dataloader = validation.to_dataloader(
        train=False,
        batch_size=batch_size,
        num_workers=2,
    )

    # ====== Seed ======
    pl.seed_everything(42)

    # ====== (Opsional) set float32 matmul precision agar Tensor Cores kepakai ======
    try:
        torch.set_float32_matmul_precision("medium")
        print("[BASELINE] torch.set_float32_matmul_precision('medium') aktif")
    except Exception as e:
        print(f"[BASELINE] Tidak bisa set float32 matmul precision: {e}")

    # ====== Pilih loss ======
    if loss_name == "mae":
        loss = MAE()
        output_size = 1
        print("[BASELINE] Menggunakan loss MAE (point forecast, output_size=1)")
    else:
        loss = QuantileLoss()
        output_size = 7
        print("[BASELINE] Menggunakan QuantileLoss (probabilistic, output_size=7)")

    # ====== Buat model TFT ======
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=learning_rate,
        hidden_size=model_cfg.get("hidden_size", 64),
        lstm_layers=model_cfg.get("lstm_layers", 2),
        dropout=model_cfg.get("dropout", 0.1),
        attention_head_size=model_cfg.get("attention_head_size", 4),
        hidden_continuous_size=model_cfg.get("hidden_continuous_size", 32),
        loss=loss,
        output_size=output_size,
        log_interval=10,
        reduce_on_plateau_patience=4,
    )

    try:
        print(f"[BASELINE] Model parameter count: {tft.size()}")
    except Exception:
        print("[BASELINE] Tidak bisa menghitung jumlah parameter dengan tft.size()")

    # ====== Callbacks ======
    lr_logger = LearningRateMonitor(logging_interval="epoch")
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=model_cfg.get("early_stopping_patience", 10),
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
        devices=1,          # 1 GPU (RTX 3050) atau 1 CPU device
        precision=precision,  # "32-true" by default → no AMP 16-bit
        gradient_clip_val=0.1,
        callbacks=[lr_logger, early_stop_callback, checkpoint_callback],
        log_every_n_steps=10,
    )

    # ====== Train ======
    print("[BASELINE] Start training TFT baseline (SELECTED TECHNICAL FEATURES, NO SENTIMENT)...")
    trainer.fit(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    print("[BASELINE] Training selesai.")
    print(f"[BASELINE] Model terbaik tersimpan di: {checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()
