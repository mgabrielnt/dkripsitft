# file: src/models/train_tft_with_sentiment.py
"""
Trainer TFT HYBRID (fitur teknikal + fitur sentimen).

VERSI INI MENGGUNAKAN SUBSET FITUR TEKNIKAL & SENTIMEN YANG SUDAH DISELEKSI
berdasarkan:
- analisis korelasi terhadap target,
- Mutual Information (MI),
- pertimbangan konseptual (price/volatility/volume & sentimen/berita).

Aturan utama:

1) Kolom yang TIDAK boleh jadi covariate real:
   - 'date'         → meta tanggal
   - 'split'        → pembagian train/val/test
   - 'time_idx'     → index waktu (sudah dipakai khusus)
   - 'day_of_week', 'month', 'is_month_end' → fitur kalender (known future real)
   - 'ticker'       → static categorical
   - target         → "close" atau "log_return_1d"

2) Fitur TEKNIKAL yang dipakai (whitelist):
   Dipilih dari MI + konsep:
   - volume/volume_ma_ratio_20  → aktivitas & tekanan volume
   - log_return_1d              → return harian
   - return_mean_5d, return_std_5d → mean & volatilitas jangka pendek
   - vol_20, bb_width_20        → volatilitas + lebar Bollinger
   - rsi_14, ma_5_div_ma_20     → momentum & trend
   - close_lag_2, close_lag_3   → lag harga
   - intraday_range_pct, atr_14 → volatilitas intraday
   - gap_return_1d              → gap open/close

3) Fitur SENTIMEN/BERITA yang dipakai (whitelist):
   Dipilih dari MI + konsep:
   - sentiment_mean, sentiment_mean_3d    → rata-rata sentimen harian & 3 hari
   - news_count, news_count_3d            → intensitas berita
   - sentiment_vol_7d, sentiment_trend_5d → volatilitas & trend sentimen
   - sentiment_shock, extreme_news        → shock & hari berita ekstrem

Jika fitur dalam whitelist tidak ditemukan di tft_master.csv,
script akan memberi WARNING dan otomatis menskip fitur tersebut.

Tambahan:
- sentiment_representation (configs/model_tft.yaml):
    - "raw"  → pakai sentimen kontinu (default).
    - "sign" → bucket: -1 / 0 / 1 berdasarkan sentiment_bucket_threshold.
- Outlier sentimen di-clip berdasarkan quantile data train (default 99.5%).
"""

import os
from typing import Dict, List, Tuple

import numpy as np
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
MODELS_DIR = os.path.join(ROOT_DIR, "models", "tft_with_sentiment")
os.makedirs(MODELS_DIR, exist_ok=True)

# ============================================================
# WHITELIST FITUR (HASIL SELEKSI TEKNIKAL + SENTIMEN)
# ============================================================

# Fitur teknikal utama (berdasarkan MI + corr + konsep)
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

# Fitur sentimen / news utama (berdasarkan MI + konsep)
SENTIMENT_FEATURES_WHITELIST: List[str] = [
    "sentiment_mean",
    "sentiment_mean_3d",
    "sentiment_vol_7d",
    "sentiment_trend_5d",
    "news_count",
    "news_count_3d",
    "sentiment_shock",
    "extreme_news",
]


def load_yaml(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ========= Utility Sentiment =========


def bucketize_sentiment(df: pd.DataFrame, threshold: float = 0.0) -> pd.DataFrame:
    """
    Konversi sentimen kontinu menjadi {-1, 0, 1} untuk eksperimen sign-only.

    Diterapkan ke kolom: ["sentiment_mean", "sentiment_mean_3d"] jika ada.
    """
    df = df.copy()
    for col in ["sentiment_mean", "sentiment_mean_3d"]:
        if col not in df.columns:
            continue
        values = df[col].astype(float)

        def _to_sign(v: float) -> float:
            if abs(v) < threshold:
                return 0.0
            return 1.0 if v > 0 else -1.0

        df[col] = values.apply(_to_sign)

    return df


def drop_constant_sentiment_features(
    df_train: pd.DataFrame, features: List[str], eps: float = 1e-9
) -> Tuple[List[str], List[str]]:
    """
    Buang fitur sentimen yang konstan (std ~ 0) agar tidak mengganggu training.
    (Kalau benar-benar konstan, informasinya memang 0.)
    """
    kept, dropped = [], []
    for col in features:
        if col not in df_train.columns:
            dropped.append(col)
            continue

        std = df_train[col].astype(float).std()
        if pd.isna(std) or std <= eps:
            dropped.append(col)
        else:
            kept.append(col)

    return kept, dropped


def clip_sentiment_outliers(
    df_train: pd.DataFrame,
    df_all: pd.DataFrame,
    features: List[str],
    quantile: float = 0.995,
) -> Tuple[pd.DataFrame, Dict[str, Tuple[float, float]]]:
    """
    Clipping outlier sentimen (mis. lonjakan news_count) berbasis quantile data train.
    """
    caps: Dict[str, Tuple[float, float]] = {}
    quantile = max(min(quantile, 0.999), 0.5)  # jaga rentang aman

    df_all = df_all.copy()

    for col in features:
        if col not in df_train.columns:
            continue

        series = df_train[col].astype(float)
        if series.empty:
            continue

        upper = series.quantile(quantile)
        lower = series.quantile(1 - quantile) if series.min() < 0 else 0.0

        if pd.isna(upper):
            continue

        caps[col] = (lower, upper)
        df_all[col] = df_all[col].astype(float).clip(lower=lower, upper=upper)

    return df_all, caps


# ========= Feature Selection (TEKNIKAL + SENTIMEN TERSELEKSI) =========


def infer_feature_sets(df: pd.DataFrame, target: str) -> Tuple[List[str], List[str]]:
    """
    Infer semua fitur numerik dari tft_master, lalu dibatasi dengan WHITELIST:

    - technical_reals : subset dari TECHNICAL_FEATURES_WHITELIST
    - sentiment_reals : subset dari SENTIMENT_FEATURES_WHITELIST

    Proses:
      1. Cari semua kandidat fitur numerik (kecuali meta/calendar/target).
      2. Klasifikasikan jadi teknikal vs sentimen via nama kolom.
      3. Ambil hanya fitur yang ada di whitelist & benar-benar muncul di data.
    """
    # kolom meta yang tidak jadi covariate
    meta_cols = {"date", "split"}

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
        # heuristik: semua kolom yang berkaitan sentimen / news / count berita / rel_sentiment
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
            # kolom di whitelist tapi tidak terdeteksi sebagai teknikal / tidak ada di data
            missing_tech.append(feat)

    if missing_tech:
        print(
            "[HYBRID] WARNING: fitur teknikal di whitelist tetapi tidak ditemukan "
            "atau tidak terklasifikasi sebagai teknikal:",
            missing_tech,
        )

    if not technical_reals:
        # fallback untuk menghindari crash kalau misal semua hilang
        print(
            "[HYBRID] WARNING: whitelist teknikal menghasilkan 0 fitur, "
            "fallback ke SEMUA fitur teknikal yang terdeteksi."
        )
        technical_reals = technical_reals_all

    # Terapkan whitelist sentimen
    sentiment_reals: List[str] = []
    missing_sent: List[str] = []
    for feat in SENTIMENT_FEATURES_WHITELIST:
        if feat in sentiment_reals_all:
            sentiment_reals.append(feat)
        else:
            missing_sent.append(feat)

    if missing_sent:
        print(
            "[HYBRID] WARNING: fitur sentimen di whitelist tetapi tidak ditemukan "
            "atau tidak terklasifikasi sebagai sentimen:",
            missing_sent,
        )

    if not sentiment_reals:
        print(
            "[HYBRID] WARNING: whitelist sentimen menghasilkan 0 fitur, "
            "fallback ke SEMUA fitur sentimen yang terdeteksi."
        )
        sentiment_reals = sentiment_reals_all

    print("\n[HYBRID] Infer fitur dari tft_master.csv (SETELAH SELEKSI)")
    print(f"[HYBRID]  - total numeric covariates  : {len(numeric_covariate_candidates)}")
    print(f"[HYBRID]  - technical_reals (FINAL)   : {len(technical_reals)} kolom")
    for c in technical_reals:
        print(f"           * {c}")
    print(f"[HYBRID]  - sentiment_reals (FINAL)   : {len(sentiment_reals)} kolom")
    for c in sentiment_reals:
        print(f"           * {c}")

    return technical_reals, sentiment_reals


def prepare_hybrid_datasets(
    df: pd.DataFrame,
    data_cfg: dict,
    model_cfg: dict,
):
    """
    Siapkan TimeSeriesDataSet untuk train/val/test HYBRID.

    Meng-handle:
    - cleaning target (drop NaN / inf di target),
    - representasi sentimen (raw vs sign),
    - clipping outlier sentimen,
    - pemilihan fitur teknikal & sentimen via whitelist.
    """

    target = model_cfg.get("target", "log_return_1d")

    sentiment_repr = str(model_cfg.get("sentiment_representation", "raw")).lower()
    sentiment_threshold = float(model_cfg.get("sentiment_bucket_threshold", 0.0))

    df = df.copy()

    # tipe data dasar
    df["time_idx"] = df["time_idx"].astype("int64")
    df["ticker"] = df["ticker"].astype("category")
    df["day_of_week"] = df["day_of_week"].astype("int64")
    df["month"] = df["month"].astype("int64")
    df["is_month_end"] = df["is_month_end"].astype("int64")

    has_sector = "sector" in df.columns
    if has_sector:
        df["sector"] = df["sector"].astype("category")

    # kolom wajib
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
        raise ValueError(f"[HYBRID] Kolom wajib hilang di tft_master: {missing_base}")

    # ====== Bersihkan NaN di kolom wajib (termasuk target) ======
    before_len = len(df)
    df = df.dropna(subset=required_base).copy()
    after_len = len(df)
    if after_len != before_len:
        print(
            f"[HYBRID] Drop baris dengan NaN di kolom wajib base: "
            f"{before_len} -> {after_len}"
        )

    # pastikan target numerik & buang nilai non-finite (inf, -inf)
    df[target] = pd.to_numeric(df[target], errors="coerce")
    non_finite_mask = ~np.isfinite(df[target].values)
    n_non_finite = int(non_finite_mask.sum())
    if n_non_finite > 0:
        print(
            f"[HYBRID] WARNING: {n_non_finite} baris dengan {target} NA/inf akan di-drop "
            "(TFT tidak mengizinkan target non-finite)."
        )
        before_len2 = len(df)
        df = df[~non_finite_mask].copy()
        after_len2 = len(df)
        print(
            f"[HYBRID] Drop baris dengan {target} non-finite: "
            f"{before_len2} -> {after_len2}"
        )

    # cek ulang NaN di kolom base
    base_na = df[required_base].isna().sum()
    if base_na.any():
        print("[HYBRID] WARNING: Masih ada NaN di kolom base setelah cleaning:")
        print(base_na)

    # Representasi sentimen sign (opsional)
    if sentiment_repr == "sign":
        df = bucketize_sentiment(df, threshold=sentiment_threshold)
        print(
            f"[HYBRID] Menggunakan representasi sentimen SIGN (-1/0/1) "
            f"dengan threshold {sentiment_threshold}"
        )
    else:
        print("[HYBRID] Menggunakan representasi sentimen RAW (kontinu)")

    print("[HYBRID] Sample columns:", df.columns.tolist())
    print("[HYBRID] Split counts:")
    print(df["split"].value_counts())

    # infer fitur teknikal & sentimen (SETELAH di-filter whitelist)
    technical_reals, sentiment_reals = infer_feature_sets(df, target=target)

    # isi NaN menjadi 0.0 untuk semua fitur real (teknikal + sentimen) yang dipakai
    for col in technical_reals + sentiment_reals:
        if col in df.columns:
            df[col] = df[col].astype(float).fillna(0.0)

    # train/val/test split
    df_train = df[df["split"] == "train"].copy()
    df_val = df[df["split"] == "val"].copy()
    df_test = df[df["split"] == "test"].copy()

    print(f"[HYBRID] Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")

    # optional clipping outlier sentimen (berbasis train)
    df, caps = clip_sentiment_outliers(
        df_train,
        df,
        sentiment_reals,
        quantile=0.995,
    )
    if caps:
        print(f"[HYBRID] Clipping sentimen dengan caps (≈0.5–99.5% quantile):")
        for k, (lo, hi) in caps.items():
            print(f"         - {k}: [{lo:.4f}, {hi:.4f}]")

    # re-split setelah clipping
    df_train = df[df["split"] == "train"].copy()
    df_val = df[df["split"] == "val"].copy()
    df_test = df[df["split"] == "test"].copy()

    # drop fitur sentimen yang konstan
    sentiment_reals, dropped_const = drop_constant_sentiment_features(
        df_train, sentiment_reals
    )
    if dropped_const:
        print(f"[HYBRID] Drop fitur sentimen konstan (std≈0): {dropped_const}")

    print("\n[HYBRID] Fitur sentimen FINAL yang dipakai TFT HYBRID:")
    for col in sentiment_reals:
        print(f"  - {col}")

    # definisi fitur untuk TFT
    static_categoricals = ["ticker"]
    if has_sector:
        static_categoricals.append("sector")

    static_reals: List[str] = []  # GroupNormalizer target_scale sudah include target stats

    time_varying_known_reals = [
        "time_idx",
        "day_of_week",
        "month",
        "is_month_end",
    ]
    time_varying_known_categoricals: List[str] = []

    time_varying_unknown_categoricals: List[str] = []
    time_varying_unknown_reals = technical_reals + sentiment_reals

    used_cols = set(
        [
            "time_idx",
            "ticker",
            "day_of_week",
            "month",
            "is_month_end",
            target,
            "split",
        ]
        + technical_reals
        + sentiment_reals
    )
    base_exclude = {"date"}
    unused_cols = sorted(c for c in df.columns if c not in used_cols and c not in base_exclude)
    print("\n[HYBRID] Kolom lain di tft_master yang TIDAK dipakai oleh TFT HYBRID:")
    if unused_cols:
        for c in unused_cols:
            print(f"  - {c}")
    else:
        print("  (semua kolom numerik utama sudah dipakai hybrid)")

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
        f"[HYBRID] Len training dataset: {len(training)}, "
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
    devices = model_cfg.get("devices", "auto")
    precision = model_cfg.get("precision", 32)
    num_workers = int(model_cfg.get("num_workers", 0))

    loss_name = str(model_cfg.get("loss", "mae")).lower()

    hidden_size = model_cfg.get("hidden_size_hybrid", model_cfg.get("hidden_size", 64))
    dropout = model_cfg.get("dropout_hybrid", model_cfg.get("dropout", 0.15))

    # ====== Load data ======
    if not os.path.exists(TFT_MASTER_PATH):
        raise FileNotFoundError(f"Tidak ditemukan: {TFT_MASTER_PATH}")

    print(f"[HYBRID] Loading {TFT_MASTER_PATH}")
    df = pd.read_csv(TFT_MASTER_PATH, parse_dates=["date"])

    # ====== Siapkan dataset ======
    training, validation, _, _, _, _ = prepare_hybrid_datasets(df, data_cfg, model_cfg)

    # ====== Dataloader ======
    pin_mem = True if str(accelerator).lower() in ("gpu", "cuda", "auto") else False

    train_dataloader = training.to_dataloader(
        train=True,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_mem,
    )
    val_dataloader = validation.to_dataloader(
        train=False,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_mem,
    )

    # ====== Seed ======
    pl.seed_everything(42)

    # ====== Pilih loss ======
    if loss_name == "mae":
        loss = MAE()
        output_size = 1
        print("[HYBRID] Menggunakan loss MAE (point forecast, output_size=1)")
    else:
        loss = QuantileLoss()
        output_size = 7
        print("[HYBRID] Menggunakan QuantileLoss (probabilistic, output_size=7)")

    # ====== Model TFT ======
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        lstm_layers=model_cfg.get("lstm_layers", 2),
        dropout=dropout,
        attention_head_size=model_cfg.get("attention_head_size", 4),
        hidden_continuous_size=model_cfg.get("hidden_continuous_size", 32),
        loss=loss,
        output_size=output_size,
        log_interval=10,
        reduce_on_plateau_patience=4,
    )

    try:
        print(f"[HYBRID] Model parameter count: {tft.size()}")
    except Exception:
        print("[HYBRID] Tidak bisa menghitung jumlah parameter dengan tft.size()")

    # ====== Callbacks ======
    lr_logger = LearningRateMonitor(logging_interval="epoch")
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=model_cfg.get("early_stopping_patience", 10),
        mode="min",
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=MODELS_DIR,
        filename="tft-with-sentiment-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        save_top_k=1,
        mode="min",
    )

    # ====== Trainer ======
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        gradient_clip_val=0.1,
        callbacks=[lr_logger, early_stop_callback, checkpoint_callback],
        log_every_n_steps=10,
    )

    # ====== Train ======
    print("[HYBRID] Start training TFT WITH SENTIMENT (SELECTED TECHNICAL + SENTIMENT FEATURES)...")
    trainer.fit(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    print("[HYBRID] Training selesai.")
    print(f"[HYBRID] Model terbaik tersimpan di: {checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()
