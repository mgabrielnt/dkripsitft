import json
import os
import warnings
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
import yaml
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss

warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision("medium")
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CONFIG_PATH = os.path.join(ROOT_DIR, "configs", "model_tft.yaml")
CONFIG_DATA_PATH = os.path.join(ROOT_DIR, "configs", "data.yaml")
OUT_DIR = Path(r"D:/skripsi/tft/models/baseline")
CAT_COLS = ["ticker", "day_of_week", "month", "is_month_end"]
REAL_COLS = ["close", "volume", "log_return_1d", "log_return_2d", "vol_20", "rsi_14", "ma_5_div_ma_20", "bb_width_20", "gap_return_1d", "intraday_range_pct"]
FORBIDDEN_TICKERS = {"BBCA.JK", "UNVR.JK"}


def load_cfg():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_allowed_tickers():
    with open(CONFIG_DATA_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return {str(t).strip().upper() for t in cfg.get("tickers", []) if str(t).strip()}


def ensure_cols(df: pd.DataFrame, cols, ctx=""):
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"{ctx} kolom wajib hilang: {miss}")


def prepare_df(path: str, time_idx_col: str, target_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    ensure_cols(df, ["split", *CAT_COLS, time_idx_col, target_col], "[BASELINE]")
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["ticker"] = df["ticker"].astype(str).str.upper()
    allowed_tickers = load_allowed_tickers()
    if allowed_tickers:
        df = df[df["ticker"].isin(allowed_tickers)].copy()
    forbidden = FORBIDDEN_TICKERS & set(df["ticker"].unique())
    if forbidden:
        raise ValueError(f"[BASELINE] ticker terlarang masih ada: {sorted(forbidden)}")
    df["split"] = df["split"].astype(str).str.strip().str.lower()
    for c in CAT_COLS:
        df[c] = df[c].astype(str)
    df[time_idx_col] = pd.to_numeric(df[time_idx_col], errors="coerce")
    if df[time_idx_col].isna().any():
        order = ["ticker", "date"] if "date" in df.columns else ["ticker"]
        df = df.sort_values(order).copy()
        df[time_idx_col] = df.groupby("ticker").cumcount().astype("int64")
    close = pd.to_numeric(df["close"], errors="coerce").replace(0, np.nan)
    if "log_return_1d" not in df.columns:
        df["log_return_1d"] = np.log(close / close.groupby(df["ticker"]).shift(1))
    df["log_return_2d"] = np.log(close / close.groupby(df["ticker"]).shift(2))
    ensure_cols(df, REAL_COLS, "[BASELINE features]")
    for c in [*REAL_COLS, target_col]:
        df[c] = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        fill = 0.0 if pd.isna(df[c].median()) else float(df[c].median())
        df[c] = df[c].fillna(fill).astype("float32")
    return df.sort_values(["ticker", time_idx_col]).reset_index(drop=True)


def build_datasets(df: pd.DataFrame, cfg: dict):
    data_cfg = cfg["data"]
    train_df, val_df = df[df["split"] == "train"].copy(), df[df["split"] == "val"].copy()
    if train_df.empty or val_df.empty:
        raise ValueError("Data train/val kosong. Periksa kolom split.")

    encoder_length = int(data_cfg["max_encoder_length"])
    prediction_length = int(data_cfg["max_prediction_length"])

    base = dict(
        time_idx=data_cfg["time_idx"],
        target=data_cfg["target"],
        group_ids=data_cfg["group_ids"],
        min_encoder_length=encoder_length,
        max_encoder_length=encoder_length,
        min_prediction_length=prediction_length,
        max_prediction_length=prediction_length,
        static_categoricals=["ticker"],
        time_varying_known_categoricals=["day_of_week", "month", "is_month_end"],
        time_varying_unknown_reals=REAL_COLS,
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
    )
    training = TimeSeriesDataSet(train_df, **base)
    validation = TimeSeriesDataSet.from_dataset(
        training, val_df, predict=False, stop_randomization=True
    )
    return training, validation


def fit_one(name: str, params: dict, training, validation, cfg: dict) -> dict:
    model_cfg, trainer_cfg = cfg.get("model", {}), cfg.get("trainer", {})
    out_dir = OUT_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = ModelCheckpoint(dirpath=str(out_dir), filename="best-checkpoint", monitor="val_loss", mode="min", save_top_k=1, auto_insert_metric_name=False)
    early = EarlyStopping(monitor="val_loss", mode="min", patience=int(trainer_cfg.get("early_stopping_patience", 10)), min_delta=float(trainer_cfg.get("early_stopping_min_delta", 1e-4)), verbose=False)
    trainer = pl.Trainer(max_epochs=int(trainer_cfg.get("max_epochs", 50)), accelerator="auto", devices=1 if torch.cuda.is_available() else "auto", gradient_clip_val=float(trainer_cfg.get("gradient_clip_val", 0.1)), callbacks=[early, ckpt], logger=False, enable_progress_bar=True, enable_model_summary=False)
    train_loader = training.to_dataloader(train=True, batch_size=int(params["batch_size"]), num_workers=0)
    val_loader = validation.to_dataloader(train=False, batch_size=max(1, int(params["batch_size"]) * 2), num_workers=0)
    model = TemporalFusionTransformer.from_dataset(training, learning_rate=float(params["learning_rate"]), hidden_size=int(params["hidden_size"]), attention_head_size=int(model_cfg.get("attention_head_size", 4)), dropout=float(model_cfg.get("dropout", 0.3)), hidden_continuous_size=max(8, int(params["hidden_size"]) // 2), loss=QuantileLoss(), reduce_on_plateau_patience=4, optimizer="Adam", log_interval=10)
    status, best_val_loss, best_path = "ok", np.nan, ""
    try:
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        best_path = ckpt.best_model_path
        if ckpt.best_model_score is not None:
            best_val_loss = float(getattr(ckpt.best_model_score, "item", lambda: ckpt.best_model_score)())
    except Exception as e:
        status = f"error: {type(e).__name__}: {e}"
    row = {"scenario": name, "learning_rate": float(params["learning_rate"]), "hidden_size": int(params["hidden_size"]), "batch_size": int(params["batch_size"]), "best_val_loss": best_val_loss, "best_model_path": best_path, "status": status}
    (out_dir / "params.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    return row


def main():
    pl.seed_everything(42, workers=True)
    cfg = load_cfg()
    data_cfg = cfg["data"]
    df = prepare_df(data_cfg["csv_path"], data_cfg["time_idx"], data_cfg["target"])
    training, validation = build_datasets(df, cfg)
    rows = [fit_one(name, params, training, validation, cfg) for name, params in cfg["scenarios"].items()]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows).sort_values(["best_val_loss", "scenario"], na_position="last")
    result.to_csv(OUT_DIR / "baseline_val_loss_results.csv", index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
