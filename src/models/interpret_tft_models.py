import os

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt

import lightning.pytorch as pl

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer

# ==== Path dasar ====
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")
CONFIG_DATA_PATH = os.path.join(ROOT_DIR, "configs", "data.yaml")
CONFIG_MODEL_PATH = os.path.join(ROOT_DIR, "configs", "model_tft.yaml")
CONFIG_EXPERIMENTS_PATH = os.path.join(ROOT_DIR, "configs", "experiments.yaml")

TFT_MASTER_PATH = os.path.join(DATA_PROCESSED_DIR, "tft_master.csv")

OUT_DIR = os.path.join(ROOT_DIR, "reports", "tft_interpretability")
os.makedirs(OUT_DIR, exist_ok=True)


def load_yaml(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def make_val_dataset(df_all: pd.DataFrame, data_cfg, model_cfg, use_sentiment: bool):
    target = model_cfg.get("target", "close")
    max_encoder_length = model_cfg.get("max_encoder_length", 60)
    max_prediction_length = model_cfg.get(
        "max_prediction_length",
        data_cfg.get("horizon", 5),
    )

    df_all = df_all.copy()
    df_all["time_idx"] = df_all["time_idx"].astype("int64")
    df_all["ticker"] = df_all["ticker"].astype("category")

    df_train = df_all[df_all["split"] == "train"].copy()
    df_val = df_all[df_all["split"] == "val"].copy()

    static_categoricals = ["ticker"]
    static_reals: list[str] = []

    time_varying_known_reals = [
        "time_idx",
        "day_of_week",
        "month",
        "is_month_end",
    ]
    time_varying_known_categoricals: list[str] = []

    time_varying_unknown_reals = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "ma_5",
        "ma_10",
        "ma_20",
        "rsi_14",
        "log_return_1d",
        "vol_20",
        "ma_5_div_ma_20",
    ]
    if use_sentiment:
        time_varying_unknown_reals += [
            "sentiment_mean",
            "news_count",
            "sentiment_mean_3d",
            "news_count_3d",
        ]

    time_varying_unknown_categoricals: list[str] = []

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

    return training, validation


def interpret_one_model(
    label: str,
    ckpt_path: str,
    use_sentiment: bool,
    df_all: pd.DataFrame,
    data_cfg,
    model_cfg,
):
    if not os.path.exists(ckpt_path):
        print(f"[WARN] Checkpoint {label} tidak ditemukan: {ckpt_path}")
        return

    print(f"\n[INFO] Interpretasi model {label} dari checkpoint: {ckpt_path}")

    training, validation = make_val_dataset(
        df_all,
        data_cfg,
        model_cfg,
        use_sentiment=use_sentiment,
    )

    val_loader = validation.to_dataloader(
        train=False,
        batch_size=model_cfg.get("batch_size", 64),
        num_workers=0,
    )

    pl.seed_everything(42)

    model = TemporalFusionTransformer.load_from_checkpoint(ckpt_path)

    # === Prediksi RAW + input X (kompatibel versi baru & lama pytorch-forecasting) ===
    pred_obj = model.predict(
        val_loader,
        mode="raw",
        return_x=True,
        trainer_kwargs=dict(accelerator="cpu"),
    )

    # Versi baru: PredictionsOutput dengan .output (dict) dan .x (dict)
    # Versi lama: bisa tuple (raw_predictions, x)
    if hasattr(pred_obj, "output") and hasattr(pred_obj, "x"):
        raw_predictions = pred_obj.output
        x = pred_obj.x
    elif isinstance(pred_obj, (list, tuple)) and len(pred_obj) >= 2:
        raw_predictions, x = pred_obj[0], pred_obj[1]
    else:
        raise RuntimeError(
            f"Tidak bisa menginterpretasi hasil model.predict, tipe: {type(pred_obj)}"
        )

    # === Global variable importance & attention ===
    interpretation = model.interpret_output(raw_predictions, reduction="sum")

    # === Plot interpretasi global (handle berbagai bentuk return) ===
    figs = model.plot_interpretation(interpretation)

    # Normalisasi ke list of (name, fig)
    fig_items: list[tuple[str, object]] = []

    if isinstance(figs, dict):
        # misal: {"attention": fig1, "static_variables": fig2, ...}
        for k, v in figs.items():
            fig_items.append((str(k), v))
    elif isinstance(figs, (list, tuple)):
        # list/tupel figure tanpa nama → kasih nama generik
        for i, f in enumerate(figs):
            fig_items.append((f"fig_{i}", f))
    else:
        # single figure
        fig_items.append(("global", figs))

    for key, fig in fig_items:
        if hasattr(fig, "savefig"):
            safe_key = key.replace(" ", "_")
            out_fig_path = os.path.join(
                OUT_DIR,
                f"interpretation_{label}_{safe_key}.png",
            )
            fig.savefig(out_fig_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"[INFO] Gambar interpretasi ({label}, {key}) disimpan di: {out_fig_path}")
        else:
            print(
                f"[WARN] Objek interpretasi untuk key '{key}' "
                f"bukan matplotlib Figure, tipe: {type(fig)}"
            )

    # === Opsional: simpan importance numerik ke CSV (kalau tersedia) ===
    for key in ["encoder_variables", "decoder_variables", "static_variables"]:
        if key in interpretation:
            imp = interpretation[key]

            # biasanya imp adalah tensor [n_variables]
            if hasattr(imp, "detach"):
                vals = imp.detach().cpu().numpy().ravel()
            else:
                vals = np.array(imp).ravel()

            # ambil nama variabel dari dataset (reals + categoricals)
            var_names = training.reals + getattr(training, "flat_categoricals", [])
            var_names = var_names[: len(vals)]

            df_imp = pd.DataFrame({"variable": var_names, "importance": vals})
            df_imp = df_imp.sort_values("importance", ascending=False)
            out_imp_path = os.path.join(OUT_DIR, f"importance_{label}_{key}.csv")
            df_imp.to_csv(out_imp_path, index=False)
            print(f"[INFO] Importance {key} disimpan di: {out_imp_path}")


def main():
    if not os.path.exists(TFT_MASTER_PATH):
        raise FileNotFoundError(f"Tidak ditemukan: {TFT_MASTER_PATH}")

    data_cfg = load_yaml(CONFIG_DATA_PATH)
    model_cfg = load_yaml(CONFIG_MODEL_PATH)
    exp_cfg = load_yaml(CONFIG_EXPERIMENTS_PATH)

    df_all = pd.read_csv(TFT_MASTER_PATH, parse_dates=["date"])

    # Baseline & hybrid checkpoint dari experiments.yaml
    baseline_ckpts = exp_cfg["tft_baseline"]["checkpoint_paths"]
    hybrid_ckpts = exp_cfg["tft_with_sentiment"]["checkpoint_paths"]

    baseline_ckpt = baseline_ckpts[0]
    hybrid_ckpt = hybrid_ckpts[0]

    # Interpretasi BASELINE (tanpa sentimen)
    interpret_one_model(
        label="baseline",
        ckpt_path=baseline_ckpt,
        use_sentiment=False,
        df_all=df_all,
        data_cfg=data_cfg,
        model_cfg=model_cfg,
    )

    # Interpretasi HYBRID (dengan sentimen)
    interpret_one_model(
        label="hybrid",
        ckpt_path=hybrid_ckpt,
        use_sentiment=True,
        df_all=df_all,
        data_cfg=data_cfg,
        model_cfg=model_cfg,
    )


if __name__ == "__main__":
    main()
