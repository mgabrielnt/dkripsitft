import os
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    confusion_matrix,
    accuracy_score,
    classification_report,
)

# ==============================
# KONFIGURASI
# ==============================

# Kalau True -> hitung juga metrik setelah bias correction (y_pred + mean residual)
APPLY_BIAS_CORRECTION = True

# Kalau False -> bagian klasifikasi bucket (F1, confusion) tidak dihitung
ENABLE_BUCKET_CLASSIFICATION = False

# ==============================
# Path setup
# ==============================

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(ROOT_DIR, "data", "processed")
REPORT_DIR = os.path.join(ROOT_DIR, "reports")
FIG_DIR = os.path.join(REPORT_DIR, "figures")

os.makedirs(FIG_DIR, exist_ok=True)


def load_predictions() -> Dict[str, pd.DataFrame]:
    """
    Load baseline & hybrid prediction CSVs.
    Expected columns: y_true, y_pred.
    """
    paths = {
        "baseline": os.path.join(DATA_DIR, "predictions_tft_baseline_test.csv"),
        "hybrid": os.path.join(DATA_DIR, "predictions_tft_with_sentiment_test.csv"),
    }

    preds: Dict[str, pd.DataFrame] = {}
    for name, path in paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Tidak menemukan file prediksi: {path}")
        df = pd.read_csv(path)
        required = {"y_true", "y_pred"}
        if not required.issubset(df.columns):
            raise ValueError(
                f"File {path} harus punya kolom {required}, "
                f"sekarang kolomnya: {df.columns.tolist()}"
            )
        preds[name] = df
    return preds


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Hitung metrik regresi standar.
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    # MAPE: hati-hati pembagian nol -> skip nol
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    # sMAPE
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    smape = np.mean(
        np.where(denom == 0, 0.0, np.abs(y_true - y_pred) / denom)
    ) * 100
    r2 = r2_score(y_true, y_pred)
    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "MAPE(%)": mape,
        "sMAPE(%)": smape,
        "R2": r2,
    }


def error_by_quantile(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 4,
) -> pd.DataFrame:
    """
    Hitung MAE & RMSE per quantile y_true.
    Ini membantu melihat apakah model lebih jelek di harga rendah / tinggi.
    """
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    q_labels = [f"Q{i+1}" for i in range(n_bins)]
    df["bucket"], bins = pd.qcut(
        df["y_true"], q=n_bins, labels=q_labels, retbins=True, duplicates="drop"
    )

    rows = []
    for q in q_labels:
        sub = df[df["bucket"] == q]
        if sub.empty:
            continue
        y_t = sub["y_true"].values
        y_p = sub["y_pred"].values
        mae = mean_absolute_error(y_t, y_p)
        mse = mean_squared_error(y_t, y_p)
        rmse = np.sqrt(mse)
        rows.append(
            {
                "bucket": q,
                "count": len(sub),
                "y_true_min": float(y_t.min()),
                "y_true_max": float(y_t.max()),
                "MAE": mae,
                "RMSE": rmse,
            }
        )

    return pd.DataFrame(rows)


def add_level_buckets(df: pd.DataFrame, n_bins: int = 3) -> Tuple[pd.DataFrame, List[str]]:
    """
    Buat kategori level harga (low / mid / high) berbasis quantile dari y_true,
    lalu mapping y_pred ke bucket yang sama (berdasarkan batas quantile y_true).
    """
    df = df.copy()
    y_true = df["y_true"].values

    quantiles = np.linspace(0, 1, n_bins + 1)
    bins = np.quantile(y_true, quantiles)
    bins = np.unique(bins)
    if len(bins) <= 2:
        # fallback: 2 bin saja
        bins = np.quantile(y_true, [0.0, 0.5, 1.0])
        bins = np.unique(bins)

    labels = [f"Q{i+1}" for i in range(len(bins) - 1)]

    df["true_bucket"] = pd.cut(
        df["y_true"],
        bins=bins,
        labels=labels,
        include_lowest=True,
    )
    df["pred_bucket"] = pd.cut(
        df["y_pred"],
        bins=bins,
        labels=labels,
        include_lowest=True,
    )

    return df, labels


def classification_metrics_from_buckets(
    df: pd.DataFrame, labels: List[str]
) -> Dict[str, Any]:
    """
    Hitung accuracy, confusion matrix, dan classification report
    dari kolom true_bucket vs pred_bucket.
    """
    df_clean = df.dropna(subset=["true_bucket", "pred_bucket"]).copy()
    if df_clean.empty:
        return {
            "accuracy": float("nan"),
            "confusion_matrix": np.zeros((len(labels), len(labels)), dtype=int),
            "classification_report": {},
        }

    y_true_cls = df_clean["true_bucket"].astype(str).values
    y_pred_cls = df_clean["pred_bucket"].astype(str).values

    acc = accuracy_score(y_true_cls, y_pred_cls)
    cm = confusion_matrix(y_true_cls, y_pred_cls, labels=labels)
    report = classification_report(
        y_true_cls,
        y_pred_cls,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": acc,
        "confusion_matrix": cm,
        "classification_report": report,
    }


def plot_confusion_heatmap(
    cm: np.ndarray,
    labels: List[str],
    title: str,
    save_path: str,
) -> None:
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.xlabel("Predicted bucket")
    plt.ylabel("True bucket")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_feature_correlation_heatmap() -> None:
    """
    Baca tft_master.csv dan plot heatmap korelasi fitur utama dengan target.
    """
    master_path = os.path.join(DATA_DIR, "tft_master.csv")
    if not os.path.exists(master_path):
        print(f"[WARN] tft_master.csv tidak ditemukan di {master_path}, skip heatmap.")
        return

    df = pd.read_csv(master_path)

    cols = [
        "close",
        "volume",
        "log_return_1d",
        "vol_20",
        "rsi_14",
        "ma_5_div_ma_20",
        "bb_width_20",
        "volume_ma_ratio_20",
        "return_mean_5d",
        "return_std_5d",
        "sentiment_mean",
        "sentiment_mean_3d",
        "news_count",
        "news_count_3d",
        "sentiment_shock",
        "extreme_news",
        "sentiment_vol_7d",
        "sentiment_trend_5d",
    ]
    cols = [c for c in cols if c in df.columns]
    if not cols:
        print("[WARN] Tidak ada kolom yang cocok untuk korelasi, skip heatmap.")
        return

    corr = df[cols].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        square=True,
    )
    plt.title("Correlation Heatmap Fitur Teknis & Sentimen")
    plt.tight_layout()
    save_path = os.path.join(FIG_DIR, "feature_correlation_heatmap.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[INFO] Heatmap korelasi fitur disimpan ke: {save_path}")


def plot_residual_hist(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    save_path: str,
) -> None:
    residuals = y_true - y_pred
    plt.figure(figsize=(6, 4))
    plt.hist(residuals, bins=20, edgecolor="black")
    plt.xlabel("Residual (y_true - y_pred)")
    plt.ylabel("Frekuensi")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_true_vs_pred_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    save_path: str,
) -> None:
    plt.figure(figsize=(5, 5))
    plt.scatter(y_true, y_pred, alpha=0.6)
    min_v = min(np.min(y_true), np.min(y_pred))
    max_v = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_v, max_v], [min_v, max_v], linestyle="--")
    plt.xlabel("y_true")
    plt.ylabel("y_pred")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def compute_bias_corrected(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Tuple[np.ndarray, float]:
    """
    Hitung bias (mean residual) dan prediksi yang sudah dikoreksi.
    y_pred_corr = y_pred + bias
    """
    residuals = y_true - y_pred
    bias = float(np.mean(residuals))
    y_pred_corr = y_pred + bias
    return y_pred_corr, bias


def main() -> None:
    preds = load_predictions()

    summary_rows = []

    for name, df in preds.items():
        print("=" * 60)
        print(f"[{name.upper()}] Regression metrics (tanpa koreksi bias)")
        y_true = df["y_true"].values
        y_pred = df["y_pred"].values

        # 1) REGRESSION METRICS (GLOBAL)
        reg = regression_metrics(y_true, y_pred)
        for k, v in reg.items():
            if "APE" in k:
                print(f"  {k:10s} = {v:8.3f}")
            else:
                print(f"  {k:10s} = {v:8.4f}")

        summary_row = {"model": name, "kind": "raw", **reg}
        summary_rows.append(summary_row)

        # 2) MAE/RMSE PER QUANTILE HARGA (tanpa koreksi)
        bucket_df = error_by_quantile(y_true, y_pred, n_bins=4)
        print(f"\n[{name.upper()}] Error per quantile harga (Q1=harga rendah ... Q4=tinggi):")
        if bucket_df.empty:
            print("  (tidak cukup data untuk menghitung per quantile)")
        else:
            for _, row in bucket_df.iterrows():
                print(
                    f"  {row['bucket']}: n={int(row['count'])}, "
                    f"range_y=[{row['y_true_min']:.1f}, {row['y_true_max']:.1f}], "
                    f"MAE={row['MAE']:.2f}, RMSE={row['RMSE']:.2f}"
                )

        # 3) Plot residual & scatter (tanpa koreksi)
        resid_path = os.path.join(FIG_DIR, f"residual_hist_{name}.png")
        plot_residual_hist(
            y_true,
            y_pred,
            title=f"TFT {name.capitalize()} – Histogram Residual (raw)",
            save_path=resid_path,
        )
        print(f"  Histogram residual disimpan ke: {resid_path}")

        scatter_path = os.path.join(FIG_DIR, f"true_vs_pred_{name}.png")
        plot_true_vs_pred_scatter(
            y_true,
            y_pred,
            title=f"TFT {name.capitalize()} – y_true vs y_pred (raw)",
            save_path=scatter_path,
        )
        print(f"  Scatter y_true vs y_pred disimpan ke: {scatter_path}")

        # 4) OPTIONAL: bucket classification (F1, confusion)
        if ENABLE_BUCKET_CLASSIFICATION:
            df_b, labels = add_level_buckets(df, n_bins=3)
            cls = classification_metrics_from_buckets(df_b, labels)

            print(f"\n[{name.upper()}] Bucket-level classification (zona harga)")
            acc = cls["accuracy"]
            if acc != acc:  # NaN check
                print("  Accuracy = NaN (kemungkinan semua bucket kosong/NaN)")
            else:
                print(f"  Accuracy = {acc:.4f}")

            report = cls["classification_report"]
            for label in labels:
                stats = report.get(label, {})
                f1 = stats.get("f1-score", float("nan"))
                support = stats.get("support", 0)
                if support == 0:
                    print(f"  {label}: F1 = NaN (support=0)")
                else:
                    print(f"  {label}: F1 = {f1:.4f}, support = {support}")

            cm = cls["confusion_matrix"]
            cm_path = os.path.join(FIG_DIR, f"confusion_heatmap_{name}.png")
            plot_confusion_heatmap(
                cm,
                labels=labels,
                title=f"TFT {name.capitalize()} – Bucket Confusion Matrix",
                save_path=cm_path,
            )
            print(f"  Confusion matrix heatmap disimpan ke: {cm_path}")

        # 5) BIAS CORRECTION (OPSIONAL)
        if APPLY_BIAS_CORRECTION:
            y_pred_corr, bias = compute_bias_corrected(y_true, y_pred)
            print(
                f"\n[{name.upper()}] Regression metrics SETELAH koreksi bias "
                f"(y_pred + mean_residual, bias={bias:.4f})"
            )
            reg_corr = regression_metrics(y_true, y_pred_corr)
            for k, v in reg_corr.items():
                if "APE" in k:
                    print(f"  {k:10s} = {v:8.3f}")
                else:
                    print(f"  {k:10s} = {v:8.4f}")

            summary_rows.append({"model": name, "kind": "bias_corrected", **reg_corr})

            # Plot residual & scatter sesudah koreksi bias
            resid_corr_path = os.path.join(FIG_DIR, f"residual_hist_{name}_bias_corrected.png")
            plot_residual_hist(
                y_true,
                y_pred_corr,
                title=f"TFT {name.capitalize()} – Histogram Residual (bias-corrected)",
                save_path=resid_corr_path,
            )
            print(f"  Histogram residual (bias-corrected) disimpan ke: {resid_corr_path}")

            scatter_corr_path = os.path.join(FIG_DIR, f"true_vs_pred_{name}_bias_corrected.png")
            plot_true_vs_pred_scatter(
                y_true,
                y_pred_corr,
                title=f"TFT {name.capitalize()} – y_true vs y_pred (bias-corrected)",
                save_path=scatter_corr_path,
            )
            print(f"  Scatter y_true vs y_pred (bias-corrected) disimpan ke: {scatter_corr_path}")

    # Simpan ringkasan global MAE/RMSE/MAPE/sMAPE/R2 ke CSV
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = os.path.join(REPORT_DIR, "tft_regression_summary.csv")
        os.makedirs(REPORT_DIR, exist_ok=True)
        summary_df.to_csv(summary_path, index=False)
        print("=" * 60)
        print(f"[INFO] Ringkasan metrik regresi disimpan ke: {summary_path}")

    # 6) Feature correlation heatmap
    print("=" * 60)
    print("[INFO] Mencari korelasi fitur dari tft_master.csv...")
    plot_feature_correlation_heatmap()


if __name__ == "__main__":
    main()
