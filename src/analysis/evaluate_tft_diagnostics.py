import os
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# ==============================
# KONFIGURASI GLOBAL
# ==============================

# Kalau True -> hitung juga metrik setelah bias correction (y_pred + mean residual)
APPLY_BIAS_CORRECTION = True

# Analisis fitur di level dataset (korelasi ke target, F-test, MI)
ENABLE_DATASET_FEATURE_ANALYSIS = True

# Analisis metrik per horizon (H+1..H+5) kalau kolom horizon ada
ENABLE_HORIZON_ANALYSIS = True

# Target utama di tft_master.csv (sesuaikan dengan yang dipakai TFT)
TARGET_COLUMN = "close"

# Kandidat fitur teknikal & sentimen (disesuaikan dengan pipeline-mu)
FEATURE_CANDIDATES: List[str] = [
    # --- Fitur teknikal ---
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
    # --- Fitur sentimen dari daily_sentiment / tft_master ---
    "sentiment_text_mean",
    "sentiment_market_mean",
    "sentiment_lex_mean",
    "sentiment_final_mean",
    "sentiment_conf_mean",
    "sentiment_conf_max",
    "strong_market_count",
    "strong_lex_count",
    "sentiment_mean",
    "sentiment_mean_3d",
    "news_count",
    "pos_count",
    "neg_count",
    "neu_count",
    "news_count_3d",
    "has_news",
    "sentiment_shock",
    "extreme_news",
    "sentiment_vol_7d",
    "sentiment_trend_5d",
]

# Kandidat kolom waktu yang mungkin ada di CSV prediksi
TIME_COLUMN_CANDIDATES: List[str] = ["time_idx", "date", "ds", "timestamp"]

# Kandidat kolom horizon yang mungkin ada di CSV prediksi
HORIZON_COLUMN_CANDIDATES: List[str] = ["horizon", "step", "offset", "h"]

# ==============================
# Path setup
# ==============================

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(ROOT_DIR, "data", "processed")
REPORT_DIR = os.path.join(ROOT_DIR, "reports")
FIG_DIR = os.path.join(REPORT_DIR, "figures")

os.makedirs(FIG_DIR, exist_ok=True)


# ==============================
# LOAD PREDICTIONS
# ==============================

def load_predictions() -> Dict[str, pd.DataFrame]:
    """
    Load baseline & hybrid prediction CSVs.
    Expected columns: y_true, y_pred (+ optional time_idx/date/horizon).

    Behaviour baru:
    - Kalau file baseline ada, tapi hybrid tidak → tetap jalan (pakai baseline saja).
    - Kalau keduanya tidak ada → raise FileNotFoundError.
    """
    paths = {
        "baseline": os.path.join(DATA_DIR, "predictions_tft_baseline_test.csv"),
        "hybrid": os.path.join(DATA_DIR, "predictions_tft_with_sentiment_test.csv"),
    }

    preds: Dict[str, pd.DataFrame] = {}
    for name, path in paths.items():
        if not os.path.exists(path):
            print(
                f"[WARN] File prediksi untuk '{name}' tidak ditemukan: {path}. "
                f"Model ini akan di-skip."
            )
            continue

        df = pd.read_csv(path)
        required = {"y_true", "y_pred"}
        if not required.issubset(df.columns):
            print(
                f"[WARN] File {path} tidak punya kolom {required}, "
                f"kolom sekarang: {df.columns.tolist()}. "
                f"Model '{name}' akan di-skip."
            )
            continue

        preds[name] = df

    if not preds:
        raise FileNotFoundError(
            "Tidak ada file prediksi yang valid (baseline/hybrid). "
            "Pastikan sudah menjalankan src.models.evaluate_tft_models terlebih dahulu."
        )

    return preds


# ==============================
# METRIK REGRESI
# ==============================

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Hitung metrik regresi standar.
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    # MAPE: hati-hati pembagian nol -> skip nol
    mask = y_true != 0
    if np.any(mask):
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = float("nan")
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
    df["bucket"], _ = pd.qcut(
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


def metrics_by_horizon(df: pd.DataFrame, model_name: str) -> None:
    """
    Hitung metrik regresi per horizon (H+1..H+K) jika ada kolom horizon.
    Cocok untuk TFT multi-horizon.
    """
    if not ENABLE_HORIZON_ANALYSIS:
        return

    horizon_col = None
    for c in HORIZON_COLUMN_CANDIDATES:
        if c in df.columns:
            horizon_col = c
            break

    if horizon_col is None:
        print(
            f"[WARN] Tidak menemukan kolom horizon "
            f"({HORIZON_COLUMN_CANDIDATES}) di prediksi {model_name}, "
            f"skip analisis per horizon."
        )
        return

    rows = []
    print(f"\n[{model_name.upper()}] Metrik regresi per horizon ({horizon_col}):")
    for h, sub in df.groupby(horizon_col):
        if len(sub) < 5:
            continue
        y_t = sub["y_true"].values
        y_p = sub["y_pred"].values
        m = regression_metrics(y_t, y_p)
        rows.append({"model": model_name, "horizon": h, **m})

        mape_str = "NaN" if np.isnan(m["MAPE(%)"]) else f"{m['MAPE(%)']:.2f}"
        print(
            f"  Horizon={h}: n={len(sub)}, "
            f"MAE={m['MAE']:.4f}, RMSE={m['RMSE']:.4f}, "
            f"MAPE={mape_str}, R2={m['R2']:.4f}"
        )

    if rows:
        out_df = pd.DataFrame(rows)
        out_path = os.path.join(REPORT_DIR, f"tft_regression_by_horizon_{model_name}.csv")
        out_df.to_csv(out_path, index=False)
        print(f"  [INFO] Ringkasan metrik per horizon disimpan ke: {out_path}")


# ==============================
# KORELASI FITUR (PAIRWISE & DENGAN TARGET)
# ==============================

def plot_feature_correlation_heatmap() -> None:
    """
    Baca tft_master.csv dan plot heatmap korelasi fitur utama dengan target.
    """
    master_path = os.path.join(DATA_DIR, "tft_master.csv")
    if not os.path.exists(master_path):
        print(f"[WARN] tft_master.csv tidak ditemukan di {master_path}, skip heatmap.")
        return

    df = pd.read_csv(master_path)

    cols = [c for c in FEATURE_CANDIDATES if c in df.columns]
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


def load_master_for_feature_analysis() -> Tuple[Optional[pd.DataFrame], List[str]]:
    """
    Load tft_master.csv dan pilih fitur yang tersedia + target.
    """
    master_path = os.path.join(DATA_DIR, "tft_master.csv")
    if not os.path.exists(master_path):
        print(f"[WARN] tft_master.csv tidak ditemukan di {master_path}, skip analisis fitur.")
        return None, []

    df = pd.read_csv(master_path)

    if TARGET_COLUMN not in df.columns:
        print(
            f"[WARN] Kolom target '{TARGET_COLUMN}' tidak ditemukan di tft_master.csv, "
            f"skip analisis fitur."
        )
        return None, []

    available = [c for c in FEATURE_CANDIDATES if c in df.columns]
    feature_cols = [c for c in available if c != TARGET_COLUMN]

    if not feature_cols:
        print("[WARN] Tidak ada fitur yang tersedia untuk analisis.")
        return None, []

    return df, feature_cols


def compute_feature_target_correlation(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
) -> pd.DataFrame:
    """
    Korelasi Pearson fitur terhadap target, diurutkan berdasarkan |corr|.
    """
    rows = []
    for col in feature_cols:
        if col not in df.columns:
            continue
        if df[col].dtype.kind not in "bifc":
            continue
        corr_val = df[col].corr(df[target_col])
        rows.append(
            {
                "feature": col,
                "corr_with_target": corr_val,
                "abs_corr": abs(corr_val),
            }
        )
    if not rows:
        print("[WARN] Tidak bisa menghitung korelasi fitur-target (tidak ada fitur numerik).")
        return pd.DataFrame()

    corr_df = pd.DataFrame(rows).sort_values("abs_corr", ascending=False)
    out_path = os.path.join(REPORT_DIR, "feature_target_correlation.csv")
    os.makedirs(REPORT_DIR, exist_ok=True)
    corr_df.to_csv(out_path, index=False)
    print(f"[INFO] Korelasi fitur-target disimpan ke: {out_path}")
    print("[INFO] Top fitur berdasarkan |corr| terhadap target:")
    for _, row in corr_df.head(10).iterrows():
        print(f"  {row['feature']:25s} corr={row['corr_with_target']:.4f}")
    return corr_df


def compute_filter_feature_scores(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
) -> pd.DataFrame:
    """
    Hitung F-test (f_regression) & Mutual Information untuk ranking fitur.
    """
    try:
        from sklearn.feature_selection import f_regression, mutual_info_regression
    except ImportError:
        print("[WARN] sklearn.feature_selection tidak tersedia, skip filter methods.")
        return pd.DataFrame()

    cols = feature_cols + [target_col]
    df_clean = df[cols].dropna()
    if df_clean.empty:
        print("[WARN] Data bersih untuk filter methods kosong, skip.")
        return pd.DataFrame()

    X = df_clean[feature_cols].values
    y = df_clean[target_col].astype(float).values

    f_vals, f_pvals = f_regression(X, y)
    mi_vals = mutual_info_regression(X, y, random_state=42)

    scores_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "F_score": f_vals,
            "F_pvalue": f_pvals,
            "MI": mi_vals,
        }
    ).sort_values("MI", ascending=False)

    out_path = os.path.join(REPORT_DIR, "feature_filter_scores.csv")
    scores_df.to_csv(out_path, index=False)
    print(f"[INFO] Hasil filter methods (F-test & MI) disimpan ke: {out_path}")
    print("[INFO] Top fitur berdasarkan Mutual Information:")
    for _, row in scores_df.head(10).iterrows():
        print(f"  {row['feature']:25s} MI={row['MI']:.4f}, F={row['F_score']:.2f}")
    return scores_df


def run_dataset_feature_analysis() -> None:
    """
    Analisis fitur di level dataset (bukan model):
    - Korelasi fitur-target
    - F-test & Mutual Information
    (semua relevan untuk regresi TFT karena target-nya sama dengan model)
    """
    if not ENABLE_DATASET_FEATURE_ANALYSIS:
        return

    df, feature_cols = load_master_for_feature_analysis()
    if df is None or not feature_cols:
        return

    print("[INFO] Analisis fitur di level dataset dimulai...")
    compute_feature_target_correlation(df, feature_cols, TARGET_COLUMN)
    compute_filter_feature_scores(df, feature_cols, TARGET_COLUMN)


# ==============================
# PLOTTING RESIDUAL & DIAGNOSTIK
# ==============================

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


def plot_residual_vs_pred(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    save_path: str,
) -> None:
    residuals = y_true - y_pred
    plt.figure(figsize=(5, 4))
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.xlabel("y_pred")
    plt.ylabel("Residual (y_true - y_pred)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_residual_over_time(
    df_pred: pd.DataFrame,
    model_name: str,
) -> None:
    """
    Plot residual vs waktu jika ada kolom waktu di df_pred.
    Sangat relevan untuk TFT (time series forecasting).
    """
    time_col = None
    for c in TIME_COLUMN_CANDIDATES:
        if c in df_pred.columns:
            time_col = c
            break

    if time_col is None:
        print(
            f"[WARN] Tidak menemukan kolom waktu ({TIME_COLUMN_CANDIDATES}) "
            f"di prediksi {model_name}, skip residual vs waktu."
        )
        return

    df = df_pred.copy()
    df["residual"] = df["y_true"] - df["y_pred"]
    df = df.sort_values(time_col)

    plt.figure(figsize=(10, 4))
    plt.plot(df[time_col].values, df["residual"].values, marker="o", linestyle="-", alpha=0.7)
    plt.axhline(0.0, color="black", linewidth=1)
    plt.xlabel(time_col)
    plt.ylabel("Residual")
    plt.title(f"TFT {model_name.capitalize()} – Residual vs waktu")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, f"residual_over_time_{model_name}.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"  Plot residual vs waktu disimpan ke: {path}")


def plot_residual_qq(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    save_path: str,
) -> None:
    """
    Q–Q plot residual vs distribusi normal (opsional, hanya kalau scipy tersedia).
    """
    try:
        from scipy import stats
    except ImportError:
        print("[WARN] Paket 'scipy' belum terinstal, skip Q–Q plot residual.")
        return

    residuals = y_true - y_pred
    if residuals.size == 0:
        print("[WARN] Residual kosong, skip Q–Q plot.")
        return

    plt.figure(figsize=(5, 5))
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# ==============================
# BIAS CORRECTION
# ==============================

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


# ==============================
# MAIN
# ==============================

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

        # 3) Metrik per horizon (kalau kolom horizon ada)
        metrics_by_horizon(df, name)

        # 4) Plot residual & scatter (tanpa koreksi)
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

        resid_vs_pred_path = os.path.join(FIG_DIR, f"residual_vs_pred_{name}.png")
        plot_residual_vs_pred(
            y_true,
            y_pred,
            title=f"TFT {name.capitalize()} – Residual vs y_pred (raw)",
            save_path=resid_vs_pred_path,
        )
        print(f"  Plot residual vs y_pred disimpan ke: {resid_vs_pred_path}")

        # Residual vs waktu (jika ada kolom waktu di df)
        plot_residual_over_time(df, name)

        # Q–Q plot residual (opsional)
        qq_path = os.path.join(FIG_DIR, f"residual_qq_{name}.png")
        plot_residual_qq(
            y_true,
            y_pred,
            title=f"TFT {name.capitalize()} – Q–Q plot residual (raw)",
            save_path=qq_path,
        )

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

            scatter_corr_path = os.path.join(
                FIG_DIR, f"true_vs_pred_{name}_bias_corrected.png"
            )
            plot_true_vs_pred_scatter(
                y_true,
                y_pred_corr,
                title=f"TFT {name.capitalize()} – y_true vs y_pred (bias-corrected)",
                save_path=scatter_corr_path,
            )
            print(
                f"  Scatter y_true vs y_pred (bias-corrected) disimpan ke: "
                f"{scatter_corr_path}"
            )

    # Simpan ringkasan global MAE/RMSE/MAPE/sMAPE/R2 ke CSV
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = os.path.join(REPORT_DIR, "tft_regression_summary.csv")
        os.makedirs(REPORT_DIR, exist_ok=True)
        summary_df.to_csv(summary_path, index=False)
        print("=" * 60)
        print(f"[INFO] Ringkasan metrik regresi disimpan ke: {summary_path}")

    # 6) Feature correlation heatmap (pairwise)
    print("=" * 60)
    print("[INFO] Mencari korelasi fitur dari tft_master.csv...")
    plot_feature_correlation_heatmap()

    # 7) Analisis fitur di level dataset (F-test, MI, korelasi ke target)
    run_dataset_feature_analysis()


if __name__ == "__main__":
    main()
