# D:\skripsi\tft\src\dashboard\app.py

import os
import subprocess
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------
# Path dasar
# ---------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")
REPORTS_DIR = os.path.join(ROOT_DIR, "reports")
FIG_DIR = os.path.join(REPORTS_DIR, "figures")

TFT_MASTER_PATH = os.path.join(DATA_PROCESSED_DIR, "tft_master.csv")
FORECAST_TIMELINE_PATH = os.path.join(DATA_PROCESSED_DIR, "tft_forecasts_timeline.csv")
REG_SUMMARY_PATH = os.path.join(REPORTS_DIR, "tft_regression_summary.csv")
BACKTEST_FULL_PATH = os.path.join(DATA_PROCESSED_DIR, "tft_backtest_full.csv")


# ---------------------------
# Helper
# ---------------------------


def run_cmd(label: str, cmd: str) -> None:
    """Jalankan perintah shell dan tampilkan log ke Streamlit."""
    st.write(f"**▶ {label}**")
    st.code(cmd, language="bash")
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=ROOT_DIR,
            capture_output=True,
            text=True,
        )
        if result.stdout:
            st.text(result.stdout)
        if result.stderr:
            st.text_area("stderr", result.stderr, height=150)
        if result.returncode != 0:
            st.error(f"Perintah gagal (exit code {result.returncode})")
        else:
            st.success("Selesai ✅")
    except Exception as e:
        st.error(f"Error saat menjalankan perintah: {e}")


@st.cache_data(show_spinner=False)
def load_tft_master() -> Optional[pd.DataFrame]:
    if not os.path.exists(TFT_MASTER_PATH):
        return None
    df = pd.read_csv(TFT_MASTER_PATH, parse_dates=["date"])
    return df


@st.cache_data(show_spinner=False)
def load_forecast_timeline() -> Optional[pd.DataFrame]:
    if not os.path.exists(FORECAST_TIMELINE_PATH):
        return None
    df = pd.read_csv(FORECAST_TIMELINE_PATH, parse_dates=["date"])
    return df


@st.cache_data(show_spinner=False)
def load_regression_summary() -> Optional[pd.DataFrame]:
    if not os.path.exists(REG_SUMMARY_PATH):
        return None
    return pd.read_csv(REG_SUMMARY_PATH)


@st.cache_data(show_spinner=False)
def load_backtest_full() -> Optional[pd.DataFrame]:
    """Load hasil rolling backtest multi-horizon."""
    if not os.path.exists(BACKTEST_FULL_PATH):
        return None
    df = pd.read_csv(BACKTEST_FULL_PATH, parse_dates=["date_target"])
    return df


def metric_safe(label: str, value):
    """Wrapper supaya nggak error kalau value bukan angka."""
    try:
        st.metric(label, value)
    except TypeError:
        st.metric(label, str(value))


# ---------------------------
# UI
# ---------------------------


def main():
    st.set_page_config(
        page_title="TFT Stock Forecast Dashboard",
        layout="wide",
    )

    st.title("📈 TFT Stock Forecast Dashboard (BBCA & BBRI)")
    st.caption("Dataset: harga saham + indikator teknikal + sentimen berita")

    # =====================
    # Sidebar: pipeline
    # =====================
    st.sidebar.header("⚙️ Pipeline & Evaluasi")

    if st.sidebar.button(
        "1️⃣ Run Full Pipeline (News + Harga + Sentimen + Train + Eval)",
        type="primary",
    ):
        with st.expander("Log Full Pipeline", expanded=True):
            cmds = [
                # 1) News
                "python -m src.data.fetch_news_rss_google",
                "python -m src.data.fetch_news_yahoo",
                "python -m src.data.merge_news_sources",
                "python -m src.data.preprocess_news_text",
                # 2) Prices + teknikal
                "python -m src.data.download_prices_yahoo",
                "python -m src.data.compute_technical_indicators",
                "python -m src.data.check_price_calendar",
                # 3) Sentimen
                "python -m src.data.gpt_sentiment_labeling",
                "python -m src.data.aggregate_daily_sentiment",
                # 4) Master dataset
                "python -m src.data.build_tft_master_dataset",
                # 5) Train + update experiments
                "python -m src.models.train_tft_baseline",
                "python -m src.models.train_tft_with_sentiment",
                "python -m src.utils.update_experiments_best_ckpt",
                # 6) Evaluasi + ringkasan
                "python -m src.models.evaluate_tft_models",
                "python -m src.analysis.evaluate_tft_diagnostics",
                "python -m src.analysis.compute_vif_features",
                # 7) Build timeline forecast (future-only) + rolling backtest penuh
                "python -m src.models.evaluate_tft_backtest",
                "python -m src.models.evaluate_tft_backtest_full",
            ]
            for c in cmds:
                run_cmd("Run", c)

    if st.sidebar.button(
        "2️⃣ Update Data Harian (News + Harga + Teknikal + Sentimen + Rebuild Master)",
        type="secondary",
    ):
        with st.expander("Log Update Harian", expanded=True):
            cmds = [
                "python -m src.data.fetch_news_rss_google",
                "python -m src.data.fetch_news_yahoo",
                "python -m src.data.merge_news_sources",
                "python -m src.data.preprocess_news_text",
                "python -m src.data.download_prices_yahoo",
                "python -m src.data.compute_technical_indicators",
                "python -m src.data.gpt_sentiment_labeling",
                "python -m src.data.aggregate_daily_sentiment",
                "python -m src.data.build_tft_master_dataset",
                "python -m src.models.evaluate_tft_backtest",
                "python -m src.models.evaluate_tft_backtest_full",
            ]
            for c in cmds:
                run_cmd("Run", c)

    if st.sidebar.button("3️⃣ Evaluasi Ulang Model + Ringkasan", type="secondary"):
        with st.expander("Log Evaluasi", expanded=True):
            cmds = [
                "python -m src.models.evaluate_tft_models",
                "python -m src.analysis.evaluate_tft_diagnostics",
                "python -m src.analysis.compute_vif_features",
                "python -m src.models.evaluate_tft_backtest",
                "python -m src.models.evaluate_tft_backtest_full",
            ]
            for c in cmds:
                run_cmd("Run", c)

    st.sidebar.markdown("---")
    st.sidebar.info(
        "Urutan aman:\n"
        "1) Update data harian\n"
        "2) Train / Evaluasi jika perlu\n"
        "3) Jalankan: `evaluate_tft_backtest` & `evaluate_tft_backtest_full`\n"
        "4) Refresh dashboard"
    )

    # =====================
    # Load data utama
    # =====================
    df_master = load_tft_master()
    df_timeline = load_forecast_timeline()
    reg_summary = load_regression_summary()
    df_bt_full = load_backtest_full()

    if df_master is None:
        st.error(
            f"File master dataset **tft_master.csv** belum ada di:\n`{TFT_MASTER_PATH}`"
        )
        st.stop()

    # Pastikan kolom dasar ada
    required_cols = {"date", "ticker", "split", "close"}
    missing = required_cols - set(df_master.columns)
    if missing:
        st.error(f"Kolom wajib hilang dari tft_master.csv: {missing}")
        st.stop()

    # =====================
    # Sidebar filter
    # =====================
    tickers = sorted(df_master["ticker"].dropna().unique())
    selected_ticker = st.sidebar.selectbox(
        "Pilih ticker", options=tickers, index=0 if tickers else None
    )

    # Rentang tanggal historis (dari master)
    hist_min_date = df_master["date"].min()
    hist_max_date = df_master["date"].max()

    # Kalau ada timeline forecast, extend max_date sampai future
    if df_timeline is not None and not df_timeline.empty:
        total_max_date = max(hist_max_date, df_timeline["date"].max())
    else:
        total_max_date = hist_max_date

    # Date filter di sidebar pakai rentang historis -> future (kalau ada)
    date_range = st.sidebar.date_input(
        "Filter tanggal (untuk chart & tabel)",
        value=(hist_min_date.date(), total_max_date.date()),
        min_value=hist_min_date.date(),
        max_value=total_max_date.date(),
    )

    if isinstance(date_range, tuple):
        start_date, end_date = date_range
    else:
        start_date = hist_min_date.date()
        end_date = total_max_date.date()

    # Filter data historis sesuai ticker & range untuk overview
    mask_hist = (
        (df_master["ticker"] == selected_ticker)
        & (df_master["date"] >= pd.to_datetime(start_date))
        & (df_master["date"] <= pd.to_datetime(end_date))
    )
    df_ticker = df_master[mask_hist].copy().sort_values("date")

    # =====================
    # Overview
    # =====================
    st.subheader("📊 Overview Data")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        metric_safe("Ticker", selected_ticker)

    with col2:
        if not df_ticker.empty:
            metric_safe("Tanggal awal (filter)", df_ticker["date"].min().strftime("%Y-%m-%d"))
        else:
            metric_safe("Tanggal awal (filter)", "-")

    with col3:
        if not df_ticker.empty:
            metric_safe("Tanggal akhir (filter)", df_ticker["date"].max().strftime("%Y-%m-%d"))
        else:
            metric_safe("Tanggal akhir (filter)", "-")

    with col4:
        metric_safe("Jumlah observasi (historis)", len(df_ticker))

    st.dataframe(
        df_ticker[["date", "split", "close"]].tail(20).reset_index(drop=True),
        width="stretch",
    )

    # =====================
    # Tabs
    # =====================
    tab_chart, tab_metrics, tab_residual, tab_backtest = st.tabs(
        [
            "📈 Chart Harga vs Forecast",
            "📋 Ringkasan Metrik",
            "🔍 Residual & Korelasi",
            "🧪 Backtest Multi-Horizon",
        ]
    )

    # ---------------------
    # TAB CHART
    # ---------------------
    with tab_chart:
        st.markdown("### 📈 Harga Aktual vs Prediksi TFT (dari awal + future)")

        if (
            (df_timeline is None or df_timeline.empty)
            and (df_bt_full is None or df_bt_full.empty)
        ):
            st.warning(
                "Belum ada file forecast/backtest.\n\n"
                "- Jalankan: `python -m src.models.evaluate_tft_backtest`\n"
                "- dan: `python -m src.models.evaluate_tft_backtest_full`"
            )
        else:
            # 1) Harga aktual (historis)
            df_price = df_ticker[["date", "close"]].copy()
            df_price = df_price.set_index("date")

            # 2) Prediksi historis H+1 dari rolling backtest (baseline & hybrid)
            df_bt_pivot = None
            if df_bt_full is not None and not df_bt_full.empty:
                df_bt_ticker = df_bt_full[
                    (df_bt_full["ticker"] == selected_ticker)
                    & (df_bt_full["horizon"] == 1)  # H+1
                ].copy()

                # filter tanggal sesuai range
                mask_bt = (
                    (df_bt_ticker["date_target"] >= pd.to_datetime(start_date))
                    & (df_bt_ticker["date_target"] <= pd.to_datetime(end_date))
                )
                df_bt_ticker = df_bt_ticker[mask_bt]

                if not df_bt_ticker.empty:
                    df_bt_pivot = df_bt_ticker.pivot_table(
                        index="date_target",
                        columns="model",
                        values="y_pred",
                        aggfunc="mean",
                    )
                    df_bt_pivot.index.name = "date"
                    df_bt_pivot = df_bt_pivot.rename(
                        columns={
                            "baseline": "pred_baseline",
                            "hybrid": "pred_hybrid",
                        }
                    )

            # 3) Prediksi future dari evaluate_tft_backtest (5 hari ke depan)
            df_future_pivot = None
            if df_timeline is not None and not df_timeline.empty:
                df_time_ticker = df_timeline[
                    df_timeline["ticker"] == selected_ticker
                ].copy()
                mask_t = (
                    (df_time_ticker["date"] >= pd.to_datetime(start_date))
                    & (df_time_ticker["date"] <= pd.to_datetime(end_date))
                )
                df_time_ticker = df_time_ticker[mask_t]

                if not df_time_ticker.empty:
                    df_future_pivot = df_time_ticker.set_index("date")[
                        [c for c in ["pred_baseline", "pred_hybrid"] if c in df_time_ticker.columns]
                    ]

            # 4) Gabungkan semua index tanggal
            idx = pd.Index([])
            if not df_price.empty:
                idx = idx.union(df_price.index)
            if df_bt_pivot is not None:
                idx = idx.union(df_bt_pivot.index)
            if df_future_pivot is not None:
                idx = idx.union(df_future_pivot.index)

            idx = idx.sort_values()

            if len(idx) == 0:
                st.warning("Tidak ada data untuk kombinasi ticker + range tanggal ini.")
            else:
                df_chart = pd.DataFrame(index=idx)

                # isi harga aktual
                if not df_price.empty:
                    df_chart["close"] = df_price["close"]

                # isi prediksi historis (backtest H+1)
                if df_bt_pivot is not None:
                    for col in ["pred_baseline", "pred_hybrid"]:
                        if col in df_bt_pivot.columns:
                            df_chart[col] = df_bt_pivot[col]

                # isi prediksi future (timeline H+1..H+5)
                if df_future_pivot is not None:
                    for col in ["pred_baseline", "pred_hybrid"]:
                        if col in df_future_pivot.columns:
                            if col in df_chart.columns:
                                # untuk future dates yang belum ada nilai → isi
                                df_chart[col] = df_chart[col].combine_first(
                                    df_future_pivot[col]
                                )
                            else:
                                df_chart[col] = df_future_pivot[col]

                # rename untuk ditampilkan
                rename_map = {"close": "Harga Aktual"}
                if "pred_baseline" in df_chart.columns:
                    rename_map["pred_baseline"] = "Prediksi Baseline (H+1)"
                if "pred_hybrid" in df_chart.columns:
                    rename_map["pred_hybrid"] = "Prediksi Hybrid (H+1)"

                df_plot = df_chart.rename(columns=rename_map)

                st.line_chart(df_plot, height=420)

                st.caption(
                    "- **Harga Aktual**: dari `tft_master.csv`.\n"
                    "- **Prediksi Baseline/Hybrid (H+1)**:\n"
                    "   - Periode historis → diambil dari **rolling backtest full** (`tft_backtest_full.csv`).\n"
                    "   - Periode future → diambil dari **forecast 5 hari ke depan** "
                    "(`tft_forecasts_timeline.csv`)."
                )

                with st.expander("Lihat data mentah (gabungan historis + future)"):
                    st.dataframe(
                        df_chart.reset_index(names="date"),
                        width="stretch",
                    )

    # ---------------------
    # TAB METRICS
    # ---------------------
    with tab_metrics:
        st.markdown("### 📋 Ringkasan Metrik Evaluasi TFT")

        if reg_summary is None or reg_summary.empty:
            st.warning(
                "File `tft_regression_summary.csv` belum ditemukan.\n\n"
                "Jalankan: `python -m src.analysis.evaluate_tft_diagnostics` "
                "untuk membuat summary."
            )
        else:
            st.dataframe(reg_summary, width="stretch")

            # Tampilkan metrik utama sebagai metric cards
            col_a, col_b, col_c = st.columns(3)

            # Ambil baseline raw
            try:
                base_raw = reg_summary[
                    (reg_summary["model"] == "baseline")
                    & (reg_summary["kind"] == "raw")
                ].iloc[0]
                with col_a:
                    metric_safe("Baseline MAE (raw)", round(base_raw["MAE"], 3))
                with col_b:
                    metric_safe("Baseline RMSE (raw)", round(base_raw["RMSE"], 3))
                with col_c:
                    metric_safe(
                        "Baseline MAPE % (raw)", round(base_raw["MAPE(%)"], 3)
                    )
            except Exception:
                pass

            st.caption(
                "Baris `bias_corrected` adalah metrik setelah koreksi bias "
                "(menambah mean residual ke prediksi)."
            )

    # ---------------------
    # TAB RESIDUAL & KORELASI
    # ---------------------
    with tab_residual:
        st.markdown("### 🔍 Visualisasi Residual & Korelasi Fitur")

        cols_resid = st.columns(2)

        # Residual histogram baseline (raw + bias-corrected)
        resid_base_path = os.path.join(FIG_DIR, "residual_hist_baseline.png")
        resid_base_bc_path = os.path.join(
            FIG_DIR, "residual_hist_baseline_bias_corrected.png"
        )

        with cols_resid[0]:
            st.markdown("**Histogram Residual – Baseline (raw)**")
            if os.path.exists(resid_base_path):
                st.image(resid_base_path)
            else:
                st.info(f"File belum ada: `{resid_base_path}`")

        with cols_resid[1]:
            st.markdown("**Histogram Residual – Baseline (bias-corrected)**")
            if os.path.exists(resid_base_bc_path):
                st.image(resid_base_bc_path)
            else:
                st.info(f"File belum ada: `{resid_base_bc_path}`")

        st.markdown("---")

        # Residual histogram hybrid (jika ada)
        cols_resid2 = st.columns(2)

        resid_hybrid_path = os.path.join(FIG_DIR, "residual_hist_hybrid.png")
        resid_hybrid_bc_path = os.path.join(
            FIG_DIR, "residual_hist_hybrid_bias_corrected.png"
        )

        with cols_resid2[0]:
            st.markdown("**Histogram Residual – Hybrid (raw)**")
            if os.path.exists(resid_hybrid_path):
                st.image(resid_hybrid_path)
            else:
                st.info(f"File belum ada: `{resid_hybrid_path}`")

        with cols_resid2[1]:
            st.markdown("**Histogram Residual – Hybrid (bias-corrected)**")
            if os.path.exists(resid_hybrid_bc_path):
                st.image(resid_hybrid_bc_path)
            else:
                st.info(f"File belum ada: `{resid_hybrid_bc_path}`")

        st.markdown("---")

        # Korelasi fitur
        st.markdown("**Heatmap Korelasi Fitur Teknis & Sentimen**")
        corr_path = os.path.join(FIG_DIR, "feature_correlation_heatmap.png")
        if os.path.exists(corr_path):
            st.image(corr_path)
        else:
            st.info(
                f"File heatmap belum ada: `{corr_path}`\n\n"
                "Pastikan sudah menjalankan: `python -m src.analysis.evaluate_tft_diagnostics`"
            )

    # ---------------------
    # TAB BACKTEST MULTI-HORIZON
    # ---------------------
    with tab_backtest:
        st.markdown("### 🧪 Rolling Backtest Multi-Horizon (H+1..H+5)")

        if df_bt_full is None or df_bt_full.empty:
            st.warning(
                "File `tft_backtest_full.csv` belum ditemukan atau kosong.\n\n"
                "Jalankan: `python -m src.models.evaluate_tft_backtest_full` "
                "dari sidebar atau terminal."
            )
        else:
            df_bt_ticker = df_bt_full[df_bt_full["ticker"] == selected_ticker].copy()

            # Filter by date range (pakai date_target)
            mask_bt = (
                (df_bt_ticker["date_target"] >= pd.to_datetime(start_date))
                & (df_bt_ticker["date_target"] <= pd.to_datetime(end_date))
            )
            df_bt_ticker = df_bt_ticker[mask_bt]

            if df_bt_ticker.empty:
                st.info(
                    "Tidak ada data backtest untuk kombinasi ticker + range tanggal ini.\n"
                    "Coba perpanjang range tanggal di sidebar."
                )
            else:
                # Hitung error
                df_bt_ticker["error"] = df_bt_ticker["y_pred"] - df_bt_ticker["y_true"]
                df_bt_ticker["abs_error"] = df_bt_ticker["error"].abs()

                st.markdown("#### 📌 Ringkasan MAE per Horizon & Model")
                mae_table = (
                    df_bt_ticker.groupby(["model", "horizon"])["abs_error"]
                    .mean()
                    .reset_index()
                    .sort_values(["model", "horizon"])
                )
                mae_table["MAE"] = mae_table["abs_error"]
                mae_table = mae_table.drop(columns=["abs_error"])

                st.dataframe(
                    mae_table.pivot(index="horizon", columns="model", values="MAE"),
                    width="stretch",
                )

                st.caption(
                    "- MAE dihitung dari rolling backtest (setiap titik waktu dengan window encoder 60 hari).\n"
                    "- Horizon = 1 artinya H+1, horizon = 5 artinya H+5."
                )

                st.markdown("#### 📈 MAE per Horizon")
                mae_pivot = mae_table.pivot(index="horizon", columns="model", values="MAE")
                st.line_chart(mae_pivot)

                st.markdown("#### 🔍 Contoh Data Backtest (10 baris terakhir)")
                st.dataframe(
                    df_bt_ticker.sort_values(["date_target", "horizon", "model"])
                    .tail(10)
                    .reset_index(drop=True),
                    width="stretch",
                )


if __name__ == "__main__":
    main()
