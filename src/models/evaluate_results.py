import os
import yaml
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ==============================================================================
# 🛠️ KONFIGURASI SPESIFIK (SESUAI REQUEST) 🛠️
# ==============================================================================

# 1. VERSI LOG (Untuk Grafik)
BASELINE_LOG_VERSION = 13    # Lokasi: logs/baseline_model/version_13
HYBRID_LOG_VERSION   = 135   # Lokasi: lightning_logs/version_135

# 2. FILE MODEL (Untuk Prediksi)
# Masukkan nama file spesifik agar tidak salah ambil.
# Jika diisi None, script akan mencari val_loss terendah di folder.
BASELINE_CKPT_EXACT = r"D:\skripsi\tft\checkpoints\baseline\tft-baseline-epoch=12-val_loss=47.51.ckpt"
HYBRID_CKPT_EXACT   = None # Biarkan None agar auto-detect yang terbaik (28.22)

# ==============================================================================
# KONFIGURASI PATH
# ==============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "model_tft.yaml"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "tft_master.csv"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
REPORTS_DIR = PROJECT_ROOT / "reports"

# LOKASI LOG FOLDER (BEDA TEMPAT)
PATH_BASELINE_LOGS = PROJECT_ROOT / "logs" / "baseline_model"
PATH_HYBRID_LOGS   = PROJECT_ROOT / "lightning_logs"

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# Setup Style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.4)

def load_config():
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r") as f:
            return yaml.safe_load(f)
    return {}

def find_checkpoint(model_type, exact_path=None):
    """
    Mencari checkpoint. Jika exact_path diisi, pakai itu.
    Jika tidak, cari yang loss terendah di folder.
    """
    # 1. Prioritas Manual
    if exact_path:
        p = Path(exact_path)
        if p.exists():
            return p
        print(f"⚠️ Warning: Checkpoint manual tidak ditemukan: {exact_path}")
        print("   -> Mencoba auto-detect...")

    # 2. Auto Detect
    search_dirs = [
        PROJECT_ROOT / "checkpoints" / model_type,
        PROJECT_ROOT / "checkpoints" / f"{model_type}_tuned",
        PROJECT_ROOT / "checkpoints" / f"{model_type}_aggressive"
    ]
    candidates = []
    for d in search_dirs:
        if d.exists():
            candidates.extend(list(d.glob("*.ckpt")))
    
    if not candidates: return None
    # Cari val_loss terkecil
    return min(candidates, key=lambda x: float(x.stem.split("val_loss=")[-1]) if "val_loss=" in x.stem else 9999)

# ==============================================================================
# BAGIAN 1: EXTRACT LOGS
# ==============================================================================

def get_log_data(version_id, base_log_dir):
    version_dir = base_log_dir / f"version_{version_id}"
    csv_path = version_dir / "metrics.csv"
    
    print(f"   📂 Memproses Log: {version_dir.name} di {base_log_dir.name}...")
    
    # Cek CSV dulu (Prioritas)
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            print(f"      ✅ Loaded metrics.csv ({len(df)} epochs).")
            return df
        except: pass

    # Fallback ke Events File
    event_files = list(version_dir.glob("events.out.tfevents*"))
    if not event_files:
        print(f"      ❌ File events tidak ditemukan.")
        return None
        
    evt_file = max(event_files, key=lambda f: f.stat().st_size)
    
    try:
        event_acc = EventAccumulator(str(evt_file))
        event_acc.Reload()
        tags = event_acc.Tags()['scalars']
        
        data = {}
        val_tag = next((t for t in ['val_loss', 'val_loss_epoch'] if t in tags), None)
        if val_tag:
            events = event_acc.Scalars(val_tag)
            data['step'] = [e.step for e in events]
            data['val_loss'] = [e.value for e in events]
        else:
            print("      ⚠️ Tag val_loss tidak ditemukan.")
            return None

        train_tag = next((t for t in ['train_loss_epoch', 'train_loss'] if t in tags), None)
        if train_tag:
            events = event_acc.Scalars(train_tag)
            train_dict = {e.step: e.value for e in events}
            if 'step' in data:
                data['train_loss'] = [train_dict.get(s, np.nan) for s in data['step']]
                
        if 'epoch' in tags:
            events = event_acc.Scalars('epoch')
            epoch_dict = {e.step: e.value for e in events}
            if 'step' in data:
                data['epoch'] = [epoch_dict.get(s, np.nan) for s in data['step']]
        else:
            data['epoch'] = range(len(data['step']))
                
        df = pd.DataFrame(data)
        df = df.ffill().bfill()
        return df
        
    except Exception as e:
        print(f"      ❌ Error: {e}")
        return None

def plot_loss_curves(hist_base, hist_sent):
    print("\n📈 Generating Loss Comparison Plots...")
    plt.figure(figsize=(10, 6))
    
    # Baseline
    if hist_base is not None and not hist_base.empty:
        min_b = hist_base['val_loss'].min()
        plt.plot(hist_base['epoch'], hist_base['val_loss'], label=f'Baseline (v{BASELINE_LOG_VERSION}, Min: {min_b:.2f})', 
                 color='gray', linestyle='--', linewidth=2)

    # Hybrid
    if hist_sent is not None and not hist_sent.empty:
        min_s = hist_sent['val_loss'].min()
        plt.plot(hist_sent['epoch'], hist_sent['val_loss'], label=f'Hybrid (v{HYBRID_LOG_VERSION}, Min: {min_s:.2f})', 
                 color='#c44e52', linewidth=2.5)
    
    plt.title(f'Validation Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Quantile Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(FIGURES_DIR / "loss_comparison.png", dpi=300)
    plt.close()

    # Individual Plots
    for name, hist, color in [("baseline", hist_base, "blue"), ("hybrid", hist_sent, "red")]:
        if hist is None or hist.empty: continue
        plt.figure(figsize=(10, 6))
        if 'train_loss' in hist.columns:
            clean = hist.dropna(subset=['train_loss'])
            plt.plot(clean['epoch'], clean['train_loss'], label='Train Loss', linestyle='--', color='gray')
        plt.plot(hist['epoch'], hist['val_loss'], label='Val Loss', linewidth=2, color=color)
        plt.title(f'Learning Curve: {name.capitalize()}')
        plt.legend()
        plt.savefig(FIGURES_DIR / f"loss_curve_{name}.png", dpi=300)
        plt.close()

# ==============================================================================
# BAGIAN 2: INTERPRETABILITY
# ==============================================================================

def plot_interpretability(model, raw_prediction, prefix):
    print(f"🎨 Generating Interpretability Plots for {prefix}...")
    
    # Feature Importance
    interpretation = model.interpret_output(raw_prediction.output, reduction="sum")
    enc_imp = pd.Series(interpretation["encoder_variables"].detach().cpu().numpy(), 
                        index=model.encoder_variables)
    
    plt.figure(figsize=(10, 8))
    enc_imp.sort_values().plot.barh(color='#4c72b0')
    plt.title(f"Encoder Feature Importance ({prefix})")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{prefix}_feature_importance.png", dpi=300)
    plt.close()
    
    # Attention
    attention = interpretation["attention"].detach().cpu().numpy()
    if attention.ndim == 2: avg_attention = attention.mean(axis=0)
    else: avg_attention = attention
        
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(len(avg_attention)) - len(avg_attention), avg_attention, color='#c44e52')
    plt.title(f"Average Attention Pattern ({prefix})")
    plt.xlabel("Encoder Steps (Past)")
    plt.savefig(FIGURES_DIR / f"{prefix}_attention.png", dpi=300)
    plt.close()
    
    # Forecast
    plt.figure(figsize=(12, 6))
    model.plot_prediction(raw_prediction.x, raw_prediction.output, idx=0, add_loss_to_title=True)
    plt.title(f"Sample Forecast: {prefix}")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{prefix}_sample_forecast.png", dpi=300)
    plt.close()

# ==============================================================================
# BAGIAN 3: PREDIKSI
# ==============================================================================

def generate_prediction_df(model_path, data_path, model_name):
    if not model_path: return None
    print(f"\n⚙️  Memproses Model: {model_name} ({model_path.name})...")
    
    try:
        tft = TemporalFusionTransformer.load_from_checkpoint(
            model_path, map_location="cpu", weights_only=False
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    for c in ['ticker', 'month', 'day_of_week']: df[c] = df[c].astype(str)
    df = df.fillna(0)

    if 'split' in df.columns: test_rows = df[df['split'] == 'test']
    else: test_rows = df.tail(1000)

    max_enc = tft.dataset_parameters['max_encoder_length']
    df_eval = df[df['time_idx'] >= (test_rows['time_idx'].min() - max_enc - 5)].copy()
    
    validation = TimeSeriesDataSet.from_parameters(
        tft.dataset_parameters, df_eval, predict=False, stop_randomization=True
    )
    val_dataloader = validation.to_dataloader(train=False, batch_size=64, num_workers=0)
    
    raw_predictions = tft.predict(val_dataloader, mode="raw", return_x=True, return_index=True)
    plot_interpretability(tft, raw_predictions, model_name.lower())
    
    preds = tft.to_prediction(raw_predictions.output).cpu().numpy()
    actuals = raw_predictions.x["decoder_target"].cpu().numpy()
    
    results = []
    price_map = df.set_index(['ticker', 'time_idx'])['close'].to_dict()
    tickers = raw_predictions.index['ticker'].values
    start_times = raw_predictions.index['time_idx'].values
    horizon = preds.shape[1]
    
    for i in range(len(preds)):
        ticker = tickers[i]
        t_start = start_times[i]
        try: prev_close = price_map.get((ticker, t_start - 1), np.nan)
        except: prev_close = np.nan
        
        for h in range(horizon):
            y_p, y_t = preds[i, h], actuals[i, h]
            correct = np.nan
            if not np.isnan(prev_close):
                act_dir = np.sign(y_t - prev_close)
                pred_dir = np.sign(y_p - prev_close)
                is_flat = (abs(y_t - prev_close) < 5) and (abs(y_p - prev_close) < 5)
                correct = 1 if (act_dir == pred_dir or is_flat) else 0
            
            results.append({
                'Model': model_name, 'ticker': ticker, 'time_idx': t_start + h,
                'horizon': f"H{h+1}", 'y_true': y_t, 'y_pred': y_p, 'correct_dir': correct
            })
            
    return pd.DataFrame(results)

# ==============================================================================
# BAGIAN 4: REPORTING LENGKAP & VISUALISASI
# ==============================================================================

def compute_group_metrics(x):
    """Fungsi helper untuk menghitung metrik pada grup dataframe"""
    y_true = x['y_true']
    y_pred = x['y_pred']
    
    if len(y_true) < 2:
        return pd.Series({'RMSE': np.nan, 'MAE': np.nan, 'MAPE': np.nan, 'R2': np.nan, 'DirAcc': np.nan})

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    r2 = r2_score(y_true, y_pred)
    da = x['correct_dir'].mean() * 100
    
    return pd.Series({
        'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'R2': r2, 'DirAcc': da
    })

def generate_comparison_reports(df_base, df_sent):
    """
    Fungsi untuk membuat CSV detil dan Plot perbandingan
    """
    print("\n[3/3] Menghasilkan Laporan & Visualisasi Detil...")
    
    # --- 1. PROSES DATA ---
    df_base['Model_Type'] = 'Baseline'
    df_sent['Model_Type'] = 'Hybrid'
    df_all = pd.concat([df_base, df_sent], ignore_index=True)

    # A. GLOBAL METRICS (Seluruh Data)
    global_metrics = df_all.groupby('Model_Type').apply(compute_group_metrics).reset_index()
    global_metrics.to_csv(REPORTS_DIR / "eval_metrics_global.csv", index=False)
    print("      📄 Saved: eval_metrics_global.csv")

    # B. HORIZON METRICS (Per H1, H2, dst)
    horizon_metrics = df_all.groupby(['Model_Type', 'horizon']).apply(compute_group_metrics).reset_index()
    horizon_metrics.to_csv(REPORTS_DIR / "eval_metrics_by_horizon.csv", index=False)
    print("      📄 Saved: eval_metrics_by_horizon.csv")

    # C. TICKER DETAILED (Ticker + Horizon)
    ticker_horizon_metrics = df_all.groupby(['Model_Type', 'ticker', 'horizon']).apply(compute_group_metrics).reset_index()
    ticker_horizon_metrics.to_csv(REPORTS_DIR / "eval_metrics_by_ticker_horizon.csv", index=False)
    print("      📄 Saved: eval_metrics_by_ticker_horizon.csv")

    # ==========================================================================
    # D. TICKER GLOBAL (Rata-rata semua horizon per saham)  <-- BAGIAN BARU
    # ==========================================================================
    ticker_global_metrics = df_all.groupby(['Model_Type', 'ticker']).apply(compute_group_metrics).reset_index()
    ticker_global_metrics.to_csv(REPORTS_DIR / "eval_metrics_by_ticker_global.csv", index=False)
    print("      📄 Saved: eval_metrics_by_ticker_global.csv (✅ Requested)")

    # --- 2. VISUALISASI ---
    metrics_to_plot = ['RMSE', 'MAE', 'MAPE', 'R2', 'DirAcc']
    
    # Plot 1: Global Comparison
    for m in metrics_to_plot:
        plt.figure(figsize=(8, 5))
        sns.barplot(data=global_metrics, x='Model_Type', y=m, palette=['gray', '#c44e52'])
        plt.title(f'Global Comparison: {m}')
        plt.ylabel(m)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"comp_global_{m}.png", dpi=300)
        plt.close()

    # Plot 2: Per Horizon Comparison
    for m in metrics_to_plot:
        plt.figure(figsize=(10, 6))
        sns.barplot(data=horizon_metrics, x='horizon', y=m, hue='Model_Type', 
                    palette={'Baseline': 'gray', 'Hybrid': '#c44e52'})
        plt.title(f'Comparison per Horizon: {m}')
        plt.ylabel(m)
        plt.legend(title='Model')
        plt.grid(axis='y', linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"comp_horizon_{m}.png", dpi=300)
        plt.close()

    # ==========================================================================
    # Plot 3: Distribusi Error per Ticker (Boxplot) <-- BAGIAN BARU
    # ==========================================================================
    # Ini untuk melihat sebaran error antar saham. Apakah model stabil di semua saham?
    for m in ['RMSE', 'DirAcc']:
        plt.figure(figsize=(12, 6))
        # Mengurutkan ticker berdasarkan median RMSE Hybrid agar rapi
        sorted_tickers = ticker_global_metrics[ticker_global_metrics['Model_Type']=='Hybrid'].sort_values(m)['ticker']
        
        # Ambil Top 20 Ticker saja biar grafik tidak penuh (opsional, kalau mau semua hapus .head(20))
        # Jika ticker < 30, tampilkan semua. Jika > 30, tampilkan 20 terbaik/terburuk.
        if len(sorted_tickers) > 30:
            top_tickers = sorted_tickers.head(10).tolist() + sorted_tickers.tail(10).tolist()
            plot_data = ticker_global_metrics[ticker_global_metrics['ticker'].isin(top_tickers)]
            title_suffix = "(Top 10 & Bottom 10)"
        else:
            plot_data = ticker_global_metrics
            title_suffix = "(All Tickers)"

        sns.barplot(data=plot_data, x='ticker', y=m, hue='Model_Type', 
                    palette={'Baseline': 'gray', 'Hybrid': '#c44e52'})
        
        plt.xticks(rotation=45, ha='right')
        plt.title(f'Comparison per Ticker: {m} {title_suffix}')
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"comp_ticker_global_{m}.png", dpi=300)
        plt.close()

    print("      📊 Saved Plots (termasuk per ticker global) ke reports/figures/")

# ==============================================================================
# MAIN PROGRAM
# ==============================================================================

def main():
    print("="*95)
    print(f"   EVALUASI MODEL SKRIPSI LENGKAP")
    print(f"   Baseline: Logs v{BASELINE_LOG_VERSION} | Ckpt: {Path(BASELINE_CKPT_EXACT).name if BASELINE_CKPT_EXACT else 'Auto'}")
    print(f"   Hybrid  : Logs v{HYBRID_LOG_VERSION} | Ckpt: Auto")
    print("="*95)
    
    # 1. GRAFIK LOSS
    print("\n[1/3] Memproses Grafik Loss...")
    hist_base = get_log_data(BASELINE_LOG_VERSION, PATH_BASELINE_LOGS)
    hist_sent = get_log_data(HYBRID_LOG_VERSION, PATH_HYBRID_LOGS)
    plot_loss_curves(hist_base, hist_sent)
    print("✅ Grafik Loss Selesai.")

    # 2. PREDIKSI
    print("\n[2/3] Memproses Prediksi...")
    ckpt_base = find_checkpoint("baseline", BASELINE_CKPT_EXACT)
    ckpt_sent = find_checkpoint("sentiment", HYBRID_CKPT_EXACT)
    
    df_base = generate_prediction_df(ckpt_base, DATA_PATH, "Baseline")
    df_sent = generate_prediction_df(ckpt_sent, DATA_PATH, "Hybrid")
    
    # Simpan Prediksi Mentah
    if df_base is not None: df_base.to_csv(REPORTS_DIR / "preds_raw_baseline.csv", index=False)
    if df_sent is not None: df_sent.to_csv(REPORTS_DIR / "preds_raw_hybrid.csv", index=False)

    # 3. REPORTING LENGKAP
    if df_base is not None and df_sent is not None:
        generate_comparison_reports(df_base, df_sent)
        
        # Tampilkan Summary Singkat di Terminal
        print("\n" + "="*60)
        print("SUMMARY SINGKAT (RMSE PER HORIZON)")
        print("-" * 60)
        
        # Hitung cepat untuk display terminal
        res_b = df_base.groupby('horizon').apply(compute_group_metrics)['RMSE']
        res_s = df_sent.groupby('horizon').apply(compute_group_metrics)['RMSE']
        
        for h in ['H1', 'H2', 'H3']: # Sesuaikan jika horizon > 3
            if h in res_b.index:
                imp = ((res_b[h] - res_s[h]) / res_b[h]) * 100
                print(f" {h} | Base: {res_b[h]:.2f} | Hybrid: {res_s[h]:.2f} | Imp: {imp:+.2f}%")
        print("="*60)
        
    print(f"\n✅ Selesai! Cek folder '{FIGURES_DIR}' dan '{REPORTS_DIR}'")

if __name__ == "__main__":
    main()