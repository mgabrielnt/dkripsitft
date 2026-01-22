import os
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from pathlib import Path

# ==============================================================================
# SETUP PATH
# ==============================================================================
# Asumsi: Script dijalankan dari root folder project (D:\skripsi\tft)
PROJECT_ROOT = Path.cwd()

# DAFTAR FOLDER YANG AKAN DI-SCAN
TARGET_DIRS = [
    PROJECT_ROOT / "lightning_logs",       # Lokasi default (Hybrid/Sentiment biasanya di sini)
    PROJECT_ROOT / "logs" / "baseline_model" # Lokasi custom (Baseline ada di sini)
]

def extract_scalars(log_file):
    print(f"      Reading: {log_file.name}...")
    
    try:
        event_acc = EventAccumulator(str(log_file))
        event_acc.Reload()
        tags = event_acc.Tags()['scalars']
    except Exception as e:
        print(f"      ⚠️ File rusak/tidak bisa dibaca: {e}")
        return pd.DataFrame()

    # --- 1. EXTRACT DATA ---
    
    # A. Ambil Epoch
    df_epoch = pd.DataFrame()
    if 'epoch' in tags:
        events = event_acc.Scalars('epoch')
        df_epoch = pd.DataFrame([{'step': e.step, 'epoch': e.value} for e in events])

    # B. Ambil Validation Loss
    val_tag = next((t for t in ['val_loss', 'val_loss_epoch', 'val_loss_step'] if t in tags), None)
    df_val = pd.DataFrame()
    if val_tag:
        events = event_acc.Scalars(val_tag)
        df_val = pd.DataFrame([{'step': e.step, 'val_loss': e.value} for e in events])

    # C. Ambil Training Loss
    train_tag = next((t for t in ['train_loss_epoch', 'train_loss', 'train_loss_step'] if t in tags), None)
    df_train = pd.DataFrame()
    if train_tag:
        events = event_acc.Scalars(train_tag)
        df_train = pd.DataFrame([{'step': e.step, 'train_loss': e.value} for e in events])

    # --- 2. PENGGABUNGAN DATA (ROBUST MERGE) ---
    
    if df_val.empty:
        return pd.DataFrame() 

    # Merge Val Loss + Epoch
    df_final = df_val.copy()
    if not df_epoch.empty:
        df_final = pd.merge(df_final, df_epoch, on='step', how='left')
    else:
        df_final['epoch'] = range(len(df_final))

    # Merge dengan Train Loss (asof merge untuk mencocokkan step terdekat)
    if not df_train.empty:
        df_merged = pd.merge_asof(
            df_final.sort_values('step'), 
            df_train.sort_values('step'), 
            on='step', 
            direction='backward'
        )
        df_final = df_merged

    return df_final

def process_directory(log_dir):
    """Fungsi helper untuk memproses satu direktori induk"""
    print(f"\n{'='*60}")
    print(f"🔍 SCANNING DIRECTORY: {log_dir}")
    print(f"{'='*60}")

    if not log_dir.exists():
        print(f"❌ Folder tidak ditemukan: {log_dir}")
        return

    # Cari subfolder (version_X)
    versions = sorted([d for d in log_dir.iterdir() if d.is_dir() and "version_" in d.name],
                      key=lambda x: int(x.name.split('_')[-1]) if x.name.split('_')[-1].isdigit() else 0)
    
    if not versions:
        print("   (Info) Tidak ada folder version_XXX di sini.")
        return

    for version_dir in versions:
        print(f"\n📂 Processing {version_dir.name}...")
        
        event_files = list(version_dir.glob("events.out.tfevents*"))
        if not event_files:
            print("   (Skip) Tidak ada file event.")
            continue
            
        latest_event = max(event_files, key=os.path.getmtime)
        
        # Filter file kecil (< 5KB biasanya corrupt/kosong)
        if latest_event.stat().st_size < 5000:
            print("   (Skip) File terlalu kecil (<5KB).")
            continue

        try:
            df = extract_scalars(latest_event)
            
            if not df.empty and 'val_loss' in df.columns:
                save_path = version_dir / "metrics.csv"
                df.to_csv(save_path, index=False)
                print(f"   ✅ Sukses! Disimpan ke: {save_path}")
                print(f"      Stats: {len(df)} epochs, Min Loss: {df['val_loss'].min():.4f}")
            else:
                print("   ⚠️  Data kosong atau val_loss tidak ditemukan.")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")

def main():
    # Loop melalui kedua folder target
    for target_dir in TARGET_DIRS:
        process_directory(target_dir)

if __name__ == "__main__":
    main()