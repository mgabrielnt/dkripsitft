"""
update_experiments_best_ckpt.py

Cari checkpoint terbaik (val_loss paling kecil) untuk:
- baseline (models/tft_baseline)
- hybrid  (models/tft_with_sentiment)

Lalu update D:/skripsi/tft/configs/experiments.yaml:

tft_baseline:
  checkpoint_paths:
    - "D:/skripsi/tft/models/tft_baseline/....ckpt"

tft_with_sentiment:
  checkpoint_paths:
    - "D:/skripsi/tft/models/tft_with_sentiment/....ckpt"
"""

import os
import re
from typing import Optional, Dict, Any

import yaml

# Lokasi root project
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

MODELS_DIR = os.path.join(ROOT_DIR, "models")
BASELINE_DIR = os.path.join(MODELS_DIR, "tft_baseline")
HYBRID_DIR = os.path.join(MODELS_DIR, "tft_with_sentiment")

EXPERIMENTS_PATH = os.path.join(ROOT_DIR, "configs", "experiments.yaml")


def _safe_parse_float_from_filename(pattern: str, fname: str) -> Optional[float]:
    """
    Cari angka float dari nama file berdasarkan regex pattern.
    Kalau gagal parse, return None dan tulis warning.
    """
    m = re.search(pattern, fname)
    if not m:
        return None

    raw = m.group(1)
    # Buang titik di ujung kalau ada (kasus: '414.8535.' dari '.ckpt')
    raw_clean = raw.rstrip(".")

    try:
        return float(raw_clean)
    except ValueError as e:
        print(f"[WARN] Gagal parse float '{raw}' dari filename '{fname}': {e}")
        return None


def _safe_parse_int_from_filename(pattern: str, fname: str) -> Optional[int]:
    """
    Cari angka int dari nama file berdasarkan regex pattern.
    Kalau gagal parse, return None.
    """
    m = re.search(pattern, fname)
    if not m:
        return None

    try:
        return int(m.group(1))
    except ValueError:
        return None


def find_best_ckpt(run_dir: str) -> Optional[str]:
    """
    Cari file .ckpt di folder run_dir dan pilih yang:
    - kalau ada 'val_loss=xxx' di namanya -> ambil yang val_loss-nya MIN
    - kalau tidak ada satupun yang punya val_loss -> ambil yang epoch/mtime paling besar
    """
    if not os.path.isdir(run_dir):
        print(f"[WARN] Folder ckpt tidak ada: {run_dir}")
        return None

    candidates = []
    for fname in os.listdir(run_dir):
        if not fname.endswith(".ckpt"):
            continue

        path = os.path.join(run_dir, fname)

        # Parse val_loss dan epoch dengan aman
        loss = _safe_parse_float_from_filename(r"val_loss=([0-9.]+)", fname)
        epoch = _safe_parse_int_from_filename(r"epoch=(\d+)", fname)
        mtime = os.path.getmtime(path)

        candidates.append(
            {
                "path": path,
                "loss": loss,
                "epoch": epoch,
                "mtime": mtime,
            }
        )

    if not candidates:
        print(f"[WARN] Tidak ada file .ckpt di {run_dir}")
        return None

    with_loss = [c for c in candidates if c["loss"] is not None]
    if with_loss:
        best = min(with_loss, key=lambda c: c["loss"])
        print(
            f"[INFO] Best ckpt di {run_dir}: "
            f"{os.path.basename(best['path'])} (val_loss={best['loss']})"
        )
    else:
        # fallback: pilih yang epoch paling besar, kalau epoch None pakai mtime
        best = max(
            candidates,
            key=lambda c: ((c["epoch"] or 0), c["mtime"]),
        )
        print(
            f"[INFO] Best ckpt di {run_dir}: "
            f"{os.path.basename(best['path'])} (tanpa val_loss, pakai epoch/mtime)"
        )

    return best["path"]


def load_experiments_config() -> Dict[str, Any]:
    if not os.path.exists(EXPERIMENTS_PATH):
        print(f"[WARN] experiments.yaml belum ada, akan dibuat baru: {EXPERIMENTS_PATH}")
        return {}
    with open(EXPERIMENTS_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg


def save_experiments_config(cfg: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(EXPERIMENTS_PATH), exist_ok=True)
    with open(EXPERIMENTS_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    print(f"[INFO] experiments.yaml berhasil diupdate: {EXPERIMENTS_PATH}")


def main() -> None:
    print(f"[INFO] ROOT_DIR = {ROOT_DIR}")
    print("[INFO] Mencari ckpt terbaik di:")
    print(f"       BASELINE: {BASELINE_DIR}")
    print(f"       HYBRID  : {HYBRID_DIR}")

    baseline_ckpt = find_best_ckpt(BASELINE_DIR)
    hybrid_ckpt = find_best_ckpt(HYBRID_DIR)

    if not baseline_ckpt and not hybrid_ckpt:
        print("[ERROR] Tidak ada ckpt yang bisa dipakai. experiments.yaml tidak diubah.")
        return

    cfg = load_experiments_config()

    # Pastikan key-nya ada
    if "tft_baseline" not in cfg:
        cfg["tft_baseline"] = {}
    if "tft_with_sentiment" not in cfg:
        cfg["tft_with_sentiment"] = {}

    if baseline_ckpt:
        path_yaml = baseline_ckpt.replace("\\", "/")
        cfg["tft_baseline"]["checkpoint_paths"] = [path_yaml]
        print(f"[INFO] Set baseline checkpoint_paths -> {path_yaml}")

    if hybrid_ckpt:
        path_yaml = hybrid_ckpt.replace("\\", "/")
        cfg["tft_with_sentiment"]["checkpoint_paths"] = [path_yaml]
        print(f"[INFO] Set hybrid   checkpoint_paths -> {path_yaml}")

    save_experiments_config(cfg)


if __name__ == "__main__":
    main()
