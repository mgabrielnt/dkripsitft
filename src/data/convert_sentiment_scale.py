"""Konversi skala sentimen 5-level ke 3-level {-1, 0, +1}.

- Input utama: data/processed/news_with_sentiment_per_article.csv
  berisi kolom `gpt_score` numerik atau label teks.
- Output: data/interim/news_with_sentiment_3class.csv dengan kolom tambahan:
  - sentiment_label_3 (-1/0/+1)
  - pos_count_3, neg_count_3, neu_count_3 (hitungan biner per artikel)
- File output tidak menimpa data mentah, supaya pipeline tetap reproducible.

CLI:
    python -m src.data.convert_sentiment_scale
"""
import os
from typing import Optional

import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")
DATA_INTERIM_DIR = os.path.join(ROOT_DIR, "data", "interim")

SRC_PATH = os.path.join(DATA_PROCESSED_DIR, "news_with_sentiment_per_article.csv")
OUT_PATH = os.path.join(DATA_INTERIM_DIR, "news_with_sentiment_3class.csv")

os.makedirs(DATA_INTERIM_DIR, exist_ok=True)


def map_score_to_label(score: Optional[float]) -> int:
    """Peta gpt_score 5-level menjadi {-1, 0, +1}.

    Aturan:
    - Nilai numerik: sign(score)
    - Label teks: very negative/negative -> -1, neutral -> 0,
      positive/very positive -> +1
    """

    if pd.isna(score):
        return 0

    # Jika sudah string label
    if isinstance(score, str):
        s = score.strip().lower()
        if s in {"very negative", "negative"}:
            return -1
        if s in {"neutral"}:
            return 0
        if s in {"positive", "very positive"}:
            return 1
        # fallback: coba parse float
        try:
            score = float(score)
        except ValueError:
            return 0

    # Numerik: sign
    if score > 0:
        return 1
    if score < 0:
        return -1
    return 0


def main():
    if not os.path.exists(SRC_PATH):
        raise FileNotFoundError(f"Tidak ditemukan: {SRC_PATH}")

    print(f"[INFO] Loading {SRC_PATH}")
    df = pd.read_csv(SRC_PATH, parse_dates=["date"])

    if "gpt_score" not in df.columns:
        raise ValueError("Kolom 'gpt_score' wajib ada di news_with_sentiment_per_article.csv")

    print("[INFO] Mengonversi gpt_score ke label 3-class (-1/0/+1)")
    df["sentiment_label_3"] = df["gpt_score"].apply(map_score_to_label)

    # Hitungan biner per artikel untuk agregasi harian
    df["pos_count_3"] = (df["sentiment_label_3"] > 0).astype(int)
    df["neg_count_3"] = (df["sentiment_label_3"] < 0).astype(int)
    df["neu_count_3"] = (df["sentiment_label_3"] == 0).astype(int)

    # Simpan ke interim (tidak menimpa data mentah)
    print(f"[INFO] Saving 3-class sentiment to {OUT_PATH}")
    df.to_csv(OUT_PATH, index=False)
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
