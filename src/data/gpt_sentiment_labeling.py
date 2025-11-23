import os
import time

import pandas as pd
import yaml
from tqdm import tqdm

from src.utils.gpt_client import classify_sentiment

# Lokasi root project
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_INTERIM_DIR = os.path.join(ROOT_DIR, "data", "interim")
DATA_PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")
CONFIG_GPT_PATH = os.path.join(ROOT_DIR, "configs", "gpt_sentiment.yaml")

os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)

SRC_PATH = os.path.join(DATA_INTERIM_DIR, "news_clean.csv")
OUT_PATH = os.path.join(DATA_PROCESSED_DIR, "news_with_sentiment_per_article.csv")


def load_score_mapping():
    with open(CONFIG_GPT_PATH, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg["score_mapping"]


def build_text_for_sentiment(row: pd.Series) -> str:
    """
    Bangun teks pendek yang paling relevan untuk analisis sentimen:
    1) title_clean / title
    2) + description_clean / description
    3) fallback ke text_clean kalau semua kosong
    """
    title = row.get("title_clean") or row.get("title") or ""
    desc = row.get("description_clean") or row.get("description") or ""

    # Handle NaN
    if pd.isna(title):
        title = ""
    if pd.isna(desc):
        desc = ""

    text_parts = []
    if str(title).strip():
        text_parts.append(str(title).strip())
    if str(desc).strip():
        text_parts.append(str(desc).strip())

    if text_parts:
        text = ". ".join(text_parts)
    else:
        # Fallback ke text_clean kalau ada
        text = row.get("text_clean", "")
        if pd.isna(text):
            text = ""

    return str(text)


def main():
    if not os.path.exists(SRC_PATH):
        raise FileNotFoundError(f"Tidak ditemukan: {SRC_PATH}")

    print(f"[INFO] Loading {SRC_PATH}")
    df = pd.read_csv(SRC_PATH, parse_dates=["date"])

    score_map = load_score_mapping()

    # 🔧 Cek apakah sudah ada hasil labeling sebelumnya → buat cache
    cache: dict[tuple, tuple[str | None, int | None]] = {}
    reused = 0
    new_calls = 0

    if os.path.exists(OUT_PATH):
        print(f"[INFO] Detected existing {OUT_PATH}, building cache...")
        df_old = pd.read_csv(OUT_PATH, parse_dates=["date"])

        # pakai (link, ticker) kalau ada, fallback ke (title_clean, date, ticker)
        for _, row in df_old.iterrows():
            if "link" in row and not pd.isna(row["link"]):
                key = (row["link"], row.get("ticker", ""))
            else:
                key = (
                    row.get("title_clean", "") or row.get("title", ""),
                    pd.to_datetime(row.get("date")).date()
                    if not pd.isna(row.get("date"))
                    else None,
                    row.get("ticker", ""),
                )
            if key not in cache:
                cache[key] = (row.get("gpt_label", None), row.get("gpt_score", None))

        print(f"[INFO] Cache size: {len(cache)} entries")

    labels: list[str] = []
    scores: list[int] = []

    print(f"[INFO] Melabeli {len(df)} berita dengan GPT (dengan caching)...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        # bangun key yang sama seperti saat bikin cache
        if "link" in df.columns and not pd.isna(row.get("link", None)):
            key = (row["link"], row.get("ticker", ""))
        else:
            key = (
                row.get("title_clean", "") or row.get("title", ""),
                pd.to_datetime(row.get("date")).date()
                if not pd.isna(row.get("date"))
                else None,
                row.get("ticker", ""),
            )

        if key in cache and cache[key][0] is not None:
            # pakai label lama dari cache
            label, score = cache[key]
            labels.append(label)  # type: ignore[arg-type]
            scores.append(int(score) if score is not None else 0)
            reused += 1
            continue

        # berita baru → panggil GPT
        text = build_text_for_sentiment(row)
        label = classify_sentiment(text)
        score = score_map.get(label, 0)

        labels.append(label)
        scores.append(score)

        cache[key] = (label, score)
        new_calls += 1

        # jeda kecil biar aman dari rate limit (opsional)
        time.sleep(0.1)

    df["gpt_label"] = labels
    df["gpt_score"] = scores

    print(f"[INFO] Reused from cache: {reused}, new GPT calls: {new_calls}")
    print(f"[INFO] Saving to {OUT_PATH}")
    df.to_csv(OUT_PATH, index=False)
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
