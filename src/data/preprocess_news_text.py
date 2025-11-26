import os
import re

import pandas as pd

# Lokasi root project
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_RAW_NEWS_DIR = os.path.join(ROOT_DIR, "data", "raw", "news")
DATA_INTERIM_DIR = os.path.join(ROOT_DIR, "data", "interim")

os.makedirs(DATA_INTERIM_DIR, exist_ok=True)

NEWS_RAW_ALL_PATH = os.path.join(DATA_RAW_NEWS_DIR, "news_raw_all_sources.csv")
NEWS_RAW_GOOGLE_PATH = os.path.join(DATA_RAW_NEWS_DIR, "news_raw_google_rss.csv")

OUT_PATH = os.path.join(DATA_INTERIM_DIR, "news_clean.csv")


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    # hapus URL
    text = re.sub(r"http\S+", " ", text)
    # hapus HTML tags
    text = re.sub(r"<.*?>", " ", text)
    # ganti newline
    text = text.replace("\n", " ").replace("\r", " ")
    # kompres spasi
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def main():
    # pilih sumber: kalau ada all_sources → pakai itu, kalau tidak fallback ke Google
    if os.path.exists(NEWS_RAW_ALL_PATH):
        src_path = NEWS_RAW_ALL_PATH
    elif os.path.exists(NEWS_RAW_GOOGLE_PATH):
        src_path = NEWS_RAW_GOOGLE_PATH
    else:
        raise FileNotFoundError(
            f"Tidak ditemukan file berita mentah: "
            f"{NEWS_RAW_ALL_PATH} atau {NEWS_RAW_GOOGLE_PATH}"
        )

    print(f"[INFO] Loading raw news from: {src_path}")
    df = pd.read_csv(src_path, parse_dates=["date"])

    # pastikan kolom minimal ada
    for col in ["title", "description", "ticker", "date"]:
        if col not in df.columns:
            df[col] = ""

    df["title"] = df["title"].fillna("").astype(str)
    df["description"] = df["description"].fillna("").astype(str)

    # gabung title + description sebagai teks mentah
    df["text_raw"] = (df["title"] + ". " + df["description"]).str.strip()

    df["title_clean"] = df["title"].apply(clean_text).str.lower()
    df["text_clean"] = df["text_raw"].apply(clean_text).str.lower()

    before = len(df)
    df = df[df["text_clean"].str.strip() != ""].copy()
    after = len(df)
    print(f"[INFO] Drop berita dengan teks kosong: {before} -> {after}")

    # deduplikasi berbasis link dan konten title_clean per ticker + tanggal
    before = len(df)
    subset_cols = [c for c in ["link", "ticker", "date", "title_clean"] if c in df.columns]
    if subset_cols:
        df = df.drop_duplicates(subset=subset_cols)
    df = df.drop_duplicates()
    print(f"[INFO] Drop duplikat (link/title) : {before} -> {len(df)}")

    # pilih kolom penting untuk labeling
    keep_cols = [
        "date",
        "ticker",
        "title",
        "description",
        "link",
        "source",
        "title_clean",
        "text_clean",
    ]
    existing_cols = [c for c in keep_cols if c in df.columns]

    df = df[existing_cols].sort_values(["ticker", "date"])

    print(f"[INFO] Saving cleaned news to: {OUT_PATH}")
    df.to_csv(OUT_PATH, index=False)
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
