import os

import pandas as pd

# Lokasi root project
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_RAW_NEWS_DIR = os.path.join(ROOT_DIR, "data", "raw", "news")

GOOGLE_PATH = os.path.join(DATA_RAW_NEWS_DIR, "news_raw_google_rss.csv")
YAHOO_PATH = os.path.join(DATA_RAW_NEWS_DIR, "news_raw_yahoo.csv")
OUT_PATH = os.path.join(DATA_RAW_NEWS_DIR, "news_raw_all_sources.csv")


def main():
    dfs = []

    if os.path.exists(GOOGLE_PATH):
        print(f"[INFO] Load Google RSS news dari: {GOOGLE_PATH}")
        df_g = pd.read_csv(GOOGLE_PATH, parse_dates=["date"])
        df_g["source_type"] = "google_news_rss"
        dfs.append(df_g)
    else:
        print(f"[WARN] File Google RSS tidak ditemukan: {GOOGLE_PATH}")

    if os.path.exists(YAHOO_PATH):
        print(f"[INFO] Load Yahoo Finance news dari: {YAHOO_PATH}")
        df_y = pd.read_csv(YAHOO_PATH, parse_dates=["date"])
        df_y["source_type"] = "yahoo_finance"
        dfs.append(df_y)
    else:
        print(f"[WARN] File Yahoo news tidak ditemukan: {YAHOO_PATH}")

    if not dfs:
        print("[ERROR] Tidak ada sumber berita yang ditemukan. Tidak ada file yang digabung.")
        return

    df_all = pd.concat(dfs, ignore_index=True)

    # buang duplikat berdasarkan link kalau ada
    if "link" in df_all.columns:
        before = len(df_all)
        df_all = df_all.drop_duplicates(subset=["link"])
        after = len(df_all)
        print(f"[INFO] Drop duplikat berdasarkan link: {before} -> {after}")
    else:
        df_all = df_all.drop_duplicates()

    df_all = df_all.sort_values(["ticker", "date"])

    print(f"[INFO] Total berita setelah merge: {len(df_all)}")
    print(f"[INFO] Menyimpan ke: {OUT_PATH}")
    df_all.to_csv(OUT_PATH, index=False)
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
