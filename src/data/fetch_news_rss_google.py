"""
fetch_news_rss_google.py

Fitur Utama:
- HISTORICAL SCRAPING: Mengambil berita Google News per bulan mulai dari start_date_history di configs/rss.yaml.
- DIRECT RSS: Mengambil feed RSS umum (market) → sekarang diberi ticker khusus "MARKET".
- Rate Limit Protection: Ada jeda waktu (sleep) agar tidak diblokir Google.
"""

import os
import time
import random
import urllib.parse
from datetime import datetime
from typing import Any, Dict, List, Optional

import feedparser
import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# PATH SETUP
# ---------------------------------------------------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_RAW_NEWS_DIR = os.path.join(ROOT_DIR, "data", "raw", "news")
os.makedirs(DATA_RAW_NEWS_DIR, exist_ok=True)

CONFIG_RSS_PATH = os.path.join(ROOT_DIR, "configs", "rss.yaml")
OUT_PATH = os.path.join(DATA_RAW_NEWS_DIR, "news_raw_google_rss.csv")


# ---------------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------------

def load_config(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config tidak ditemukan: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_published(entry: feedparser.FeedParserDict) -> Optional[datetime.date]:
    """Mencoba parsing tanggal dari format RSS yang berantakan."""
    # Coba field parsed bawaan feedparser
    parsed = entry.get("published_parsed") or entry.get("updated_parsed")
    if parsed:
        try:
            return datetime(*parsed[:6]).date()
        except Exception:
            pass

    # Coba parsing string manual
    pub_str = entry.get("published") or entry.get("pubDate")
    if pub_str:
        from email.utils import parsedate_to_datetime

        try:
            return parsedate_to_datetime(pub_str).date()
        except Exception:
            return None
    return None


def build_google_news_rss_url(query: str, language: str = "id") -> str:
    base_url = "https://news.google.com/rss/search"
    lang_lower = (language or "id").lower()

    if lang_lower == "id":
        hl, gl, ceid = "id-ID", "ID", "ID:id"
    elif lang_lower == "en":
        hl, gl, ceid = "en-US", "US", "US:en"
    else:
        hl, gl, ceid = f"{lang_lower}-ID", "ID", f"ID:{lang_lower}"

    params = {"q": query, "hl": hl, "gl": gl, "ceid": ceid}
    return base_url + "?" + urllib.parse.urlencode(params)


# ---------------------------------------------------------------------------
# CORE LOGIC: ITERATIVE FETCHING (HISTORICAL GOOGLE NEWS)
# ---------------------------------------------------------------------------

def fetch_google_history(
    query: str,
    ticker: str,
    language: str,
    start_date_str: str = "2020-01-01",
    max_articles_per_query: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Ambil arsip Google News bulanan untuk 1 kombinasi (ticker, query).
    Jika max_articles_per_query tidak None, hentikan setelah mencapai batas tsb.
    """
    ticker = (ticker or "UNKNOWN").upper()

    # Konversi string tanggal ke object date
    try:
        start_dt = datetime.strptime(str(start_date_str), "%Y-%m-%d")
    except ValueError:
        start_dt = datetime(2020, 1, 1)  # Default fallback

    end_dt = datetime.now()

    print(f"\n[START] Fetching Google History untuk {ticker}")
    print(f"       Query : {query}")
    print(f"       Range : {start_dt.date()} s.d. {end_dt.date()}")

    # Rentang waktu per bulan (Monthly Start)
    date_ranges = pd.date_range(start=start_dt, end=end_dt, freq="MS")

    all_records: List[Dict[str, Any]] = []

    # Loop setiap bulan
    for i in range(len(date_ranges) - 1):
        curr_start = date_ranges[i].date()
        curr_end = date_ranges[i + 1].date()

        # Tambahkan filter waktu ke query:
        #   after:YYYY-MM-DD before:YYYY-MM-DD
        period_query = f"{query} after:{curr_start} before:{curr_end}"

        url = build_google_news_rss_url(period_query, language=language)

        try:
            feed = feedparser.parse(url)
            entries = feed.entries

            if entries:
                print(f"    -> [{curr_start} s/d {curr_end}] : {len(entries)} artikel.")

                for entry in entries:
                    date_obj = parse_published(entry)
                    if not date_obj:
                        continue

                    link = entry.get("link", "")

                    # Cegah duplikat dalam sesi ini (berdasarkan link saja)
                    if any(r["link"] == link for r in all_records):
                        continue

                    all_records.append(
                        {
                            "date": date_obj,
                            "ticker": ticker,
                            "query": query,  # Simpan query asli (tanpa after/before)
                            "query_type": "google_history",
                            "language": language,
                            "title": entry.get("title", ""),
                            "description": entry.get("summary", ""),
                            "link": link,
                            "source": entry.get("source", {}).get("title", "GoogleNews"),
                            "published_raw": str(date_obj),
                        }
                    )

                    # Batas maksimal artikel per (ticker, query)
                    if max_articles_per_query is not None and len(all_records) >= max_articles_per_query:
                        print(
                            f"    [INFO] Mencapai max_articles_per_query={max_articles_per_query}, stop lebih awal."
                        )
                        break
            # else: boleh di-silent, supaya log tidak terlalu rame

        except Exception as e:
            print(f"    [ERROR] Periode {curr_start} : {e}")

        # Stop juga di level bulan jika sudah penuh
        if max_articles_per_query is not None and len(all_records) >= max_articles_per_query:
            break

        # PENTING: Sleep acak 2-4 detik agar tidak kena blokir (HTTP 429)
        sleep_sec = random.uniform(2.0, 4.0)
        time.sleep(sleep_sec)

    print(f"[DONE] Total terkumpul untuk {ticker}: {len(all_records)} artikel.\n")
    return all_records


# ---------------------------------------------------------------------------
# CORE LOGIC: DIRECT RSS (MARKET-WIDE)
# ---------------------------------------------------------------------------

def fetch_rss_direct(rss_url: str, ticker: str, source_name: str) -> List[Dict[str, Any]]:
    """
    Ambil RSS umum (misal: Kontan keuangan, CNBC market).
    Sekarang biasanya diberi ticker "MARKET" di configs/rss.yaml.
    """
    ticker = (ticker or "UNKNOWN").upper()
    print(f"[INFO] Fetching RSS Direct: {source_name} (ticker={ticker})")

    try:
        feed = feedparser.parse(rss_url)
        records: List[Dict[str, Any]] = []

        for entry in feed.entries:
            date_obj = parse_published(entry)
            if not date_obj:
                continue

            records.append(
                {
                    "date": date_obj,
                    "ticker": ticker,
                    "query": rss_url,
                    "query_type": "rss_direct",
                    "language": "id",  # Asumsi mayoritas ID
                    "title": entry.get("title", ""),
                    "description": entry.get("summary", ""),
                    "link": entry.get("link", ""),
                    "source": source_name,
                    "published_raw": str(date_obj),
                }
            )

        print(f"       -> Dapat {len(records)} artikel.")
        return records

    except Exception as e:
        print(f"       [ERROR] RSS Direct gagal: {e}")
        return []


# ---------------------------------------------------------------------------
# MAIN PROGRAM
# ---------------------------------------------------------------------------

def main():
    config = load_config(CONFIG_RSS_PATH)

    # Ambil konfigurasi tanggal mulai & batas artikel
    start_date_history = config.get("start_date_history", "2020-01-01")
    default_language = config.get("default_language", "id")
    max_articles_per_query = config.get("max_articles_per_query", None)
    if isinstance(max_articles_per_query, str) and max_articles_per_query.strip() == "":
        max_articles_per_query = None

    queries = config.get("queries", [])

    final_data: List[Dict[str, Any]] = []

    print("===================================================")
    print(f" MULAI SCRAPING DATA BERITA DARI {start_date_history}")
    print("===================================================")
    print(" NOTE: Proses ini akan memakan waktu karena ada jeda")
    print("       (sleep) antar bulan agar tidak diblokir Google.")
    print("===================================================\n")

    for q in queries:
        q_type = q.get("type")
        ticker = q.get("ticker", "UNKNOWN")
        ticker = (ticker or "UNKNOWN").upper()

        if q_type == "google":
            query_text = q.get("query")
            lang = q.get("language", default_language)

            recs = fetch_google_history(
                query=query_text,
                ticker=ticker,
                language=lang,
                start_date_str=start_date_history,
                max_articles_per_query=max_articles_per_query,
            )
            final_data.extend(recs)

        elif q_type == "rss":
            url = q.get("rss_url")
            src = q.get("source_name", "RSS")
            recs = fetch_rss_direct(url, ticker, src)
            final_data.extend(recs)

    # --- SAVE PROCESS ---
    if not final_data:
        print("\n[WARN] Tidak ada data yang berhasil diambil.")
        return

    df_new = pd.DataFrame(final_data)

    # Bersihkan format tanggal
    df_new["date"] = pd.to_datetime(df_new["date"]).dt.date

    # Cek apakah file lama ada (untuk append/merge)
    if os.path.exists(OUT_PATH):
        print(f"\n[INFO] Menggabungkan dengan database lama: {OUT_PATH}")
        df_old = pd.read_csv(OUT_PATH)
        df_old["date"] = pd.to_datetime(df_old["date"]).dt.date

        # Gabung
        df_combined = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_combined = df_new

    # Hapus duplikat (link + ticker), supaya per (ticker, berita) unik
    before_dedupe = len(df_combined)
    df_combined = df_combined.drop_duplicates(subset=["link", "ticker"], keep="last")
    after_dedupe = len(df_combined)

    # Sortir biar rapi
    df_combined = df_combined.sort_values(by=["ticker", "date"]).reset_index(drop=True)

    print(f"\n[INFO] Statistik Data:")
    print(f"       Total baris (sebelum dedupe) : {before_dedupe}")
    print(f"       Total baris (setelah dedupe) : {after_dedupe}")

    df_combined.to_csv(OUT_PATH, index=False)
    print(f"[SUCCESS] Data tersimpan di: {OUT_PATH}")


if __name__ == "__main__":
    main()
