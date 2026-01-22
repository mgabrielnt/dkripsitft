"""
fetch_news_rss_google.py

Fitur Utama:
- SMART RESUME: Otomatis mendeteksi tanggal terakhir di CSV dan hanya mengambil data baru.
- HISTORICAL SCRAPING: Mengambil berita Google News per bulan.
- DIRECT RSS: Mengambil feed RSS umum.
- Rate Limit Protection: Ada jeda waktu (sleep).
"""

import os
import time
import random
import urllib.parse
from datetime import datetime, timedelta
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
OUT_PATH = os.path.join(DATA_RAW_NEWS_DIR, "news_raw_all_sources.csv")


# ---------------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------------

def load_config(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config tidak ditemukan: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def get_last_date_from_csv(csv_path: str, ticker: str) -> Optional[datetime.date]:
    """
    Cek tanggal terakhir untuk ticker tertentu di file CSV yang sudah ada.
    """
    if not os.path.exists(csv_path):
        return None
    
    try:
        # Baca hanya kolom ticker dan date untuk efisiensi
        df = pd.read_csv(csv_path, usecols=["ticker", "date"])
        df = df[df["ticker"] == ticker]
        
        if df.empty:
            return None
            
        df["date"] = pd.to_datetime(df["date"])
        last_date = df["date"].max().date()
        return last_date
    except Exception as e:
        print(f"[WARN] Gagal membaca last date dari CSV: {e}")
        return None

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
    start_date: datetime,
    max_articles_per_query: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Ambil arsip Google News bulanan.
    """
    end_dt = datetime.now()
    
    # Jika start date sudah lewat hari ini (misal baru run pagi tadi), skip
    if start_date.date() >= end_dt.date():
        print(f"[SKIP] Data {ticker} sudah up-to-date ({start_date.date()}).")
        return []

    print(f"\n[START] Fetching Google History untuk {ticker}")
    print(f"      Query : {query}")
    print(f"      Range : {start_date.date()} s.d. {end_dt.date()}")

    # Rentang waktu per bulan (Monthly Start)
    # Gunakan period 'M' agar mencakup sisa bulan ini
    date_ranges = pd.date_range(start=start_date, end=end_dt, freq="MS")
    
    # Jika range kosong (misal baru beda beberapa hari), paksa loop sekali
    if len(date_ranges) == 0:
        date_ranges = pd.DatetimeIndex([start_date])

    all_records: List[Dict[str, Any]] = []

    # Loop logic yang aman
    current_cursor = start_date
    
    while current_cursor < end_dt:
        # Set next cursor ke awal bulan depan, atau hari ini jika sudah lewat
        next_month = (current_cursor.replace(day=1) + timedelta(days=32)).replace(day=1)
        next_cursor = min(next_month, end_dt)
        
        curr_start_str = current_cursor.strftime("%Y-%m-%d")
        curr_end_str = next_cursor.strftime("%Y-%m-%d")

        # Tambahkan filter waktu ke query:
        period_query = f"{query} after:{curr_start_str} before:{curr_end_str}"
        url = build_google_news_rss_url(period_query, language=language)

        try:
            feed = feedparser.parse(url)
            entries = feed.entries

            if entries:
                print(f"    -> [{curr_start_str} s/d {curr_end_str}] : {len(entries)} artikel.")
                for entry in entries:
                    date_obj = parse_published(entry)
                    if not date_obj: continue

                    link = entry.get("link", "")
                    if any(r["link"] == link for r in all_records): continue

                    all_records.append({
                        "date": date_obj,
                        "ticker": ticker,
                        "query": query,
                        "query_type": "google_history",
                        "language": language,
                        "title": entry.get("title", ""),
                        "description": entry.get("summary", ""),
                        "link": link,
                        "source": entry.get("source", {}).get("title", "GoogleNews"),
                        "published_raw": str(date_obj),
                    })

                    if max_articles_per_query and len(all_records) >= max_articles_per_query:
                        break
            else:
                pass # Silent if empty

        except Exception as e:
            print(f"    [ERROR] Periode {curr_start_str} : {e}")

        if max_articles_per_query and len(all_records) >= max_articles_per_query:
            print(f"    [INFO] Mencapai limit {max_articles_per_query} artikel.")
            break

        # Move cursor
        current_cursor = next_cursor
        
        # Sleep acak
        time.sleep(random.uniform(2.0, 4.0))

    print(f"[DONE] Total terkumpul untuk {ticker}: {len(all_records)} artikel.\n")
    return all_records


# ---------------------------------------------------------------------------
# CORE LOGIC: DIRECT RSS
# ---------------------------------------------------------------------------

def fetch_rss_direct(rss_url: str, ticker: str, source_name: str) -> List[Dict[str, Any]]:
    ticker = (ticker or "UNKNOWN").upper()
    print(f"[INFO] Fetching RSS Direct: {source_name} (ticker={ticker})")
    try:
        feed = feedparser.parse(rss_url)
        records: List[Dict[str, Any]] = []
        for entry in feed.entries:
            date_obj = parse_published(entry)
            if not date_obj: continue
            records.append({
                "date": date_obj,
                "ticker": ticker,
                "query": rss_url,
                "query_type": "rss_direct",
                "language": "id",
                "title": entry.get("title", ""),
                "description": entry.get("summary", ""),
                "link": entry.get("link", ""),
                "source": source_name,
                "published_raw": str(date_obj),
            })
        print(f"      -> Dapat {len(records)} artikel.")
        return records
    except Exception as e:
        print(f"      [ERROR] RSS Direct gagal: {e}")
        return []


# ---------------------------------------------------------------------------
# MAIN PROGRAM
# ---------------------------------------------------------------------------

def main():
    config = load_config(CONFIG_RSS_PATH)
    
    # Config Defaults
    default_start_str = config.get("start_date_history", "2020-01-01")
    default_language = config.get("default_language", "id")
    max_articles = config.get("max_articles_per_query", None)
    if isinstance(max_articles, str) and not max_articles.strip(): max_articles = None

    queries = config.get("queries", [])
    final_data: List[Dict[str, Any]] = []

    print("===================================================")
    print("      SMART NEWS SCRAPING (AUTO-RESUME)")
    print("===================================================")

    for q in queries:
        q_type = q.get("type")
        ticker = (q.get("ticker") or "UNKNOWN").upper()

        if q_type == "google":
            query_text = q.get("query")
            lang = q.get("language", default_language)
            
            # --- SMART RESUME LOGIC ---
            last_recorded_date = get_last_date_from_csv(OUT_PATH, ticker)
            
            if last_recorded_date:
                # Lanjut H+1 dari data terakhir
                start_fetch_dt = datetime.combine(last_recorded_date, datetime.min.time()) + timedelta(days=1)
                print(f"Detected existing data for {ticker}. Resuming from {start_fetch_dt.date()}...")
            else:
                # Mulai dari awal (Config)
                try:
                    start_fetch_dt = datetime.strptime(str(default_start_str), "%Y-%m-%d")
                except:
                    start_fetch_dt = datetime(2020, 1, 1)
                print(f"No existing data for {ticker}. Starting fresh from {start_fetch_dt.date()}...")
            
            # Fetch
            recs = fetch_google_history(
                query=query_text,
                ticker=ticker,
                language=lang,
                start_date=start_fetch_dt,
                max_articles_per_query=max_articles,
            )
            final_data.extend(recs)

        elif q_type == "rss":
            # RSS Direct selalu fetch karena isinya real-time/latest
            url = q.get("rss_url")
            src = q.get("source_name", "RSS")
            recs = fetch_rss_direct(url, ticker, src)
            final_data.extend(recs)

    # --- SAVE PROCESS ---
    if not final_data:
        print("\n[INFO] Tidak ada berita BARU yang ditemukan (Data sudah up-to-date).")
        return

    df_new = pd.DataFrame(final_data)
    df_new["date"] = pd.to_datetime(df_new["date"]).dt.date

    if os.path.exists(OUT_PATH):
        print(f"\n[INFO] Menambahkan {len(df_new)} berita baru ke database lama...")
        df_old = pd.read_csv(OUT_PATH)
        df_old["date"] = pd.to_datetime(df_old["date"]).dt.date
        df_combined = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_combined = df_new

    # Deduplikasi Final (Link + Ticker)
    before_dedupe = len(df_combined)
    df_combined = df_combined.drop_duplicates(subset=["link", "ticker"], keep="last")
    after_dedupe = len(df_combined)

    # Sorting
    df_combined = df_combined.sort_values(by=["ticker", "date"]).reset_index(drop=True)

    print(f"\n[INFO] Statistik Data:")
    print(f"      Total baris (sebelum dedupe) : {before_dedupe}")
    print(f"      Total baris (setelah dedupe) : {after_dedupe}")
    print(f"      Data baru yang tersimpan   : {after_dedupe - (len(df_old) if 'df_old' in locals() else 0)}")

    df_combined.to_csv(OUT_PATH, index=False)
    print(f"[SUCCESS] Database diperbarui: {OUT_PATH}")

if __name__ == "__main__":
    main()