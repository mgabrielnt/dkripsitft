"""
fetch_news_rss_google.py

Mengambil berita dari:
- Google News RSS (type: "google")
- RSS langsung (Yahoo Finance, Kontan, Kompas, dll.) (type: "rss")

Config dibaca dari: configs/rss.yaml
Struktur minimal config:

default_language: "id"
lookback_years: 5
max_articles_per_query: 100

queries:
  - type: "google"
    ticker: "BBCA.JK"
    language: "id"
    query: 'BBCA saham OR "Bank Central Asia"'
  - type: "rss"
    ticker: "BBCA.JK"
    rss_url: "https://feeds.finance.yahoo.com/rss/2.0/headline?s=BBCA.JK&region=US&lang=en-US"
    source_name: "Yahoo Finance"

Output CSV:
data/raw/news/news_raw_google_rss.csv

Kolom:
    date
    ticker
    query
    query_type
    language
    title
    description
    link
    source
    published_raw
"""

import os
from datetime import datetime, timedelta, timezone
import urllib.parse
from typing import Any, Dict, List, Optional

import feedparser
import pandas as pd
import yaml
from email.utils import parsedate_to_datetime

# Lokasi root project
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_RAW_NEWS_DIR = os.path.join(ROOT_DIR, "data", "raw", "news")
CONFIG_RSS_PATH = os.path.join(ROOT_DIR, "configs", "rss.yaml")

os.makedirs(DATA_RAW_NEWS_DIR, exist_ok=True)

OUT_PATH = os.path.join(DATA_RAW_NEWS_DIR, "news_raw_google_rss.csv")


# ===================== CONFIG HANDLING =====================


def load_config(path: str) -> Dict[str, Any]:
    """
    Load config YAML. Kalau file tidak ada atau kosong, raise error yang jelas.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Config RSS tidak ditemukan: {path}. "
            f"Buat file rss.yaml sesuai struktur yang diharapkan."
        )

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        raise ValueError(f"Config RSS kosong: {path}")

    return cfg


def normalize_lookback_years(raw_value: Any) -> Optional[int]:
    """
    Normalisasi lookback_years dari config.

    Aturan:
    - None / "none" / "null" / ""  -> None  (tidak pakai filter tahun)
    - int/float > 0                -> int(raw_value)
    - Tipe lain / tidak valid      -> None (dan kasih warning)
    """
    if raw_value is None:
        return None

    if isinstance(raw_value, str):
        txt = raw_value.strip().lower()
        if txt in {"", "none", "null"}:
            return None
        try:
            val = int(txt)
        except ValueError:
            print(
                f"[WARN] lookback_years di config ('{raw_value}') bukan angka valid. "
                "Filter tahun dimatikan."
            )
            return None
        return val if val > 0 else None

    if isinstance(raw_value, (int, float)):
        if raw_value <= 0:
            return None
        return int(raw_value)

    print(
        f"[WARN] Tipe lookback_years tidak dikenali ({type(raw_value)}). "
        "Filter tahun dimatikan."
    )
    return None


def normalize_max_articles(raw_value: Any) -> Optional[int]:
    """
    Normalisasi max_articles_per_query.

    - None / 0 / negatif -> None (artinya tidak dibatasi)
    - int/float > 0      -> int
    - string angka       -> parse ke int
    - lain-lain          -> None + warning
    """
    if raw_value is None:
        return None

    if isinstance(raw_value, str):
        txt = raw_value.strip()
        if txt == "":
            return None
        try:
            val = int(txt)
        except ValueError:
            print(
                f"[WARN] max_articles_per_query di config ('{raw_value}') bukan angka valid. "
                "Jumlah artikel per query tidak dibatasi."
            )
            return None
        return val if val > 0 else None

    if isinstance(raw_value, (int, float)):
        if raw_value <= 0:
            return None
        return int(raw_value)

    print(
        f"[WARN] Tipe max_articles_per_query tidak dikenali ({type(raw_value)}). "
        "Jumlah artikel per query tidak dibatasi."
    )
    return None


# ===================== GOOGLE NEWS =====================


def build_google_news_rss_url(query: str, language: str = "id") -> str:
    """
    Bangun URL RSS Google News.

    Catatan:
    - hl  : bahasa interface (misal 'id-ID', 'en-ID')
    - gl  : country code (ID)
    - ceid: country-language (misal 'ID:id', 'ID:en')
    """
    base_url = "https://news.google.com/rss/search"

    # Coba bikin kode region-language yang masuk akal
    lang_lower = (language or "id").lower()
    if lang_lower == "id":
        hl = "id-ID"
        ceid = "ID:id"
    else:
        # Misal "en" -> "en-ID" dan "ID:en"
        hl = f"{lang_lower}-ID"
        ceid = f"ID:{lang_lower}"

    params = {
        "q": query,
        "hl": hl,
        "gl": "ID",
        "ceid": ceid,
    }
    url = base_url + "?" + urllib.parse.urlencode(params)
    return url


def parse_published(entry: feedparser.FeedParserDict) -> Optional[datetime.date]:
    """
    Ambil tanggal dari entry RSS:
    1) Coba pakai published_parsed / updated_parsed (struct_time)
    2) Kalau tidak ada, fallback ke string published/pubDate (RFC822)
    Return: datetime.date atau None kalau gagal.
    """
    parsed = entry.get("published_parsed") or entry.get("updated_parsed")
    if parsed:
        try:
            dt = datetime(*parsed[:6])
            return dt.date()
        except Exception:
            pass

    published_str = entry.get("published") or entry.get("pubDate")
    if published_str:
        try:
            dt = parsedate_to_datetime(published_str)
            return dt.date()
        except Exception:
            return None

    return None


def fetch_news_google(
    query: str,
    ticker: str,
    language: str,
    lookback_years: Optional[int] = None,
    max_articles: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Ambil berita dari Google News RSS untuk satu query-ticker.
    """
    url = build_google_news_rss_url(query, language=language)
    print(
        f"[INFO] Fetching GOOGLE RSS for query='{query}' "
        f"ticker='{ticker}' lang='{language}'"
    )
    feed = feedparser.parse(url)

    if getattr(feed, "bozo", False):
        print(f"[WARN] Feed Google untuk '{query}' bozo=True (mungkin RSS error).")

    total_entries = len(feed.entries)
    print(f"[INFO]  -> dapat {total_entries} entries mentah (Google) untuk {ticker}")

    records: List[Dict[str, Any]] = []

    # Batas tanggal bawah kalau pakai lookback_years
    min_date = None
    if lookback_years is not None:
        today = datetime.now(timezone.utc).date()
        min_date = today - timedelta(days=365 * lookback_years)
        print(f"[INFO]  -> Filter tanggal aktif (>= {min_date})")

    for entry in feed.entries:
        date = parse_published(entry)
        if date is None:
            continue

        if min_date is not None and date < min_date:
            continue

        title = entry.get("title", "") or ""
        summary = entry.get("summary", "") or ""
        link = entry.get("link", "") or ""

        source = ""
        src = entry.get("source")
        if isinstance(src, dict):
            source = src.get("title", "") or ""
        elif isinstance(src, str):
            source = src

        published_raw = entry.get("published", "") or entry.get("pubDate", "") or ""

        records.append(
            {
                "date": date,
                "ticker": ticker,
                "query": query,
                "query_type": "google",
                "language": language,
                "title": title,
                "description": summary,
                "link": link,
                "source": source or "GoogleNews",
                "published_raw": published_raw,
            }
        )

        if max_articles is not None and len(records) >= max_articles:
            print(
                f"[INFO]  -> Mencapai batas max_articles={max_articles} "
                f"untuk {ticker}, berhenti."
            )
            break

    print(
        f"[INFO] Selesai fetch GOOGLE untuk {ticker}: "
        f"{len(records)} artikel dengan tanggal valid."
    )
    return records


# ===================== DIRECT RSS (YAHOO, KONTAN, DLL) =====================


def fetch_news_rss_direct(
    rss_url: str,
    ticker: str,
    source_name: Optional[str],
    lookback_years: Optional[int] = None,
    max_articles: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Ambil berita dari RSS langsung (Yahoo, Kontan, CNBC, dll.).
    """
    print(f"[INFO] Fetching DIRECT RSS for ticker='{ticker}' url='{rss_url}'")
    feed = feedparser.parse(rss_url)

    if getattr(feed, "bozo", False):
        print(f"[WARN] Feed direct RSS '{rss_url}' bozo=True (mungkin RSS error).")

    total_entries = len(feed.entries)
    print(f"[INFO]  -> dapat {total_entries} entries mentah (direct RSS) untuk {ticker}")

    records: List[Dict[str, Any]] = []

    min_date = None
    if lookback_years is not None:
        today = datetime.now(timezone.utc).date()
        min_date = today - timedelta(days=365 * lookback_years)
        print(f"[INFO]  -> Filter tanggal aktif (>= {min_date})")

    for entry in feed.entries:
        date = parse_published(entry)
        if date is None:
            continue
        if min_date is not None and date < min_date:
            continue

        title = entry.get("title", "") or ""
        summary = entry.get("summary", "") or ""
        link = entry.get("link", "") or ""

        # Sumber: pakai source_name dari config kalau ada, kalau tidak pakai source di RSS
        src = source_name or entry.get("source") or ""
        if isinstance(src, dict):
            src = src.get("title", "") or ""

        published_raw = entry.get("published", "") or entry.get("pubDate", "") or ""

        records.append(
            {
                "date": date,
                "ticker": ticker,
                "query": rss_url,
                "query_type": "rss_direct",
                "language": "",
                "title": title,
                "description": summary,
                "link": link,
                "source": src or "RSS",
                "published_raw": published_raw,
            }
        )

        if max_articles is not None and len(records) >= max_articles:
            print(
                f"[INFO]  -> Mencapai batas max_articles={max_articles} "
                f"untuk {ticker}, berhenti."
            )
            break

    print(
        f"[INFO] Selesai fetch DIRECT RSS untuk {ticker}: "
        f"{len(records)} artikel dengan tanggal valid."
    )
    return records


# ===================== MAIN =====================


def main() -> None:
    config = load_config(CONFIG_RSS_PATH)

    default_language = config.get("default_language", "id")
    queries_cfg = config.get("queries", []) or []

    max_articles_per_query = normalize_max_articles(
        config.get("max_articles_per_query", None)
    )
    lookback_years = normalize_lookback_years(config.get("lookback_years", None))

    print("[INFO] Konfigurasi RSS:")
    print(f"       default_language       : {default_language}")
    print(f"       lookback_years         : {lookback_years}")
    print(f"       max_articles_per_query : {max_articles_per_query}")
    print(f"       jumlah queries         : {len(queries_cfg)}")

    all_records: List[Dict[str, Any]] = []

    for idx, q in enumerate(queries_cfg, start=1):
        q_type = q.get("type", "google")
        ticker = q.get("ticker")

        if not ticker:
            print(f"[WARN] Query ke-{idx} tidak punya ticker, skip.")
            continue

        if q_type == "google":
            query = q.get("query")
            if not query:
                print(f"[WARN] Query GOOGLE untuk ticker={ticker} tidak punya 'query', skip.")
                continue

            language = q.get("language", default_language)
            recs = fetch_news_google(
                query=query,
                ticker=ticker,
                language=language,
                lookback_years=lookback_years,
                max_articles=max_articles_per_query,
            )

        elif q_type == "rss":
            rss_url = q.get("rss_url")
            if not rss_url:
                print(f"[WARN] Query RSS untuk ticker={ticker} tidak punya 'rss_url', skip.")
                continue

            source_name = q.get("source_name")
            recs = fetch_news_rss_direct(
                rss_url=rss_url,
                ticker=ticker,
                source_name=source_name,
                lookback_years=lookback_years,
                max_articles=max_articles_per_query,
            )

        else:
            print(f"[WARN] Jenis query tidak dikenal: {q_type}, skip.")
            continue

        all_records.extend(recs)

    if not all_records:
        print("[WARN] Tidak ada berita yang berhasil diambil (tanggal valid).")
        pd.DataFrame(
            columns=[
                "date",
                "ticker",
                "query",
                "query_type",
                "language",
                "title",
                "description",
                "link",
                "source",
                "published_raw",
            ]
        ).to_csv(OUT_PATH, index=False)
        print(f"[INFO] Menyimpan file kosong ke: {OUT_PATH}")
        return

    df = pd.DataFrame(all_records)

    # Sedikit rapi:
    # - pastikan kolom date bertipe date (bukan string), lalu sort
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.sort_values(["ticker", "date", "source", "title"])

    # Optional: deduplikasi berdasarkan (ticker, title, link, date)
    df = df.drop_duplicates(subset=["ticker", "title", "link", "date"])

    print(f"[INFO] Total berita (semua ticker & sumber, setelah dedupe): {len(df)}")
    print(f"[INFO] Menyimpan ke: {OUT_PATH}")
    df.to_csv(OUT_PATH, index=False)
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
