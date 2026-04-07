import os
import random
import time
import urllib.parse
from datetime import date, datetime, timedelta
from email.utils import parsedate_to_datetime
from typing import Any

import feedparser
import pandas as pd
import yaml

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CFG_PATH = os.path.join(ROOT_DIR, "configs", "rss.yaml")
DATA_CFG_PATH = os.path.join(ROOT_DIR, "configs", "data.yaml")
OUT_PATH = os.path.join(ROOT_DIR, "data", "raw", "news", "news_raw_all_sources.csv")

BASE_COLS = ["date", "ticker", "language", "title", "description", "link", "source"]


def load_yaml(path: str) -> dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_allowed_tickers(data_cfg: dict[str, Any]) -> set[str]:
    return {
        str(t).strip().upper()
        for t in data_cfg.get("tickers", [])
        if str(t).strip()
    }


def to_date(raw: Any) -> date | None:
    if raw is None:
        return None
    try:
        dt = pd.to_datetime(raw, errors="coerce")
        return None if pd.isna(dt) else dt.date()
    except Exception:
        return None


def parse_published(entry: Any) -> date | None:
    for key in ("published", "updated"):
        raw = entry.get(key)
        if not raw:
            continue
        try:
            return parsedate_to_datetime(raw).date()
        except Exception:
            dt = pd.to_datetime(raw, errors="coerce")
            if not pd.isna(dt):
                return dt.date()
    return None


def read_existing_output(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=BASE_COLS)
    try:
        return pd.read_csv(path).reindex(columns=BASE_COLS)
    except Exception:
        return pd.DataFrame(columns=BASE_COLS)


def last_saved_date(path: str, ticker: str) -> date | None:
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, usecols=["ticker", "date"])
        df = df[df["ticker"].astype(str).str.upper() == ticker.upper()]
        if df.empty:
            return None
        dt = pd.to_datetime(df["date"], errors="coerce").max()
        return None if pd.isna(dt) else dt.date()
    except Exception:
        return None


def google_url(query: str, language: str) -> str:
    hl = f"{language}-ID" if language == "id" else language
    params = {
        "q": query,
        "hl": hl,
        "gl": "ID",
        "ceid": f"ID:{language}",
        "output": "rss",
    }
    return "https://news.google.com/rss/search?" + urllib.parse.urlencode(params)


def make_row(
    published: date,
    ticker: str,
    language: str,
    title: str,
    description: str,
    link: str,
    source: str,
) -> dict[str, Any]:
    return {
        "date": published,
        "ticker": ticker,
        "language": language,
        "title": title,
        "description": description,
        "link": link,
        "source": source,
    }


def fetch_google(query: str, ticker: str, language: str, start: date, end: date, limit: int | None):
    rows, seen = [], set()
    cursor = start

    while cursor <= end and (limit is None or len(rows) < limit):
        nxt = min(cursor + timedelta(days=31), end + timedelta(days=1))
        q = f"{query} after:{cursor.isoformat()} before:{nxt.isoformat()}"
        feed = feedparser.parse(google_url(q, language))

        for entry in getattr(feed, "entries", []):
            published = parse_published(entry)
            link = str(entry.get("link", "")).strip()
            if not published or not link or link in seen:
                continue

            rows.append(
                make_row(
                    published=published,
                    ticker=ticker,
                    language=language,
                    title=entry.get("title", ""),
                    description=entry.get("summary", ""),
                    link=link,
                    source=entry.get("source", {}).get("title", "GoogleNews"),
                )
            )
            seen.add(link)

            if limit is not None and len(rows) >= limit:
                break

        cursor = nxt
        time.sleep(random.uniform(1.2, 2.2))

    return rows


def fetch_direct(url: str, ticker: str, source_name: str, language: str):
    rows = []
    feed = feedparser.parse(url)

    for entry in getattr(feed, "entries", []):
        published = parse_published(entry)
        link = str(entry.get("link", "")).strip()
        if not published or not link:
            continue

        rows.append(
            make_row(
                published=published,
                ticker=ticker,
                language=language,
                title=entry.get("title", ""),
                description=entry.get("summary", ""),
                link=link,
                source=source_name,
            )
        )

    return rows


def main():
    cfg = load_yaml(CFG_PATH)
    data_cfg = load_yaml(DATA_CFG_PATH)
    allowed_tickers = get_allowed_tickers(data_cfg)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    default_start = to_date(cfg.get("start_date_history", data_cfg.get("start_date", "2017-01-01")))
    default_lang = str(cfg.get("default_language", "id"))
    max_articles = cfg.get("max_articles_per_query")
    today = to_date(data_cfg.get("end_date")) or datetime.now().date()

    all_rows = []

    print("===================================================")
    print("      SMART NEWS SCRAPING (AUTO-RESUME)")
    print("===================================================")
    print(f"[INFO] Rentang pengambilan: {default_start} s.d. {today}")

    for q in cfg.get("queries", []):
        qtype = str(q.get("type", "")).lower()
        ticker = str(q.get("ticker", "UNKNOWN")).upper()
        lang = str(q.get("language", default_lang))

        if allowed_tickers and ticker not in allowed_tickers:
            print(f"[SKIP] {ticker}: tidak ada di data.yaml.")
            continue

        if qtype == "google":
            last_dt = last_saved_date(OUT_PATH, ticker)
            start = max(default_start, last_dt + timedelta(days=1)) if last_dt else default_start
            if start > today:
                print(f"[SKIP] {ticker}: tidak ada periode baru.")
                continue

            rows = fetch_google(str(q.get("query", "")), ticker, lang, start, today, max_articles)
            print(f"[INFO] {ticker} google_history -> {len(rows)} artikel baru")
            all_rows.extend(rows)

        elif qtype == "rss":
            rows = fetch_direct(str(q.get("url", "")), ticker, str(q.get("source", "RSS")), lang)
            print(f"[INFO] {ticker} rss_direct -> {len(rows)} artikel")
            all_rows.extend(rows)

    old = read_existing_output(OUT_PATH)
    if not old.empty:
        old["ticker"] = old["ticker"].astype(str).str.upper()
        if allowed_tickers:
            old = old[old["ticker"].isin(allowed_tickers)].copy()

    new = pd.DataFrame(all_rows).reindex(columns=BASE_COLS)
    out = pd.concat([old, new], ignore_index=True).reindex(columns=BASE_COLS)

    if not out.empty:
        out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date
        out["ticker"] = out["ticker"].astype(str).str.upper()
        out["language"] = out["language"].fillna("").astype(str).str.lower()
        out["title"] = out["title"].fillna("").astype(str)
        out["description"] = out["description"].fillna("").astype(str)
        out["link"] = out["link"].astype(str).str.strip()
        out["source"] = out["source"].fillna("").astype(str)

        out = out.dropna(subset=["date", "ticker", "link"])
        out = out[out["date"] <= today]
        if allowed_tickers:
            out = out[out["ticker"].isin(allowed_tickers)].copy()
        out = out.drop_duplicates(subset=["ticker", "link", "date"], keep="last")
        out = out.sort_values(["ticker", "date", "link"]).reset_index(drop=True)
        out = out.reindex(columns=BASE_COLS)

    out.to_csv(OUT_PATH, index=False)
    print(f"[INFO] Total rows tersimpan: {len(out)}")
    print(f"[INFO] Saving to: {OUT_PATH}")
    print("[INFO] Kolom final:", ",".join(out.columns.tolist()))
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
