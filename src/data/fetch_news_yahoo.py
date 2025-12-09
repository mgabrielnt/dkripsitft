"""
fetch_news_yahoo.py

Mengambil berita dari Yahoo Finance untuk daftar ticker
dan menyimpannya ke CSV: data/raw/news/news_raw_yahoo.csv

- Membaca konfigurasi dari configs/yahoo_news.yaml
- Field penting di config:
    - tickers: list string, contoh ["BBCA.JK", "BBRI.JK"]
    - lookback_years:
        * None / null  -> tidak ada filter tahun (ambil semua)
        * int > 0      -> hanya ambil berita dalam N tahun terakhir
"""

import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

try:
    import yfinance as yf
except ImportError as e:
    raise ImportError(
        "Module 'yfinance' belum terinstall. Jalankan: pip install yfinance"
    ) from e

# Lokasi root project
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_RAW_NEWS_DIR = os.path.join(ROOT_DIR, "data", "raw", "news")
os.makedirs(DATA_RAW_NEWS_DIR, exist_ok=True)

CONFIG_YAHOO_PATH = os.path.join(ROOT_DIR, "configs", "yahoo_news.yaml")
OUT_PATH = os.path.join(DATA_RAW_NEWS_DIR, "news_raw_yahoo.csv")

DEFAULT_CONFIG: Dict[str, Any] = {
    "tickers": ["BBCA.JK", "BBRI.JK"],
    "lookback_years": None,
}


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------
def load_config(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        print(f"[WARN] Config Yahoo tidak ditemukan: {path}, pakai default: {DEFAULT_CONFIG}")
        return DEFAULT_CONFIG.copy()

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    merged = DEFAULT_CONFIG.copy()
    merged.update({k: v for k, v in cfg.items() if v is not None})
    return merged


def normalize_lookback_years(raw_value: Any) -> Optional[int]:
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


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main() -> None:
    cfg = load_config(CONFIG_YAHOO_PATH)

    tickers: List[str] = cfg.get("tickers", DEFAULT_CONFIG["tickers"])
    raw_lookback = cfg.get("lookback_years", DEFAULT_CONFIG["lookback_years"])
    lookback_years = normalize_lookback_years(raw_lookback)

    print("[INFO] Konfigurasi Yahoo News:")
    print(f"       tickers        : {tickers}")
    print(f"       lookback_years : {lookback_years}")

    min_date = None
    if lookback_years is not None:
        today = datetime.now(timezone.utc).date()
        min_date = today - timedelta(days=365 * lookback_years)
        print(f"[INFO] Filter tanggal aktif. Hanya ambil berita >= {min_date}")
    else:
        print("[INFO] Filter tanggal NONAKTIF (ambil semua berita yang dikembalikan Yahoo).")

    all_records: List[Dict[str, Any]] = []

    for ticker in tickers:
        print(f"[INFO] Fetching Yahoo Finance news for {ticker}")
        yf_ticker = yf.Ticker(ticker)

        news_list = getattr(yf_ticker, "news", []) or []
        print(f"[INFO] Yahoo Finance mengembalikan {len(news_list)} item mentah untuk {ticker}.")

        if not news_list:
            print(f"[WARN] Tidak ada news dari Yahoo untuk {ticker}.")
            continue

        kept_for_ticker = 0

        for item in news_list:
            # Debug bisa dimatikan jika sudah yakin
            # print("[DEBUG RAW ITEM]", item)

            content = item.get("content") or {}

            # TITLE
            title = content.get("title") or item.get("title") or ""

            # SUMMARY / DESCRIPTION
            description = (
                content.get("summary")
                or content.get("description")
                or item.get("summary")
                or item.get("description")
                or ""
            )

            # SOURCE
            provider = content.get("provider") or {}
            source = provider.get("displayName") or item.get("publisher") or "Yahoo Finance"

            # LINK
            canonical = content.get("canonicalUrl") or {}
            click = content.get("clickThroughUrl") or {}
            link = (
                canonical.get("url")
                or (click.get("url") if isinstance(click, dict) else None)
                or content.get("previewUrl")
                or item.get("link")
                or ""
            )

            # WAKTU PUBLIKASI
            pub_time = item.get("providerPublishTime")
            dt_utc: Optional[datetime] = None
            pub_epoch: Optional[int] = None

            if isinstance(pub_time, (int, float)):
                dt_utc = datetime.fromtimestamp(pub_time, tz=timezone.utc)
                pub_epoch = int(pub_time)

            if dt_utc is None:
                pub_date_str = (
                    content.get("pubDate")
                    or content.get("displayTime")
                    or item.get("pubDate")
                    or item.get("displayTime")
                )
                if pub_date_str:
                    try:
                        dt_utc = datetime.fromisoformat(pub_date_str.replace("Z", "+00:00"))
                        pub_epoch = int(dt_utc.timestamp())
                    except Exception as e:
                        print(
                            f"[DEBUG] Gagal parse pubDate '{pub_date_str}' "
                            f"untuk {ticker}: {e}"
                        )

            if dt_utc is None:
                print(
                    f"[DEBUG] Tidak ada timestamp valid untuk {ticker}, "
                    f"judul: {title[:50]!r}... → skip"
                )
                continue

            date_utc = dt_utc.date()

            if min_date is not None and date_utc < min_date:
                continue

            all_records.append(
                {
                    "date": date_utc.isoformat(),
                    "ticker": ticker,
                    "query": "yahoo_finance",
                    "title": title,
                    "description": description,
                    "link": link,
                    "source": source,
                    "published_raw": pub_epoch,
                    "published_dt_utc": dt_utc.isoformat(),
                }
            )
            kept_for_ticker += 1

        print(
            f"[INFO] Total berita yang disimpan untuk {ticker}: {kept_for_ticker} "
            f"dari {len(news_list)} item mentah."
        )

    if not all_records and not os.path.exists(OUT_PATH):
        print("[WARN] Tidak ada berita Yahoo yang berhasil diambil (setelah filter).")
        df_empty = pd.DataFrame(
            columns=[
                "date",
                "ticker",
                "query",
                "title",
                "description",
                "link",
                "source",
                "published_raw",
                "published_dt_utc",
            ]
        )
        df_empty.to_csv(OUT_PATH, index=False)
        print(f"[INFO] Menyimpan file kosong ke: {OUT_PATH}")
        return

    df_new = pd.DataFrame(all_records)
    if not df_new.empty:
        df_new["date"] = pd.to_datetime(df_new["date"], errors="coerce").dt.date
        df_new = df_new.sort_values(["ticker", "date", "published_raw"])

    if os.path.exists(OUT_PATH):
        print(f"[INFO] Ditemukan file lama: {OUT_PATH}, lakukan merge incremental.")
        df_old = pd.read_csv(OUT_PATH, parse_dates=["date"])
        df_old["date"] = df_old["date"].dt.date
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new

    if not df_all.empty:
        before = len(df_all)
        df_all = df_all.drop_duplicates(
            subset=["ticker", "date", "title", "link"], keep="last"
        )
        after = len(df_all)
        print(f"[INFO] Drop duplikat (ticker,date,title,link): {before} -> {after}")
        df_all = df_all.sort_values(["ticker", "date", "published_raw"]).reset_index(
            drop=True
        )

    print(f"[INFO] Menyimpan berita Yahoo ke: {OUT_PATH}")
    df_all.to_csv(OUT_PATH, index=False)
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
