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

Output CSV kolom:
    date            : tanggal (UTC) dalam format YYYY-MM-DD
    ticker          : ticker saham (misal "BBCA.JK")
    query           : penanda sumber, di sini "yahoo_finance"
    title           : judul berita
    description     : ringkasan/summary (kalau ada)
    link            : URL ke berita
    source          : nama publisher
    published_raw   : epoch time (detik, UTC)
    published_dt_utc: datetime lengkap UTC (ISO format)
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

# Lokasi root project (sesuaikan dengan strukturmu)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_RAW_NEWS_DIR = os.path.join(ROOT_DIR, "data", "raw", "news")
CONFIG_YAHOO_PATH = os.path.join(ROOT_DIR, "configs", "yahoo_news.yaml")

os.makedirs(DATA_RAW_NEWS_DIR, exist_ok=True)

OUT_PATH = os.path.join(DATA_RAW_NEWS_DIR, "news_raw_yahoo.csv")

DEFAULT_CONFIG: Dict[str, Any] = {
    "tickers": ["BBCA.JK", "BBRI.JK"],
    # Default: None supaya TIDAK ada filter tahun kalau config tidak diisi
    "lookback_years": None,
}


def load_config(path: str) -> Dict[str, Any]:
    """Load konfigurasi YAML. Kalau tidak ada, pakai DEFAULT_CONFIG."""
    if not os.path.exists(path):
        print(f"[WARN] Config Yahoo tidak ditemukan: {path}, pakai default: {DEFAULT_CONFIG}")
        return DEFAULT_CONFIG.copy()

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # Gabungkan dengan default (yang di YAML override)
    merged = DEFAULT_CONFIG.copy()
    merged.update({k: v for k, v in cfg.items() if v is not None})

    return merged


def normalize_lookback_years(raw_value: Any) -> Optional[int]:
    """
    Normalisasi nilai lookback_years dari config.

    Aturan:
    - None / "none" / "null" / ""  -> None  (tidak pakai filter tahun)
    - int/float > 0                -> int(raw_value)
    - Tipe lain atau tidak valid   -> None (dan kasih warning)
    """
    if raw_value is None:
        return None

    # Kalau string: coba parse
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

    # Kalau numeric
    if isinstance(raw_value, (int, float)):
        if raw_value <= 0:
            return None
        return int(raw_value)

    # Tipe lain
    print(
        f"[WARN] Tipe lookback_years tidak dikenali ({type(raw_value)}). "
        "Filter tahun dimatikan."
    )
    return None


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

        # yfinance menjadikan .news sebagai property (list of dict)
        news_list = getattr(yf_ticker, "news", []) or []

        print(f"[INFO] Yahoo Finance mengembalikan {len(news_list)} item mentah untuk {ticker}.")

        if not news_list:
            print(f"[WARN] Tidak ada news dari Yahoo untuk {ticker}.")
            continue

        kept_for_ticker = 0

        for item in news_list:
            print("[DEBUG RAW ITEM]", item)  # supaya kelihatan struktur aslinya

            # Yahoo format baru: field penting ada di item["content"]
            content = item.get("content") or {}

            # ====== TITLE ======
            title = (
                content.get("title")
                or item.get("title")
                or ""
            )

            # ====== SUMMARY / DESCRIPTION ======
            description = (
                content.get("summary")
                or content.get("description")
                or item.get("summary")
                or item.get("description")
                or ""
            )

            # ====== SOURCE / PUBLISHER ======
            provider = content.get("provider") or {}
            source = (
                provider.get("displayName")
                or item.get("publisher")
                or "Yahoo Finance"
            )

            # ====== LINK ======
            canonical = content.get("canonicalUrl") or {}
            click = content.get("clickThroughUrl") or {}
            link = (
                canonical.get("url")
                or (click.get("url") if isinstance(click, dict) else None)
                or content.get("previewUrl")
                or item.get("link")
                or ""
            )

            # ====== WAKTU PUBLIKASI ======
            pub_time = item.get("providerPublishTime")
            dt_utc: Optional[datetime] = None
            pub_epoch: Optional[int] = None

            # Format lama: providerPublishTime = epoch seconds
            if isinstance(pub_time, (int, float)):
                dt_utc = datetime.fromtimestamp(pub_time, tz=timezone.utc)
                pub_epoch = int(pub_time)

            # Kalau belum dapat, coba format baru: ISO datetime di pubDate/displayTime
            if dt_utc is None:
                pub_date_str = (
                    content.get("pubDate")
                    or content.get("displayTime")
                    or item.get("pubDate")
                    or item.get("displayTime")
                )
                if pub_date_str:
                    try:
                        # Contoh: "2025-09-12T16:32:00Z"
                        dt_utc = datetime.fromisoformat(
                            pub_date_str.replace("Z", "+00:00")
                        )
                        pub_epoch = int(dt_utc.timestamp())
                    except Exception as e:
                        print(
                            f"[DEBUG] Gagal parse pubDate '{pub_date_str}' "
                            f"untuk {ticker}: {e}"
                        )

            if dt_utc is None:
                # Kalau masih tidak ada timestamp yang valid, skip saja
                print(
                    f"[DEBUG] Tidak ada timestamp valid untuk {ticker}, "
                    f"judul: {title[:50]!r}... → skip"
                )
                continue

            date_utc = dt_utc.date()

            # Filter berdasarkan min_date (kalau aktif)
            if min_date is not None and date_utc < min_date:
                print(
                    f"[DEBUG] News terlalu lama ({date_utc}) < {min_date}, "
                    f"ticker={ticker}, judul={title[:50]!r}... → skip"
                )
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

    # Setelah semua ticker diproses
    if not all_records:
        print("[WARN] Tidak ada berita Yahoo yang berhasil diambil (setelah filter).")
        # tetap simpan CSV kosong supaya pipeline tidak error
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

    df = pd.DataFrame(all_records)

    # Pastikan tipe tanggal benar (kalau mau dipakai groupby di step berikutnya)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date

    # Sort biar rapi
    df = df.sort_values(["ticker", "date", "published_raw"])

    print(f"[INFO] Total berita Yahoo (semua ticker, setelah filter): {len(df)}")
    print(f"[INFO] Rentang tanggal: {df['date'].min()}  s/d  {df['date'].max()}")
    print(f"[INFO] Menyimpan ke: {OUT_PATH}")
    df.to_csv(OUT_PATH, index=False)
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
