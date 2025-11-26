"""
Scrape arsip HTML berita untuk menambah cakupan histori (hingga 5 tahun).

Konfigurasi di configs/html_archives.yaml:

lookback_years: 5
max_pages: 2  # per tanggal
sources:
  - name: "Tempo - Bisnis"
    ticker: "IDX"
    language: "id"
    url_template: "https://www.tempo.co/indeks/{year}-{month:02d}-{day:02d}/bisnis?page={page}"
    article_selector: "article a, div.card a, h2 a, h3 a"
    allow_patterns: ["tempo.co"]
    base_url: "https://www.tempo.co"
    date_from_url_regex: "(\\d{4}/\\d{2}/\\d{2})"

Output CSV: data/raw/news/news_raw_html_archives.csv
Kolom minimal: date, ticker, query, query_type, language, title, description, link, source, published_raw
"""

import os
import re
from datetime import date, timedelta
from typing import Any, Dict, Iterable, List, Optional, Sequence
from urllib.parse import urljoin

import pandas as pd
import yaml

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError as e:  # pragma: no cover - env dependency guard
    raise ImportError(
        "Butuh package requests dan beautifulsoup4. Jalankan: "
        "pip install requests beautifulsoup4"
    ) from e

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_RAW_NEWS_DIR = os.path.join(ROOT_DIR, "data", "raw", "news")
CONFIG_HTML_PATH = os.path.join(ROOT_DIR, "configs", "html_archives.yaml")

os.makedirs(DATA_RAW_NEWS_DIR, exist_ok=True)

OUT_PATH = os.path.join(DATA_RAW_NEWS_DIR, "news_raw_html_archives.csv")
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)


# ===================== CONFIG =====================


def load_config(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Config HTML archive tidak ditemukan: {path}. "
            "Salin template configs/html_archives.yaml."
        )

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    return cfg


def _normalize_int(raw: Any, default: Optional[int] = None) -> Optional[int]:
    if raw is None:
        return default
    if isinstance(raw, str):
        raw = raw.strip()
        if raw == "":
            return default
        if raw.lower() in {"none", "null"}:
            return default
        try:
            raw = int(raw)
        except ValueError:
            return default
    if isinstance(raw, (int, float)):
        return int(raw) if raw > 0 else default
    return default


def daterange_days(end_date: date, years: int) -> Iterable[date]:
    start_date = end_date - timedelta(days=365 * years)
    current = end_date
    while current >= start_date:
        yield current
        current -= timedelta(days=1)


# ===================== SCRAPING =====================


def fetch_html(url: str) -> Optional[str]:
    try:
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=20)
    except requests.RequestException as exc:
        print(f"[WARN] Gagal fetch {url}: {exc}")
        return None

    if resp.status_code >= 400:
        print(f"[WARN] HTTP {resp.status_code} untuk {url}")
        return None

    return resp.text


def normalize_link(href: str, base_url: Optional[str]) -> str:
    href = href.strip()
    if base_url:
        return urljoin(base_url, href)
    return href


def allowed_link(href: str, allow_patterns: Optional[List[str]]) -> bool:
    if not href:
        return False
    if allow_patterns:
        href_low = href.lower()
        for pat in allow_patterns:
            if not pat:
                continue
            if re.search(pat, href, flags=re.IGNORECASE):
                return True
            if pat.lower() in href_low:
                return True
        return False
    return True


def extract_date_from_url(href: str, regex: Optional[str]) -> Optional[str]:
    if not href or not regex:
        return None
    m = re.search(regex, href)
    if not m:
        return None
    # Ambil grup pertama sebagai string tanggal (YYYY/MM/DD atau YYYY-MM-DD)
    raw = m.group(1)
    raw = raw.replace("/", "-")
    parts = raw.split("-")
    if len(parts) < 3:
        return None
    try:
        y, mth, d = int(parts[0]), int(parts[1]), int(parts[2])
        return date(y, mth, d).isoformat()
    except ValueError:
        return None


def parse_articles(
    html: str,
    page_date: date,
    source_cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    soup = BeautifulSoup(html, "html.parser")

    tags_to_strip = source_cfg.get("tags_to_strip") or []
    for selector in tags_to_strip:
        for tag in soup.select(selector):
            tag.decompose()

    selector = source_cfg.get("article_selector")
    anchors = soup.select(selector) if selector else soup.find_all("a")

    allow_patterns = source_cfg.get("allow_patterns") or []
    base_url = source_cfg.get("base_url")
    date_regex = source_cfg.get("date_from_url_regex")

    records: List[Dict[str, Any]] = []

    for a in anchors:
        href = a.get("href") or ""
        href = href.strip()
        if not allowed_link(href, allow_patterns):
            continue

        link = normalize_link(href, base_url)
        title = a.get_text(strip=True) or a.get("title", "") or ""
        if not title:
            continue

        parsed_date = extract_date_from_url(link, date_regex) or page_date.isoformat()

        records.append(
            {
                "date": parsed_date,
                "title": title,
                "link": link,
            }
        )

    return records


# ===================== MAIN =====================


def main() -> None:
    cfg = load_config(CONFIG_HTML_PATH)

    lookback_years = _normalize_int(cfg.get("lookback_years"), default=3)
    max_pages = _normalize_int(cfg.get("max_pages"), default=2)
    sources_cfg = cfg.get("sources") or []

    print("[INFO] Konfigurasi HTML scraping:")
    print(f"       lookback_years : {lookback_years}")
    print(f"       max_pages      : {max_pages}")
    print(f"       total sources  : {len(sources_cfg)}")

    if not sources_cfg:
        print("[WARN] Tidak ada sumber di config. Keluar.")
        return

    today = date.today()

    all_records: List[Dict[str, Any]] = []

    for src_idx, src in enumerate(sources_cfg, start=1):
        name = src.get("name", f"source_{src_idx}")
        ticker = src.get("ticker", "IDX")
        language = src.get("language", "id")
        templates: Sequence[str] = src.get("url_templates") or []

        # Backward compatibility: single template key
        if not templates:
            single = src.get("url_template")
            if single:
                templates = [single]

        templates = [t for t in templates if t]

        if not templates:
            print(f"[WARN] Source '{name}' tidak punya url_template/url_templates, skip.")
            continue

        print(f"[INFO] === Source: {name} (ticker={ticker}) ===")

        for current_date in daterange_days(today, lookback_years):
            for page in range(1, (max_pages or 1) + 1):
                html = None
                last_url = None

                for tmpl_idx, tmpl in enumerate(templates):
                    url = tmpl.format(
                        year=current_date.year,
                        month=current_date.month,
                        day=current_date.day,
                        page=page,
                    )
                    last_url = url
                    html = fetch_html(url)
                    if html:
                        if tmpl_idx > 0:
                            print(
                                f"[INFO] Template ke-{tmpl_idx + 1} dipakai untuk {name}: {url}"
                            )
                        break

                if not html:
                    # kalau semua template gagal, lanjut ke tanggal berikutnya
                    if page == 1:
                        break
                    continue

                records = parse_articles(html, page_date=current_date, source_cfg=src)
                if not records:
                    # tidak ada artikel, lanjut tanggal berikutnya
                    if page == 1:
                        break
                    continue

                for rec in records:
                    rec.update(
                        {
                            "ticker": ticker,
                            "query": last_url,
                            "query_type": "html_archive",
                            "language": language,
                            "description": "",
                            "source": name,
                            "published_raw": rec.get("date", ""),
                        }
                    )

                all_records.extend(records)

    if not all_records:
        print("[WARN] Tidak ada artikel yang terkumpul dari HTML scraping.")
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
    df["date"] = pd.to_datetime(df["date"]).dt.date

    before = len(df)
    df = df.drop_duplicates(subset=["ticker", "title", "link", "date"])
    after = len(df)
    print(f"[INFO] Drop duplikat HTML: {before} -> {after}")

    df = df.sort_values(["ticker", "date", "source", "title"])

    print(f"[INFO] Total artikel HTML setelah dedupe: {len(df)}")
    print(f"[INFO] Menyimpan ke: {OUT_PATH}")
    df.to_csv(OUT_PATH, index=False)
    print("[INFO] Selesai.")


if __name__ == "__main__":
    main()
