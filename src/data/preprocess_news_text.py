# preprocess_news_text.py

import html
import os
import re
import unicodedata

import pandas as pd
import yaml

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RAW_DIR = os.path.join(ROOT_DIR, 'data', 'raw', 'news')
OUT_PATH = os.path.join(ROOT_DIR, 'data', 'interim', 'news_clean.csv')
SRC_ALL = os.path.join(RAW_DIR, 'news_raw_all_sources.csv')
SRC_GOOGLE = os.path.join(RAW_DIR, 'news_raw_google_rss.csv')
CONFIG_DATA_PATH = os.path.join(ROOT_DIR, 'configs', 'data.yaml')

ALLOWED_LANGUAGES = {'id'}

COMPANY = {
    'BBRI.JK': ['bbri', 'bank rakyat indonesia', 'bri'],
    'BMRI.JK': ['bmri', 'bank mandiri', 'mandiri'],
    'TLKM.JK': ['tlkm', 'telkom indonesia', 'telkom'],
    'ASII.JK': ['asii', 'astra international', 'astra'],
}

SECTOR = {
    'BBRI.JK': ['brimo', 'brilink', 'kur bri'],
    'BMRI.JK': ['livin', 'kopra'],
    'TLKM.JK': ['indihome', 'telkomsel', 'mitratel'],
    'ASII.JK': ['auto2000', 'astra financial', 'united tractors'],
}

FINANCE = {
    'saham', 'emiten', 'dividen', 'buyback', 'stock split',
    'laba', 'kinerja', 'laporan keuangan', 'ihsg', 'bei',
    'idx', 'harga saham', 'target harga'
}


def load_allowed_tickers() -> set[str]:
    with open(CONFIG_DATA_PATH, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f) or {}
    return {str(t).strip().upper() for t in cfg.get('tickers', []) if str(t).strip()}


def load_news() -> pd.DataFrame:
    src = SRC_ALL if os.path.exists(SRC_ALL) else SRC_GOOGLE
    if not os.path.exists(src):
        raise FileNotFoundError(f'File berita tidak ditemukan: {src}')
    return pd.read_csv(src)


def normalize_text(text: str) -> str:
    text = html.unescape(str(text or ''))
    text = unicodedata.normalize('NFKC', text).lower()
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'https?://\S+', ' ', text)
    text = re.sub(r'[^\w\s.%/-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def is_relevant(text: str, ticker: str) -> bool:
    company_hits = any(k in text for k in COMPANY.get(ticker, []))
    sector_hits = any(k in text for k in SECTOR.get(ticker, []))
    finance_hits = any(k in text for k in FINANCE)
    return company_hits or (sector_hits and finance_hits)


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    allowed_tickers = load_allowed_tickers()

    df = load_news().copy()
    if 'date' not in df.columns or 'ticker' not in df.columns:
        raise KeyError("Kolom wajib 'date' dan 'ticker' tidak ditemukan.")

    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['ticker'] = df['ticker'].astype(str).str.upper()
    df['language'] = df.get('language', 'id').astype(str).str.lower()

    if allowed_tickers:
        df = df[df['ticker'].isin(allowed_tickers)].copy()

    df = df[df['language'].isin(ALLOWED_LANGUAGES)].copy()

    for c in ['title', 'description', 'link']:
        if c not in df.columns:
            df[c] = ''

    df['title_clean'] = df['title'].map(normalize_text)
    df['description_clean'] = df['description'].map(normalize_text)
    df['text_for_label'] = (
        df['title_clean'].fillna('') + '. ' + df['description_clean'].fillna('')
    ).str.strip('. ').str.strip()

    df['is_relevant'] = [
        is_relevant(text, ticker)
        for text, ticker in zip(df['text_for_label'], df['ticker'])
    ]

    df = df[df['is_relevant']].copy()
    df = df[df['text_for_label'].ne('')].copy()

    out = df[['date', 'ticker', 'link', 'text_for_label']].copy()
    out['link'] = out['link'].astype(str).str.strip()

    out = out.dropna(subset=['date', 'ticker'])
    out = out.drop_duplicates(
        subset=['ticker', 'date', 'link', 'text_for_label'],
        keep='first',
    )
    out = out.sort_values(['ticker', 'date', 'link']).reset_index(drop=True)

    out.to_csv(OUT_PATH, index=False)
    print(f'[INFO] Final shape : {out.shape[0]} rows, {out.shape[1]} columns')
    print(f'[INFO] Saving cleaned news to: {OUT_PATH}')
    print('[INFO] Done.')


if __name__ == '__main__':
    main()
