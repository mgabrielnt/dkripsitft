# gpt_sentiment_labeling.py

import hashlib
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import yaml

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.utils.gpt_client import MODEL_NAME, TICKER_TO_COMPANY, classify_sentiment

SRC_PATH = os.path.join(ROOT_DIR, 'data', 'interim', 'news_clean.csv')
PRICES_PATH = os.path.join(ROOT_DIR, 'data', 'interim', 'prices_with_indicators.csv')
OUT_PATH = os.path.join(ROOT_DIR, 'data', 'processed', 'news_with_sentiment_per_article.csv')
GPT_CFG_PATH = os.path.join(ROOT_DIR, 'configs', 'gpt_sentiment.yaml')
SENT_CFG_PATH = os.path.join(ROOT_DIR, 'configs', 'sentiment.yaml')
DATA_CFG_PATH = os.path.join(ROOT_DIR, 'configs', 'data.yaml')


def load_yaml(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def get_allowed_tickers(cfg: dict) -> set[str]:
    return {
        str(t).strip().upper()
        for t in cfg.get('tickers', [])
        if str(t).strip()
    }


def parse_end_date(raw):
    if raw is None:
        return None
    dt = pd.to_datetime(raw, errors='coerce')
    return None if pd.isna(dt) else dt.normalize()


def lex_score(text: str, pos: set[str], neg: set[str]) -> float:
    toks = [t for t in str(text).lower().split() if t]
    if not toks:
        return 0.0
    score = sum(1 for t in toks if t in pos) - sum(1 for t in toks if t in neg)
    return score / math.sqrt(len(toks))


def calibrate(values: pd.Series, method: str, q1: float, q2: float) -> tuple[float, float]:
    values = values.dropna().abs()
    if values.empty:
        return 0.0, 0.0
    if method == 'std':
        std = float(values.std())
        return q1 * std, q2 * std
    return float(values.quantile(q1)), float(values.quantile(q2))


def market_lookup(prices: pd.DataFrame, horizon: int, cfg: dict) -> tuple[dict[tuple[str, str], float], float, float]:
    prices = prices[['ticker', 'date', 'close']].copy()
    prices['date'] = pd.to_datetime(prices['date'], errors='coerce')
    prices['close'] = pd.to_numeric(prices['close'], errors='coerce').replace(0, pd.NA)
    prices = prices.sort_values(['ticker', 'date'])

    prices['close_tph'] = prices.groupby('ticker')['close'].shift(-horizon)
    prices['ret'] = (prices['close_tph'] - prices['close']) / prices['close']
    prices['bench'] = prices.groupby('date')['ret'].transform('mean')
    prices['abnormal_return'] = prices['ret'] - prices['bench'].fillna(0.0)

    mcfg = cfg.get('market', {})
    theta1, theta2 = calibrate(
        prices['abnormal_return'],
        mcfg.get('threshold_method', 'quantile'),
        float(mcfg.get('theta_q1', 0.75)),
        float(mcfg.get('theta_q2', 0.9)),
    )

    lookup = {
        (r.ticker, str(r.date.date())): float(r.abnormal_return)
        for r in prices.itertuples()
        if pd.notna(r.abnormal_return)
    }
    return lookup, theta1, theta2


def shift_to_next_monday(d: pd.Timestamp) -> pd.Timestamp:
    return d + pd.Timedelta(days=7 - d.weekday()) if pd.notna(d) and d.weekday() >= 5 else d


def sign_by_threshold(x: float, t: float) -> int:
    return 1 if x >= t else (-1 if x <= -t else 0)


def final_label(l_text: int, l_market: int, l_lex: int) -> int:
    if l_text != 0:
        agree = (l_market == l_text) + (l_lex == l_text)
        oppose = (l_market == -l_text) + (l_lex == -l_text)
        if oppose == 2:
            return 0
        return l_text

    if l_market != 0:
        return l_market
    return l_lex


def main():
    if not os.path.exists(SRC_PATH) or not os.path.exists(PRICES_PATH):
        raise FileNotFoundError('news_clean.csv atau prices_with_indicators.csv tidak ditemukan.')

    gpt_cfg = load_yaml(GPT_CFG_PATH)
    sent_cfg = load_yaml(SENT_CFG_PATH)
    data_cfg = load_yaml(DATA_CFG_PATH)
    allowed_tickers = get_allowed_tickers(data_cfg)

    score_map = gpt_cfg.get('score_mapping', {'NEGATIF': -1, 'NETRAL': 0, 'POSITIF': 1})
    pos = set(sent_cfg.get('lexicon', {}).get('positive_words', []))
    neg = set(sent_cfg.get('lexicon', {}).get('negative_words', []))
    horizon = int(data_cfg.get('horizon', 3))

    df = pd.read_csv(SRC_PATH, parse_dates=['date'])
    prices = pd.read_csv(PRICES_PATH)

    required_cols = {'date', 'ticker', 'link', 'text_for_label'}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f'Kolom wajib tidak ditemukan di news_clean.csv: {sorted(missing)}')

    df['ticker'] = df['ticker'].astype(str).str.upper()
    df['link'] = df['link'].astype(str).str.strip()
    df['text_for_label'] = df['text_for_label'].astype(str).str.strip()
    df = df[df['text_for_label'].ne('')].copy()

    if 'ticker' in prices.columns:
        prices['ticker'] = prices['ticker'].astype(str).str.upper()

    if allowed_tickers:
        df = df[df['ticker'].isin(allowed_tickers)].copy()
        if 'ticker' in prices.columns:
            prices = prices[prices['ticker'].isin(allowed_tickers)].copy()

    end_date = parse_end_date(data_cfg.get('end_date'))
    if end_date is not None:
        df = df[df['date'] <= end_date].copy()
        if 'date' in prices.columns:
            prices['date'] = pd.to_datetime(prices['date'], errors='coerce')
            prices = prices[prices['date'] <= end_date].copy()

    market_map, theta1, theta2 = market_lookup(prices, horizon, sent_cfg)

    cache: dict[tuple[str, str, str], tuple[str, int]] = {}
    if os.path.exists(OUT_PATH):
        old = pd.read_csv(OUT_PATH)
        if 'ticker' in old.columns:
            old['ticker'] = old['ticker'].astype(str).str.upper()
            if allowed_tickers:
                old = old[old['ticker'].isin(allowed_tickers)].copy()
        needed_cache_cols = {'link', 'ticker', 'text_for_label', 'l_text_label', 'l_text'}
        if needed_cache_cols.issubset(old.columns):
            for r in old.itertuples():
                txt = str(getattr(r, 'text_for_label', '') or '')
                key = (
                    str(getattr(r, 'link', '')),
                    str(getattr(r, 'ticker', '')).upper(),
                    hashlib.md5(txt.encode('utf-8')).hexdigest(),
                )
                cache[key] = (
                    str(getattr(r, 'l_text_label', 'NETRAL')),
                    int(getattr(r, 'l_text', 0)),
                )

    keys = [
        (
            str(r.link or ''),
            str(r.ticker or '').upper(),
            hashlib.md5(str(r.text_for_label).encode('utf-8')).hexdigest(),
        )
        for r in df.itertuples()
    ]

    labels = ['NETRAL'] * len(df)
    scores = [0] * len(df)
    pending: list[int] = []

    for i, key in enumerate(keys):
        if key in cache:
            labels[i], scores[i] = cache[key]
        else:
            pending.append(i)

    def worker(i: int):
        row = df.iloc[i]
        label = classify_sentiment(
            row['text_for_label'],
            ticker=str(row.get('ticker', '')),
            company=TICKER_TO_COMPANY.get(str(row.get('ticker', '')), ''),
        )
        return i, label

    if pending:
        with ThreadPoolExecutor(max_workers=int(gpt_cfg.get('max_workers', 1))) as ex:
            futures = [ex.submit(worker, i) for i in pending]
            for fut in as_completed(futures):
                i, label = fut.result()
                labels[i] = label
                scores[i] = int(score_map.get(label, 0))

    df['l_text_label'] = labels
    df['l_text'] = scores

    df['lex_score_norm'] = df['text_for_label'].map(lambda x: lex_score(x, pos, neg))
    tau1, tau2 = calibrate(
        df['lex_score_norm'],
        sent_cfg.get('lexicon', {}).get('threshold_method', 'quantile'),
        float(sent_cfg.get('lexicon', {}).get('tau_q1', 0.7)),
        float(sent_cfg.get('lexicon', {}).get('tau_q2', 0.9)),
    )
    df['l_lex'] = df['lex_score_norm'].map(lambda x: sign_by_threshold(float(x), tau1))

    df['event_date'] = df['date'].map(shift_to_next_monday)
    df['abnormal_return'] = [
        market_map.get((str(t), str(pd.to_datetime(d).date())), 0.0)
        for t, d in zip(df['ticker'], df['event_date'])
    ]
    df['l_market'] = df['abnormal_return'].map(lambda x: sign_by_threshold(float(x), theta1))

    df['l_final'] = [
        final_label(int(a), int(b), int(c))
        for a, b, c in zip(df['l_text'], df['l_market'], df['l_lex'])
    ]

    out = df[
        ['date', 'ticker', 'link', 'text_for_label', 'l_text_label', 'l_text', 'l_final']
    ].copy()

    if allowed_tickers:
        out = out[out['ticker'].astype(str).str.upper().isin(allowed_tickers)].copy()

    out = out.drop_duplicates(
        subset=['ticker', 'date', 'link', 'text_for_label'],
        keep='last',
    ).sort_values(['ticker', 'date', 'link']).reset_index(drop=True)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    out.to_csv(OUT_PATH, index=False)

    print(f'[INFO] Loading {SRC_PATH}')
    print(f'[INFO] Horizon market signal = {horizon} hari')
    print(f'[INFO] θ1 (market)={theta1:.4f}, θ2 (market strong)={theta2:.4f}')
    print(f'[INFO] τ1 (lex)={tau1:.4f}, τ2 (lex strong)={tau2:.4f}')
    print(f'[INFO] Model GPT = {MODEL_NAME}')
    print(f'[INFO] Saving to {OUT_PATH}')


if __name__ == '__main__':
    main()
