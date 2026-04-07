import os
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RAW_PATH = os.path.join(ROOT_DIR, 'data', 'raw', 'prices', 'prices_all_raw.csv')
OUT_PATH = os.path.join(ROOT_DIR, 'data', 'interim', 'prices_with_indicators.csv')
CFG_PATH = os.path.join(ROOT_DIR, 'configs', 'data.yaml')
KEEP_COLS = [
    'ticker', 'date', 'close', 'volume', 'log_return_1d', 'vol_20', 'rsi_14',
    'ma_5_div_ma_20', 'bb_width_20', 'gap_return_1d', 'intraday_range_pct',
]


def load_yaml(path: str) -> dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def parse_end_date(raw: Any):
    if raw is None:
        return None
    dt = pd.to_datetime(raw, errors='coerce')
    return None if pd.isna(dt) else dt.normalize()


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).rolling(period, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def normalize_raw(df: pd.DataFrame) -> pd.DataFrame:
    if 'date' not in df.columns:
        raise KeyError("Kolom 'date' tidak ditemukan di prices_all_raw.csv.")
    df = df[df['date'].notna()].copy()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    base_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    suffixes = ['', '.1', '.2', '.3', '.4', '.5']
    for base in base_cols:
        cols = [base + s for s in suffixes if base + s in df.columns]
        if cols:
            ser = pd.Series(np.nan, index=df.index)
            for c in cols:
                ser = ser.where(~ser.isna(), pd.to_numeric(df[c], errors='coerce'))
            df[base] = ser
    if 'ticker' not in df.columns:
        raise KeyError("Kolom 'ticker' tidak ditemukan pada raw merged.")
    df['ticker'] = df['ticker'].astype(str).str.upper()
    return df.sort_values(['ticker', 'date']).drop_duplicates(['ticker', 'date'])


def apply_filters(df: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    for c in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    allowed = {str(t).strip().upper() for t in tickers if str(t).strip()}
    if allowed:
        df = df[df['ticker'].isin(allowed)].copy()
    df = df[df['Volume'].fillna(0) > 0].copy()
    return df.sort_values(['ticker', 'date']).copy()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    eps = 1e-8
    for ticker, g in df.groupby('ticker', sort=True):
        g = g.sort_values('date').copy()
        g = g.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
        prev_close = g['close'].shift(1)
        ma5 = g['close'].rolling(5, min_periods=5).mean()
        ma20 = g['close'].rolling(20, min_periods=20).mean()
        std20 = g['close'].rolling(20, min_periods=20).std()
        g['log_return_1d'] = np.log(g['close'] / prev_close.replace(0, np.nan))
        g['vol_20'] = g['log_return_1d'].rolling(20, min_periods=20).std()
        g['rsi_14'] = compute_rsi(g['close'], 14)
        g['ma_5_div_ma_20'] = ma5 / (ma20 + eps)
        g['bb_width_20'] = ((ma20 + 2 * std20) - (ma20 - 2 * std20)) / (ma20 + eps)
        g['gap_return_1d'] = np.log(g['open'] / prev_close.replace(0, np.nan))
        g['intraday_range_pct'] = (g['high'] - g['low']) / (g['close'] + eps)
        out.append(g[KEEP_COLS])
    result = pd.concat(out, ignore_index=True)
    need = [c for c in KEEP_COLS if c not in {'ticker', 'date', 'close', 'volume'}]
    return result.dropna(subset=need).sort_values(['ticker', 'date']).reset_index(drop=True)


def main():
    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(f'File tidak ditemukan: {RAW_PATH}')
    cfg = load_yaml(CFG_PATH)
    tickers = list(cfg.get('tickers', []))
    raw = normalize_raw(pd.read_csv(RAW_PATH))
    end_date = parse_end_date(cfg.get('end_date'))
    if end_date is not None:
        raw = raw[raw['date'] <= end_date].copy()
    filt = apply_filters(raw, tickers)
    result = add_indicators(filt)

    forbidden = {'BBCA.JK', 'UNVR.JK'} & set(result['ticker'].astype(str).str.upper().unique())
    if forbidden:
        raise ValueError(f'Ticker terlarang masih ada di prices_with_indicators.csv: {sorted(forbidden)}')

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    result.to_csv(OUT_PATH, index=False)
    print('[INFO] Kolom yang disimpan di prices_with_indicators.csv:')
    print(result.columns.tolist())
    print(f'\n[INFO] Shape akhir: {result.shape}')
    print(f'[INFO] Saving prices with indicators to {OUT_PATH}')
    print('[INFO] Done.')


if __name__ == '__main__':
    main()
