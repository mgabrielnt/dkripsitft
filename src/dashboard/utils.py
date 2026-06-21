import re
import pandas as pd

DROP_NAMES = {"n", "count", "jumlah", "split", "index", "unnamed0", "unnamed"}

def norm(text):
    return re.sub(r"[^a-z0-9]+", "", str(text).lower().replace("²", "2"))

def fmt(value, decimals=0):
    if value is None or pd.isna(value):
        return "-"
    try:
        if decimals == 0:
            return f"{float(value):,.0f}".replace(",", ".")
        return f"{float(value):,.{decimals}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return str(value)

def date_col(df):
    if df is None:
        return None
    names = {"date", "targetdate", "publishedat", "publishdate", "datetime", "timestamp"}
    return next((col for col in df.columns if norm(col) in names), None)

def find_col(df, candidates):
    if df is None:
        return None
    mapping = {norm(col): col for col in df.columns}
    for candidate in candidates:
        if norm(candidate) in mapping:
            return mapping[norm(candidate)]
    return next((col for col in df.columns if any(norm(c) in norm(col) or norm(col) in norm(c) for c in candidates)), None)

def find_contains_col(df, include, exclude=None):
    if df is None:
        return None
    exclude = exclude or []
    for col in df.columns:
        low = norm(col)
        if all(norm(i) in low for i in include) and not any(norm(e) in low for e in exclude):
            return col
    return None

def filter_df(df, ticker, dates):
    if df is None:
        return None
    out = df.copy()
    if ticker and "ticker" in out.columns:
        out = out[out["ticker"].astype(str).eq(str(ticker))]
    dc = date_col(out)
    if dc and dates and len(dates) == 2:
        start, end = pd.to_datetime(dates[0]), pd.to_datetime(dates[1])
        out = out[(out[dc] >= start) & (out[dc] <= end)]
    return out
