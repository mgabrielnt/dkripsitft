import os
import pandas as pd
import yaml

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")
SRC_PATH = os.path.join(DATA_PROCESSED_DIR, "news_with_sentiment_per_article.csv")
OUT_PATH = os.path.join(DATA_PROCESSED_DIR, "daily_sentiment.csv")
CONFIG_DATA_PATH = os.path.join(ROOT_DIR, "configs", "data.yaml")

KEEP_COLS = [
    "date",
    "ticker",
    "sentiment_final_mean",
    "news_count_3d",
    "sentiment_mean_3d",
    "sentiment_ema_7d",
    "sentiment_trend_7d",
]


def load_data_cfg() -> dict:
    with open(CONFIG_DATA_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_allowed_tickers() -> set[str]:
    cfg = load_data_cfg()
    return {str(t).strip().upper() for t in cfg.get("tickers", []) if str(t).strip()}


def parse_end_date(raw):
    if raw is None:
        return None
    dt = pd.to_datetime(raw, errors="coerce")
    return None if pd.isna(dt) else dt.normalize()


def shift_to_next_monday(d: pd.Timestamp) -> pd.Timestamp:
    if pd.isna(d):
        return d
    return d + pd.Timedelta(days=7 - d.weekday()) if d.weekday() >= 5 else d


def build_daily_features(g: pd.DataFrame) -> pd.DataFrame:
    daily = (
        g.groupby("date_shifted", as_index=False)
        .agg(
            sentiment_final_mean=("l_final", "mean"),
            news_count=("l_final", "size"),
        )
        .rename(columns={"date_shifted": "date"})
        .sort_values("date")
    )

    if daily.empty:
        return pd.DataFrame(columns=["date"] + [c for c in KEEP_COLS if c != "date"])

    idx = pd.bdate_range(daily["date"].min(), daily["date"].max())
    daily = (
        daily.set_index("date")
        .reindex(idx)
        .rename_axis("date")
        .reset_index()
    )

    daily[["sentiment_final_mean", "news_count"]] = (
        daily[["sentiment_final_mean", "news_count"]].fillna(0.0)
    )

    daily["sentiment_mean_3d"] = (
        daily["sentiment_final_mean"].rolling(3, min_periods=1).mean()
    )
    daily["news_count_3d"] = (
        daily["news_count"].rolling(3, min_periods=1).sum()
    )
    daily["sentiment_ema_7d"] = (
        daily["sentiment_final_mean"].ewm(span=7, adjust=False).mean()
    )

    prior_mean = daily["sentiment_final_mean"].shift(1).rolling(7, min_periods=1).mean()
    daily["sentiment_trend_7d"] = (
        daily["sentiment_final_mean"] - prior_mean
    ).fillna(0.0)

    return daily


def main():
    if not os.path.exists(SRC_PATH):
        raise FileNotFoundError(f"File tidak ditemukan: {SRC_PATH}")

    df = pd.read_csv(SRC_PATH, parse_dates=["date"])

    required_cols = {"date", "ticker", "l_final"}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(
            f"Kolom wajib tidak ditemukan di news_with_sentiment_per_article.csv: {sorted(missing)}"
        )

    cfg = load_data_cfg()
    end_date = parse_end_date(cfg.get("end_date"))
    allowed_tickers = load_allowed_tickers()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["ticker"] = df["ticker"].astype(str).str.upper()
    df["l_final"] = pd.to_numeric(df["l_final"], errors="coerce")

    if allowed_tickers:
        df = df[df["ticker"].isin(allowed_tickers)].copy()

    if "link" in df.columns:
        df["link"] = df["link"].astype(str).str.strip()
    if "text_for_label" in df.columns:
        df["text_for_label"] = df["text_for_label"].astype(str).str.strip()

    df = df.dropna(subset=["date", "ticker", "l_final"]).copy()

    if end_date is not None:
        df = df[df["date"] <= end_date].copy()

    dedup_cols = ["ticker", "date"]
    if "link" in df.columns:
        dedup_cols.append("link")
    elif "text_for_label" in df.columns:
        dedup_cols.append("text_for_label")

    df = df.drop_duplicates(subset=dedup_cols, keep="last").copy()
    df["date_shifted"] = df["date"].apply(shift_to_next_monday)

    out = []
    for ticker, g in df.groupby("ticker", sort=True):
        daily = build_daily_features(g.copy())
        if daily.empty:
            continue
        daily["ticker"] = ticker
        out.append(daily[KEEP_COLS])

    if not out:
        raise ValueError("Data kosong setelah agregasi sentimen.")

    result = (
        pd.concat(out, ignore_index=True)
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )

    if end_date is not None:
        result = result[result["date"] <= end_date].copy()
    if allowed_tickers:
        result = result[result["ticker"].isin(allowed_tickers)].copy()

    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
    result.to_csv(OUT_PATH, index=False)

    print(f"saved -> {OUT_PATH}")
    print(result.columns.tolist())


if __name__ == "__main__":
    main()
