# file: src/data/aggregate_daily_sentiment.py
"""
Agregasi sentimen harian berbasis label multi-sumber {-1,0,+1}.

Pipeline:
1) news_clean.csv (interim) -> python -m src.data.gpt_sentiment_labeling
   menghasilkan data/processed/news_with_sentiment_per_article.csv dengan
   kolom L_text, L_market, L_lex, L_final, sentiment_conf, dst.
2) File ini menggeser tanggal weekend ke Senin, lalu agregasi harian per ticker
   untuk membentuk daily_sentiment.csv.
   Termasuk ticker khusus "MARKET" (berita makro/pasar luas) yang akan muncul
   sebagai satu seri terpisah dengan ticker="MARKET".
3) daily_sentiment.csv dipakai di build_tft_master_dataset.py untuk join ke harga.

Output utama per (ticker, date):
- sentiment_text_mean / market_mean / lex_mean / final_mean
- news_count, pos_count, neg_count, neu_count
- sentiment_conf_mean, sentiment_conf_max, sentiment_conf_sum
- strong_market_count, strong_lex_count
- Fitur yang UTAMA untuk TFT regresi:
    * sentiment_mean_3d       : rata-rata sentimen final 3 hari (rolling)
    * sentiment_mean_5d       : rata-rata sentimen final 5 hari (rolling)
    * sentiment_mean_conf_weighted : sentimen harian berbobot confidence
    * news_count_3d           : jumlah berita 3 hari (rolling)
    * news_count_7d           : jumlah berita 7 hari (rolling)
    * sentiment_vol_7d        : volatilitas sentimen 7 hari (rolling std)
    * rel_sentiment_mean      : sentimen relatif terhadap pasar (MARKET)
    * rel_sentiment_mean_3d   : sentimen 3D relatif terhadap pasar
- Fitur tambahan (opsional tetapi mungkin berguna):
    * sentiment_shock, extreme_news, sentiment_trend_5d
    * sentiment_pos_ratio, sentiment_neg_ratio, sentiment_balance_ratio
    * sentiment_sign, sentiment_majority_sign,
      sentiment_is_bullish_day, sentiment_is_bearish_day
    * high_news_day : hari dengan volume berita sangat tinggi (top 10% per ticker)
"""

import os
from typing import List

import pandas as pd

# Lokasi root project
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_INTERIM_DIR = os.path.join(ROOT_DIR, "data", "interim")
DATA_PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")

SRC_PATH = os.path.join(DATA_PROCESSED_DIR, "news_with_sentiment_per_article.csv")
OUT_PATH = os.path.join(DATA_PROCESSED_DIR, "daily_sentiment.csv")


def shift_to_next_monday(d: pd.Timestamp) -> pd.Timestamp:
    """Geser berita weekend ke hari Senin berikutnya agar selaras dengan hari bursa."""
    if pd.isna(d):
        return d
    wd = d.weekday()  # 0=Mon ... 5=Sat, 6=Sun
    if wd >= 5:
        return d + pd.Timedelta(days=7 - wd)
    return d


def compute_extreme_flag(series: pd.Series, q: float = 0.9) -> pd.Series:
    """Tandai nilai ekstrem berdasarkan quantile absolut."""
    threshold = series.abs().quantile(q)
    if pd.isna(threshold) or threshold == 0:
        threshold = series.abs().max()  # fallback minimal, bisa nol juga
    return (series.abs() > threshold).astype(int)


def main():
    if not os.path.exists(SRC_PATH):
        raise FileNotFoundError(
            f"Tidak ditemukan file input sentimen artikel: {SRC_PATH}. Jalankan gpt_sentiment_labeling dulu."
        )

    print(f"[INFO] Loading {SRC_PATH}")
    df = pd.read_csv(SRC_PATH, parse_dates=["date", "event_date"])

    required_cols = {"ticker", "l_text", "l_market", "l_lex", "l_final", "sentiment_conf"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Kolom wajib hilang di file input: {missing}")

    # Drop duplikat artikel (misal re-run crawling)
    before = len(df)
    subset_cols: List[str] = []
    for c in ["date", "ticker", "title", "link"]:
        if c in df.columns:
            subset_cols.append(c)
    if not subset_cols:
        subset_cols = ["date", "ticker"]

    df = df.drop_duplicates(subset=subset_cols).copy()
    after = len(df)
    print(f"[INFO] Drop duplikat: {before - after} baris dibuang, sisa {after}")

    df["date"] = pd.to_datetime(df["date"])
    df["date_shifted"] = df["date"].apply(shift_to_next_monday)

    # flag has_news per artikel (nantinya di-agg)
    df["has_news"] = 1

    all_daily = []

    for ticker, g in df.groupby("ticker"):
        g = g.copy()

        # Produk l_final * confidence → dipakai untuk weighted mean
        g["l_final_conf_weighted"] = g["l_final"] * g["sentiment_conf"]

        agg_map = dict(
            sentiment_text_mean=("l_text", "mean"),
            sentiment_market_mean=("l_market", "mean"),
            sentiment_lex_mean=("l_lex", "mean"),
            sentiment_final_mean=("l_final", "mean"),
            news_count=("l_final", "size"),
            pos_count=("l_final", lambda x: (x > 0).sum()),
            neg_count=("l_final", lambda x: (x < 0).sum()),
            neu_count=("l_final", lambda x: (x == 0).sum()),
            sentiment_conf_mean=("sentiment_conf", "mean"),
            sentiment_conf_max=("sentiment_conf", "max"),
            sentiment_conf_sum=("sentiment_conf", "sum"),
            has_news=("has_news", "max"),
            # Weighted sum untuk l_final berbobot confidence
            sentiment_conf_weighted_sum=("l_final_conf_weighted", "sum"),
        )

        if "strong_market_signal" in g.columns:
            agg_map["strong_market_count"] = ("strong_market_signal", "sum")
        else:
            agg_map["strong_market_count"] = ("l_final", lambda x: 0)

        if "strong_lex_signal" in g.columns:
            agg_map["strong_lex_count"] = ("strong_lex_signal", "sum")
        else:
            agg_map["strong_lex_count"] = ("l_final", lambda x: 0)

        daily = g.groupby("date_shifted").agg(**agg_map).reset_index()

        if daily.empty:
            continue

        # Reindex ke semua hari kerja antara min & max tanggal (supaya align dengan harga)
        start_date = daily["date_shifted"].min()
        end_date = daily["date_shifted"].max()
        full_idx = pd.bdate_range(start=start_date, end=end_date, freq="B")

        daily = daily.set_index("date_shifted").reindex(full_idx)
        daily.index.name = "date"

        daily["ticker"] = ticker

        # isi kosong dengan 0 (hari tanpa berita)
        sentiment_cols = [
            "sentiment_text_mean",
            "sentiment_market_mean",
            "sentiment_lex_mean",
            "sentiment_final_mean",
            "sentiment_conf_mean",
            "sentiment_conf_weighted_sum",
        ]
        count_cols = [
            "news_count",
            "pos_count",
            "neg_count",
            "neu_count",
            "strong_market_count",
            "strong_lex_count",
            "sentiment_conf_sum",
        ]
        binary_cols = ["has_news"]

        daily[sentiment_cols] = daily[sentiment_cols].fillna(0.0)
        daily[count_cols] = daily[count_cols].fillna(0)
        daily[binary_cols] = daily[binary_cols].fillna(0)

        # ========= FEATURE TURUNAN DASAR =========
        # 1) sentiment_mean & rolling 3 hari (basis untuk TFT)
        daily["sentiment_mean"] = daily["sentiment_final_mean"]
        daily["sentiment_mean_3d"] = daily["sentiment_mean"].rolling(
            window=3, min_periods=1
        ).mean()
        daily["news_count_3d"] = daily["news_count"].rolling(
            window=3, min_periods=1
        ).sum()

        # 1b) rata-rata 5 hari
        daily["sentiment_mean_5d"] = daily["sentiment_mean"].rolling(
            window=5, min_periods=1
        ).mean()

        # 1c) confidence-weighted daily sentiment
        daily["sentiment_mean_conf_weighted"] = daily["sentiment_mean"]
        valid_conf = daily["sentiment_conf_sum"] > 0
        daily.loc[valid_conf, "sentiment_mean_conf_weighted"] = (
            daily.loc[valid_conf, "sentiment_conf_weighted_sum"]
            / daily.loc[valid_conf, "sentiment_conf_sum"]
        )

        # 2) Sentiment shock = deviasi terhadap rata-rata 3 hari sebelumnya (opsional)
        prev3 = daily["sentiment_mean"].shift(1).rolling(window=3, min_periods=1).mean()
        daily["sentiment_shock"] = (daily["sentiment_mean"] - prev3).fillna(0.0)

        # 3) Extreme news: threshold berdasarkan quantile 0.9 abs(shock) per ticker (opsional)
        daily["extreme_news"] = compute_extreme_flag(daily["sentiment_shock"], q=0.9)

        # 4) Volatilitas sentimen 7 hari (rolling std sentiment_mean) → fitur utama
        daily["sentiment_vol_7d"] = (
            daily["sentiment_mean"].rolling(window=7, min_periods=3).std().fillna(0.0)
        )

        # 5) Tren sentimen 5 hari: bandingkan level sekarang dengan rata-rata 5 hari sebelumnya (opsional)
        prev5_mean = daily["sentiment_mean"].shift(1).rolling(
            window=5, min_periods=1
        ).mean()
        daily["sentiment_trend_5d"] = (daily["sentiment_mean"] - prev5_mean).fillna(0.0)

        # 6) Intensitas berita: rolling 7 hari + flag hari dengan volume berita tinggi
        daily["news_count_7d"] = daily["news_count"].rolling(
            window=7, min_periods=1
        ).sum()

        news_with_any = daily.loc[daily["news_count"] > 0, "news_count"]
        if news_with_any.empty:
            daily["high_news_day"] = 0
        else:
            high_q = news_with_any.quantile(0.9)
            if pd.isna(high_q) or high_q <= 0:
                daily["high_news_day"] = 0
            else:
                daily["high_news_day"] = (daily["news_count"] >= high_q).astype(int)

        # 7) Rasio & sinyal polarity harian
        valid_mask = daily["news_count"] > 0
        daily["sentiment_pos_ratio"] = 0.0
        daily["sentiment_neg_ratio"] = 0.0
        daily["sentiment_balance_ratio"] = 0.0

        daily.loc[valid_mask, "sentiment_pos_ratio"] = (
            daily.loc[valid_mask, "pos_count"] / daily.loc[valid_mask, "news_count"]
        )
        daily.loc[valid_mask, "sentiment_neg_ratio"] = (
            daily.loc[valid_mask, "neg_count"] / daily.loc[valid_mask, "news_count"]
        )
        daily.loc[valid_mask, "sentiment_balance_ratio"] = (
            (daily.loc[valid_mask, "pos_count"] - daily.loc[valid_mask, "neg_count"])
            / daily.loc[valid_mask, "news_count"]
        )

        daily["sentiment_sign"] = 0
        daily.loc[daily["sentiment_mean"] > 0, "sentiment_sign"] = 1
        daily.loc[daily["sentiment_mean"] < 0, "sentiment_sign"] = -1

        daily["sentiment_majority_sign"] = 0
        daily.loc[daily["pos_count"] > daily["neg_count"], "sentiment_majority_sign"] = 1
        daily.loc[daily["pos_count"] < daily["neg_count"], "sentiment_majority_sign"] = -1

        daily["sentiment_is_bullish_day"] = (daily["sentiment_mean"] > 0).astype(int)
        daily["sentiment_is_bearish_day"] = (daily["sentiment_mean"] < 0).astype(int)

        all_daily.append(daily.reset_index())

    if not all_daily:
        print("[WARN] Tidak ada data harian setelah agregasi.")
        empty_cols = [
            "date",
            "ticker",
            "sentiment_text_mean",
            "sentiment_market_mean",
            "sentiment_lex_mean",
            "sentiment_final_mean",
            "news_count",
            "pos_count",
            "neg_count",
            "neu_count",
            "sentiment_conf_mean",
            "sentiment_conf_max",
            "sentiment_conf_sum",
            "strong_market_count",
            "strong_lex_count",
            "sentiment_mean",
            "sentiment_mean_3d",
            "sentiment_mean_5d",
            "sentiment_mean_conf_weighted",
            "news_count_3d",
            "news_count_7d",
            "has_news",
            "sentiment_shock",
            "extreme_news",
            "sentiment_vol_7d",
            "sentiment_trend_5d",
            "high_news_day",
            "sentiment_pos_ratio",
            "sentiment_neg_ratio",
            "sentiment_balance_ratio",
            "sentiment_sign",
            "sentiment_majority_sign",
            "sentiment_is_bullish_day",
            "sentiment_is_bearish_day",
        ]
        pd.DataFrame(columns=empty_cols).to_csv(OUT_PATH, index=False)
        print(f"[INFO] Saving empty daily_sentiment to {OUT_PATH}")
        return

    df_daily = pd.concat(all_daily, ignore_index=True)

    # ========= Fitur relatif terhadap pasar (ticker "MARKET") =========
    if "ticker" in df_daily.columns and "date" in df_daily.columns:
        market_mask = df_daily["ticker"] == "MARKET"
        if market_mask.any():
            market_daily = df_daily.loc[
                market_mask,
                ["date", "sentiment_mean", "sentiment_mean_3d", "news_count_3d"],
            ].rename(
                columns={
                    "sentiment_mean": "market_sentiment_mean",
                    "sentiment_mean_3d": "market_sentiment_mean_3d",
                    "news_count_3d": "market_news_count_3d",
                }
            )

            df_daily = df_daily.merge(market_daily, on="date", how="left")

            for col in [
                "market_sentiment_mean",
                "market_sentiment_mean_3d",
                "market_news_count_3d",
            ]:
                if col in df_daily.columns:
                    df_daily[col] = df_daily[col].fillna(0.0)

            df_daily["rel_sentiment_mean"] = (
                df_daily["sentiment_mean"] - df_daily["market_sentiment_mean"]
            )
            df_daily["rel_sentiment_mean_3d"] = (
                df_daily["sentiment_mean_3d"] - df_daily["market_sentiment_mean_3d"]
            )
            df_daily["rel_news_count_3d"] = (
                df_daily["news_count_3d"] - df_daily["market_news_count_3d"]
            )

            is_market = df_daily["ticker"] == "MARKET"
            df_daily.loc[
                is_market,
                ["rel_sentiment_mean", "rel_sentiment_mean_3d", "rel_news_count_3d"],
            ] = 0.0

    # ========= Final safety fill & sort =========
    sentiment_cols_all = [
        "sentiment_text_mean",
        "sentiment_market_mean",
        "sentiment_lex_mean",
        "sentiment_final_mean",
        "sentiment_conf_mean",
        "sentiment_conf_max",
        "sentiment_conf_weighted_sum",
        "sentiment_mean",
        "sentiment_mean_3d",
        "sentiment_mean_5d",
        "sentiment_mean_conf_weighted",
        "sentiment_shock",
        "sentiment_vol_7d",
        "sentiment_trend_5d",
        "market_sentiment_mean",
        "market_sentiment_mean_3d",
        "rel_sentiment_mean",
        "rel_sentiment_mean_3d",
        "sentiment_pos_ratio",
        "sentiment_neg_ratio",
        "sentiment_balance_ratio",
    ]
    count_cols_all = [
        "news_count",
        "pos_count",
        "neg_count",
        "neu_count",
        "news_count_3d",
        "news_count_7d",
        "strong_market_count",
        "strong_lex_count",
        "sentiment_conf_sum",
        "market_news_count_3d",
        "rel_news_count_3d",
    ]
    binary_cols_all = [
        "has_news",
        "extreme_news",
        "high_news_day",
        "sentiment_is_bullish_day",
        "sentiment_is_bearish_day",
    ]

    for col in sentiment_cols_all:
        if col in df_daily.columns:
            df_daily[col] = df_daily[col].astype(float).fillna(0.0)
    for col in count_cols_all:
        if col in df_daily.columns:
            df_daily[col] = df_daily[col].fillna(0).astype(int)
    for col in binary_cols_all:
        if col in df_daily.columns:
            df_daily[col] = df_daily[col].fillna(0).astype(int)

    df_daily = df_daily.sort_values(["ticker", "date"]).reset_index(drop=True)

    print(f"[INFO] Total baris harian (setelah shifting weekend): {len(df_daily)}")
    print(f"[INFO] Saving to {OUT_PATH}")
    df_daily.to_csv(OUT_PATH, index=False)
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
