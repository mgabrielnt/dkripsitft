"""Labeling multi-sumber untuk berita per artikel.

Langkah utama (per artikel dan per ticker):
1. Bangun `text_for_label` (judul + lead/description + kalimat relevan) lalu
   klasifikasi L_text via GPT 3-kelas (NEGATIF/NETRAL/POSITIF).
2. Hitung skor leksikon (S_lex) + normalisasi, kalibrasi τ dari distribusi skor,
   lalu turunkan L_lex {-1,0,+1}.
3. Hitung abnormal return berbasis harga t-1 vs t+1 per ticker, kalibrasi θ,
   lalu turunkan L_market {-1,0,+1}.
4. Gabungkan dengan majority rule → L_final, dan skor sentiment_conf {0,1,2}.

Output: data/processed/news_with_sentiment_per_article.csv
berisi teks untuk labeling + semua label (L_text, L_market, L_lex, L_final)
serta indikator sinyal kuat.
"""
import math
import os

import pandas as pd
import yaml
from tqdm import tqdm

from src.utils.gpt_client import classify_sentiment

# Lokasi root project
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_INTERIM_DIR = os.path.join(ROOT_DIR, "data", "interim")
DATA_PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")
CONFIG_GPT_PATH = os.path.join(ROOT_DIR, "configs", "gpt_sentiment.yaml")
CONFIG_SENTIMENT_PATH = os.path.join(ROOT_DIR, "configs", "sentiment.yaml")
PRICES_PATH = os.path.join(DATA_INTERIM_DIR, "prices_with_indicators.csv")

os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)

SRC_PATH = os.path.join(DATA_INTERIM_DIR, "news_clean.csv")
OUT_PATH = os.path.join(DATA_PROCESSED_DIR, "news_with_sentiment_per_article.csv")


# ---------------------------------------------------------------------------
# Helpers untuk load config
# ---------------------------------------------------------------------------
def load_score_mapping() -> dict[str, int]:
    with open(CONFIG_GPT_PATH, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg["score_mapping"]


def load_sentiment_config() -> dict:
    with open(CONFIG_SENTIMENT_PATH, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Helper teks & leksikon
# ---------------------------------------------------------------------------
def build_text_for_sentiment(row: pd.Series) -> str:
    """Bangun teks relevan sesuai spesifikasi unit labeling.

    - Utamakan title + description/lead.
    - Kalau kosong, fallback ke text_clean.
    - Disiapkan per (ticker, artikel), sehingga cocok untuk berita multi-emiten.
    """

    title = row.get("title_clean") or row.get("title") or ""
    desc = row.get("description_clean") or row.get("description") or ""

    if pd.isna(title):
        title = ""
    if pd.isna(desc):
        desc = ""

    text_parts: list[str] = []
    if str(title).strip():
        text_parts.append(str(title).strip())
    if str(desc).strip():
        text_parts.append(str(desc).strip())

    if text_parts:
        text = ". ".join(text_parts)
    else:
        text = row.get("text_clean", "")
        if pd.isna(text):
            text = ""

    return str(text)


def tokenize(text: str) -> list[str]:
    return [t for t in str(text).lower().split() if t]


def compute_lex_score(text: str, pos_words: set[str], neg_words: set[str]) -> float:
    tokens = tokenize(text)
    if not tokens:
        return 0.0

    score = 0
    for tok in tokens:
        if tok in pos_words:
            score += 1
        if tok in neg_words:
            score -= 1

    return score / math.sqrt(len(tokens))


# ---------------------------------------------------------------------------
# Helper pasar & threshold
# ---------------------------------------------------------------------------
def shift_to_next_monday(d: pd.Timestamp) -> pd.Timestamp:
    if pd.isna(d):
        return d
    wd = d.weekday()
    if wd >= 5:
        return d + pd.Timedelta(days=7 - wd)
    return d


def calibrate_threshold(values: pd.Series, method: str, primary: float, secondary: float) -> tuple[float, float]:
    values = values.dropna().abs()
    if values.empty:
        return 0.0, 0.0

    if method == "std":
        std = values.std()
        return primary * std, secondary * std

    # default pakai quantile
    return (
        float(values.quantile(primary)),
        float(values.quantile(secondary)),
    )


def prepare_market_lookup(prices: pd.DataFrame, cfg: dict) -> tuple[dict[tuple[str, str], float], float, float]:
    if prices.empty:
        return {}, 0.0, 0.0

    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)

    prices["close_prev"] = prices.groupby("ticker")["close"].shift(1)
    prices["close_next"] = prices.groupby("ticker")["close"].shift(-1)

    prices["return_window"] = (prices["close_next"] - prices["close_prev"]) / prices["close_prev"]

    benchmark_col = cfg.get("benchmark_column")
    if benchmark_col and benchmark_col in prices.columns:
        prices["benchmark_return"] = prices[benchmark_col]
    else:
        prices["benchmark_return"] = prices.groupby("date")["return_window"].transform("mean")

    prices["abnormal_return"] = prices["return_window"] - prices["benchmark_return"].fillna(0.0)

    theta1, theta2 = calibrate_threshold(
        prices["abnormal_return"],
        cfg.get("threshold_method", "quantile"),
        float(cfg.get("theta_q1", 0.75)),
        float(cfg.get("theta_q2", 0.9)),
    )

    lookup: dict[tuple[str, str], float] = {}
    for _, row in prices.iterrows():
        if pd.isna(row.get("abnormal_return")):
            continue
        lookup[(row["ticker"], str(pd.to_datetime(row["date"]).date()))] = float(row["abnormal_return"])

    return lookup, theta1, theta2


def label_market(ar: float, theta1: float) -> int:
    if pd.isna(ar):
        return 0
    if ar >= theta1:
        return 1
    if ar <= -theta1:
        return -1
    return 0


def label_lex(score: float, tau1: float) -> int:
    if score >= tau1:
        return 1
    if score <= -tau1:
        return -1
    return 0


def compute_final_and_conf(row: pd.Series, theta1: float, tau1: float) -> tuple[int, int]:
    labels = [row.get("l_text", 0), row.get("l_market", 0), row.get("l_lex", 0)]
    pos_votes = sum(1 for x in labels if x > 0)
    neg_votes = sum(1 for x in labels if x < 0)

    if pos_votes >= 2:
        final = 1
    elif neg_votes >= 2:
        final = -1
    else:
        final = 0

    strong_sig = bool(row.get("strong_market_signal", False) or row.get("strong_lex_signal", False))
    conf = 1 if (pos_votes >= 2 or neg_votes >= 2) else 0

    if labels[0] == labels[1] == labels[2] != 0 and strong_sig:
        conf = 2

    if row.get("l_text", 0) == 0 and abs(row.get("abnormal_return", 0.0)) < theta1 and abs(row.get("lex_score_norm", 0.0)) < tau1:
        conf = 0

    return final, conf


# ---------------------------------------------------------------------------
# Pipeline utama
# ---------------------------------------------------------------------------
def main():
    if not os.path.exists(SRC_PATH):
        raise FileNotFoundError(f"Tidak ditemukan: {SRC_PATH}")

    if not os.path.exists(PRICES_PATH):
        raise FileNotFoundError(f"Tidak ditemukan data harga: {PRICES_PATH}")

    print(f"[INFO] Loading {SRC_PATH}")
    df = pd.read_csv(SRC_PATH, parse_dates=["date"])

    score_map = load_score_mapping()
    sent_cfg = load_sentiment_config()

    pos_words = set(sent_cfg.get("lexicon", {}).get("positive_words", []))
    neg_words = set(sent_cfg.get("lexicon", {}).get("negative_words", []))

    # Siapkan lookup abnormal return dan threshold pasar
    print(f"[INFO] Loading prices for market signal from {PRICES_PATH}")
    prices = pd.read_csv(PRICES_PATH)
    market_lookup, theta1, theta2 = prepare_market_lookup(prices, sent_cfg.get("market", {}))
    print(f"[INFO] θ1 (market)={theta1:.4f}, θ2 (market strong)={theta2:.4f}")

    # Cache GPT kalau sudah pernah melabel
    cache: dict[tuple, tuple[str, int, str]] = {}
    reused = 0
    new_calls = 0

    if os.path.exists(OUT_PATH):
        print(f"[INFO] Detected existing {OUT_PATH}, building cache (only L_text)...")
        df_old = pd.read_csv(OUT_PATH, parse_dates=["date"])
        if "l_text" in df_old.columns:
            for _, row in df_old.iterrows():
                key = (
                    row.get("link", ""),
                    row.get("ticker", ""),
                )
                if key not in cache:
                    cache[key] = (
                        row.get("gpt_label", row.get("l_text_label", "NETRAL")),
                        int(row.get("l_text", 0)),
                        row.get("text_for_label", ""),
                    )
            print(f"[INFO] Cache size: {len(cache)} entries")
        else:
            print("[WARN] Cache dilewati karena format lama (5-level) terdeteksi.")

    l_text_labels: list[str] = []
    l_text_scores: list[int] = []
    text_for_labels: list[str] = []

    print(f"[INFO] Melabeli {len(df)} berita (L_text) dengan GPT 3-level + caching...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        key = (
            row.get("link", ""),
            row.get("ticker", ""),
        )

        if key in cache:
            label_str, label_num, cached_text = cache[key]
            reused += 1
            text_for_label = cached_text
        else:
            text_for_label = build_text_for_sentiment(row)
            label_str = classify_sentiment(text_for_label)
            label_num = int(score_map.get(label_str, 0))
            cache[key] = (label_str, label_num, text_for_label)
            new_calls += 1

        l_text_labels.append(label_str)
        l_text_scores.append(label_num)
        text_for_labels.append(text_for_label)

        # jeda kecil opsional (tidak dipakai supaya cepat)

    df["text_for_label"] = text_for_labels
    df["l_text_label"] = l_text_labels
    df["l_text"] = l_text_scores
    df["gpt_label"] = l_text_labels  # kompatibilitas lama
    df["gpt_score"] = l_text_scores

    # Lexicon score
    lex_scores = [compute_lex_score(t, pos_words, neg_words) for t in text_for_labels]
    df["lex_score_norm"] = lex_scores

    tau1, tau2 = calibrate_threshold(
        pd.Series(lex_scores),
        sent_cfg.get("lexicon", {}).get("threshold_method", "quantile"),
        float(sent_cfg.get("lexicon", {}).get("tau_q1", 0.7)),
        float(sent_cfg.get("lexicon", {}).get("tau_q2", 0.9)),
    )
    print(f"[INFO] τ1 (lex)={tau1:.4f}, τ2 (lex strong)={tau2:.4f}")

    df["l_lex"] = df["lex_score_norm"].apply(lambda s: label_lex(float(s), tau1))
    df["strong_lex_signal"] = df["lex_score_norm"].abs() >= tau2 if tau2 > 0 else False

    # Market label: ambil abnormal return berdasarkan tanggal (shift weekend -> Senin)
    df["event_date"] = df["date"].apply(shift_to_next_monday)
    df["abnormal_return"] = df.apply(
        lambda r: market_lookup.get((r.get("ticker", ""), str(pd.to_datetime(r.get("event_date")).date())), 0.0),
        axis=1,
    )
    df["l_market"] = df["abnormal_return"].apply(lambda ar: label_market(float(ar), theta1))
    df["strong_market_signal"] = df["abnormal_return"].abs() >= theta2 if theta2 > 0 else False

    # Final label + confidence
    finals_conf = df.apply(lambda r: compute_final_and_conf(r, theta1, tau1), axis=1)
    df["l_final"], df["sentiment_conf"] = zip(*finals_conf)

    print(f"[INFO] Reused from cache: {reused}, new GPT calls: {new_calls}")
    print(f"[INFO] Saving to {OUT_PATH}")
    df.to_csv(OUT_PATH, index=False)
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
