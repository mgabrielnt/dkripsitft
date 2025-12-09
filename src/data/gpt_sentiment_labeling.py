# file: src/data/gpt_sentiment_labeling.py

import math
import os
from typing import Dict, Tuple, List

import pandas as pd
import yaml
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.utils.gpt_client import classify_sentiment, TICKER_TO_COMPANY, MODEL_NAME

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
# Helpers config
# ---------------------------------------------------------------------------
def load_score_mapping() -> Dict[str, int]:
    with open(CONFIG_GPT_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg["score_mapping"]


def load_sentiment_config() -> dict:
    with open(CONFIG_SENTIMENT_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Helper teks & leksikon
# ---------------------------------------------------------------------------
def build_text_for_sentiment(row: pd.Series) -> str:
    """
    Ambil teks yang akan dikirim ke GPT:

    - Utamakan title_clean + description_clean
    - Kalau kosong, fallback ke text_clean
    """
    title = row.get("title_clean") or row.get("title") or ""
    desc = row.get("description_clean") or row.get("description") or ""

    if pd.isna(title):
        title = ""
    if pd.isna(desc):
        desc = ""

    text_parts: List[str] = []
    t = str(title).strip()
    d = str(desc).strip()

    if t:
        text_parts.append(t)
    if d:
        text_parts.append(d)

    if text_parts:
        text = ". ".join(text_parts)
    else:
        text = row.get("text_clean", "")
        if pd.isna(text):
            text = ""

    return str(text)


def tokenize(text: str) -> List[str]:
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

    # Normalisasi sederhana supaya skala tidak meledak di teks panjang
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


def calibrate_threshold(
    values: pd.Series, method: str, primary: float, secondary: float
) -> Tuple[float, float]:
    """
    Kalibrasi tau/theta:

    - method == "std"    → primary * std, secondary * std
    - method == "quantile" (default) → quantile(|x|)

    primary = batas signifikan, secondary = batas sinyal kuat.
    """
    values = values.dropna().abs()
    if values.empty:
        return 0.0, 0.0

    if method == "std":
        std = values.std()
        return primary * std, secondary * std

    return float(values.quantile(primary)), float(values.quantile(secondary))


def prepare_market_lookup(
    prices: pd.DataFrame, cfg: dict
) -> Tuple[Dict[Tuple[str, str], float], float, float]:
    if prices.empty:
        return {}, 0.0, 0.0

    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)

    prices["close_prev"] = prices.groupby("ticker")["close"].shift(1)
    prices["close_next"] = prices.groupby("ticker")["close"].shift(-1)

    prices["return_window"] = (prices["close_next"] - prices["close_prev"]) / prices[
        "close_prev"
    ]

    benchmark_col = cfg.get("benchmark_column")
    if benchmark_col and benchmark_col in prices.columns:
        prices["benchmark_return"] = prices[benchmark_col]
    else:
        prices["benchmark_return"] = prices.groupby("date")["return_window"].transform(
            "mean"
        )

    prices["abnormal_return"] = (
        prices["return_window"] - prices["benchmark_return"].fillna(0.0)
    )

    theta1, theta2 = calibrate_threshold(
        prices["abnormal_return"],
        cfg.get("threshold_method", "quantile"),
        float(cfg.get("theta_q1", 0.75)),
        float(cfg.get("theta_q2", 0.9)),
    )

    lookup: Dict[Tuple[str, str], float] = {}
    for _, row in prices.iterrows():
        if pd.isna(row.get("abnormal_return")):
            continue
        lookup[(row["ticker"], str(pd.to_datetime(row["date"]).date()))] = float(
            row["abnormal_return"]
        )

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


def compute_final_and_conf(row: pd.Series, theta1: float, tau1: float) -> Tuple[int, int]:
    """
    Gabungkan L_text (GPT), L_market, L_lex menjadi L_final dan sentiment_conf.

    PERBAIKAN PENTING (berdasarkan hasil data kamu):
    - GPT (l_text) tetap sinyal utama, tapi TIDAK LAGI boleh membuat terlalu banyak NEGATIF.
    - Kalau GPT bilang NEGATIF, label final NEGATIF hanya diberikan kalau:
        * market atau lexicon juga NEGATIF (l_market == -1 atau l_lex == -1), atau
        * ada strong_market_signal / strong_lex_signal.
      Kalau tidak ada dukungan → diturunkan ke NETRAL.
    - GPT POSITIF tetap dipercaya selama tidak dibantah keras oleh market/lex.
    - GPT NETRAL → pakai kombinasi market + lex seperti sebelumnya.
    """
    l_text = int(row.get("l_text", 0))
    l_market = int(row.get("l_market", 0))
    l_lex = int(row.get("l_lex", 0))

    strong_market = bool(row.get("strong_market_signal", False))
    strong_lex = bool(row.get("strong_lex_signal", False))

    ar = float(row.get("abnormal_return", 0.0))
    lex_score = float(row.get("lex_score_norm", 0.0))

    final = 0
    conf = 0

    # Sinyal searah / berlawanan dari market & lex (diskrit)
    evidence_neg = (l_market == -1) or (l_lex == -1)
    evidence_pos = (l_market == 1) or (l_lex == 1)

    # 1) GPT NEGATIF → sekarang wajib ada bukti tambahan
    if l_text == -1:
        if evidence_neg:
            # GPT negatif + market/lex negatif → boleh NEGATIF
            final = -1
            conf = 2 if (strong_market or strong_lex) else 1
        elif evidence_pos:
            # GPT negatif tapi market/lex positif → kompromi netral
            final = 0
            conf = 1
        else:
            # GPT sendirian tanpa dukungan → downgrade jadi netral
            final = 0
            conf = 0

    # 2) GPT POSITIF → tetap anchor utama, tapi boleh dikoreksi jika data sangat negatif
    elif l_text == 1:
        if evidence_pos and not evidence_neg:
            # GPT positif + market/lex searah
            final = 1
            conf = 2 if (strong_market or strong_lex) else 1
        elif evidence_neg and not evidence_pos:
            # Market/lex jelas negatif, GPT positif → kompromi netral
            final = 0
            conf = 1
        else:
            # GPT sendirian tapi tidak dibantah keras → tetap positif
            final = 1
            conf = 1

    # 3) GPT NETRAL → gunakan kombinasi market + lex (mirip desain awal)
    else:
        if l_market == 0 and l_lex == 0:
            final = 0
            conf = 0
        else:
            if l_market == l_lex != 0:
                # Keduanya searah dan non-zero
                final = l_market
                conf = 2 if (strong_market or strong_lex) else 1
            else:
                # Hanya salah satu yang non-zero → prioritas market, lalu lex
                if l_market != 0:
                    final = l_market
                else:
                    final = l_lex
                conf = 1

    # Jika final tetap netral dan kedua sinyal sangat lemah → turunkan confidence
    if final == 0 and abs(ar) < theta1 and abs(lex_score) < tau1:
        conf = 0

    return final, conf


# ---------------------------------------------------------------------------
# Helper relevansi emiten vs teks
# ---------------------------------------------------------------------------

def is_relevant_to_ticker(text: str, ticker: str, row: pd.Series) -> bool:
    """
    Berita yang sudah masuk ke data/interim/news_clean.csv SUDAH difilter
    di src/data/preprocess_news_text.py menggunakan kamus keyword per-ticker
    (termasuk ticker khusus "MARKET").

    Di tahap labeling GPT, kita tidak melakukan filter ulang yang berbeda,
    supaya:
    - tidak ada berita relevan yang ter-drop hanya karena aturan keyword ganda,
    - jumlah baris yang dilabeli GPT 1:1 dengan isi news_clean.csv,
    - cache news_with_sentiment_per_article.csv tetap konsisten.

    Fungsi ini dipertahankan demi kompatibilitas kode, tetapi selalu
    mengembalikan True.
    """
    return True


# ---------------------------------------------------------------------------
# MAIN
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

    print(f"[INFO] Loading prices for market signal from {PRICES_PATH}")
    prices = pd.read_csv(PRICES_PATH)
    market_lookup, theta1, theta2 = prepare_market_lookup(
        prices, sent_cfg.get("market", {})
    )
    print(f"[INFO] θ1 (market)={theta1:.4f}, θ2 (market strong)={theta2:.4f}")

    # ---------------- CACHE dari file lama ----------------
    cache: Dict[Tuple[str, str], Tuple[str, int, str]] = {}
    df_old: pd.DataFrame | None = None

    if os.path.exists(OUT_PATH):
        print(f"[INFO] Detected existing {OUT_PATH}, building cache (only L_text)...")
        df_old = pd.read_csv(OUT_PATH, parse_dates=["date"])
        if "l_text" in df_old.columns:
            for _, row in df_old.iterrows():
                # Kalau di run sebelumnya berita dianggap TIDAK relevan,
                # jangan dimasukkan ke cache supaya bisa dihitung ulang dengan aturan baru.
                if row.get("relevant_to_ticker") is False:
                    continue

                key = (row.get("link", ""), row.get("ticker", ""))
                if key not in cache:
                    cache[key] = (
                        row.get("gpt_label", row.get("l_text_label", "NETRAL")),
                        int(row.get("l_text", 0)),
                        row.get("text_for_label", ""),
                    )
            print(f"[INFO] Cache size: {len(cache)} entries")
        else:
            print("[WARN] Cache dilewati karena format lama terdeteksi (tanpa l_text).")

    # ---------------- Siapkan teks, relevansi, dan cache reuse ----------------
    n = len(df)
    texts_for_label: List[str] = [""] * n
    l_text_labels: List[str] = ["NETRAL"] * n
    l_text_scores: List[int] = [0] * n
    relevant_flags: List[bool] = [False] * n

    indices_need_gpt: List[int] = []
    reused_from_cache = 0

    print(f"[INFO] Menyiapkan teks & menggunakan cache untuk {n} berita...")

    for i, row in tqdm(df.iterrows(), total=n):
        text_for_label = build_text_for_sentiment(row)
        texts_for_label[i] = text_for_label

        ticker = str(row.get("ticker", ""))
        is_rel = is_relevant_to_ticker(text_for_label, ticker, row)
        relevant_flags[i] = is_rel

        key = (row.get("link", ""), ticker)

        if not is_rel:
            # Berita tidak relevan dengan emiten → paksa NETRAL, skip GPT
            l_text_labels[i] = "NETRAL"
            l_text_scores[i] = 0
            continue

        # Relevan, coba pakai cache lama
        if key in cache:
            label_str, label_num, cached_text = cache[key]
            if cached_text:
                texts_for_label[i] = cached_text
            l_text_labels[i] = label_str
            l_text_scores[i] = label_num
            reused_from_cache += 1
        else:
            indices_need_gpt.append(i)

    print(f"[INFO] Artikel lama (pakai cache GPT): {reused_from_cache}")
    print(f"[INFO] Artikel baru (perlu GPT): {len(indices_need_gpt)}")

    # ---------------- Label GPT untuk artikel baru (paralel) ----------------
    if indices_need_gpt:
        with open(CONFIG_GPT_PATH, "r", encoding="utf-8") as f:
            gpt_cfg = yaml.safe_load(f)
        max_workers = int(gpt_cfg.get("max_workers", 3))

        print(
            f"[INFO] Melabeli berita baru dengan {MODEL_NAME} (max_workers={max_workers})..."
        )

        def _worker(idx: int) -> Tuple[int, str]:
            text = texts_for_label[idx]
            row = df.iloc[idx]
            ticker = str(row.get("ticker", "") or "")
            company = TICKER_TO_COMPANY.get(ticker, "")
            label = classify_sentiment(text, ticker=ticker, company=company)
            return idx, label

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_worker, idx): idx for idx in indices_need_gpt}

            for fut in tqdm(
                as_completed(futures), total=len(futures), desc="GPT labeling"
            ):
                idx = futures[fut]
                try:
                    i, label_str = fut.result()
                except Exception as e:
                    print(f"[WARN] GPT worker error pada index {idx}: {e}")
                    i = idx
                    label_str = "NETRAL"

                label_num = int(score_map.get(label_str, 0))
                l_text_labels[i] = label_str
                l_text_scores[i] = label_num

    # Isi ke dataframe
    df["text_for_label"] = texts_for_label
    df["relevant_to_ticker"] = relevant_flags
    df["l_text_label"] = l_text_labels
    df["l_text"] = l_text_scores
    df["gpt_label"] = l_text_labels
    df["gpt_score"] = l_text_scores

    # Lexicon score
    lex_scores = [
        compute_lex_score(t, pos_words, neg_words) if t else 0.0
        for t in texts_for_label
    ]
    df["lex_score_norm"] = lex_scores

    # Kalibrasi threshold lexicon HANYA dari berita relevan (lebih bersih)
    lex_series_for_calib = df.loc[df["relevant_to_ticker"], "lex_score_norm"]
    tau1, tau2 = calibrate_threshold(
        lex_series_for_calib,
        sent_cfg.get("lexicon", {}).get("threshold_method", "quantile"),
        float(sent_cfg.get("lexicon", {}).get("tau_q1", 0.7)),
        float(sent_cfg.get("lexicon", {}).get("tau_q2", 0.9)),
    )
    print(f"[INFO] τ1 (lex)={tau1:.4f}, τ2 (lex strong)={tau2:.4f}")

    df["l_lex"] = df["lex_score_norm"].apply(lambda s: label_lex(float(s), tau1))
    df["strong_lex_signal"] = df["lex_score_norm"].abs() >= tau2 if tau2 > 0 else False

    # Market label (event-study ke return harga)
    df["event_date"] = df["date"].apply(shift_to_next_monday)
    df["abnormal_return"] = df.apply(
        lambda r: market_lookup.get(
            (r.get("ticker", ""), str(pd.to_datetime(r.get("event_date")).date())),
            0.0,
        ),
        axis=1,
    )
    df["l_market"] = df["abnormal_return"].apply(
        lambda ar: label_market(float(ar), theta1)
    )
    df["strong_market_signal"] = (
        df["abnormal_return"].abs() >= theta2 if theta2 > 0 else False
    )

    # Final label & confidence (dengan logika baru)
    finals_conf = df.apply(lambda r: compute_final_and_conf(r, theta1, tau1), axis=1)
    df["l_final"], df["sentiment_conf"] = zip(*finals_conf)

    # Info distribusi label untuk debugging
    try:
        dist = df["l_final"].value_counts().sort_index()
        print("[INFO] Distribusi l_final (per artikel):")
        print(dist.to_string())
        print("[INFO] Proporsi l_final:")
        print((dist / dist.sum()).round(4).to_string())

        ct = pd.crosstab(df["gpt_label"], df["l_final"]).sort_index()
        print("[INFO] Crosstab gpt_label vs l_final:")
        print(ct.to_string())
    except Exception:
        pass

    # ---------------- Merge incremental dengan file lama ----------------
    if df_old is not None:
        subset_cols = ["ticker", "date", "link"]

        df["date"] = pd.to_datetime(df["date"]).dt.date
        df_old["date"] = pd.to_datetime(df_old["date"]).dt.date

        df_all = pd.concat([df_old, df], ignore_index=True)
        before = len(df_all)
        df_all = df_all.drop_duplicates(subset=subset_cols, keep="last")
        after = len(df_all)
        print(f"[INFO] Merge incremental news_with_sentiment: {before} -> {after}")
    else:
        df_all = df

    df_all = df_all.sort_values(["ticker", "date"]).reset_index(drop=True)

    print(f"[INFO] Saving to {OUT_PATH}")
    df_all.to_csv(OUT_PATH, index=False)
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
