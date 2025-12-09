import os
import re
import html
import unicodedata
from typing import List, Optional

import pandas as pd

# =============================================================================
# KONFIGURASI GLOBAL
# =============================================================================

# Lokasi root project (dua level di atas file ini)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

DATA_RAW_NEWS_DIR = os.path.join(ROOT_DIR, "data", "raw", "news")
DATA_INTERIM_DIR = os.path.join(ROOT_DIR, "data", "interim")

os.makedirs(DATA_INTERIM_DIR, exist_ok=True)

# Input yang mungkin tersedia
NEWS_RAW_ALL_PATH = os.path.join(DATA_RAW_NEWS_DIR, "news_raw_all_sources.csv")
NEWS_RAW_GOOGLE_PATH = os.path.join(DATA_RAW_NEWS_DIR, "news_raw_google_rss.csv")

# Output
OUT_PATH = os.path.join(DATA_INTERIM_DIR, "news_clean.csv")

# Hanya pakai bahasa ini (bisa diubah kalau nanti mau tambah "en" dll)
ALLOWED_LANGUAGES = {"id"}

# -----------------------------------------------------------------------------
# KAMUS KEYWORD PER EMITEN & PASAR
# -----------------------------------------------------------------------------
# TICKER_COMPANY_KEYWORDS:
#   - Nama resmi emiten (PT ..., Bank ..., dsb.)
#   - Singkatan yang sangat khas (BBCA, BRI, Telkom, Astra, Unilever)
#
# TICKER_SECTOR_KEYWORDS:
#   - Brand produk / anak usaha yang kuat mewakili emiten tersebut.
#   - HANYA akan dianggap relevan jika ada konteks finansial (FINANCE_KEYWORDS).
#
# FINANCE_KEYWORDS:
#   - Istilah pasar modal, laporan keuangan, makroekonomi yang umum muncul di
#     berita saham Indonesia (IHSG, emiten, dividen, laba bersih, OJK, BI rate, dll.)
# -----------------------------------------------------------------------------

TICKER_COMPANY_KEYWORDS = {
    "BBCA.JK": [
        "bbca",
        "bank central asia",
        "pt bank central asia",
        "bca",
        "bank bca",
        "grup bca",
    ],
    "BBRI.JK": [
        "bbri",
        "bank rakyat indonesia",
        "pt bank rakyat indonesia",
        "bri",
        "bank bri",
    ],
    "BMRI.JK": [
        "bmri",
        "bank mandiri",
        "pt bank mandiri",
        "bank mandiri (persero)",
    ],
    "TLKM.JK": [
        "tlkm",
        "telkom indonesia",
        "pt telkom indonesia",
        "telkom",
        "telkom group",
    ],
    "ASII.JK": [
        "asii",
        "astra international",
        "pt astra international",
        "grup astra",
        "astra",
    ],
    "UNVR.JK": [
        "unvr",
        "unilever indonesia",
        "pt unilever indonesia",
        "pt unilever indonesia tbk",
        "unilever",
    ],
    "MARKET": [
        "ihsg",
        "indeks harga saham gabungan",
        "bursa efek indonesia",
        "bei",
        "idx",
        "pasar modal",
        "pasar saham",
        "bursa saham",
        "emiten",
        "lq45",
        "idx30",
    ],
}

TICKER_SECTOR_KEYWORDS = {
    "BBCA.JK": [
        "kartu kredit bca",
        "debit bca",
        "kpr bca",
        "kpa bca",
        "kku bca",
        "kks bca",
        "bca mobile",
        "klikbca",
        "sakuku",
        "flazz bca",
        "bca digital",
        "mybca",
        "bank",
    ],
    "BBRI.JK": [
        "kur bri",
        "kredit usaha rakyat bri",
        "bri link",
        "brilink",
        "brimo",
        "simpedes",
        "britama",
        "bri life",
        "bri finance",
        "bri remittance",
        "bank",
    ],
    "BMRI.JK": [
        "livin' by mandiri",
        "livin by mandiri",
        "kopra by mandiri",
        "kopra mandiri",
        "mandiri online",
        "mandiri internet bisnis",
        "mandiri debit",
        "kartu kredit mandiri",
        "mandiri sekuritas",
        "mandiri taspen",
        "mandiri taspen pos",
        "bank",
    ],
    "TLKM.JK": [
        "indihome",
        "telkomsel",
        "grapari",
        "neucentrix",
        "telkomsat",
        "mitratel",
        "telkomsigma",
        "telin",
        "telkom infra",
        "telkominfra",
        "telkom akses",
        "metranet",
        "internet",
        "telkom property",
    ],
    "ASII.JK": [
        "astra daihatsu",
        "astra isuzu",
        "astra honda motor",
        "toyota astra",
        "astra toyota",
        "auto2000",
        "united tractors",
        "untr",
        "astra agro lestari",
        "aali",
        "fifgroup",
        "astra credit companies",
        "acc",
        "mobil88",
        "toyota trust",
        "astra financial",
        "astra life",
    ],
    "UNVR.JK": [
        "pepsodent",
        "rinso",
        "lifebuoy",
        "lux",
        "sunsilk",
        "clear",
        "rexona",
        "vaseline",
        "molto",
        "sunlight",
        "super pell",
        "superpell",
        "wipol",
        "dove",
        "pond's",
        "ponds",
        "zwitsal",
        "cif",
        "citra",
        "blue band",
        "royco",
        "jawara",
        "walls",
        "wall's",
    ],
    "MARKET": [
        "saham",
        "obligasi",
        "reksadana",
        "rights issue",
        "ipo",
        "buyback",
        "stock split",
        "dividen",
        "laba bersih",
        "laba rugi",
        "earning per share",
        "eps",
        "book value",
        "kapitalisasi pasar",
        "market cap",
        "valuasi",
        "price to earnings",
        "p e ratio",
    ],
}

FINANCE_KEYWORDS = [
    # Pasar modal & instrumen
    "saham",
    "emiten",
    "ipo",
    "right issue",
    "rights issue",
    "dividen",
    "buyback",
    "stock split",
    "obligasi",
    "obligasi korporasi",
    "reksadana",
    "nav",
    "unit penyertaan",
    "perdagangan saham",
    "perdagangan bursa",
    "bursa efek indonesia",
    "bei",
    "idx",
    "ihsg",
    "indeks harga saham gabungan",
    "sbn",
    "surat berharga negara",
    "yield",
    "pasar modal",
    "pasar saham",
    # Laporan keuangan & rasio
    "laba bersih",
    "laba rugi",
    "pendapatan bunga",
    "aset produktif",
    "aset",
    "liabilitas",
    "ekuitas",
    "margin laba",
    "kinerja keuangan",
    "laporan keuangan",
    "fee based income",
    "pendapatan bunga bersih",
    "non performing loan",
    "npl",
    "rasio kecukupan modal",
    "car",
    "rasio likuiditas",
    "p e ratio",
    "price to book value",
    "pbv",
    "kapitalisasi pasar",
    "market cap",
    "harga penutupan",
    "harga saham",
    "target harga",
    "rekomendasi beli",
    "rekomendasi jual",
    "earning per share",
    "eps",
    "book value",
    # Waktu pelaporan
    "kuartal i",
    "kuartal ii",
    "kuartal iii",
    "kuartal iv",
    "q1",
    "q2",
    "q3",
    "q4",
    "tahun buku",
    # Regulator & suku bunga
    "bank indonesia",
    "bi rate",
    "suku bunga acuan",
    "ojk",
    "otoritas jasa keuangan",
    "lps",
    "lembaga penjamin simpanan",
    # Makroekonomi yang relevan ke pasar keuangan
    "inflasi",
    "deflasi",
    "pdb",
    "produk domestik bruto",
    "pertumbuhan ekonomi",
    "resesi",
    "krisis finansial",
    "krisis keuangan",
    "kurs rupiah",
    "nilai tukar",
    "rupiah menguat",
    "rupiah melemah",
    "ekonomi indonesia",
    "ekonomi global",
    "surplus neraca",
    "defisit neraca",
    "subsidi bbm",
    "harga bbm",
    "suku bunga",
]


# =============================================================================
# FUNGSI UTILITAS
# =============================================================================

def clean_text(text: str) -> str:
    """
    Bersihkan teks dari HTML, URL, whitespace berlebih, dan normalisasi unicode.
    Return string lowercased yang siap dipakai ke model / labeling.
    """
    if not isinstance(text, str):
        return ""

    # Normalisasi unicode (hilangkan karakter aneh, kombinasi, dll.)
    text = unicodedata.normalize("NFKC", text)

    # Decode HTML entity: &nbsp;, &amp;, dll.
    text = html.unescape(text)

    # Hapus URL (http, https, dan www.)
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"www\.[^\s]+", " ", text)

    # Hapus tag HTML (misal <a href=...>, <font>, dll.)
    text = re.sub(r"<[^>]+>", " ", text)

    # Ganti newline / carriage return dengan spasi
    text = text.replace("\n", " ").replace("\r", " ")

    # Hilangkan non-breaking space
    text = text.replace("\xa0", " ")

    # Hilangkan spasi berulang
    text = re.sub(r"\s+", " ", text)

    # Buang spasi di depan/belakang dan lowercase
    return text.strip().lower()


def pick_date_column(df: pd.DataFrame) -> Optional[str]:
    """
    Pilih kolom tanggal terbaik dari kandidat yang tersedia.
    Urutan prioritas:
      1. "date"
      2. "published_dt_utc"
      3. "published_raw"
    """
    candidates = ["date", "published_dt_utc", "published_raw"]
    for col in candidates:
        if col in df.columns:
            return col
    return None


def ensure_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Pastikan setiap kolom dalam 'cols' ada di df.
    Kalau belum ada, buat kolom kosong.
    """
    for col in cols:
        if col not in df.columns:
            df[col] = ""
    return df


def text_contains_any(text: str, keywords: List[str]) -> bool:
    """
    Cek apakah 'text' (sudah lower-case & dibersihkan) mengandung
    salah satu keyword (juga diasumsikan lower-case).
    Dipakai untuk match nama bank, brand, istilah keuangan, dll.
    """
    if not isinstance(text, str) or not text:
        return False

    txt = f" {text.lower()} "
    for kw in keywords:
        kw_l = kw.lower().strip()
        if not kw_l:
            continue
        # Pakai dua cara:
        # 1) pattern dengan spasi (supaya 'bca' tidak match 'bencana')
        # 2) fallback "in" biasa untuk frasa panjang
        if f" {kw_l} " in txt or kw_l in txt:
            return True
    return False


def is_relevant_news(ticker: str, text_clean: str) -> bool:
    """
    Tentukan apakah sebuah berita relevan dengan ticker tertentu.

    Aturan:
    - Untuk emiten individu (BBCA, BBRI, BMRI, TLKM, ASII, UNVR):
        relevan jika:
          * ada nama perusahaan/grup (TICKER_COMPANY_KEYWORDS), ATAU
          * (ada brand/anak usaha sektor (TICKER_SECTOR_KEYWORDS)
             DAN ada konteks finansial/makro (FINANCE_KEYWORDS))
    - Untuk ticker khusus "MARKET":
        relevan jika ada konteks finansial/makro (FINANCE_KEYWORDS)
        atau kata kunci pasar/indeks di TICKER_COMPANY_KEYWORDS/MARKET.
    """
    if not isinstance(ticker, str):
        return False

    t = ticker.strip().upper()
    if not t:
        return False

    text_clean = (text_clean or "").lower()
    if not text_clean:
        return False

    comp_kws = TICKER_COMPANY_KEYWORDS.get(t, [])
    sector_kws = TICKER_SECTOR_KEYWORDS.get(t, [])

    has_company_kw = text_contains_any(text_clean, comp_kws)
    has_sector_kw = text_contains_any(text_clean, sector_kws)
    has_finance_kw = text_contains_any(text_clean, FINANCE_KEYWORDS)

    if t == "MARKET":
        # MARKET mewakili sentimen pasar luas (IHSG, makro, dll.)
        return has_finance_kw or has_company_kw or has_sector_kw

    # Emiten individu: lebih ketat
    return bool(has_company_kw or (has_sector_kw and has_finance_kw))


# =============================================================================
# MAIN CLEANING PIPELINE
# =============================================================================

def main():
    # -------------------------------------------------------------------------
    # 1. Tentukan sumber file input
    # -------------------------------------------------------------------------
    if os.path.exists(NEWS_RAW_ALL_PATH):
        src_path = NEWS_RAW_ALL_PATH
    elif os.path.exists(NEWS_RAW_GOOGLE_PATH):
        src_path = NEWS_RAW_GOOGLE_PATH
    else:
        raise FileNotFoundError(
            f"Tidak ditemukan file berita mentah: "
            f"{NEWS_RAW_ALL_PATH} atau {NEWS_RAW_GOOGLE_PATH}"
        )

    print(f"[INFO] Loading raw news from: {src_path}")

    # Baca CSV tanpa parse_dates dulu (supaya fleksibel kalau nama kolom tanggal beda)
    df = pd.read_csv(src_path)

    print(f"[INFO] Raw shape : {df.shape[0]} rows, {df.shape[1]} columns")
    if "ticker" in df.columns:
        print("[INFO] Sample ticker counts:\n", df["ticker"].value_counts().head())

    # -------------------------------------------------------------------------
    # 2. Pastikan kolom penting minimal ada
    # -------------------------------------------------------------------------
    df = ensure_columns(df, ["title", "description", "ticker"])

    # Bersihkan tipe data & strip whitespace dasar
    df["title"] = df["title"].fillna("").astype(str).str.strip()
    df["description"] = df["description"].fillna("").astype(str).str.strip()
    df["ticker"] = df["ticker"].fillna("").astype(str).str.strip().str.upper()

    # -------------------------------------------------------------------------
    # 3. Normalisasi kolom tanggal -> df["date"]
    # -------------------------------------------------------------------------
    date_col = pick_date_column(df)
    if date_col is None:
        raise ValueError(
            "Tidak menemukan kolom tanggal yang cocok. "
            "Harus ada salah satu dari: 'date', 'published_dt_utc', 'published_raw'."
        )

    print(f"[INFO] Menggunakan kolom tanggal: {date_col}")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df["date"] = df[date_col].dt.date  # simpan sebagai date (YYYY-MM-DD)

    before = len(df)
    df = df[~df["date"].isna()].copy()
    print(f"[INFO] Drop baris tanpa tanggal valid : {before} -> {len(df)}")

    # -------------------------------------------------------------------------
    # 4. Filter berdasarkan bahasa (kalau kolom 'language' tersedia)
    # -------------------------------------------------------------------------
    if "language" in df.columns and ALLOWED_LANGUAGES:
        before = len(df)
        df = df[df["language"].isin(ALLOWED_LANGUAGES)].copy()
        print(
            f"[INFO] Filter language (hanya {ALLOWED_LANGUAGES}) : "
            f"{before} -> {len(df)}"
        )

    # -------------------------------------------------------------------------
    # 5. Bentuk teks gabungan dan bersihkan
    # -------------------------------------------------------------------------
    df["text_raw"] = (df["title"] + ". " + df["description"]).str.strip()

    df["title_clean"] = df["title"].apply(clean_text)
    df["text_clean"] = df["text_raw"].apply(clean_text)

    # Drop baris yang teks bersihnya kosong (berarti cuma URL / HTML / noise)
    before = len(df)
    df = df[df["text_clean"].str.strip() != ""].copy()
    print(f"[INFO] Drop berita dengan teks kosong : {before} -> {len(df)}")

    # Pastikan ticker tidak kosong
    before = len(df)
    df = df[df["ticker"].str.strip() != ""].copy()
    print(f"[INFO] Drop baris tanpa ticker       : {before} -> {len(df)}")

    # -------------------------------------------------------------------------
    # 6. Filter relevansi berita per ticker (BERDASARKAN KEYWORD)
    # -------------------------------------------------------------------------
    if "ticker" in df.columns and "text_clean" in df.columns:
        before = len(df)
        df["is_relevant"] = df.apply(
            lambda row: is_relevant_news(row["ticker"], row["text_clean"]),
            axis=1,
        )
        df = df[df["is_relevant"]].copy()
        print(f"[INFO] Filter relevansi per ticker : {before} -> {len(df)}")
    else:
        print("[WARN] Kolom 'ticker' atau 'text_clean' tidak ditemukan, skip filter relevansi.")

    # -------------------------------------------------------------------------
    # 7. Dedup (supaya tidak double label / double hitung)
    # -------------------------------------------------------------------------
    before = len(df)
    subset_cols = [c for c in ["link", "ticker", "date", "title_clean"] if c in df.columns]

    if subset_cols:
        df = df.drop_duplicates(subset=subset_cols, keep="first")
        print(
            f"[INFO] Drop duplikat berdasarkan {subset_cols} : "
            f"{before} -> {len(df)}"
        )
        before = len(df)

    # Dedup full row untuk jaga-jaga
    df = df.drop_duplicates()
    print(f"[INFO] Drop duplikat full row              : {before} -> {len(df)}")

    # -------------------------------------------------------------------------
    # 8. Pilih kolom yang disimpan (utama + meta yang berguna)
    # -------------------------------------------------------------------------
    base_cols = [
        "date",
        "ticker",
        "title",
        "description",
        "link",
        "source",
        "title_clean",
        "text_clean",
        "text_raw",
    ]

    meta_cols = [
        "query",
        "query_type",
        "language",
        "source_type",
        "published_raw",
        "published_dt_utc",
        "is_relevant",
    ]

    keep_cols = [c for c in base_cols + meta_cols if c in df.columns]
    df = df[keep_cols].sort_values(["ticker", "date", "title_clean"])

    print(f"[INFO] Final shape : {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"[INFO] Columns     : {keep_cols}")

    # -------------------------------------------------------------------------
    # 9. Simpan ke CSV
    # -------------------------------------------------------------------------
    print(f"[INFO] Saving cleaned news to: {OUT_PATH}")
    df.to_csv(OUT_PATH, index=False)
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
