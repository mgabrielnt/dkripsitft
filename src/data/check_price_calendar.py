import os
import pandas as pd

# Lokasi root project (sesuaikan dengan strukturmu)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_INTERIM_DIR = os.path.join(ROOT_DIR, "data", "interim")

PRICES_WITH_IND_PATH = os.path.join(DATA_INTERIM_DIR, "prices_with_indicators.csv")

# Output laporan (opsional, biar bisa dicek di Excel)
MISSING_BDAYS_PATH = os.path.join(DATA_INTERIM_DIR, "missing_business_days.csv")
SUSPICIOUS_GAPS_PATH = os.path.join(DATA_INTERIM_DIR, "suspicious_gaps_over_3days.csv")


def load_prices_with_indicators() -> pd.DataFrame:
    """Load prices_with_indicators.csv dan pastikan kolom date sudah datetime."""
    if not os.path.exists(PRICES_WITH_IND_PATH):
        raise FileNotFoundError(f"File tidak ditemukan: {PRICES_WITH_IND_PATH}")

    print(f"[INFO] Loading: {PRICES_WITH_IND_PATH}")
    df = pd.read_csv(PRICES_WITH_IND_PATH, parse_dates=["date"])

    # Buat aman: sort dan buang duplikat jika ada
    df = df.sort_values(["ticker", "date"]).drop_duplicates(
        subset=["ticker", "date"], keep="first"
    )

    return df


def add_diff_days(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tambah kolom diff_days:
    - selisih (dalam hari) antara tanggal sekarang dan tanggal sebelumnya
      per ticker.
    """
    df = df.sort_values(["ticker", "date"]).copy()
    df["diff_days"] = (
        df.groupby("ticker")["date"].diff().dt.days
    )  # baris pertama per ticker = NaN
    return df


def summarize_diff_days(df: pd.DataFrame):
    """
    Tampilkan ringkasan distribusi diff_days per ticker di terminal.
    Ini berguna buat lihat:
    - 1 hari  -> normal
    - 3 hari  -> Jumat ke Senin (weekend)
    - >3 hari -> libur panjang / data hilang
    """
    print("\n================ RINGKASAN diff_days PER TICKER ================")
    for ticker, g in df.groupby("ticker"):
        print(f"\n=== {ticker} ===")
        vc = g["diff_days"].value_counts().sort_index()
        print(vc)


def find_suspicious_gaps(df: pd.DataFrame, threshold: int = 3) -> pd.DataFrame:
    """
    Ambil baris-baris di mana diff_days > threshold (default: 3 hari).
    Ini kandidat 'loncatan aneh' yang perlu dicek manual:
    - bisa libur panjang bursa,
    - bisa juga data bolong.
    """
    suspicious = df[df["diff_days"] > threshold].copy()
    suspicious = suspicious[["ticker", "date", "diff_days"]].sort_values(
        ["ticker", "date"]
    )
    return suspicious


def find_missing_business_days(df: pd.DataFrame) -> pd.DataFrame:
    """
    Untuk setiap ticker, buat kalender hari kerja (Mon–Fri) dari tanggal min–max,
    lalu cari tanggal hari kerja yang tidak muncul di data.

    Catatan:
    - Ini tidak tahu libur bursa resmi, hanya tahu weekend.
    - Jadi tanggal 'hilang' bisa:
      * benar-benar libur bursa (Lebaran, Natal, dll) -> OK
      * atau data kamu yang bolong                 -> perlu dicek.
    """
    rows = []

    for ticker, g in df.groupby("ticker"):
        g = g.sort_values("date")
        start = g["date"].min()
        end = g["date"].max()

        # Kalender hari kerja (business days) antara start–end
        all_bdays = pd.date_range(start, end, freq="B")  # Mon–Fri

        existing_dates = set(g["date"].dt.normalize().unique())
        bdays_set = set(all_bdays.normalize())

        missing = sorted(bdays_set - existing_dates)

        print(
            f"[INFO] {ticker}: "
            f"total hari kerja seharusnya = {len(all_bdays)}, "
            f"data ada = {len(g)}, "
            f"hari kerja hilang = {len(missing)}"
        )

        for d in missing:
            rows.append({"ticker": ticker, "missing_date": d})

    missing_df = pd.DataFrame(rows)
    if not missing_df.empty:
        missing_df = missing_df.sort_values(["ticker", "missing_date"])

    return missing_df


def main():
    # 1. Load data
    df = load_prices_with_indicators()

    print("\n[INFO] Kolom yang tersedia:")
    print(df.columns.tolist())

    print("\n[INFO] Daftar ticker:")
    print(df["ticker"].unique())

    # 2. Tambah diff_days
    df = add_diff_days(df)

    # 3. Ringkas diff_days per ticker (print ke terminal)
    summarize_diff_days(df)

    # 4. Cari gap 'mencurigakan' (loncat lebih dari 3 hari)
    suspicious = find_suspicious_gaps(df, threshold=3)
    if suspicious.empty:
        print("\n[INFO] Tidak ada gap > 3 hari. Kalender lumayan rapi 🙌")
    else:
        print(
            f"\n[WARN] Ditemukan {len(suspicious)} baris dengan diff_days > 3. "
            f"Detail disimpan ke: {SUSPICIOUS_GAPS_PATH}"
        )
        suspicious.to_csv(SUSPICIOUS_GAPS_PATH, index=False)

    # 5. Cari hari kerja (Mon–Fri) yang hilang dari data
    missing_bdays = find_missing_business_days(df)
    if missing_bdays.empty:
        print("\n[INFO] Tidak ada hari kerja yang hilang dari data (menurut kalender Mon–Fri).")
    else:
        print(
            f"\n[WARN] Ditemukan {len(missing_bdays)} tanggal hari kerja "
            f"yang tidak ada di data. Detail disimpan ke: {MISSING_BDAYS_PATH}"
        )
        missing_bdays.to_csv(MISSING_BDAYS_PATH, index=False)

    print("\n[INFO] Selesai cek kalender harga.\n")

    df = pd.read_csv("data/interim/prices_with_indicators.csv")

    print(df.isna().sum())   # cek jumlah NaN per kolom




if __name__ == "__main__":
    main()
