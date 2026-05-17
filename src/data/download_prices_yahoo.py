import os
import pandas as pd
import yfinance as yf
import yaml

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_CFG_PATH = os.path.join(ROOT_DIR, "configs", "data.yaml")
OUT_PATH = os.path.join(ROOT_DIR, "data", "raw", "prices", "prices_all_raw.csv")

def main():
    # Baca config dari data.yaml
    with open(DATA_CFG_PATH, "r") as f:
        cfg = yaml.safe_load(f)

    TICKERS = cfg["tickers"]
    # Dikunci manual ke 2 Jan 2017 sesuai keinginan Anda
    START_DATE = "2017-01-02" 
    END_DATE_CFG = pd.to_datetime(cfg["end_date"]).date()
    END_DATE_YF = (END_DATE_CFG + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    print(f"Mendownload harga {TICKERS} dari {START_DATE} s.d {END_DATE_CFG}...")
    
    # auto_adjust=False untuk mencegah error hilangnya 'Adj Close'
    df = yf.download(TICKERS, start=START_DATE, end=END_DATE_YF, group_by="ticker", auto_adjust=False)
    
    all_data = []
    
    for ticker in TICKERS:
        t_df = df[ticker].copy().reset_index()
        t_df.rename(columns={'Date': 'date'}, inplace=True)
        t_df['date'] = pd.to_datetime(t_df['date']).dt.date
        t_df.insert(1, 'ticker', ticker)
        
        # HAPUS BARIS KOSONG: Buang semua tanggal merah / akhir pekan yang tidak ada harganya
        t_df = t_df.dropna(subset=['Close'])
        
        all_data.append(t_df)

    final_df = pd.concat(all_data, ignore_index=True)
    cols = ['date', 'ticker', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    final_df = final_df[cols].sort_values(['ticker', 'date'])
    
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    final_df.to_csv(OUT_PATH, index=False)
    print(f"Selesai! {len(final_df)} baris data hari bursa aktif tersimpan di {OUT_PATH}")

if __name__ == "__main__":
    main()