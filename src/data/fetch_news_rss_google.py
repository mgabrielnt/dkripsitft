import os, time, urllib.parse, yaml
from datetime import timedelta
import feedparser, pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CFG_PATH = os.path.join(ROOT_DIR, "configs", "rss.yaml")
DATA_CFG_PATH = os.path.join(ROOT_DIR, "configs", "data.yaml")
OUT_PATH = os.path.join(ROOT_DIR, "data", "raw", "news", "news_raw_all_sources.csv")
BASE_COLS = ["date", "ticker", "language", "title", "description", "link", "source"]

def get_news_per_month(query, ticker, start_dt, end_dt):
    all_rows, cursor = [], start_dt
    
    while cursor <= end_dt:
        # Tentukan tanggal akhir untuk bulan yang sedang berjalan
        next_month = cursor.replace(day=28) + timedelta(days=4)
        end_of_month = next_month - timedelta(days=next_month.day)
        chunk_end = min(end_of_month, end_dt)
        
        print(f"  -> {ticker}: Mengambil data {cursor.strftime('%b %Y')} ({cursor} s/d {chunk_end})")
        
        # Format query ke Google (mundur 1 hari untuk toleransi timezone)
        after_str = (cursor - timedelta(days=1)).isoformat()
        before_str = (chunk_end + timedelta(days=1)).isoformat()
        q = f'{query} after:{after_str} before:{before_str}'
        
        url = "https://news.google.com/rss/search?" + urllib.parse.urlencode({"q": q, "hl": "id-ID", "gl": "ID"})
        
        for e in getattr(feedparser.parse(url), "entries", []):
            pub = pd.to_datetime(e.get("published", ""), errors="coerce")
            # Filter ketat: Hanya simpan berita yang pas di bulan/periode tersebut
            if pd.isna(pub) or pub.date() < cursor or pub.date() > chunk_end: continue
            
            all_rows.append({
                "date": pub.date(), "ticker": ticker, "language": "id",
                "title": e.get("title", ""), "description": e.get("summary", ""),
                "link": str(e.get("link", "")).strip(), "source": e.get("source", {}).get("title", "GoogleNews")
            })
            
        cursor = chunk_end + timedelta(days=1)
        time.sleep(1.5) # Jeda antar bulan agar tidak terkena limit
        
    return all_rows

def main():
    with open(CFG_PATH, "r") as f: rss_cfg = yaml.safe_load(f)
    with open(DATA_CFG_PATH, "r") as f: data_cfg = yaml.safe_load(f)
    
    START_DATE = pd.to_datetime("2017-01-02").date()
    END_DATE = pd.to_datetime(data_cfg["end_date"]).date()
    
    all_rows = []
    print(f"Scraping berita BERTILIK PER BULAN dari {START_DATE} s/d {END_DATE}...\n")
    
    for q in rss_cfg.get("queries", []):
        if q.get("type") == "google":
            print(f"Mulai query untuk {q['ticker']}...")
            all_rows.extend(get_news_per_month(q["query"], q["ticker"], START_DATE, END_DATE))
            
    df = pd.DataFrame(all_rows, columns=BASE_COLS).drop_duplicates(subset=["ticker", "link", "date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"\nSelesai! {len(df)} baris berita tersimpan ke {OUT_PATH}")

if __name__ == "__main__":
    main()