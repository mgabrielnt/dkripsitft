# dkripsitft

Pipeline ringkas untuk labeling sentimen multi-sumber dan membangun dataset TFT.

## Prasyarat
- Python 3.10+
- Variabel lingkungan `OPENAI_API_KEY` terisi (untuk L_text via GPT). Jika ingin hanya
  menjalankan agregasi tanpa GPT, pastikan `data/processed/news_with_sentiment_per_article.csv`
  sudah ada terlebih dahulu.
- File input yang dibutuhkan:
  - `data/interim/news_clean.csv`
  - `data/interim/prices_with_indicators.csv`

## Perintah cepat dari terminal
Jalankan dari root repo (folder ini). Seluruh langkah sudah diringkas dalam skrip
berikut:

```bash
bash scripts/run_sentiment_pipeline.sh
```

Skrip tersebut menjalankan tiga modul secara berurutan:

1. **Label berita per artikel & ticker**
   ```bash
   python -m src.data.gpt_sentiment_labeling
   ```
   Output: `data/processed/news_with_sentiment_per_article.csv` berisi kolom
   `l_text`, `l_market`, `l_lex`, `l_final`, dan `sentiment_conf`.

2. **Agregasi harian**
   ```bash
   python -m src.data.aggregate_daily_sentiment
   ```
   Output: `data/processed/daily_sentiment.csv` dengan agregasi per ticker per hari
   (sudah shifting weekend → Senin).

3. **Bangun master dataset TFT**
   ```bash
   python -m src.data.build_tft_master_dataset
   ```
Output: `data/processed/tft_master.csv` yang siap dipakai untuk training TFT.

## Urutan perintah lengkap: dari pengambilan data sampai evaluasi
Semua dijalankan dari root repo.

1. **Ambil berita mentah** (butuh config `configs/rss.yaml` dan `configs/yahoo_news.yaml`)
   ```bash
   python -m src.data.fetch_news_rss_google
   python -m src.data.fetch_news_yahoo
   python -m src.data.merge_news_sources
   python -m src.data.preprocess_news_text
   ```

2. **Ambil harga & hitung indikator teknikal** (butuh `configs/data.yaml`)
   ```bash
   python -m src.data.download_prices_yahoo
   python -m src.data.compute_technical_indicators
   python -m src.data.check_price_calendar  # opsional untuk audit kalender
   ```

3. **Labeling sentimen per artikel** (butuh `OPENAI_API_KEY`)
   ```bash
   python -m src.data.gpt_sentiment_labeling
   python -m src.data.aggregate_daily_sentiment
   ```

4. **Bangun dataset TFT (gabung harga + sentimen)**
   ```bash
   python -m src.data.build_tft_master_dataset
   ```

5. **Training model**
   ```bash
   python -m src.models.train_tft_baseline
   python -m src.models.train_tft_with_sentiment
   ```

6. **Evaluasi & backtest**
   ```bash
   python -m src.models.evaluate_tft_models
   python -m src.models.evaluate_tft_backtest
   ```

Semua langkah di atas bisa dijalankan sekaligus dengan:

```bash
bash scripts/run_end_to_end.sh
```

## Testing cepat
Untuk memastikan modul bisa di-import, jalankan:

```bash
python -m compileall src
```
