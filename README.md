# dkripsitft

Pipeline ringkas untuk labeling sentimen multi-sumber dan membangun dataset TFT.

## Prasyarat
- Python 3.10+
- Dependency utama: `pandas`, `pyyaml`, `feedparser`, `yfinance`, `requests`,
  `beautifulsoup4` (install via `pip install -r requirements.txt` atau langsung sesuai kebutuhan).
- Variabel lingkungan `OPENAI_API_KEY` terisi (untuk L_text via GPT). Jika ingin hanya
  menjalankan agregasi tanpa GPT, pastikan `data/processed/news_with_sentiment_per_article.csv`
  sudah ada terlebih dahulu.
- File input yang dibutuhkan (akan dihasilkan otomatis jika menjalankan pipeline lengkap):
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

0. **Scrape arsip HTML (deep history 5 tahun)** — sumber Tempo, CNN, CNBC, Kumparan, Kompas
   ```bash
   python -m src.data.scrape_html_archives
   ```

1. **Ambil berita mentah (RSS + Google News)** — butuh `configs/rss.yaml` dan `configs/yahoo_news.yaml`
   ```bash
   python -m src.data.fetch_news_rss_google
   python -m src.data.fetch_news_yahoo
   python -m src.data.merge_news_sources
   python -m src.data.preprocess_news_text
   ```
   Catatan sumber RSS yang sudah diaktifkan di `configs/rss.yaml`:
   - Tempo Nasional/Bisnis, CNN Indonesia Ekonomi/Nasional, CNBC Indonesia News/Market,
     Kompas, Kumparan, Kontan, Yahoo Finance per ticker BBCA/BBRI.
   - Google News query termasuk `site:` untuk Kontan/Kompas serta bahasa Inggris/Indonesia
     dengan lookback 5 tahun.

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

### Referensi cepat semua modul (jalan dari root repo)
Jika ingin menjalankan setiap file secara manual dalam urutan lengkap, gunakan
perintah berikut. Jalankan di direktori root repo (contoh: `D:\skripsi\tft`).

```bash
# 0) Scrape arsip HTML (opsional tapi disarankan untuk memperbanyak histori)
python -m src.data.scrape_html_archives

# 1) Ambil & gabung berita (RSS, Google News, Yahoo Finance)
python -m src.data.fetch_news_rss_google
python -m src.data.fetch_news_yahoo
python -m src.data.merge_news_sources
python -m src.data.preprocess_news_text

# 2) Ambil harga dan indikator
python -m src.data.download_prices_yahoo
python -m src.data.compute_technical_indicators
python -m src.data.check_price_calendar          # opsional

# 3) Labeling sentimen per artikel + agregasi harian
python -m src.data.gpt_sentiment_labeling
python -m src.data.aggregate_daily_sentiment

# 4) Konversi skala (jika butuh label 5 kelas lama)
python -m src.data.convert_sentiment_scale       # opsional

# 5) Bangun master dataset TFT
python -m src.data.build_tft_master_dataset

# 6) Analisis fitur
python -m src.analysis.compute_vif_features

# 7) Training model
python -m src.models.train_tft_baseline
python -m src.models.train_tft_with_sentiment

# 8) Evaluasi & interpretasi
python -m src.models.evaluate_tft_models
python -m src.models.evaluate_tft_backtest
python -m src.models.analyze_news_effect         # dampak berita ke harga
python -m src.models.interpret_tft_models        # interpretasi fitur/attention

# 9) Utilitas eksperimen
python -m src.utils.update_experiments_best_ckpt # sinkronisasi checkpoint terbaik
```

## Testing cepat
Untuk memastikan modul bisa di-import, jalankan:

```bash
python -m compileall src
```

## Lokasi rumus & aturan sentimen
Semua aturan NEGATIF/NETRAL/POSITIF dan tiga sumber label (L_text, L_market, L_lex) sudah
diterapkan di kode berikut:

* **Kelas polaritas & prompt GPT** – mapping {-1,0,+1} ada di `configs/gpt_sentiment.yaml`.
* **Konfigurasi threshold** – θ (market) dan τ (lexicon) beserta daftar kata keuangan ada di
  `configs/sentiment.yaml`.
* **Unit teks per emiten** – konstruksi `text_for_label` (judul + lead/kalimat relevan) dan
  label GPT/lexicon/market + voting mayoritas → `L_final` & `sentiment_conf` ada di
  `src/data/gpt_sentiment_labeling.py`.
* **Agregasi harian untuk TFT** – rata-rata `L_text`, `L_market`, `L_lex`, `L_final`, hitung
  `news_count`, `pos/neg/neu`, sinyal kuat, rolling/shock, dan ekspor ke
  `data/processed/daily_sentiment.csv` ada di `src/data/aggregate_daily_sentiment.py`.

Dengan menjalankan pipeline di atas, dataset TFT akan memuat fitur sentimen observasi harian
berbasis ketiga sumber tersebut.
