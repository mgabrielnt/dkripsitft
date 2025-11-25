#!/usr/bin/env bash
set -euo pipefail

# Jalankan seluruh pipeline: ambil data harga & berita -> pra-proses -> labeling sentimen ->
# bangun dataset -> train -> evaluasi.
# Semua perintah dijalankan dari root repo.

# 1) Ambil berita
python -m src.data.fetch_news_rss_google
python -m src.data.fetch_news_yahoo
python -m src.data.merge_news_sources
python -m src.data.preprocess_news_text

# 2) Ambil harga dan hitung indikator teknikal
python -m src.data.download_prices_yahoo
python -m src.data.compute_technical_indicators
python -m src.data.check_price_calendar

# 3) Labeling sentimen dan agregasi
python -m src.data.gpt_sentiment_labeling
python -m src.data.aggregate_daily_sentiment

# 4) Bangun dataset TFT (gabung sentimen + harga)
python -m src.data.build_tft_master_dataset

# 5) Training TFT baseline & hybrid
python -m src.models.train_tft_baseline
python -m src.models.train_tft_with_sentiment

# 6) Evaluasi & backtest sederhana
python -m src.models.evaluate_tft_models
python -m src.models.evaluate_tft_backtest
