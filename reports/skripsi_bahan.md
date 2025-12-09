# Bahan Skripsi (Ringkasan Repo)

Dokumen ini merangkum struktur repo dan poin penting untuk penulisan skripsi.

## Struktur Direktori
- `configs/`: konfigurasi data, model, sentiment, RSS/Google News, eksperimen.
- `src/data/`: pipeline pengambilan harga, berita, labeling GPT, agregasi sentimen, dan pembuatan dataset TFT.
- `src/models/`: training TFT baseline & hybrid, evaluasi, backtest, interpretasi.
- `src/analysis/`: analisis fitur (VIF, diagnostik).
- `src/utils/`: klien GPT dan utilitas update checkpoint.
- `data/`: hasil interim (harga + indikator, berita bersih).
- `models/`: checkpoint TFT baseline & hybrid.
- `reports/`: skor korelasi/feature filter dan grafik residual.

## Konfigurasi Kunci
- `configs/data.yaml`: tickers BEI (BBCA.JK, BBRI.JK, TLKM.JK, ASII.JK, BMRI.JK, UNVR.JK), start_date 2017-01-01, horizon 5, split 70/15/15.
- `configs/model_tft.yaml`: target `close`, encoder 60 hari, horizon 5; baseline hidden_size 64 dropout 0.1; hybrid hidden_size 128 dropout 0.2; batch_size 64, lr 5e-4, early stopping patience 10, accelerator gpu.
- `configs/rss.yaml`: query Google News per ticker (termasuk site:kontan), start 2017-01-01, market RSS (Kontan Keuangan, CNBC Market, CNN Ekonomi) dengan ticker MARKET.
- `configs/gpt_sentiment.yaml`: model gpt-5-nano, label {NEGATIF, NETRAL, POSITIF} dengan skor {-1,0,1}, max_completion_tokens 64, max_chars 2000, max_workers 5.
- `configs/sentiment.yaml`: threshold market quantile (theta_q1=0.75, theta_q2=0.9) dan lexicon threshold std (tau_q1=0.5, tau_q2=1.0), daftar kata positif/negatif.
- `configs/experiments.yaml`: checkpoint utama baseline `tft-baseline-epoch=09-val_loss=377.0987.ckpt`, hybrid `tft-with-sentiment-epoch=04-val_loss=303.5286.ckpt`.

## Pipeline Data Singkat
1. `src/data/download_prices_yahoo.py`: unduh OHLCV harian via yfinance sesuai tickers & start_date.
2. `src/data/compute_technical_indicators.py`: bersihkan harga, hitung indikator (log_return_1d, vol_20, rsi_14, ma_5_div_ma_20, bb_width_20, volume_ma_ratio_20, price_zscore_20, volume_zscore_20, return_mean_5d, return_std_5d, intraday_range_pct, atr_14, gap_return_1d, lag close_2/3).
3. `src/data/fetch_news_rss_google.py` + `fetch_news_yahoo.py` + `merge_news_sources.py` + `preprocess_news_text.py`: ambil berita Google News/RSS/Yahoo, bersihkan teks & mapping ticker.
4. `src/data/gpt_sentiment_labeling.py`: bangun text_for_label (title+desc), klasifikasi GPT → l_text; hitung lexicon & market abnormal return; voting compute_final_and_conf → l_final, sentiment_conf; simpan `news_with_sentiment_per_article.csv`.
5. `src/data/aggregate_daily_sentiment.py`: shift weekend ke Senin, agregasi per ticker per hari (sentiment_mean, sentiment_mean_3d/5d, news_count_3d/7d, sentiment_vol_7d, sentiment_trend_5d, sentiment_shock, extreme_news, rel_sentiment_* dll.) → `daily_sentiment.csv`.
6. `src/data/build_tft_master_dataset.py`: merge harga+sentimen, tambah fitur kalender, hitung time_idx & split train/val/test → `tft_master.csv`.

## Model & Eksperimen
- **Baseline** (`src/models/train_tft_baseline.py`): hanya fitur teknikal whitelist, target `close`; GroupNormalizer per ticker; encoder 60, horizon 5; Trainer dengan EarlyStopping(val_loss) dan checkpoint terbaik.
- **Hybrid** (`src/models/train_tft_with_sentiment.py`): fitur teknikal + fitur sentimen whitelist (sentiment_mean, sentiment_mean_3d, sentiment_vol_7d, sentiment_trend_5d, news_count, news_count_3d, sentiment_shock, extreme_news); opsi sentiment raw/sign, clipping outlier 99.5%; hidden_size 128 dropout 0.2.
- Evaluasi: `src/models/evaluate_tft_models.py` untuk prediksi test multi-horizon; `evaluate_tft_backtest.py` & `evaluate_tft_backtest_full.py` untuk rolling backtest; `interpret_tft_models.py` untuk importance/attention; `analyze_news_effect.py` untuk dampak berita.

## Artefak Hasil
- Checkpoint di `models/tft_baseline/` dan `models/tft_with_sentiment/` sesuai experiments.yaml.
- Laporan fitur di `reports/feature_filter_scores.csv`, `feature_target_correlation.csv`, grafik residual & true-vs-pred di `reports/figures/`.
- Interim data: `data/interim/prices_with_indicators.csv`, `data/interim/news_clean.csv`; gap kalender di `missing_business_days.csv`, `suspicious_gaps_over_3days.csv`.

