# Bahan Lanjutan Skripsi (Detail Teknis)

## B. DETAIL DATASET & PREPROCESSING

### B.1 Ringkasan Dataset `tft_master.csv`

> Catatan: file `data/processed/tft_master.csv` **tidak ada di repo** sehingga statistik numerik perlu dihasilkan dengan menjalankan pipeline (`python -m src.data.build_tft_master_dataset`).

- Konfigurasi target dan cakupan data:
  - Ticker: BBCA.JK, BBRI.JK, TLKM.JK, ASII.JK, BMRI.JK, UNVR.JK (lihat: `configs/data.yaml`).
  - Start date harga/berita: 2017-01-01 (lihat: `configs/data.yaml`, `configs/rss.yaml`).
  - Horizon prediksi: 5 hari bursa (`max_prediction_length=5`).
- Status ketersediaan: TIDAK TERSEDIA DI REPO – jalankan pipeline untuk membuat `tft_master.csv`.

**Tabel B.1 – Ringkasan Dataset tft_master (perlu dieksekusi ulang)**

| Ticker | Jumlah Baris | Tanggal Awal | Tanggal Akhir |
|--------|--------------|--------------|---------------|
| BBCA.JK | TIDAK TERSEDIA DI REPO | TIDAK TERSEDIA DI REPO | TIDAK TERSEDIA DI REPO |
| BBRI.JK | TIDAK TERSEDIA DI REPO | TIDAK TERSEDIA DI REPO | TIDAK TERSEDIA DI REPO |
| TLKM.JK | TIDAK TERSEDIA DI REPO | TIDAK TERSEDIA DI REPO | TIDAK TERSEDIA DI REPO |
| ASII.JK | TIDAK TERSEDIA DI REPO | TIDAK TERSEDIA DI REPO | TIDAK TERSEDIA DI REPO |
| BMRI.JK | TIDAK TERSEDIA DI REPO | TIDAK TERSEDIA DI REPO | TIDAK TERSEDIA DI REPO |
| UNVR.JK | TIDAK TERSEDIA DI REPO | TIDAK TERSEDIA DI REPO | TIDAK TERSEDIA DI REPO |

**Tabel B.2 – Statistik Deskriptif Fitur Utama (perlu dihitung setelah dataset dibuat)**

| Fitur | Mean | Std | Min | Max |
|-------|------|-----|-----|-----|
| close | TIDAK TERSEDIA DI REPO | TIDAK TERSEDIA DI REPO | TIDAK TERSEDIA DI REPO | TIDAK TERSEDIA DI REPO |
| log_return_1d | TIDAK TERSEDIA DI REPO | TIDAK TERSEDIA DI REPO | TIDAK TERSEDIA DI REPO | TIDAK TERSEDIA DI REPO |
| rsi_14 | TIDAK TERSEDIA DI REPO | TIDAK TERSEDIA DI REPO | TIDAK TERSEDIA DI REPO | TIDAK TERSEDIA DI REPO |
| vol_20 | TIDAK TERSEDIA DI REPO | TIDAK TERSEDIA DI REPO | TIDAK TERSEDIA DI REPO | TIDAK TERSEDIA DI REPO |
| bb_width_20 | TIDAK TERSEDIA DI REPO | TIDAK TERSEDIA DI REPO | TIDAK TERSEDIA DI REPO | TIDAK TERSEDIA DI REPO |
| volume_ma_ratio_20 | TIDAK TERSEDIA DI REPO | TIDAK TERSEDIA DI REPO | TIDAK TERSEDIA DI REPO | TIDAK TERSEDIA DI REPO |
| return_std_5d | TIDAK TERSEDIA DI REPO | TIDAK TERSEDIA DI REPO | TIDAK TERSEDIA DI REPO | TIDAK TERSEDIA DI REPO |
| sentiment_mean | TIDAK TERSEDIA DI REPO | TIDAK TERSEDIA DI REPO | TIDAK TERSEDIA DI REPO | TIDAK TERSEDIA DI REPO |
| sentiment_mean_3d | TIDAK TERSEDIA DI REPO | TIDAK TERSEDIA DI REPO | TIDAK TERSEDIA DI REPO | TIDAK TERSEDIA DI REPO |
| news_count | TIDAK TERSEDIA DI REPO | TIDAK TERSEDIA DI REPO | TIDAK TERSEDIA DI REPO | TIDAK TERSEDIA DI REPO |
| sentiment_vol_7d | TIDAK TERSEDIA DI REPO | TIDAK TERSEDIA DI REPO | TIDAK TERSEDIA DI REPO | TIDAK TERSEDIA DI REPO |
| sentiment_shock | TIDAK TERSEDIA DI REPO | TIDAK TERSEDIA DI REPO | TIDAK TERSEDIA DI REPO | TIDAK TERSEDIA DI REPO |

Pemanfaatan statistik deskriptif (setelah dieksekusi): untuk menggambarkan skala harga vs volatilitas return, distribusi indikator teknikal, distribusi intensitas berita, serta karakter sebaran sentimen harian di Bab IV.

### B.2 Pembagian Train / Validation / Test

- Rasio split berdasarkan config: train 70%, val 15%, test 15% (lihat: `configs/data.yaml`).
- Penetapan `split` dilakukan kronologis berbasis `time_idx` setelah penggabungan harga + sentimen (lihat: `src/data/build_tft_master_dataset.py`).
- Karena `tft_master.csv` belum tersedia, jumlah baris dan rentang tanggal per split perlu dihasilkan ulang.

**Tabel B.3 – Pembagian Data Train/Validation/Test (perlu dieksekusi ulang)**

| Split | Jumlah Baris | Tanggal Awal | Tanggal Akhir |
|-------|--------------|--------------|---------------|
| Train | TIDAK TERSEDIA DI REPO | TIDAK TERSEDIA DI REPO | TIDAK TERSEDIA DI REPO |
| Val | TIDAK TERSEDIA DI REPO | TIDAK TERSEDIA DI REPO | TIDAK TERSEDIA DI REPO |
| Test | TIDAK TERSEDIA DI REPO | TIDAK TERSEDIA DI REPO | TIDAK TERSEDIA DI REPO |

Catatan: pembagian mengikuti praktik time-series split (non-random), sehingga data validasi dan uji selalu berada setelah periode pelatihan.

## C. DETAIL INDIKATOR & FITUR

### C.1 Definisi Fitur Teknis (`src/data/compute_technical_indicators.py`)

- **log_return_1d**  
  Definisi: log return harian.  
  Rumus: $$r_t = \ln\left(\frac{P_t}{P_{t-1}}\right)$$
- **vol_20**  
  Definisi: volatilitas 20 hari sebagai simpangan baku log return 20 hari.  
  Rumus: $$\sigma_{20,t} = \sqrt{\frac{1}{19}\sum_{i=0}^{19}(r_{t-i} - \bar{r}_{20,t})^2}$$
- **rsi_14**  
  Definisi: Relative Strength Index 14 hari menggunakan rata-rata gain/loss.  
  Rumus: $$RSI_t = 100 - \frac{100}{1 + \frac{\text{avg\_gain}_{14,t}}{\text{avg\_loss}_{14,t}}}$$
- **ma_5_div_ma_20**  
  Definisi: rasio MA5 terhadap MA20 untuk tren jangka pendek vs menengah.  
  Rumus: $$\text{MA5/MA20}_t = \frac{\text{MA}_5(P_t)}{\text{MA}_{20}(P_t)}$$
- **bb_width_20**  
  Definisi: lebar Bollinger Band 20 hari, \((\text{upper}-\text{lower})/\text{MA20}\).  
  Rumus: $$\text{BBWidth}_{20,t} = \frac{(\text{MA}_{20,t} + 2\sigma_{20,t}) - (\text{MA}_{20,t} - 2\sigma_{20,t})}{\text{MA}_{20,t}+\varepsilon}$$
- **price_zscore_20**  
  Definisi: z-score harga terhadap MA20.  
  Rumus: $$z_{20,t} = \frac{P_t - \text{MA}_{20,t}}{\sigma_{20,t}+\varepsilon}$$
- **volume_ma_ratio_20**  
  Definisi: rasio volume terhadap MA volume 20 hari.  
  Rumus: $$\text{VolRatio}_{20,t} = \frac{V_t}{\text{MA}_{20}(V_t)+\varepsilon}$$
- **volume_zscore_20**  
  Definisi: z-score volume terhadap MA volume 20 hari.  
  Rumus: $$z^{(V)}_{20,t} = \frac{V_t - \text{MA}_{20}(V_t)}{\sigma_{20}^{(V)}+\varepsilon}$$
- **close_lag_2, close_lag_3**  
  Definisi: harga penutupan t-2, t-3 sebagai lag.
- **return_mean_5d**  
  Definisi: rata-rata log return 5 hari.  
  Rumus: $$\bar{r}_{5,t} = \frac{1}{5}\sum_{i=0}^{4} r_{t-i}$$
- **return_std_5d**  
  Definisi: simpangan baku log return 5 hari.  
- **intraday_range_pct**  
  Definisi: rentang intraday relatif ke close.  
  Rumus: $$\frac{H_t - L_t}{P_t + \varepsilon}$$
- **atr_14**  
  Definisi: Average True Range 14 hari (rata-rata rolling true range).  
  Rumus: $$ATR_{14,t} = \frac{1}{14}\sum_{i=0}^{13} TR_{t-i}$$ dengan \(TR_t = \max(H_t-L_t, |H_t-P_{t-1}|, |L_t-P_{t-1}|)\).
- **gap_return_1d**  
  Definisi: log gap overnight antara open hari ini dan close kemarin.  
  Rumus: $$g_t = \ln\left(\frac{O_t}{P_{t-1}+\varepsilon}\right)$$

Parameter: MA5/MA20, RSI14, volatilitas 20 hari, rolling mean/std 5 hari, ATR14, quantile clipping epsilon kecil untuk hindari divisi nol.

### C.2 Definisi Fitur Sentimen

Alur labeling per artikel (`src/data/gpt_sentiment_labeling.py`):
- `l_text`: output GPT (gpt-5-nano) dengan label {NEGATIF, NETRAL, POSITIF} → skor {-1,0,1}.
- `l_market`: label pasar berbasis abnormal return harga window (close_next - close_prev)/close_prev relatif benchmark rata-rata tanggal; threshold kuantil \(\theta_1, \theta_2\) (0.75, 0.9 quantile |AR|) → strong signal jika |AR| ≥ \(\theta_2\).
- `l_lex`: label leksikon dari skor token positif-negatif, dengan threshold berbasis simpangan baku: \(\tau_1 = 0.5\sigma, \tau_2 = 1.0\sigma\).
- Voting `compute_final_and_conf`: GPT menjadi anchor; GPT negatif butuh dukungan market/lex untuk tetap negatif, GPT positif dipertahankan kecuali ditolak keras; GPT netral memakai gabungan market/lex; confidence dinaikkan jika strong signal.

Agregasi harian (`src/data/aggregate_daily_sentiment.py`):
- **sentiment_mean**: rata-rata \(l_{final}\) harian.  
  Rumus: $$s_t = \frac{1}{n_t}\sum_{i=1}^{n_t} l^{(i)}_{final,t}$$
- **sentiment_mean_3d / 5d**: rata-rata rolling 3 atau 5 hari dari `sentiment_mean`.  
  Rumus: $$s^{(3)}_t = \frac{1}{3}\sum_{i=0}^{2} s_{t-i}$$
- **sentiment_mean_conf_weighted**: rata-rata harian berbobot confidence jika tersedia.  
  Pseudocode: `mean = sentiment_conf_weighted_sum / sentiment_conf_sum` bila `sentiment_conf_sum>0`, else `sentiment_mean`.
- **news_count**: jumlah berita harian \(n_t\); **news_count_3d/7d** rolling sum 3/7 hari.  
  Rumus: $$n^{(3)}_t = \sum_{i=0}^{2} n_{t-i}$$
- **sentiment_vol_7d**: simpangan baku rolling 7 hari dari `sentiment_mean`.  
  Rumus: $$\sigma^{(s)}_{7,t} = \text{std}(s_{t-6},...,s_t)$$
- **sentiment_trend_5d**: selisih level sekarang dengan rata-rata 5 hari sebelumnya.  
  Rumus: $$\Delta s^{(5)}_t = s_t - \bar{s}_{t-1,5}$$
- **sentiment_shock**: deviasi dari rata-rata 3 hari sebelumnya (shifted).  
  Rumus: $$shock_t = s_t - \bar{s}_{t-1,3}$$
- **extreme_news**: indikator 1 jika |shock| > quantile 0.9 absolut per ticker; 0 selainnya.
- **rel_sentiment_mean / rel_sentiment_mean_3d**: sentimen relatif terhadap ticker MARKET (pasar) pada tanggal yang sama.  
  Rumus: $$rel_t = s_t - s^{(market)}_t$$
- **rel_news_count_3d**: selisih news_count_3d ticker vs MARKET.  
- **sentiment_pos_ratio / sentiment_neg_ratio / sentiment_balance_ratio**: rasio jumlah berita positif/negatif terhadap total; balance = (pos - neg)/total.
- **sentiment_vol_7d**, **high_news_day**, **sentiment_is_bullish_day**, **sentiment_is_bearish_day** sesuai logika biner pada kode agregasi.

Interpretasi pasar modal: `sentiment_shock` menandai lonjakan emosional, `sentiment_vol_7d` menangkap ketidakpastian berita, `rel_sentiment_*` membandingkan emiten dengan sentimen pasar umum (ticker MARKET).

## D. DETAIL MODEL TFT & TRAINING

### D.1 Konfigurasi Model Baseline vs Hybrid

**Tabel D.1 – Perbandingan Konfigurasi TFT Baseline vs Hybrid**

| Aspek | Baseline | Hybrid |
|-------|----------|--------|
| Target | close | close |
| Fitur input | Fitur teknikal whitelist: close, volume, log_return_1d, vol_20, rsi_14, ma_5_div_ma_20, bb_width_20, price_zscore_20, volume_ma_ratio_20, volume_zscore_20, close_lag_2, close_lag_3, return_mean_5d, return_std_5d, intraday_range_pct, atr_14, gap_return_1d | Fitur teknikal di atas + fitur sentimen whitelist (sentiment_mean, sentiment_mean_3d, sentiment_vol_7d, sentiment_trend_5d, news_count, news_count_3d, sentiment_shock, extreme_news) dengan opsi representasi raw/sign dan clipping 99.5% |
| hidden_size | 64 | 128 |
| dropout | 0.10 | 0.20 |
| max_encoder_length | 60 | 60 |
| max_prediction_length | 5 | 5 |
| batch_size | 64 | 64 |
| learning_rate | 5e-4 | 5e-4 |
| loss function | MAE (opsional QuantileLoss) | MAE (opsional QuantileLoss) |
| optimizer | default Lightning (Adam) via TFT | default Lightning (Adam) via TFT |
| early stopping (monitor) | val_loss, patience 10 | val_loss, patience 10 |
| checkpoint terbaik | `models/tft_baseline/tft-baseline-epoch=09-val_loss=377.0987.ckpt` | `models/tft_with_sentiment/tft-with-sentiment-epoch=04-val_loss=303.5286.ckpt` |

Perbedaan: hybrid menambah covariate sentimen dan kapasitas model (hidden_size/dropout) untuk menangkap interaksi berita-harga; baseline lebih ringan hanya teknikal.

### D.2 TimeSeriesDataSet & Peran Fitur

- Static categorical: `ticker` (opsional `sector` jika ada).  
- Time-varying known reals: `time_idx`, `day_of_week`, `month`, `is_month_end`.  
- Time-varying unknown reals: baseline → fitur teknikal whitelist; hybrid → teknikal + sentimen whitelist.  
- Target: `close`; group_ids: `["ticker"]`; normalizer: `GroupNormalizer(softplus)`.

**Tabel D.2 – Peran Fitur (contoh sesuai skrip)**

| Nama Fitur | Tipe | Peran | Keterangan |
|------------|------|-------|------------|
| ticker | categorical | static | identitas saham |
| time_idx | real | known_future | indeks waktu global |
| day_of_week | real/int | known_future | fitur kalender |
| month | real/int | known_future | fitur kalender |
| is_month_end | real/int | known_future | akhir bulan |
| close | real | target / observed_past | harga penutupan |
| log_return_1d | real | observed_past | return harian |
| rsi_14 | real | observed_past | momentum |
| vol_20 | real | observed_past | volatilitas 20 hari |
| ma_5_div_ma_20 | real | observed_past | rasio MA |
| bb_width_20 | real | observed_past | lebar Bollinger |
| price_zscore_20 | real | observed_past | z-score harga |
| volume_ma_ratio_20 | real | observed_past | volume relatif |
| volume_zscore_20 | real | observed_past | z-score volume |
| close_lag_2 / close_lag_3 | real | observed_past | lag harga |
| return_mean_5d / return_std_5d | real | observed_past | statistik return 5 hari |
| intraday_range_pct | real | observed_past | range intraday |
| atr_14 | real | observed_past | ATR 14 |
| gap_return_1d | real | observed_past | gap overnight |
| sentiment_mean, sentiment_mean_3d, sentiment_vol_7d, sentiment_trend_5d, sentiment_shock | real | observed_past (hybrid) | level/tren/vol sentimen |
| news_count, news_count_3d, extreme_news | real/int | observed_past (hybrid) | intensitas berita |

### D.3 Training & Hyperparameter

- Optimizer: default Adam dari `TemporalFusionTransformer.from_dataset`.  
- Learning rate: 5e-4; batch_size: 64; max_epochs: 100.  
- Gradient clipping: 0.1.  
- Early stopping: monitor `val_loss`, patience 10.  
- Trainer: accelerator "gpu" (devices=1) untuk baseline; hybrid `devices` diambil dari config (default 1), precision `32-true` (baseline) / `precision` config (hybrid), log interval 10 step.  
- Loss: MAE (output_size=1) default; QuantileLoss (output_size=7) opsional jika `loss` di config diubah.  
- Seed: 42; allow_missing_timesteps=True; add_relative_time_idx, add_target_scales, add_encoder_length diaktifkan.

## E. DETAIL EVALUASI & BACKTEST

### E.1 Definisi Metrik

- **MAE**: $$MAE = \frac{1}{N}\sum_{i=1}^N |\hat{y}_i - y_i|$$
- **RMSE**: $$RMSE = \sqrt{\frac{1}{N}\sum_{i=1}^N (\hat{y}_i - y_i)^2}$$
- **MAPE**: $$MAPE = \frac{100\%}{N}\sum_{i=1}^N \left|\frac{\hat{y}_i - y_i}{y_i + \varepsilon}\right|$$
- **sMAPE** (tersedia di backtest script jika diaktifkan): $$sMAPE = \frac{100\%}{N}\sum_{i=1}^N \frac{|\hat{y}_i - y_i|}{(|y_i| + |\hat{y}_i|)/2 + \varepsilon}$$
- **R^2** (jika dihitung): $$R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$

Penghitungan di skrip `evaluate_tft_models.py`: metrik global dihitung dari matriks \(y_{true}\) dan \(y_{pred}\) berukuran (n_\text{seri}, horizon), lalu dicetak per horizon (H+1 s.d. H+5). Backtest penuh (`evaluate_tft_backtest*.py`) menyiapkan struktur serupa untuk penilaian rolling.

### E.2 Ringkasan Hasil

- Tidak ada file hasil numerik backtest/test di repo (mis. `tft_master.csv` dan output evaluasi tidak tersedia).  
- Untuk memperoleh metrik: jalankan `python -m src.models.evaluate_tft_models` setelah `tft_master.csv` dan checkpoint tersedia; untuk rolling backtest gunakan `python -m src.models.evaluate_tft_backtest_full` sesuai `configs/experiments.yaml`.

**Tabel E.1 – Contoh Perbandingan Kinerja (perlu dieksekusi)**

| Model | Ticker | Horizon | MAE | RMSE | MAPE | sMAPE | R² |
|-------|--------|---------|-----|------|------|-------|----|
| baseline | TIDAK TERSEDIA DI REPO | H+1 | TIDAK TERSEDIA | TIDAK TERSEDIA | TIDAK TERSEDIA | TIDAK TERSEDIA | TIDAK TERSEDIA |
| hybrid | TIDAK TERSEDIA DI REPO | H+1 | TIDAK TERSEDIA | TIDAK TERSEDIA | TIDAK TERSEDIA | TIDAK TERSEDIA | TIDAK TERSEDIA |
| baseline | TIDAK TERSEDIA DI REPO | H+5 | TIDAK TERSEDIA | TIDAK TERSEDIA | TIDAK TERSEDIA | TIDAK TERSEDIA | TIDAK TERSEDIA |
| hybrid | TIDAK TERSEDIA DI REPO | H+5 | TIDAK TERSEDIA | TIDAK TERSEDIA | TIDAK TERSEDIA | TIDAK TERSEDIA | TIDAK TERSEDIA |

### E.3 Analisis Kualitatif

- Artefak tersedia: `reports/feature_filter_scores.csv`, `reports/feature_target_correlation.csv`, serta grafik di `reports/figures/` (true vs pred/residual) yang dapat dipakai untuk insight visual setelah angka metrik dihitung.
- Perlu menjalankan evaluasi/backtest untuk menghasilkan plot interpretasi TFT (`src/models/interpret_tft_models.py`) dan analisis efek berita (`src/models/analyze_news_effect.py`).
- Insight yang dapat dituliskan setelah eksekusi: tren error per horizon (skrip mencetak MAE/RMSE/MAPE H+1–H+5), perbandingan bucket true/pred di `add_buckets_and_save_forecasts`, dan perbedaan kinerja baseline vs hybrid (improvement persentase dihitung dengan `safe_improvement`).
