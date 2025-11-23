# dkripsitft

Panduan ringkas untuk menjalankan pipeline Temporal Fusion Transformer (TFT) baseline dan hybrid dengan fitur sentimen.

## Prasyarat
- Jalankan semua perintah dari root repo ini (`/workspace/dkripsitft`).
- Sudah menyiapkan environment Python dengan dependensi Lightning, PyTorch Forecasting, dan scikit-learn.
- Dataset hasil pra-pemrosesan tersedia di `data/processed/tft_master.csv`.
- Konfigurasi berikut ada dan konsisten:
  - `configs/data.yaml` dan `configs/model_tft.yaml` untuk data serta parameter model.
  - `configs/experiments.yaml` berisi lokasi checkpoint untuk evaluasi, backtest, interpretasi, dan analisis efek berita.

### Opsi representasi sentimen (hybrid)
Atur pada `configs/model_tft.yaml` di bagian `tft_model`:
- `sentiment_representation`: `raw` (default) untuk nilai kontinu, atau `sign` untuk membucket ke {-1, 0, 1}.
- `sentiment_bucket_threshold`: ambang batas absolut saat memakai `sign` (misal `0.05` agar nilai di antara -0.05 dan 0.05 menjadi 0).

## Perintah utama
- **Latih TFT baseline (tanpa sentimen)**  
  ```bash
  python src/models/train_tft_baseline.py
  ```

- **Latih TFT hybrid (dengan sentimen)**  
  ```bash
  python src/models/train_tft_with_sentiment.py
  ```
  Menghormati opsi `sentiment_representation` dan `sentiment_bucket_threshold` jika diaktifkan.

- **Evaluasi model (baseline/hybrid) pada test set**  
  ```bash
  python src/models/evaluate_tft_models.py
  ```
  Mengambil checkpoint dari `configs/experiments.yaml`, mencetak MAE/RMSE/MAPE, dan menyimpan prediksi ke `data/processed/`.

- **Backtest per-window (early/mid/late)**  
  ```bash
  python src/models/evaluate_tft_backtest.py
  ```
  Mencetak metrik per horizon untuk tiap window menggunakan checkpoint pada `configs/experiments.yaml`.

- **Interpretasi model (variable importance, dsb.)**  
  ```bash
  python src/models/interpret_tft_models.py
  ```
  Menghasilkan analisis interpretabilitas; otomatis melewati hybrid jika kolom sentimen tidak tersedia.

- **Analisis efek berita (bandingkan baseline vs hybrid)**  
  ```bash
  python src/models/analyze_news_effect.py
  ```
  Memisahkan subset dengan/ tanpa berita, menghitung metrik per horizon, dan memanfaatkan opsi pembucketan sentimen bila diaktifkan.
