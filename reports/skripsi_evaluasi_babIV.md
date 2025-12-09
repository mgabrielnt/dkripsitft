# Bahan Evaluasi & Pembahasan (Bab IV)

## E.1 Definisi dan Peran Metrik Evaluasi
- **Mean Absolute Error (MAE)**
  - Definisi: rata-rata absolut selisih antara nilai aktual dan prediksi.
  - Rumus: \(MAE = \frac{1}{N}\sum_{i=1}^{N} |y_i - \hat{y}_i|\).
- **Root Mean Squared Error (RMSE)**
  - Definisi: akar dari rata-rata kuadrat selisih, menekankan error besar.
  - Rumus: \(RMSE = \sqrt{\frac{1}{N}\sum_{i=1}^{N} (y_i - \hat{y}_i)^2}\).
- **Mean Absolute Percentage Error (MAPE)**
  - Definisi: rata-rata persentase error absolut terhadap nilai aktual.
  - Rumus: \(MAPE = \frac{100}{N}\sum_{i=1}^{N} \left|\frac{y_i - \hat{y}_i}{y_i}\right|\).
- **Symmetric MAPE (sMAPE)**
  - Definisi: persentase error yang dibagi rata-rata nilai aktual dan prediksi, lebih stabil ketika nilai mendekati nol.
  - Rumus: \(sMAPE = \frac{100}{N}\sum_{i=1}^{N} \frac{|y_i - \hat{y}_i|}{(|y_i| + |\hat{y}_i|)/2}\).
- **Koefisien Determinasi (R²)**
  - Definisi: proporsi variasi target yang dapat dijelaskan oleh model.
  - Rumus: \(R^2 = 1 - \frac{\sum_{i=1}^{N}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{N}(y_i - \bar{y})^2}\).
- **Peran dalam repo**
  - `val_loss` saat training TFT menggunakan MAE (lihat konfigurasi model di `configs/model_tft.yaml`).
  - `reports/tft_regression_summary.csv` memuat MAE, MSE, RMSE, MAPE, sMAPE, dan R² untuk model baseline vs hybrid (raw dan bias correction). Angka horizon-spesifik tidak tersedia karena file backtest per horizon belum ada.

## E.2 Hasil Backtest & Evaluasi Global
- **Ketersediaan data**: File backtest multi-horizon seperti `data/processed/tft_backtest_full.csv` tidak ada di repo sehingga metrik per horizon belum dapat dihitung. Hasil yang tersedia berasal dari `reports/tft_regression_summary.csv` (evaluasi agregat).
- **Metrik agregat baseline vs hybrid**

| Model    | Versi             | MAE    | RMSE    | MAPE (%) | sMAPE (%) | R²     |
|----------|-------------------|--------|---------|----------|-----------|--------|
| Baseline | Raw               | 253.83 | 297.92  | 5.75     | 5.52      | 0.9764 |
| Baseline | Bias corrected    | 151.19 | 177.99  | 3.16     | 3.16      | 0.9916 |
| Hybrid   | Raw               | 214.55 | 274.75  | 5.00     | 4.79      | 0.9799 |
| Hybrid   | Bias corrected    | 198.11 | 240.38  | 4.83     | 4.78      | 0.9846 |

- **Catatan tambahan**
  - Angka di atas merupakan ringkasan global (tidak terpisah per horizon atau ticker). Untuk memperoleh metrik per horizon/ticker, perlu menjalankan `src/models/evaluate_tft_backtest_full.py` setelah `tft_master.csv` dan checkpoint tersedia.
  - Bias correction menurunkan MAE/RMSE signifikan untuk baseline dan hybrid; baseline bias-corrected mencapai MAE ~151 vs hybrid bias-corrected ~198, menunjukkan bias correction lebih menguntungkan model baseline pada ringkasan ini.

## E.3 Analisis Korelasi & Grafik Fitur
- **Korelasi fitur terhadap target** (`reports/feature_target_correlation.csv`)

| Fitur              | Korelasi | |Fitur| (absolut) |
|--------------------|----------|-------------------|
| vol_20             | -0.2935  | 0.2935            |
| volume             | -0.2652  | 0.2652            |
| bb_width_20        | -0.2542  | 0.2542            |
| return_std_5d      | -0.2005  | 0.2005            |
| sentiment_vol_7d   | -0.0734  | 0.0734            |
| neu_count          | 0.0669   | 0.0669            |
| ma_5_div_ma_20     | 0.0506   | 0.0506            |
| strong_market_count| -0.0472  | 0.0472            |
| sentiment_conf_mean| -0.0460  | 0.0460            |
| news_count_3d      | 0.0439   | 0.0439            |

- **Skor seleksi fitur (F-score & Mutual Information)** (`reports/feature_filter_scores.csv`)
  - F-score tertinggi: vol_20, volume, bb_width_20, return_std_5d, sentiment_vol_7d (menunjukkan kontribusi kuat terhadap variasi target).
  - Mutual information tertinggi: log_return_1d, return_mean_5d, vol_20, bb_width_20, volume (indikasi hubungan non-linear yang informatif).

- **Grafik residual & true-vs-pred** (`reports/figures/`)
  - Terdapat plot residual histogram, QQ-plot, residual vs prediksi, serta true-vs-pred untuk baseline dan hybrid (dengan/ tanpa bias correction).
  - Insight kualitatif yang dapat diambil:
    - Distribusi residual relatif simetris dengan sedikit penyebaran heavy-tail (terlihat pada QQ-plot baseline/hybrid).
    - Plot residual vs pred menunjukkan tidak ada pola non-linear mencolok, namun variasi residual meningkat pada prediksi ekstrem.
    - Grafik true-vs-pred bias-corrected lebih mendekati garis diagonal dibanding versi raw, mendukung penurunan MAE/RMSE setelah koreksi bias.

## E.4 Ringkasan Perbandingan Baseline vs Hybrid
- Hybrid raw mengungguli baseline raw pada MAE/RMSE (214 vs 254; 275 vs 298), namun setelah bias correction baseline justru menjadi terbaik (MAE ~151 vs 198 pada hybrid).
- Tidak tersedia metrik per horizon/ticker; kesimpulan awal hanya berbasis agregat. Perlu menjalankan backtest untuk melihat apakah fitur sentimen memberi keuntungan konsisten pada horizon tertentu atau ticker spesifik.
- Korelasi dan F-score menunjukkan dominasi fitur volatilitas/volume (vol_20, volume, bb_width_20, return_std_5d); fitur sentimen seperti sentiment_vol_7d dan neu_count memiliki korelasi kecil namun tetap muncul di 10 besar, sehingga layak dievaluasi lanjut pada analisis horizon-spesifik.
- Bias correction penting untuk kedua model; dapat dipakai sebagai poin pembahasan kenapa hasil raw masih menyisakan bias sistematis.

## Catatan Eksekusi Lanjutan
- Untuk memperoleh tabel metrik per horizon/ticker sesuai permintaan Bab IV, jalankan pipeline hingga menghasilkan `data/processed/tft_master.csv`, lalu evaluasi dengan `src/models/evaluate_tft_backtest_full.py` untuk menghasilkan file backtest multi-horizon.
