# Perbaikan Deploy Streamlit Dashboard StockForecast

## File yang harus diganti / ditambahkan

1. Ganti file berikut di repository GitHub:

```text
src/dashboard/app.py
```

dengan file `src/dashboard/app.py` dari paket ini.

2. Tambahkan file berikut di root repository:

```text
requirements.txt
streamlit_app.py
```

## Path checkpoint final yang dipakai

Dashboard sudah dikunci memakai checkpoint berikut:

```python
("TFT", "S5", ROOT / "modelssss/baseline/S5/best-checkpoint.ckpt")
("LLM-TFT", "S1", ROOT / "modelssss/hybrid/S1/best-checkpoint.ckpt")
```

## Pengaturan Streamlit Community Cloud

Saat membuat app di Streamlit Cloud:

- Repository: `mgabrielnt/dkripsitft`
- Branch: `main`
- Main file path yang paling aman: `streamlit_app.py`
- Python version: pilih `3.11` pada Advanced settings

## Secret / environment variable opsional

Jika ingin fitur update data harian dan labeling GPT aktif, tambahkan secret:

```toml
OPENAI_API_KEY = "isi_api_key"
STOCKFORECAST_AUTO_UPDATE = "true"
STOCKFORECAST_UPDATE_TIMEOUT = "180"
```

Jika tidak ada secret tersebut, dashboard tetap dapat dibuka. Tombol prediksi tetap memakai checkpoint yang sudah ada.

## Catatan penting

- Auto update data harian dimatikan secara default agar deploy tidak timeout.
- Dashboard tidak melakukan training ulang.
- Jika file checkpoint terlalu kecil atau Git LFS belum aktif, dashboard tidak crash; pesan error akan tampil di bagian "Catatan checkpoint".
- Jika `tft_master.csv` belum ada, dashboard tetap terbuka, tetapi prediksi tidak dapat dijalankan sampai dataset master tersedia.
