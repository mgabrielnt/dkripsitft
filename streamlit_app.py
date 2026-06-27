"""Root entrypoint untuk Streamlit Community Cloud.

Pakai file ini sebagai main file path jika ingin deploy dari root repo.
File ini menjalankan dashboard utama di src/dashboard/app.py.
"""

from pathlib import Path
import runpy

APP_PATH = Path(__file__).resolve().parent / "src" / "dashboard" / "app.py"
runpy.run_path(str(APP_PATH), run_name="__main__")
