# file: src/utils/gpt_client.py

import os
import time
from functools import lru_cache
from typing import Literal, Optional

from dotenv import load_dotenv
from openai import OpenAI
import yaml

# Lokasi root project
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CONFIG_GPT_PATH = os.path.join(ROOT_DIR, "configs", "gpt_sentiment.yaml")

# Load .env
load_dotenv()

# Load konfigurasi GPT dari YAML
with open(CONFIG_GPT_PATH, "r", encoding="utf-8") as f:
    GPT_CONFIG = yaml.safe_load(f)

MODEL_NAME = GPT_CONFIG.get("model_name", "gpt-5-nano")
LABELS = GPT_CONFIG["labels"]
TEMPERATURE = float(GPT_CONFIG.get("temperature", 1.0))
MAX_TOKENS = int(
    GPT_CONFIG.get("max_completion_tokens", GPT_CONFIG.get("max_tokens", 8))
)
MAX_CHARS = int(GPT_CONFIG.get("text", {}).get("max_chars", 2000))

# Config retry & debug
RETRY_MAX = int(GPT_CONFIG.get("rate_limit_retry_max_attempts", 5))
RETRY_BASE_DELAY = float(GPT_CONFIG.get("rate_limit_initial_delay", 0.5))
DEBUG_PRINT_FIRST_N = int(GPT_CONFIG.get("debug_print_first_n", 0))
REASONING_EFFORT = GPT_CONFIG.get("reasoning_effort", "minimal")

SentimentLabel = Literal["NEGATIF", "NETRAL", "POSITIF"]

# Mapping ticker → nama emiten (dipakai saat membangun prompt)
TICKER_TO_COMPANY = {
    "BBCA.JK": "Bank Central Asia",
    "BBRI.JK": "Bank Rakyat Indonesia",
    "BMRI.JK": "Bank Mandiri",
    "TLKM.JK": "Telkom Indonesia",
    "ASII.JK": "Astra International",
    "UNVR.JK": "Unilever Indonesia",
    # Ticker khusus untuk berita makro/pasar luas
    "MARKET": "Pasar saham Indonesia (IHSG & emiten BEI)",
}

_debug_counter = 0  # untuk debug print output mentah beberapa kali pertama


def build_prompt(text: str, ticker: str = "", company: str = "") -> str:
    """
    Prompt 3-level untuk L_text (NEGATIF / NETRAL / POSITIF).

    Fokus:
    - Sudut pandang investor saham jangka pendek (1–5 hari).
    - Menggunakan konteks emiten / sektor yang sesuai dengan ticker.
    - Menilai apakah berita cenderung mendorong HARGA NAIK / TURUN.
    - PERBAIKAN: lebih ketat kapan boleh pakai label NEGATIF,
      dan jelaskan dengan jelas kasus-kasus yang HARUS NETRAL.
    """
    labels_str = ", ".join(LABELS)

    ticker_upper = (ticker or "").upper()

    # Deskripsi identitas yang dianalisis
    if ticker_upper == "MARKET":
        # Berita makro / pasar luas, bukan 1 emiten spesifik
        stock_id = (
            "pasar saham Indonesia secara umum "
            "(IHSG dan emiten di Bursa Efek Indonesia)"
        )
    elif ticker and company:
        stock_id = f"{ticker} ({company})"
    elif ticker:
        stock_id = ticker
    else:
        stock_id = "emiten yang diberitakan"

    # Penjelasan konteks khusus untuk MARKET vs emiten biasa
    if ticker_upper == "MARKET":
        focus_block = (
            "- Nilai sentimen BERITA berikut terhadap PASAR SAHAM INDONESIA SECARA UMUM, "
            "bukan hanya satu emiten.\n"
            "- Fokus pada dampak ke IHSG dan minat beli/jual investor terhadap saham blue-chip "
            "perbankan, telekomunikasi, otomotif, dan consumer goods secara keseluruhan.\n"
        )
    else:
        focus_block = (
            f"- Nilai sentimen BERITA berikut terhadap saham {stock_id}.\n"
            "- Fokus pada bagaimana berita ini mempengaruhi minat beli/jual dan ekspektasi "
            f"harga saham {stock_id} dalam 1–5 hari ke depan.\n"
        )

    prompt = f"""
Anda adalah analis sentimen saham untuk Bursa Efek Indonesia.

Tugas utama:
{focus_block}
- Gunakan sudut pandang investor saham jangka pendek (sekitar 1–5 hari ke depan).
- Pertimbangkan juga berita makro/industri yang MUNGKIN berdampak pada {stock_id}
  (misalnya perubahan suku bunga, kebijakan pemerintah, regulasi OJK/BI, perkembangan
  sektor perbankan/telekomunikasi/otomotif/konsumsi, dll).
- Tentukan bagaimana berita ini MUNGKIN mempengaruhi minat beli/jual dan ekspektasi
  HARGA {stock_id} dalam 1–5 hari ke depan.

Label yang diizinkan (balas HANYA SATU kata persis seperti ini):
- NEGATIF : berita utama jelas cenderung menurunkan minat beli, menambah risiko,
            atau memberi indikasi harga bisa TURUN dalam jangka sangat pendek.
- NETRAL  : berita lebih bersifat informatif/administratif dan dampaknya terhadap harga
            tidak jelas (tidak jelas lebih baik atau lebih buruk bagi investor).
- POSITIF : berita utama cenderung meningkatkan minat beli, mengurangi risiko,
            atau memberi indikasi harga bisa NAIK.

Pedoman penting:
- Jika keseluruhan isi berita lebih dominan BURUK dan ada dampak nyata terhadap {stock_id}
  (misalnya penurunan kinerja, risiko hukum, kerugian investor, penurunan permintaan,
  kegagalan proyek penting) → pilih NEGATIF.
- Jika keseluruhan isi berita lebih dominan BAIK untuk {stock_id}
  (misalnya kinerja membaik, laba meningkat, dividen besar, ekspansi bisnis yang
  kredibel, kontrak besar, sentimen pasar membaik) → pilih POSITIF.
- Pilih NETRAL jika berita TIDAK memberi arah jelas (baik maupun buruk) bagi investor
  jangka pendek, ATAU jika berita hanya bersifat informatif tanpa dampak langsung
  pada prospek laba/harga.

Kasus-kasus yang HARUS dianggap NETRAL (kecuali ada konteks jelas sangat positif/negatif):
- Artikel HOW-TO atau tutorial layanan:
  • cara membeli token listrik, cara top-up, cara menggunakan aplikasi/mobile banking,
    cara mendaftar suatu layanan, cara beli saham, dsb.
- Berita administratif atau struktural:
  • pengangkatan/rotasi direksi/komisaris tanpa kontroversi,
    perubahan alamat kantor, pembukaan cabang, penutupan kantor kecil,
    penandatanganan MoU umum yang belum ada dampak finansial jelas.
- Artikel profil/biografi:
  • riwayat karier seseorang, latar belakang pimpinan, wawancara umum
    tanpa ada informasi rugi besar, skandal, atau masalah kinerja serius.
- Artikel edukasi/informasi umum:
  • edukasi investasi, tips menabung, daftar produk/layanan bank/telekomunikasi,
    ringkasan fitur aplikasi, dsb, tanpa fokus spesifik pada kinerja saham.
- Berita yang hanya menggambarkan peristiwa biasa (kunjungan pejabat, pertemuan,
  konferensi) TANPA menyebut dampak positif/negatif yang kuat terhadap kinerja
  perusahaan atau harga saham.

JANGAN pilih NEGATIF hanya karena:
- ada kata "menjual/melepas" saham/aset jika konteksnya adalah aksi korporasi biasa
  (misalnya penataan portofolio, divestasi wajar, atau transaksi strategis),
  dan tidak ada indikasi kerugian besar atau masalah serius.
- berita berisi kekhawatiran abstrak tanpa bukti, sementara sebagian besar isi berita
  netral atau bahkan positif.

Jika berita mengandung hal baik dan buruk secara seimbang TANPA dominan yang kuat,
LEBIH BAIK pilih NETRAL daripada memaksa POSITIF atau NEGATIF.

Contoh pola umum (tidak wajib dihafal, hanya membantu Anda menilai):
- PERBANKAN (BBCA, BBRI, BMRI):
  • Kenaikan suku bunga acuan, pengetatan kredit, kenaikan NPL, sanksi OJK → cenderung NEGATIF.
  • Penurunan suku bunga, stimulus kredit, pertumbuhan laba kuat, kualitas aset membaik → cenderung POSITIF.
- TELEKOMUNIKASI (TLKM):
  • Gangguan layanan besar, sanksi regulator, kehilangan lisensi spektrum → cenderung NEGATIF.
  • Ekspansi jaringan, proyek infrastruktur digital, pertumbuhan pelanggan/data yang kuat → cenderung POSITIF.
- OTOMOTIF (ASII):
  • Penurunan tajam penjualan mobil, tekanan biaya tinggi, pelemahan daya beli → cenderung NEGATIF.
  • Kenaikan penjualan, insentif pajak/EV, stimulus konsumsi otomotif → cenderung POSITIF.
- CONSUMER GOODS (UNVR):
  • Pelemahan konsumsi rumah tangga, kenaikan tajam biaya bahan baku tanpa kompensasi → NEGATIF.
  • Penguatan konsumsi, inovasi produk sukses, penguatan margin laba → POSITIF.

Jawab SELALU dengan SATU KATA SAJA, tanpa simbol atau tambahan lain,
salah satu dari: {labels_str}

Teks berita:
\"\"\"{text}\"\"\""""
    return prompt.strip()


def _normalize_output(raw: str) -> SentimentLabel:
    """
    Normalisasi jawaban model ke salah satu label resmi.
    """
    if not isinstance(raw, str):
        return "NETRAL"

    up = raw.strip().upper()

    # DEBUG: cetak beberapa output awal untuk cek apakah model patuh instruksi
    global _debug_counter
    if _debug_counter < DEBUG_PRINT_FIRST_N:
        print(f"[DEBUG] RAW GPT OUTPUT: {up!r}")
        _debug_counter += 1

    # Izinkan model menulis misalnya "SANGAT POSITIF", "CENDERUNG NEGATIF" dll
    if "NEGATIF" in up:
        return "NEGATIF"
    if "POSITIF" in up or "POSITIVE" in up or "BULLISH" in up:
        return "POSITIF"
    if "NETRAL" in up or "NEUTRAL" in up:
        return "NETRAL"
    if "BEARISH" in up:
        return "NEGATIF"

    # Fallback: cocokkan persis dengan salah satu label di config
    for label in LABELS:
        if up == label.upper():
            return label  # type: ignore[return-value]

    # Kalau tetap aneh → NETRAL
    return "NETRAL"


@lru_cache(maxsize=1)
def _get_client() -> OpenAI:
    """Lazy instantiate klien OpenAI supaya impor modul tidak gagal tanpa API key."""
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    if not api_key:
        raise ValueError("OPENAI_API_KEY tidak ditemukan di .env")

    return OpenAI(api_key=api_key, base_url=base_url)


def _truncate_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.strip()
    if len(text) > MAX_CHARS:
        return text[:MAX_CHARS]
    return text


def classify_sentiment(
    text: str,
    ticker: str = "",
    company: str = "",
) -> SentimentLabel:
    """
    Klasifikasi 3-level (NEGATIF/NETRAL/POSITIF) untuk L_text.

    Fitur:
    - Potong teks panjang (hemat token).
    - Inject konteks ticker/emiten langsung ke prompt.
    - Retry otomatis kalau kena rate limit (429).
    - Debug beberapa output pertama.
    """
    # Kalau teks kosong → NETRAL
    if not isinstance(text, str) or text.strip() == "":
        return "NETRAL"

    processed = _truncate_text(text)

    # Kalau company tidak dikirim eksplisit, coba lookup dari mapping
    if ticker and not company:
        company = TICKER_TO_COMPANY.get(ticker, "")

    prompt = build_prompt(processed, ticker=ticker, company=company)

    system_msg = (
        "Anda adalah model analisis sentimen saham 3-kelas "
        "(NEGATIF, NETRAL, POSITIF) yang SANGAT KONSISTEN.\n"
        "Selalu utamakan NETRAL jika dampak berita terhadap harga saham tidak jelas.\n"
        "Hanya gunakan NEGATIF jika berita jelas buruk bagi investor, "
        "dan hanya gunakan POSITIF jika berita jelas menguntungkan.\n"
        "Jawab SELALU dengan SATU KATA saja tanpa penjelasan tambahan."
    )

    client = _get_client()

    # Hanya model reasoning yang boleh dikasih reasoning_effort
    supports_reasoning = (
        MODEL_NAME.startswith("gpt-5")  # misal gpt-5-nano, gpt-5.1-mini, dll
        or "o1" in MODEL_NAME
        or "o3" in MODEL_NAME
    )

    def _call_once(user_content: str) -> str:
        kwargs = dict(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_content},
            ],
            max_completion_tokens=MAX_TOKENS,
        )

        if supports_reasoning:
            kwargs["reasoning_effort"] = REASONING_EFFORT

        # Untuk non gpt-5-nano, boleh set temperature dari config
        if not MODEL_NAME.startswith("gpt-5-nano"):
            kwargs["temperature"] = TEMPERATURE

        resp = client.chat.completions.create(**kwargs)  # type: ignore[arg-type]
        msg = resp.choices[0].message
        content = msg.content

        # Antisipasi format content baru (list of parts)
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict):
                    parts.append(str(part.get("text", "")))
                else:
                    txt = getattr(part, "text", None)
                    parts.append(str(txt) if txt is not None else str(part))
            content = "".join(parts)

        return (content or "").strip()

    last_err: Optional[Exception] = None

    if ticker and company:
        stock_id = f"{ticker} ({company})"
    elif ticker:
        stock_id = ticker
    else:
        stock_id = "emiten terkait"

    for attempt in range(RETRY_MAX):
        try:
            raw = _call_once(prompt)

            # Kalau kosong, coba ulang dengan prompt super simple + konteks ticker
            if not raw:
                simple_prompt = (
                    "Klasifikasikan sentimen berita saham berikut terhadap "
                    f"{stock_id} menjadi satu kata saja: NEGATIF, NETRAL, atau POSITIF.\n"
                    "Jawab STRICT hanya satu kata dari tiga itu.\n\n"
                    f"Berita: {processed}"
                )
                raw = _call_once(simple_prompt)

            if not raw:
                print("[WARN] GPT masih mengembalikan string kosong, fallback NETRAL.")
                return "NETRAL"

            return _normalize_output(raw)

        except Exception as e:
            last_err = e
            msg = str(e).lower()
            if "rate limit" in msg or "429" in msg:
                # Exponential backoff kalau kena rate limit
                wait = RETRY_BASE_DELAY * (2**attempt)
                print(
                    f"[WARN] Rate limit {MODEL_NAME} (attempt {attempt+1}/{RETRY_MAX}), "
                    f"sleep {wait:.2f} detik..."
                )
                time.sleep(wait)
                continue

            print(f"[WARN] classify_sentiment non-rate-limit error: {e}")
            return "NETRAL"

    print(
        f"[WARN] classify_sentiment gagal setelah {RETRY_MAX} percobaan. "
        f"Last error: {last_err}"
    )
    return "NETRAL"
