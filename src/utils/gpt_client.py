import os
from functools import lru_cache
from typing import Literal

from dotenv import load_dotenv
from openai import OpenAI
import yaml

# Lokasi root project
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CONFIG_GPT_PATH = os.path.join(ROOT_DIR, "configs", "gpt_sentiment.yaml")

# Load .env dari root project
load_dotenv()

# Load konfigurasi sentimen dari YAML (tidak tergantung kunci API)
with open(CONFIG_GPT_PATH, "r", encoding="utf-8") as f:
    GPT_CONFIG = yaml.safe_load(f)

MODEL_NAME = GPT_CONFIG.get("model_name", "gpt-5-nano")
LABELS = GPT_CONFIG["labels"]
TEMPERATURE = float(GPT_CONFIG.get("temperature", 1.0))
MAX_TOKENS = int(
    GPT_CONFIG.get("max_completion_tokens", GPT_CONFIG.get("max_tokens", 5))
)
MAX_CHARS = int(GPT_CONFIG.get("text", {}).get("max_chars", 2000))

SentimentLabel = Literal[
    "NEGATIF",
    "NETRAL",
    "POSITIF",
]


def build_prompt(text: str) -> str:
    """
    Prompt 3-level untuk L_text sesuai spesifikasi multi-sumber.

    Fokus utama: apakah isi teks (judul + lead + kalimat relevan ke emiten)
    membuat investor ingin beli/jual saham perusahaan tersebut.
    """
    labels_str = ", ".join(LABELS)
    prompt = f"""
Anda adalah analis sentimen khusus berita emiten.

Label yang diizinkan:
- NEGATIF (-1): berita utama bertone buruk, memicu kekhawatiran investor
- NETRAL  (0) : deskriptif/administratif atau dampak tidak jelas
- POSITIF (+1): berita utama bertone baik, mendorong optimisme investor

Aturan ringkas:
- POSITIF jika kinerja/prospek membaik, aksi korporasi pro-pemegang saham,
  penilaian eksternal positif, atau kabar operasional kuat.
- NEGATIF jika kinerja/prospek memburuk, aksi korporasi merugikan,
  downgrade/penilaian negatif, masalah hukum/reputasi, atau gangguan operasional serius.
- NETRAL untuk pengumuman administratif, pro-kontra seimbang, atau dampak sangat ambigu.

Kembalikan HANYA salah satu kata ini: {labels_str}

  Teks berita:
  \"\"\"{text}\"\"\"
  """
    return prompt.strip()


def _normalize_output(raw: str) -> SentimentLabel:
    """
    Normalisasi jawaban model ke salah satu label resmi.
    """
    if not isinstance(raw, str):
        return "NETRAL"

    up = raw.strip().upper()

    # Tangani bentuk dengan spasi / underscore
    if "NEGATIF" in up:
        return "NEGATIF"
    if "POSITIF" in up or "POSITIVE" in up:
        return "POSITIF"
    if "NETRAL" in up or "NEUTRAL" in up:
        return "NETRAL"

    # Fallback: kalau persis salah satu label (case-insensitive)
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


def classify_sentiment(text: str) -> SentimentLabel:
    """Klasifikasi 3-level (NEGATIF/NETRAL/POSITIF) untuk L_text."""
    # Kalau teks kosong → anggap netral saja
    if not isinstance(text, str) or text.strip() == "":
        return "NETRAL"

    # Batasi panjang teks supaya hemat token (mis. 2000 karakter pertama)
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS]

    prompt = build_prompt(text)

    # Siapkan payload untuk chat.completions
    kwargs = dict(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": "Anda adalah model analisis sentimen yang sangat konsisten dan tidak pernah memberi penjelasan tambahan.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        max_completion_tokens=MAX_TOKENS,
    )

    # Untuk gpt-5-nano, temperature HARUS default (1) → jangan dikirim.
    if not MODEL_NAME.startswith("gpt-5-nano"):
        kwargs["temperature"] = TEMPERATURE

    try:
        response = _get_client().chat.completions.create(**kwargs)  # type: ignore[arg-type]
        raw = (response.choices[0].message.content or "").strip()
        return _normalize_output(raw)
    except Exception as e:
        # Kalau API error, log ke console dan fallback NETRAL
        print(f"[WARN] classify_sentiment error: {e}")
        return "NETRAL"
