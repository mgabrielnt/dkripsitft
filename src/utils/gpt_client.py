import os
from typing import Literal

from dotenv import load_dotenv
from openai import OpenAI
import yaml

# Lokasi root project
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CONFIG_GPT_PATH = os.path.join(ROOT_DIR, "configs", "gpt_sentiment.yaml")

# Load .env dari root project
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY tidak ditemukan di .env")

# Client OpenAI
client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
)

# Load konfigurasi sentimen dari YAML
with open(CONFIG_GPT_PATH, "r", encoding="utf-8") as f:
    GPT_CONFIG = yaml.safe_load(f)

MODEL_NAME = GPT_CONFIG.get("model_name", "gpt-5-nano")
LABELS = GPT_CONFIG["labels"]
TEMPERATURE = float(GPT_CONFIG.get("temperature", 1.0))
MAX_TOKENS = int(
    GPT_CONFIG.get("max_completion_tokens", GPT_CONFIG.get("max_tokens", 5))
)

SentimentLabel = Literal[
    "SANGAT_NEGATIF",
    "NEGATIF",
    "NETRAL",
    "POSITIF",
    "SANGAT_POSITIF",
]


def build_prompt(text: str) -> str:
    """
    Prompt untuk klasifikasi sentimen berita saham 5-level.
    Fokus pada dampak jangka sangat pendek (1–5 hari ke depan).
    """
    labels_str = ", ".join(LABELS)
    prompt = f"""
Anda adalah analis sentimen khusus berita saham.

Tugas Anda:
1. Baca teks berita berikut.
2. Nilai DAMPAK berita terhadap pergerakan harga saham
   dalam jangka sangat pendek (1–5 hari ke depan).
3. Klasifikasikan sentimen menjadi salah satu dari:
   - SANGAT_NEGATIF : sangat mungkin menurunkan harga secara signifikan
   - NEGATIF        : cenderung menurunkan harga
   - NETRAL         : tidak jelas / dampak kecil
   - POSITIF        : cenderung menaikkan harga
   - SANGAT_POSITIF : sangat mungkin menaikkan harga secara signifikan
4. Jawab HANYA dengan salah satu kata berikut (tanpa penjelasan lain):
   {labels_str}

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
    if "SANGAT" in up and "NEGATIF" in up:
        return "SANGAT_NEGATIF"
    if "SANGAT" in up and "POSITIF" in up:
        return "SANGAT_POSITIF"
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


def classify_sentiment(text: str) -> SentimentLabel:
    """
    Mengembalikan salah satu label:
    'SANGAT_NEGATIF', 'NEGATIF', 'NETRAL', 'POSITIF', 'SANGAT_POSITIF'.
    """
    # Kalau teks kosong → anggap netral saja
    if not isinstance(text, str) or text.strip() == "":
        return "NETRAL"

    # Batasi panjang teks supaya hemat token (mis. 1000 karakter pertama)
    if len(text) > 2000:
        text = text[:2000]

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
        response = client.chat.completions.create(**kwargs)  # type: ignore[arg-type]
        raw = (response.choices[0].message.content or "").strip()
        return _normalize_output(raw)
    except Exception as e:
        # Kalau API error, log ke console dan fallback NETRAL
        print(f"[WARN] classify_sentiment error: {e}")
        return "NETRAL"
