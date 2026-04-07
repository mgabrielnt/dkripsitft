import os
import time
from functools import lru_cache
from typing import Literal

from dotenv import load_dotenv
from openai import OpenAI
import yaml

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
GPT_CFG_PATH = os.path.join(ROOT_DIR, 'configs', 'gpt_sentiment.yaml')
DATA_CFG_PATH = os.path.join(ROOT_DIR, 'configs', 'data.yaml')
load_dotenv()
with open(GPT_CFG_PATH, 'r', encoding='utf-8') as f:
    GPT_CFG = yaml.safe_load(f) or {}
with open(DATA_CFG_PATH, 'r', encoding='utf-8') as f:
    DATA_CFG = yaml.safe_load(f) or {}

MODEL_NAME = str(GPT_CFG.get('model_name', 'gpt-5-nano'))
LABELS = list(GPT_CFG.get('labels', ['NEGATIF', 'NETRAL', 'POSITIF']))
TEMPERATURE = float(GPT_CFG.get('temperature', 1.0))
MAX_TOKENS = int(GPT_CFG.get('max_completion_tokens', GPT_CFG.get('max_tokens', 16)))
MAX_CHARS = int(GPT_CFG.get('text', {}).get('max_chars', 2000))
RETRY_MAX = int(GPT_CFG.get('rate_limit_retry_max_attempts', 5))
RETRY_BASE = float(GPT_CFG.get('rate_limit_initial_delay', 0.5))
DEBUG_PRINT_FIRST_N = int(GPT_CFG.get('debug_print_first_n', 0))
REASONING_EFFORT = GPT_CFG.get('reasoning_effort', 'minimal')
PREDICTION_HORIZON_DAYS = int(DATA_CFG.get('horizon', 3))
SentimentLabel = Literal['NEGATIF', 'NETRAL', 'POSITIF']
TICKER_TO_COMPANY = {
    'BBRI.JK': 'Bank Rakyat Indonesia',
    'BMRI.JK': 'Bank Mandiri',
    'TLKM.JK': 'Telkom Indonesia',
    'ASII.JK': 'Astra International',
}
_debug_counter = 0


def build_prompt(text: str, ticker: str = '', company: str = '') -> str:
    stock_id = f'{ticker} ({company})' if ticker and company else (ticker or 'emiten yang diberitakan')
    return (
        'Anda adalah analis sentimen saham untuk Bursa Efek Indonesia.\n\n'
        f'Tugas: nilai dampak berita berikut terhadap saham {stock_id} untuk horizon {PREDICTION_HORIZON_DAYS} hari ke depan.\n'
        'Gunakan sudut pandang investor jangka pendek.\n'
        'Jawab HANYA SATU KATA dari tiga label ini: NEGATIF, NETRAL, POSITIF.\n\n'
        'NEGATIF = berita jelas buruk bagi minat beli atau prospek harga jangka pendek.\n'
        'NETRAL = berita informatif atau dampaknya tidak jelas.\n'
        'POSITIF = berita jelas baik bagi minat beli atau prospek harga jangka pendek.\n\n'
        f'Teks berita:\n"""{text}"""'
    )


def normalize_output(raw: str) -> SentimentLabel:
    global _debug_counter
    up = str(raw or '').strip().upper()
    if _debug_counter < DEBUG_PRINT_FIRST_N:
        print(f'[DEBUG] RAW GPT OUTPUT: {up!r}')
        _debug_counter += 1
    if 'NEGATIF' in up or 'BEARISH' in up:
        return 'NEGATIF'
    if 'POSITIF' in up or 'POSITIVE' in up or 'BULLISH' in up:
        return 'POSITIF'
    if 'NETRAL' in up or 'NEUTRAL' in up:
        return 'NETRAL'
    return 'NETRAL'


@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError('OPENAI_API_KEY tidak ditemukan di .env')
    return OpenAI(api_key=api_key, base_url=os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1'))


def extract_content(msg) -> str:
    content = getattr(msg, 'content', '')
    if isinstance(content, list):
        parts = []
        for part in content:
            parts.append(str(part.get('text', '')) if isinstance(part, dict) else str(getattr(part, 'text', part)))
        return ''.join(parts).strip()
    return str(content or '').strip()


def classify_sentiment(text: str, ticker: str = '', company: str = '') -> SentimentLabel:
    if not isinstance(text, str) or not text.strip():
        return 'NETRAL'
    processed = text.strip()[:MAX_CHARS]
    company = company or TICKER_TO_COMPANY.get(ticker, '')
    system_msg = (
        'Anda adalah model analisis sentimen saham 3-kelas yang sangat konsisten. '
        'Jawab hanya satu kata: NEGATIF, NETRAL, atau POSITIF.'
    )
    supports_reasoning = MODEL_NAME.startswith('gpt-5') or 'o1' in MODEL_NAME or 'o3' in MODEL_NAME
    for attempt in range(RETRY_MAX):
        try:
            kwargs = {
                'model': MODEL_NAME,
                'messages': [{'role': 'system', 'content': system_msg}, {'role': 'user', 'content': build_prompt(processed, ticker, company)}],
                'max_completion_tokens': MAX_TOKENS,
            }
            if supports_reasoning:
                kwargs['reasoning_effort'] = REASONING_EFFORT
            if not MODEL_NAME.startswith('gpt-5-nano'):
                kwargs['temperature'] = TEMPERATURE
            resp = get_client().chat.completions.create(**kwargs)
            content = extract_content(resp.choices[0].message)
            if not content:
                simple = f'Klasifikasikan sentimen berita ini untuk saham {ticker or company or "emiten"} menjadi NEGATIF, NETRAL, atau POSITIF. Jawab satu kata saja.\n\nBerita: {processed}'
                resp = get_client().chat.completions.create(model=MODEL_NAME, messages=[{'role': 'system', 'content': system_msg}, {'role': 'user', 'content': simple}], max_completion_tokens=MAX_TOKENS)
                content = extract_content(resp.choices[0].message)
            return normalize_output(content)
        except Exception as e:
            msg = str(e).lower()
            if '429' in msg or 'rate limit' in msg:
                time.sleep(RETRY_BASE * (2 ** attempt))
                continue
            print(f'[WARN] classify_sentiment error: {e}')
            return 'NETRAL'
    print(f'[WARN] classify_sentiment gagal setelah {RETRY_MAX} percobaan.')
    return 'NETRAL'
