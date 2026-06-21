from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
REPORTSS = ROOT / "reportss"
REPORTS = ROOT / "reports"
CONFIG_PATH = ROOT / "configs" / "model_tft.yaml"

COLORS = ["#38BDF8", "#F97316", "#22C55E", "#A855F7", "#F43F5E", "#14B8A6"]
MODEL_COLORS = {
    "Encoder 15 Hari": "#E5E7EB",
    "TFT": "#38BDF8",
    "LLM-TFT": "#F97316",
    "TFT S5": "#38BDF8",
    "LLM-TFT S1": "#F97316",
}

PRICE_PATHS = [INTERIM/"prices_with_indicators.csv", PROCESSED/"prices_with_indicators.csv"]
NEWS_PATHS = [INTERIM/"news_clean.csv", PROCESSED/"news_clean.csv"]
ARTICLE_PATHS = [PROCESSED/"news_with_sentiment_per_article.csv", INTERIM/"article_sentiment.csv"]
DAILY_SENTIMENT_PATHS = [PROCESSED/"daily_sentiment.csv", INTERIM/"daily_sentiment.csv"]
MASTER_PATHS = [PROCESSED/"tft_master.csv", INTERIM/"tft_master.csv"]

EVAL_GLOBAL_PATHS = [REPORTSS/"eval_metrics_global.csv", REPORTS/"eval_metrics_global.csv"]
EVAL_TICKER_PATHS = [REPORTSS/"eval_metrics_by_ticker_global.csv", REPORTS/"eval_metrics_by_ticker_global.csv"]
EVAL_HORIZON_PATHS = [REPORTSS/"eval_metrics_by_horizon.csv", REPORTS/"eval_metrics_by_horizon.csv"]
ATTENTION_PATHS = [REPORTSS/"attention_comparison.csv", REPORTSS/"attention_weights.csv"]
PREDICTION_PATHS = [REPORTSS/"backtest_predictions.csv", REPORTSS/"predictions.csv", PROCESSED/"predictions.csv"]

CHECKPOINTS = [
    ("TFT", "S5", ROOT/"modelssss/baseline/S5/best-checkpoint.ckpt"),
    ("LLM-TFT", "S1", ROOT/"modelssss/hybrid/S1/best-checkpoint.ckpt"),
]

TECH_FEATURES = ["close", "volume", "log_return_1d", "log_return_2d", "vol_20", "rsi_14",
                 "ma_5_div_ma_20", "bb_width_20", "gap_return_1d", "intraday_range_pct"]
SENT_FEATURES = ["news_count_3d", "sentiment_mean_3d", "sentiment_ema_7d",
                 "sentiment_trend_7d", "sentiment_delta_1d", "sentiment_dir_signal"]
FINAL_LABELS = ["l_final", "final_label", "label_final", "sentiment_final", "sentiment"]
