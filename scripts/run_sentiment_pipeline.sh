#!/usr/bin/env bash
set -euo pipefail

# Simple wrapper to run the full sentiment → aggregation → TFT build pipeline.
# Usage: bash scripts/run_sentiment_pipeline.sh
# Requires:
#   - data/interim/news_clean.csv
#   - data/interim/prices_with_indicators.csv
#   - OPENAI_API_KEY set (for GPT L_text labeling)

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

python -m src.data.gpt_sentiment_labeling
python -m src.data.aggregate_daily_sentiment
python -m src.data.build_tft_master_dataset
