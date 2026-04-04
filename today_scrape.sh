#!/bin/bash
# 当日の出走表を取得するスクリプト（開催がなければスキップ）
cd "$(dirname "$0")"
TODAY=$(date +%Y-%m-%d)
source .venv/bin/activate
python -m src.cli scrape --date "$TODAY" >> logs/today_scrape.log 2>&1
