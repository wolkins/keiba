#!/bin/bash
# 当日の出走表を取得するスクリプト（開催がなければスキップ）
cd "$(dirname "$0")"
TODAY=$(date +%Y-%m-%d)
source .venv/bin/activate

# 当日の WIN5 対象5R を JRA 公式から取得
# (非WIN5日は JRA 側に掲載が無いので空返却で終了、副作用なし)
python -m src.cli scrape-win5-targets --date "$TODAY" >> logs/today_scrape.log 2>&1

# 当日の出走表取得
python -m src.cli scrape --date "$TODAY" >> logs/today_scrape.log 2>&1

# 出走表取得後に Win5TargetRace と Race の紐付け
python -m src.cli inspect-win5-targets --date "$TODAY" --resolve >> logs/today_scrape.log 2>&1
