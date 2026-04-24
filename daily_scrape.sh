#!/bin/bash
# 前日分のデータを自動取得する日次スクリプト
cd "$(dirname "$0")"
YESTERDAY=$(date -d "yesterday" +%Y-%m-%d)
source .venv/bin/activate

# 前日のレース結果+オッズ取得
python -m src.cli scrape --date "$YESTERDAY" --with-odds >> logs/daily_scrape.log 2>&1

# 前日が WIN5 開催日だった場合、Race と Win5TargetRace の紐付けを更新
# (非WIN5日は「未登録」メッセージだけで終了するので毎日実行して安全)
python -m src.cli inspect-win5-targets --date "$YESTERDAY" --resolve >> logs/daily_scrape.log 2>&1
