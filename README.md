# keiba

中央競馬（JRA）予想システム。netkeiba.com からデータを取得し、LightGBM LambdaRank + アンサンブルで着順を予測する。

## セットアップ

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
mkdir -p data
```

## CLI

```bash
# 当日データ取得
python -m src.cli.main scrape --date 2025-06-01

# 期間一括取得
python -m src.cli.main scrape-range --from 2024-04-01 [--to 2025-03-31]

# 予測
python -m src.cli.main predict --date 2025-06-01 [--venue 東京] [--mode accuracy|roi]

# モデル学習
python -m src.cli.main train [--min-races 50]

# DB状況確認
python -m src.cli.main status
```

## Web UI

```bash
python -m src.cli.main serve [--port 8002] [--reload]
# http://localhost:8002
```

FastAPI + Jinja2 + htmx。デフォルトポート 8002。

## サーバー運用

```bash
# バックグラウンドでデータ取得
nohup .venv/bin/python -m src.cli.main scrape-range --from 2024-04-01 > scrape.log 2>&1 &

# Web UIをバックグラウンド起動
nohup .venv/bin/python -m src.cli.main serve --port 8002 > serve.log 2>&1 &

# ログ確認
tail -f scrape.log
```

## 構成

```
src/
  scraper/netkeiba.py       # netkeiba.com からスクレイピング
  parser/store.py           # SQLite (data/keiba.db) への格納
  predictor/
    features.py             # 特徴量 (馬/騎手/血統/コース/馬場/展開, 63個)
    model.py                # LightGBM LambdaRank + アンサンブル
    calibration.py          # Isotonic回帰で確率キャリブレーション
    ensemble.py             # 3モデルスタッキング (Ranker+Win+Top3 → Ridge)
    evaluation.py           # ウォークフォワードCV + NDCG/MRR/ROI指標
  cli/main.py               # CLI エントリポイント
  web/                      # FastAPI + Jinja2 + htmx
data/keiba.db               # SQLite DB
```

## 競馬場

| コード | 場名 |
|--------|------|
| 01 | 札幌 |
| 02 | 函館 |
| 03 | 福島 |
| 04 | 新潟 |
| 05 | 東京 |
| 06 | 中山 |
| 07 | 中京 |
| 08 | 京都 |
| 09 | 阪神 |
| 10 | 小倉 |
