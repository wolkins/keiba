"""アプリケーション設定"""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# .env ファイルの読み込み
_env_file = PROJECT_ROOT / ".env"
if _env_file.exists():
    for line in _env_file.read_text().strip().splitlines():
        if "=" in line and not line.startswith("#"):
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip())

DB_PATH = PROJECT_ROOT / "data" / "keiba.db"
DB_URL = os.environ.get("DATABASE_URL", f"sqlite:///{DB_PATH}")

REQUEST_DELAY = 3.0
DAY_PAUSE = 30.0
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)

DECAY_HALF_LIFE_DAYS = 120  # 競馬は長めの半減期

# JRA競馬場コード
COURSE_CODES = {
    "01": "札幌", "02": "函館", "03": "福島", "04": "新潟",
    "05": "東京", "06": "中山", "07": "中京", "08": "京都",
    "09": "阪神", "10": "小倉",
}
