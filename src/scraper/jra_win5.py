"""JRA公式 WIN5 対象レース スクレイパー

https://www.jra.go.jp/kouza/win5/info/racelist.html から対象5レースを取得する。
ヒューリスティック推定は行わない (対象レースを誤って別のレースに置換すると、
WIN5 は購入時点で全券無効になるため)。
"""
import re
import time
from datetime import date, datetime

import requests
from bs4 import BeautifulSoup

from src.common.config import COURSE_CODES, REQUEST_DELAY, USER_AGENT

WIN5_RACELIST_URL = "https://www.jra.go.jp/kouza/win5/info/racelist.html"

# JRA表記の競馬場名 → 既存 COURSE_CODES の逆引き
_NAME_TO_CODE = {name: code for code, name in COURSE_CODES.items()}


class Win5Scraper:
    """JRA公式 WIN5 ページからのスクレイパー"""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self._last_request = 0.0

    def _get(self, url: str, encoding: str = "shift_jis") -> BeautifulSoup:
        elapsed = time.time() - self._last_request
        if elapsed < REQUEST_DELAY:
            time.sleep(REQUEST_DELAY - elapsed)

        resp = self.session.get(url, timeout=30)
        resp.raise_for_status()
        resp.encoding = encoding
        self._last_request = time.time()
        return BeautifulSoup(resp.text, "lxml")

    def scrape_target_races(self, target_date: str | date) -> list[dict]:
        """指定日のWIN5対象5レースを取得

        Args:
            target_date: "YYYY-MM-DD" or datetime.date

        Returns:
            [
                {
                    "race_date": "2026-04-26",
                    "leg_index": 1,
                    "racecourse_name": "京都",
                    "racecourse_code": "08",
                    "race_number": 10,
                    "post_time": "14時50分",
                    "close_time": "14時45分",
                    "source": "jra",
                    "source_url": "https://www.jra.go.jp/...",
                },
                ... (5件)
            ]
            対象日が見つからなければ空リスト。
        """
        if isinstance(target_date, str):
            target_date = datetime.strptime(target_date, "%Y-%m-%d").date()

        soup = self._get(WIN5_RACELIST_URL)
        table = soup.find("table", class_=re.compile(r"win5list"))
        if not table:
            return []

        tbody = table.find("tbody") or table
        rows = tbody.find_all("tr")

        month_day_label = f"{target_date.month}月{target_date.day}日"

        for row in rows:
            cells = row.find_all("td")
            if len(cells) < 7:
                continue

            date_cell_text = cells[0].get_text(strip=True)
            # "4月26日（日曜）" 形式
            if not date_cell_text.startswith(month_day_label):
                continue

            # 締切時刻
            close_time_tag = cells[1].find("strong")
            close_time = close_time_tag.get_text(strip=True) if close_time_tag else None

            results = []
            for i, cell in enumerate(cells[2:7], start=1):
                race_span = cell.find("span", class_="race")
                time_span = cell.find("span", class_="time")
                if not race_span:
                    continue

                race_label = race_span.get_text(strip=True)  # "京都10R"
                m = re.match(r"([^\d]+)(\d+)R", race_label)
                if not m:
                    continue

                racecourse_name = m.group(1).strip()
                race_number = int(m.group(2))
                racecourse_code = _NAME_TO_CODE.get(racecourse_name)

                post_time_text = time_span.get_text(strip=True) if time_span else ""
                # "14時50分 発走" から時刻部分だけ抽出
                pm = re.search(r"(\d+時\d+分)", post_time_text)
                post_time = pm.group(1) if pm else None

                results.append({
                    "race_date": target_date.isoformat(),
                    "leg_index": i,
                    "racecourse_name": racecourse_name,
                    "racecourse_code": racecourse_code,
                    "race_number": race_number,
                    "post_time": post_time,
                    "close_time": close_time,
                    "source": "jra",
                    "source_url": WIN5_RACELIST_URL,
                })

            if len(results) == 5:
                return results
            # 5未満なら壊れているので空で返す
            return []

        return []
