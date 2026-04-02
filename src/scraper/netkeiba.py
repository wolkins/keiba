"""netkeiba.com スクレイパー (中央競馬)"""
import re
import time
from datetime import datetime

import requests
from bs4 import BeautifulSoup

from src.common.config import COURSE_CODES, REQUEST_DELAY, USER_AGENT

BASE_URL = "https://db.netkeiba.com"


class NetkeibaScraper:
    """netkeiba.com からレース結果を取得"""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self._last_request = 0

    def _get(self, url: str, encoding: str = "EUC-JP") -> BeautifulSoup:
        """レート制限付きGETリクエスト"""
        elapsed = time.time() - self._last_request
        if elapsed < REQUEST_DELAY:
            time.sleep(REQUEST_DELAY - elapsed)

        resp = self.session.get(url, timeout=30)
        resp.raise_for_status()
        resp.encoding = encoding
        self._last_request = time.time()
        return BeautifulSoup(resp.text, "lxml")

    # ------------------------------------------------------------------
    # レース一覧
    # ------------------------------------------------------------------
    def scrape_race_list(self, date_str: str) -> list[dict]:
        """指定日のレースID一覧を取得

        Args:
            date_str: "YYYY-MM-DD"

        Returns:
            [{"race_id": "202505021211", "race_date": "2025-06-01",
              "course_code": "05", "race_number": 11}, ...]
        """
        ymd = date_str.replace("-", "")
        url = f"{BASE_URL}/race/list/{ymd}/"
        soup = self._get(url)

        races = []
        # レースリンク: /race/XXXXXXXXXXXX/ (12桁)
        links = soup.find_all("a", href=re.compile(r"/race/(\d{12})/"))
        seen = set()
        for link in links:
            m = re.search(r"/race/(\d{12})/", link.get("href", ""))
            if not m:
                continue
            race_id = m.group(1)
            if race_id in seen:
                continue
            seen.add(race_id)

            # race_id 構造: YYYY(4) + 競馬場(2) + 回次(2) + 日次(2) + レース番号(2)
            course_code = race_id[4:6]

            # JRA(中央)のみ: 場コード 01-10。それ以外は地方競馬なのでスキップ
            if course_code not in COURSE_CODES:
                continue

            race_number = int(race_id[10:12])

            races.append({
                "race_id": race_id,
                "race_date": date_str,
                "course_code": course_code,
                "race_number": race_number,
            })

        return races

    # ------------------------------------------------------------------
    # レース結果
    # ------------------------------------------------------------------
    def scrape_race_result(self, race_id: str) -> dict:
        """レース結果を取得

        Args:
            race_id: 12桁レースID

        Returns:
            {"race": {...}, "entries": [...]}
        """
        url = f"{BASE_URL}/race/{race_id}/"
        soup = self._get(url)

        race_info = self._parse_race_info(soup)
        entries = self._parse_entries(soup)

        return {"race": race_info, "entries": entries}

    def _parse_race_info(self, soup: BeautifulSoup) -> dict:
        """レース情報(条件)をパース"""
        info = {
            "race_name": "",
            "grade": "",
            "surface": "",
            "distance": 0,
            "course_type": "",
            "weather": "",
            "track_condition": "",
        }

        # レース名: <dl class="racedata fc"> の <h1> 等
        name_tag = soup.find("h1", class_="racedata_title") or soup.select_one("dl.racedata h1")
        if not name_tag:
            # 別パターン: class="racedata fc" 内の dt
            racedata = soup.find("dl", class_=re.compile(r"racedata"))
            if racedata:
                dt = racedata.find("dt")
                if dt:
                    name_tag = dt
        if name_tag:
            info["race_name"] = name_tag.get_text(strip=True)

        # グレード判定
        info["grade"] = self._detect_grade(soup, info["race_name"])

        # レース条件テキスト: "芝右2400m / 天候 : 晴 / 芝 : 良"
        # <diary_snap_cut> タグや <span> 内、あるいは <p class="smalltxt"> 等に含まれる
        condition_text = ""

        # パターン1: <diary_snap_cut> 内の <span>
        diary_snap = soup.find("diary_snap_cut")
        if diary_snap:
            condition_text = diary_snap.get_text(" ", strip=True)

        # パターン2: dl.racedata 内の dd/span
        if not condition_text:
            racedata = soup.find("dl", class_=re.compile(r"racedata"))
            if racedata:
                spans = racedata.find_all("span")
                for span in spans:
                    txt = span.get_text(strip=True)
                    if re.search(r"(芝|ダ|障)", txt):
                        condition_text = txt
                        break
                if not condition_text:
                    dd = racedata.find("dd")
                    if dd:
                        condition_text = dd.get_text(" ", strip=True)

        # パターン3: smalltxt 等
        if not condition_text:
            small = soup.find("p", class_="smalltxt")
            if small:
                condition_text = small.get_text(" ", strip=True)

        if condition_text:
            self._parse_condition_text(condition_text, info)

        return info

    def _detect_grade(self, soup: BeautifulSoup, race_name: str) -> str:
        """グレードを検出"""
        # アイコン画像から検出
        for img in soup.find_all("img"):
            src = img.get("src", "") + img.get("alt", "")
            if "icon_grade_g1" in src.lower() or "GI" in src:
                return "G1"
            if "icon_grade_g2" in src.lower() or "GII" in src:
                return "G2"
            if "icon_grade_g3" in src.lower() or "GIII" in src:
                return "G3"

        # クラスからの検出
        title_el = soup.find(class_=re.compile(r"Icon_GradeType"))
        if title_el:
            cls = " ".join(title_el.get("class", []))
            if "Icon_GradeType1" in cls:
                return "G1"
            if "Icon_GradeType2" in cls:
                return "G2"
            if "Icon_GradeType3" in cls:
                return "G3"

        # テキストから検出
        text = race_name
        if "(G1)" in text or "（G1）" in text or "(GI)" in text:
            return "G1"
        if "(G2)" in text or "（G2）" in text or "(GII)" in text:
            return "G2"
        if "(G3)" in text or "（G3）" in text or "(GIII)" in text:
            return "G3"
        if "オープン" in text or "(OP)" in text or "(L)" in text:
            return "OP"
        if "(L)" in text:
            return "L"

        return ""

    def _parse_condition_text(self, text: str, info: dict):
        """コンディションテキストをパース

        例: "芝右2400m / 天候 : 晴 / 芝 : 良"
            "ダ左1200m / 天候 : 曇 / ダート : 稍重"
        """
        # 馬場・距離
        m = re.search(r"(芝|ダ|障)\s*(右|左|直線|右内|左内|右外|左外)?\s*(\d{3,5})m", text)
        if m:
            surface_raw = m.group(1)
            info["surface"] = "ダート" if surface_raw == "ダ" else surface_raw
            info["course_type"] = (m.group(2) or "").replace("内", "").replace("外", "")
            info["distance"] = int(m.group(3))

        # 天候
        w = re.search(r"天候\s*[:：]\s*(\S+)", text)
        if w:
            info["weather"] = w.group(1)

        # 馬場状態
        c = re.search(r"(?:芝|ダート|障害)\s*[:：]\s*(良|稍重|重|不良)", text)
        if c:
            info["track_condition"] = c.group(1)

    def _parse_entries(self, soup: BeautifulSoup) -> list[dict]:
        """出走馬テーブルをパース"""
        entries = []

        # テーブル検出: class="race_table_01 nk_tb_common" 等
        table = soup.find("table", class_=re.compile(r"race_table"))
        if not table:
            table = soup.find("table", {"summary": re.compile(r"レース結果")})
        if not table:
            return entries

        rows = table.find_all("tr")
        for row in rows:
            cells = row.find_all("td")
            if len(cells) < 10:
                continue

            entry = self._parse_entry_row(cells)
            if entry:
                entries.append(entry)

        return entries

    def _parse_entry_row(self, cells: list) -> dict | None:
        """結果テーブルの1行をパース

        典型的な列順序:
        着順, 枠番, 馬番, 馬名, 性齢, 斤量, 騎手, タイム, 着差, 通過,
        上り, 単勝, 人気, 馬体重, 調教師, (馬主, 賞金...)
        """
        texts = [c.get_text(strip=True) for c in cells]
        if len(texts) < 10:
            return None

        # 着順 (除外/取消/中止 対応)
        finish_str = texts[0]
        finish_position = None
        if finish_str.isdigit():
            finish_position = int(finish_str)
        elif finish_str in ("除", "取", "中", "失"):
            finish_position = None  # 異常終了
        else:
            return None  # ヘッダ行等

        # 枠番・馬番
        frame_str = texts[1]
        horse_num_str = texts[2]
        if not horse_num_str.isdigit():
            return None

        frame_number = int(frame_str) if frame_str.isdigit() else None
        horse_number = int(horse_num_str)

        entry = {
            "finish_position": finish_position,
            "frame_number": frame_number,
            "horse_number": horse_number,
        }

        # 馬名・馬ID (リンクから取得)
        horse_cell = cells[3]
        horse_link = horse_cell.find("a", href=re.compile(r"/horse/"))
        if horse_link:
            entry["horse_name"] = horse_link.get_text(strip=True)
            hid_m = re.search(r"/horse/(\w+)", horse_link.get("href", ""))
            if hid_m:
                entry["horse_id"] = hid_m.group(1)
        else:
            entry["horse_name"] = texts[3]
            entry["horse_id"] = ""

        # 性齢
        entry["sex_age"] = texts[4]

        # 斤量
        try:
            entry["weight_carry"] = float(texts[5])
        except (ValueError, IndexError):
            entry["weight_carry"] = None

        # 騎手名・騎手ID
        jockey_cell = cells[6]
        jockey_link = jockey_cell.find("a", href=re.compile(r"/jockey/"))
        if jockey_link:
            entry["jockey_name"] = jockey_link.get_text(strip=True)
            jid_m = re.search(r"/jockey/(?:result/recent/)?(\d+)", jockey_link.get("href", ""))
            if jid_m:
                entry["jockey_id"] = jid_m.group(1)
            else:
                entry["jockey_id"] = ""
        else:
            entry["jockey_name"] = texts[6]
            entry["jockey_id"] = ""

        # タイム
        entry["finish_time"] = texts[7] if len(texts) > 7 else ""

        # 着差
        entry["margin"] = texts[8] if len(texts) > 8 else ""

        # 通過順
        entry["passing"] = ""
        # 上がり3F
        entry["last_3f"] = None
        # 単勝オッズ
        entry["odds_win"] = None
        # 人気
        entry["popularity"] = None

        # 後半のカラムはサイトのバリエーションがあるため
        # インデックスで取得しつつフォールバック
        if len(texts) > 9:
            entry["passing"] = texts[9]

        if len(texts) > 10:
            try:
                entry["last_3f"] = float(texts[10])
            except ValueError:
                pass

        if len(texts) > 11:
            try:
                entry["odds_win"] = float(texts[11])
            except ValueError:
                pass

        if len(texts) > 12:
            try:
                entry["popularity"] = int(texts[12])
            except ValueError:
                pass

        # 馬体重 "480(+2)" or "480(-4)" or "計不"
        entry["horse_weight"] = None
        entry["weight_diff"] = None
        if len(texts) > 13:
            self._parse_horse_weight(texts[13], entry)

        # 調教師 (馬体重の次のカラム、あるいはリンクから)
        entry["trainer"] = ""
        # 調教師はリンクから取得を試みる
        for i, cell in enumerate(cells):
            trainer_link = cell.find("a", href=re.compile(r"/trainer/"))
            if trainer_link:
                entry["trainer"] = trainer_link.get_text(strip=True)
                break

        return entry

    def _parse_horse_weight(self, text: str, entry: dict):
        """馬体重テキストをパース: "480(+2)", "480(-4)", "480(0)", "計不" """
        m = re.match(r"(\d+)\(([+\-]?\d+)\)", text)
        if m:
            entry["horse_weight"] = int(m.group(1))
            entry["weight_diff"] = int(m.group(2))
