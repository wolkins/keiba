"""netkeiba.com スクレイパー (中央競馬)"""
import re
import time
from datetime import datetime

import requests
from bs4 import BeautifulSoup

from src.common.config import COURSE_CODES, REQUEST_DELAY, USER_AGENT

BASE_URL = "https://db.netkeiba.com"
RACE_URL = "https://race.netkeiba.com"


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
    # 当日レース一覧 (race.netkeiba.com)
    # ------------------------------------------------------------------
    def scrape_today_race_list(self, date_str: str) -> list[dict]:
        """当日の出走表からレースID一覧を取得 (結果未確定のレース用)

        Args:
            date_str: "YYYY-MM-DD"

        Returns:
            scrape_race_list() と同じ形式
        """
        ymd = date_str.replace("-", "")
        url = f"{RACE_URL}/top/race_list_sub.html?kaisai_date={ymd}"
        soup = self._get(url)

        races = []
        links = soup.find_all("a", href=re.compile(r"race_id=(\d{12})"))
        seen = set()
        for link in links:
            m = re.search(r"race_id=(\d{12})", link.get("href", ""))
            if not m:
                continue
            race_id = m.group(1)
            if race_id in seen:
                continue
            seen.add(race_id)

            course_code = race_id[4:6]
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
    # 当日出走表 (race.netkeiba.com)
    # ------------------------------------------------------------------
    def scrape_shutuba(self, race_id: str) -> dict:
        """出走表ページから出馬情報を取得 (結果未確定のレース用)

        Args:
            race_id: 12桁レースID

        Returns:
            scrape_race_result() と同じ形式 (finish_position は None)
        """
        url = f"{RACE_URL}/race/shutuba.html?race_id={race_id}"
        soup = self._get(url)

        race_info = self._parse_shutuba_race_info(soup)
        entries = self._parse_shutuba_entries(soup)

        return {"race": race_info, "entries": entries}

    def _parse_shutuba_race_info(self, soup: BeautifulSoup) -> dict:
        """出走表ページからレース情報をパース"""
        info = {
            "race_name": "",
            "grade": "",
            "surface": "",
            "distance": 0,
            "course_type": "",
            "weather": "",
            "track_condition": "",
        }

        # レース名
        name_tag = soup.select_one(".RaceName")
        if name_tag:
            info["race_name"] = name_tag.get_text(strip=True)

        # グレード
        info["grade"] = self._detect_grade(soup, info["race_name"])

        # レース条件: "芝1600m" 等
        data_tag = soup.select_one(".RaceData01") or soup.select_one(".RaceData")
        if data_tag:
            text = data_tag.get_text(" ", strip=True)
            self._parse_condition_text(text, info)

        return info

    def _parse_shutuba_entries(self, soup: BeautifulSoup) -> list[dict]:
        """出走表テーブルをパース"""
        entries = []
        table = soup.find("table", class_=re.compile(r"Shutuba_Table|shutuba"))
        if not table:
            table = soup.find("table", class_=re.compile(r"race_table"))
        if not table:
            return entries

        rows = table.find_all("tr")
        for row in rows:
            cells = row.find_all("td")
            if len(cells) < 5:
                continue

            entry = self._parse_shutuba_row(cells)
            if entry:
                entries.append(entry)

        return entries

    def _parse_shutuba_row(self, cells: list) -> dict | None:
        """出走表の1行をパース"""
        texts = [c.get_text(strip=True) for c in cells]

        # 枠番・馬番を探す
        frame_number = None
        horse_number = None
        for i, t in enumerate(texts[:4]):
            if t.isdigit():
                if frame_number is None:
                    frame_number = int(t)
                elif horse_number is None:
                    horse_number = int(t)
                    break

        if horse_number is None:
            return None

        entry = {
            "finish_position": None,  # 未確定
            "frame_number": frame_number,
            "horse_number": horse_number,
        }

        # 馬名・馬ID
        for cell in cells:
            horse_link = cell.find("a", href=re.compile(r"/horse/"))
            if horse_link:
                entry["horse_name"] = horse_link.get_text(strip=True)
                hid_m = re.search(r"/horse/(\w+)", horse_link.get("href", ""))
                if hid_m:
                    entry["horse_id"] = hid_m.group(1)
                break
        else:
            entry["horse_name"] = ""
            entry["horse_id"] = ""

        # 騎手名・騎手ID
        for cell in cells:
            jockey_link = cell.find("a", href=re.compile(r"/jockey/"))
            if jockey_link:
                entry["jockey_name"] = jockey_link.get_text(strip=True)
                jid_m = re.search(r"/jockey/(?:result/recent/)?(\d+)", jockey_link.get("href", ""))
                entry["jockey_id"] = jid_m.group(1) if jid_m else ""
                break
        else:
            entry["jockey_name"] = ""
            entry["jockey_id"] = ""

        # 調教師
        entry["trainer"] = ""
        for cell in cells:
            trainer_link = cell.find("a", href=re.compile(r"/trainer/"))
            if trainer_link:
                entry["trainer"] = trainer_link.get_text(strip=True)
                break

        # 性齢・斤量: 数字以外のテキストから探す
        entry["sex_age"] = ""
        entry["weight_carry"] = None
        for t in texts:
            if re.match(r"^[牡牝セ]\d+$", t):
                entry["sex_age"] = t
            try:
                v = float(t)
                if 40 < v < 70:  # 斤量の範囲
                    entry["weight_carry"] = v
            except ValueError:
                pass

        # 馬体重
        entry["horse_weight"] = None
        entry["weight_diff"] = None
        for t in texts:
            self._parse_horse_weight(t, entry)
            if entry["horse_weight"]:
                break

        # 出走表には結果系データなし
        entry["finish_time"] = ""
        entry["margin"] = ""
        entry["last_3f"] = None
        entry["odds_win"] = None
        entry["popularity"] = None
        entry["passing"] = ""

        return entry

    # ------------------------------------------------------------------
    # レース結果 (db.netkeiba.com)
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

        # 後半カラムは列数がサイトによって異なるため、内容ベースで検出
        entry["passing"] = ""
        entry["last_3f"] = None
        entry["odds_win"] = None
        entry["popularity"] = None
        entry["horse_weight"] = None
        entry["weight_diff"] = None

        for i in range(8, len(texts)):
            t = texts[i].strip()
            if not t or t == "**":
                continue

            # 通過順: "3-3-2-1" パターン
            if re.match(r"^\d+-\d+", t) and not entry["passing"]:
                entry["passing"] = t
                continue

            # 上がり3F: 30-40秒台の小数
            if entry["last_3f"] is None:
                try:
                    v = float(t)
                    if 30.0 <= v <= 45.0:
                        entry["last_3f"] = v
                        continue
                except ValueError:
                    pass

            # 単勝オッズ: 1.0-999.9の小数
            if entry["odds_win"] is None:
                try:
                    v = float(t)
                    if 1.0 <= v <= 9999.9 and "." in t:
                        entry["odds_win"] = v
                        continue
                except ValueError:
                    pass

            # 人気: 1-18の整数
            if entry["popularity"] is None and t.isdigit():
                v = int(t)
                if 1 <= v <= 30:
                    entry["popularity"] = v
                    continue

            # 馬体重: "480(+2)" パターン
            if entry["horse_weight"] is None:
                self._parse_horse_weight(t, entry)

        # 調教師 (馬体重の次のカラム、あるいはリンクから)
        entry["trainer"] = ""
        # 調教師はリンクから取得を試みる
        for i, cell in enumerate(cells):
            trainer_link = cell.find("a", href=re.compile(r"/trainer/"))
            if trainer_link:
                entry["trainer"] = trainer_link.get_text(strip=True)
                break

        return entry

    # ------------------------------------------------------------------
    # オッズ (race.netkeiba.com)
    # ------------------------------------------------------------------
    def scrape_odds(self, race_id: str) -> list[dict]:
        """単勝・複勝オッズを取得

        Returns:
            [{"bet_type": "win", "combination": "1", "odds_value": 3.5}, ...]
        """
        url = f"{RACE_URL}/odds/index.html?race_id={race_id}&type=b1"
        try:
            soup = self._get(url, encoding="UTF-8")
        except Exception:
            return []

        odds_list = []

        # 単勝オッズテーブル
        for row in soup.select("tr.Odds_Table_Row, table#odds_tan_block tr"):
            cells = row.find_all("td")
            if len(cells) < 2:
                continue
            texts = [c.get_text(strip=True) for c in cells]
            # 馬番と単勝オッズを探す
            num = None
            for t in texts:
                if t.isdigit() and num is None:
                    num = t
            for t in texts:
                try:
                    v = float(t)
                    if num and 1.0 <= v <= 9999.9:
                        odds_list.append({"bet_type": "win", "combination": num, "odds_value": v})
                        break
                except ValueError:
                    pass

        # 取得できなければ結果ページの単勝オッズを使う
        if not odds_list:
            result = self.scrape_race_result(race_id)
            for e in result.get("entries", []):
                if e.get("odds_win") and e.get("horse_number"):
                    odds_list.append({
                        "bet_type": "win",
                        "combination": str(e["horse_number"]),
                        "odds_value": e["odds_win"],
                    })

        return odds_list

    def _parse_horse_weight(self, text: str, entry: dict):
        """馬体重テキストをパース: "480(+2)", "480(-4)", "480(0)", "計不" """
        m = re.match(r"(\d+)\(([+\-]?\d+)\)", text)
        if m:
            entry["horse_weight"] = int(m.group(1))
            entry["weight_diff"] = int(m.group(2))
