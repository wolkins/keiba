"""スクレイピングデータをDBに格納するパーサー (中央競馬)"""
from datetime import datetime

from sqlalchemy.orm import Session

from src.common.database import (
    Horse, Jockey, Odds, Race, RaceEntry, Racecourse, get_session,
)


def store_race_result(session: Session, race_id_str: str, data: dict) -> Race | None:
    """scrape_race_result() の結果をDBに格納

    Args:
        session: SQLAlchemy Session
        race_id_str: 12桁レースID
        data: {"race": {...}, "entries": [...]}

    Returns:
        Race オブジェクト or None
    """
    race_info = data.get("race", {})
    entries_data = data.get("entries", [])

    # 競馬場の特定
    course_code = race_id_str[4:6]
    racecourse = session.query(Racecourse).filter_by(code=course_code).first()
    if not racecourse:
        print(f"  [WARN] 競馬場 code={course_code} が見つかりません")
        return None

    # レース日付: race_id の先頭4桁が年
    year = int(race_id_str[0:4])
    # race_id だけでは正確な日付がわからないため、
    # date_str がデータに含まれない場合はダミー日付を使う
    # (呼び出し側で race_date を渡すことを推奨)
    race_date = data.get("race_date")
    if race_date:
        if isinstance(race_date, str):
            race_date = datetime.strptime(race_date, "%Y-%m-%d").date()
    else:
        # 日付不明の場合は年の1/1をフォールバック
        race_date = datetime(year, 1, 1).date()

    race_number = int(race_id_str[10:12])

    # ------------------------------------------------------------------
    # Race upsert
    # ------------------------------------------------------------------
    race = session.query(Race).filter_by(race_id=race_id_str).first()
    if not race:
        race = Race(
            race_id=race_id_str,
            race_date=race_date,
            racecourse_id=racecourse.id,
            race_number=race_number,
            race_name=race_info.get("race_name", ""),
            grade=race_info.get("grade", ""),
            surface=race_info.get("surface", ""),
            distance=race_info.get("distance"),
            course_type=race_info.get("course_type", ""),
            weather=race_info.get("weather", ""),
            track_condition=race_info.get("track_condition", ""),
            n_runners=len(entries_data),
            status=data.get("status", "finished"),
        )
        session.add(race)
        session.flush()
    else:
        # 既存レースの情報を更新
        race.race_name = race_info.get("race_name") or race.race_name
        race.grade = race_info.get("grade") or race.grade
        race.surface = race_info.get("surface") or race.surface
        race.distance = race_info.get("distance") or race.distance
        race.course_type = race_info.get("course_type") or race.course_type
        race.weather = race_info.get("weather") or race.weather
        race.track_condition = race_info.get("track_condition") or race.track_condition
        race.n_runners = len(entries_data) or race.n_runners
        # scheduled → finished への更新は許可、逆は不可
        new_status = data.get("status", "finished")
        if new_status == "finished" or race.status != "finished":
            race.status = new_status
        if race_date:
            race.race_date = race_date

    # ------------------------------------------------------------------
    # 出走馬の格納
    # ------------------------------------------------------------------
    for e in entries_data:
        horse_number = e.get("horse_number")
        if horse_number is None:
            continue

        # Horse upsert
        horse = _upsert_horse(session, e)

        # Jockey upsert
        jockey = _upsert_jockey(session, e)

        # RaceEntry upsert
        entry = session.query(RaceEntry).filter_by(
            race_id=race.id, horse_number=horse_number
        ).first()

        if not entry:
            entry = RaceEntry(
                race_id=race.id,
                horse_id=horse.id if horse else None,
                jockey_id=jockey.id if jockey else None,
                frame_number=e.get("frame_number"),
                horse_number=horse_number,
                sex_age=e.get("sex_age", ""),
                weight_carry=e.get("weight_carry"),
                horse_weight=e.get("horse_weight"),
                weight_diff=e.get("weight_diff"),
                popularity=e.get("popularity"),
                odds_win=e.get("odds_win"),
                finish_position=e.get("finish_position"),
                finish_time=e.get("finish_time", ""),
                margin=e.get("margin", ""),
                last_3f=e.get("last_3f"),
                passing=e.get("passing", ""),
                win_technique=e.get("win_technique", ""),
            )
            session.add(entry)
        else:
            # 既存エントリーの更新
            entry.horse_id = horse.id if horse else entry.horse_id
            entry.jockey_id = jockey.id if jockey else entry.jockey_id
            entry.frame_number = e.get("frame_number") or entry.frame_number
            entry.sex_age = e.get("sex_age") or entry.sex_age
            entry.weight_carry = e.get("weight_carry") or entry.weight_carry
            entry.horse_weight = e.get("horse_weight") if e.get("horse_weight") is not None else entry.horse_weight
            entry.weight_diff = e.get("weight_diff") if e.get("weight_diff") is not None else entry.weight_diff
            entry.popularity = e.get("popularity") or entry.popularity
            entry.odds_win = e.get("odds_win") or entry.odds_win
            entry.finish_position = e.get("finish_position") if e.get("finish_position") is not None else entry.finish_position
            entry.finish_time = e.get("finish_time") or entry.finish_time
            entry.margin = e.get("margin") if e.get("margin") is not None else entry.margin
            entry.last_3f = e.get("last_3f") or entry.last_3f
            entry.passing = e.get("passing") or entry.passing

    session.commit()
    return race


def _upsert_horse(session: Session, entry_data: dict) -> Horse | None:
    """馬の upsert"""
    horse_id_str = str(entry_data.get("horse_id", "")).strip()
    horse_name = entry_data.get("horse_name", "").strip()

    if not horse_id_str and not horse_name:
        return None

    horse = None
    if horse_id_str:
        horse = session.query(Horse).filter_by(horse_id=horse_id_str).first()

    if horse is None:
        # 性齢から性別を取得
        sex = ""
        sex_age = entry_data.get("sex_age", "")
        if sex_age:
            sex = sex_age[0] if sex_age[0] in ("牡", "牝", "セ") else ""

        horse = Horse(
            horse_id=horse_id_str or f"name_{horse_name}",
            name=horse_name,
            sex=sex,
            trainer=entry_data.get("trainer", ""),
        )
        session.add(horse)
        session.flush()
    else:
        if horse_name:
            horse.name = horse_name
        if entry_data.get("trainer"):
            horse.trainer = entry_data["trainer"]

    return horse


def _upsert_jockey(session: Session, entry_data: dict) -> Jockey | None:
    """騎手の upsert"""
    jockey_id_str = str(entry_data.get("jockey_id", "")).strip()
    jockey_name = entry_data.get("jockey_name", "").strip()

    if not jockey_id_str and not jockey_name:
        return None

    jockey = None
    if jockey_id_str:
        jockey = session.query(Jockey).filter_by(jockey_id=jockey_id_str).first()

    if jockey is None:
        jockey = Jockey(
            jockey_id=jockey_id_str or f"name_{jockey_name}",
            name=jockey_name,
        )
        session.add(jockey)
        session.flush()
    else:
        if jockey_name:
            jockey.name = jockey_name

    return jockey


def store_odds(session: Session, race: Race, odds_list: list[dict]):
    """オッズデータをDBに格納

    Args:
        session: SQLAlchemy Session
        race: Race オブジェクト
        odds_list: [{"bet_type": "win", "combination": "5", "odds_value": 2.5}, ...]
    """
    for o in odds_list:
        if not o.get("combination") or o.get("odds_value") is None:
            continue

        existing = session.query(Odds).filter_by(
            race_id=race.id,
            bet_type=o.get("bet_type", ""),
            combination=o["combination"],
        ).first()

        if not existing:
            session.add(Odds(
                race_id=race.id,
                bet_type=o.get("bet_type", ""),
                combination=o["combination"],
                odds_value=o["odds_value"],
            ))
        else:
            existing.odds_value = o["odds_value"]
            existing.captured_at = datetime.now()

    session.commit()
