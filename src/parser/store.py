"""スクレイピングデータをDBに格納するパーサー (中央競馬)"""
import json
from datetime import datetime

from sqlalchemy import and_, func
from sqlalchemy.orm import Session

from src.common.database import (
    Horse, Jockey, Odds, OddsSnapshot, Race, RaceEntry, Racecourse,
    Win5PayoutHistory, Win5Run, Win5TargetRace,
    WIN5_OPTIMIZER_VERSION, WIN5_RUN_MODE_HIT, WIN5_TARGET_SOURCE_JRA,
    WIN5_TARGET_STATUS_SCHEDULED, WIN5_UNIT_PRICE,
    get_session,
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

    Odds (最新値) を upsert しつつ、OddsSnapshot に同時点のスナップショットを append する。
    同一 captured_at の重複書き込みは握り潰す。

    Args:
        session: SQLAlchemy Session
        race: Race オブジェクト
        odds_list: [{"bet_type": "win", "combination": "5", "odds_value": 2.5}, ...]
    """
    captured_at = datetime.now()

    for o in odds_list:
        if not o.get("combination") or o.get("odds_value") is None:
            continue

        bet_type = o.get("bet_type", "")
        combination = o["combination"]
        odds_value = o["odds_value"]

        existing = session.query(Odds).filter_by(
            race_id=race.id,
            bet_type=bet_type,
            combination=combination,
        ).first()

        if not existing:
            session.add(Odds(
                race_id=race.id,
                bet_type=bet_type,
                combination=combination,
                odds_value=odds_value,
                captured_at=captured_at,
            ))
        else:
            existing.odds_value = odds_value
            existing.captured_at = captured_at

        snapshot_exists = session.query(OddsSnapshot).filter_by(
            race_id=race.id,
            bet_type=bet_type,
            combination=combination,
            captured_at=captured_at,
        ).first()
        if not snapshot_exists:
            session.add(OddsSnapshot(
                race_id=race.id,
                bet_type=bet_type,
                combination=combination,
                odds_value=odds_value,
                captured_at=captured_at,
            ))

    session.commit()


def store_win5_target_races(
    session: Session,
    targets: list[dict],
) -> list[Win5TargetRace]:
    """WIN5 対象5レースを DB に保存 (upsert)

    Args:
        session: SQLAlchemy Session
        targets: Win5Scraper.scrape_target_races() の戻り値形式

    Returns:
        保存 (or 更新) された Win5TargetRace のリスト
    """
    saved = []
    for t in targets:
        race_date = t["race_date"]
        if isinstance(race_date, str):
            race_date = datetime.strptime(race_date, "%Y-%m-%d").date()

        existing = session.query(Win5TargetRace).filter_by(
            race_date=race_date,
            leg_index=t["leg_index"],
        ).first()

        # Race 内部PK の解決 (race_date + racecourse_code + race_number で突合)
        race_pk = None
        if t.get("racecourse_code"):
            racecourse = session.query(Racecourse).filter_by(code=t["racecourse_code"]).first()
            if racecourse:
                race = session.query(Race).filter_by(
                    race_date=race_date,
                    racecourse_id=racecourse.id,
                    race_number=t["race_number"],
                ).first()
                if race:
                    race_pk = race.id

        if not existing:
            target = Win5TargetRace(
                race_date=race_date,
                leg_index=t["leg_index"],
                racecourse_name=t["racecourse_name"],
                racecourse_code=t.get("racecourse_code"),
                race_number=t["race_number"],
                post_time=t.get("post_time"),
                close_time=t.get("close_time"),
                race_id=race_pk,
                source=t.get("source", WIN5_TARGET_SOURCE_JRA),
                source_url=t.get("source_url"),
                status=WIN5_TARGET_STATUS_SCHEDULED,
            )
            session.add(target)
            saved.append(target)
        else:
            existing.racecourse_name = t["racecourse_name"]
            existing.racecourse_code = t.get("racecourse_code") or existing.racecourse_code
            existing.race_number = t["race_number"]
            existing.post_time = t.get("post_time") or existing.post_time
            existing.close_time = t.get("close_time") or existing.close_time
            if race_pk is not None:
                existing.race_id = race_pk
            existing.source = t.get("source", existing.source)
            existing.source_url = t.get("source_url") or existing.source_url
            saved.append(existing)

    session.commit()
    return saved


def resolve_win5_race_links(session: Session, race_date) -> int:
    """Win5TargetRace のうち race_id 未解決の行について、Race との紐付けを試みる

    スクレイプ当時に Race が未取得でも、後から race データを scrape した後に
    このヘルパを呼べば紐付けが埋まる。

    Returns:
        新規に紐付けに成功した件数
    """
    if isinstance(race_date, str):
        race_date = datetime.strptime(race_date, "%Y-%m-%d").date()

    targets = session.query(Win5TargetRace).filter(
        Win5TargetRace.race_date == race_date,
        Win5TargetRace.race_id.is_(None),
    ).all()

    resolved = 0
    for t in targets:
        if not t.racecourse_code:
            continue
        racecourse = session.query(Racecourse).filter_by(code=t.racecourse_code).first()
        if not racecourse:
            continue
        race = session.query(Race).filter_by(
            race_date=race_date,
            racecourse_id=racecourse.id,
            race_number=t.race_number,
        ).first()
        if race:
            t.race_id = race.id
            resolved += 1

    if resolved:
        session.commit()
    return resolved


def store_win5_payout_history(
    session: Session,
    record: dict,
) -> Win5PayoutHistory:
    """WIN5 払戻履歴を1件 upsert (race_date UNIQUE)

    Args:
        record: 以下のキーを受け付ける (全て optional, race_date は必須)
            race_date (str or date) 必須
            winning_combination: "3-7-12-5-9"
            race_ids: [int, int, int, int, int]
            total_sales: int
            winning_tickets: int
            payout_per_ticket: int
            carryover_in: int
            carryover_out: int
            jackpot_flag: bool
            source: str (default "manual")
            source_url: str
            notes: str
    """
    race_date = record.get("race_date")
    if race_date is None:
        raise ValueError("race_date is required")
    if isinstance(race_date, str):
        race_date = datetime.strptime(race_date, "%Y-%m-%d").date()

    existing = session.query(Win5PayoutHistory).filter_by(race_date=race_date).first()
    race_ids_json = (
        json.dumps(record["race_ids"])
        if record.get("race_ids") is not None else None
    )

    if existing:
        for field in [
            "winning_combination", "total_sales", "winning_tickets",
            "payout_per_ticket", "carryover_in", "carryover_out",
            "jackpot_flag", "source", "source_url", "notes",
        ]:
            if field in record and record[field] is not None:
                setattr(existing, field, record[field])
        if race_ids_json is not None:
            existing.race_ids_json = race_ids_json
        row = existing
    else:
        row = Win5PayoutHistory(
            race_date=race_date,
            winning_combination=record.get("winning_combination"),
            race_ids_json=race_ids_json,
            total_sales=record.get("total_sales"),
            winning_tickets=record.get("winning_tickets"),
            payout_per_ticket=record.get("payout_per_ticket"),
            carryover_in=record.get("carryover_in", 0),
            carryover_out=record.get("carryover_out", 0),
            jackpot_flag=record.get("jackpot_flag", False),
            source=record.get("source", "manual"),
            source_url=record.get("source_url"),
            notes=record.get("notes"),
        )
        session.add(row)

    session.commit()
    return row


def store_win5_run(
    session: Session,
    race_date,
    recommendation,
    target_races: list[Win5TargetRace],
    as_of: datetime | None = None,
) -> Win5Run:
    """WIN5 推奨買い目を Win5Run として保存 (append-only)

    Args:
        session: DBセッション
        race_date: date
        recommendation: Win5Recommendation
        target_races: この run が参照した Win5TargetRace 5件
        as_of: 実行時刻。None なら now()

    Returns:
        保存された Win5Run
    """
    if isinstance(race_date, str):
        race_date = datetime.strptime(race_date, "%Y-%m-%d").date()
    if as_of is None:
        as_of = datetime.now()

    tickets_json = json.dumps([
        {
            "horse_numbers": t.horse_numbers,
            "combination_key": t.combination_key,
            "prob": t.combo_probability,
            "amount": t.amount,
        }
        for t in recommendation.tickets
    ], ensure_ascii=False)

    target_races_json = json.dumps([
        {
            "leg_index": t.leg_index,
            "racecourse_code": t.racecourse_code,
            "racecourse_name": t.racecourse_name,
            "race_number": t.race_number,
            "race_id": t.race_id,
            "post_time": t.post_time,
        }
        for t in sorted(target_races, key=lambda x: x.leg_index)
    ], ensure_ascii=False)

    run = Win5Run(
        race_date=race_date,
        as_of=as_of,
        mode=recommendation.mode,
        budget=recommendation.budget,
        unit_amount=recommendation.unit_amount,
        total_tickets=recommendation.total_tickets,
        total_cost=recommendation.total_cost,
        hit_probability_sum=recommendation.hit_probability_sum,
        top_ticket_probability=recommendation.top_ticket_probability,
        coverage_threshold=recommendation.coverage_threshold,
        optimizer_version=WIN5_OPTIMIZER_VERSION,
        tickets_json=tickets_json,
        target_races_json=target_races_json,
    )
    session.add(run)
    session.commit()
    return run


def get_odds_snapshot(
    session: Session,
    race_id: int,
    bet_type: str = "win",
    as_of: datetime | None = None,
) -> dict[str, float]:
    """as_of 時点での最新オッズスナップショットを返す

    同一 (race_id, bet_type, combination) について captured_at <= as_of の中で
    最大の captured_at を持つ行を採用する。
    as_of=None の場合は全期間中の最新。

    Args:
        session: SQLAlchemy Session
        race_id: Race.id (内部 PK)
        bet_type: "win" / "place" 等
        as_of: 基準時刻。None なら最新。

    Returns:
        {"1": 3.2, "5": 12.5, ...}  combination -> odds_value
    """
    filters = [
        OddsSnapshot.race_id == race_id,
        OddsSnapshot.bet_type == bet_type,
    ]
    if as_of is not None:
        filters.append(OddsSnapshot.captured_at <= as_of)

    latest_at = (
        session.query(
            OddsSnapshot.combination.label("combination"),
            func.max(OddsSnapshot.captured_at).label("latest_at"),
        )
        .filter(and_(*filters))
        .group_by(OddsSnapshot.combination)
        .subquery()
    )

    rows = (
        session.query(OddsSnapshot)
        .join(
            latest_at,
            and_(
                OddsSnapshot.combination == latest_at.c.combination,
                OddsSnapshot.captured_at == latest_at.c.latest_at,
            ),
        )
        .filter(
            OddsSnapshot.race_id == race_id,
            OddsSnapshot.bet_type == bet_type,
        )
        .all()
    )

    return {row.combination: row.odds_value for row in rows}
