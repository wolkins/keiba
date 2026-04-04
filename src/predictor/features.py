"""特徴量エンジニアリング (中央競馬)

競馬特有の要素:
  - 血統(父・母父)による芝/ダート適性
  - 騎手の乗り替わり・相性
  - 枠番×距離の有利不利
  - 脚質(逃げ/先行/差し/追込)と展開予測
  - コース適性(右回り/左回り/直線)
"""
import re

import numpy as np
import pandas as pd
from sqlalchemy import and_, or_
from sqlalchemy.orm import Session

from src.common.database import Horse, Jockey, Odds, Race, RaceEntry, Racecourse


# =============================================================
# 大種牡馬スコア (芝/ダート適性含む)
# =============================================================

SIRE_SCORES = {
    # 芝向き大種牡馬
    "ディープインパクト": (90, 95, 50),  # (総合, 芝適性, ダート適性)
    "キングカメハメハ": (85, 75, 80),
    "ハーツクライ": (80, 85, 55),
    "ロードカナロア": (85, 80, 65),
    "エピファネイア": (80, 85, 50),
    "キタサンブラック": (80, 80, 55),
    "ドゥラメンテ": (80, 80, 60),
    "モーリス": (75, 80, 55),
    "オルフェーヴル": (80, 80, 60),
    "ステイゴールド": (75, 80, 45),
    "サトノクラウン": (70, 75, 50),
    "リアルスティール": (70, 75, 50),
    "スワーヴリチャード": (75, 70, 70),
    # ダート向き
    "ヘニーヒューズ": (75, 35, 90),
    "パイロ": (70, 30, 85),
    "シニスターミニスター": (70, 25, 85),
    "ゴールドアリュール": (70, 30, 85),
    "コパノリッキー": (65, 25, 80),
    "ルヴァンスレーヴ": (65, 30, 80),
    "マインドユアビスケッツ": (65, 30, 80),
    # 万能型
    "サトノダイヤモンド": (70, 70, 65),
    "ジャスタウェイ": (70, 75, 55),
    "ダイワメジャー": (75, 70, 65),
}

_DEFAULT_SIRE_SCORE = (50, 50, 50)


def _sire_score(name: str | None) -> tuple[int, int, int]:
    """種牡馬名からスコアを返す (総合, 芝適性, ダート適性)"""
    if not name:
        return _DEFAULT_SIRE_SCORE
    for key, val in SIRE_SCORES.items():
        if key in name:
            return val
    return _DEFAULT_SIRE_SCORE


# =============================================================
# グレード数値化
# =============================================================

GRADE_MAP = {
    "G1": 10, "G2": 8, "G3": 7, "OP": 6, "L": 5.5,
    "3勝": 5, "2勝": 4, "1勝": 3, "未勝利": 2, "新馬": 1,
}


def _grade_to_num(grade: str | None) -> float:
    if not grade:
        return 3.0
    for key, val in GRADE_MAP.items():
        if key in grade:
            return val
    return 3.0


# =============================================================
# 性別数値化
# =============================================================

SEX_MAP = {"牡": 0, "牝": 1, "セ": 2}


def _sex_to_num(sex_age: str | None) -> tuple[int, int]:
    """性齢文字列 (例: '牡3') から (性別番号, 年齢) を返す"""
    if not sex_age:
        return (0, 4)
    sex_age = sex_age.strip()
    sex_str = sex_age[0] if sex_age else ""
    sex_num = SEX_MAP.get(sex_str, 0)
    age_match = re.search(r"(\d+)", sex_age)
    age = int(age_match.group(1)) if age_match else 4
    return (sex_num, age)


# =============================================================
# タイム文字列→秒変換
# =============================================================

def _time_to_seconds(time_str: str | None) -> float | None:
    """'1:34.5' → 94.5"""
    if not time_str:
        return None
    time_str = time_str.strip()
    m = re.match(r"(\d+):(\d+)\.(\d+)", time_str)
    if m:
        return int(m.group(1)) * 60 + int(m.group(2)) + int(m.group(3)) / 10
    m2 = re.match(r"(\d+)\.(\d+)", time_str)
    if m2:
        return int(m2.group(1)) + int(m2.group(2)) / 10
    return None


# =============================================================
# 脚質判定
# =============================================================

def _judge_running_style(passing: str | None) -> int:
    """通過順から脚質を判定: 1=逃げ, 2=先行, 3=差し, 4=追込"""
    if not passing:
        return 3  # 不明は差し扱い
    parts = passing.replace("-", ",").split(",")
    try:
        first = int(parts[0].strip())
    except (ValueError, IndexError):
        return 3
    if first == 1:
        return 1  # 逃げ
    elif first <= 3:
        return 2  # 先行
    elif first <= 6:
        return 3  # 差し
    else:
        return 4  # 追込


# =============================================================
# 履歴取得 (データリーク防止)
# =============================================================

def _get_horse_history(session: Session, horse_id: int, current_race_id: int,
                       limit: int = 10) -> list[tuple[RaceEntry, Race]]:
    """馬の過去レース結果を取得 (データリーク防止)"""
    current_race = session.query(Race).filter_by(id=current_race_id).first()
    if not current_race:
        return []

    past_entries = (
        session.query(RaceEntry, Race)
        .join(Race, RaceEntry.race_id == Race.id)
        .filter(
            RaceEntry.horse_id == horse_id,
            RaceEntry.finish_position.isnot(None),
            or_(
                Race.race_date < current_race.race_date,
                and_(
                    Race.race_date == current_race.race_date,
                    Race.id < current_race_id,
                ),
            ),
        )
        .order_by(Race.race_date.desc(), Race.id.desc())
        .limit(limit)
        .all()
    )
    return past_entries


# =============================================================
# 騎手履歴取得
# =============================================================

def _get_jockey_history(session: Session, jockey_id: int, current_race_id: int,
                        limit: int = 50) -> list[tuple[RaceEntry, Race]]:
    """騎手の過去レース結果を取得"""
    current_race = session.query(Race).filter_by(id=current_race_id).first()
    if not current_race:
        return []

    past_entries = (
        session.query(RaceEntry, Race)
        .join(Race, RaceEntry.race_id == Race.id)
        .filter(
            RaceEntry.jockey_id == jockey_id,
            RaceEntry.finish_position.isnot(None),
            or_(
                Race.race_date < current_race.race_date,
                and_(
                    Race.race_date == current_race.race_date,
                    Race.id < current_race_id,
                ),
            ),
        )
        .order_by(Race.race_date.desc(), Race.id.desc())
        .limit(limit)
        .all()
    )
    return past_entries


# =============================================================
# 成績特徴量 (直近10走)
# =============================================================

def _calc_performance_features(history: list[tuple[RaceEntry, Race]]) -> dict:
    """直近走成績からトレンド特徴量を算出"""
    features = {
        "recent_races_count": len(history),
        "recent_avg_finish": 0.0,
        "recent_win_rate": 0.0,
        "recent_top2_rate": 0.0,
        "recent_top3_rate": 0.0,
        "finish_trend_slope": 0.0,
    }

    if not history:
        return features

    positions = [e.finish_position for e, r in history if e.finish_position]
    if not positions:
        return features

    weights = [0.9 ** i for i in range(len(positions))]
    features["recent_avg_finish"] = float(np.average(positions, weights=weights))
    features["recent_win_rate"] = sum(1 for p in positions if p == 1) / len(positions)
    features["recent_top2_rate"] = sum(1 for p in positions if p <= 2) / len(positions)
    features["recent_top3_rate"] = sum(1 for p in positions if p <= 3) / len(positions)

    if len(positions) >= 3:
        x = np.arange(len(positions), dtype=float)
        slope = np.polyfit(x, positions, 1)[0]
        features["finish_trend_slope"] = float(slope)

    return features


# =============================================================
# タイム特徴量
# =============================================================

def _calc_time_features(entry: RaceEntry, race: Race,
                        history: list[tuple[RaceEntry, Race]]) -> dict:
    """上がり3F, タイムから特徴量を算出"""
    features = {
        "last_3f": entry.last_3f or 0.0,
        "recent_avg_last_3f": 0.0,
        "last_3f_trend": 0.0,
        "recent_avg_time_per_m": 0.0,  # 距離別標準化タイム
    }

    last_3fs = [e.last_3f for e, r in history if e.last_3f and e.last_3f > 0]
    if last_3fs:
        weights = [0.9 ** i for i in range(len(last_3fs))]
        features["recent_avg_last_3f"] = float(np.average(last_3fs, weights=weights))
        if len(last_3fs) >= 3:
            x = np.arange(len(last_3fs), dtype=float)
            slope = np.polyfit(x, last_3fs, 1)[0]
            features["last_3f_trend"] = float(slope)  # 負=改善

    # 距離別標準化タイム (秒/m)
    time_per_m_list = []
    for e, r in history:
        t = _time_to_seconds(e.finish_time)
        if t and r.distance and r.distance > 0:
            time_per_m_list.append(t / r.distance)
    if time_per_m_list:
        features["recent_avg_time_per_m"] = float(np.mean(time_per_m_list))

    return features


# =============================================================
# 血統特徴量
# =============================================================

def _calc_bloodline_features(horse: Horse | None, surface: str | None) -> dict:
    """血統(父・母父)から特徴量を算出"""
    features = {
        "sire_score": 50.0,
        "sire_turf_apt": 50.0,
        "sire_dirt_apt": 50.0,
        "broodmare_sire_score": 50.0,
        "bloodline_surface_apt": 0.0,  # 父×surface適性
    }

    if not horse:
        return features

    s_total, s_turf, s_dirt = _sire_score(horse.father)
    features["sire_score"] = float(s_total)
    features["sire_turf_apt"] = float(s_turf)
    features["sire_dirt_apt"] = float(s_dirt)

    ms_total, ms_turf, ms_dirt = _sire_score(horse.mother_father)
    features["broodmare_sire_score"] = float(ms_total)

    # 父×surface適性
    if surface and "芝" in surface:
        features["bloodline_surface_apt"] = (s_turf + ms_turf) / 2
    elif surface and "ダ" in surface:
        features["bloodline_surface_apt"] = (s_dirt + ms_dirt) / 2
    else:
        features["bloodline_surface_apt"] = (s_total + ms_total) / 2

    return features


# =============================================================
# 騎手特徴量
# =============================================================

def _calc_jockey_features(entry: RaceEntry, race: Race,
                          jockey_history: list[tuple[RaceEntry, Race]],
                          horse_history: list[tuple[RaceEntry, Race]]) -> dict:
    """騎手の成績・相性を算出"""
    features = {
        "jockey_win_rate": 0.0,
        "jockey_top3_rate": 0.0,
        "jockey_surface_win_rate": 0.0,
        "jockey_distance_win_rate": 0.0,
        "jockey_change": 0,  # 乗り替わりフラグ
    }

    if jockey_history:
        positions = [e.finish_position for e, r in jockey_history if e.finish_position]
        if positions:
            features["jockey_win_rate"] = sum(1 for p in positions if p == 1) / len(positions)
            features["jockey_top3_rate"] = sum(1 for p in positions if p <= 3) / len(positions)

        # 同surface成績
        surface = (race.surface or "").strip()
        if surface:
            same_surface = [e.finish_position for e, r in jockey_history
                           if e.finish_position and r.surface and surface in r.surface]
            if same_surface:
                features["jockey_surface_win_rate"] = sum(1 for p in same_surface if p == 1) / len(same_surface)

        # 同距離帯成績 (±200m)
        dist = race.distance or 0
        if dist > 0:
            same_dist = [e.finish_position for e, r in jockey_history
                        if e.finish_position and r.distance
                        and abs(r.distance - dist) <= 200]
            if same_dist:
                features["jockey_distance_win_rate"] = sum(1 for p in same_dist if p == 1) / len(same_dist)

    # 乗り替わり判定
    if horse_history:
        prev_entry, prev_race = horse_history[0]
        if prev_entry.jockey_id and entry.jockey_id:
            if prev_entry.jockey_id != entry.jockey_id:
                features["jockey_change"] = 1

    return features


# =============================================================
# コース適性
# =============================================================

def _calc_course_features(race: Race, history: list[tuple[RaceEntry, Race]]) -> dict:
    """コース適性: 同競馬場/同距離/同surface/回り方向"""
    features = {
        "same_course_avg": 0.0,
        "same_course_top3": 0.0,
        "same_distance_avg": 0.0,
        "same_distance_top3": 0.0,
        "same_surface_avg": 0.0,
        "same_surface_top3": 0.0,
        "same_direction_avg": 0.0,
    }

    if not history:
        return features

    surface = (race.surface or "").strip()
    distance = race.distance or 0
    course_type = (race.course_type or "").strip()
    racecourse_id = race.racecourse_id

    # 同競馬場
    same_course = [e.finish_position for e, r in history
                   if e.finish_position and r.racecourse_id == racecourse_id]
    if same_course:
        features["same_course_avg"] = float(np.mean(same_course))
        features["same_course_top3"] = sum(1 for p in same_course if p <= 3) / len(same_course)

    # 同距離 (±200m)
    if distance > 0:
        same_dist = [e.finish_position for e, r in history
                     if e.finish_position and r.distance
                     and abs(r.distance - distance) <= 200]
        if same_dist:
            features["same_distance_avg"] = float(np.mean(same_dist))
            features["same_distance_top3"] = sum(1 for p in same_dist if p <= 3) / len(same_dist)

    # 同surface
    if surface:
        same_surf = [e.finish_position for e, r in history
                     if e.finish_position and r.surface and surface in r.surface]
        if same_surf:
            features["same_surface_avg"] = float(np.mean(same_surf))
            features["same_surface_top3"] = sum(1 for p in same_surf if p <= 3) / len(same_surf)

    # 同回り方向
    if course_type:
        same_dir = [e.finish_position for e, r in history
                    if e.finish_position and r.course_type and course_type in r.course_type]
        if same_dir:
            features["same_direction_avg"] = float(np.mean(same_dir))

    return features


# =============================================================
# 馬場・天候適性
# =============================================================

def _calc_track_condition_features(race: Race, history: list[tuple[RaceEntry, Race]]) -> dict:
    """馬場状態別の成績"""
    features = {
        "good_track_avg": 0.0,
        "good_track_top3": 0.0,
        "heavy_track_avg": 0.0,
        "heavy_track_top3": 0.0,
        "track_pref_diff": 0.0,  # 良走路と重馬場の成績差
    }

    if not history:
        return features

    good_pos = []
    heavy_pos = []
    for e, r in history:
        if not e.finish_position:
            continue
        cond = (r.track_condition or "").strip()
        if cond in ("良", "稍重"):
            good_pos.append(e.finish_position)
        elif cond in ("重", "不良"):
            heavy_pos.append(e.finish_position)

    if good_pos:
        features["good_track_avg"] = float(np.mean(good_pos))
        features["good_track_top3"] = sum(1 for p in good_pos if p <= 3) / len(good_pos)
    if heavy_pos:
        features["heavy_track_avg"] = float(np.mean(heavy_pos))
        features["heavy_track_top3"] = sum(1 for p in heavy_pos if p <= 3) / len(heavy_pos)

    # 馬場適性差 (良走路の方が良い = 正)
    if good_pos and heavy_pos:
        features["track_pref_diff"] = features["heavy_track_avg"] - features["good_track_avg"]

    return features


# =============================================================
# 展開予測
# =============================================================

def _calc_pace_features(entry: RaceEntry, race: Race,
                        history: list[tuple[RaceEntry, Race]],
                        all_entries: list[RaceEntry]) -> dict:
    """脚質判定とペース予測"""
    # 自分の脚質判定 (直近走の通過順から)
    styles = []
    for e, r in history:
        s = _judge_running_style(e.passing)
        styles.append(s)

    if styles:
        my_style = int(round(np.mean(styles[:3])))  # 直近3走の平均
    else:
        my_style = 3  # 不明は差し

    # レース内各脚質の頭数 (全出走馬の前走を見る)
    style_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    for e in all_entries:
        s = _judge_running_style(e.passing)
        style_counts[s] = style_counts.get(s, 0) + 1

    n_runners = race.n_runners or len(all_entries)
    pace_pressure = (style_counts.get(1, 0) + style_counts.get(2, 0)) / max(1, n_runners)

    features = {
        "running_style": my_style,
        "is_front_runner": 1 if my_style <= 2 else 0,
        "is_closer": 1 if my_style >= 3 else 0,
        "same_style_count": style_counts.get(my_style, 0),
        "pace_pressure": pace_pressure,  # 前に行く馬の割合(高い=ハイペース予測)
    }

    return features


# =============================================================
# レース情報
# =============================================================

def _calc_race_info_features(entry: RaceEntry, race: Race) -> dict:
    """レース属性の特徴量"""
    n_runners = race.n_runners or 0
    distance = race.distance or 0
    frame = entry.frame_number or 0

    # 枠番有利度: 短距離は内枠有利、長距離は影響少
    if distance > 0 and frame > 0 and n_runners > 0:
        frame_advantage = (1 - (frame - 1) / max(1, 8 - 1)) * max(0, (2000 - distance) / 1000)
    else:
        frame_advantage = 0.0

    # 馬場状態のエンコード
    cond = (race.track_condition or "").strip()
    track_cond_num = {"良": 0, "稍重": 1, "重": 2, "不良": 3}.get(cond, 0)

    # surface
    surface = (race.surface or "").strip()
    is_turf = 1 if "芝" in surface else 0
    is_dirt = 1 if "ダ" in surface else 0

    # 天候
    weather = (race.weather or "").strip()
    weather_num = {"晴": 0, "曇": 1, "小雨": 2, "雨": 3, "雪": 4}.get(weather, 0)
    is_rainy = 1 if weather in ("小雨", "雨", "雪") else 0

    return {
        "grade_num": _grade_to_num(race.grade),
        "n_runners": n_runners,
        "distance": distance,
        "distance_category": _distance_category(distance),
        "frame_number": frame,
        "horse_number": entry.horse_number,
        "frame_advantage": frame_advantage,
        "track_condition_num": track_cond_num,
        "is_turf": is_turf,
        "is_dirt": is_dirt,
        "weather_num": weather_num,
        "is_rainy": is_rainy,
    }


def _distance_category(distance: int) -> int:
    """距離カテゴリ: 1=スプリント, 2=マイル, 3=中距離, 4=長距離"""
    if distance <= 1400:
        return 1
    elif distance <= 1800:
        return 2
    elif distance <= 2200:
        return 3
    else:
        return 4


# =============================================================
# 交互作用
# =============================================================

def _calc_interaction_features(f: dict) -> dict:
    """既存特徴量の交差項"""
    return {
        # 脚質×馬場状態
        "style_x_track_cond": f.get("running_style", 3) * f.get("track_condition_num", 0),
        # 血統×surface
        "bloodline_x_surface": f.get("bloodline_surface_apt", 50) * f.get("is_turf", 0)
                               + f.get("bloodline_surface_apt", 50) * f.get("is_dirt", 0),
        # 枠番×距離カテゴリ
        "frame_x_distance": f.get("frame_advantage", 0) * f.get("distance_category", 2),
        # 騎手勝率×馬の直近成績
        "jockey_x_horse_form": f.get("jockey_win_rate", 0) * (10 - min(10, f.get("recent_avg_finish", 5))),
        # ペース圧×脚質
        "pace_x_style": f.get("pace_pressure", 0.5) * (5 - f.get("running_style", 3)),
        # 天候×馬場適性(雨天時に重馬場得意馬が有利)
        "weather_x_track_pref": f.get("is_rainy", 0) * f.get("heavy_track_top3", 0),
    }


# =============================================================
# メイン特徴量構築
# =============================================================

def build_features_for_race(session: Session, race: Race) -> pd.DataFrame:
    """レースの全エントリーから特徴量DataFrameを生成"""
    entries = session.query(RaceEntry).filter_by(race_id=race.id).all()
    racecourse = session.query(Racecourse).filter_by(id=race.racecourse_id).first()

    if not entries:
        return pd.DataFrame()

    rows = []
    for entry in entries:
        horse = session.query(Horse).filter_by(id=entry.horse_id).first() if entry.horse_id else None
        jockey = session.query(Jockey).filter_by(id=entry.jockey_id).first() if entry.jockey_id else None

        horse_history = _get_horse_history(session, entry.horse_id, race.id, limit=10) if entry.horse_id else []
        jockey_history = _get_jockey_history(session, entry.jockey_id, race.id, limit=50) if entry.jockey_id else []

        row = _build_entry_features(entry, horse, jockey, race, racecourse,
                                    entries, horse_history, jockey_history, session)
        rows.append(row)

    df = pd.DataFrame(rows)
    df = _add_relative_features(df)
    return df


def _build_entry_features(entry: RaceEntry, horse: Horse | None,
                           jockey: Jockey | None, race: Race,
                           racecourse: Racecourse | None,
                           all_entries: list[RaceEntry],
                           horse_history: list[tuple[RaceEntry, Race]],
                           jockey_history: list[tuple[RaceEntry, Race]],
                           session: Session) -> dict:
    """1エントリーの全特徴量を生成"""
    features = {"horse_number": entry.horse_number}

    # === 馬基本 ===
    sex_num, age = _sex_to_num(entry.sex_age)
    features["sex"] = sex_num
    features["age"] = age
    features["weight_carry"] = entry.weight_carry or 0.0
    features["horse_weight"] = entry.horse_weight or 0
    features["weight_diff"] = entry.weight_diff or 0

    # === 成績(直近10走) ===
    features.update(_calc_performance_features(horse_history))

    # === タイム ===
    features.update(_calc_time_features(entry, race, horse_history))

    # === 血統 ===
    features.update(_calc_bloodline_features(horse, race.surface))

    # === 騎手 ===
    features.update(_calc_jockey_features(entry, race, jockey_history, horse_history))

    # === コース適性 ===
    features.update(_calc_course_features(race, horse_history))

    # === 馬場・天候 ===
    features.update(_calc_track_condition_features(race, horse_history))

    # === 展開予測 ===
    features.update(_calc_pace_features(entry, race, horse_history, all_entries))

    # === レース情報 ===
    features.update(_calc_race_info_features(entry, race))

    # === 交互作用 ===
    features.update(_calc_interaction_features(features))

    # === ターゲット ===
    if entry.finish_position is not None:
        features["finish_position"] = entry.finish_position

    return features


# =============================================================
# レース内相対特徴量
# =============================================================

def _add_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    """レース内相対評価"""
    if df.empty:
        return df

    # 上がり3F rank / z-score
    if "last_3f" in df.columns:
        valid = df["last_3f"] > 0
        if valid.any():
            df["last_3f_rank"] = df.loc[valid, "last_3f"].rank(method="min")
            mean_v = df.loc[valid, "last_3f"].mean()
            std_v = df.loc[valid, "last_3f"].std()
            df["last_3f_z"] = (df["last_3f"] - mean_v) / std_v if std_v > 0 else 0
        else:
            df["last_3f_rank"] = 8.0
            df["last_3f_z"] = 0.0
        df["last_3f_rank"] = df["last_3f_rank"].fillna(8.0)
        df["last_3f_z"] = df["last_3f_z"].fillna(0.0)

    # 斤量 rank
    if "weight_carry" in df.columns:
        df["weight_carry_rank"] = df["weight_carry"].rank(method="min")

    # 馬体重 rank
    if "horse_weight" in df.columns:
        valid = df["horse_weight"] > 0
        if valid.any():
            df["horse_weight_rank"] = df.loc[valid, "horse_weight"].rank(method="min")
        else:
            df["horse_weight_rank"] = 8.0
        df["horse_weight_rank"] = df["horse_weight_rank"].fillna(8.0)

    # 直近成績 rank
    if "recent_avg_finish" in df.columns:
        valid = df["recent_avg_finish"] > 0
        if valid.any():
            df["form_rank"] = df.loc[valid, "recent_avg_finish"].rank(method="min")
        else:
            df["form_rank"] = 8.0
        df["form_rank"] = df["form_rank"].fillna(8.0)

    # 騎手勝率 rank
    if "jockey_win_rate" in df.columns:
        df["jockey_rank"] = df["jockey_win_rate"].rank(ascending=False, method="min")

    return df


# =============================================================
# FEATURE_COLUMNS
# =============================================================

FEATURE_COLUMNS = [
    # 馬基本 (5)
    "sex", "age", "weight_carry", "horse_weight", "weight_diff",
    # 成績・直近10走 (6)
    "recent_races_count", "recent_avg_finish", "recent_win_rate",
    "recent_top2_rate", "recent_top3_rate", "finish_trend_slope",
    # タイム (4)
    "last_3f", "recent_avg_last_3f", "last_3f_trend", "recent_avg_time_per_m",
    # 血統 (5)
    "sire_score", "sire_turf_apt", "sire_dirt_apt",
    "broodmare_sire_score", "bloodline_surface_apt",
    # 騎手 (5)
    "jockey_win_rate", "jockey_top3_rate", "jockey_surface_win_rate",
    "jockey_distance_win_rate", "jockey_change",
    # コース適性 (7)
    "same_course_avg", "same_course_top3",
    "same_distance_avg", "same_distance_top3",
    "same_surface_avg", "same_surface_top3",
    "same_direction_avg",
    # 馬場・天候 (5)
    "good_track_avg", "good_track_top3",
    "heavy_track_avg", "heavy_track_top3",
    "track_pref_diff",
    # 展開予測 (5)
    "running_style", "is_front_runner", "is_closer",
    "same_style_count", "pace_pressure",
    # レース情報 (12)
    "grade_num", "n_runners", "distance", "distance_category",
    "frame_number", "horse_number", "frame_advantage",
    "track_condition_num", "is_turf", "is_dirt",
    "weather_num", "is_rainy",
    # レース内相対 (7)
    "last_3f_rank", "last_3f_z",
    "weight_carry_rank", "horse_weight_rank",
    "form_rank", "jockey_rank",
    # 交互作用 (6)
    "style_x_track_cond", "bloodline_x_surface",
    "frame_x_distance", "jockey_x_horse_form", "pace_x_style",
    "weather_x_track_pref",
]
# 合計: 67個
