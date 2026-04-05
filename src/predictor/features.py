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
# ローテーション・間隔
# =============================================================

def _calc_rotation_features(race: Race, history: list[tuple[RaceEntry, Race]]) -> dict:
    """前走間隔・休み明け・距離変更"""
    features = {
        "days_since_last": 60.0,  # デフォルト=休み明け相当
        "rest_bucket": 3,  # 0=連闘, 1=中1-2週, 2=中3-8週, 3=休み明け
        "is_fresh": 0,  # 休み明けフラグ(中10週以上)
        "is_second_up": 0,  # 叩き2走目フラグ
        "distance_change": 0,  # 前走比距離増減(m)
        "surface_switch": 0,  # 芝ダ替わりフラグ
    }

    if not history:
        return features

    prev_entry, prev_race = history[0]
    if race.race_date and prev_race.race_date:
        days = (race.race_date - prev_race.race_date).days
        features["days_since_last"] = float(days)
        if days <= 8:
            features["rest_bucket"] = 0
        elif days <= 21:
            features["rest_bucket"] = 1
        elif days <= 63:
            features["rest_bucket"] = 2
        else:
            features["rest_bucket"] = 3
            features["is_fresh"] = 1

    # 叩き2走目判定
    if len(history) >= 2:
        prev2_entry, prev2_race = history[1]
        if prev_race.race_date and prev2_race.race_date:
            gap_before = (prev_race.race_date - prev2_race.race_date).days
            if gap_before > 70:  # 前走が休み明けだった
                features["is_second_up"] = 1

    # 距離変更
    if race.distance and prev_race.distance:
        features["distance_change"] = race.distance - prev_race.distance

    # 芝ダ替わり
    cur_surface = (race.surface or "").strip()
    prev_surface = (prev_race.surface or "").strip()
    if cur_surface and prev_surface and cur_surface != prev_surface:
        features["surface_switch"] = 1

    return features


# =============================================================
# スピード指数
# =============================================================

def _calc_speed_features(history: list[tuple[RaceEntry, Race]]) -> dict:
    """走破タイムベースのスピード指数"""
    features = {
        "speed_figure_last": 0.0,
        "speed_figure_best3": 0.0,
        "speed_figure_avg3": 0.0,
        "margin_to_winner_avg": 0.0,  # 勝ち馬とのタイム差平均
        "finish_std": 0.0,  # 着順安定性
    }

    if not history:
        return features

    # スピード指数: 距離・馬場補正した標準化タイム
    # 基準: 芝2000m良=120秒, ダート1800m良=112秒 として偏差値化
    speed_figs = []
    for e, r in history:
        t = _time_to_seconds(e.finish_time)
        if t and r.distance and r.distance > 0:
            # 基準タイム(距離比例で推定)
            base_time = r.distance * 0.06  # 芝良の近似基準
            cond = (r.track_condition or "").strip()
            cond_adj = {"良": 0, "稍重": 0.5, "重": 1.5, "不良": 2.5}.get(cond, 0)
            adjusted_time = t - cond_adj
            # 速いほど高い指数
            fig = (base_time - adjusted_time) / base_time * 100 + 50
            speed_figs.append(fig)

    if speed_figs:
        features["speed_figure_last"] = speed_figs[0]
        top3 = sorted(speed_figs, reverse=True)[:3]
        features["speed_figure_best3"] = top3[0]
        features["speed_figure_avg3"] = float(np.mean(speed_figs[:3]))

    # 着順安定性
    positions = [e.finish_position for e, r in history if e.finish_position]
    if len(positions) >= 3:
        features["finish_std"] = float(np.std(positions))

    return features


_winner_time_cache: dict[int, float | None] = {}
_winner_cache_computed = False


def _precompute_winner_times(session: Session, before_date):
    """全レースの勝ち馬タイムを事前にキャッシュ"""
    global _winner_time_cache, _winner_cache_computed
    if _winner_cache_computed:
        return

    winners = (
        session.query(RaceEntry.race_id, RaceEntry.finish_time)
        .join(Race, RaceEntry.race_id == Race.id)
        .filter(
            RaceEntry.finish_position == 1,
            RaceEntry.finish_time.isnot(None),
            Race.race_date < before_date,
        )
        .all()
    )

    for race_id, finish_time in winners:
        _winner_time_cache[race_id] = _time_to_seconds(finish_time)

    _winner_cache_computed = True


def _calc_margin_features(session: Session, history: list[tuple[RaceEntry, Race]]) -> dict:
    """勝ち馬とのタイム差(キャッシュ版)"""
    margins = []
    for e, r in history[:5]:
        my_time = _time_to_seconds(e.finish_time)
        if not my_time:
            continue
        winner_time = _winner_time_cache.get(r.id)
        if winner_time:
            margins.append(my_time - winner_time)

    return {
        "margin_to_winner_avg": float(np.mean(margins)) if margins else 0.0,
    }


# =============================================================
# 厩舎特徴量 (キャッシュ版)
# =============================================================

_trainer_stats_cache: dict[str, tuple[float, float]] = {}
_trainer_jockey_cache: dict[tuple[str, int], float] = {}
_trainer_cache_computed = False


def _precompute_trainer_stats(session: Session, before_date):
    """全厩舎の成績を事前に一括計算"""
    global _trainer_stats_cache, _trainer_jockey_cache, _trainer_cache_computed
    if _trainer_cache_computed:
        return

    from collections import defaultdict

    results = (
        session.query(
            Horse.trainer,
            RaceEntry.finish_position,
            RaceEntry.jockey_id,
        )
        .join(Horse, RaceEntry.horse_id == Horse.id)
        .join(Race, RaceEntry.race_id == Race.id)
        .filter(
            RaceEntry.finish_position.isnot(None),
            Horse.trainer.isnot(None),
            Race.race_date < before_date,
        )
        .all()
    )

    trainer_positions = defaultdict(list)
    combo_positions = defaultdict(list)

    for trainer, finish_pos, jockey_id in results:
        trainer = (trainer or "").strip()
        if not trainer:
            continue
        trainer_positions[trainer].append(finish_pos)
        if jockey_id:
            combo_positions[(trainer, jockey_id)].append(finish_pos)

    for trainer, positions in trainer_positions.items():
        win_rate = sum(1 for p in positions if p == 1) / len(positions)
        top3_rate = sum(1 for p in positions if p <= 3) / len(positions)
        _trainer_stats_cache[trainer] = (win_rate, top3_rate)

    for key, positions in combo_positions.items():
        _trainer_jockey_cache[key] = sum(1 for p in positions if p == 1) / len(positions)

    _trainer_cache_computed = True


def _calc_trainer_features(horse: Horse | None, race: Race,
                           session: Session) -> dict:
    """厩舎成績(キャッシュから取得)"""
    features = {
        "trainer_win_rate": 0.0,
        "trainer_top3_rate": 0.0,
    }

    if not horse or not horse.trainer:
        return features

    trainer_name = horse.trainer.strip()
    stats = _trainer_stats_cache.get(trainer_name)
    if stats:
        features["trainer_win_rate"] = stats[0]
        features["trainer_top3_rate"] = stats[1]

    return features


def _calc_jockey_trainer_combo(entry: RaceEntry, horse: Horse | None,
                                race: Race, session: Session) -> dict:
    """騎手×厩舎コンビの勝率(キャッシュから取得)"""
    if not horse or not horse.trainer or not entry.jockey_id:
        return {"jockey_trainer_combo_win": 0.0}

    trainer_name = horse.trainer.strip()
    combo_rate = _trainer_jockey_cache.get((trainer_name, entry.jockey_id), 0.0)
    return {"jockey_trainer_combo_win": combo_rate}


# =============================================================
# 前半位置取り(Early Speed)
# =============================================================

def _calc_early_speed_features(history: list[tuple[RaceEntry, Race]]) -> dict:
    """通過順から先行力を算出"""
    features = {
        "early_position_avg": 0.0,  # 前半位置取り平均
        "late_gain_avg": 0.0,  # 後半の追い上げ幅平均
    }

    early_positions = []
    late_gains = []
    for e, r in history:
        if not e.passing:
            continue
        parts = e.passing.replace("-", ",").split(",")
        try:
            nums = [int(p.strip()) for p in parts if p.strip().isdigit()]
        except ValueError:
            continue
        if not nums:
            continue

        n_runners = r.n_runners or 18
        # 正規化(1位=1.0, 最下位=0.0)
        early_pos = nums[0] / max(1, n_runners)
        early_positions.append(early_pos)

        # 後半の追い上げ(最初の通過順 - 最後の通過順)
        if len(nums) >= 2:
            late_gains.append(nums[0] - nums[-1])

    if early_positions:
        features["early_position_avg"] = float(np.mean(early_positions))
    if late_gains:
        features["late_gain_avg"] = float(np.mean(late_gains))

    return features


# =============================================================
# Eloレーティング
# =============================================================

# グローバルキャッシュ(セッション内で再利用)
_elo_cache: dict[int, float] = {}
_elo_computed = False


def _compute_elo_ratings(session: Session, before_race: Race):
    """全馬のEloレーティングを計算(当該レース前まで) - 一括プリロード版"""
    global _elo_cache, _elo_computed
    if _elo_computed:
        return

    K = 16  # Elo K-factor
    _elo_cache.clear()

    # 全確定レースのIDを日付順で取得
    race_ids = (
        session.query(Race.id)
        .filter(Race.status == "finished", Race.race_date < before_race.race_date)
        .order_by(Race.race_date, Race.id)
        .all()
    )
    race_id_list = [r[0] for r in race_ids]

    if not race_id_list:
        _elo_computed = True
        return

    # 全エントリーを一括取得してレースごとにグループ化
    all_entries = (
        session.query(RaceEntry.race_id, RaceEntry.horse_id, RaceEntry.finish_position)
        .filter(
            RaceEntry.race_id.in_(race_id_list),
            RaceEntry.finish_position.isnot(None),
            RaceEntry.horse_id.isnot(None),
        )
        .all()
    )

    # レースIDごとにグループ化
    from collections import defaultdict
    race_entries_map = defaultdict(list)
    for race_id, horse_id, finish_pos in all_entries:
        race_entries_map[race_id].append((horse_id, finish_pos))

    for race_id in race_id_list:
        entries = race_entries_map.get(race_id, [])
        if len(entries) < 2:
            continue

        ratings = {hid: _elo_cache.get(hid, 1500.0) for hid, _ in entries}
        n = len(entries)
        new_ratings = {}

        for hid, fp in entries:
            r_self = ratings[hid]
            delta = 0.0
            for hid2, fp2 in entries:
                if hid2 == hid:
                    continue
                r_opp = ratings[hid2]
                expected = 1.0 / (1.0 + 10 ** ((r_opp - r_self) / 400))
                actual = 1.0 if fp < fp2 else (0.5 if fp == fp2 else 0.0)
                delta += K * (actual - expected) / (n - 1)
            new_ratings[hid] = r_self + delta

        _elo_cache.update(new_ratings)

    _elo_computed = True


def _get_elo_rating(horse_id: int | None) -> float:
    if horse_id is None:
        return 1500.0
    return _elo_cache.get(horse_id, 1500.0)


# =============================================================
# クラス補正
# =============================================================

def _calc_class_adjusted_features(history: list[tuple[RaceEntry, Race]]) -> dict:
    """レースグレードを考慮した成績補正"""
    features = {
        "class_adjusted_avg": 0.0,
        "grade_change": 0.0,  # 今回と前走のクラス差
    }

    if not history:
        return features

    # グレード補正した着順(高グレードほど着順の価値が高い)
    adjusted = []
    for e, r in history:
        if not e.finish_position:
            continue
        grade_val = _grade_to_num(r.grade)
        # 高クラスでの好走を高評価: 補正着順 = 着順 - (グレード値 - 3) * 0.5
        adj = e.finish_position - (grade_val - 3) * 0.5
        adjusted.append(adj)

    if adjusted:
        features["class_adjusted_avg"] = float(np.mean(adjusted[:5]))

    return features


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
        # 距離変更×脚質(短縮は先行有利)
        "dist_change_x_style": f.get("distance_change", 0) * f.get("running_style", 3),
        # 休み明け×厩舎力
        "fresh_x_trainer": f.get("is_fresh", 0) * f.get("trainer_win_rate", 0),
    }


# =============================================================
# メイン特徴量構築
# =============================================================

# Racecourseキャッシュ
_racecourse_cache: dict[int, Racecourse] = {}


def _ensure_racecourse_cache(session: Session):
    global _racecourse_cache
    if _racecourse_cache:
        return
    for rc in session.query(Racecourse).all():
        _racecourse_cache[rc.id] = rc


# Horse/Jockeyキャッシュ
_horse_cache: dict[int, Horse] = {}
_jockey_cache: dict[int, Jockey] = {}


def _ensure_horse_jockey_cache(session: Session):
    global _horse_cache, _jockey_cache
    if not _horse_cache:
        for h in session.query(Horse).all():
            _horse_cache[h.id] = h
    if not _jockey_cache:
        for j in session.query(Jockey).all():
            _jockey_cache[j.id] = j


def build_features_for_race(session: Session, race: Race) -> pd.DataFrame:
    """レースの全エントリーから特徴量DataFrameを生成"""
    # 事前キャッシュ計算(初回のみ)
    global _elo_computed, _trainer_cache_computed, _winner_cache_computed
    _ensure_racecourse_cache(session)
    _ensure_horse_jockey_cache(session)
    if not _elo_computed:
        _compute_elo_ratings(session, race)
    if not _trainer_cache_computed:
        _precompute_trainer_stats(session, race.race_date)
    if not _winner_cache_computed:
        _precompute_winner_times(session, race.race_date)

    entries = session.query(RaceEntry).filter_by(race_id=race.id).all()
    racecourse = _racecourse_cache.get(race.racecourse_id)

    if not entries:
        return pd.DataFrame()

    # horse_history/jockey_historyを一括プリロード
    horse_ids = {e.horse_id for e in entries if e.horse_id}
    jockey_ids = {e.jockey_id for e in entries if e.jockey_id}

    horse_histories = _batch_get_horse_histories(session, horse_ids, race)
    jockey_histories = _batch_get_jockey_histories(session, jockey_ids, race)

    rows = []
    for entry in entries:
        horse = _horse_cache.get(entry.horse_id) if entry.horse_id else None
        jockey = _jockey_cache.get(entry.jockey_id) if entry.jockey_id else None

        horse_history = horse_histories.get(entry.horse_id, []) if entry.horse_id else []
        jockey_history = jockey_histories.get(entry.jockey_id, []) if entry.jockey_id else []

        row = _build_entry_features(entry, horse, jockey, race, racecourse,
                                    entries, horse_history, jockey_history, session)
        rows.append(row)

    df = pd.DataFrame(rows)
    df = _add_relative_features(df)
    return df


def _batch_get_horse_histories(session: Session, horse_ids: set[int],
                                current_race: Race) -> dict[int, list[tuple[RaceEntry, Race]]]:
    """複数馬の履歴を一括取得"""
    if not horse_ids:
        return {}

    all_entries = (
        session.query(RaceEntry, Race)
        .join(Race, RaceEntry.race_id == Race.id)
        .filter(
            RaceEntry.horse_id.in_(horse_ids),
            RaceEntry.finish_position.isnot(None),
            or_(
                Race.race_date < current_race.race_date,
                and_(
                    Race.race_date == current_race.race_date,
                    Race.id < current_race.id,
                ),
            ),
        )
        .order_by(RaceEntry.horse_id, Race.race_date.desc(), Race.id.desc())
        .all()
    )

    from collections import defaultdict
    result = defaultdict(list)
    for entry, race in all_entries:
        if len(result[entry.horse_id]) < 10:
            result[entry.horse_id].append((entry, race))

    return dict(result)


def _batch_get_jockey_histories(session: Session, jockey_ids: set[int],
                                 current_race: Race) -> dict[int, list[tuple[RaceEntry, Race]]]:
    """複数騎手の履歴を一括取得"""
    if not jockey_ids:
        return {}

    all_entries = (
        session.query(RaceEntry, Race)
        .join(Race, RaceEntry.race_id == Race.id)
        .filter(
            RaceEntry.jockey_id.in_(jockey_ids),
            RaceEntry.finish_position.isnot(None),
            or_(
                Race.race_date < current_race.race_date,
                and_(
                    Race.race_date == current_race.race_date,
                    Race.id < current_race.id,
                ),
            ),
        )
        .order_by(RaceEntry.jockey_id, Race.race_date.desc(), Race.id.desc())
        .all()
    )

    from collections import defaultdict
    result = defaultdict(list)
    for entry, race in all_entries:
        if len(result[entry.jockey_id]) < 50:
            result[entry.jockey_id].append((entry, race))

    return dict(result)


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

    # === ローテーション ===
    features.update(_calc_rotation_features(race, horse_history))

    # === スピード指数 ===
    features.update(_calc_speed_features(horse_history))
    features.update(_calc_margin_features(session, horse_history))

    # === 厩舎 ===
    features.update(_calc_trainer_features(horse, race, session))
    features.update(_calc_jockey_trainer_combo(entry, horse, race, session))

    # === 前半位置取り ===
    features.update(_calc_early_speed_features(horse_history))

    # === Eloレーティング ===
    features["elo_rating"] = _get_elo_rating(entry.horse_id)

    # === クラス補正 ===
    features.update(_calc_class_adjusted_features(horse_history))

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

    # Eloレーティング rank / z-score
    if "elo_rating" in df.columns:
        df["elo_rank"] = df["elo_rating"].rank(ascending=False, method="min")
        mean_e = df["elo_rating"].mean()
        std_e = df["elo_rating"].std()
        df["elo_z"] = (df["elo_rating"] - mean_e) / std_e if std_e > 0 else 0

    # スピード指数 rank
    if "speed_figure_avg3" in df.columns:
        df["speed_rank"] = df["speed_figure_avg3"].rank(ascending=False, method="min")

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
    # ローテーション (6)
    "days_since_last", "rest_bucket", "is_fresh", "is_second_up",
    "distance_change", "surface_switch",
    # スピード指数 (5)
    "speed_figure_last", "speed_figure_best3", "speed_figure_avg3",
    "margin_to_winner_avg", "finish_std",
    # 厩舎 (3)
    "trainer_win_rate", "trainer_top3_rate", "jockey_trainer_combo_win",
    # 前半位置取り (2)
    "early_position_avg", "late_gain_avg",
    # Elo・クラス (3)
    "elo_rating", "class_adjusted_avg",
    # レース情報 (12)
    "grade_num", "n_runners", "distance", "distance_category",
    "frame_number", "horse_number", "frame_advantage",
    "track_condition_num", "is_turf", "is_dirt",
    "weather_num", "is_rainy",
    # レース内相対 (10)
    "last_3f_rank", "last_3f_z",
    "weight_carry_rank", "horse_weight_rank",
    "form_rank", "jockey_rank",
    "elo_rank", "elo_z", "speed_rank",
    # 交互作用 (8)
    "style_x_track_cond", "bloodline_x_surface",
    "frame_x_distance", "jockey_x_horse_form", "pace_x_style",
    "weather_x_track_pref", "dist_change_x_style", "fresh_x_trainer",
]
# 合計: 86個
