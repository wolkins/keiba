"""データベースモデル定義 (中央競馬)"""
from datetime import datetime

from sqlalchemy import (
    Boolean, Column, Date, DateTime, Float, ForeignKey,
    Integer, String, Text, UniqueConstraint, create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session, relationship, sessionmaker

from .config import DB_URL


class Base(DeclarativeBase):
    pass


class Racecourse(Base):
    """競馬場マスタ"""
    __tablename__ = "racecourses"

    id = Column(Integer, primary_key=True)
    code = Column(String(4), unique=True, nullable=False)
    name = Column(String(50), nullable=False)
    surface_type = Column(String(10))  # 芝/ダート/障害

    races = relationship("Race", back_populates="racecourse")


class Horse(Base):
    """馬マスタ"""
    __tablename__ = "horses"

    id = Column(Integer, primary_key=True)
    horse_id = Column(String(20), unique=True, nullable=False)  # netkeiba horse_id
    name = Column(String(50), nullable=False)
    sex = Column(String(2))         # 牡/牝/セ
    birth_year = Column(Integer)
    color = Column(String(20))      # 毛色
    father = Column(String(50))     # 父
    mother = Column(String(50))     # 母
    mother_father = Column(String(50))  # 母父
    trainer = Column(String(50))    # 調教師
    owner = Column(String(50))      # 馬主
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

    entries = relationship("RaceEntry", back_populates="horse")


class Jockey(Base):
    """騎手マスタ"""
    __tablename__ = "jockeys"

    id = Column(Integer, primary_key=True)
    jockey_id = Column(String(10), unique=True, nullable=False)
    name = Column(String(50), nullable=False)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

    entries = relationship("RaceEntry", back_populates="jockey")


class Race(Base):
    """レース情報"""
    __tablename__ = "races"

    id = Column(Integer, primary_key=True)
    race_id = Column(String(20), unique=True, nullable=False)  # 12桁ID
    race_date = Column(Date, nullable=False)
    racecourse_id = Column(Integer, ForeignKey("racecourses.id"), nullable=False)
    race_number = Column(Integer, nullable=False)
    race_name = Column(String(100))
    grade = Column(String(10))       # G1,G2,G3,OP,L,3勝,2勝,1勝,未勝利
    surface = Column(String(10))     # 芝/ダート
    distance = Column(Integer)       # 距離(m)
    course_type = Column(String(10)) # 右/左/直線
    weather = Column(String(10))     # 晴/曇/雨/雪
    track_condition = Column(String(10))  # 良/稍重/重/不良
    n_runners = Column(Integer)      # 出走頭数
    status = Column(String(20), default="scheduled")

    racecourse = relationship("Racecourse", back_populates="races")
    entries = relationship("RaceEntry", back_populates="race")
    odds = relationship("Odds", back_populates="race")

    __table_args__ = (
        UniqueConstraint("race_date", "racecourse_id", "race_number", name="uq_race"),
    )


class RaceEntry(Base):
    """出走表"""
    __tablename__ = "race_entries"

    id = Column(Integer, primary_key=True)
    race_id = Column(Integer, ForeignKey("races.id"), nullable=False)
    horse_id = Column(Integer, ForeignKey("horses.id"), nullable=True)
    jockey_id = Column(Integer, ForeignKey("jockeys.id"), nullable=True)

    frame_number = Column(Integer)    # 枠番 (1-8)
    horse_number = Column(Integer, nullable=False)  # 馬番 (1-18)
    sex_age = Column(String(10))      # 性齢 (牡3, 牝4 等)
    weight_carry = Column(Float)      # 斤量
    horse_weight = Column(Integer)    # 馬体重
    weight_diff = Column(Integer)     # 馬体重増減
    popularity = Column(Integer)      # 人気
    odds_win = Column(Float)          # 単勝オッズ (注意: 締切時点の最終値。学習時のリーク元になる可能性あり。時系列整合の厳密な用途では OddsSnapshot を as_of 指定で参照すること)

    # 結果
    finish_position = Column(Integer)
    finish_time = Column(String(10))  # タイム "1:34.5"
    margin = Column(String(20))       # 着差
    last_3f = Column(Float)           # 上がり3F
    passing = Column(String(20))      # 通過順 "3-3-2-1"
    win_technique = Column(String(20))  # 脚質判定

    race = relationship("Race", back_populates="entries")
    horse = relationship("Horse", back_populates="entries")
    jockey = relationship("Jockey", back_populates="entries")

    __table_args__ = (
        UniqueConstraint("race_id", "horse_number", name="uq_entry"),
    )


class Odds(Base):
    """オッズ (最新値キャッシュ)

    注意: 同一 (race_id, bet_type, combination) は上書きされ、最後の captured_at の値しか残らない。
    時系列整合が必要な用途 (WIN5 評価, バックテスト) では OddsSnapshot を参照すること。
    """
    __tablename__ = "odds"

    id = Column(Integer, primary_key=True)
    race_id = Column(Integer, ForeignKey("races.id"), nullable=False)
    bet_type = Column(String(20), nullable=False)
    combination = Column(String(30), nullable=False)
    odds_value = Column(Float, nullable=False)
    captured_at = Column(DateTime, default=datetime.now)

    race = relationship("Race", back_populates="odds")

    __table_args__ = (
        UniqueConstraint("race_id", "bet_type", "combination", name="uq_odds"),
    )


class OddsSnapshot(Base):
    """オッズ時系列スナップショット (append-only)

    スクレイプ毎に追記し、as_of 時点の市場確率評価・バックテストに使用する。
    """
    __tablename__ = "odds_snapshots"

    id = Column(Integer, primary_key=True)
    race_id = Column(Integer, ForeignKey("races.id"), nullable=False)
    bet_type = Column(String(20), nullable=False)
    combination = Column(String(30), nullable=False)
    odds_value = Column(Float, nullable=False)
    captured_at = Column(DateTime, nullable=False, default=datetime.now, index=True)

    __table_args__ = (
        UniqueConstraint(
            "race_id", "bet_type", "combination", "captured_at",
            name="uq_odds_snapshot",
        ),
    )


WIN5_TARGET_SOURCE_JRA = "jra"
WIN5_TARGET_STATUS_SCHEDULED = "scheduled"
WIN5_TARGET_STATUS_CONFIRMED = "confirmed"
WIN5_TARGET_STATUS_CANCELLED = "cancelled"


class Win5TargetRace(Base):
    """WIN5対象5レース (official only)

    JRA 公式の対象レース一覧から取り込む。
    ヒューリスティック推定行は保存しない (誤発注事故防止のため)。
    Race への紐付け (race_id) は後から解決してよい。
    """
    __tablename__ = "win5_target_races"

    id = Column(Integer, primary_key=True)
    race_date = Column(Date, nullable=False, index=True)
    leg_index = Column(Integer, nullable=False)  # 1..5

    # JRA 公式から得られる一次情報
    racecourse_code = Column(String(4), nullable=True)   # "05" 等。名称からの変換値
    racecourse_name = Column(String(50), nullable=False) # "東京" 等 (JRA表記そのまま)
    race_number = Column(Integer, nullable=False)
    post_time = Column(String(10), nullable=True)        # "14時50分"
    close_time = Column(String(10), nullable=True)       # WIN5締切時刻

    # Race との紐付け (スクレイピング未取得時は null)
    race_id = Column(Integer, ForeignKey("races.id"), nullable=True)

    source = Column(String(20), nullable=False, default=WIN5_TARGET_SOURCE_JRA)
    source_url = Column(String(255), nullable=True)
    status = Column(String(20), nullable=False, default=WIN5_TARGET_STATUS_SCHEDULED)

    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

    race = relationship("Race")

    __table_args__ = (
        UniqueConstraint("race_date", "leg_index", name="uq_win5_target_date_leg"),
    )


WIN5_RUN_MODE_HIT = "hit"
WIN5_RUN_MODE_EV = "ev"
WIN5_OPTIMIZER_VERSION = "v1"
WIN5_UNIT_PRICE = 200
WIN5_POOL_RATE = 0.7  # JRA WIN5 払戻率


class Win5Run(Base):
    """WIN5 推奨買い目 (1回の最適化実行単位)

    チケット詳細と対象5Rスナップショットは tickets_json / target_races_json に
    JSON で保持する (正規化はしない: チケット数は最大で budget//200 程度で
    行数が爆発するリスクがある + 参照は run 単位が支配的)。
    """
    __tablename__ = "win5_runs"

    id = Column(Integer, primary_key=True)
    race_date = Column(Date, nullable=False, index=True)
    as_of = Column(DateTime, nullable=False, default=datetime.now)

    mode = Column(String(20), nullable=False, default=WIN5_RUN_MODE_HIT)
    budget = Column(Integer, nullable=False)
    unit_amount = Column(Integer, nullable=False, default=WIN5_UNIT_PRICE)
    total_tickets = Column(Integer, nullable=False)
    total_cost = Column(Integer, nullable=False)

    hit_probability_sum = Column(Float, nullable=True)       # 買った全点の的中確率の和 (= 想定hit rate)
    top_ticket_probability = Column(Float, nullable=True)    # 最も確度の高い1点の確率
    coverage_threshold = Column(Float, nullable=True)        # 候補絞り込みの累積確率閾値
    optimizer_version = Column(String(20), nullable=False, default=WIN5_OPTIMIZER_VERSION)

    tickets_json = Column(Text, nullable=False)              # list[{horse_numbers, prob, amount}]
    target_races_json = Column(Text, nullable=False)         # list[{leg_index, racecourse_code, race_number, race_id}]

    created_at = Column(DateTime, default=datetime.now)


class Win5PayoutHistory(Base):
    """WIN5 過去払戻履歴 (manual entry 前提)

    JRA 公式は完全な払戻履歴を構造化データで公開していないため、手動入力 or
    半自動スクレイプ (キャリーオーバー発生日のみ) で蓄積する。
    配当モデル v2 (回帰学習) の教師データとして使用する。
    """
    __tablename__ = "win5_payout_history"

    id = Column(Integer, primary_key=True)
    race_date = Column(Date, nullable=False, unique=True, index=True)

    # 5R 勝ち馬組合せ (例: "3-7-12-5-9")
    winning_combination = Column(String(40), nullable=True)
    # 5R の race_id を JSON 配列で保持 (例: [123, 124, 125, 126, 127])
    race_ids_json = Column(Text, nullable=True)

    # 売上・票数・払戻
    total_sales = Column(Integer, nullable=True)           # 発売金額(円)
    winning_tickets = Column(Integer, nullable=True)       # 的中票数
    payout_per_ticket = Column(Integer, nullable=True)     # 1票あたり払戻金(円)

    # キャリーオーバー
    carryover_in = Column(Integer, nullable=True, default=0)    # 当日持越金
    carryover_out = Column(Integer, nullable=True, default=0)   # 次週へ持越
    jackpot_flag = Column(Boolean, default=False)               # キャリーオーバー到達時True

    # メタ
    source = Column(String(20), nullable=False, default="manual")  # manual / jra / inferred
    source_url = Column(String(255), nullable=True)
    notes = Column(Text, nullable=True)

    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)


class PredictionResult(Base):
    """予想結果の記録"""
    __tablename__ = "prediction_results"

    id = Column(Integer, primary_key=True)
    race_id = Column(Integer, ForeignKey("races.id"), nullable=False)
    mode = Column(String(20), nullable=False)
    bet_type = Column(String(20), nullable=False)
    combination = Column(String(30), nullable=False)
    confidence = Column(Float)
    expected_value = Column(Float)
    is_hit = Column(Boolean)
    payout = Column(Float)
    created_at = Column(DateTime, default=datetime.now)


engine = create_engine(DB_URL, echo=False)


def init_db():
    Base.metadata.create_all(engine)


def get_session() -> Session:
    return sessionmaker(bind=engine)()


RACECOURSES = [
    {"code": "01", "name": "札幌"},
    {"code": "02", "name": "函館"},
    {"code": "03", "name": "福島"},
    {"code": "04", "name": "新潟"},
    {"code": "05", "name": "東京"},
    {"code": "06", "name": "中山"},
    {"code": "07", "name": "中京"},
    {"code": "08", "name": "京都"},
    {"code": "09", "name": "阪神"},
    {"code": "10", "name": "小倉"},
]


def seed_racecourses():
    session = get_session()
    try:
        for rc in RACECOURSES:
            existing = session.query(Racecourse).filter_by(code=rc["code"]).first()
            if not existing:
                session.add(Racecourse(**rc))
        session.commit()
    finally:
        session.close()
