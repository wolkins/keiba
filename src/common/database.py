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
    odds_win = Column(Float)          # 単勝オッズ

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
    """オッズ"""
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
