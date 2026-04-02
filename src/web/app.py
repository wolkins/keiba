"""中央競馬予想 Web アプリケーション (FastAPI + Jinja2 + htmx)"""
from datetime import date, datetime
from pathlib import Path

from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.common.database import (
    Horse, Jockey, Odds, Race, RaceEntry, Racecourse, get_session, init_db, seed_racecourses,
)
from src.predictor.model import KeibaPredictor

WEB_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = WEB_DIR / "templates"
STATIC_DIR = WEB_DIR / "static"

app = FastAPI(title="中央競馬予想システム")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)


@app.on_event("startup")
def startup():
    init_db()
    seed_racecourses()


def _render(request: Request, template: str, context: dict) -> HTMLResponse:
    return templates.TemplateResponse(request, template, context)


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    session = get_session()
    try:
        stats = {
            "races_total": session.query(Race).count(),
            "races_finished": session.query(Race).filter(Race.status == "finished").count(),
            "entries_total": session.query(RaceEntry).count(),
            "entries_with_result": session.query(RaceEntry).filter(RaceEntry.finish_position.isnot(None)).count(),
            "horses_total": session.query(Horse).count(),
            "jockeys_total": session.query(Jockey).count(),
            "odds_total": session.query(Odds).count(),
        }

        today = date.today()
        today_races = (
            session.query(Race).filter(Race.race_date == today)
            .order_by(Race.racecourse_id, Race.race_number).all()
        )

        today_race_list = []
        for race in today_races:
            rc = session.query(Racecourse).filter_by(id=race.racecourse_id).first()
            n = session.query(RaceEntry).filter_by(race_id=race.id).count()
            today_race_list.append({
                "id": race.id, "venue": rc.name if rc else "不明",
                "race_number": race.race_number, "race_name": race.race_name or "",
                "grade": race.grade or "", "status": race.status, "n_entries": n,
                "surface": race.surface or "", "distance": race.distance or 0,
            })

        recent_dates = [
            r[0] for r in session.query(Race.race_date).distinct()
            .order_by(Race.race_date.desc()).limit(10).all()
        ]

        return _render(request, "dashboard.html", {
            "stats": stats, "today": today.isoformat(),
            "today_races": today_race_list, "recent_dates": recent_dates,
        })
    finally:
        session.close()


@app.get("/races", response_class=HTMLResponse)
def races_by_date(request: Request, d: str = Query(default="")):
    target = d or date.today().isoformat()
    session = get_session()
    try:
        dt = datetime.strptime(target, "%Y-%m-%d").date()
        races = session.query(Race).filter(Race.race_date == dt).order_by(Race.racecourse_id, Race.race_number).all()

        race_list = []
        for race in races:
            rc = session.query(Racecourse).filter_by(id=race.racecourse_id).first()
            n = session.query(RaceEntry).filter_by(race_id=race.id).count()
            race_list.append({
                "id": race.id, "venue": rc.name if rc else "不明",
                "race_number": race.race_number, "race_name": race.race_name or "",
                "grade": race.grade or "", "status": race.status, "n_entries": n,
                "surface": race.surface or "", "distance": race.distance or 0,
            })

        if request.headers.get("HX-Request"):
            return _render(request, "_race_list.html", {"races": race_list, "target_date": target})

        return _render(request, "races.html", {"races": race_list, "target_date": target})
    finally:
        session.close()


@app.get("/race/{race_id}", response_class=HTMLResponse)
def race_detail(request: Request, race_id: int, mode: str = Query(default="accuracy")):
    session = get_session()
    try:
        race = session.query(Race).filter_by(id=race_id).first()
        if not race:
            return HTMLResponse("<h2>レースが見つかりません</h2>", status_code=404)

        rc = session.query(Racecourse).filter_by(id=race.racecourse_id).first()
        entries = session.query(RaceEntry).filter_by(race_id=race.id).order_by(RaceEntry.horse_number).all()

        entry_list = []
        for e in entries:
            horse = session.query(Horse).filter_by(id=e.horse_id).first() if e.horse_id else None
            jockey = session.query(Jockey).filter_by(id=e.jockey_id).first() if e.jockey_id else None
            entry_list.append({
                "frame_number": e.frame_number,
                "horse_number": e.horse_number,
                "horse_name": horse.name if horse else "不明",
                "jockey_name": jockey.name if jockey else "不明",
                "sex_age": e.sex_age or "",
                "weight_carry": e.weight_carry or 0,
                "horse_weight": e.horse_weight,
                "weight_diff": e.weight_diff,
                "last_3f": e.last_3f,
                "popularity": e.popularity,
                "odds_win": e.odds_win,
                "finish_position": e.finish_position,
                "finish_time": e.finish_time or "",
                "margin": e.margin or "",
                "passing": e.passing or "",
            })

        predictor = KeibaPredictor(mode=mode)
        predictions = predictor.predict(session, race)
        bets = predictor.suggest_bets(predictions, budget=1000)

        return _render(request, "race_detail.html", {
            "race": {
                "id": race.id, "date": race.race_date.isoformat(),
                "venue": rc.name if rc else "不明", "race_number": race.race_number,
                "race_name": race.race_name or "", "grade": race.grade or "",
                "surface": race.surface or "", "distance": race.distance or 0,
                "course_type": race.course_type or "",
                "status": race.status, "n_runners": race.n_runners or 0,
                "weather": race.weather or "", "track_condition": race.track_condition or "",
            },
            "entries": entry_list, "predictions": predictions,
            "bets": bets, "mode": mode,
        })
    finally:
        session.close()


@app.get("/predict/{race_id}", response_class=HTMLResponse)
def predict_partial(request: Request, race_id: int, mode: str = Query(default="accuracy")):
    session = get_session()
    try:
        race = session.query(Race).filter_by(id=race_id).first()
        if not race:
            return HTMLResponse("<p>レースが見つかりません</p>")

        predictor = KeibaPredictor(mode=mode)
        predictions = predictor.predict(session, race)
        bets = predictor.suggest_bets(predictions, budget=1000)

        return _render(request, "_predictions.html", {
            "predictions": predictions, "bets": bets,
            "mode": mode, "race_id": race_id,
        })
    finally:
        session.close()
