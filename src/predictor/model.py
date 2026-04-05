"""予測モデル - LightGBM LambdaRank (中央競馬)"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except (ImportError, OSError):
    HAS_LIGHTGBM = False
    from sklearn.ensemble import GradientBoostingRegressor

from src.common.config import PROJECT_ROOT
from src.common.database import Horse, Jockey, Odds, Race, RaceEntry, Racecourse, get_session

from .calibration import ProbabilityCalibrator
from .ensemble import EnsemblePredictor
from .evaluation import evaluate_predictions
from .features import FEATURE_COLUMNS, build_features_for_race

MODEL_DIR = PROJECT_ROOT / "data" / "models"


class KeibaPredictor:
    """中央競馬予測モデル"""

    def __init__(self, mode: str = "accuracy"):
        """
        Args:
            mode: 'accuracy' (的中率重視) or 'roi' (回収率重視)
        """
        self.mode = mode
        self.model = None
        self._feature_cols = FEATURE_COLUMNS
        self._model_path = MODEL_DIR / "model_ranker.pkl"
        self._calibrator_win = ProbabilityCalibrator("win")
        self._calibrator_top3 = ProbabilityCalibrator("top3")
        self._ensemble = None

    def train(self, session: Session, min_races: int = 50) -> dict:
        """LambdaRankモデルを学習

        特徴量構築 → LambdaRank学習 → キャリブレータfit → アンサンブル学習 → 多軸評価
        """
        races = (
            session.query(Race)
            .filter(Race.status == "finished")
            .order_by(Race.race_date)
            .all()
        )
        if len(races) < min_races:
            return {"error": f"学習データが不足しています ({len(races)}/{min_races}レース)"}

        # === 特徴量構築 ===
        all_dfs = []
        race_date_map = {}
        for race in races:
            df = build_features_for_race(session, race)
            if not df.empty and "finish_position" in df.columns:
                df["race_id"] = race.id
                all_dfs.append(df)
                race_date_map[race.id] = race.race_date

        if not all_dfs:
            return {"error": "有効なトレーニングデータがありません"}

        full_df = pd.concat(all_dfs, ignore_index=True)

        # ラベル: 着順を逆転 (1着が最大値)
        max_pos = full_df["finish_position"].max()
        full_df["label"] = (max_pos + 1 - full_df["finish_position"]).clip(lower=0)

        # === 時間減衰重み ===
        from src.common.config import DECAY_HALF_LIFE_DAYS
        reference_date = max(race_date_map.values())
        full_df["sample_weight"] = full_df["race_id"].map(
            lambda rid: np.exp(
                -np.log(2) * (reference_date - race_date_map.get(rid, reference_date)).days
                / DECAY_HALF_LIFE_DAYS
            )
        )

        # === 時系列分割 ===
        unique_races = full_df["race_id"].unique()
        split_idx = int(len(unique_races) * 0.8)
        train_race_ids = set(unique_races[:split_idx])
        valid_race_ids = set(unique_races[split_idx:])

        train_df = full_df[full_df["race_id"].isin(train_race_ids)].sort_values("race_id")
        valid_df = full_df[full_df["race_id"].isin(valid_race_ids)].sort_values("race_id")

        available_cols = [c for c in FEATURE_COLUMNS if c in train_df.columns]

        X_train = train_df[available_cols].fillna(0)
        y_train = train_df["label"]
        w_train = train_df["sample_weight"]
        group_train = train_df.groupby("race_id").size().tolist()

        X_valid = valid_df[available_cols].fillna(0)
        y_valid = valid_df["label"]
        group_valid = valid_df.groupby("race_id").size().tolist()

        # === LambdaRank学習 (seedアンサンブル) ===
        if HAS_LIGHTGBM:
            # label_gain: 1着に重みを集中
            n_labels = int(max_pos) + 1
            label_gain = [0.0] * n_labels
            for i in range(n_labels):
                # 1着(label=max_pos)=100, 2着=10, 3着=5, 4着以下=着順に応じて漸減
                pos = max_pos - i  # 実際の着順
                if pos <= 0:
                    label_gain[i] = 100.0
                elif pos == 1:
                    label_gain[i] = 10.0
                elif pos == 2:
                    label_gain[i] = 5.0
                else:
                    label_gain[i] = max(0.0, 3.0 - pos * 0.1)

            base_params = {
                "objective": "lambdarank",
                "metric": "ndcg",
                "ndcg_eval_at": [1, 3],
                "label_gain": label_gain,
                "lambdarank_truncation_level": 5,
                "learning_rate": 0.02,
                "num_leaves": 63,
                "min_data_in_leaf": max(50, len(train_df) // 60),
                "feature_fraction": 0.75,
                "bagging_fraction": 0.8,
                "bagging_freq": 3,
                "lambda_l1": 0.5,
                "lambda_l2": 5.0,
                "min_gain_to_split": 0.05,
                "n_estimators": 2000,
                "verbose": -1,
            }

            # seedアンサンブル: 5モデルの平均
            n_seeds = 5
            models = []
            for seed in range(n_seeds):
                params = {**base_params, "random_state": seed * 42 + 7}
                r = lgb.LGBMRanker(**params)
                callbacks = [lgb.early_stopping(100, verbose=False)]
                if len(valid_df) > 0 and group_valid:
                    r.fit(
                        X_train, y_train, group=group_train,
                        sample_weight=w_train,
                        eval_set=[(X_valid, y_valid)],
                        eval_group=[group_valid],
                        callbacks=callbacks,
                    )
                else:
                    r.fit(X_train, y_train, group=group_train, sample_weight=w_train)
                models.append(r)

            ranker = models[0]  # メインモデル(保存用)
        else:
            ranker = GradientBoostingRegressor(
                n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42,
            )
            ranker.fit(X_train, y_train, sample_weight=w_train)

        self.model = ranker
        self._seed_models = models if HAS_LIGHTGBM else []
        self._feature_cols = available_cols

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        with open(self._model_path, "wb") as f:
            pickle.dump({"model": ranker, "features": available_cols,
                         "seed_models": models if HAS_LIGHTGBM else []}, f)

        # === キャリブレータfit + 評価 ===
        eval_results = {}
        if len(valid_df) > 0:
            valid_df = valid_df.copy()
            valid_scores = ranker.predict(X_valid)
            valid_df["pred_score"] = valid_scores

            # キャリブレータをfit (検証データで)
            win_labels = (valid_df["finish_position"] == 1).astype(int).values
            top3_labels = (valid_df["finish_position"] <= 3).astype(int).values
            self._calibrator_win.fit(valid_scores, win_labels)
            self._calibrator_top3.fit(valid_scores, top3_labels)
            self._calibrator_win.save()
            self._calibrator_top3.save()

            # オッズ情報を付加
            odds_map = {}
            for rid in valid_race_ids:
                odds_rows = session.query(Odds).filter_by(race_id=int(rid), bet_type="win").all()
                for o in odds_rows:
                    try:
                        odds_map[(int(rid), int(o.combination))] = o.odds_value
                    except (ValueError, TypeError):
                        pass
            valid_df["odds_value"] = valid_df.apply(
                lambda row: odds_map.get((int(row["race_id"]), int(row["horse_number"])), None),
                axis=1,
            )

            # 多軸指標
            eval_results = evaluate_predictions(valid_df, session=session)

        # === アンサンブル学習 ===
        ensemble_results = {}
        if HAS_LIGHTGBM and len(valid_df) > 0:
            try:
                self._ensemble = EnsemblePredictor()
                ens_result = self._ensemble.train(
                    X_train, y_train, group_train, w_train,
                    X_valid, y_valid, group_valid,
                    train_df["finish_position"], valid_df["finish_position"],
                )
                ensemble_results["ensemble"] = ens_result.get("layer1_models", [])
            except Exception as e:
                ensemble_results["ensemble_error"] = str(e)

        return {
            "n_races": len(races),
            "n_train_races": len(train_race_ids),
            "n_valid_races": len(valid_race_ids),
            "n_samples": len(full_df),
            "n_features": len(available_cols),
            "engine": "LightGBM" if HAS_LIGHTGBM else "sklearn",
            **eval_results,
            **ensemble_results,
        }

    def load(self) -> bool:
        """保存済みモデルを読み込み"""
        if self._model_path.exists():
            with open(self._model_path, "rb") as f:
                data = pickle.load(f)
            if isinstance(data, dict):
                self.model = data["model"]
                self._feature_cols = data.get("features", FEATURE_COLUMNS)
                self._seed_models = data.get("seed_models", [])
            else:
                self.model = data
                self._feature_cols = FEATURE_COLUMNS
                self._seed_models = []
            # キャリブレータも読み込み (なくても動作する)
            self._calibrator_win.load()
            self._calibrator_top3.load()
            return True
        return False

    def predict(self, session: Session, race: Race) -> list[dict]:
        """レースの予想

        アンサンブル予測 → キャリブレーション → 期待値計算
        """
        if self.model is None:
            if not self.load():
                return self._predict_by_stats(session, race)

        df = build_features_for_race(session, race)
        if df.empty:
            return []

        available_cols = [c for c in self._feature_cols if c in df.columns]
        X = df[available_cols].fillna(0)

        # seedアンサンブル予測（複数モデルの平均）
        if hasattr(self, '_seed_models') and len(self._seed_models) > 1:
            all_scores = np.array([m.predict(X) for m in self._seed_models])
            scores = all_scores.mean(axis=0)
        else:
            # フォールバック: 単体モデル
            ensemble = self._ensemble
            if ensemble is None:
                ensemble = EnsemblePredictor()
                if ensemble.load():
                    self._ensemble = ensemble
                else:
                    ensemble = None

            if ensemble is not None:
                try:
                    scores = ensemble.predict(X)
                except Exception:
                    scores = self.model.predict(X)
            else:
                scores = self.model.predict(X)

        s_min, s_max = scores.min(), scores.max()
        if s_max > s_min:
            norm_scores = (scores - s_min) / (s_max - s_min)
        else:
            norm_scores = np.ones_like(scores) * 0.5

        # キャリブレーション済み確率
        calibrated_win = self._calibrator_win.predict_proba(scores)
        calibrated_top3 = self._calibrator_top3.predict_proba(scores)

        # オッズ取得
        odds_data = session.query(Odds).filter_by(race_id=race.id, bet_type="win").all()
        odds_map = {}
        for o in odds_data:
            try:
                odds_map[int(o.combination)] = o.odds_value
            except ValueError:
                pass

        # エントリー情報
        entries = session.query(RaceEntry).filter_by(race_id=race.id).all()
        entry_map = {e.horse_number: e for e in entries}

        results = []
        for i, row in df.iterrows():
            horse_num = int(row["horse_number"])
            entry = entry_map.get(horse_num)
            score = float(norm_scores[i])
            win_prob = float(calibrated_win[i])
            top3_prob = float(calibrated_top3[i])
            odds_val = odds_map.get(horse_num, 0)
            expected_value = win_prob * odds_val if odds_val else None

            if self.mode == "roi" and expected_value is not None:
                final_score = expected_value
            else:
                final_score = score

            # 馬名取得
            horse_name = "不明"
            if entry and entry.horse:
                horse_name = entry.horse.name
            elif entry and entry.horse_id:
                horse = session.query(Horse).filter_by(id=entry.horse_id).first()
                if horse:
                    horse_name = horse.name

            # 騎手名取得
            jockey_name = ""
            if entry and entry.jockey:
                jockey_name = entry.jockey.name
            elif entry and entry.jockey_id:
                jockey = session.query(Jockey).filter_by(id=entry.jockey_id).first()
                if jockey:
                    jockey_name = jockey.name

            results.append({
                "horse_number": horse_num,
                "horse_name": horse_name,
                "jockey_name": jockey_name,
                "frame_number": entry.frame_number if entry else 0,
                "weight_carry": entry.weight_carry if entry else 0,
                "odds": odds_val,
                "score": round(final_score, 4),
                "probability": round(score, 4),
                "calibrated_win_prob": round(win_prob, 4),
                "calibrated_top3_prob": round(top3_prob, 4),
                "expected_value": round(expected_value, 4) if expected_value else None,
                "recommendation": self._get_recommendation(score),
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    def _predict_by_stats(self, session: Session, race: Race) -> list[dict]:
        """モデルなし時の統計ベース予測"""
        entries = session.query(RaceEntry).filter_by(race_id=race.id).all()
        if not entries:
            return []

        results = []
        for entry in entries:
            popularity = entry.popularity or 99
            odds = entry.odds_win or 99.9
            weight_carry = entry.weight_carry or 55.0

            # 人気順ベース + オッズの逆数 + 斤量補正
            score = (1 / max(1, popularity)) * 50 + (1 / max(1, odds)) * 30 - (weight_carry - 55) * 0.5

            horse_name = "不明"
            if entry.horse_id:
                horse = session.query(Horse).filter_by(id=entry.horse_id).first()
                if horse:
                    horse_name = horse.name

            jockey_name = ""
            if entry.jockey_id:
                jockey = session.query(Jockey).filter_by(id=entry.jockey_id).first()
                if jockey:
                    jockey_name = jockey.name

            results.append({
                "horse_number": entry.horse_number,
                "horse_name": horse_name,
                "jockey_name": jockey_name,
                "frame_number": entry.frame_number or 0,
                "weight_carry": weight_carry,
                "odds": entry.odds_win or 0,
                "score": round(score, 4),
                "probability": round(max(0, min(1, score / 100)), 4),
                "calibrated_win_prob": 0.0,
                "calibrated_top3_prob": 0.0,
                "expected_value": None,
                "recommendation": self._get_recommendation(max(0, min(1, score / 100))),
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    def _get_recommendation(self, score: float) -> str:
        """スコアから印を付与"""
        if score >= 0.7:
            return "◎ 本命"
        elif score >= 0.5:
            return "○ 対抗"
        elif score >= 0.35:
            return "▲ 単穴"
        elif score >= 0.2:
            return "△ 連下"
        else:
            return "×"

    def suggest_bets(self, predictions: list[dict], budget: int = 1000) -> list[dict]:
        """推奨買い目を生成"""
        if not predictions:
            return []

        top3 = predictions[:3]
        bets = []

        if self.mode == "accuracy":
            # 的中率重視: 単勝・馬連・三連単・ワイド
            bets.append({
                "bet_type": "単勝",
                "combination": str(top3[0]["horse_number"]),
                "amount": budget // 4,
                "reason": f"◎ {top3[0]['horse_name']} (win={top3[0].get('calibrated_win_prob', 0):.1%})",
            })
            if len(top3) >= 2:
                bets.append({
                    "bet_type": "馬連",
                    "combination": f"{top3[0]['horse_number']}-{top3[1]['horse_number']}",
                    "amount": budget // 4,
                    "reason": f"◎ {top3[0]['horse_name']} - ○ {top3[1]['horse_name']}",
                })
            if len(top3) >= 3:
                # 三連単: 1着固定で2-3着の組み合わせ
                n1 = top3[0]["horse_number"]
                n2 = top3[1]["horse_number"]
                n3 = top3[2]["horse_number"]
                bets.append({
                    "bet_type": "三連単",
                    "combination": f"{n1}→{n2}→{n3}",
                    "amount": budget // 6,
                    "reason": f"本線 ◎→○→▲",
                })
                bets.append({
                    "bet_type": "三連単",
                    "combination": f"{n1}→{n3}→{n2}",
                    "amount": budget // 6,
                    "reason": f"裏目 ◎→▲→○",
                })
                bets.append({
                    "bet_type": "ワイド",
                    "combination": f"{top3[0]['horse_number']}-{top3[2]['horse_number']}",
                    "amount": budget // 6,
                    "reason": f"◎ {top3[0]['horse_name']} - ▲ {top3[2]['horse_name']}",
                })
        else:
            # 回収率重視: EV > 1.0 の馬に単勝 (ケリー基準)
            by_ev = sorted(
                [p for p in predictions if p.get("expected_value") and p["expected_value"] > 1.0],
                key=lambda x: x["expected_value"],
                reverse=True,
            )
            if by_ev:
                for p in by_ev[:3]:
                    prob = p.get("calibrated_win_prob", 0)
                    odds = p.get("odds", 0)
                    if odds > 0 and prob > 0:
                        kelly = (prob * odds - 1) / (odds - 1) if odds > 1 else 0
                        kelly_fraction = max(0, kelly * 0.25)
                        amount = int(budget * kelly_fraction)
                        if amount >= 100:
                            bets.append({
                                "bet_type": "単勝",
                                "combination": str(p["horse_number"]),
                                "amount": amount,
                                "reason": f"EV={p['expected_value']:.2f} {p['horse_name']}",
                            })
            if not bets:
                # EV買い目なし → 的中率重視にフォールバック
                bets.append({
                    "bet_type": "単勝",
                    "combination": str(top3[0]["horse_number"]),
                    "amount": budget // 3,
                    "reason": f"EV不明: ◎ {top3[0]['horse_name']}",
                })
                if len(top3) >= 2:
                    bets.append({
                        "bet_type": "馬連",
                        "combination": f"{top3[0]['horse_number']}-{top3[1]['horse_number']}",
                        "amount": budget // 3,
                        "reason": f"EV不明: {top3[0]['horse_name']} - {top3[1]['horse_name']}",
                    })

        return bets
