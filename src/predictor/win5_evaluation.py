"""WIN5 バックテスト評価

walk_forward_cv と同じOOF予測をベースに、Win5TargetRace の5R組合せ単位で
hit probability 最大化モード (各レース top-1 を1点買い) を評価する。

EV モードと配当モデルはここでは扱わない (制度変更 2026-04-25 以降の
データが蓄積されるまで配当モデル学習を待つため)。
"""
from collections import defaultdict
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from src.common.database import Win5TargetRace

from .evaluation import produce_oof_predictions


# 制度変更日: 2026-04-25 から土曜 WIN5 通年発売開始
DEFAULT_REGIME_SPLIT_DATE = date(2026, 4, 25)


class Win5Evaluator:
    """OOF 予測 + Win5TargetRace から日次 WIN5 hit rate を算出"""

    def __init__(self, regime_split_date: date = DEFAULT_REGIME_SPLIT_DATE):
        self.regime_split_date = regime_split_date

    def evaluate(
        self,
        session: Session,
        oof_df: Optional[pd.DataFrame] = None,
        n_splits: int = 4,
        gap_days: int = 7,
    ) -> dict:
        """日次 WIN5 バックテスト

        Args:
            session: DBセッション
            oof_df: 事前生成済みの OOF DataFrame (None なら内部で生成)
            n_splits / gap_days: oof_df=None 時の produce_oof_predictions パラメータ

        Returns:
            dict:
                overall: 全期間集計
                pre_regime_change: 2026-04-25 より前
                post_regime_change: 2026-04-25 以降
                day_results: 日次の生データ
                regime_split_date: 分割日 (ISO)
                warnings: 注意メッセージ
        """
        warnings = []

        if oof_df is None:
            oof_df = produce_oof_predictions(session, n_splits=n_splits, gap_days=gap_days)
        if oof_df is None or oof_df.empty:
            return {"error": "OOF予測なし (訓練データ不足の可能性)"}

        # Win5TargetRace を race_id 解決済みのものだけ取得
        targets = (
            session.query(Win5TargetRace)
            .filter(Win5TargetRace.race_id.isnot(None))
            .order_by(Win5TargetRace.race_date, Win5TargetRace.leg_index)
            .all()
        )
        unresolved_count = (
            session.query(Win5TargetRace)
            .filter(Win5TargetRace.race_id.is_(None))
            .count()
        )
        if unresolved_count > 0:
            warnings.append(
                f"race_id 未解決の Win5TargetRace が {unresolved_count}件あり、評価から除外"
            )
        if not targets:
            return {
                "error": "Win5TargetRace (race_id 解決済み) が 0件。scrape-win5-targets + scrape をしてから inspect-win5-targets --resolve を実行してください",
                "warnings": warnings,
            }

        # 日付別にグルーピング
        by_date: dict[date, list[Win5TargetRace]] = defaultdict(list)
        for t in targets:
            by_date[t.race_date].append(t)

        # OOF をレース別にインデックス化
        oof_by_race: dict[int, pd.DataFrame] = {
            int(rid): g for rid, g in oof_df.groupby("race_id")
        }

        day_results = []
        for race_date_key, target_list in sorted(by_date.items()):
            if len(target_list) < 5:
                warnings.append(f"{race_date_key}: leg数不足 ({len(target_list)}/5) — skip")
                continue

            target_list.sort(key=lambda t: t.leg_index)
            legs_data = []
            incomplete_reason = None

            for t in target_list[:5]:
                g = oof_by_race.get(t.race_id)
                if g is None or g.empty:
                    incomplete_reason = f"race_id={t.race_id} OOF未収録"
                    break
                winner = g[g["finish_position"] == 1]
                if winner.empty:
                    incomplete_reason = f"race_id={t.race_id} 1着不明"
                    break
                top1_row = g.sort_values("pred_score", ascending=False).iloc[0]
                legs_data.append({
                    "race_id": t.race_id,
                    "leg_index": t.leg_index,
                    "top1_pick": int(top1_row["horse_number"]),
                    "top1_prob": float(top1_row["pred_win_prob_softmax"]),
                    "actual_winner": int(winner.iloc[0]["horse_number"]),
                    "winner_predicted_prob": float(winner.iloc[0]["pred_win_prob_softmax"]),
                })

            if incomplete_reason is not None or len(legs_data) != 5:
                warnings.append(f"{race_date_key}: {incomplete_reason or '5leg未揃い'}")
                continue

            top1_picks = [d["top1_pick"] for d in legs_data]
            actual_winners = [d["actual_winner"] for d in legs_data]
            predicted_hit_prob = float(np.prod([d["top1_prob"] for d in legs_data]))
            winner_predicted_prob = float(
                np.prod([d["winner_predicted_prob"] for d in legs_data])
            )
            per_leg_hits = [p == a for p, a in zip(top1_picks, actual_winners)]
            hit = all(per_leg_hits)

            day_results.append({
                "race_date": race_date_key,
                "hit": hit,
                "n_leg_hits": sum(per_leg_hits),
                "predicted_hit_prob": predicted_hit_prob,
                "winner_predicted_prob": winner_predicted_prob,
                "top1_picks": top1_picks,
                "actual_winners": actual_winners,
            })

        pre = [d for d in day_results if d["race_date"] < self.regime_split_date]
        post = [d for d in day_results if d["race_date"] >= self.regime_split_date]

        return {
            "overall": _aggregate_days(day_results),
            "pre_regime_change": _aggregate_days(pre),
            "post_regime_change": _aggregate_days(post),
            "regime_split_date": self.regime_split_date.isoformat(),
            "day_results": day_results,
            "warnings": warnings,
        }


def _aggregate_days(day_results: list[dict]) -> dict:
    """日次結果を集計"""
    n = len(day_results)
    if n == 0:
        return {"n_days": 0}

    n_hits = sum(1 for d in day_results if d["hit"])
    mean_pred_prob = float(np.mean([d["predicted_hit_prob"] for d in day_results]))
    mean_winner_prob = float(np.mean([d["winner_predicted_prob"] for d in day_results]))
    # 5Rのうち何本当たったか (部分一致)
    mean_leg_hits = float(np.mean([d["n_leg_hits"] for d in day_results]))

    return {
        "n_days": n,
        "actual_hit_rate": n_hits / n,
        "mean_predicted_hit_prob": mean_pred_prob,
        "mean_winner_predicted_prob": mean_winner_prob,
        "mean_leg_hits_per_day": mean_leg_hits,  # 0.0〜5.0
        "leg_top1_hit_rate": mean_leg_hits / 5.0,
    }
