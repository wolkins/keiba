"""WIN5 買い目最適化 (hit-only mode v1)

方針:
- 各レースで累積確率 ≥ coverage_threshold になるまで馬を候補採用
- 直積を作り、各組合せの確率積を降順でソート
- budget // unit_amount 点を上位から採択

EV モードと配当モデルは未実装 (制度変更 2026-04-25 以降のデータ蓄積待ち)。
"""
from dataclasses import dataclass, field
from itertools import product
from typing import Optional

import numpy as np


@dataclass(slots=True)
class Win5TicketRecommendation:
    horse_numbers: list[int]          # [3, 7, 12, 5, 9]
    combination_key: str              # "3-7-12-5-9"
    combo_probability: float          # 5レース勝率の積
    amount: int


@dataclass(slots=True)
class Win5Recommendation:
    mode: str
    budget: int
    unit_amount: int
    coverage_threshold: float
    total_tickets: int
    total_cost: int

    # 各レッグの採用馬番リスト (sorted by prob desc)
    selected_horses_by_leg: list[list[int]]
    # 参考: 各レッグの採用馬の予測確率 (絞り込み後、再正規化はしない)
    selected_probs_by_leg: list[list[float]]

    tickets: list[Win5TicketRecommendation]
    # 買った全点の勝率積の和 = この組合せセットで的中する総確率 (≈想定hit rate)
    hit_probability_sum: float
    # 最も確度の高い1点の確率
    top_ticket_probability: float

    # 入力として使った正規化後の勝率 (各レッグ, 18頭分で0埋めされる形式で保持するのは重いので
    # dataclass に詰めない。必要なら外で保持)
    meta: dict = field(default_factory=dict)


class Win5Optimizer:
    """hit-only モード v1 の最適化器"""

    def __init__(self, unit_amount: int = 200):
        self.unit_amount = unit_amount

    def generate_tickets(
        self,
        race_probs: list[np.ndarray],
        horse_numbers: list[list[int]],
        budget: int,
        coverage_threshold: float = 0.9,
        mode: str = "hit",
    ) -> Win5Recommendation:
        """WIN5 推奨買い目を生成

        Args:
            race_probs: 長さ5のリスト。各要素はそのレースの馬別勝率 np.ndarray (未正規化も可)
            horse_numbers: 長さ5のリスト。各要素は race_probs と同順の馬番リスト
            budget: 予算(円)
            coverage_threshold: 候補絞り込みの累積確率閾値 (0-1)
            mode: "hit" のみサポート

        Returns:
            Win5Recommendation
        """
        if mode != "hit":
            raise NotImplementedError(f"mode={mode} は v1 では未サポート (hitのみ)")
        if len(race_probs) != 5 or len(horse_numbers) != 5:
            raise ValueError("5レース分の入力が必要")

        max_tickets = budget // self.unit_amount
        if max_tickets <= 0:
            return _empty_recommendation(mode, budget, self.unit_amount, coverage_threshold)

        # 各レッグでレース内正規化 → 累積確率で候補絞り込み
        selected_horses_by_leg: list[list[int]] = []
        selected_probs_by_leg: list[list[float]] = []
        for probs_raw, horse_nums in zip(race_probs, horse_numbers):
            probs_raw = np.asarray(probs_raw, dtype=float)
            if probs_raw.size == 0 or probs_raw.sum() <= 0:
                # 情報なし → 1頭目のみに賭ける (fallback)
                fallback_horse = [horse_nums[0]] if horse_nums else []
                fallback_prob = [1.0] if fallback_horse else []
                selected_horses_by_leg.append(fallback_horse)
                selected_probs_by_leg.append(fallback_prob)
                continue
            probs = probs_raw / probs_raw.sum()
            order = np.argsort(-probs)
            cum = 0.0
            leg_horses: list[int] = []
            leg_probs: list[float] = []
            for idx in order:
                leg_horses.append(int(horse_nums[idx]))
                leg_probs.append(float(probs[idx]))
                cum += float(probs[idx])
                if cum >= coverage_threshold:
                    break
            selected_horses_by_leg.append(leg_horses)
            selected_probs_by_leg.append(leg_probs)

        # 直積 (この時点で各レッグは少数なので組合せ数は大きすぎない想定)
        all_combos = []
        for idx_tuple in product(*[range(len(h)) for h in selected_horses_by_leg]):
            combo_prob = 1.0
            for leg, i in enumerate(idx_tuple):
                combo_prob *= selected_probs_by_leg[leg][i]
            all_combos.append((combo_prob, idx_tuple))

        # 勝率積降順
        all_combos.sort(key=lambda x: x[0], reverse=True)

        # 予算内 top-N
        selected_combos = all_combos[:max_tickets]

        tickets: list[Win5TicketRecommendation] = []
        for combo_prob, idx_tuple in selected_combos:
            horse_nums = [
                selected_horses_by_leg[leg][i] for leg, i in enumerate(idx_tuple)
            ]
            key = "-".join(str(n) for n in horse_nums)
            tickets.append(Win5TicketRecommendation(
                horse_numbers=horse_nums,
                combination_key=key,
                combo_probability=combo_prob,
                amount=self.unit_amount,
            ))

        hit_prob_sum = float(sum(t.combo_probability for t in tickets))
        top_prob = float(tickets[0].combo_probability) if tickets else 0.0
        total_cost = len(tickets) * self.unit_amount

        return Win5Recommendation(
            mode=mode,
            budget=budget,
            unit_amount=self.unit_amount,
            coverage_threshold=coverage_threshold,
            total_tickets=len(tickets),
            total_cost=total_cost,
            selected_horses_by_leg=selected_horses_by_leg,
            selected_probs_by_leg=selected_probs_by_leg,
            tickets=tickets,
            hit_probability_sum=hit_prob_sum,
            top_ticket_probability=top_prob,
            meta={
                "n_combinations_evaluated": len(all_combos),
                "n_horses_per_leg": [len(h) for h in selected_horses_by_leg],
            },
        )


def build_win5_race_inputs(
    predictions_by_race: list[list[dict]],
    prob_key: str = "raw_win_prob",
) -> tuple[list[np.ndarray], list[list[int]]]:
    """KeibaPredictor.predict() の結果を Win5Optimizer 入力へ変換

    Args:
        predictions_by_race: 5レース分。各要素は KeibaPredictor.predict() の戻り値 list[dict]
        prob_key: 使う確率フィールド。raw_win_prob (丸めなし) を推奨

    Returns:
        (race_probs, horse_numbers):
            race_probs[i] は np.ndarray、同順の horse_numbers[i] と対応
    """
    race_probs: list[np.ndarray] = []
    horse_numbers: list[list[int]] = []
    for preds in predictions_by_race:
        probs = np.array([float(p.get(prob_key, 0.0) or 0.0) for p in preds])
        horses = [int(p["horse_number"]) for p in preds]
        race_probs.append(probs)
        horse_numbers.append(horses)
    return race_probs, horse_numbers


def _empty_recommendation(
    mode: str, budget: int, unit_amount: int, coverage_threshold: float,
) -> Win5Recommendation:
    return Win5Recommendation(
        mode=mode,
        budget=budget,
        unit_amount=unit_amount,
        coverage_threshold=coverage_threshold,
        total_tickets=0,
        total_cost=0,
        selected_horses_by_leg=[[] for _ in range(5)],
        selected_probs_by_leg=[[] for _ in range(5)],
        tickets=[],
        hit_probability_sum=0.0,
        top_ticket_probability=0.0,
    )
