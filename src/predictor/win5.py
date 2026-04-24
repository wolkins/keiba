"""WIN5 買い目最適化

モード:
- hit: 確率積 (各レース top-1 を集めた組合せ) 降順
- ev: expected_value = model_prob × 想定払戻 / unit 降順
  配当モデルは v1 (市場暗黙確率近似) のみ。履歴回帰は未実装。

警告 (EVモード):
- 市場暗黙確率と実際のWIN5票数には強い乖離があり、v1 は近似でしか無い
- 制度変更 2026-04-25 以降のデータが溜まるまで、EV モード出力は
  「参考値」として扱う。資金管理はフラクショナルケリー等の分散抑制が必須。
"""
from dataclasses import dataclass, field
from itertools import product
from typing import Optional

import numpy as np


WIN5_POOL_RATE = 0.7  # JRA WIN5 払戻率 (70%)


@dataclass(slots=True)
class Win5TicketRecommendation:
    horse_numbers: list[int]          # [3, 7, 12, 5, 9]
    combination_key: str              # "3-7-12-5-9"
    combo_probability: float          # 5レース勝率の積 (モデル)
    amount: int
    # EVモード時のみ意味を持つ (hit モードでは None)
    combo_market_probability: Optional[float] = None
    expected_payout: Optional[float] = None      # 1票あたり想定払戻(円)
    expected_value: Optional[float] = None       # per unit (>1 で期待値プラス)


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


class Win5DividendModel:
    """WIN5 配当モデル v1 (市場暗黙確率近似)

    履歴データを必要としない簡易モデル。
    expected_winning_tickets = total_sales_est/unit * combo_market_prob * bias_adj
    expected_payout_per_ticket = (total_sales_est * pool_rate + carryover) / max(1, exp_tickets)

    制約:
    - combo_market_prob は5レース市場確率の独立積。実際の WIN5 票数分布とは乖離がある
    - bias_adj は履歴回帰で学習する余地があるが v1 は固定値 1.0
    - carryover=0, 通常日の売上 3億円前後を想定
    """
    def __init__(
        self,
        pool_rate: float = WIN5_POOL_RATE,
        unit_amount: int = 200,
        bias_adj: float = 1.0,
    ):
        self.pool_rate = pool_rate
        self.unit_amount = unit_amount
        self.bias_adj = bias_adj

    def predict_dividend(
        self,
        combo_market_prob: float,
        total_sales_est: int,
        carryover: int = 0,
    ) -> float:
        """1票あたり想定払戻金(円)

        Args:
            combo_market_prob: 5レース市場暗黙確率の積 (0〜1)
            total_sales_est: 想定発売金額(円)
            carryover: 持越金(円)
        """
        if total_sales_est <= 0:
            return 0.0
        exp_tickets_bought = total_sales_est / self.unit_amount
        exp_winning_tickets = max(
            1.0,
            exp_tickets_bought * max(combo_market_prob, 1e-12) * self.bias_adj,
        )
        pool = total_sales_est * self.pool_rate + max(0, carryover)
        return pool / exp_winning_tickets


class Win5Optimizer:
    """WIN5 買い目最適化 (hit / ev)"""

    def __init__(
        self,
        unit_amount: int = 200,
        dividend_model: Optional[Win5DividendModel] = None,
    ):
        self.unit_amount = unit_amount
        self.dividend_model = dividend_model or Win5DividendModel(unit_amount=unit_amount)

    def generate_tickets(
        self,
        race_probs: list[np.ndarray],
        horse_numbers: list[list[int]],
        budget: int,
        coverage_threshold: float = 0.9,
        mode: str = "hit",
        market_probs: Optional[list[np.ndarray]] = None,
        total_sales_est: Optional[int] = None,
        carryover: int = 0,
        min_ev: float = 0.0,
    ) -> Win5Recommendation:
        """WIN5 推奨買い目を生成

        Args:
            race_probs: 長さ5。各要素はレース内の馬別モデル勝率 (未正規化可)
            horse_numbers: 長さ5。race_probs と同順の馬番
            budget: 予算(円)
            coverage_threshold: 候補絞り込みの累積確率閾値 (0-1)
            mode: "hit" (確率積最大) or "ev" (期待値最大)
            market_probs: EVモード必須。長さ5、各要素はレース内の馬別市場暗黙確率
            total_sales_est: EVモード必須。想定発売金額(円)
            carryover: キャリーオーバー(円)。0なら通常日
            min_ev: EVモード時の採択下限 (expected_value >= min_ev のみ採用)

        Returns:
            Win5Recommendation
        """
        if mode not in ("hit", "ev"):
            raise ValueError(f"mode={mode} は未サポート (hit / ev)")
        if len(race_probs) != 5 or len(horse_numbers) != 5:
            raise ValueError("5レース分の入力が必要")
        if mode == "ev":
            if market_probs is None or len(market_probs) != 5:
                raise ValueError("EVモードには market_probs (長さ5) が必要")
            if total_sales_est is None or total_sales_est <= 0:
                raise ValueError("EVモードには total_sales_est (>0) が必要")

        max_tickets = budget // self.unit_amount
        if max_tickets <= 0:
            return _empty_recommendation(mode, budget, self.unit_amount, coverage_threshold)

        # 各レッグでモデル確率を正規化 → 累積確率で候補絞り込み
        selected_horses_by_leg: list[list[int]] = []
        selected_probs_by_leg: list[list[float]] = []
        selected_market_probs_by_leg: list[list[float]] = []
        # 市場確率 (正規化後) を horse_number -> prob でルックアップできるように作る
        market_prob_maps = _normalize_market_probs(market_probs, horse_numbers) if market_probs is not None else None

        for leg_idx, (probs_raw, horse_nums) in enumerate(zip(race_probs, horse_numbers)):
            probs_raw = np.asarray(probs_raw, dtype=float)
            if probs_raw.size == 0 or probs_raw.sum() <= 0:
                fallback_horse = [horse_nums[0]] if horse_nums else []
                fallback_prob = [1.0] if fallback_horse else []
                selected_horses_by_leg.append(fallback_horse)
                selected_probs_by_leg.append(fallback_prob)
                if market_prob_maps is not None:
                    mkt = market_prob_maps[leg_idx]
                    selected_market_probs_by_leg.append(
                        [mkt.get(fallback_horse[0], 0.0)] if fallback_horse else []
                    )
                else:
                    selected_market_probs_by_leg.append([])
                continue
            probs = probs_raw / probs_raw.sum()
            order = np.argsort(-probs)
            cum = 0.0
            leg_horses: list[int] = []
            leg_probs: list[float] = []
            leg_market_probs: list[float] = []
            for idx in order:
                h = int(horse_nums[idx])
                leg_horses.append(h)
                leg_probs.append(float(probs[idx]))
                if market_prob_maps is not None:
                    leg_market_probs.append(float(market_prob_maps[leg_idx].get(h, 0.0)))
                cum += float(probs[idx])
                if cum >= coverage_threshold:
                    break
            selected_horses_by_leg.append(leg_horses)
            selected_probs_by_leg.append(leg_probs)
            selected_market_probs_by_leg.append(leg_market_probs)

        # 直積
        all_combos = []
        for idx_tuple in product(*[range(len(h)) for h in selected_horses_by_leg]):
            combo_model_prob = 1.0
            combo_market_prob = 1.0
            for leg, i in enumerate(idx_tuple):
                combo_model_prob *= selected_probs_by_leg[leg][i]
                if selected_market_probs_by_leg[leg]:
                    combo_market_prob *= selected_market_probs_by_leg[leg][i]
            all_combos.append((idx_tuple, combo_model_prob, combo_market_prob))

        # モード別スコアリング
        scored: list[tuple[float, tuple, float, float, Optional[float], Optional[float]]] = []
        # (rank_score, idx_tuple, model_prob, market_prob, expected_payout, expected_value)
        for idx_tuple, mp, cmp_ in all_combos:
            if mode == "hit":
                scored.append((mp, idx_tuple, mp, cmp_, None, None))
            else:  # ev
                expected_payout = self.dividend_model.predict_dividend(
                    combo_market_prob=cmp_,
                    total_sales_est=total_sales_est,
                    carryover=carryover,
                )
                expected_value = (mp * expected_payout) / self.unit_amount
                if expected_value < min_ev:
                    continue
                scored.append((expected_value, idx_tuple, mp, cmp_, expected_payout, expected_value))

        scored.sort(key=lambda x: x[0], reverse=True)
        selected_scored = scored[:max_tickets]

        tickets: list[Win5TicketRecommendation] = []
        for rank_score, idx_tuple, mp, cmp_, ep, ev in selected_scored:
            horse_nums = [
                selected_horses_by_leg[leg][i] for leg, i in enumerate(idx_tuple)
            ]
            key = "-".join(str(n) for n in horse_nums)
            tickets.append(Win5TicketRecommendation(
                horse_numbers=horse_nums,
                combination_key=key,
                combo_probability=mp,
                amount=self.unit_amount,
                combo_market_probability=cmp_ if mode == "ev" else None,
                expected_payout=ep,
                expected_value=ev,
            ))

        hit_prob_sum = float(sum(t.combo_probability for t in tickets))
        top_prob = float(tickets[0].combo_probability) if tickets else 0.0
        total_cost = len(tickets) * self.unit_amount

        meta = {
            "n_combinations_evaluated": len(all_combos),
            "n_horses_per_leg": [len(h) for h in selected_horses_by_leg],
        }
        if mode == "ev":
            meta["total_sales_est"] = total_sales_est
            meta["carryover"] = carryover
            meta["min_ev"] = min_ev
            meta["n_combinations_above_min_ev"] = len(scored)
            if tickets:
                ev_values = [t.expected_value for t in tickets if t.expected_value is not None]
                if ev_values:
                    meta["mean_expected_value"] = float(np.mean(ev_values))
                    meta["top_expected_value"] = float(max(ev_values))

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
            meta=meta,
        )


def _normalize_market_probs(
    market_probs: list[np.ndarray],
    horse_numbers: list[list[int]],
) -> list[dict[int, float]]:
    """各レッグの市場確率をレース内で正規化 (オーバーラウンド除去) → {horse_number: prob} map"""
    maps = []
    for probs_raw, horse_nums in zip(market_probs, horse_numbers):
        arr = np.asarray(probs_raw, dtype=float)
        if arr.size == 0 or arr.sum() <= 0:
            maps.append({int(h): 0.0 for h in horse_nums})
            continue
        normed = arr / arr.sum()
        maps.append({int(h): float(p) for h, p in zip(horse_nums, normed)})
    return maps


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


def build_win5_market_inputs(
    predictions_by_race: list[list[dict]],
    odds_key: str = "odds",
) -> list[np.ndarray]:
    """predict() dict の odds から市場暗黙確率 (未正規化: 1/odds) を抽出

    正規化は Win5Optimizer 側で行うので、ここでは 1/odds の生値を返す。
    odds=0 や欠損は 0.0 扱い。

    Returns:
        長さ5の list[np.ndarray]、各要素は predictions と同順の 1/odds 配列
    """
    market_probs: list[np.ndarray] = []
    for preds in predictions_by_race:
        vals = []
        for p in preds:
            odds = p.get(odds_key, 0) or 0
            vals.append(1.0 / odds if odds > 0 else 0.0)
        market_probs.append(np.array(vals))
    return market_probs


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
