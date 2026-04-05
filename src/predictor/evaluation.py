"""評価基盤: ウォークフォワードCV + 多軸指標"""
import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from src.common.database import Odds, Race

from .features import FEATURE_COLUMNS, build_features_for_race


def evaluate_predictions(valid_df: pd.DataFrame, session: Session = None) -> dict:
    """予測結果に対して多軸指標を算出する

    valid_df には以下のカラムが必要:
      race_id, horse_number, finish_position, pred_score
    オプション: odds_value (収益性能の算出に使用)

    Returns:
        dict: ndcg3, mrr, top1_hit_rate, top3_exact_rate, roi_simulation, avg_win_payout
    """
    race_ids = valid_df["race_id"].unique()

    ndcg_list = []
    mrr_list = []
    top1_hits = 0
    top3_exact = 0
    total_bet = 0
    total_payout = 0.0
    win_payouts = []
    n_races = 0

    for rid in race_ids:
        race_data = valid_df[valid_df["race_id"] == rid].copy()
        if race_data.empty or race_data["finish_position"].isna().all():
            continue
        n_races += 1

        # finish_positionがNaNの行を除外（出走取消等）
        race_data = race_data.dropna(subset=["finish_position"])
        if race_data.empty:
            continue

        # 予測スコア降順でソート
        race_data = race_data.sort_values("pred_score", ascending=False).reset_index(drop=True)
        positions = race_data["finish_position"].values

        # --- NDCG@3 ---
        ndcg_list.append(_ndcg_at_k(positions, k=3))

        # --- MRR (1着の予測ランク逆数) ---
        rank_of_winner = np.where(positions == 1)[0]
        if len(rank_of_winner) > 0:
            mrr_list.append(1.0 / (rank_of_winner[0] + 1))
        else:
            mrr_list.append(0.0)

        # --- Top1 hit rate ---
        if len(positions) > 0 and positions[0] == 1:
            top1_hits += 1

        # --- Top3 exact match ---
        pred_top3 = set(race_data.head(3)["horse_number"].values)
        actual_top3 = set(race_data.nsmallest(3, "finish_position")["horse_number"].values)
        if pred_top3 == actual_top3:
            top3_exact += 1

        # --- 収益シミュレーション: 単勝上位1点100円 ---
        total_bet += 100
        top1_horse = race_data.iloc[0]["horse_number"]
        top1_finish = race_data.iloc[0]["finish_position"]
        if top1_finish == 1:
            # オッズを取得
            odds_val = None
            if "odds_value" in race_data.columns:
                odds_val = race_data.iloc[0].get("odds_value")
            if odds_val and odds_val > 0:
                payout = odds_val * 100
                total_payout += payout
                win_payouts.append(payout)
            elif session is not None:
                odds_row = (
                    session.query(Odds)
                    .filter_by(race_id=int(rid), bet_type="win")
                    .filter(Odds.combination == str(int(top1_horse)))
                    .first()
                )
                if odds_row and odds_row.odds_value:
                    payout = odds_row.odds_value * 100
                    total_payout += payout
                    win_payouts.append(payout)

    if n_races == 0:
        return {
            "ndcg3": 0.0, "mrr": 0.0,
            "top1_hit_rate": 0.0, "top3_exact_rate": 0.0,
            "roi_simulation": 0.0, "avg_win_payout": 0.0,
            "n_eval_races": 0,
        }

    return {
        "ndcg3": float(np.mean(ndcg_list)),
        "mrr": float(np.mean(mrr_list)),
        "top1_hit_rate": top1_hits / n_races,
        "top3_exact_rate": top3_exact / n_races,
        "roi_simulation": total_payout / total_bet if total_bet > 0 else 0.0,
        "avg_win_payout": float(np.mean(win_payouts)) if win_payouts else 0.0,
        "n_eval_races": n_races,
    }


def _ndcg_at_k(positions: np.ndarray, k: int = 3) -> float:
    """予測順位の上位k件に対するNDCGを算出

    relevance = max_position + 1 - finish_position (高いほど良い着順)
    """
    max_pos = positions.max() if len(positions) > 0 else 1
    relevance = np.clip(max_pos + 1 - positions, 0, None)

    # DCG@k
    top_k_rel = relevance[:k]
    dcg = np.sum(top_k_rel / np.log2(np.arange(1, len(top_k_rel) + 1) + 1))

    # ideal DCG@k
    ideal_rel = np.sort(relevance)[::-1][:k]
    idcg = np.sum(ideal_rel / np.log2(np.arange(1, len(ideal_rel) + 1) + 1))

    if idcg == 0:
        return 0.0
    return dcg / idcg


def evaluate_by_group(results_df: pd.DataFrame, group_key: str, session: Session = None) -> dict:
    """層別評価: group_key (例: 'racecourse_id', 'surface') でグループ化して指標算出

    results_df には race_id, horse_number, finish_position, pred_score と group_key カラムが必要

    Returns:
        dict: {group_value: evaluate_predictions結果}
    """
    if group_key not in results_df.columns:
        return {}

    group_results = {}
    for gval, group_df in results_df.groupby(group_key):
        group_results[gval] = evaluate_predictions(group_df, session=session)
    return group_results


def walk_forward_cv(session: Session, n_splits: int = 4, gap_days: int = 7) -> dict:
    """ウォークフォワードCVでモデル評価

    Args:
        session: DBセッション
        n_splits: fold数
        gap_days: 学習期間と検証期間のギャップ日数

    Returns:
        dict: fold_results (各foldの指標), mean (平均), std (標準偏差)
    """
    from datetime import timedelta

    import lightgbm as lgb

    from src.common.config import DECAY_HALF_LIFE_DAYS

    # 全レースを日付順で取得
    races = (
        session.query(Race)
        .filter(Race.status == "finished")
        .order_by(Race.race_date)
        .all()
    )
    if len(races) < 50:
        return {"error": f"レース数不足 ({len(races)}/50)"}

    # レースごとに特徴量を構築
    all_dfs = []
    race_date_map = {}
    for race in races:
        df = build_features_for_race(session, race)
        if not df.empty and "finish_position" in df.columns:
            df["race_id"] = race.id
            df["race_date"] = race.race_date
            all_dfs.append(df)
            race_date_map[race.id] = race.race_date

    if not all_dfs:
        return {"error": "有効なデータなし"}

    full_df = pd.concat(all_dfs, ignore_index=True)
    max_pos = full_df["finish_position"].max()
    full_df["label"] = (max_pos + 1 - full_df["finish_position"]).clip(lower=0)

    # 日付のユニーク値からfold分割を決定
    unique_dates = sorted(full_df["race_date"].unique())
    n_dates = len(unique_dates)
    # 検証期間は全体の約20%をn_splitsで分割
    valid_size = max(1, n_dates // (n_splits + 4))

    fold_results = []

    for fold_i in range(n_splits):
        # 検証期間の開始位置: 後半部分をn_splits等分
        valid_start_idx = n_dates - (n_splits - fold_i) * valid_size
        valid_end_idx = valid_start_idx + valid_size
        if valid_start_idx < 1:
            continue

        valid_dates = set(unique_dates[valid_start_idx:valid_end_idx])

        # gap_days前までを学習データとする
        gap_threshold = min(valid_dates) - timedelta(days=gap_days)
        train_dates = set(d for d in unique_dates[:valid_start_idx] if d <= gap_threshold)

        if not train_dates or not valid_dates:
            continue

        train_df = full_df[full_df["race_date"].isin(train_dates)].copy()
        valid_df = full_df[full_df["race_date"].isin(valid_dates)].copy()

        if len(train_df) < 30 or len(valid_df) < 10:
            continue

        # 時間減衰重み
        reference_date = max(train_dates)
        train_df["sample_weight"] = train_df["race_date"].apply(
            lambda d: np.exp(
                -np.log(2) * (reference_date - d).days / DECAY_HALF_LIFE_DAYS
            )
        )

        available_cols = [c for c in FEATURE_COLUMNS if c in train_df.columns]

        X_train = train_df[available_cols].fillna(0)
        y_train = train_df["label"]
        w_train = train_df["sample_weight"]
        group_train = train_df.groupby("race_id").size().tolist()

        X_valid = valid_df[available_cols].fillna(0)
        y_valid = valid_df["label"]
        group_valid = valid_df.groupby("race_id").size().tolist()

        # LightGBM LambdaRank (label_gain + seedアンサンブル)
        max_pos = int(train_df["finish_position"].max())
        n_labels = max_pos + 1
        label_gain = [0.0] * n_labels
        for i in range(n_labels):
            pos = max_pos - i
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

        # seedアンサンブル: 3モデル(evaluateは速度重視で少なめ)
        seed_scores = []
        for seed in range(3):
            params = {**base_params, "random_state": seed * 42 + 7}
            ranker = lgb.LGBMRanker(**params)
            callbacks = [lgb.early_stopping(100, verbose=False)]
            ranker.fit(
                X_train, y_train, group=group_train,
                sample_weight=w_train,
                eval_set=[(X_valid, y_valid)],
                eval_group=[group_valid],
                callbacks=callbacks,
            )
            seed_scores.append(ranker.predict(X_valid))

        # 予測（seedアンサンブルの平均）
        valid_df["pred_score"] = np.mean(seed_scores, axis=0)

        # オッズ情報を付加
        odds_map = {}
        for rid in valid_df["race_id"].unique():
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

        metrics = evaluate_predictions(valid_df, session=session)
        metrics["fold"] = fold_i + 1
        metrics["n_train_dates"] = len(train_dates)
        metrics["n_valid_dates"] = len(valid_dates)
        fold_results.append(metrics)

    if not fold_results:
        return {"error": "有効なfoldを作成できませんでした"}

    # 平均・標準偏差
    metric_keys = ["ndcg3", "mrr", "top1_hit_rate", "top3_exact_rate", "roi_simulation"]
    mean_metrics = {}
    std_metrics = {}
    for key in metric_keys:
        values = [f[key] for f in fold_results if key in f]
        if values:
            mean_metrics[key] = float(np.mean(values))
            std_metrics[key] = float(np.std(values))

    return {
        "fold_results": fold_results,
        "mean": mean_metrics,
        "std": std_metrics,
        "n_folds": len(fold_results),
    }
