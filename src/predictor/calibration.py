"""確率キャリブレーション: LambdaRankスコア → 校正確率"""
import pickle
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression

from src.common.config import PROJECT_ROOT

MODEL_DIR = PROJECT_ROOT / "data" / "models"


class ProbabilityCalibrator:
    """Isotonic回帰による確率キャリブレータ"""

    def __init__(self, name: str = "win"):
        """
        Args:
            name: キャリブレータ名 ('win' or 'top3')
        """
        self.name = name
        self.model = IsotonicRegression(out_of_bounds="clip")
        self._fitted = False
        self._save_path = MODEL_DIR / f"calibrator_{name}.pkl"

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> "ProbabilityCalibrator":
        """LambdaRankスコアとbinaryラベルでIsotonic回帰をfit

        Args:
            scores: LambdaRankの予測スコア
            labels: バイナリラベル (win: 1着=1, それ以外=0 / top3: 3着以内=1, それ以外=0)

        Returns:
            self
        """
        scores = np.asarray(scores, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.float64)

        if len(scores) < 2:
            self._fitted = False
            return self

        self.model.fit(scores, labels)
        self._fitted = True
        return self

    def predict_proba(self, scores: np.ndarray) -> np.ndarray:
        """校正された確率を返す

        Args:
            scores: LambdaRankの予測スコア

        Returns:
            校正済み確率 (0-1)
        """
        scores = np.asarray(scores, dtype=np.float64)
        if not self._fitted:
            # 未fitの場合はmin-max正規化でフォールバック
            s_min, s_max = scores.min(), scores.max()
            if s_max > s_min:
                return (scores - s_min) / (s_max - s_min)
            return np.ones_like(scores) * 0.5

        proba = self.model.predict(scores)
        return np.clip(proba, 0.0, 1.0)

    def save(self) -> None:
        """キャリブレータをファイルに保存"""
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        with open(self._save_path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "fitted": self._fitted,
                "name": self.name,
            }, f)

    def load(self) -> bool:
        """キャリブレータをファイルから読み込み

        Returns:
            bool: 読み込みに成功したかどうか
        """
        if not self._save_path.exists():
            return False
        with open(self._save_path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self._fitted = data["fitted"]
        self.name = data.get("name", self.name)
        return True

    @property
    def is_fitted(self) -> bool:
        return self._fitted
