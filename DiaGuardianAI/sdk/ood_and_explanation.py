"""Out-of-Distribution detection and self-explanation utilities."""

from __future__ import annotations

import numpy as np
from typing import Any, Dict, Tuple


class OODDetector:
    """Simple uncertainty-based out-of-distribution detector."""

    def __init__(self, model: Any, threshold: float = 0.95) -> None:
        self.model = model
        self.threshold = float(threshold)

    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return mean and std deviation of predictions using Monte Carlo sampling."""
        preds = [self.model.predict(X) for _ in range(10)]
        mean_pred = np.mean(preds, axis=0)
        std_pred = np.std(preds, axis=0)
        return mean_pred, std_pred

    def is_ood(self, X: np.ndarray) -> bool:
        """Check if input appears out-of-distribution based on prediction variance."""
        _, std_pred = self.predict_with_uncertainty(X)
        return np.max(std_pred) > self.threshold


class SelfExplainer:
    """Generate simple natural language explanations for recommendations."""

    def explain(self, input_context: Dict[str, Any], recommendation: Dict[str, Any]) -> str:
        bolus = recommendation.get("bolus", 0)
        basal = recommendation.get("basal", 0)
        trend = input_context.get("trend")
        meal_carbs = input_context.get("meal_carbs")
        safety = recommendation.get("safety_level", "unknown")
        explanation = (
            f"\u03A0\u03C1\u03BF\u03C4\u03B5\u03AF\u03BD\u03C9 {bolus}U bolus "
            f"\u03BA\u03B1\u03B9 {basal}U/hr basal "
            f"\u03B5\u03C0\u03B5\u03B9\u03B4\u03AE {trend} "
            f"\u03C0\u03B1\u03C1\u03B1\u03C4\u03B7\u03C1\u03AE\u03C3\u03B5 {meal_carbs}g. "
            f"\u0395\u03C0\u03AF\u03C0\u03B5\u03B4\u03BF \u03B1\u03C3\u03C6\u03AC\u03BB\u03B5\u03B9\u03B1\u03C2: {safety}."
        )
        return explanation
