import numpy as np
import sys
import types

# Mock optional dependency to avoid import errors during tests
sys.modules.setdefault('optuna', types.ModuleType('optuna'))

from DiaGuardianAI.models.transformer_zoo import TransformerZoo

class DummyPredictor:
    def __init__(self, horizons):
        self.is_trained = True
        self.horizons = horizons
    def predict_multi_horizon(self, X):
        return {h: np.full(X.shape[0], float(h)) for h in self.horizons}
    def get_model_info(self):
        return {
            "name": "dummy",
            "is_trained": True,
            "input_dim": len(self.horizons),
            "prediction_horizons": self.horizons,
        }


def test_evaluate_all_models_dummy():
    horizons = [10, 20]
    zoo = TransformerZoo(input_dim=3)
    zoo.prediction_horizons = horizons
    zoo.models = {"dummy": DummyPredictor(horizons)}

    X = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
    y = np.array([[10, 20], [10, 20]], dtype=float)

    results = zoo.evaluate_all_models(X, y)
    assert "dummy" in results
    for h in horizons:
        assert h in results["dummy"]
        assert results["dummy"][h] == 0.0
