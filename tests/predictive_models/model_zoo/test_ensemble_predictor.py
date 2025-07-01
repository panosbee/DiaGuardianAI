import numpy as np
import pytest
import sys
import types

# Mock optional dependency to avoid import errors during tests
sys.modules.setdefault('optuna', types.ModuleType('optuna'))

from DiaGuardianAI.predictive_models.model_zoo.ensemble_predictor import EnsemblePredictor
from DiaGuardianAI.core.base_classes import BasePredictiveModel

class DummyModel(BasePredictiveModel):
    def __init__(self, value: float, output_len: int = 3):
        super().__init__()
        self.value = value
        self.output_len = output_len
        self.trained = False

    def train(self, X_train, y_train):
        self.trained = True

    def predict(self, X_current_state):
        return {
            "mean": [self.value] * self.output_len,
            "std_dev": [0.0] * self.output_len,
        }

    def save(self, path: str):
        pass

    def load(self, path: str):
        pass


def test_train_calls_submodels_and_average_prediction():
    models = [DummyModel(1.0), DummyModel(2.0)]
    ens = EnsemblePredictor(models=models, strategy="average")
    X = np.zeros((1, 1))
    y = np.array([0.0, 0.0, 0.0])
    ens.train(X, y)
    assert all(m.trained for m in models)
    preds = ens.predict(X)
    assert preds["mean"] == [1.5, 1.5, 1.5]


def test_stacking_trains_meta_learner_and_predicts():
    models = [DummyModel(1.0), DummyModel(2.0)]
    ens = EnsemblePredictor(models=models, strategy="stacking")
    X = np.zeros((1, 1))
    y = np.array([1.5, 1.5, 1.5])
    ens.train(X, y)
    assert hasattr(ens.meta_learner, "coef_")
    preds = ens.predict(X)
    assert np.allclose(preds["mean"], y)
