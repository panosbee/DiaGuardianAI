import os
import sys
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from DiaGuardianAI.predictive_models.model_zoo.ensemble_predictor import EnsemblePredictor
from DiaGuardianAI.core.base_classes import BasePredictiveModel

class DummyModel(BasePredictiveModel):
    def __init__(self, value, length=3):
        self.value = value
        self.length = length
    def train(self, data, targets=None, **kwargs):
        pass
    def predict(self, current_input, **kwargs):
        mean = [self.value for _ in range(self.length)]
        std = [0.1 * self.value for _ in range(self.length)]
        return {"mean": mean, "std_dev": std}
    def save(self, path):
        pass
    def load(self, path):
        pass

def test_ensemble_average_strategy():
    models = [DummyModel(100), DummyModel(110)]
    ensemble = EnsemblePredictor(models=models, strategy="average")
    preds = ensemble.predict(None)
    assert preds["mean"] == [105.0, 105.0, 105.0]
    assert all(isinstance(v, float) for v in preds["std_dev"])


def test_ensemble_weighted_average_strategy():
    models = [DummyModel(80), DummyModel(120)]
    ensemble = EnsemblePredictor(models=models, strategy="weighted_average", weights=[0.25, 0.75])
    preds = ensemble.predict(None)
    expected = 80*0.25 + 120*0.75
    assert preds["mean"][0] == expected
