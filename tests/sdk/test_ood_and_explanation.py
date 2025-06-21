import numpy as np
from DiaGuardianAI.sdk.ood_and_explanation import OODDetector, SelfExplainer

class DummyModel:
    def predict(self, X):
        return X + np.random.normal(0, 1, size=X.shape)

def test_ood_detector():
    model = DummyModel()
    detector = OODDetector(model, threshold=0.5)
    data = np.zeros((5, 1))
    assert isinstance(detector.is_ood(data), bool)

def test_self_explainer():
    explainer = SelfExplainer()
    context = {"trend": "rising", "meal_carbs": 50}
    rec = {"bolus": 2, "basal": 1.2, "safety_level": "normal"}
    explanation = explainer.explain(context, rec)
    assert "bolus" in explanation
