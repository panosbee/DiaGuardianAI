import os
import sys
from datetime import datetime, timedelta
import numpy as np
import pytest

project_root_for_tests = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root_for_tests not in sys.path:
    sys.path.insert(0, project_root_for_tests)

from DiaGuardianAI.agents.meal_detector import MealDetector


def _make_sequence(start: float, delta: float, length: int = 12):
    return [start + delta * i for i in range(length)]


def _timestamps(length: int):
    base = datetime(2024, 1, 1, 12, 0)
    return [base + timedelta(minutes=5 * i) for i in range(length)]


@pytest.fixture
def ml_detector():
    return MealDetector(
        detection_method="ml_based",
        params={"model_config": {"n_estimators": 10, "random_state": 42}},
    )


def test_ml_training_and_detection(ml_detector):
    meal_seqs = [_make_sequence(100, 5), _make_sequence(110, 4)]
    no_meal_seqs = [_make_sequence(100, 0), _make_sequence(120, -1)]

    X = []
    y = []
    for seq in meal_seqs:
        X.append(ml_detector._extract_features_for_ml(seq, _timestamps(len(seq))))
        y.append(1)
    for seq in no_meal_seqs:
        X.append(ml_detector._extract_features_for_ml(seq, _timestamps(len(seq))))
        y.append(0)

    X = np.vstack(X)
    y = np.array(y)

    ml_detector.train_model(X, y)

    meal_test = _make_sequence(105, 6)
    no_meal_test = _make_sequence(115, 0)

    prob_meal, _, _ = ml_detector.detect_meal_event(meal_test, _timestamps(len(meal_test)))
    prob_none, _, _ = ml_detector.detect_meal_event(no_meal_test, _timestamps(len(no_meal_test)))

    assert 0.0 <= prob_meal <= 1.0
    assert 0.0 <= prob_none <= 1.0
    assert prob_meal > prob_none


def test_untrained_ml_detector_returns_zero():
    detector = MealDetector(detection_method="ml_based")
    prob, t, carbs = detector.detect_meal_event([100, 102, 104], _timestamps(3))
    assert prob == 0.0
    assert t is None
    assert carbs is None
