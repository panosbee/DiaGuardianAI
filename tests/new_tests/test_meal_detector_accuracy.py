import os
import sys
from datetime import datetime, timedelta

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from DiaGuardianAI.agents.meal_detector import MealDetector


def test_meal_detector_rule_based_detects_meal():
    detector = MealDetector(
        detection_method="rule_based",
        params={"rise_threshold_mg_dl": 20, "time_window_minutes": 30, "min_samples_in_window": 3}
    )
    start = datetime(2023, 1, 1, 12, 0)
    cgm_history = [100, 105, 115, 130]
    timestamps = [start + timedelta(minutes=5*i) for i in range(len(cgm_history))]
    prob, est_time, carbs = detector.detect_meal_event(cgm_history, timestamps)
    assert prob > 0.5
    assert est_time == timestamps[0]
    assert carbs is not None


def test_meal_detector_rule_based_no_meal():
    detector = MealDetector(
        detection_method="rule_based",
        params={"rise_threshold_mg_dl": 25, "time_window_minutes": 30, "min_samples_in_window": 3}
    )
    start = datetime(2023, 1, 1, 12, 0)
    cgm_history = [100, 102, 101, 103]
    timestamps = [start + timedelta(minutes=5*i) for i in range(len(cgm_history))]
    prob, est_time, carbs = detector.detect_meal_event(cgm_history, timestamps)
    assert prob == 0.0
    assert est_time is None
    assert carbs is None
