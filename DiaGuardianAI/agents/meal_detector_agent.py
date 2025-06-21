# DiaGuardianAI Meal Detector Agent
# Analyzes CGM data and other inputs to detect unannounced meal events.

import sys
import os
from typing import Optional, List, Dict, Any

# Ensure the DiaGuardianAI package is discoverable
if __package__ is None or __package__ == "":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from DiaGuardianAI.core.base_classes import BaseAgent, BasePredictiveModel
from DiaGuardianAI.agents.meal_detector import (
    MealDetector,
)  # Import the MealDetector class


class MealDetectorAgent(BaseAgent):
    """
    Detects meal events based on CGM data patterns and other contextual signals.

    This agent could use various techniques, from simple rule-based heuristics
    (e.g., rapid rise in CGM) to more complex machine learning models trained
    to identify meal-induced glucose excursions.
    """

    def __init__(
        self,
        state_dim: int,  # May not be directly used if not making 'actions' in RL sense
        action_space: Any,  # May be minimal or conceptual for this agent
        predictor: Optional[BasePredictiveModel] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Initializes the MealDetectorAgent.

        Args:
            state_dim (int): Dimensionality of any state representation this agent might use.
            action_space (Any): Defines any action space (might be conceptual, e.g., 'signal_meal').
            predictor (Optional[BasePredictiveModel]): A predictive model, if useful for detection.
            config (Optional[Dict[str, Any]]): Configuration parameters for the meal detector
                                               (e.g., thresholds, model paths).
            **kwargs: Additional keyword arguments.
        """
        super().__init__(state_dim, action_space, predictor, **kwargs)
        self.config = config if config else {}

        # Instantiate the underlying MealDetector
        # Pass relevant parts of agent's config to MealDetector's params
        meal_detector_params = self.config.get("meal_detector_params", {})
        meal_detector_method = self.config.get(
            "meal_detection_method", "rule_based"
        )  # e.g. "rule_based" or "ml_based"
        meal_detector_model_path = self.config.get("meal_detector_model_path", None)

        self.meal_detector_instance = MealDetector(
            detection_method=meal_detector_method,
            model_path=meal_detector_model_path,
            params=meal_detector_params,
        )
        print(
            f"MealDetectorAgent initialized, wrapping MealDetector with method: {meal_detector_method}."
        )

    def decide_action(self, current_state: Any, **kwargs) -> Optional[Dict[str, Any]]:
        """
        This method is part of BaseAgent. For MealDetector, it's repurposed to
        trigger meal detection based on current_state (which should include CGM history).
        It doesn't 'decide an action' in the RL sense but rather 'performs detection'.
        """
        cgm_history = current_state.get("cgm_history")  # List[float]
        timestamps = current_state.get(
            "timestamps"
        )  # Optional[List[datetime]] or similar
        # other_features could be other parts of current_state if needed by MealDetector
        # For now, we'll pass a subset or None if not explicitly defined for MealDetector

        if not cgm_history:
            print("MealDetectorAgent: CGM history not available in current_state.")
            return None  # Or a dict indicating no detection

        # Call the underlying MealDetector's detect_meal_event method
        prob, time, carbs_g = self.meal_detector_instance.detect_meal_event(
            cgm_history=cgm_history,
            timestamps=timestamps,
            other_features=None,  # Pass other relevant context if MealDetector is designed for it
        )

        if prob > 0:  # Assuming prob > 0 means some level of meal likelihood
            return {
                "meal_probability": prob,
                "estimated_start_time": time,  # This might be None
                "estimated_meal_carbs_g": carbs_g,  # This might be None
            }
        return None  # No meal detected or low probability

    # The detect_meal_event method is now removed from MealDetectorAgent,
    # as the core logic resides in the MealDetector class.
    # The decide_action method of MealDetectorAgent will call
    # self.meal_detector_instance.detect_meal_event()

    def learn(self, experience: Any):
        """
        (Optional) Trains the meal detection model if it's adaptive.
        This might involve supervised learning on labeled meal/no-meal data.
        """
        # Delegate training to the underlying MealDetector if applicable
        if hasattr(self.meal_detector_instance, "train_model"):
            try:
                X_train = experience.get("features")
                y_train = experience.get("labels")
                if X_train is not None and y_train is not None:
                    self.meal_detector_instance.train_model(X_train, y_train)
                else:
                    print("MealDetectorAgent.learn: Incomplete experience provided.")
            except Exception as e:
                print(f"MealDetectorAgent.learn: Error during training - {e}")
        else:
            print(
                "MealDetectorAgent.learn: Underlying detector has no train_model method."
            )

    def save(self, path: str):
        """(Optional) Saves the meal detection model."""
        if hasattr(self.meal_detector_instance, "save_model"):
            try:
                self.meal_detector_instance.save_model(path)
            except Exception as e:
                print(f"MealDetectorAgent.save: Error saving model - {e}")
        else:
            print(
                "MealDetectorAgent.save: Underlying detector has no save_model method."
            )

    def load(self, path: str):
        """(Optional) Loads a pre-trained meal detection model."""
        if hasattr(self.meal_detector_instance, "load_model"):
            try:
                self.meal_detector_instance.load_model(path)
            except Exception as e:
                print(f"MealDetectorAgent.load: Error loading model - {e}")
        else:
            print(
                "MealDetectorAgent.load: Underlying detector has no load_model method."
            )


if __name__ == "__main__":
    print("--- MealDetectorAgent Example ---")

    # Example configuration for MealDetectorAgent, which will pass params to MealDetector
    config = {
        "meal_detection_method": "rule_based",  # or "ml_based"
        "meal_detector_params": {
            "rise_threshold_mg_dl": 15,
            "time_window_minutes": 20,
            "min_samples_in_window": 3,
            "carbs_small_g": 20,  # Override default
            "carbs_medium_g": 50,
            "carbs_large_g": 80,
            # For ml_based, could add "model_config": {"n_estimators": 100}
        },
        # meal_detector_model_path could be added here if using a pre-trained ML model
    }

    # Dummy action space and state_dim for BaseAgent compatibility
    dummy_action_space = None
    dummy_state_dim = 1

    meal_detector = MealDetectorAgent(
        state_dim=dummy_state_dim, action_space=dummy_action_space, config=config
    )

    # Test case 1: No meal
    cgm_history_no_meal = [
        100.0,
        102.0,
        101.0,
        103.0,
        102.0,
        104.0,
        105.0,
    ]  # 7 points (35 min)
    print(f"\nTesting with no meal CGM: {cgm_history_no_meal}")
    # The decide_action method is used to trigger detection
    detection_result_1 = meal_detector.decide_action(
        {"cgm_history": cgm_history_no_meal}
    )
    if detection_result_1:
        print(f"  Result: Meal DETECTED (Unexpected): {detection_result_1}")
    else:
        print(f"  Result: No meal detected (Expected).")

    # Test case 2: Clear meal signature
    # Window is 30 min (6 points). Rise threshold 25.
    # Start of window: 100. End of window: 130. Abs Rise = 30. (Meets 25)
    # RoC over last 3 points (125, 130, 132): (132-125)/2 = 7/2 = 3.5 mg/dL/interval. (Meets 3.0)
    cgm_history_with_meal = [98.0, 100.0, 105.0, 115.0, 125.0, 130.0, 132.0]  # Length 7
    # Rise window (last 6 points): [100, 105, 115, 125, 130, 132]. Rise = 132-100 = 32.
    # RoC window (last 3 points): [125, 130, 132]. RoC = (132-125)/2 = 3.5
    print(f"\nTesting with meal CGM: {cgm_history_with_meal}")
    # Prepare a dummy current_state that includes timestamps
    from datetime import datetime, timedelta

    start_ts = datetime(2023, 1, 1, 12, 0, 0)
    timestamps_meal = [
        start_ts + timedelta(minutes=i * 5) for i in range(len(cgm_history_with_meal))
    ]
    current_state_meal = {
        "cgm_history": cgm_history_with_meal,
        "timestamps": timestamps_meal,
    }

    detection_result_2 = meal_detector.decide_action(current_state_meal)
    if detection_result_2:
        print(f"  Result: Meal DETECTED (Expected): {detection_result_2}")
    else:
        print(f"  Result: No meal detected (Unexpected).")

    # Test case 3: Insufficient rise
    cgm_history_small_rise = [
        100.0,
        102.0,
        105.0,
        108.0,
        110.0,
        112.0,
        115.0,
    ]  # Rise of 15 in last 6 points
    print(f"\nTesting with small rise CGM: {cgm_history_small_rise}")
    detection_result_3 = meal_detector.decide_action(
        {"cgm_history": cgm_history_small_rise}
    )
    if detection_result_3:
        print(f"  Result: Meal DETECTED (Unexpected): {detection_result_3}")
    else:
        print(f"  Result: No meal detected (Expected).")

    # Test case 4: Insufficient data
    cgm_history_short = [100.0, 110.0, 120.0]  # Only 3 points
    print(f"\nTesting with insufficient data: {cgm_history_short}")
    detection_result_4 = meal_detector.decide_action({"cgm_history": cgm_history_short})
    if detection_result_4:
        print(f"  Result: Meal DETECTED (Unexpected): {detection_result_4}")
    else:
        print(f"  Result: No meal detected (Expected due to insufficient data).")

    # Test case 5: Meets absolute rise, fails RoC
    # Abs Rise window (last 6): [100, 102, 105, 110, 120, 128]. Rise = 128-100 = 28. (Meets 25)
    # RoC window (last 3): [110, 120, 128]. RoC = (128-110)/2 = 18/2 = 9.0 (Meets 3.0) -> This example actually meets both.
    # Let's make RoC fail: Need slower rise at the end.
    # Abs Rise window: [100, 110, 120, 125, 126, 127]. Rise = 127-100 = 27. (Meets 25)
    # RoC window: [125, 126, 127]. RoC = (127-125)/2 = 1.0. (Fails 3.0)
    cgm_history_abs_rise_fails_roc = [90.0, 100.0, 110.0, 120.0, 125.0, 126.0, 127.0]
    print(f"\nTesting with Abs Rise Met, RoC Fail: {cgm_history_abs_rise_fails_roc}")
    detection_result_5 = meal_detector.decide_action(
        {"cgm_history": cgm_history_abs_rise_fails_roc}
    )
    if detection_result_5:
        print(f"  Result: Meal DETECTED (Unexpected): {detection_result_5}")
    else:
        print(f"  Result: No meal detected (Expected).")

    # Test case 6: Fails absolute rise, meets RoC
    # Abs Rise window (last 6): [100, 102, 103, 108, 113, 118]. Rise = 118-100 = 18. (Fails 25)
    # RoC window (last 3): [108, 113, 118]. RoC = (118-108)/2 = 5.0. (Meets 3.0)
    cgm_history_roc_met_fails_abs = [90.0, 100.0, 102.0, 103.0, 108.0, 113.0, 118.0]
    print(f"\nTesting with RoC Met, Abs Rise Fail: {cgm_history_roc_met_fails_abs}")
    detection_result_6 = meal_detector.decide_action(
        {"cgm_history": cgm_history_roc_met_fails_abs}
    )
    if detection_result_6:
        print(f"  Result: Meal DETECTED (Unexpected): {detection_result_6}")
    else:
        print(f"  Result: No meal detected (Expected).")

    # Test case 7: Insufficient data for RoC (but enough for abs rise window)
    # roc_detection_points = 3. Need at least 3 points.
    # cgm_points_in_window = 6.
    # This case should be caught by the initial length check in detect_meal_event if len < roc_detection_points
    # If len is e.g. 4, it's < cgm_points_in_window (6) but > roc_detection_points (3)
    # The initial check is: len(cgm_history) < self.cgm_points_in_window OR len(cgm_history) < self.roc_detection_points
    # So if roc_detection_points is smaller, it's len < cgm_points_in_window that matters first.
    # If cgm_points_in_window is smaller, it's len < roc_detection_points that matters first.
    # Let's test where len(cgm_history) is between roc_detection_points and cgm_points_in_window
    # Example: roc_detection_points = 3, cgm_points_in_window = 6. History length = 4.
    # len(cgm_history) < self.cgm_points_in_window (4 < 6) is true. So it returns None. Correct.

    # Test with just enough for RoC but not for window (if roc_detection_points > cgm_points_in_window, which is unlikely here)
    # Let's assume roc_detection_points = 3, cgm_points_in_window = 2 (by changing config for this test)
    short_window_config = {
        "min_cgm_rise_threshold_mg_dl": 10,
        "detection_window_minutes": 5,  # cgm_points_in_window = 1, but min is 2 for rise. Let's make it 10 min -> 2 pts
        "min_roc_mg_dl_per_interval": 3.0,
        "roc_detection_points": 3,
    }
    # meal_detector_short_win = MealDetectorAgent(dummy_state_dim, dummy_action_space, config=short_window_config)
    # cgm_history_short_win_test = [100.0, 110.0, 120.0] # len=3. roc_pts=3 (ok). win_pts=2 (ok for slice).
    # Rise over 2 pts: 120-110=10. (Meets 10).
    # RoC over 3 pts: (120-100)/2 = 10. (Meets 3.0).
    # print(f"\nTesting with short window config: {cgm_history_short_win_test}")
    # detection_result_7 = meal_detector_short_win.decide_action({"cgm_history": cgm_history_short_win_test})
    # if detection_result_7:
    #     print(f"  Result: Meal DETECTED (Expected): {detection_result_7}")
    # else:
    #     print(f"  Result: No meal detected (Unexpected).")
    # This test is a bit complex due to interaction of two window sizes. The initial combined length check should handle it.

    print("\nMealDetectorAgent example run complete.")
