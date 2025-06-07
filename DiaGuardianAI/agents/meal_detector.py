# DiaGuardianAI Meal Detector
# Sub-system for detecting meals from CGM data and other available information.

from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import sys # Added sys
import os # Added os
from datetime import timedelta # For timestamp calculations

# Ensure the DiaGuardianAI package is discoverable
if __package__ is None or __package__ == '':
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
 
from sklearn.ensemble import RandomForestClassifier # Import RandomForestClassifier
import joblib # For saving/loading sklearn models


class MealDetector:
    """Detects meal events from CGM data and other available information.

    This class can employ rule-based heuristics or machine learning models
    to identify the occurrence of meals. Its output can be used by
    decision-making agents to inform insulin dosing strategies. The plan
    includes enhancing this to estimate meal size and GI.

    Attributes:
        detection_method (str): The method used for meal detection,
            either "rule_based" or "ml_based".
        params (Dict[str, Any]): Parameters specific to the chosen
            detection method.
        ml_model (Optional[Any]): The machine learning model instance, if
            `detection_method` is "ml_based". (Placeholder)
        rise_threshold (float): For rule-based detection, the minimum
            glucose rise (mg/dL) to consider as a potential meal.
        time_window_minutes (int): For rule-based detection, the time
            window (minutes) over which the `rise_threshold` is checked.
        min_samples_in_window (int): For rule-based detection, the
            minimum number of CGM samples required within the
            `time_window_minutes`.
    """
    def __init__(self, detection_method: str = "rule_based",
                 model_path: Optional[str] = None,
                 params: Optional[Dict[str, Any]] = None):
        """Initializes the MealDetector.

        Args:
            detection_method (str): The method for meal detection.
                Supported: "rule_based", "ml_based". Defaults to
                "rule_based".
            model_path (Optional[str]): Path to a pre-trained machine
                learning model, used if `detection_method` is "ml_based".
                Defaults to None.
            params (Optional[Dict[str, Any]]): Parameters for the chosen
                detection method. For "rule_based", example:
                `{"rise_threshold_mg_dl": 20, "time_window_minutes": 30,
                "min_samples_in_window": 3, "carbs_small_g": 15, 
                "carbs_medium_g": 45, "carbs_large_g": 75}`. 
                For "ml_based", this could contain hyperparameters for model 
                initialization if `model_path` is not provided. Defaults to an empty dict.
        """
        self.detection_method: str = detection_method.lower()
        self.params: Dict[str, Any] = params if params else {}
        self.ml_model: Optional[Any] = None

        # Initialize carb estimation parameters for all instances
        self.carbs_small_g: float = float(self.params.get("carbs_small_g", 15.0))
        self.carbs_medium_g: float = float(self.params.get("carbs_medium_g", 45.0))
        self.carbs_large_g: float = float(self.params.get("carbs_large_g", 75.0))

        if self.detection_method == "ml_based":
            if model_path:
                self.load_model(model_path)
            else:
                model_config = self.params.get("model_config", {}) # e.g., {"n_estimators": 100, "max_depth": 10}
                self.ml_model = RandomForestClassifier(**model_config)
                print(
                    f"MealDetector (ml_based): Initialized new RandomForestClassifier "
                    f"with config: {model_config if model_config else 'default'}."
                )
        elif self.detection_method == "rule_based":
            self.rise_threshold: float = float(
                self.params.get("rise_threshold_mg_dl", 20.0)
            )
            self.time_window_minutes: int = int(
                self.params.get("time_window_minutes", 30)
            )
            self.min_samples_in_window: int = int(
                self.params.get("min_samples_in_window", 3)
            )
            # self.carbs_small_g, medium_g, large_g moved to be initialized for all instances

            if not self.time_window_minutes > 0 or \
               not self.min_samples_in_window > 0:
                raise ValueError(
                    "time_window_minutes and min_samples_in_window must be "
                    "positive for rule-based method."
                )
            print(
                f"MealDetector (rule_based) initialized: Threshold "
                f"{self.rise_threshold} mg/dL over {self.time_window_minutes} "
                f"min (min {self.min_samples_in_window} samples). "
                # Print statement for carb estimates will use the instance attributes
            )
        else:
            raise ValueError(
                f"Unsupported meal detection_method: {self.detection_method}. "
                f"Supported: 'rule_based', 'ml_based'."
            )

    def detect_meal_event(self, cgm_history: List[float],
                          timestamps: Optional[List[Any]] = None,
                          other_features: Optional[Dict[str, Any]] = None
                         ) -> Tuple[float, Optional[Any], Optional[float]]: # Changed return type for size to float (carbs)
        """Analyzes CGM history and other features to detect a meal event.

        Args:
            cgm_history (List[float]): A list of recent CGM readings,
                typically ordered from oldest to newest.
            timestamps (Optional[List[Any]]): A list of timestamps
                corresponding to each CGM reading in `cgm_history`. Used
                for rule-based time window calculations or feature
                engineering. Defaults to None.
            other_features (Optional[Dict[str, Any]]): A dictionary of
                other contextual features that might aid meal detection
                (e.g., time of day, activity level from sensors).
                Defaults to None.

        Returns:
            Tuple[float, Optional[Any], Optional[float]]: A tuple containing:
            - meal_probability (float): An estimated probability of a meal
              occurring, ranging from 0.0 (no meal) to 1.0 (high
              certainty of meal).
            - estimated_start_time (Optional[Any]): The estimated start
              time of the detected meal. The type depends on the
              `timestamps` provided (e.g., datetime object). None if no
              meal is detected or time cannot be estimated.
            - estimated_carbs_g (Optional[float]): An estimation of the
              meal's carbohydrate content in grams. None if no meal is
              detected or carbs cannot be estimated.
        """
        if self.detection_method == "rule_based":
            return self._detect_meal_rule_based(cgm_history, timestamps)
        elif self.detection_method == "ml_based":
            if self.ml_model is None: # Explicitly check for None
                print("MealDetector (ml_based): ML model not initialized. Returning no detection.")
                return 0.0, None, None
            
            # Check if the model is fitted. For RandomForestClassifier, 'estimators_' is a good indicator.
            # However, to be more general for sklearn estimators, we can check for 'is_fitted()',
            # or for attributes that are set after fitting.
            # A common way for sklearn is to check for an attribute like `classes_` or `n_features_in_`.
            # For now, let's assume if it's not None, we try to use it, and errors during predict_proba
            # will indicate it's not trained. The AttributeError catch handles this.
            
            features = self._extract_features_for_ml(cgm_history, timestamps, other_features)
            if features is None or len(features) == 0: # Should be handled by _extract_features_for_ml to return zeros
                 print("MealDetector (ml_based): No features extracted. Returning no detection.")
                 return 0.0, None, None

            try:
                # Assuming the model is trained and has predict_proba
                # Reshape features to (1, n_features) for a single sample
                probabilities = self.ml_model.predict_proba(features.reshape(1, -1))[0]
                meal_probability = float(probabilities[1]) # Assuming class 1 is "meal"
            except AttributeError: # Model might not be trained yet or doesn't have predict_proba
                print("MealDetector (ml_based): Model not trained or lacks predict_proba. Using placeholder probability.")
                meal_probability = np.random.rand() # Placeholder if not trained
            except Exception as e:
                print(f"MealDetector (ml_based): Error during prediction: {e}. Using placeholder probability.")
                meal_probability = np.random.rand() # Placeholder

            detection_threshold = self.params.get("ml_detection_threshold", 0.5)
            if meal_probability > detection_threshold:
                est_time = timestamps[-1] if timestamps and len(timestamps) > 0 else None
                # Placeholder for carb estimation from ML model. For now, use medium.
                # This should be refined: model predicts carbs, or another heuristic.
                estimated_carbs = self.carbs_medium_g 
                return meal_probability, est_time, estimated_carbs
            else:
                return meal_probability, None, None
        
        # Fallback if detection method is somehow not covered (should be caught in __init__)
        return 0.0, None, None # Return None for carbs

    def _detect_meal_rule_based(self, cgm_history: List[float],
                                timestamps: Optional[List[Any]] = None
                               ) -> Tuple[float, Optional[Any], Optional[float]]: # Changed return type for size to float (carbs)
        """Implements simple rule-based meal detection.

        Looks for a significant rise in CGM values over a defined time
        window.

        Args:
            cgm_history (List[float]): Recent CGM readings.
            timestamps (Optional[List[Any]]): Timestamps for `cgm_history`.

        Returns:
            Tuple[float, Optional[Any], Optional[float]]: Meal probability,
                estimated start time, and estimated carbs in grams.
        """
        if not cgm_history or len(cgm_history) < self.min_samples_in_window:
            return 0.0, None, None

        relevant_history_segment = []
        relevant_timestamps_segment = []

        if timestamps and len(timestamps) == len(cgm_history):
            # Use timestamps to define the window
            current_time = timestamps[-1]
            window_start_time = current_time - timedelta(minutes=self.time_window_minutes)
            
            for i in range(len(timestamps) -1, -1, -1):
                if timestamps[i] >= window_start_time:
                    relevant_history_segment.insert(0, cgm_history[i])
                    relevant_timestamps_segment.insert(0, timestamps[i])
                else:
                    break # Window exceeded
            
            if len(relevant_history_segment) < self.min_samples_in_window:
                return 0.0, None, None # Not enough samples in the time window
        else:
            # Fallback to sample count if timestamps are not reliable or not provided
            cgm_sample_rate_minutes = self.params.get("cgm_sample_rate_minutes_for_rule", 5)
            samples_for_window = self.time_window_minutes // cgm_sample_rate_minutes
            samples_for_window = max(samples_for_window, self.min_samples_in_window) # Ensure at least min_samples

            if len(cgm_history) < samples_for_window:
                relevant_history_segment = cgm_history # Use all available if less than desired window but >= min_samples
            else:
                relevant_history_segment = cgm_history[-samples_for_window:]
            # Timestamps are not used for start time estimation in this fallback
            relevant_timestamps_segment = timestamps[-len(relevant_history_segment):] if timestamps and len(timestamps) >= len(relevant_history_segment) else [None] * len(relevant_history_segment)


        if len(relevant_history_segment) < 2:  # Need at least two points to calculate a change
            return 0.0, None, None

        glucose_change = relevant_history_segment[-1] - relevant_history_segment[0]

        # Optional: Calculate rate of change
        # time_diff_minutes = 0
        # if relevant_timestamps_segment[0] and relevant_timestamps_segment[-1] and isinstance(relevant_timestamps_segment[0], type(relevant_timestamps_segment[-1])):
        #     time_diff = relevant_timestamps_segment[-1] - relevant_timestamps_segment[0]
        #     time_diff_minutes = time_diff.total_seconds() / 60.0
        # rate_of_change = glucose_change / time_diff_minutes if time_diff_minutes > 0 else float('inf')
        # Could add a rule: e.g. rate_of_change > X mg/dL/min

        if glucose_change >= self.rise_threshold:
            probability = 0.85  # Assign a relatively high probability
            
            est_start_time = relevant_timestamps_segment[0] if relevant_timestamps_segment and relevant_timestamps_segment[0] is not None else None
            
            # Estimate carbs based on rise magnitude
            estimated_carbs = self.carbs_small_g # Default to small
            if glucose_change > self.rise_threshold * 2.5:  # e.g., >50mg/dL rise if threshold is 20
                estimated_carbs = self.carbs_large_g
            elif glucose_change > self.rise_threshold * 1.5:  # e.g., >30mg/dL rise
                estimated_carbs = self.carbs_medium_g
            return probability, est_start_time, estimated_carbs
        
        return 0.0, None, None  # No meal detected by rule, return None for carbs


    def _extract_features_for_ml(self, cgm_history: List[float],
                                 timestamps: Optional[List[Any]] = None,
                                 other_features: Optional[Dict[str, Any]] = None
                                ) -> np.ndarray:
        """Extracts features from CGM history and other data for the ML model. (Placeholder)

        This method would transform raw input data into a numerical
        feature vector that the machine learning model can process.

        Args:
            cgm_history (List[float]): Recent CGM readings.
            timestamps (Optional[List[Any]]): Timestamps for `cgm_history`.
            other_features (Optional[Dict[str, Any]]): Other contextual
                features.

        Returns:
            np.ndarray: A NumPy array of features.
        """
        # TODO: Implement comprehensive feature engineering for ML-based detection.
        # Examples:
        # - Fourier transform components or wavelet coefficients.
        # - Features from `other_features` (e.g., activity levels).
        
        # Current features: mean, std, min, max, overall_slope (5 features)
        # Will add: hour_sin, hour_cos, day_sin, day_cos (4 features)
        # Total: 9 features if timestamps are available, otherwise 5.
        # The ML model will need to handle potentially variable feature lengths or ensure consistent input.
        # For now, we'll append time features if possible.

        if not cgm_history:
            # If no CGM history, return zeros for all potential features.
            # Current target: mean, std, min, max (4) + overall_slope, roc_last_3, roc_last_2 (3) + time (4) = 11 features
            return np.zeros(11, dtype=float)

        cgm_array = np.array(cgm_history, dtype=float)
        features = []

        # 1. Basic Statistical Features
        features.append(np.mean(cgm_array) if len(cgm_array) > 0 else 0.0)
        features.append(np.std(cgm_array) if len(cgm_array) > 1 else 0.0)
        features.append(np.min(cgm_array) if len(cgm_array) > 0 else 0.0)
        features.append(np.max(cgm_array) if len(cgm_array) > 0 else 0.0)

        # 2. Slope / Rate of Change
        if len(cgm_array) >= 2:
            # Overall slope (start to end of the provided history segment)
            overall_slope = (cgm_array[-1] - cgm_array[0]) / max(1, len(cgm_array)-1) # Slope per interval
            features.append(overall_slope)
            
            # RoC over last 3 samples (approx 15 mins if 5-min samples)
            if len(cgm_array) >= 4: # Need at least 4 points for 3 intervals
                roc_last_3 = (cgm_array[-1] - cgm_array[-4]) / 3.0 # Avg change per interval
                features.append(roc_last_3)
            else:
                features.append(0.0) # Not enough data for this RoC

            # RoC over last 2 samples (approx 10 mins)
            if len(cgm_array) >= 3: # Need at least 3 points for 2 intervals
                roc_last_2 = (cgm_array[-1] - cgm_array[-3]) / 2.0
                features.append(roc_last_2)
            else:
                features.append(0.0)
        else:
            features.append(0.0) # Overall Slope
            features.append(0.0) # RoC last 3
            features.append(0.0) # RoC last 2

        # Add time-based features if timestamps are available
        if timestamps and len(timestamps) > 0:
            last_timestamp = timestamps[-1]
            if hasattr(last_timestamp, 'hour') and hasattr(last_timestamp, 'weekday'): # Check if datetime-like
                hour_of_day = last_timestamp.hour
                day_of_week = last_timestamp.weekday() # Monday=0, Sunday=6

                features.append(np.sin(2 * np.pi * hour_of_day / 24.0))
                features.append(np.cos(2 * np.pi * hour_of_day / 24.0))
                features.append(np.sin(2 * np.pi * day_of_week / 7.0))
                features.append(np.cos(2 * np.pi * day_of_week / 7.0))
            else: # Timestamps not datetime objects or don't have needed attributes
                features.extend([0.0, 0.0, 0.0, 0.0]) # Pad with neutral time features
        else: # No timestamps provided
            features.extend([0.0, 0.0, 0.0, 0.0]) # Pad with neutral time features
        
        # Current feature count: mean, std, min, max (4) + overall_slope, roc_last_3, roc_last_2 (3) + time (4) = 11 features
        # If no cgm_history, it returns np.zeros(9) - this needs to be consistent.
        # Let's ensure a consistent length of 11 features.

        # TODO: Add features from other_features if provided

        # Ensure the feature vector has a consistent length,
        # padding or truncating if necessary, based on what the ML model expects.
        # For now, the number of features is dynamic based on above calculations.
        # An actual ML model would require a fixed feature vector size.
        # The placeholder num_features_expected is removed as we are now generating specific features.
        
        # Example: If a model expects exactly N features, pad/truncate here.
        # expected_ml_features = self.params.get("ml_model_num_features", 5) 
        # current_features_len = len(features)
        # if current_features_len < expected_ml_features:
        #     features.extend([0.0] * (expected_ml_features - current_features_len))
        # elif current_features_len > expected_ml_features:
        #     features = features[:expected_ml_features]
            
        return np.array(features, dtype=float)

    def train_model(self, X_train: np.ndarray, y_train: np.ndarray):
        """Trains the ML-based meal detector. (Placeholder)

        Args:
            X_train (np.ndarray): Training features, where each row is a
                sample and columns are features.
            y_train (np.ndarray): Training labels (e.g., 0 for no meal,
                1 for meal).
        """
        if self.detection_method == "ml_based":
            if self.ml_model is not None: # Simplified check: if model exists, assume it has fit
                # We know it's a RandomForestClassifier from __init__ if not loaded
                try:
                    self.ml_model.fit(X_train, y_train)  # Actual training call
                    print(
                        f"MealDetector (ml_based): RandomForestClassifier training "
                        f"complete with X_train shape {X_train.shape}, "
                        f"y_train shape {y_train.shape}."
                    )
                except Exception as e:
                    print(f"MealDetector (ml_based): Error during model training: {e}")
            else:
                print(
                    "MealDetector (ml_based): ML model not initialized. Cannot train."
                )
        else:
            print(
                "MealDetector: Training is only applicable for 'ml_based' "
                "detection method."
            )

    def save_model(self, path: str):
        """Saves the trained ML model. (Placeholder)

        Args:
            path (str): The file path where the model should be saved.
        """
        if self.detection_method == "ml_based" and self.ml_model:
            try:
                joblib.dump(self.ml_model, path)
                print(f"MealDetector (ml_based): RandomForestClassifier model saved to {path}.")
            except Exception as e:
                print(f"MealDetector (ml_based): Error saving model to {path}: {e}")
        else:
            print(
                "MealDetector: No ML model to save (not ml_based or model "
                "not available)."
            )

    def load_model(self, path: str):
        """Loads a pre-trained ML model. (Placeholder)

        Args:
            path (str): The file path from which to load the model.
        """
        if self.detection_method == "ml_based":
            try:
                self.ml_model = joblib.load(path)
                print(f"MealDetector (ml_based): RandomForestClassifier model loaded successfully from {path}.")
            except FileNotFoundError:
                print(f"MealDetector (ml_based): Error - Model file not found at {path}.")
                self.ml_model = None  # Ensure model is None if loading fails
            except Exception as e:
                print(f"MealDetector (ml_based): Error loading model from {path}: {e}")
                self.ml_model = None
        else:
            print(
                "MealDetector: Model loading is only applicable for 'ml_based' "
                "detection method."
            )


if __name__ == '__main__':
    # Rule-based example
    detector_rule = MealDetector(
        detection_method="rule_based", 
        params={
            "rise_threshold_mg_dl": 15, 
            "time_window_minutes": 20,
            "min_samples_in_window": 3
            # carbs_small_g, etc., will use defaults from __init__
        }
    )
    
    cgm_no_meal = [100.0, 102.0, 101.0, 103.0, 102.0, 104.0]
    cgm_meal_start = [100.0, 105.0, 115.0, 130.0, 140.0, 150.0]  # Rise of 50 in 25 mins (5 samples)
    
    # Timestamps (assuming 5 min intervals)
    from datetime import datetime # Already imported at top
    start_ts = datetime(2023, 1, 1, 12, 0, 0) # This is a Sunday
    timestamps_no_meal = [
        start_ts + timedelta(minutes=i*5) for i in range(len(cgm_no_meal))
    ]
    timestamps_meal_start = [
        start_ts + timedelta(minutes=i*5) for i in range(len(cgm_meal_start))
    ]

    prob_nm, time_nm, carbs_nm = detector_rule.detect_meal_event(
        cgm_no_meal, timestamps_no_meal
    )
    print(f"Rule-based (No Meal): Prob={prob_nm:.2f}, Time={time_nm}, Carbs={carbs_nm}g")

    prob_m, time_m, carbs_m = detector_rule.detect_meal_event(
        cgm_meal_start, timestamps_meal_start
    )
    print(f"Rule-based (Meal): Prob={prob_m:.2f}, Time={time_m}, Carbs={carbs_m}g")

    # ML-based example (placeholders)
    print("\n--- ML Feature Extraction Test ---")
    detector_ml_for_features = MealDetector(
        detection_method="ml_based",
        params={"model_config": {"n_estimators": 10}} # Example config for RF
    ) 
    
    # Test with cgm_meal_start data 
    # timestamps_meal_start[-1] is datetime(2023, 1, 1, 12, 25, 0) -> hour=12, weekday=6 (Sunday)
    features_meal = detector_ml_for_features._extract_features_for_ml(cgm_meal_start, timestamps_meal_start)
    print(f"Features for cgm_meal_start ({len(features_meal)} features): {np.round(features_meal, 4)}")

    # Test with cgm_no_meal data
    features_no_meal = detector_ml_for_features._extract_features_for_ml(cgm_no_meal, timestamps_no_meal)
    print(f"Features for cgm_no_meal ({len(features_no_meal)} features): {np.round(features_no_meal, 4)}")

    # Test with short CGM history
    cgm_short = [100.0, 105.0]
    timestamps_short = [start_ts, start_ts + timedelta(minutes=5)] # Ends at 12:05 Sunday
    features_short = detector_ml_for_features._extract_features_for_ml(cgm_short, timestamps_short)
    print(f"Features for cgm_short ({len(features_short)} features): {np.round(features_short, 4)}")
    
    # Test with empty CGM history
    features_empty = detector_ml_for_features._extract_features_for_ml([], []) 
    print(f"Features for empty CGM ({len(features_empty)} features): {np.round(features_empty, 4)}")
    
    # Test without timestamps
    features_no_ts = detector_ml_for_features._extract_features_for_ml(cgm_meal_start, None)
    print(f"Features for cgm_meal_start (no timestamps) ({len(features_no_ts)} features): {np.round(features_no_ts, 4)}")

    # Test ML-based detection (model is untrained, so predict_proba will be placeholder or error)
    print("\n--- ML Detection Test (Untrained Model) ---")
    prob_ml_detect, time_ml_detect, carbs_ml_detect = detector_ml_for_features.detect_meal_event(
        cgm_meal_start, timestamps_meal_start
    )
    print(f"ML-based Detection (Meal Data): Prob={prob_ml_detect:.2f}, Time={time_ml_detect}, Carbs={carbs_ml_detect}g")

    # Example of training (conceptual, needs actual data)
    # print("\n--- ML Training Example (Conceptual) ---")
    # X_dummy_train = np.random.rand(10, 9) # 10 samples, 9 features
    # y_dummy_train = np.random.randint(0, 2, 10)
    # detector_ml_for_features.train_model(X_dummy_train, y_dummy_train)
    # detector_ml_for_features.save_model("dummy_rf_meal_detector.joblib")
    # detector_ml_for_features.load_model("dummy_rf_meal_detector.joblib")
    
    print("\nMealDetector example run complete.")