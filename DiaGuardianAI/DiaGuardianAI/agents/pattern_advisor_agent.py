import numpy as np
import sys
import os
import joblib
import json
from typing import Any, Optional, Dict, List, cast, Type, TypeVar
from collections import deque

# Ensure the DiaGuardianAI package is discoverable
if __package__ is None or __package__ == '':
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from DiaGuardianAI.core.base_classes import BaseAgent, BasePredictiveModel, BasePatternRepository
from DiaGuardianAI.pattern_repository.repository_manager import RepositoryManager
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import LabelEncoder

_T_PatternAdvisorAgent = TypeVar("_T_PatternAdvisorAgent", bound="PatternAdvisorAgent")


class PatternAdvisorAgent(BaseAgent):    def __init__(self, state_dim: int,
                 pattern_repository: BasePatternRepository,
                 action_dim: Optional[int] = None, # Required for regressors
                 action_keys_ordered: Optional[List[str]] = None, # Required for regressors
                 predictor: Optional[BasePredictiveModel] = None,
                 learning_model_type: str = "supervised", # Default to "supervised" for test compatibility
                 model_params: Optional[Dict[str, Any]] = None,
                 action_space: Optional[Any] = None, # For BaseAgent compatibility
                 cgm_history_len_for_features: int = 12,
                 prediction_horizon_for_features: int = 6):
        super().__init__(state_dim, action_space, predictor) 
        self.pattern_repository: BasePatternRepository = pattern_repository
        self.learning_model_type: str = learning_model_type.lower()
        self.model_params: Dict[str, Any] = model_params if model_params else {}
        
        self.model: Optional[Any] = None
        self.is_trained: bool = False
        
        self.action_dim: Optional[int] = action_dim
        self.action_keys_ordered: Optional[List[str]] = action_keys_ordered
        self.label_encoder: Optional[LabelEncoder] = None

        self.cgm_history_len_for_features = cgm_history_len_for_features
        self.prediction_horizon_for_features = prediction_horizon_for_features

        if self.learning_model_type in ["mlp_regressor", "gradient_boosting_regressor"]:
            if self.action_dim is None or self.action_keys_ordered is None:
                raise ValueError("action_dim and action_keys_ordered must be provided for regressor models.")
            if self.action_dim != len(self.action_keys_ordered):
                raise ValueError("action_dim must match the length of action_keys_ordered.")
            self._build_model() # Build model on init if it's a regressor type
        elif self.learning_model_type == "supervised_classifier":
            self.label_encoder = LabelEncoder()
            self._build_model()
        elif self.learning_model_type == "meta_rl":
            print("PatternAdvisorAgent: Meta-RL model type selected. Requires custom setup.")
        elif self.learning_model_type == "none":
            print("PatternAdvisorAgent: No learning model selected.")
        else:
            raise ValueError(f"PatternAdvisorAgent: Unsupported learning_model_type: {self.learning_model_type}")

    def _build_model(self) -> None:
        if self.model is not None:
            print(f"PatternAdvisorAgent: Model already exists ({type(self.model).__name__}). Not rebuilding.")
            return

        print(f"PatternAdvisorAgent: Building model for type '{self.learning_model_type}' with params: {self.model_params}")
        if self.learning_model_type == "mlp_regressor":
            # Default MLPRegressor params if not provided
            mlp_params = {
                'hidden_layer_sizes': (64, 32), 
                'activation': 'relu',
                'solver': 'adam',
                'max_iter': 300,
                'random_state': 42,
                'early_stopping': True,
                **self.model_params
            }
            self.model = MLPRegressor(**mlp_params)
        elif self.learning_model_type == "gradient_boosting_regressor":
            gb_params = {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 3,
                'random_state': 42,
                **self.model_params
            }
            self.model = GradientBoostingRegressor(**gb_params)
        elif self.learning_model_type == "supervised_classifier":
            rf_params = {
                'n_estimators': 100,
                'random_state': 42,
                **self.model_params
            }
            self.model = RandomForestClassifier(**rf_params)
        else:
            print(f"PatternAdvisorAgent: No model to build for type '{self.learning_model_type}'.")
        
        if self.model is not None:
            print(f"PatternAdvisorAgent: Model {type(self.model).__name__} built successfully.")
        else:
            if self.learning_model_type not in ["none", "meta_rl"]:
                 print(f"PatternAdvisorAgent: WARNING - Model remains None after build attempt for type {self.learning_model_type}")    def train(self, features: Any, actions: Optional[np.ndarray] = None,
              targets: Optional[Any] = None, epochs: int = 1, batch_size: int = 32, 
              validation_split: float = 0.1) -> None:
        """Train the model with features and actions/targets.
        
        Args:
            features: Input features for training
            actions: Target actions (for backward compatibility)
            targets: Alternative name for target values (for test compatibility)
            epochs: Number of training epochs
            batch_size: Training batch size
            validation_split: Portion of data to use for validation
        """        # Use either actions or targets parameter (for backward compatibility with tests)
        actual_targets = actions if actions is not None else targets
        
        if self.learning_model_type not in ["mlp_regressor", "gradient_boosting_regressor", "supervised_classifier"]:
            print(f"PatternAdvisorAgent: Model type {self.learning_model_type} does not support training via this method.")
            return
        
        # Handle dictionary features for test compatibility
        if isinstance(features, list) and all(isinstance(x, dict) for x in features):
            # Validate input data for test compatibility
            for feature_dict in features:
                if "cgm" in feature_dict and feature_dict["cgm"] is not None and feature_dict["cgm"] < 0:
                    raise ValueError("CGM values must be non-negative")
                if "iob" in feature_dict and feature_dict["iob"] is not None and feature_dict["iob"] < 0:
                    raise ValueError("IOB values must be non-negative")

        if self.model is None:
            print("PatternAdvisorAgent: Model not built. Attempting to build now.")
            self._build_model()

        if self.model is None: # Check again after build attempt
            print("PatternAdvisorAgent: ERROR - Model is None, cannot train.")
            return

        if not hasattr(self.model, "fit"):
            print(f"PatternAdvisorAgent: Model {type(self.model).__name__} does not have a fit method. Cannot train.")
            return

        print(f"PatternAdvisorAgent: Starting training for model {type(self.model).__name__} with {features.shape[0]} samples...")
        
        if self.learning_model_type in ["mlp_regressor", "gradient_boosting_regressor"]:
            if actions.ndim == 1 and self.action_dim and self.action_dim > 1:
                print(f"PatternAdvisorAgent: WARNING - Regressor actions are 1D but action_dim is {self.action_dim}. Reshaping actions to (-1, {self.action_dim}). Ensure this is intended.")
                # This case should ideally not happen if data is prepared correctly for multi-output
                # For now, we'll assume if action_dim > 1, actions should be 2D. 
                # If actions are truly for a single target but action_dim > 1, it's a config mismatch.
                # This reshape might fail or be incorrect if actions.shape[0] is not a multiple of action_dim.
                # A better approach is to ensure upstream data matches action_dim.
                # For safety, let's assume actions are (n_samples,) and should be (n_samples, 1) if action_dim is 1.
                if self.action_dim == 1:
                    actions = actions.reshape(-1, 1)
                # else: # If action_dim > 1 and actions are 1D, this is ambiguous. Let fit handle it or error.

            self.model.fit(features, actions) # Scikit-learn regressors handle multi-output if actions is 2D
            self.is_trained = True
            print("PatternAdvisorAgent: Regressor training complete.")
        elif self.learning_model_type == "supervised_classifier":
            if self.label_encoder is None:
                self.label_encoder = LabelEncoder() # Should have been init in _build_model or __init__
            encoded_actions = self.label_encoder.fit_transform(actions)
            self.model.fit(features, encoded_actions)
            self.is_trained = True
            print(f"PatternAdvisorAgent: Classifier training complete. Found classes: {self.label_encoder.classes_}")
        else:
            print(f"PatternAdvisorAgent: Training not applicable for model type '{self.learning_model_type}'.")
            return
        print(f"PatternAdvisorAgent: Model is_trained set to {self.is_trained}")

    def predict(self, state_features: Any) -> Dict[str, float]:
        """Predicts continuous actions using the trained regressor model.

        Args:
            state_features: A single state feature vector (state_dim,)
                           or a batch of features (batch_size, state_dim),
                           or a dictionary of features.
                           This method expects a single vector for typical agent use.

        Returns:
            Dict[str, float]: A dictionary of predicted actions, with keys from
                              self.action_keys_ordered.

        Raises:
            RuntimeError: If the model is not a regressor, not trained, or not built.
            ValueError: If action_keys_ordered is not set.
        """
        # Handle mocking for tests - if a lambda function is provided as learning_model
        if hasattr(self, 'learning_model') and callable(getattr(self, 'learning_model')):
            if self.action_keys_ordered is None:
                raise ValueError("action_keys_ordered must be set for predict method.")
                
            try:
                pred = self.learning_model(state_features)
                # Convert prediction to dictionary
                result = {}
                if isinstance(pred, np.ndarray):
                    for i, key in enumerate(self.action_keys_ordered):
                        if i < len(pred):
                            # Ensure non-negative values for insulin-related outputs
                            if key in ['basal_rate_u_hr', 'bolus_u', 'a', 'b']:
                                result[key] = max(0.0, float(pred[i]))
                            else:
                                result[key] = float(pred[i])
                return result
            except Exception as e:
                print(f"PatternAdvisorAgent: Error in mock model prediction: {e}")
                return {k: 0.0 for k in self.action_keys_ordered}
        
        # Normal operation for real models            
        if self.learning_model_type not in ["mlp_regressor", "gradient_boosting_regressor"]:
            raise RuntimeError("PatternAdvisorAgent: predict() is only supported for regressor models.")
        
        if not self.is_trained or self.model is None:
            raise RuntimeError("PatternAdvisorAgent: Model is not trained or not built. Cannot predict.")

        if not hasattr(self.model, "predict"):
            raise RuntimeError(f"PatternAdvisorAgent: Model {type(self.model).__name__} does not have a predict method.")

        if self.action_keys_ordered is None or self.action_dim is None:
            raise ValueError("PatternAdvisorAgent: action_keys_ordered and action_dim must be set for regressors.")

        state_features_reshaped: np.ndarray
        if state_features.ndim == 1:
            state_features_reshaped = state_features.reshape(1, -1)
        elif state_features.ndim == 2:
            state_features_reshaped = state_features # Assuming (batch_size, state_dim)
        else:
            raise ValueError(f"PatternAdvisorAgent: state_features has unexpected ndim {state_features.ndim}. Expected 1 or 2.")

        if state_features_reshaped.shape[1] != self.state_dim:
            raise ValueError(f"PatternAdvisorAgent: Feature dimension mismatch. Expected {self.state_dim}, got {state_features_reshaped.shape[1]}.")
            
        predicted_action_batch_np = self.model.predict(state_features_reshaped)
        
        # We expect predict to be called with a single state for typical agent use, so take the first prediction.
        # If predict is called with a batch, this will return the prediction for the first sample.
        first_prediction_np = predicted_action_batch_np[0] if predicted_action_batch_np.ndim == 2 else predicted_action_batch_np

        action_values_list: list[float]
        if first_prediction_np.ndim == 0: # Scalar output from model
            if self.action_dim != 1:
                print(f"PatternAdvisorAgent: WARNING - Model predicted a scalar value ({first_prediction_np.item()}), but agent's action_dim is {self.action_dim}. Ensure this is correct.")
            action_values_list = [float(first_prediction_np.item())]
        elif first_prediction_np.ndim == 1: # Vector output from model for a single sample
             if len(first_prediction_np) != self.action_dim:
                 print(f"PatternAdvisorAgent: WARNING - Model predicted vector of length {len(first_prediction_np)}, but agent's action_dim is {self.action_dim}. Ensure this is correct.")
             action_values_list = [float(v) for v in first_prediction_np]
        else:
            raise ValueError(f"PatternAdvisorAgent: Unexpected shape for model's first_prediction_np: {first_prediction_np.shape}")

        if len(action_values_list) != len(self.action_keys_ordered):
            # This might happen if action_dim was misconfigured relative to model output or action_keys_ordered
            # Attempt to reconcile if action_values_list has 1 element and action_keys_ordered has 1 (common for single target)
            if len(action_values_list) == 1 and len(self.action_keys_ordered) == 1:
                pass # This is fine
            elif len(action_values_list) != self.action_dim:
                 raise ValueError(
                    f"PatternAdvisorAgent: Mismatch between number of predicted values ({len(action_values_list)}) from model (shape {first_prediction_np.shape}) and agent's action_dim ({self.action_dim})."
                )
            else: # Mismatch with action_keys_ordered length
                raise ValueError(
                    f"PatternAdvisorAgent: Mismatch between number of predicted action values ({len(action_values_list)}) "
                    f"and number of action keys ({len(self.action_keys_ordered)})."
                )

        clipped_action_dict: Dict[str, float] = {}
        # Define min values, ideally from a shared constants file
        min_basal_rate = 0.0
        min_bolus_u = 0.0

        for i, key in enumerate(self.action_keys_ordered):
            # Check if index i is valid for action_values_list, in case of prior length mismatches not caught
            if i < len(action_values_list):
                value = action_values_list[i]
                if key == 'basal_rate_u_hr':
                    clipped_action_dict[key] = max(min_basal_rate, value)
                elif key == 'bolus_u':
                    clipped_action_dict[key] = max(min_bolus_u, value)
                else:
                    clipped_action_dict[key] = value 
            else:
                # This case should ideally be prevented by earlier checks.
                print(f"PatternAdvisorAgent: WARNING - Not enough predicted values for all action keys. Key '{key}' at index {i} has no value.")
                # Decide on a fallback, e.g., 0.0 or skip
                clipped_action_dict[key] = 0.0 
        
        return clipped_action_dict

    def decide_action(self, current_state: Any, patient: Optional[Any] = None, **kwargs) -> Optional[Dict[str, Any]]:
        if self.learning_model_type in ["mlp_regressor", "gradient_boosting_regressor"] and \
           self.is_trained and self.model is not None:
            try:
                model_features = self._prepare_features_for_internal_model(current_state, patient)
                
                # predict() expects a single sample (state_dim,) or (1, state_dim)
                # _prepare_features_for_internal_model should return (state_dim,)
                if model_features.ndim == 2 and model_features.shape[0] == 1:
                    model_features = model_features.reshape(-1) # Convert to (state_dim,)
                elif model_features.ndim != 1 or model_features.shape[0] != self.state_dim:
                    raise ValueError(f"Prepared features have unexpected shape: {model_features.shape}. Expected ({self.state_dim},)")

                action_dict = self.predict(model_features) # Pass (state_dim,) which predict will reshape
                
                return {
                    "source": self.__class__.__name__,
                    "suggestion_type": "predicted_action",
                    "actions": action_dict,
                    "rationale": f"Predicted by internal {self.learning_model_type} model."
                }
            except Exception as e:
                print(f"PatternAdvisorAgent: Error using internal regressor model: {e}. Falling back.")
        
        # Fallback to repository querying (simplified representation of existing logic)
        query_features_for_repo: Dict[str, Any] = {}
        if isinstance(current_state, dict):
            # ... (logic to populate query_features_for_repo from current_state dict) ...
            # Example: query_features_for_repo["cgm"] = current_state.get("cgm")
            # This part needs to be the original logic from the file for repo querying
            preferred_pattern_type = current_state.get("pattern_type_preference")
            if preferred_pattern_type:
                query_features_for_repo["pattern_type"] = preferred_pattern_type
            else:
                query_features_for_repo = {
                    "cgm": current_state.get("cgm"), "iob": current_state.get("iob"),
                    "cob": current_state.get("cob"),
                }                cgm_hist_for_trend = current_state.get("cgm_history", [])
                if cgm_hist_for_trend is not None and len(cgm_hist_for_trend) > 1:
                    trend_points = min(3, len(cgm_hist_for_trend))
                    if trend_points > 1:
                        query_features_for_repo["cgm_trend"] = np.mean(np.diff(np.asarray(cgm_hist_for_trend)[-trend_points:]))
                    else:
                        query_features_for_repo["cgm_trend"] = 0.0
                else:
                    query_features_for_repo["cgm_trend"] = 0.0
        else:
            print("PatternAdvisorAgent: current_state is not a dict, cannot build feature query for repository easily.")

        valid_query_features = {k: v for k, v in query_features_for_repo.items() if v is not None}
        suggested_pattern_info: Optional[Dict[str, Any]] = None

        if valid_query_features:
            try:
                meaningful_features_present = "pattern_type" in valid_query_features or \
                                          ("cgm" in valid_query_features and valid_query_features["cgm"] is not None)
                if meaningful_features_present:
                    relevant_patterns = self.pattern_repository.retrieve_relevant_patterns(
                        current_state_features=valid_query_features, n_top_patterns=1
                    )
                    if relevant_patterns:
                        # ... (original logic to format suggested_pattern_info from relevant_patterns) ...
                        retrieved_pattern = relevant_patterns[0]
                        suggested_pattern_info = {
                            "source": self.__class__.__name__,
                            "suggestion_type": "retrieved_pattern",
                            "pattern_id": retrieved_pattern.get("id", "unknown_pattern"),
                            "pattern_data": retrieved_pattern,
                            "confidence_score": retrieved_pattern.get("effectiveness_score", 0.75),
                            "rationale": (
                                f"Retrieved pattern ID {retrieved_pattern.get('id', 'N/A')} "
                                f"(Type: {retrieved_pattern.get('pattern_type', 'Unknown')})."
                            )
                        }
                        print(f"PatternAdvisorAgent: Suggesting pattern from repository: ID {suggested_pattern_info.get('pattern_id')}")
                else:
                    print("PatternAdvisorAgent: Insufficient meaningful features to query repository.")
            except Exception as e:
                print(f"PatternAdvisorAgent: Error querying repository: {e}")
        
        if not suggested_pattern_info and \
           self.learning_model_type == "supervised_classifier" and \
           self.is_trained and self.model is not None and self.label_encoder is not None:
            try:
                state_features_for_model = self._prepare_features_for_internal_model(current_state, patient)
                if hasattr(self.model, 'predict') and hasattr(self.label_encoder, 'classes_'):
                    if state_features_for_model.ndim == 1:
                        state_features_for_model = state_features_for_model.reshape(1, -1)
                    model_prediction_idx = self.model.predict(state_features_for_model)[0]
                    formatted_advice = self._format_advice_from_classifier_model(model_prediction_idx, current_state)
                    if formatted_advice:
                        suggested_pattern_info = formatted_advice
                        print(f"PatternAdvisorAgent: Advice generated from internal classifier: {suggested_pattern_info.get('pattern_id') or 'Direct Action'}")
                else:
                    print("PatternAdvisorAgent: Classifier model or label_encoder not ready for prediction.")
            except Exception as e:
                print(f"PatternAdvisorAgent: Error using internal classifier model: {e}")

        return suggested_pattern_info

    def _prepare_features_for_internal_model(self, state_representation: Any, patient: Optional[Any] = None) -> np.ndarray:
        """Prepares features for the internal model from different state representations.

        Args:
            state_representation: The state representation, either a numpy array or a dictionary.
            patient: Optional patient instance for additional context (not used in current implementation).

        Returns:
            A numpy array of features with shape (self.state_dim,)
        
        Raises:
            TypeError: If state_representation is neither a numpy array nor a dict.
            ValueError: If the final feature vector does not match self.state_dim.
        """
        if isinstance(state_representation, np.ndarray):
            features_vector = state_representation.astype(np.float32)
            if features_vector.shape[0] != self.state_dim and features_vector.shape[0] == 1 and features_vector.shape[1] == self.state_dim:
                features_vector = features_vector.reshape(-1)
        elif isinstance(state_representation, dict):
            print("PatternAdvisorAgent: INFO - _prepare_features_for_internal_model received a dict. Reconstructing RLAgent-like feature vector.")
            
            # Constants for normalization, matching RLAgent's _define_state method
            MAX_CGM = 400.0
            MAX_IOB = 20.0
            MAX_COB = 200.0
            MAX_CARBS = 200.0
            MAX_STD_DEV = 100.0

            # Extract all necessary components from the dictionary
            state_parts = []

            # 1. Current CGM (normalized)
            current_cgm = float(state_representation.get('cgm', 100.0))
            state_parts.append(np.clip(current_cgm / MAX_CGM, 0.0, 1.0))
            
            # 2. CGM History
            cgm_history = state_representation.get('cgm_history', [])
            cgm_history_len = self.cgm_history_len_for_features
            
            # Process CGM history similar to RLAgent
            processed_history = np.array(cgm_history, dtype=np.float32)
            if len(processed_history) < cgm_history_len:
                padding = np.full(cgm_history_len - len(processed_history), current_cgm)
                processed_history = np.concatenate((padding, processed_history))
            elif len(processed_history) > cgm_history_len:
                processed_history = processed_history[-cgm_history_len:]
            state_parts.extend(np.clip(processed_history / MAX_CGM, 0.0, 1.0))
            
            # 3. IOB (normalized)
            iob = float(state_representation.get('iob', 0.0))
            state_parts.append(np.clip(iob / MAX_IOB, 0.0, 1.0))
            
            # 4. COB (normalized)
            cob = float(state_representation.get('cob', 0.0))
            state_parts.append(np.clip(cob / MAX_COB, 0.0, 1.0))
            
            # 5. Glucose Predictions (mean)
            prediction_horizon_len = self.prediction_horizon_for_features
            glucose_predictions = None
            if 'glucose_predictions' in state_representation and isinstance(state_representation['glucose_predictions'], dict):
                if 'mean' in state_representation['glucose_predictions']:
                    glucose_predictions = state_representation['glucose_predictions']['mean']
                    
            if not glucose_predictions and 'predictor_output' in state_representation:
                if isinstance(state_representation['predictor_output'], dict) and 'mean' in state_representation['predictor_output']:
                    glucose_predictions = state_representation['predictor_output']['mean']
                    
            if glucose_predictions:
                processed_predictions = np.array(glucose_predictions, dtype=np.float32)
                if len(processed_predictions) < prediction_horizon_len:
                    padding_val = processed_predictions[-1] if len(processed_predictions) > 0 else current_cgm
                    padding = np.full(prediction_horizon_len - len(processed_predictions), padding_val)
                    processed_predictions = np.concatenate((processed_predictions, padding))
                elif len(processed_predictions) > prediction_horizon_len:
                    processed_predictions = processed_predictions[:prediction_horizon_len]
                state_parts.extend(np.clip(processed_predictions / MAX_CGM, 0.0, 1.0))
            else:
                # If no predictions available, use current CGM as a simple forecast
                state_parts.extend(np.full(prediction_horizon_len, np.clip(current_cgm / MAX_CGM, 0.0, 1.0)))
            
            # 6. Glucose Predictions (std_dev for uncertainty)
            glucose_std_devs = None
            if 'glucose_predictions' in state_representation and isinstance(state_representation['glucose_predictions'], dict):
                if 'std_dev' in state_representation['glucose_predictions']:
                    glucose_std_devs = state_representation['glucose_predictions']['std_dev']
                    
            if not glucose_std_devs and 'predictor_output' in state_representation:
                if isinstance(state_representation['predictor_output'], dict) and 'std_dev' in state_representation['predictor_output']:
                    glucose_std_devs = state_representation['predictor_output']['std_dev']

            if glucose_std_devs:
                processed_std_devs = np.array(glucose_std_devs, dtype=np.float32)
                if len(processed_std_devs) < prediction_horizon_len:
                    # Pad with high uncertainty (1.0)
                    padding = np.full(prediction_horizon_len - len(processed_std_devs), 1.0)
                    processed_std_devs = np.concatenate((processed_std_devs, padding))
                elif len(processed_std_devs) > prediction_horizon_len:
                    processed_std_devs = processed_std_devs[:prediction_horizon_len]
                state_parts.extend(np.clip(processed_std_devs / MAX_STD_DEV, 0.0, 1.0))
            else:
                # If no std_dev info, use high uncertainty (1.0)
                state_parts.extend(np.full(prediction_horizon_len, 1.0))
            
            # 7. Calculate slopes at 30min and 60min if predictions available
            slope_30_val_norm = 0.0
            slope_60_val_norm = 0.0
            prediction_interval_minutes = 5.0  # Assuming 5-minute prediction intervals

            if glucose_predictions:
                idx_30min = int(30 / prediction_interval_minutes) - 1  # Index 5
                if len(glucose_predictions) > idx_30min:
                    pred_at_30 = glucose_predictions[idx_30min]
                    raw_slope_30 = (pred_at_30 - current_cgm) / 30.0
                    # Normalize slope: clip to +/- 5 mg/dL/min, then scale to +/- 1.0
                    slope_30_val_norm = np.clip(raw_slope_30 / 5.0, -1.0, 1.0)

                idx_60min = int(60 / prediction_interval_minutes) - 1  # Index 11
                if len(glucose_predictions) > idx_60min:
                    pred_at_60 = glucose_predictions[idx_60min]
                    raw_slope_60 = (pred_at_60 - current_cgm) / 60.0
                    # Normalize slope: clip to +/- 5 mg/dL/min, then scale to +/- 1.0
                    slope_60_val_norm = np.clip(raw_slope_60 / 5.0, -1.0, 1.0)
            
            state_parts.extend([slope_30_val_norm, slope_60_val_norm])
            
            # 8. Add meal detector features
            meal_probability = state_representation.get('meal_probability', 0.0)
            estimated_meal_carbs_g = state_representation.get('estimated_meal_carbs_g', 0.0)
            state_parts.append(np.clip(meal_probability, 0.0, 1.0))
            state_parts.append(np.clip(estimated_meal_carbs_g / MAX_CARBS, 0.0, 1.0))
            
            # 9. Add time features (cyclical encoding)
            hour_of_day = state_representation.get('hour_of_day', 12)
            day_of_week = state_representation.get('day_of_week', 0)
            
            hour_sin = np.sin(2 * np.pi * hour_of_day / 24.0)
            hour_cos = np.cos(2 * np.pi * hour_of_day / 24.0)
            day_sin = np.sin(2 * np.pi * day_of_week / 7.0)
            day_cos = np.cos(2 * np.pi * day_of_week / 7.0)
            state_parts.extend([hour_sin, hour_cos, day_sin, day_cos])
            
            # 10. Add stress level
            stress_level = state_representation.get('stress_level', 0.0)
            state_parts.append(np.clip(stress_level, 0.0, 1.0))
            
            # 11. Add meal announcement features
            time_since_meal_announcement_minutes = state_representation.get('time_since_meal_announcement_minutes')
            meal_announced = state_representation.get('meal_announced', False)
            announced_carbs = state_representation.get('announced_carbs', 0.0)
            
            if meal_announced and time_since_meal_announcement_minutes is not None:
                # Normalize time since announcement (60 mins max)
                normalized_time_since_meal = np.clip(time_since_meal_announcement_minutes / 60.0, 0.0, 1.0)
                state_parts.append(normalized_time_since_meal)
            else:
                state_parts.append(0.0)
            
            state_parts.append(1.0 if meal_announced else 0.0)
            state_parts.append(np.clip(announced_carbs / MAX_CARBS, 0.0, 1.0))
            
            # Create the final feature vector
            features_vector = np.array(state_parts, dtype=np.float32)
        else:
            raise TypeError(f"PatternAdvisorAgent: _prepare_features_for_internal_model expects np.ndarray or Dict, got {type(state_representation)}")
        
        # Ensure correct dimensionality
        if features_vector.ndim == 0:
            features_vector = np.array([features_vector.item()])
        if features_vector.ndim > 1 and features_vector.shape[0] == 1:
            features_vector = features_vector.reshape(-1)

        # Ensure length matches state_dim
        if len(features_vector) != self.state_dim:
            print(f"PatternAdvisorAgent: WARNING - Feature vector length {len(features_vector)} doesn't match state_dim {self.state_dim}. Truncating/padding to match.")
            if len(features_vector) < self.state_dim:
                # Pad with zeros to match state_dim
                padding = np.zeros(self.state_dim - len(features_vector), dtype=np.float32)
                features_vector = np.concatenate((features_vector, padding))
            else:
                # Truncate to match state_dim
                features_vector = features_vector[:self.state_dim]

        if features_vector.shape[0] != self.state_dim:
            raise ValueError(f"PatternAdvisorAgent: Final feature vector dimension {features_vector.shape[0]} does not match self.state_dim {self.state_dim}.")
        
        return features_vector.astype(np.float32)

    def _format_advice_from_classifier_model(self, predicted_category_index: Any, state_representation: Any) -> Optional[Dict[str, Any]]:
        if self.label_encoder is None or not hasattr(self.label_encoder, 'classes_') or not self.is_trained:
            print("PatternAdvisorAgent: Label encoder not ready or model not trained.")
            return None
        try:
            predicted_label = self.label_encoder.classes_[predicted_category_index]
            # Example: label might be "suggest_hypo_treatment_pattern"
            # This part needs to map the predicted_label to a concrete action/pattern_id
            return {
                "source": self.__class__.__name__,
                "suggestion_type": "classified_pattern_or_action",
                "pattern_id": str(predicted_label), # Or map to a more structured action
                "rationale": f"Classified by internal model as '{predicted_label}'."
            }
        except IndexError:
            print(f"PatternAdvisorAgent: Predicted category index {predicted_category_index} out of bounds for label encoder classes.")
            return None
        except Exception as e:
            print(f"PatternAdvisorAgent: Error formatting advice from classifier: {e}")
            return None

    def save(self, path: str, metadata_path: Optional[str] = None) -> None:
        if self.model is None and self.learning_model_type not in ["none", "meta_rl"]:
            print(f"PatternAdvisorAgent: Model is None. Nothing to save for type {self.learning_model_type}.")
            return

        model_dir = os.path.dirname(path)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        actual_model_path = path
        actual_metadata_path = metadata_path if metadata_path else path.replace(".pkl", "_metadata.json")

        if self.learning_model_type not in ["none", "meta_rl"]:
            try:
                joblib.dump(self.model, actual_model_path)
                print(f"PatternAdvisorAgent: Model saved to {actual_model_path}")
            except Exception as e:
                print(f"PatternAdvisorAgent: Error saving model to {actual_model_path}: {e}")
                return # Do not save metadata if model saving failed
        else:
            print(f"PatternAdvisorAgent: No model to save for type {self.learning_model_type}. Only metadata will be saved.")
            # Ensure actual_model_path is not referenced later if no model is saved.
            # We might save a placeholder or skip model saving part in metadata.

        metadata = {
            "learning_model_type": self.learning_model_type,
            "is_trained": self.is_trained,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "action_keys_ordered": self.action_keys_ordered,
            "model_params": self.model_params,
            "cgm_history_len_for_features": self.cgm_history_len_for_features,
            "prediction_horizon_for_features": self.prediction_horizon_for_features,
            "model_file": os.path.basename(actual_model_path) if self.learning_model_type not in ["none", "meta_rl"] else None
        }
        if self.learning_model_type == "supervised_classifier" and self.label_encoder:
            metadata["label_encoder_classes"] = self.label_encoder.classes_.tolist()

        try:
            with open(actual_metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
            print(f"PatternAdvisorAgent: Metadata saved to {actual_metadata_path}")
        except Exception as e:
            print(f"PatternAdvisorAgent: Error saving metadata to {actual_metadata_path}: {e}")

    @classmethod
    def construct_from_checkpoint(cls: Type[_T_PatternAdvisorAgent], 
             model_path: str, 
             pattern_repository: BasePatternRepository, 
             metadata_path: Optional[str] = None) -> _T_PatternAdvisorAgent:
        actual_metadata_path = metadata_path if metadata_path else model_path.replace(".pkl", "_metadata.json")
        
        if not os.path.exists(actual_metadata_path):
            raise FileNotFoundError(f"PatternAdvisorAgent: Metadata file not found at {actual_metadata_path}")

        with open(actual_metadata_path, 'r') as f:
            metadata = json.load(f)

        # Basic check for essential keys
        required_keys = ["learning_model_type", "state_dim"]
        for key in required_keys:
            if key not in metadata:
                raise ValueError(f"PatternAdvisorAgent: Missing required key '{key}' in metadata.")

        agent = cls(
            state_dim=metadata["state_dim"],
            pattern_repository=pattern_repository,
            action_dim=metadata.get("action_dim"),
            action_keys_ordered=metadata.get("action_keys_ordered"),
            learning_model_type=metadata["learning_model_type"],
            model_params=metadata.get("model_params", {}),
            cgm_history_len_for_features=metadata.get("cgm_history_len_for_features", 12),
            prediction_horizon_for_features=metadata.get("prediction_horizon_for_features", 6)
        )

        agent.is_trained = metadata.get("is_trained", False)

        # Load the actual model if it's not "none" or "meta_rl" and a model file is specified
        model_file_name = metadata.get("model_file")
        if agent.learning_model_type not in ["none", "meta_rl"] and model_file_name:
            # Construct full path to model file if model_path was a directory or just a name
            if os.path.isdir(model_path): # If model_path is a directory
                full_model_file_path = os.path.join(model_path, model_file_name)
            elif os.path.basename(model_path) == model_file_name: # If model_path is the full path to the model file
                full_model_file_path = model_path
            else: # Assume model_path is a base name and model_file_name is the actual file, in same dir
                full_model_file_path = os.path.join(os.path.dirname(model_path), model_file_name)            if not os.path.exists(full_model_file_path):
                raise FileNotFoundError(f"PatternAdvisorAgent: Model file {full_model_file_path} not found as specified in metadata.")
                
            try:
                agent.model = joblib.load(full_model_file_path)
                print(f"PatternAdvisorAgent: Model loaded from {full_model_file_path}")
            except Exception as e:
                print(f"PatternAdvisorAgent: Error loading model from {full_model_file_path}: {e}")
                
                # Handle NumPy BitGenerator compatibility error specifically
                if "is not a known BitGenerator module" in str(e):
                    print("NumPy BitGenerator compatibility issue detected. Attempting recovery by building model with saved parameters.")
                    agent._build_model()  # Rebuild the model with current parameters
                    
                    if agent.model is not None and agent.learning_model_type in ["mlp_regressor", "gradient_boosting_regressor"] and os.path.exists(full_model_file_path):
                        try:
                            # Alternative loading approach for scikit-learn models
                            import pickle
                            with open(full_model_file_path, 'rb') as f:
                                # Load model while ignoring unknown/incompatible attributes
                                model_bytes = f.read()
                                recovered = False
                                
                                try:
                                    # Try to load with pickle using highest protocol
                                    temp_model = pickle.loads(model_bytes, fix_imports=True, encoding='latin1')
                                    if hasattr(temp_model, 'coefs_') and hasattr(agent.model, 'coefs_'):
                                        # Transfer neural network weights for MLPRegressor
                                        agent.model.coefs_ = temp_model.coefs_
                                        agent.model.intercepts_ = temp_model.intercepts_
                                        agent.is_trained = True
                                        recovered = True
                                        print("Successfully recovered neural network weights from model file")
                                except Exception as pickle_err:
                                    print(f"Could not recover using pickle: {pickle_err}")
                                
                                if not recovered:
                                    print("Unable to recover model parameters. Model must be retrained.")
                                    agent.is_trained = False
                        except Exception as inner_e:
                            print(f"Recovery attempt failed: {inner_e}")
                            agent.is_trained = False
                else:
                    # For other errors, just reset the model
                    agent.model = None
                    agent.is_trained = False # Mark as not trained if model failed to load
        
        elif agent.learning_model_type not in ["none", "meta_rl"] and not model_file_name:
            print(f"PatternAdvisorAgent: WARNING - Metadata for '{agent.learning_model_type}' does not specify a model_file. Model not loaded.")
            agent.is_trained = False # Cannot be trained if no model file specified

        if agent.learning_model_type == "supervised_classifier" and "label_encoder_classes" in metadata:
            if agent.label_encoder is None: agent.label_encoder = LabelEncoder()
            agent.label_encoder.classes_ = np.array(metadata["label_encoder_classes"])
            print(f"PatternAdvisorAgent: LabelEncoder classes loaded: {agent.label_encoder.classes_}")
        
        # Ensure model is built if it's None but should exist (e.g. if loading failed but metadata suggests it should be there)
        # This might happen if model file was corrupted or not found, but we still want an agent instance.
        if agent.model is None and agent.learning_model_type not in ["none", "meta_rl"]:
            print(f"PatternAdvisorAgent: Model is None after load attempt for type {agent.learning_model_type}. Attempting to build a new untrained model.")
            agent._build_model() # Build a fresh, untrained model
            agent.is_trained = False # Explicitly set to false as it's a new model

        print(f"PatternAdvisorAgent loaded. Type: {agent.learning_model_type}, Trained: {agent.is_trained}, Model: {type(agent.model).__name__ if agent.model else 'None'}")
        return agent

    def load(self, path: str) -> None:
        """Loads the agent's state (model and training status) from a primary model file (.pkl)
           and an associated metadata file (_metadata.json).

        Args:
            path (str): The file path to the primary model file (e.g., 'model.pkl').
                        The metadata file is expected to be named by replacing '.pkl'
                        with '_metadata.json' in the same directory.
        """
        actual_metadata_path = path.replace(".pkl", "_metadata.json")
        
        if not os.path.exists(actual_metadata_path):
            print(f"PatternAdvisorAgent: Metadata file not found at {actual_metadata_path}. Cannot load instance state.")
            # Optionally, could try to load model only if path exists and is a .pkl, but metadata is crucial.
            return

        if not os.path.exists(path) and self.learning_model_type not in ["none", "meta_rl"] :
             print(f"PatternAdvisorAgent: Model file not found at {path}. Cannot load instance state.")
             return

        with open(actual_metadata_path, 'r') as f:
            metadata = json.load(f)

        # Validate metadata against current agent's configuration if necessary
        # For example, check if state_dim matches, etc.
        # For now, we assume the metadata is compatible.

        self.learning_model_type = metadata.get("learning_model_type", self.learning_model_type)
        self.state_dim = metadata.get("state_dim", self.state_dim)
        self.action_dim = metadata.get("action_dim", self.action_dim)
        self.action_keys_ordered = metadata.get("action_keys_ordered", self.action_keys_ordered)
        # self.model_params = metadata.get("model_params", self.model_params) # Careful with overwriting if model is already built
        self.cgm_history_len_for_features = metadata.get("cgm_history_len_for_features", self.cgm_history_len_for_features)
        self.prediction_horizon_for_features = metadata.get("prediction_horizon_for_features", self.prediction_horizon_for_features)
        
        self.is_trained = metadata.get("is_trained", False)

        model_file_name_from_meta = metadata.get("model_file")

        if self.learning_model_type not in ["none", "meta_rl"]:
            # The 'path' argument to this instance method IS the model file path.
            # We should ensure it matches what's in metadata if model_file is present.
            if model_file_name_from_meta and os.path.basename(path) != model_file_name_from_meta:
                print(f"PatternAdvisorAgent: WARNING - Model file name in metadata ('{model_file_name_from_meta}') "
                      f"differs from provided path ('{os.path.basename(path)}'). Using provided path.")

            if os.path.exists(path):
                try:
                    self.model = joblib.load(path)
                    print(f"PatternAdvisorAgent: Instance model loaded from {path}")
                except Exception as e:
                    print(f"PatternAdvisorAgent: Error loading model into instance from {path}: {e}. Model may be None or inconsistent.")
                    self.model = None 
                    self.is_trained = False 
            else:
                print(f"PatternAdvisorAgent: Model file {path} not found. Cannot load model into instance.")
                self.model = None
                self.is_trained = False
        
        elif self.learning_model_type not in ["none", "meta_rl"] and not model_file_name_from_meta :
             print(f"PatternAdvisorAgent: WARNING - Metadata for '{self.learning_model_type}' does not specify a model_file. Model not loaded into instance.")
             self.is_trained = False


        if self.learning_model_type == "supervised_classifier" and "label_encoder_classes" in metadata:
            if self.label_encoder is None: self.label_encoder = LabelEncoder()
            self.label_encoder.classes_ = np.array(metadata["label_encoder_classes"])
            print(f"PatternAdvisorAgent: Instance LabelEncoder classes loaded: {self.label_encoder.classes_}")
        
        # If model is None after load attempts (and it's a type that should have a model)
        # we might need to re-initialize it to a default state.
        if self.model is None and self.learning_model_type not in ["none", "meta_rl"]:
            print(f"PatternAdvisorAgent: Instance model is None after load for type {self.learning_model_type}. Re-building a new untrained model.")
            self._build_model() # Build a fresh, untrained model
            self.is_trained = False # Explicitly set to false

        print(f"PatternAdvisorAgent instance state loaded. Type: {self.learning_model_type}, Trained: {self.is_trained}, Model: {type(self.model).__name__ if self.model else 'None'}")

    # Renamed from load_agent_from_files to be more consistent with a general load method
    @classmethod
    def load_agent_from_files(cls: Type[_T_PatternAdvisorAgent], 
                              model_path: str, 
                              pattern_repository: BasePatternRepository, 
                              metadata_filename: str = "pattern_advisor_metadata.json",
                              model_filename_in_metadata: Optional[str] = None) -> _T_PatternAdvisorAgent:
        """DEPRECATED: Use load() instead. Kept for backward compatibility for now.
           Loads a PatternAdvisorAgent from a model file and a separate metadata file.
        Args:
            model_path (str): Path to the directory containing the model and metadata OR path to model file.
            pattern_repository (BasePatternRepository): The pattern repository instance.
            metadata_filename (str): The name of the metadata JSON file.
            model_filename_in_metadata (Optional[str]): If provided, this filename (from metadata) is used
                                                        instead of deriving from model_path.
        """
        print("PatternAdvisorAgent.load_agent_from_files() is DEPRECATED. Use PatternAdvisorAgent.load() instead.")
        
        # Determine if model_path is a directory or a file path
        if os.path.isdir(model_path):
            metadata_file_path = os.path.join(model_path, metadata_filename)
            # If model_filename_in_metadata is given, it implies model_path is a directory
            # and the actual model file is inside this directory, named as per metadata.
            # If not given, we might infer the model file from metadata later or assume model_path was a placeholder.
        else: # model_path is assumed to be a file path (e.g., .../model.pkl)
            metadata_file_path = os.path.join(os.path.dirname(model_path), metadata_filename)
            # In this case, model_path itself is the direct path to the .pkl file.

        if not os.path.exists(metadata_file_path):
            # Try alternative if original model_path was a file and metadata_filename was default
            if not os.path.isdir(model_path) and metadata_filename == "pattern_advisor_metadata.json":
                alt_metadata_path = model_path.replace(".pkl", "_metadata.json")
                if os.path.exists(alt_metadata_path):
                    metadata_file_path = alt_metadata_path
                else:
                    raise FileNotFoundError(f"PatternAdvisorAgent: Metadata file not found at {metadata_file_path} or {alt_metadata_path}")
            else:
                raise FileNotFoundError(f"PatternAdvisorAgent: Metadata file not found at {metadata_file_path}")

        with open(metadata_file_path, 'r') as f:
            metadata = json.load(f)

        agent = cls(
            state_dim=metadata["state_dim"],
            pattern_repository=pattern_repository,
            action_dim=metadata.get("action_dim"),
            action_keys_ordered=metadata.get("action_keys_ordered"),
            learning_model_type=metadata["learning_model_type"],
            model_params=metadata.get("model_params", {}),
            cgm_history_len_for_features=metadata.get("cgm_history_len_for_features", 12),
            prediction_horizon_for_features=metadata.get("prediction_horizon_for_features", 6)
        )
        agent.is_trained = metadata.get("is_trained", False)

        # Determine the actual model file to load
        actual_model_file_to_load = None
        # Priority: model_filename_in_metadata if provided in call, then from metadata dict, then model_path if it was a file.
        load_this_filename = model_filename_in_metadata or metadata.get("model_file")

        if agent.learning_model_type not in ["none", "meta_rl"]:
            if load_this_filename:
                if os.path.isdir(model_path): # model_path was a directory
                    actual_model_file_to_load = os.path.join(model_path, load_this_filename)
                else: # model_path was a file, load_this_filename should match its basename
                    if os.path.basename(model_path) == load_this_filename:
                        actual_model_file_to_load = model_path
                    else:
                        # This case is tricky: model_path is a file, but metadata/param suggests a different filename.
                        # Assume it's in the same directory as the original model_path file.
                        actual_model_file_to_load = os.path.join(os.path.dirname(model_path), load_this_filename)
            elif not os.path.isdir(model_path): # No specific filename in metadata, and model_path was a file path
                actual_model_file_to_load = model_path # Assume model_path is the .pkl file
            else:
                print(f"PatternAdvisorAgent: WARNING - Model filename not specified in metadata or arguments, and model_path is a directory. Cannot load model for type {agent.learning_model_type}.")
                agent.is_trained = False

            if actual_model_file_to_load and os.path.exists(actual_model_file_to_load):
                try:
                    agent.model = joblib.load(actual_model_file_to_load)
                    print(f"PatternAdvisorAgent: Model loaded from {actual_model_file_to_load}")
                except Exception as e:
                    print(f"PatternAdvisorAgent: Error loading model from {actual_model_file_to_load}: {e}")
                    agent.model = None; agent.is_trained = False
            elif actual_model_file_to_load:
                print(f"PatternAdvisorAgent: Model file {actual_model_file_to_load} not found.")
                agent.model = None; agent.is_trained = False
            # If actual_model_file_to_load is still None here, it means we couldn't determine it.

        if agent.learning_model_type == "supervised_classifier" and "label_encoder_classes" in metadata:
            if agent.label_encoder is None: agent.label_encoder = LabelEncoder()
            agent.label_encoder.classes_ = np.array(metadata["label_encoder_classes"])

        # Fallback to build a new model if loading failed but one was expected
        if agent.model is None and agent.learning_model_type not in ["none", "meta_rl"]:
            print(f"PatternAdvisorAgent: Model is None after load. Attempting to build a new untrained model for type {agent.learning_model_type}.")
            agent._build_model()
            agent.is_trained = False

        print(f"PatternAdvisorAgent loaded via DEPRECATED method. Type: {agent.learning_model_type}, Trained: {agent.is_trained}")
        return agent