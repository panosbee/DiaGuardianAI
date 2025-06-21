# DiaGuardianAI Decision Agent
# The primary RL agent for managing insulin delivery and recognizing meal patterns.

import sys # Added sys
import os # Added os

# Ensure the DiaGuardianAI package is discoverable
if __package__ is None or __package__ == '':
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')) # Adjusted path for agents directory
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from DiaGuardianAI.core.base_classes import BaseAgent, BasePredictiveModel # PatternAdvisorAgent is a BaseAgent
# Ensure PatternAdvisorAgent is importable if its specific type is needed, or use BaseAgent
# from DiaGuardianAI.agents.pattern_advisor_agent import PatternAdvisorAgent
from DiaGuardianAI.predictive_models.model_zoo.lstm_predictor import LSTMPredictor # Added import
from typing import Any, Optional, Dict, List, Tuple, cast
import numpy as np
import pandas as pd # Added import
import gymnasium as gym
from gymnasium.spaces import Box, Dict as GymDict, Space
from stable_baselines3 import PPO, SAC # Import SAC
from stable_baselines3.common.vec_env import DummyVecEnv # For creating a dummy env if needed by PPO/SAC

# Imports for __main__ testing will be moved into the __main__ block
# from DiaGuardianAI.agents.pattern_advisor_agent import PatternAdvisorAgent # Moved
from DiaGuardianAI.pattern_repository.repository_manager import RepositoryManager
from DiaGuardianAI.learning import MetaLearner, FederatedClient, SimpleOODDetector


# Minimal Gym Environment for SB3 compatibility
class _MinimalGymEnv(gym.Env):
    """
    A minimal Gymnasium environment that is initialized with pre-defined
    observation and action spaces. Its methods are placeholders as it's
    primarily used to satisfy SB3 model constructor requirements when
    the actual environment interaction is handled externally.
    """
    metadata = {'render_modes': ['human'], 'render_fps': 30}
    # Match base class type hints for attributes
    observation_space: Space[np.ndarray]
    action_space: Space[np.ndarray] # Changed from Space[Dict[str, Any]] as Box yields np.ndarray

    def __init__(self, observation_space: gym.spaces.Box, action_space: gym.spaces.Box): # Changed action_space type to Box
        super().__init__()
        self.observation_space = observation_space # Assigning Box to Space[np.ndarray]
        self.action_space = action_space       # Assigning Box to Space[np.ndarray]
        
        # Use cast for Pylance when accessing specific attributes for np.zeros
        current_obs_box_space = cast(gym.spaces.Box, self.observation_space)
        obs_shape: Tuple[int, ...] = current_obs_box_space.shape
        obs_dtype: np.dtype = current_obs_box_space.dtype
        self.current_obs: np.ndarray = np.zeros(obs_shape, dtype=obs_dtype)

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # Placeholder: actual step logic is handled by SimulationEngine
        # Return current observation, zero reward, done=False, truncated=False, empty info
        return self.current_obs, 0.0, False, False, {}

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed, options=options)
        # Placeholder: actual reset logic is handled by SimulationEngine/training loop
        current_obs_box_space = cast(gym.spaces.Box, self.observation_space)
        obs_shape: Tuple[int, ...] = current_obs_box_space.shape
        obs_dtype: np.dtype = current_obs_box_space.dtype
        self.current_obs = np.zeros(obs_shape, dtype=obs_dtype)
        return self.current_obs, {}

    def render(self):
        """Render the current observation to the console."""
        print(f"Current observation: {self.current_obs}")

    def close(self):
        """Perform minimal environment cleanup."""
        # Nothing persistent to clean up, but method provided for completeness
        print("_MinimalGymEnv closed.")


class RLAgent(BaseAgent):
    """Reinforcement Learning agent for diabetes management.
    (Docstring remains the same)
    """
    # Define expected structure for action_space if it were a dict (for non-gymnasium use)
    # Example:
    # ACTION_SPACE_STRUCTURE = {
    #     "bolus_u": {"low": 0.0, "high": 20.0, "type": "continuous"},
    #     "basal_rate_u_hr": {"low": 0.0, "high": 5.0, "type": "continuous"}
    # }
    # Or for discrete adjustments:
    # "basal_adjustment": {"n": 3} # 0: decrease, 1: none, 2: increase

    def __init__(self, state_dim: int, action_space_definition: Any,
                 predictor: Optional[BasePredictiveModel] = None,
                 meal_detector: Optional[BaseAgent] = None,
                 pattern_advisor: Optional[BaseAgent] = None, # New parameter for PatternAdvisorAgent
                 cgm_history_len: int = 24, # Changed default to 24 for LSTMPredictor
                 prediction_horizon_len: int = 6,
                 rl_algorithm: str = "PPO",
                 rl_params: Optional[Dict[str, Any]] = None,
                 learning_rate: float = 3e-4):
       """Initializes the RL Decision Agent."""
       super().__init__(state_dim, action_space_definition, predictor)
       self.meal_detector = meal_detector
       self.pattern_advisor = pattern_advisor # Store the pattern advisor
       self.cgm_history_len: int = cgm_history_len
       self.prediction_horizon_len: int = prediction_horizon_len
       self.rl_algorithm_name: str = rl_algorithm
       self.rl_params: Dict[str, Any] = rl_params if rl_params else {}
       self.learning_rate: float = learning_rate
       self.rl_model: Optional[Any] = None

       self.observation_space_gym: Optional[gym.spaces.Box] = None
       self.action_space_gym: Optional[gym.spaces.Box] = None # Changed from GymDict to Box
       self.action_keys_ordered: List[str] = [] # To store the order of action components

       self._setup_gym_spaces()
       self._initialize_rl_model()

       # Initialize advanced learning helpers
       self.meta_learner = MetaLearner(self.rl_model, algorithm="maml")
       self.federated_client = FederatedClient(self.rl_model, client_id="rl_agent", buffer_capacity=1000)
       self.ood_detector = SimpleOODDetector()
       self.latest_obs: Optional[np.ndarray] = None

       print(
           f"RLAgent initialized with algorithm: {self.rl_algorithm_name}, "
           f"state_dim: {self.state_dim}, action_space: {self.action_space}"
           # self.action_space is from BaseAgent, which stores action_space_definition
       )

    def _setup_gym_spaces(self):
        """Sets up Gymnasium observation and action spaces."""
        # Observation Space
        self.observation_space_gym = Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32)

        # Action Space - convert from action_space_definition to a single Box
        if not isinstance(self.action_space, dict): # self.action_space is the original definition
            raise ValueError("action_space_definition in RLAgent must be a dictionary.")

        low_bounds = []
        high_bounds = []
        self.action_keys_ordered = [] # Ensure it's reset if called multiple times

        # Sort keys to ensure consistent order if action_space dict is not ordered (Python < 3.7)
        # For Python 3.7+, dicts are insertion ordered, but sorting is safer for broader compatibility.
        sorted_action_names = sorted(self.action_space.keys())

        for action_name in sorted_action_names:
            props = self.action_space[action_name]
            if not isinstance(props, dict) or "low" not in props or "high" not in props:
                raise ValueError(
                    f"Each action in action_space_definition must be a dict with 'low' and 'high' keys. "
                    f"Error with action: {action_name}"
                )
            low_bounds.append(props["low"])
            high_bounds.append(props["high"])
            self.action_keys_ordered.append(action_name)
        
        if not low_bounds: # Should not happen if action_space is not empty
             raise ValueError("Action space definition resulted in empty bounds.")

        self.action_space_gym = Box(
            low=np.array(low_bounds, dtype=np.float32),
            high=np.array(high_bounds, dtype=np.float32),
            dtype=np.float32
        )

        print(f"RLAgent: Observation space: {self.observation_space_gym}")
        print(f"RLAgent: Action space (Box): {self.action_space_gym}, Ordered Keys: {self.action_keys_ordered}")


    def _initialize_rl_model(self):
        """Helper to instantiate the RL model."""
        if self.observation_space_gym is None or self.action_space_gym is None:
            print("Error: Observation or action space not defined. Cannot initialize RL model.")
            return

        # Create a dummy environment instance (needed for both PPO and SAC)
        dummy_env_instance = _MinimalGymEnv(
            observation_space=self.observation_space_gym,
            action_space=self.action_space_gym
        )
        dummy_vec_env = DummyVecEnv([lambda: dummy_env_instance])
        policy_str = self.rl_params.get("policy", "MlpPolicy") # Common for PPO and SAC MlpPolicy
        policy_kwargs_for_model = self.rl_params.get("policy_kwargs", {}).copy()


        if self.rl_algorithm_name.upper() == "PPO":
            ppo_constructor_params = {
                "learning_rate": self.learning_rate,
                "n_steps": 2048,
                "batch_size": 64,
                "n_epochs": 10,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
                "ent_coef": 0.0,
                "vf_coef": 0.5,
                "max_grad_norm": 0.5,
                "verbose": self.rl_params.get("verbose", 0),
                "tensorboard_log": self.rl_params.get("tensorboard_log", None)
            }
            # Override defaults with any user-provided rl_params for PPO constructor
            for key, value in self.rl_params.items():
                if key not in ["policy", "policy_kwargs", "learning_rate"]: # learning_rate is handled
                    ppo_constructor_params[key] = value
            
            self.rl_model = PPO(
                policy=policy_str,
                env=dummy_vec_env,
                policy_kwargs=policy_kwargs_for_model if policy_kwargs_for_model else None,
                **ppo_constructor_params
            )
            print(f"RLAgent: PPO model initialized with policy '{policy_str}', learning rate {ppo_constructor_params['learning_rate']}.")

        elif self.rl_algorithm_name.upper() == "SAC":
            sac_constructor_params = {
                "learning_rate": self.learning_rate,
                "buffer_size": self.rl_params.get("buffer_size", 1_000_000), # Default SAC buffer_size
                "batch_size": self.rl_params.get("batch_size", 256),       # Default SAC batch_size
                "gamma": self.rl_params.get("gamma", 0.99),
                "tau": self.rl_params.get("tau", 0.005),
                "train_freq": self.rl_params.get("train_freq", 1), # Tuple (1, "step") or int
                "gradient_steps": self.rl_params.get("gradient_steps", 1),
                "learning_starts": self.rl_params.get("learning_starts", 100),
                "ent_coef": self.rl_params.get("ent_coef", 'auto'), # SAC specific
                "verbose": self.rl_params.get("verbose", 0),
                "tensorboard_log": self.rl_params.get("tensorboard_log", None)
                # use_sde, sde_sample_freq, replay_buffer_class, replay_buffer_kwargs etc. can be added
            }
            # Override defaults with any user-provided rl_params for SAC constructor
            for key, value in self.rl_params.items():
                 if key not in ["policy", "policy_kwargs", "learning_rate"]: # learning_rate is handled
                    sac_constructor_params[key] = value

            self.rl_model = SAC(
                policy=policy_str, # SAC also uses MlpPolicy by default for continuous
                env=dummy_vec_env,
                policy_kwargs=policy_kwargs_for_model if policy_kwargs_for_model else None,
                **sac_constructor_params
            )
            print(f"RLAgent: SAC model initialized with policy '{policy_str}', learning rate {sac_constructor_params['learning_rate']}.")
        else:
            print(f"RLAgent: RL algorithm '{self.rl_algorithm_name}' not supported or placeholder.")
            self.rl_model = None

    def _define_state(self,
                      current_cgm: float,
                      cgm_history: List[float],
                      iob: float,
                      cob: float,
                      glucose_predictions: Optional[List[float]],
                      glucose_predictions_std_dev: Optional[List[float]] = None, # Added std_dev
                      meal_announced: bool = False,
                      announced_carbs: float = 0.0,
                      hour_of_day: int = 12, # Default to noon
                      day_of_week: int = 0,  # Default to Monday
                      stress_level: float = 0.0, # Default to no stress
                      time_since_meal_announcement_minutes: Optional[float] = None, # New feature
                      meal_probability: float = 0.0, # From MealDetectorAgent
                      estimated_meal_carbs_g: float = 0.0 # From MealDetectorAgent
                      ) -> np.ndarray:
        """Constructs the numerical state vector from various inputs."""
        state_parts = []
        state_parts.append(np.clip(current_cgm / 400.0, 0.0, 1.0))
        
        processed_history = np.array(cgm_history, dtype=np.float32)
        if len(processed_history) < self.cgm_history_len:
            padding = np.full(self.cgm_history_len - len(processed_history), current_cgm)
            processed_history = np.concatenate((padding, processed_history))
        elif len(processed_history) > self.cgm_history_len:
            processed_history = processed_history[-self.cgm_history_len:]
        state_parts.extend(np.clip(processed_history / 400.0, 0.0, 1.0))
        
        state_parts.append(np.clip(iob / 20.0, 0.0, 1.0))
        state_parts.append(np.clip(cob / 200.0, 0.0, 1.0))
        
        if glucose_predictions:
            processed_predictions = np.array(glucose_predictions, dtype=np.float32)
            if len(processed_predictions) < self.prediction_horizon_len:
                padding_val = processed_predictions[-1] if len(processed_predictions) > 0 else current_cgm
                padding = np.full(self.prediction_horizon_len - len(processed_predictions), padding_val)
                processed_predictions = np.concatenate((processed_predictions, padding))
            elif len(processed_predictions) > self.prediction_horizon_len:
                processed_predictions = processed_predictions[:self.prediction_horizon_len]
            state_parts.extend(np.clip(processed_predictions / 400.0, 0.0, 1.0))
        else:
            state_parts.extend(np.full(self.prediction_horizon_len, np.clip(current_cgm / 400.0, 0.0, 1.0)))

        # Add prediction standard deviations
        if glucose_predictions_std_dev:
            processed_std_devs = np.array(glucose_predictions_std_dev, dtype=np.float32)
            if len(processed_std_devs) < self.prediction_horizon_len:
                # Pad with a high uncertainty (e.g., normalized max possible std dev, or just a fixed high value)
                # For now, let's use a normalized value like 1.0 (max std dev / 400, assuming std dev could be large)
                # A more principled padding value might be derived from typical prediction errors.
                padding_val_std = 1.0 # Placeholder for high uncertainty
                padding_std = np.full(self.prediction_horizon_len - len(processed_std_devs), padding_val_std)
                processed_std_devs = np.concatenate((processed_std_devs, padding_std))
            elif len(processed_std_devs) > self.prediction_horizon_len:
                processed_std_devs = processed_std_devs[:self.prediction_horizon_len]
            # Normalize std dev. Assuming std_dev might be in same units as CGM.
            # Max std dev could be e.g. 50-100 mg/dL. Normalizing by 100 might be reasonable.
            state_parts.extend(np.clip(processed_std_devs / 100.0, 0.0, 1.0))
        else:
            # If no std_dev provided, fill with a value indicating high uncertainty (e.g., 1.0 after normalization)
            state_parts.extend(np.full(self.prediction_horizon_len, 1.0))

        # Calculate and add prediction slopes
        # Assuming 5-minute prediction intervals
        prediction_interval_minutes = 5.0
        slope_30_val_norm = 0.0
        slope_60_val_norm = 0.0

        if glucose_predictions:
            idx_30min = int(30 / prediction_interval_minutes) - 1 # Index 5
            if len(glucose_predictions) > idx_30min:
                pred_at_30 = glucose_predictions[idx_30min]
                raw_slope_30 = (pred_at_30 - current_cgm) / 30.0
                # Normalize slope: clip to +/- 5 mg/dL/min, then scale to +/- 1.0
                slope_30_val_norm = np.clip(raw_slope_30 / 5.0, -1.0, 1.0)

            idx_60min = int(60 / prediction_interval_minutes) - 1 # Index 11
            if len(glucose_predictions) > idx_60min:
                pred_at_60 = glucose_predictions[idx_60min]
                raw_slope_60 = (pred_at_60 - current_cgm) / 60.0
                # Normalize slope: clip to +/- 5 mg/dL/min, then scale to +/- 1.0
                slope_60_val_norm = np.clip(raw_slope_60 / 5.0, -1.0, 1.0)
        
        state_parts.extend([slope_30_val_norm, slope_60_val_norm])

        # Add features from MealDetectorAgent
        state_parts.append(np.clip(meal_probability, 0.0, 1.0))
        state_parts.append(np.clip(estimated_meal_carbs_g / 200.0, 0.0, 1.0)) # Normalize, assuming max 200g

        # Add cyclical time features
        hour_sin = np.sin(2 * np.pi * hour_of_day / 24.0)
        hour_cos = np.cos(2 * np.pi * hour_of_day / 24.0)
        day_sin = np.sin(2 * np.pi * day_of_week / 7.0)
        day_cos = np.cos(2 * np.pi * day_of_week / 7.0)
        state_parts.extend([hour_sin, hour_cos, day_sin, day_cos])

        # Add stress level
        state_parts.append(np.clip(stress_level, 0.0, 1.0))

        # Add time since meal announcement
        if meal_announced and time_since_meal_announcement_minutes is not None:
            # Normalize, e.g., up to 60 minutes. Values beyond this might be clipped or handled by the agent.
            normalized_time_since_meal = np.clip(time_since_meal_announcement_minutes / 60.0, 0.0, 1.0)
            state_parts.append(normalized_time_since_meal)
        else:
            # If no meal announced or time not provided, use a neutral/default value (e.g., 0 or -1 if clipping allows)
            # Using 0.0 implies either just announced or not applicable.
            state_parts.append(0.0)
            
        state_parts.append(1.0 if meal_announced else 0.0)
        state_parts.append(np.clip(announced_carbs / 200.0, 0.0, 1.0))
        
        state_vector = np.array(state_parts, dtype=np.float32)
        
        # Updated expected_dim: + self.prediction_horizon_len for std_devs + 2 for slopes + 2 for meal_detector_features + 4 for time features + 1 for stress + 1 for time_since_meal
        expected_dim = 1 + self.cgm_history_len + 1 + 1 + self.prediction_horizon_len + self.prediction_horizon_len + 2 + 2 + 2 + 4 + 1 + 1
        if self.state_dim != expected_dim:
              print(f"Warning: RLAgent state_dim in __init__ ({self.state_dim}) does not match calculated "
                    f"state dimension ({expected_dim}). This is problematic if state_dim was used to init observation space.")
        
        # Ensure state_vector matches self.state_dim if it was pre-calculated and fixed
        if len(state_vector) != self.state_dim:
            print(f"Warning: Constructed state length {len(state_vector)} does not match "
                  f"agent's state_dim {self.state_dim}. Truncating/Padding.")
            if len(state_vector) < self.state_dim:
                padding = np.zeros(self.state_dim - len(state_vector), dtype=np.float32)
                state_vector = np.concatenate((state_vector, padding))
            else:
                state_vector = state_vector[:self.state_dim]
        return state_vector

    def decide_action(self, current_state: Dict[str, Any], **kwargs) -> Dict[str, float]: # Renamed current_patient_state to current_state
        """Decides an action based on the current state information."""
        glucose_pred_dict: Optional[Dict[str, List[float]]] = None
        meal_detector_output: Optional[Dict[str, Any]] = None
        advisor_suggestion: Optional[Dict[str, Any]] = None

        # 1. Get suggestion from Pattern Advisor if available
        if self.pattern_advisor:
            try:
                advisor_suggestion = self.pattern_advisor.decide_action(current_state, **kwargs)
                if advisor_suggestion:
                    print(f"RLAgent: Received suggestion from PatternAdvisor: {advisor_suggestion}")
            except Exception as e:
                print(f"RLAgent: Error getting suggestion from PatternAdvisor: {e}")
                advisor_suggestion = None
        
        # 2. Get predictions from internal predictor (self.predictor)
        if self.predictor and isinstance(self.predictor, LSTMPredictor):
            # LSTMPredictor expects a DataFrame with 'cgm_mg_dl', 'bolus_U', 'carbs_g'
            required_input_len = self.predictor.input_seq_len

            cgm_hist = current_state.get('cgm_history', [])
            # Assuming 'bolus_history' and 'carbs_history' are provided in current_state
            # and correspond to the cgm_history time points.
            # These would typically be 0 if no bolus/carbs at that specific time step.
            bolus_hist = current_state.get('bolus_history', [0.0] * len(cgm_hist))
            carbs_hist = current_state.get('carbs_history', [0.0] * len(cgm_hist))

            if not cgm_hist: # If cgm_history is empty or not provided, use current CGM
                 cgm_hist = [current_state.get('cgm', 100.0)] # Default to 100 if cgm also missing
                 # Ensure bolus/carbs hists have at least one element if cgm_hist was empty
                 if not bolus_hist: bolus_hist = [0.0]
                 if not carbs_hist: carbs_hist = [0.0]


            # Ensure all histories are of the same length, pad if necessary for consistency
            # This part is tricky as LSTMPredictor expects full sequences.
            # For now, we rely on SimulationEngine to provide consistent length histories.
            # The LSTMPredictor.predict method itself handles cases where len(df) < input_seq_len.

            if len(cgm_hist) >= 1 and len(bolus_hist) == len(cgm_hist) and len(carbs_hist) == len(cgm_hist):
                # Take the most recent 'required_input_len' entries if available,
                # otherwise LSTMPredictor.predict will handle shorter sequences (naive persistence).
                start_idx = max(0, len(cgm_hist) - required_input_len) # Ensure we don't go negative

                predictor_input_df = pd.DataFrame({
                    'cgm_mg_dl': cgm_hist[start_idx:],
                    'bolus_U': bolus_hist[start_idx:],
                    'carbs_g': carbs_hist[start_idx:]
                })
                
                # LSTMPredictor.predict handles if len(predictor_input_df) < required_input_len
                # by returning naive persistence.
                try:
                    glucose_pred_dict = self.predictor.predict(predictor_input_df)
                except Exception as e:
                    print(f"RLAgent: Error getting predictions from LSTMPredictor: {e}")
                    glucose_pred_dict = None
            else:
                print(f"RLAgent: Mismatch in history lengths or insufficient cgm_history for LSTMPredictor. "
                      f"CGM: {len(cgm_hist)}, Bolus: {len(bolus_hist)}, Carbs: {len(carbs_hist)}. Required: {required_input_len}")
                glucose_pred_dict = None
        elif self.predictor: # For other predictor types that might expect np.array
            cgm_for_predictor = current_state.get('cgm_history', [current_state['cgm']])
            if not cgm_for_predictor: cgm_for_predictor = [current_state.get('cgm', 100.0)]
            predictor_input_sequence = np.array(cgm_for_predictor, dtype=np.float32)
            # Reshape based on what a generic BasePredictor might expect (e.g., (1, seq_len, n_features))
            # This part is speculative as BasePredictor.predict is generic.
            # Assuming n_features=1 (just CGM) for a generic case.
            predictor_input_for_model = predictor_input_sequence.reshape(1, -1, 1)
            try:
                glucose_pred_dict = self.predictor.predict(predictor_input_for_model)
            except Exception as e:
                print(f"RLAgent: Error getting predictions from generic predictor: {e}")
                glucose_pred_dict = None
        
        mean_glucose_predictions = glucose_pred_dict["mean"] if glucose_pred_dict and "mean" in glucose_pred_dict else None
        std_dev_glucose_predictions = glucose_pred_dict["std_dev"] if glucose_pred_dict and "std_dev" in glucose_pred_dict else None

        if self.meal_detector:
            try:
                # The meal_detector's decide_action might need specific parts of current_state
                # For now, pass the whole current_state, assuming it can handle it.
                meal_detector_output = self.meal_detector.decide_action(current_state, **kwargs)
            except Exception as e:
                print(f"RLAgent: Error getting meal detection: {e}")
                meal_detector_output = None
        
        # Extract info from meal_detector_output or use defaults from current_state
        meal_announced_flag = current_state.get('meal_announced', False)
        announced_carbs_val = current_state.get('announced_carbs', 0.0)
        meal_probability_val = 0.0
        estimated_meal_carbs_g_val = 0.0

        if meal_detector_output:
            meal_probability_val = meal_detector_output.get('meal_probability', 0.0)
            # If meal_detector provides carb estimation, it might override announced_carbs
            # or supplement it. For now, let's assume it can provide its own estimate.
            estimated_meal_carbs_g_val = meal_detector_output.get('estimated_meal_carbs_g', 0.0)
            # The meal_announced_flag could also be influenced by meal_probability
            if meal_probability_val > 0.5 and not meal_announced_flag : # Example threshold
                 meal_announced_flag = True # If detector is confident, treat as announced
                 if estimated_meal_carbs_g_val > 0 and announced_carbs_val == 0:
                     announced_carbs_val = estimated_meal_carbs_g_val # Use detector's carbs if not otherwise announced

        state_vector = self._define_state(
            current_cgm=current_state['cgm'],
            cgm_history=current_state.get('cgm_history', [current_state['cgm']]),
            iob=current_state['iob'],
            cob=current_state['cob'],
            glucose_predictions=mean_glucose_predictions,
            glucose_predictions_std_dev=std_dev_glucose_predictions,
            meal_announced=meal_announced_flag, # Use potentially updated flag
            announced_carbs=announced_carbs_val, # Use potentially updated carbs
            hour_of_day=current_state.get('hour_of_day', 12),
            day_of_week=current_state.get('day_of_week', 0),
            stress_level=current_state.get('stress_level', 0.0),
            time_since_meal_announcement_minutes=current_state.get('time_since_meal_announcement_minutes'),
            meal_probability=meal_probability_val,
            estimated_meal_carbs_g=estimated_meal_carbs_g_val
        )

        # Store latest observation for rendering/debugging
        self.latest_obs = state_vector

        # Simple out-of-distribution check with probability
        ood_prob = None
        if self.ood_detector:
            ood_prob = self.ood_detector.probability_ood(state_vector)
            if ood_prob > 0.5:
                print(
                    f"\u26a0\ufe0f RLAgent: out-of-distribution probability {ood_prob:.2f}"
                )

        # 3. Decide final action: Use advisor's suggestion or RL model's output
        final_action_dict: Dict[str, float] = {}
        use_advisor_action = False

        if advisor_suggestion:
            potential_action = None
            # Check for direct action suggestion first
            if "suggested_action" in advisor_suggestion and isinstance(advisor_suggestion["suggested_action"], dict):
                potential_action = advisor_suggestion["suggested_action"]
            # Else, check if it's a retrieved pattern with action data
            elif "pattern_data" in advisor_suggestion and isinstance(advisor_suggestion["pattern_data"], dict):
                retrieved_pattern_data = advisor_suggestion["pattern_data"].get("data")
                if isinstance(retrieved_pattern_data, dict):
                    potential_action = retrieved_pattern_data
            
            if potential_action:
                # Check if the potential action contains all necessary keys defined by the RLAgent's action space
                is_complete = all(key in potential_action for key in self.action_keys_ordered)
                if is_complete:
                    try:
                        # Construct the action dict using only the keys defined in RLAgent's action space
                        final_action_dict = {key: float(potential_action[key]) for key in self.action_keys_ordered}
                        use_advisor_action = True
                        print(f"RLAgent: Using action from PatternAdvisor: {final_action_dict}")
                    except (ValueError, TypeError, KeyError) as e:
                        print(f"RLAgent: Error processing advisor action values: {e}. Reverting to RL model.")
                        use_advisor_action = False # Ensure fallback
                else:
                    print(f"RLAgent: Advisor suggestion incomplete or keys mismatch. Advisor keys: {list(potential_action.keys())}, Expected: {self.action_keys_ordered}")
            else:
                print("RLAgent: Advisor suggestion received, but no actionable content found in expected format.")

        if not use_advisor_action:
            if self.rl_model:
                action_sb3, _states = self.rl_model.predict(state_vector, deterministic=True)
                # Convert SB3 action (np.ndarray from Box space) to a dictionary
                current_action_dict = {}
                if len(action_sb3) == len(self.action_keys_ordered):
                    for i, key in enumerate(self.action_keys_ordered):
                        current_action_dict[key] = float(action_sb3[i])
                    final_action_dict = current_action_dict
                    print(f"RLAgent: Using action from RL model: {final_action_dict}")
                else:
                    print(f"RLAgent: ERROR - Mismatch between SB3 action length ({len(action_sb3)}) and ordered keys ({len(self.action_keys_ordered)}). Using placeholder.")
                    # Fallback to placeholder logic to prevent crashes
                    final_action_dict = {key: 0.0 for key in self.action_keys_ordered}
                    if "bolus_u" in final_action_dict: final_action_dict["bolus_u"] = 0.0
                    if "basal_rate_u_hr" in final_action_dict: final_action_dict["basal_rate_u_hr"] = 1.0 # Default basal
            else:
                # Fallback to placeholder logic if no RL model and no usable advisor suggestion
                print("RLAgent: No RL model and no usable advisor suggestion. Using placeholder action logic.")
                final_action_dict = {}
                for key in self.action_keys_ordered:
                    if key == "bolus_u":
                        final_action_dict[key] = np.random.uniform(0, 0.5) if current_state.get('meal_announced', False) else 0.0
                    elif key == "basal_rate_u_hr":
                        final_action_dict[key] = np.random.uniform(0.5, 1.5)
                    else:
                        final_action_dict[key] = 0.0
                
                # If action_keys_ordered is empty (should not happen if _setup_gym_spaces ran)
                if not final_action_dict and isinstance(self.action_space, dict):
                     final_action_dict = {
                        "bolus_u": np.random.uniform(0, 0.5) if current_state.get('meal_announced', False) else 0.0,
                        "basal_rate_u_hr": np.random.uniform(0.5, 1.5)
                     }
                     for key_def in self.action_space.keys():
                        if key_def not in final_action_dict:
                            final_action_dict[key_def] = 0.0
        
        return {k: float(v) for k, v in final_action_dict.items()} if final_action_dict else {}

    def learn(self, experience: Tuple[Any, Any, float, Any, bool]):
        """
        This method is deprecated for PPO. PPO training is handled by train_rl_model,
        which calls the underlying SB3 model's learn() method.
        """
        raise NotImplementedError(
            "The RLAgent.learn(experience) method is deprecated for PPO. "
            "Use RLAgent.train_rl_model(total_timesteps) for training."
        )

    def train_rl_model(self, total_timesteps: int, callback: Any = None):
        """
        Trains the underlying Stable Baselines3 RL model (e.g., PPO).

        This method directly calls the `learn()` method of the `self.rl_model`
        (e.g., an SB3 PPO instance), which is responsible for collecting rollouts
        from its environment and updating its policy.

        Args:
            total_timesteps (int): The total number of samples (environment steps)
                                   to train on.
            callback (Optional[stable_baselines3.common.callbacks.BaseCallback]):
                A callback or list of callbacks for monitoring training, saving models, etc.
                (e.g., EvalCallback, StopTrainingOnRewardThreshold).
        """
        if self.rl_model and hasattr(self.rl_model, 'learn'):
            print(f"RLAgent: Starting training of {self.rl_algorithm_name} model for {total_timesteps} timesteps...")
            try:
                self.rl_model.learn(
                    total_timesteps=total_timesteps,
                    callback=callback,
                    # reset_num_timesteps can be True (default) to reset timestep counter for new learning,
                    # or False to continue counting from previous .learn() calls.
                    # For typical PPO training sessions, True is often desired for a fresh run.
                    # If you are incrementally training, you might set it to False.
                    # SB3 PPO's learn method handles this.
                )
                print(f"RLAgent: Training of {self.rl_algorithm_name} model complete.")
            except Exception as e:
                print(f"RLAgent: Error during RL model training: {e}")
                # Potentially re-raise or handle more gracefully
                raise
        elif not self.rl_model:
            print("RLAgent: No RL model initialized. Cannot train.")
        else: # self.rl_model exists but doesn't have 'learn'
             print(f"RLAgent: RL model ({type(self.rl_model)}) does not have a 'learn' method. Cannot train.")

    def personalize(self, support_data, query_data=None):
        """Run a lightweight meta-learning adaptation step."""
        if self.meta_learner and self.rl_model:
            self.rl_model = self.meta_learner.adapt(support_data, query_data)
        else:
            print("RLAgent: MetaLearner not initialized; skipping personalization.")

    def save(self, path: str):
        """Saves the agent's learned RL model. (Placeholder)"""
        if self.rl_model and hasattr(self.rl_model, 'save'):
            self.rl_model.save(path)
            print(f"RLAgent: {self.rl_algorithm_name} model saved to {path}")
        else:
            print(f"RLAgent: No RL model to save or model does not support saving (model: {self.rl_model}).")

    def load(self, path: str):
        """Loads a pre-trained RL model for the agent."""
        if self.observation_space_gym is None or self.action_space_gym is None:
            print("CRITICAL ERROR: Observation/Action space not set up before attempting to load model.")
            return

        dummy_env_instance = _MinimalGymEnv(
            observation_space=self.observation_space_gym,
            action_space=self.action_space_gym
        )
        dummy_vec_env = DummyVecEnv([lambda: dummy_env_instance])

        model_class = None
        if self.rl_algorithm_name.upper() == "PPO":
            model_class = PPO
        elif self.rl_algorithm_name.upper() == "SAC":
            model_class = SAC
        # Add other algorithms here if supported in the future
        # elif self.rl_algorithm_name.upper() == "TD3":
        #     model_class = TD3

        if model_class:
            try:
                self.rl_model = model_class.load(path, env=dummy_vec_env)
                # Verification of spaces after load
                if hasattr(self.rl_model, 'observation_space') and self.rl_model.observation_space != self.observation_space_gym:
                    print(f"Warning: Loaded {self.rl_algorithm_name} model's observation space differs from agent's.")
                if hasattr(self.rl_model, 'action_space') and self.rl_model.action_space != self.action_space_gym:
                    print(f"Warning: Loaded {self.rl_algorithm_name} model's action space differs from agent's.")
                print(f"RLAgent: {self.rl_algorithm_name} model loaded from {path}")
            except Exception as e:
                print(f"RLAgent: Error loading {self.rl_algorithm_name} model from {path}: {e}")
                self.rl_model = None # Ensure model is None if loading fails
        else:
            print(f"RLAgent: Loading not implemented for algorithm '{self.rl_algorithm_name}'.")

    def store_experience(self, experience: Any) -> None:
        """Add an experience tuple to the local replay buffer."""
        if self.federated_client:
            self.federated_client.replay_buffer.add(experience)

    def continual_train_step(self, batch_size: int = 32) -> None:
        """Trigger a continual learning update from the replay buffer."""
        if self.federated_client:
            self.federated_client.continual_update(batch_size)

    def federated_round(self, data) -> None:
        """Run a simple federated learning round with local data."""
        if self.federated_client:
            self.federated_client.train_local(data)
            self.federated_client.continual_update()
            self.federated_client.share_updates()

    def render(self) -> None:
        """Minimal render showing the latest observation."""
        if self.latest_obs is not None:
            print(f"RLAgent: latest observation {self.latest_obs}")
        else:
            print("RLAgent: no observation available to render")

    def close(self) -> None:
        """Clean up any resources associated with the RL model."""
        if self.rl_model and hasattr(self.rl_model, 'env') and hasattr(self.rl_model.env, 'close'):
            try:
                self.rl_model.env.close()
            except Exception as e:
                print(f"RLAgent: error closing environment: {e}")

if __name__ == '__main__':
    # Moved imports for __main__ block to resolve circular dependencies
    from DiaGuardianAI.agents.pattern_advisor_agent import PatternAdvisorAgent
    # RepositoryManager is already imported at the top level, which is fine if not part of the cycle.

    # Example of defining an action space (requires gymnasium)
    # try:
    #     import gymnasium as gym
    #     action_space_example = gym.spaces.Dict({
    #         "basal_rate_adjustment_type": gym.spaces.Discrete(3),  # 0: maintain, 1: increase %, 2: decrease %  # noqa: E501
    #         "basal_rate_adjustment_value_percent": gym.spaces.Box(
    #             low=0, high=50, shape=(1,), dtype=np.float32
    #         ),
    #         "bolus_type": gym.spaces.Discrete(2),  # 0: no bolus, 1: standard bolus  # noqa: E501
    #         "bolus_u": gym.spaces.Box(
    #             low=0, high=20, shape=(1,), dtype=np.float32
    #         )
    #     })
    #     state_dim_example = 20  # Example state dimension
    #     agent = RLAgent(
    #         state_dim=state_dim_example,
    #         action_space=action_space_example, rl_algorithm="PPO"
    # Example usage:
    # Define a simple action space structure (if not using gymnasium)
    # This is just for the placeholder logic to interpret.
    action_space_def_example = {
        "bolus_u": {"low": 0.0, "high": 15.0}, # Units
        "basal_rate_u_hr": {"low": 0.0, "high": 3.0} # Units per hour
        # Could add: "basal_rate_change_percent", "temp_basal_u_hr", "temp_basal_duration_minutes"
    }

    # Calculate state_dim based on the components in _define_state
    # 1 (cgm) + cgm_history_len (24) + 1 (iob) + 1 (cob) + prediction_horizon_len (6 for mean) + prediction_horizon_len (6 for std_dev) + 2 (slopes) + 2 (meal_detector_features) + 4 (time) + 1 (stress) + 1 (time_since_meal_announce) + 1 (meal_announced) + 1 (announced_carbs)
    cgm_hist_len_example = 24 # Changed to 24
    pred_horizon_len_example = 6 # This is for how many direct predictions are in state
    # Recalculate based on RLAgent._define_state structure:
    # 1 (cgm) + cgm_hist_len + 1 (iob) + 1 (cob) + pred_horizon (mean) + pred_horizon (std_dev) +
    # 2 (slopes) + 2 (meal_detector_out) + 4 (time) + 1 (stress) + 1 (time_since_meal) + 1 (meal_announced) + 1 (announced_carbs)
    calculated_state_dim = (1 + cgm_hist_len_example + 1 + 1 +
                            pred_horizon_len_example + pred_horizon_len_example +
                            2 + 2 + 4 + 1 + 1 + 1 + 1) # = 1 + 24 + 1 + 1 + 6 + 6 + 2 + 2 + 4 + 1 + 1 + 1 + 1 = 51
 
    # Load the LSTMPredictor
    loaded_lstm_predictor: Optional[LSTMPredictor] = None
    # Construct path to the model directory relative to project_root
    # Determine project_root_main specifically for __main__ execution
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    # Assuming decision_agent.py is in DiaGuardianAI/agents/
    project_root_main = os.path.abspath(os.path.join(current_file_dir, '..', '..'))

    model_dir_relative = os.path.join("DiaGuardianAI", "models", "lstm_predictor_example_run")
    lstm_model_path = os.path.join(project_root_main, model_dir_relative)

    print(f"RLAgent __main__: Attempting to load LSTMPredictor from: {lstm_model_path}")
    if os.path.exists(lstm_model_path):
        try:
            # Instantiate LSTMPredictor first, then call load (instance method)
            # Use dummy parameters for init, as load will overwrite them from config
            loaded_lstm_predictor = LSTMPredictor(input_seq_len=1, output_seq_len=1, n_features=1)
            loaded_lstm_predictor.load(lstm_model_path) # Call load on the instance
            print(f"RLAgent __main__: LSTMPredictor loaded successfully: {type(loaded_lstm_predictor)}")
        except Exception as e:
            print(f"RLAgent __main__: Failed to load LSTMPredictor: {e}")
            loaded_lstm_predictor = None
    else:
        print(f"RLAgent __main__: LSTMPredictor model path not found: {lstm_model_path}")
        loaded_lstm_predictor = None

    agent = RLAgent(
        state_dim=calculated_state_dim,
        action_space_definition=action_space_def_example,
        cgm_history_len=cgm_hist_len_example,
        prediction_horizon_len=pred_horizon_len_example,
        predictor=loaded_lstm_predictor, # Pass the loaded predictor
        meal_detector=None, # No actual meal detector for this simple test
        # pattern_advisor=None, # Will be replaced by a real advisor instance
        rl_algorithm="SAC", # Using SAC for example, can be PPO
        rl_params={"learning_starts": 10, "verbose": 0} # Added verbose to quiet SB3 init
    )

    # --- Setup PatternAdvisorAgent for testing RLAgent integration (using TRAINED internal model) ---
    print("\n--- Testing RLAgent with TRAINED PatternAdvisorAgent (internal model) ---")
    
    # Path to the trained advisor model
    trained_advisor_model_path = os.path.join(project_root_main, "DiaGuardianAI", "models",
                                              "pattern_advisor_agent_model", "pattern_advisor_supervised.joblib")

    pattern_advisor_instance = None # Initialize for broader scope
    dummy_repo_for_loaded_advisor = None # Initialize for cleanup scope
    dummy_repo_for_loaded_advisor_path = None # Initialize for cleanup scope

    if not os.path.exists(trained_advisor_model_path):
        print(f"ERROR: Trained PatternAdvisorAgent model not found at {trained_advisor_model_path}")
        print("Please run DiaGuardianAI/examples/train_pattern_advisor.py first.")
        # Fallback to no advisor or skip this test part
        pattern_advisor_instance = None
    else:
        # Dummy repository for the loaded advisor (it won't be used if internal model predicts)
        dummy_repo_for_loaded_advisor_path = "dummy_decision_agent_trained_advisor_repo.sqlite"
        if os.path.exists(dummy_repo_for_loaded_advisor_path):
            os.remove(dummy_repo_for_loaded_advisor_path)
        
        dummy_repo_for_loaded_advisor = RepositoryManager(db_path=dummy_repo_for_loaded_advisor_path)

        # Instantiate PatternAdvisorAgent - its state_dim MUST match RLAgent's obs space (calculated_state_dim)
        # The loaded model was trained on features of this length.
        # The advisor model was trained with features of length calculated_state_dim (39).
        advisor_model_expected_state_dim = calculated_state_dim # Use calculated_state_dim (39)
        pattern_advisor_instance = PatternAdvisorAgent(
            state_dim=advisor_model_expected_state_dim, # Use calculated_state_dim (39)
            action_space="PredictActionCategory", # Placeholder
            pattern_repository=dummy_repo_for_loaded_advisor, # Needs a repo instance
            learning_model_type="supervised_classifier" # Must match the type of the saved model
        )
        try:
            pattern_advisor_instance.load(trained_advisor_model_path)
            print(f"RLAgent __main__: Successfully loaded trained PatternAdvisorAgent from {trained_advisor_model_path}")
        except Exception as e:
            print(f"RLAgent __main__: FAILED to load trained PatternAdvisorAgent: {e}")
            pattern_advisor_instance = None # Fallback if loading fails
        
        # NOTE: Cleanup for dummy_repo_for_loaded_advisor is moved after its use.

    # Initialize RLAgent with the (potentially trained) PatternAdvisor
    # RLAgent itself uses calculated_state_dim (39) for its own state vector construction.
    agent = RLAgent(
        state_dim=calculated_state_dim,
        action_space_definition=action_space_def_example,
        cgm_history_len=cgm_hist_len_example,
        prediction_horizon_len=pred_horizon_len_example,
        predictor=loaded_lstm_predictor,
        meal_detector=None,
        pattern_advisor=pattern_advisor_instance, # Pass the created advisor
        rl_algorithm="SAC",
        rl_params={"learning_starts": 10, "verbose": 0}
    )
    print("RLAgent __main__: RLAgent re-initialized WITH PatternAdvisorAgent.")
    # --- End PatternAdvisorAgent setup ---
    if loaded_lstm_predictor:
        print("RLAgent __main__: RLAgent initialized WITH LSTMPredictor.")
    else:
        print("RLAgent __main__: RLAgent initialized WITHOUT LSTMPredictor (loading failed or path not found).")

    dummy_patient_state_example = {
        'cgm': 120.0, # Feature used by PatternAdvisor
        'cgm_history': [120.0 - (i * 0.1) for i in range(cgm_hist_len_example)][::-1], # 24 entries
        'bolus_history': [0.0] * cgm_hist_len_example, # Added for LSTMPredictor input, now 24 entries
        'carbs_history': [0.0] * cgm_hist_len_example, # Added for LSTMPredictor input, now 24 entries
        'iob': 1.5, # Feature used by PatternAdvisor
        'cob': 20.0, # Feature used by PatternAdvisor
        'meal_announced': True,
        'announced_carbs': 30.0,
        'hour_of_day': 14, # e.g., 2 PM
        'day_of_week': 2,  # e.g., Wednesday (0=Mon, 1=Tue, 2=Wed)
        'stress_level': 0.2,
        'time_since_meal_announcement_minutes': 10.0
        # REMOVED 'pattern_type_preference' to force advisor to use internal model
        # 'meal_probability': 0.8,
        # 'estimated_meal_carbs_g': 50.0
    }
    
    if pattern_advisor_instance: # Only proceed if advisor was loaded
        print(f"\nRLAgent __main__: Calling decide_action with TRAINED PatternAdvisor (using internal model). State: {dummy_patient_state_example}")
        # The RLAgent's decide_action will pass the current_state (dict) to the advisor.
        # The advisor's _prepare_features_for_internal_model will then process this dict
        # (or ideally, RLAgent would pass its state_vector directly if advisor expects that).
        # For now, let's ensure the advisor's _prepare_features_for_internal_model can handle the dict.
        # The current PatternAdvisorAgent._prepare_features_for_internal_model expects RLAgent's obs vector.
        # So, RLAgent should pass its state_vector to advisor.decide_action if that's the case.
        # This is getting complex. For this test, let's assume PatternAdvisorAgent.decide_action
        # correctly calls its _prepare_features_for_internal_model with the current_state dict,
        # and _prepare_features_for_internal_model can create the RLAgent-like feature vector from it.
        # The current `_prepare_features_for_internal_model` in PatternAdvisorAgent expects an np.ndarray (RLAgent's obs)
        # or a dict. If it gets a dict, it tries to build features. This should work.

        action = agent.decide_action(dummy_patient_state_example)
        print(f"RLAgent __main__: RLAgent decided action: {action}")

        # Check if the suggestion came from the advisor's internal model
        # This requires checking the 'suggestion_type' in the advisor_suggestion dict,
        # which is internal to RLAgent.decide_action.
        # For now, we'll infer by seeing if it's NOT the RL model's action or a known repo pattern.
        # A more direct check would involve RLAgent logging the source of its chosen action.
        # Let's assume if an action is returned and it's not the default RL model action, it might be from the advisor.
        # The advisor's _format_advice_from_internal_model returns a dict with "suggestion_type": "model_predicted_action_category"
        # We need RLAgent to log this. For now, we just observe the printout from RLAgent.
    else:
        print("\nRLAgent __main__: Skipping test with trained PatternAdvisor as it failed to load.")
        # Optionally, run with no advisor or the repo-based advisor test again
        # For now, just end this part of the test.
        action = agent.decide_action(dummy_patient_state_example) # Run with whatever advisor agent has (None if load failed)
        print(f"RLAgent __main__: RLAgent decided action (advisor load failed/skipped): {action}")


        print(f"RLAgent __main__: RLAgent decided action (advisor load failed/skipped): {action}")

    # Cleanup for the dummy_repo_for_loaded_advisor (used by the trained advisor instance)
    if dummy_repo_for_loaded_advisor is not None and hasattr(dummy_repo_for_loaded_advisor, 'conn') and dummy_repo_for_loaded_advisor.conn:
        dummy_repo_for_loaded_advisor.conn.close()
        print("RLAgent __main__: Closed DB connection for dummy_repo_for_loaded_advisor.")
    
    if dummy_repo_for_loaded_advisor_path is not None and os.path.exists(dummy_repo_for_loaded_advisor_path):
        os.remove(dummy_repo_for_loaded_advisor_path)
        print(f"RLAgent __main__: Cleaned up {dummy_repo_for_loaded_advisor_path}.")

    # Dummy experience for learn method (now deprecated for PPO/SAC via this direct call)
    # dummy_experience = (
    #     np.random.rand(calculated_state_dim), # state
    #     action, # action (dict)
    #     1.0, # reward
    #     np.random.rand(calculated_state_dim), # next_state
    #     False # done
    # )
    # agent.learn(dummy_experience) # This will now raise NotImplementedError

    agent.save("./dummy_rl_agent_before_train_example")
    agent.load("./dummy_rl_agent_before_train_example")
    print(f"Agent saved and loaded from ./dummy_rl_agent_before_train_example")

    # Example of initiating training
    # Note: The _MinimalGymEnv used by the agent is not interactive for real training yet.
    # This is a conceptual demonstration.
    print("\n--- Example: Initiating Agent Training (Conceptual) ---")
    try:
        print("Calling agent.train_rl_model(total_timesteps=50)...") # Very few timesteps
        agent.train_rl_model(total_timesteps=50)
        # SB3's .learn() will print its own progress if verbose > 0.
        # The _MinimalGymEnv will just loop with static observations.
        agent.save("./dummy_rl_agent_after_train_example")
        print("Conceptual training finished and model saved to ./dummy_rl_agent_after_train_example")
    except NotImplementedError as e:
        print(f"Note: {e}") # This would happen if learn(experience) was called directly
    except Exception as e:
        print(f"Error during conceptual training example: {e}")
        print("This might be expected if the _MinimalGymEnv is not fully interactive for training yet.")


    print("\nRLAgent (DecisionAgent) example run complete.")