# DiaGuardianAI Decision Agent
# The primary RL agent for managing insulin delivery and recognizing meal patterns.

import sys
import os

if __package__ is None or __package__ == '':
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from DiaGuardianAI.core.base_classes import BaseAgent, BasePredictiveModel
from DiaGuardianAI.predictive_models.model_zoo.lstm_predictor import LSTMPredictor
from typing import Any, Optional, Dict, List, Tuple, cast
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium.spaces import Box, Dict as GymDict, Space
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv

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
    observation_space: Space[np.ndarray]
    action_space: Space[np.ndarray]

    def __init__(self, observation_space: gym.spaces.Box, action_space: gym.spaces.Box):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        current_obs_box_space = cast(gym.spaces.Box, self.observation_space)
        obs_shape: Tuple[int, ...] = current_obs_box_space.shape
        obs_dtype: np.dtype = current_obs_box_space.dtype
        self.current_obs: np.ndarray = np.zeros(obs_shape, dtype=obs_dtype)

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        return self.current_obs, 0.0, False, False, {}

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed, options=options)
        current_obs_box_space = cast(gym.spaces.Box, self.observation_space)
        obs_shape: Tuple[int, ...] = current_obs_box_space.shape
        obs_dtype: np.dtype = current_obs_box_space.dtype
        self.current_obs = np.zeros(obs_shape, dtype=obs_dtype)
        return self.current_obs, {}

    def render(self):
        print(f"Current observation: {self.current_obs}")

    def close(self):
        print("_MinimalGymEnv closed.")

class RLAgent(BaseAgent):
    """
    Reinforcement Learning agent for diabetes management.
    """

    def __init__(self, state_dim: int, action_space_definition: Any,
                 predictor: Optional[BasePredictiveModel] = None,
                 meal_detector: Optional[BaseAgent] = None,
                 pattern_advisor: Optional[BaseAgent] = None,
                 cgm_history_len: int = 24,
                 prediction_horizon_len: int = 6,
                 rl_algorithm: str = "PPO",
                 rl_params: Optional[Dict[str, Any]] = None,
                 learning_rate: float = 3e-4):
        super().__init__(state_dim, action_space_definition, predictor)
        self.meal_detector = meal_detector
        self.pattern_advisor = pattern_advisor
        self.cgm_history_len: int = cgm_history_len
        self.prediction_horizon_len: int = prediction_horizon_len
        self.rl_algorithm_name: str = rl_algorithm
        self.rl_params: Dict[str, Any] = rl_params if rl_params else {}
        self.learning_rate: float = learning_rate
        self.rl_model: Optional[Any] = None

        self.observation_space_gym: Optional[gym.spaces.Box] = None
        self.action_space_gym: Optional[gym.spaces.Box] = None
        self.action_keys_ordered: List[str] = []

        self._setup_gym_spaces()
        self._initialize_rl_model()

        self.meta_learner = MetaLearner(self.rl_model, algorithm="maml")
        self.federated_client = FederatedClient(self.rl_model, client_id="rl_agent")
        self.ood_detector = SimpleOODDetector()

        print(
            f"RLAgent initialized with algorithm: {self.rl_algorithm_name}, "
            f"state_dim: {self.state_dim}, action_space: {self.action_space}"
        )

    def _setup_gym_spaces(self):
        self.observation_space_gym = Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32)
        if not isinstance(self.action_space, dict):
            raise ValueError("action_space_definition in RLAgent must be a dictionary.")

        low_bounds = []
        high_bounds = []
        self.action_keys_ordered = []
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
        if not low_bounds:
            raise ValueError("Action space definition resulted in empty bounds.")
        self.action_space_gym = Box(
            low=np.array(low_bounds, dtype=np.float32),
            high=np.array(high_bounds, dtype=np.float32),
            dtype=np.float32
        )
        print(f"RLAgent: Observation space: {self.observation_space_gym}")
        print(f"RLAgent: Action space (Box): {self.action_space_gym}, Ordered Keys: {self.action_keys_ordered}")

    def _initialize_rl_model(self):
        if self.observation_space_gym is None or self.action_space_gym is None:
            print("Error: Observation or action space not defined. Cannot initialize RL model.")
            return
        dummy_env_instance = _MinimalGymEnv(
            observation_space=self.observation_space_gym,
            action_space=self.action_space_gym
        )
        dummy_vec_env = DummyVecEnv([lambda: dummy_env_instance])
        policy_str = self.rl_params.get("policy", "MlpPolicy")
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
            for key, value in self.rl_params.items():
                if key not in ["policy", "policy_kwargs", "learning_rate"]:
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
                "buffer_size": self.rl_params.get("buffer_size", 1_000_000),
                "batch_size": self.rl_params.get("batch_size", 256),
                "gamma": self.rl_params.get("gamma", 0.99),
                "tau": self.rl_params.get("tau", 0.005),
                "train_freq": self.rl_params.get("train_freq", 1),
                "gradient_steps": self.rl_params.get("gradient_steps", 1),
                "learning_starts": self.rl_params.get("learning_starts", 100),
                "ent_coef": self.rl_params.get("ent_coef", 'auto'),
                "verbose": self.rl_params.get("verbose", 0),
                "tensorboard_log": self.rl_params.get("tensorboard_log", None)
            }
            for key, value in self.rl_params.items():
                if key not in ["policy", "policy_kwargs", "learning_rate"]:
                    sac_constructor_params[key] = value
            self.rl_model = SAC(
                policy=policy_str,
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
                      glucose_predictions_std_dev: Optional[List[float]] = None,
                      meal_announced: bool = False,
                      announced_carbs: float = 0.0,
                      hour_of_day: int = 12,
                      day_of_week: int = 0,
                      stress_level: float = 0.0,
                      time_since_meal_announcement_minutes: Optional[float] = None,
                      meal_probability: float = 0.0,
                      estimated_meal_carbs_g: float = 0.0
                      ) -> np.ndarray:
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
        if glucose_predictions_std_dev:
            processed_std_devs = np.array(glucose_predictions_std_dev, dtype=np.float32)
            if len(processed_std_devs) < self.prediction_horizon_len:
                padding_val_std = 1.0
                padding_std = np.full(self.prediction_horizon_len - len(processed_std_devs), padding_val_std)
                processed_std_devs = np.concatenate((processed_std_devs, padding_std))
            elif len(processed_std_devs) > self.prediction_horizon_len:
                processed_std_devs = processed_std_devs[:self.prediction_horizon_len]
            state_parts.extend(np.clip(processed_std_devs / 100.0, 0.0, 1.0))
        else:
            state_parts.extend(np.full(self.prediction_horizon_len, 1.0))
        prediction_interval_minutes = 5.0
        slope_30_val_norm = 0.0
        slope_60_val_norm = 0.0
        if glucose_predictions:
            idx_30min = int(30 / prediction_interval_minutes) - 1
            if len(glucose_predictions) > idx_30min:
                pred_at_30 = glucose_predictions[idx_30min]
                raw_slope_30 = (pred_at_30 - current_cgm) / 30.0
                slope_30_val_norm = np.clip(raw_slope_30 / 5.0, -1.0, 1.0)
            idx_60min = int(60 / prediction_interval_minutes) - 1
            if len(glucose_predictions) > idx_60min:
                pred_at_60 = glucose_predictions[idx_60min]
                raw_slope_60 = (pred_at_60 - current_cgm) / 60.0
                slope_60_val_norm = np.clip(raw_slope_60 / 5.0, -1.0, 1.0)
        state_parts.extend([slope_30_val_norm, slope_60_val_norm])
        state_parts.append(np.clip(meal_probability, 0.0, 1.0))
        state_parts.append(np.clip(estimated_meal_carbs_g / 200.0, 0.0, 1.0))
        hour_sin = np.sin(2 * np.pi * hour_of_day / 24.0)
        hour_cos = np.cos(2 * np.pi * hour_of_day / 24.0)
        day_sin = np.sin(2 * np.pi * day_of_week / 7.0)
        day_cos = np.cos(2 * np.pi * day_of_week / 7.0)
        state_parts.extend([hour_sin, hour_cos, day_sin, day_cos])
        state_parts.append(np.clip(stress_level, 0.0, 1.0))
        if meal_announced and time_since_meal_announcement_minutes is not None:
            normalized_time_since_meal = np.clip(time_since_meal_announcement_minutes / 60.0, 0.0, 1.0)
            state_parts.append(normalized_time_since_meal)
        else:
            state_parts.append(0.0)
        state_parts.append(1.0 if meal_announced else 0.0)
        state_parts.append(np.clip(announced_carbs / 200.0, 0.0, 1.0))
        state_vector = np.array(state_parts, dtype=np.float32)
        expected_dim = 1 + self.cgm_history_len + 1 + 1 + self.prediction_horizon_len + self.prediction_horizon_len + 2 + 2 + 2 + 4 + 1 + 1
        if self.state_dim != expected_dim:
            print(f"Warning: RLAgent state_dim in __init__ ({self.state_dim}) does not match calculated "
                  f"state dimension ({expected_dim}). This is problematic if state_dim was used to init observation space.")
        if len(state_vector) != self.state_dim:
            print(f"Warning: Constructed state length {len(state_vector)} does not match "
                  f"agent's state_dim {self.state_dim}. Truncating/Padding.")
            if len(state_vector) < self.state_dim:
                padding = np.zeros(self.state_dim - len(state_vector), dtype=np.float32)
                state_vector = np.concatenate((state_vector, padding))
            else:
                state_vector = state_vector[:self.state_dim]
        return state_vector

    def decide_action(self, current_state: Dict[str, Any], **kwargs) -> Dict[str, float]:
        glucose_pred_dict: Optional[Dict[str, List[float]]] = None
        meal_detector_output: Optional[Dict[str, Any]] = None
        advisor_suggestion: Optional[Dict[str, Any]] = None
        if self.pattern_advisor:
            try:
                advisor_suggestion = self.pattern_advisor.decide_action(current_state, **kwargs)
                if advisor_suggestion:
                    print(f"RLAgent: Received suggestion from PatternAdvisor: {advisor_suggestion}")
            except Exception as e:
                print(f"RLAgent: Error getting suggestion from PatternAdvisor: {e}")
                advisor_suggestion = None
        if self.predictor and isinstance(self.predictor, LSTMPredictor):
            required_input_len = self.predictor.input_seq_len
            cgm_hist = current_state.get('cgm_history', [])
            bolus_hist = current_state.get('bolus_history', [0.0] * len(cgm_hist))
            carbs_hist = current_state.get('carbs_history', [0.0] * len(cgm_hist))
            if not cgm_hist:
                cgm_hist = [current_state.get('cgm', 100.0)]
                if not bolus_hist: bolus_hist = [0.0]
                if not carbs_hist: carbs_hist = [0.0]
            if len(cgm_hist) >= 1 and len(bolus_hist) == len(cgm_hist) and len(carbs_hist) == len(cgm_hist):
                start_idx = max(0, len(cgm_hist) - required_input_len)
                predictor_input_df = pd.DataFrame({
                    'cgm_mg_dl': cgm_hist[start_idx:],
                    'bolus_U': bolus_hist[start_idx:],
                    'carbs_g': carbs_hist[start_idx:]
                })
                try:
                    glucose_pred_dict = self.predictor.predict(predictor_input_df)
                except Exception as e:
                    print(f"RLAgent: Error getting predictions from LSTMPredictor: {e}")
                    glucose_pred_dict = None
            else:
                print(f"RLAgent: Mismatch in history lengths or insufficient cgm_history for LSTMPredictor. "
                      f"CGM: {len(cgm_hist)}, Bolus: {len(bolus_hist)}, Carbs: {len(carbs_hist)}. Required: {required_input_len}")
                glucose_pred_dict = None
        elif self.predictor:
            cgm_for_predictor = current_state.get('cgm_history', [current_state['cgm']])
            if not cgm_for_predictor: cgm_for_predictor = [current_state.get('cgm', 100.0)]
            predictor_input_sequence = np.array(cgm_for_predictor, dtype=np.float32)
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
                meal_detector_output = self.meal_detector.decide_action(current_state, **kwargs)
            except Exception as e:
                print(f"RLAgent: Error getting meal detection: {e}")
                meal_detector_output = None
        meal_announced_flag = current_state.get('meal_announced', False)
        announced_carbs_val = current_state.get('announced_carbs', 0.0)
        meal_probability_val = 0.0
        estimated_meal_carbs_g_val = 0.0
        if meal_detector_output:
            meal_probability_val = meal_detector_output.get('meal_probability', 0.0)
            estimated_meal_carbs_g_val = meal_detector_output.get('estimated_meal_carbs_g', 0.0)
            if meal_probability_val > 0.5 and not meal_announced_flag:
                meal_announced_flag = True
                if estimated_meal_carbs_g_val > 0 and announced_carbs_val == 0:
                    announced_carbs_val = estimated_meal_carbs_g_val
        state_vector = self._define_state(
            current_cgm=current_state['cgm'],
            cgm_history=current_state.get('cgm_history', [current_state['cgm']]),
            iob=current_state['iob'],
            cob=current_state['cob'],
            glucose_predictions=mean_glucose_predictions,
            glucose_predictions_std_dev=std_dev_glucose_predictions,
            meal_announced=meal_announced_flag,
            announced_carbs=announced_carbs_val,
            hour_of_day=current_state.get('hour_of_day', 12),
            day_of_week=current_state.get('day_of_week', 0),
            stress_level=current_state.get('stress_level', 0.0),
            time_since_meal_announcement_minutes=current_state.get('time_since_meal_announcement_minutes'),
            meal_probability=meal_probability_val,
            estimated_meal_carbs_g=estimated_meal_carbs_g_val
        )
        if self.ood_detector and self.ood_detector.is_ood(state_vector):
            print("\u26a0\ufe0f RLAgent: out-of-distribution state detected")
        final_action_dict: Dict[str, float] = {}
        use_advisor_action = False
        if advisor_suggestion:
            potential_action = None
            if "suggested_action" in advisor_suggestion and isinstance(advisor_suggestion["suggested_action"], dict):
                potential_action = advisor_suggestion["suggested_action"]
            elif "pattern_data" in advisor_suggestion and isinstance(advisor_suggestion["pattern_data"], dict):
                retrieved_pattern_data = advisor_suggestion["pattern_data"].get("data")
                if isinstance(retrieved_pattern_data, dict):
                    potential_action = retrieved_pattern_data
            if potential_action:
                is_complete = all(key in potential_action for key in self.action_keys_ordered)
                if is_complete:
                    try:
                        final_action_dict = {key: float(potential_action[key]) for key in self.action_keys_ordered}
                        use_advisor_action = True
                        print(f"RLAgent: Using action from PatternAdvisor: {final_action_dict}")
                    except (ValueError, TypeError, KeyError) as e:
                        print(f"RLAgent: Error processing advisor action values: {e}. Reverting to RL model.")
                        use_advisor_action = False
                else:
                    print(f"RLAgent: Advisor suggestion incomplete or keys mismatch. Advisor keys: {list(potential_action.keys())}, Expected: {self.action_keys_ordered}")
            else:
                print("RLAgent: Advisor suggestion received, but no actionable content found in expected format.")
        if not use_advisor_action:
            if self.rl_model:
                action_sb3, _states = self.rl_model.predict(state_vector, deterministic=True)
                current_action_dict = {}
                if len(action_sb3) == len(self.action_keys_ordered):
                    for i, key in enumerate(self.action_keys_ordered):
                        current_action_dict[key] = float(action_sb3[i])
                    final_action_dict = current_action_dict
                    print(f"RLAgent: Using action from RL model: {final_action_dict}")
                else:
                    print(f"RLAgent: ERROR - Mismatch between SB3 action length ({len(action_sb3)}) and ordered keys ({len(self.action_keys_ordered)}). Using placeholder.")
                    final_action_dict = {key: 0.0 for key in self.action_keys_ordered}
                    if "bolus_u" in final_action_dict: final_action_dict["bolus_u"] = 0.0
                    if "basal_rate_u_hr" in final_action_dict: final_action_dict["basal_rate_u_hr"] = 1.0
            else:
                print("RLAgent: No RL model and no usable advisor suggestion. Using placeholder action logic.")
                final_action_dict = {}
                for key in self.action_keys_ordered:
                    if key == "bolus_u":
                        final_action_dict[key] = np.random.uniform(0, 0.5) if current_state.get('meal_announced', False) else 0.0
                    elif key == "basal_rate_u_hr":
                        final_action_dict[key] = np.random.uniform(0.5, 1.5)
                    else:
                        final_action_dict[key] = 0.0
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
        raise NotImplementedError(
            "The RLAgent.learn(experience) method is deprecated for PPO. "
            "Use RLAgent.train_rl_model(total_timesteps) for training."
        )

    def train_rl_model(self, total_timesteps: int, callback: Any = None):
        if self.rl_model and hasattr(self.rl_model, 'learn'):
            print(f"RLAgent: Starting training of {self.rl_algorithm_name} model for {total_timesteps} timesteps...")
            try:
                self.rl_model.learn(
                    total_timesteps=total_timesteps,
                    callback=callback,
                )
                print(f"RLAgent: Training of {self.rl_algorithm_name} model complete.")
            except Exception as e:
                print(f"RLAgent: Error during RL model training: {e}")
                raise
        elif not self.rl_model:
            print("RLAgent: No RL model initialized. Cannot train.")
        else:
            print(f"RLAgent: RL model ({type(self.rl_model)}) does not have a 'learn' method. Cannot train.")

    def personalize(self, support_data, query_data=None):
        if self.meta_learner and self.rl_model:
            self.rl_model = self.meta_learner.adapt(support_data, query_data)
        else:
            print("RLAgent: MetaLearner not initialized; skipping personalization.")

    def save(self, path: str):
        if self.rl_model and hasattr(self.rl_model, 'save'):
            self.rl_model.save(path)
            print(f"RLAgent: {self.rl_algorithm_name} model saved to {path}")
        else:
            print(f"RLAgent: No RL model to save or model does not support saving (model: {self.rl_model}).")

    def load(self, path: str):
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
        if model_class:
            try:
                self.rl_model = model_class.load(path, env=dummy_vec_env)
                if hasattr(self.rl_model, 'observation_space') and self.rl_model.observation_space != self.observation_space_gym:
                    print(f"Warning: Loaded {self.rl_algorithm_name} model's observation space differs from agent's.")
                if hasattr(self.rl_model, 'action_space') and self.rl_model.action_space != self.action_space_gym:
                    print(f"Warning: Loaded {self.rl_algorithm_name} model's action space differs from agent's.")
                print(f"RLAgent: {self.rl_algorithm_name} model loaded from {path}")
            except Exception as e:
                print(f"RLAgent: Error loading {self.rl_algorithm_name} model from {path}: {e}")
                self.rl_model = None
        else:
            print(f"RLAgent: Loading not implemented for algorithm '{self.rl_algorithm_name}'.")

if __name__ == '__main__':
    from DiaGuardianAI.agents.pattern_advisor_agent import PatternAdvisorAgent
    action_space_def_example = {
        "bolus_u": {"low": 0.0, "high": 15.0},
        "basal_rate_u_hr": {"low": 0.0, "high": 3.0}
    }
    cgm_hist_len_example = 24
    pred_horizon_len_example = 6
    calculated_state_dim = (1 + cgm_hist_len_example + 1 + 1 +
                            pred_horizon_len_example + pred_horizon_len_example +
                            2 + 2 + 4 + 1 + 1 + 1 + 1)
    loaded_lstm_predictor: Optional[LSTMPredictor] = None
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    project_root_main = os.path.abspath(os.path.join(current_file_dir, '..', '..'))
    model_dir_relative = os.path.join("DiaGuardianAI", "models", "lstm_predictor_example_run")
    lstm_model_path = os.path.join(project_root_main, model_dir_relative)
    print(f"RLAgent __main__: Attempting to load LSTMPredictor from: {lstm_model_path}")
    if os.path.exists(lstm_model_path):
        try:
            loaded_lstm_predictor = LSTMPredictor(input_seq_len=1, output_seq_len=1, n_features=1)
            loaded_lstm_predictor.load(lstm_model_path)
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
        predictor=loaded_lstm_predictor,
        meal_detector=None,
        rl_algorithm="SAC",
        rl_params={"learning_starts": 10, "verbose": 0}
    )
    print("\n--- Testing RLAgent with TRAINED PatternAdvisorAgent (internal model) ---")
    trained_advisor_model_path = os.path.join(project_root_main, "DiaGuardianAI", "models",
                                              "pattern_advisor_agent_model", "pattern_advisor_supervised.joblib")
    pattern_advisor_instance = None
    dummy_repo_for_loaded_advisor = None
    dummy_repo_for_loaded_advisor_path = None
    if not os.path.exists(trained_advisor_model_path):
        print(f"ERROR: Trained PatternAdvisorAgent model not found at {trained_advisor_model_path}")
        print("Please run DiaGuardianAI/examples/train_pattern_advisor.py first.")
        pattern_advisor_instance = None
    else:
        dummy_repo_for_loaded_advisor_path = "dummy_decision_agent_trained_advisor_repo.sqlite"
        if os.path.exists(dummy_repo_for_loaded_advisor_path):
            os.remove(dummy_repo_for_loaded_advisor_path)
        dummy_repo_for_loaded_advisor = RepositoryManager(db_path=dummy_repo_for_loaded_advisor_path)
        advisor_model_expected_state_dim = calculated_state_dim
        pattern_advisor_instance = PatternAdvisorAgent(
            state_dim=advisor_model_expected_state_dim,
            action_space="PredictActionCategory",
            pattern_repository=dummy_repo_for_loaded_advisor,
            learning_model_type="supervised_classifier"
        )
        try:
            pattern_advisor_instance.load(trained_advisor_model_path)
            print(f"RLAgent __main__: Successfully loaded trained PatternAdvisorAgent from {trained_advisor_model_path}")
        except Exception as e:
            print(f"RLAgent __main__: FAILED to load trained PatternAdvisorAgent: {e}")
            pattern_advisor_instance = None
    agent = RLAgent(
        state_dim=calculated_state_dim,
        action_space_definition=action_space_def_example,
        cgm_history_len=cgm_hist_len_example,
        prediction_horizon_len=pred_horizon_len_example,
        predictor=loaded_lstm_predictor,
        meal_detector=None,
        pattern_advisor=pattern_advisor_instance,
        rl_algorithm="SAC",
        rl_params={"learning_starts": 10, "verbose": 0}
    )
    print("RLAgent __main__: RLAgent re-initialized WITH PatternAdvisorAgent.")
    if loaded_lstm_predictor:
        print("RLAgent __main__: RLAgent initialized WITH LSTMPredictor.")
    else:
        print("RLAgent __main__: RLAgent initialized WITHOUT LSTMPredictor (loading failed or path not found).")
    dummy_patient_state_example = {
        'cgm': 120.0,
        'cgm_history': [120.0 - (i * 0.1) for i in range(cgm_hist_len_example)][::-1],
        'bolus_history': [0.0] * cgm_hist_len_example,
        'carbs_history': [0.0] * cgm_hist_len_example,
        'iob': 1.5,
        'cob': 20.0,
        'meal_announced': True,
        'announced_carbs': 30.0,
        'hour_of_day': 14,
        'day_of_week': 2,
        'stress_level': 0.2,
        'time_since_meal_announcement_minutes': 10.0
    }
    if pattern_advisor_instance:
        print(f"\nRLAgent __main__: Calling decide_action with TRAINED PatternAdvisor (using internal model). State: {dummy_patient_state_example}")
        action = agent.decide_action(dummy_patient_state_example)
        print(f"RLAgent __main__: RLAgent decided action: {action}")
    else:
        print("\nRLAgent __main__: Skipping test with trained PatternAdvisor as it failed to load.")
        action = agent.decide_action(dummy_patient_state_example)
        print(f"RLAgent __main__: RLAgent decided action (advisor load failed/skipped): {action}")
        print(f"RLAgent __main__: RLAgent decided action (advisor load failed/skipped): {action}")
    if dummy_repo_for_loaded_advisor is not None and hasattr(dummy_repo_for_loaded_advisor, 'conn') and dummy_repo_for_loaded_advisor.conn:
        dummy_repo_for_loaded_advisor.conn.close()
        print("RLAgent __main__: Closed DB connection for dummy_repo_for_loaded_advisor.")
    if dummy_repo_for_loaded_advisor_path is not None and os.path.exists(dummy_repo_for_loaded_advisor_path):
        os.remove(dummy_repo_for_loaded_advisor_path)
        print(f"RLAgent __main__: Cleaned up {dummy_repo_for_loaded_advisor_path}.")
    agent.save("./dummy_rl_agent_before_train_example")
    agent.load("./dummy_rl_agent_before_train_example")
    print(f"Agent saved and loaded from ./dummy_rl_agent_before_train_example")
    print("\n--- Example: Initiating Agent Training (Conceptual) ---")
    try:
        print("Calling agent.train_rl_model(total_timesteps=50)...")
        agent.train_rl_model(total_timesteps=50)
        agent.save("./dummy_rl_agent_after_train_example")
        print("Conceptual training finished and model saved to ./dummy_rl_agent_after_train_example")
    except NotImplementedError as e:
        print(f"Note: {e}")
    except Exception as e:
        print(f"Error during conceptual training example: {e}")
        print("This might be expected if the _MinimalGymEnv is not fully interactive for training yet.")
    print("\nRLAgent (DecisionAgent) example run complete.")