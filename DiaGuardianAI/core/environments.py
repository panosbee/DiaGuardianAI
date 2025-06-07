# DiaGuardianAI/core/environments.py
# Gymnasium-compatible environment for training RL agents.

import sys
import os
from typing import Dict, Any, Optional, List, Tuple, cast, TYPE_CHECKING

import gymnasium as gym
from gymnasium.spaces import Box, Space
import numpy as np
import pandas as pd # Added import

# Ensure the DiaGuardianAI package is discoverable
# This logic helps if the script is run directly or as part of the package.
# Standardize import path resolution
project_root_env = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root_env not in sys.path:
    sys.path.insert(0, project_root_env)

from DiaGuardianAI.core.base_classes import BaseSyntheticPatient
from DiaGuardianAI.predictive_models.model_zoo.lstm_predictor import LSTMPredictor

if TYPE_CHECKING:
    from DiaGuardianAI.agents.decision_agent import RLAgent


class DiaGuardianEnv(gym.Env):
    """
    A Gymnasium environment for training RL agents on the DiaGuardianAI
    synthetic patient model.
    """
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}

    def __init__(self,
                 patient: BaseSyntheticPatient,
                 agent_for_specs: 'RLAgent', # Use forward reference string for type hint
                 sim_config: Dict[str, Any]):
        """
        Initializes the DiaGuardian Environment.

        Args:
            patient (BaseSyntheticPatient): An instance of the synthetic patient.
            agent_for_specs (RLAgent): An RLAgent instance, used to define
                observation/action spaces and how to construct the state vector.
                Its internal observation_space_gym and action_space_gym must be initialized.
            sim_config (Dict[str, Any]): Configuration dictionary.
                Expected keys:
                "time_step_minutes" (int): Duration of each simulation step.
                "max_episode_steps" (int): Max steps per episode.
                "meal_schedule" (Optional[Dict[int, float]]): Time (minutes) to carb (g).
                "exercise_schedule" (Optional[Dict[int, Dict[str, Any]]]): Time (minutes) to exercise event.
                "reward_params" (Optional[Dict[str, float]]): Parameters for reward calculation.
        """
        super().__init__()

        self.patient = patient
        self.agent_for_specs = agent_for_specs
        self.sim_config = sim_config

        self.time_step_minutes: int = self.sim_config.get("time_step_minutes", 5)
        self.max_episode_steps: int = self.sim_config.get("max_episode_steps", 288) # Default 1 day

        # Spaces are defined by the agent's configuration. Assert they are initialized.
        if self.agent_for_specs.observation_space_gym is None:
            raise ValueError("RLAgent used for specs must have observation_space_gym initialized.")
        if self.agent_for_specs.action_space_gym is None:
            raise ValueError("RLAgent used for specs must have action_space_gym initialized.")
        
        self.observation_space: Space = self.agent_for_specs.observation_space_gym
        self.action_space: Space = self.agent_for_specs.action_space_gym
        
        self.cgm_history_len: int = self.agent_for_specs.cgm_history_len
        self.cgm_history_buffer: List[float] = []
        self.bolus_history_buffer: List[float] = [] # New buffer for bolus history
        self.carbs_history_buffer: List[float] = [] # New buffer for carbs history
        # Assuming history length for bolus/carbs is same as CGM for predictor input alignment
        self.action_carbs_history_len: int = self.cgm_history_len

        self.current_episode_step: int = 0
        self.current_episode_time_minutes: int = 0

        # State for post-meal spike penalty
        self.last_meal_time_minutes: Optional[int] = None
        self.cgm_at_last_meal_ingestion: Optional[float] = None
 
        default_reward_params = {
            "TARGET_RANGE_LOW": 70.0, "TARGET_RANGE_HIGH": 180.0,
            "HYPO_THRESHOLD": 70.0, "SEVERE_HYPO_THRESHOLD": 54.0,
            "HYPER_THRESHOLD": 180.0, "SEVERE_HYPER_THRESHOLD": 250.0,
            "REWARD_IN_RANGE": 1.0,
            "PENALTY_HYPO": -2.0,
            "PENALTY_SEVERE_HYPO": -10.0,
            "PENALTY_HYPER": -1.0,
            "PENALTY_SEVERE_HYPER": -5.0,
            "MAX_HYPO_DEVIATION_SCALE_FACTOR": 5.0,
            "MAX_HYPER_DEVIATION_SCALE_FACTOR": 2.0,
            "PENALTY_GLUCOSE_VARIABILITY": -0.1, # Penalty factor for CGM std dev
            "VARIABILITY_THRESHOLD_STD": 15.0, # Std dev above which penalty applies
            "PENALTY_RAPID_ROC_DROP": -0.5, # Penalty for rapid drops
            "PENALTY_RAPID_ROC_RISE": -0.3, # Penalty for rapid rises
            "ROC_THRESHOLD_MG_DL_PER_STEP": 10.0, # mg/dL change per step to trigger RoC penalty
            "PENALTY_POST_MEAL_SPIKE": -2.0,
            "POST_MEAL_SPIKE_THRESHOLD_MG_DL": 50.0,
            "POST_MEAL_WINDOW_MINUTES": 120,
            "OPTIMAL_RANGE_LOW": 90.0,
            "OPTIMAL_RANGE_HIGH": 140.0,
            "MAX_SHAPED_TIR_REWARD": 1.0,
            "PENALTY_SMALL_CORRECTION_BOLUS": -0.2, # New: Penalty for small, potentially unnecessary boluses
            "SMALL_CORRECTION_BOLUS_MAX_U": 0.5,    # New: Max U for a bolus to be considered "small"
            "SMALL_CORRECTION_GLUCOSE_OK_LOW": 80.0, # New: Lower glucose bound where small correction is penalized
            "SMALL_CORRECTION_GLUCOSE_OK_HIGH": 160.0 # New: Upper glucose bound where small correction is penalized
        }
        # Start with defaults, then update with any user-provided params
        self.reward_params = default_reward_params.copy()
        user_reward_params = self.sim_config.get("reward_params", {})
        self.reward_params.update(user_reward_params)
        
        # Now, these keys are guaranteed to exist in self.reward_params
        self.reward_params["MAX_HYPO_DEVIATION_SCALE"] = max(1e-6, self.reward_params["HYPO_THRESHOLD"] - self.reward_params["SEVERE_HYPO_THRESHOLD"])
        self.reward_params["MAX_HYPER_DEVIATION_SCALE"] = max(1e-6, self.reward_params["SEVERE_HYPER_THRESHOLD"] - self.reward_params["HYPER_THRESHOLD"])

    def _get_observation(self) -> np.ndarray:
        current_cgm = self.patient.get_cgm_reading()
        internal_states = self.patient.get_internal_states()

        self.cgm_history_buffer.append(current_cgm)
        if len(self.cgm_history_buffer) > self.cgm_history_len:
            self.cgm_history_buffer.pop(0)
        
        padded_cgm_history = list(self.cgm_history_buffer)
        if len(padded_cgm_history) < self.cgm_history_len:
            padding_val = current_cgm if not padded_cgm_history else padded_cgm_history[0]
            padding = [padding_val] * (self.cgm_history_len - len(padded_cgm_history))
            padded_cgm_history = padding + padded_cgm_history
        
        # The observation vector should be what the agent's policy network expects as input.
        # RLAgent._define_state constructs this vector.
        # We assume that for training, the agent might not have its own predictor's output
        # available *before* this observation is formed, or that _define_state handles None predictions.
        
        # Generate predictions if the agent_for_specs has a predictor
        mean_glucose_predictions: Optional[List[float]] = None
        std_dev_glucose_predictions: Optional[List[float]] = None # For UQ
        pred_dict: Optional[Dict[str, Any]] = None

        if self.agent_for_specs.predictor:
            predictor_instance = self.agent_for_specs.predictor
            required_pred_input_len = getattr(predictor_instance, 'input_seq_len', self.cgm_history_len)

            # Pad bolus and carbs history similar to CGM history
            padded_bolus_history = list(self.bolus_history_buffer)
            if len(padded_bolus_history) < self.action_carbs_history_len:
                padding = [0.0] * (self.action_carbs_history_len - len(padded_bolus_history))
                padded_bolus_history = padding + padded_bolus_history
            
            padded_carbs_history = list(self.carbs_history_buffer)
            if len(padded_carbs_history) < self.action_carbs_history_len:
                padding = [0.0] * (self.action_carbs_history_len - len(padded_carbs_history))
                padded_carbs_history = padding + padded_carbs_history

            if isinstance(predictor_instance, LSTMPredictor):
                # Ensure histories are sliced to the length expected by LSTMPredictor's input_seq_len
                start_idx_cgm = max(0, len(padded_cgm_history) - required_pred_input_len)
                start_idx_bolus = max(0, len(padded_bolus_history) - required_pred_input_len)
                start_idx_carbs = max(0, len(padded_carbs_history) - required_pred_input_len)

                predictor_input_df = pd.DataFrame({
                    'cgm_mg_dl': padded_cgm_history[start_idx_cgm:],
                    'bolus_U': padded_bolus_history[start_idx_bolus:],
                    'carbs_g': padded_carbs_history[start_idx_carbs:]
                })
                # Ensure all columns in DF have the same length, matching required_pred_input_len
                # This might require more careful padding/slicing if original histories differ in length significantly
                # For now, assume they are kept aligned and LSTMPredictor.predict handles shorter than input_seq_len
                
                try:
                    pred_dict = predictor_instance.predict(predictor_input_df)
                except Exception as e:
                    print(f"DiaGuardianEnv: Error getting predictions from LSTMPredictor: {e}")
                    pred_dict = None
            else: # Generic predictor expecting a numpy array (e.g., just CGM history)
                predictor_input_sequence = np.array(padded_cgm_history, dtype=np.float32)
                if predictor_input_sequence.ndim == 1:
                    predictor_input_for_model = predictor_input_sequence.reshape(1, -1, 1) # (batch, seq, features=1)
                elif predictor_input_sequence.ndim == 2 and predictor_input_sequence.shape[0] == 1:
                    predictor_input_for_model = predictor_input_sequence.reshape(1, -1, 1)
                else:
                    print(f"Warning: Unexpected shape for generic predictor input sequence: {predictor_input_sequence.shape}")
                    predictor_input_for_model = np.zeros((1, self.cgm_history_len, 1), dtype=np.float32)
                try:
                    pred_dict = predictor_instance.predict(predictor_input_for_model)
                except Exception as e:
                    print(f"DiaGuardianEnv: Error getting predictions from generic predictor: {e}")
                    pred_dict = None
            
            if pred_dict:
                mean_glucose_predictions = pred_dict.get("mean")
                std_dev_glucose_predictions = pred_dict.get("std_dev")
            else: # Ensure they are None if pred_dict is None
                mean_glucose_predictions = None # Fallback
                std_dev_glucose_predictions = None # Fallback

        observation_vector = self.agent_for_specs._define_state(
            current_cgm=current_cgm,
            cgm_history=padded_cgm_history,
            iob=internal_states.get("iob", 0.0), # Use total IOB from patient model
            cob=internal_states.get("cob", 0.0),
            glucose_predictions=mean_glucose_predictions, # Pass mean predictions
            glucose_predictions_std_dev=std_dev_glucose_predictions, # Pass std_dev predictions
            meal_announced=internal_states.get("meal_announced_this_step", False),
            announced_carbs=internal_states.get("announced_carbs_this_step", 0.0)
        )
        return observation_vector.astype(self.observation_space.dtype)

    def _calculate_reward(self, cgm_after_action: float, carbs_ingested_this_step: float, action_taken: Dict[str, float]) -> float: # Added action_taken
        rp = self.reward_params
        reward = 0.0

        # TIR and Glycemic Excursion Penalties
        if cgm_after_action < rp["HYPO_THRESHOLD"]:
            if cgm_after_action < rp["SEVERE_HYPO_THRESHOLD"]:
                reward += rp["PENALTY_SEVERE_HYPO"]
                deviation = rp["SEVERE_HYPO_THRESHOLD"] - cgm_after_action
                reward += rp["PENALTY_HYPO"] * (deviation / rp["MAX_HYPO_DEVIATION_SCALE"]) * rp["MAX_HYPO_DEVIATION_SCALE_FACTOR"]
            else:
                deviation = rp["HYPO_THRESHOLD"] - cgm_after_action
                reward += rp["PENALTY_HYPO"] * (deviation / rp["MAX_HYPO_DEVIATION_SCALE"])
        elif cgm_after_action > rp["HYPER_THRESHOLD"]:
            if cgm_after_action > rp["SEVERE_HYPER_THRESHOLD"]:
                reward += rp["PENALTY_SEVERE_HYPER"]
                deviation = cgm_after_action - rp["SEVERE_HYPER_THRESHOLD"]
                reward += rp["PENALTY_HYPER"] * (deviation / rp["MAX_HYPER_DEVIATION_SCALE"]) * rp["MAX_HYPER_DEVIATION_SCALE_FACTOR"]
            else:
                deviation = cgm_after_action - rp["HYPER_THRESHOLD"]
                reward += rp["PENALTY_HYPER"] * (deviation / rp["MAX_HYPER_DEVIATION_SCALE"])
        elif rp["OPTIMAL_RANGE_LOW"] <= cgm_after_action <= rp["OPTIMAL_RANGE_HIGH"]:
            # Inside optimal sub-range
            reward = rp["MAX_SHAPED_TIR_REWARD"]
        elif rp["TARGET_RANGE_LOW"] <= cgm_after_action < rp["OPTIMAL_RANGE_LOW"]:
            # In lower part of target range, but below optimal
            # Linearly scale reward from 0 at TARGET_RANGE_LOW to MAX_SHAPED_TIR_REWARD at OPTIMAL_RANGE_LOW
            reward = rp["MAX_SHAPED_TIR_REWARD"] * \
                     ((cgm_after_action - rp["TARGET_RANGE_LOW"]) /
                      max(1e-6, rp["OPTIMAL_RANGE_LOW"] - rp["TARGET_RANGE_LOW"]))
        elif rp["OPTIMAL_RANGE_HIGH"] < cgm_after_action <= rp["TARGET_RANGE_HIGH"]:
            # In upper part of target range, but above optimal
            # Linearly scale reward from MAX_SHAPED_TIR_REWARD at OPTIMAL_RANGE_HIGH to 0 at TARGET_RANGE_HIGH
            reward = rp["MAX_SHAPED_TIR_REWARD"] * \
                     (1.0 - (cgm_after_action - rp["OPTIMAL_RANGE_HIGH"]) /
                             max(1e-6, rp["TARGET_RANGE_HIGH"] - rp["OPTIMAL_RANGE_HIGH"]))
        # If cgm_after_action is exactly TARGET_RANGE_LOW or TARGET_RANGE_HIGH, reward will be 0 from these conditions.
        # If it's outside TIR, the hypo/hyper penalties above will apply.
        # Ensure reward doesn't go below 0 from TIR shaping if MAX_SHAPED_TIR_REWARD is positive.
        reward = max(0, reward) if rp["MAX_SHAPED_TIR_REWARD"] > 0 and \
                                   rp["TARGET_RANGE_LOW"] <= cgm_after_action <= rp["TARGET_RANGE_HIGH"] else reward


        # Glucose Variability Penalty (based on standard deviation of recent CGM history)
        if len(self.cgm_history_buffer) >= 2: # Need at least 2 points to calculate std dev
            cgm_std_dev = np.std(self.cgm_history_buffer)
            if cgm_std_dev > rp["VARIABILITY_THRESHOLD_STD"]:
                # Penalize proportionally to how much std dev exceeds threshold
                variability_penalty_magnitude = (cgm_std_dev - rp["VARIABILITY_THRESHOLD_STD"]) / rp["VARIABILITY_THRESHOLD_STD"]
                reward += rp["PENALTY_GLUCOSE_VARIABILITY"] * variability_penalty_magnitude

        # Rate of Change (RoC) Penalty
        if len(self.cgm_history_buffer) >= 2:
            previous_cgm = self.cgm_history_buffer[-2] # Second to last reading
            roc = cgm_after_action - previous_cgm # Change over one time step
            
            if roc < -rp["ROC_THRESHOLD_MG_DL_PER_STEP"]: # Rapid drop
                # Penalize proportionally to how much drop exceeds threshold
                roc_drop_penalty_magnitude = (abs(roc) - rp["ROC_THRESHOLD_MG_DL_PER_STEP"]) / rp["ROC_THRESHOLD_MG_DL_PER_STEP"]
                reward += rp["PENALTY_RAPID_ROC_DROP"] * roc_drop_penalty_magnitude
            elif roc > rp["ROC_THRESHOLD_MG_DL_PER_STEP"]: # Rapid rise
                # Penalize proportionally to how much rise exceeds threshold
                roc_rise_penalty_magnitude = (roc - rp["ROC_THRESHOLD_MG_DL_PER_STEP"]) / rp["ROC_THRESHOLD_MG_DL_PER_STEP"]
                reward += rp["PENALTY_RAPID_ROC_RISE"] * roc_rise_penalty_magnitude

        # Post-Meal Spike Penalty
        if carbs_ingested_this_step > 0: # A meal was just ingested in the step leading to cgm_after_action
            self.last_meal_time_minutes = self.current_episode_time_minutes # Time of current step
            self.cgm_at_last_meal_ingestion = cgm_after_action # CGM right after meal action
        
        if self.last_meal_time_minutes is not None and self.cgm_at_last_meal_ingestion is not None:
            time_since_last_meal = self.current_episode_time_minutes - self.last_meal_time_minutes
            if 0 < time_since_last_meal <= rp["POST_MEAL_WINDOW_MINUTES"]:
                cgm_spike = cgm_after_action - self.cgm_at_last_meal_ingestion
                if cgm_spike > rp["POST_MEAL_SPIKE_THRESHOLD_MG_DL"]:
                    # Penalize proportionally to how much spike exceeds threshold
                    spike_penalty_magnitude = (cgm_spike - rp["POST_MEAL_SPIKE_THRESHOLD_MG_DL"]) / rp["POST_MEAL_SPIKE_THRESHOLD_MG_DL"]
                    reward += rp["PENALTY_POST_MEAL_SPIKE"] * spike_penalty_magnitude
            elif time_since_last_meal > rp["POST_MEAL_WINDOW_MINUTES"]:
                # Reset if outside window, to only consider the first significant spike post-meal
                self.last_meal_time_minutes = None
                self.cgm_at_last_meal_ingestion = None

        # User Burden Penalty: Small unnecessary correction boluses
        bolus_u_taken = action_taken.get("bolus_u", 0.0)
        if 0 < bolus_u_taken <= rp["SMALL_CORRECTION_BOLUS_MAX_U"]:
            if rp["SMALL_CORRECTION_GLUCOSE_OK_LOW"] <= cgm_after_action <= rp["SMALL_CORRECTION_GLUCOSE_OK_HIGH"]:
                reward += rp["PENALTY_SMALL_CORRECTION_BOLUS"]
                
        return float(reward)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action_dict = {}
        if len(action) == len(self.agent_for_specs.action_keys_ordered):
            for i, key in enumerate(self.agent_for_specs.action_keys_ordered):
                action_dict[key] = float(action[i])
        else:
            print(f"DiaGuardianEnv ERROR: Action length mismatch. Expected {len(self.agent_for_specs.action_keys_ordered)}, got {len(action)}")
            action_dict = {key: 0.0 for key in self.agent_for_specs.action_keys_ordered}

        bolus_insulin = action_dict.get("bolus_u", 0.0)
        basal_rate_u_hr = action_dict.get("basal_rate_u_hr")
        
        basal_for_step = 0.0
        if basal_rate_u_hr is not None:
            basal_for_step = basal_rate_u_hr * (self.time_step_minutes / 60.0)

        carbs_ingested = self.sim_config.get("meal_schedule", {}).get(self.current_episode_time_minutes, 0.0)
        protein_ingested = self.sim_config.get("protein_schedule", {}).get(self.current_episode_time_minutes, 0.0)
        fat_ingested = self.sim_config.get("fat_schedule", {}).get(self.current_episode_time_minutes, 0.0)
        exercise_event = self.sim_config.get("exercise_schedule", {}).get(self.current_episode_time_minutes, None)

        carbs_details_for_step: Optional[Dict[str, Any]] = None
        if carbs_ingested > 0:
            carbs_details_for_step = {
                "grams": carbs_ingested,
                "gi_factor": self.sim_config.get("meal_gi_factors", {}).get(self.current_episode_time_minutes, 1.0)
            }

        # Update bolus and carbs history buffers
        self.bolus_history_buffer.append(bolus_insulin) # Log the bolus applied
        if len(self.bolus_history_buffer) > self.action_carbs_history_len:
            self.bolus_history_buffer.pop(0)

        self.carbs_history_buffer.append(carbs_ingested) # Log the carbs ingested
        if len(self.carbs_history_buffer) > self.action_carbs_history_len:
            self.carbs_history_buffer.pop(0)

        self.patient.step(
            basal_insulin=basal_for_step, bolus_insulin=bolus_insulin,
            carbs_details=carbs_details_for_step,
            protein_ingested=protein_ingested,
            fat_ingested=fat_ingested, exercise_event=exercise_event
        )

        observation = self._get_observation()
        cgm_after_action = self.patient.get_cgm_reading() # Get CGM for reward *after* patient step
        reward = self._calculate_reward(cgm_after_action, carbs_ingested, action_dict)
 
        self.current_episode_step += 1
        self.current_episode_time_minutes += self.time_step_minutes

        done = self.current_episode_step >= self.max_episode_steps
        truncated = False 

        info = {"cgm": cgm_after_action, "iob": self.patient.get_internal_states().get("iob",0), "cob": self.patient.get_internal_states().get("cob",0)}
        return observation, reward, done, truncated, info

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        initial_patient_state_options = options.get("patient_initial_state", None) if options else None
        
        # Ensure BaseSyntheticPatient has a reset method
        if not hasattr(self.patient, 'reset') or not callable(getattr(self.patient, 'reset')):
            raise NotImplementedError("The patient model must implement a 'reset' method.")
        self.patient.reset(initial_state_override=initial_patient_state_options)

        self.cgm_history_buffer = []
        self.bolus_history_buffer = [] # Reset bolus history
        self.carbs_history_buffer = [] # Reset carbs history
        self.current_episode_step = 0
        self.current_episode_time_minutes = 0
        self.last_meal_time_minutes = None
        self.cgm_at_last_meal_ingestion = None
        
        initial_cgm = self.patient.get_cgm_reading()
        initial_bolus = 0.0 # Assuming no bolus at reset
        initial_carbs = 0.0 # Assuming no carbs at reset

        for _ in range(self.cgm_history_len): # Use self.cgm_history_len for all three
            self.cgm_history_buffer.append(initial_cgm)
            self.bolus_history_buffer.append(initial_bolus)
            self.carbs_history_buffer.append(initial_carbs)

        observation = self._get_observation()
        info = {"initial_cgm": initial_cgm}
        return observation, info

    def render(self, mode='human'):
        if mode == 'human':
            cgm = self.patient.get_cgm_reading()
            iob = self.patient.get_internal_states().get("iob",0)
            cob = self.patient.get_internal_states().get("cob",0)
            print(f"Step: {self.current_episode_step}, Time: {self.current_episode_time_minutes}m, CGM: {cgm:.2f}, IOB: {iob:.2f}, COB: {cob:.2f}")
        elif mode == 'rgb_array':
            pass 
        # else: # gymnasium.Env does not have a render method with mode by default
        #     pass

    def close(self):
        pass

if __name__ == '__main__':
    print("DiaGuardianEnv - Example Usage")

    # Attempt to import concrete classes needed for the example
    try:
        from DiaGuardianAI.data_generation.synthetic_patient_model import SyntheticPatient
        # RLAgent is already imported at the top
    except ImportError as e:
        print(f"Could not import SyntheticPatient or RLAgent for example: {e}")
        print("Please ensure you are running this script from the project root or DiaGuardianAI is in PYTHONPATH.")
        sys.exit(1)

    # 1. Define dummy patient parameters
    patient_params = {
        "name": "TestPatient",
        "glucose_history_mg_dl": [100.0] * 288, # Initial history
        "insulin_history_U": [0.0] * 288,
        "carb_history_g": [0.0] * 288,
        "initial_state": {"G": 100.0, "I": 10.0, "X": 0.0, "Q1": 100.0, "Q2": 100.0},
        "body_weight_kg": 70,
        "insulin_sensitivity_factor_mg_dl_U": 50,
        "carb_insulin_ratio_g_U": 10,
        "basal_profile_U_hr": {"00:00": 0.8, "06:00": 0.9, "12:00": 0.85, "18:00": 0.75},
        "target_glucose_mg_dl": 110,
        "simulation_time_step_minutes": 5, # Match env time step
        "sensor_noise_type": "gaussian",
        "sensor_noise_param": 2.0, # std dev for gaussian noise
        "action_latency_minutes": 0,
        "insulin_action_duration_hours": 4.0,
        "carb_absorption_duration_minutes": 180,
        "default_phys_params": {} # Use internal defaults
    }
    dummy_patient = SyntheticPatient(params=patient_params)

    # 2. Define dummy RLAgent parameters and initialize it
    # State dim needs to match RLAgent's _define_state calculation
    # Current RLAgent state_dim (as of last modification): 39
    # 1 (cgm) + 12 (hist) + 1 (iob) + 1 (cob) + 6 (pred_mean) + 6 (pred_std) +
    # 2 (meal_flags) + 2 (slopes) + 2 (meal_detector_out) + 4 (time) + 1 (stress) + 1 (time_since_meal)
    agent_state_dim = 39
    agent_action_space_def = {
        "bolus_u": {"low": 0.0, "high": 15.0},
        "basal_rate_u_hr": {"low": 0.0, "high": 3.0}
    }
    dummy_agent_for_specs = RLAgent(
        state_dim=agent_state_dim,
        action_space_definition=agent_action_space_def,
        cgm_history_len=12, # Must match RLAgent's expectation for state
        prediction_horizon_len=6, # Must match RLAgent's expectation for state
        rl_algorithm="SAC" # Or PPO, doesn't matter much for just getting spaces
    )

    # 3. Define simulation configuration for the environment
    env_sim_config = {
        "time_step_minutes": 5,
        "max_episode_steps": 60, # Run for 5 hours (60 steps * 5 min)
        "meal_schedule": {
            0: 50.0, # 50g carbs at time 0
            180: 30.0 # 30g carbs at time 180 minutes (3 hours)
        },
        "reward_params": { # Using default reward params from DiaGuardianEnv for now
            # "PENALTY_SEVERE_HYPO": -20.0 # Example of overriding a default
        }
    }

    try:
        env = DiaGuardianEnv(
            patient=dummy_patient,
            agent_for_specs=dummy_agent_for_specs,
            sim_config=env_sim_config
        )
        obs, info = env.reset()
        print(f"Initial Obs: {obs[:5]}... (len: {len(obs)}), Info: {info}")

        total_reward = 0
        for step_num in range(env_sim_config["max_episode_steps"]):
            action = env.action_space.sample() # Agent takes random actions
            obs, reward, done, truncated, info = env.step(action)
            env.render() # Prints CGM, IOB, COB
            print(f"Step {step_num+1}: Action: {action}, Reward: {reward:.4f}, CGM: {info.get('cgm', 'N/A'):.2f}")
            # print(f"   Full Obs: {obs[:5]}...") # Optionally print part of observation
            total_reward += reward
            if done or truncated:
                print("Episode finished.")
                break
        env.close()
        print(f"\nTotal reward for episode: {total_reward:.4f}")

    except Exception as e:
        print(f"Error in DiaGuardianEnv example run: {e}")
        import traceback
        traceback.print_exc()

    print("\nDiaGuardianEnv example run complete.")