# DiaGuardianAI - Train PPO Agent Example
# This script demonstrates how to train an RLAgent using PPO from Stable Baselines3
# with the DiaGuardianEnv.

import sys
import os
import numpy as np
from typing import Optional # Added import

# Ensure the DiaGuardianAI package is discoverable
# Define project_root reliably
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from DiaGuardianAI.core.environments import DiaGuardianEnv
from DiaGuardianAI.data_generation.synthetic_patient_model import SyntheticPatient
from DiaGuardianAI.agents.decision_agent import RLAgent
from DiaGuardianAI.predictive_models.model_zoo.lstm_predictor import LSTMPredictor
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env # To verify env compatibility

def train_ppo_diabetes_agent():
    print("--- Starting PPO Agent Training Example ---")

    # 1. Patient Setup
    patient_params = {
        "name": "Adult#PPOTrain001", "base_glucose": 100.0, "isf": 50.0, "cr": 10.0,
        "cir": 10.0, "basal_insulin_rate": 0.8, "weight_kg": 70.0, # Slightly lower basal
        "initial_glucose": 140.0, "time_step_minutes": 5
    }
    try:
        patient = SyntheticPatient(params=patient_params)
    except Exception as e:
        print(f"Error initializing SyntheticPatient: {e}")
        return

    # 2. Agent Setup (for specs for the environment)
    cgm_hist_len = 24 # Changed to 24 to match LSTMPredictor's expected input_seq_len
    pred_horizon = 6
    # state_dim calculation based on RLAgent._define_state:
    # 1 (cgm) + cgm_hist_len + 1 (iob) + 1 (cob) + pred_horizon (mean) + pred_horizon (std_dev) +
    # 2 (slopes) + 2 (meal_detector_features) + 4 (time_features) + 1 (stress) +
    # 1 (time_since_meal) + 1 (meal_announced_flag) + 1 (announced_carbs)
    state_dim = cgm_hist_len + (2 * pred_horizon) + 15 # Corrected calculation: 12 + 12 + 15 = 39
    print(f"Calculated state_dim for RLAgent: {state_dim}")
 
    action_space_def = {
        "bolus_u": {"low": 0.0, "high": 10.0},
        "basal_rate_u_hr": {"low": 0.0, "high": 3.0}
    }
    # Predictor for RLAgent (used by DiaGuardianEnv._get_observation to generate part of the state)
    # LSTMPredictor expects: input_seq_len, output_seq_len, n_features, hidden_units, num_layers
    # For this example, let's assume the predictor uses cgm_hist_len as input_seq_len and n_features=1 (just CGM)
    # or n_features=3 if we provide dummy bolus/carbs history.
    # The LSTMPredictor.predict method expects a DataFrame.
    # DiaGuardianEnv._get_observation will need to prepare this DataFrame.
    
    # For simplicity in this training script, if the LSTMPredictor is used by the env,
    # it will need historical bolus and carbs. We can provide dummy ones or load a real one.
    # Let's load the example LSTM predictor if available.
    loaded_lstm_predictor: Optional[LSTMPredictor] = None
    model_dir_relative = os.path.join("DiaGuardianAI", "models", "lstm_predictor_example_run")
    lstm_model_path = os.path.join(project_root, model_dir_relative) # project_root from top of file

    print(f"Train PPO: Attempting to load LSTMPredictor from: {lstm_model_path}")
    if os.path.exists(lstm_model_path):
        try:
            loaded_lstm_predictor = LSTMPredictor(input_seq_len=1, output_seq_len=1, n_features=1) # Dummy init
            loaded_lstm_predictor.load(lstm_model_path)
            print(f"Train PPO: LSTMPredictor loaded successfully for agent_for_env_specs.")
            # The RLAgent's _define_state method will use its own self.prediction_horizon_len
            # for padding/truncating predictions from any predictor. So, we don't need to
            # change pred_horizon or state_dim here based on the loaded LSTM.
            # The RLAgent will use the pred_horizon it's initialized with (6 in this case).
            # if loaded_lstm_predictor.output_seq_len != pred_horizon:
            #     print(f"Warning: pred_horizon ({pred_horizon}) differs from loaded LSTM output_seq_len ({loaded_lstm_predictor.output_seq_len}). RLAgent will use its configured horizon.")
                # pred_horizon = loaded_lstm_predictor.output_seq_len # DO NOT CHANGE
                # state_dim = cgm_hist_len + (2 * pred_horizon) + 15 # DO NOT RECALCULATE
                # print(f"Re-calculated state_dim: {state_dim}")

        except Exception as e:
            print(f"Train PPO: Failed to load LSTMPredictor: {e}. Using None for agent's predictor.")
            loaded_lstm_predictor = None
    else:
        print(f"Train PPO: LSTMPredictor model path not found: {lstm_model_path}. Using None for agent's predictor.")
        loaded_lstm_predictor = None

    agent_for_env_specs = RLAgent(
        state_dim=state_dim, action_space_definition=action_space_def,
        predictor=loaded_lstm_predictor, # Pass the loaded or None predictor
        cgm_history_len=cgm_hist_len, prediction_horizon_len=pred_horizon
    )

    # 3. Environment Configuration
    env_sim_config = {
        "time_step_minutes": 5,
        "max_episode_steps": 288, # 1 day per episode for training
        "meal_schedule": {
            60 * 8: 50.0,  60 * 13: 70.0, 60 * 19: 60.0
        },
        "exercise_schedule": {
            60 * 16: {"duration_minutes": 30, "intensity_factor": 1.0}
        },
        "reward_params": {} # Use default reward params
    }

    # 4. Create and Check DiaGuardianEnv
    try:
        train_env = DiaGuardianEnv(
            patient=patient,
            agent_for_specs=agent_for_env_specs,
            sim_config=env_sim_config
        )
        print("\nDiaGuardianEnv created for training.")
        # It's a good practice to check your custom environment with SB3's checker
        print("Running Stable Baselines3 environment checker...")
        check_env(train_env)
        print("Environment check passed successfully.")
    except Exception as e:
        print(f"Error creating or checking DiaGuardianEnv: {e}")
        return

    # 5. PPO Model Setup and Training
    # Define PPO hyperparameters
    # These are example values and should be tuned.
    ppo_hyperparams = {
        "policy": "MlpPolicy",
        "learning_rate": 3e-4,
        "n_steps": 2048, # Number of steps to run for each environment per update
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.0,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "verbose": 1, # Set to 1 to see training progress
        "tensorboard_log": "./ppo_diaguardian_tensorboard/"
    }

    model_save_path = "./trained_ppo_diaguardian_agent"
    
    try:
        print(f"\n--- Initializing PPO Model ({ppo_hyperparams['policy']}) ---")
        # If continuing training, load model: model = PPO.load(model_save_path, env=train_env)
        # For a new training run:
        ppo_model = PPO(env=train_env, **ppo_hyperparams)

        total_training_timesteps = 20000 # Example: short training run
        print(f"\n--- Starting PPO Training for {total_training_timesteps} Timesteps ---")
        ppo_model.learn(total_timesteps=total_training_timesteps, progress_bar=True)
        
        print("\n--- PPO Training Complete ---")
        ppo_model.save(model_save_path)
        print(f"Trained PPO model saved to {model_save_path}.zip")

    except Exception as e:
        print(f"Error during PPO model setup or training: {e}")
        import traceback
        traceback.print_exc()
        return

    # 6. (Optional) Test the trained agent
    print("\n--- Testing Trained Agent (Conceptual) ---")
    # Load the trained PPO model directly (it is the agent policy)
    # For actual deployment/use, you might load its weights into your RLAgent structure,
    # or use the SB3 model directly for predictions if RLAgent is adapted.
    
    # For now, let's re-use the train_env to see a few steps with the trained policy.
    # In a real scenario, you'd use a separate evaluation environment.
    try:
        loaded_ppo_model = PPO.load(model_save_path, env=train_env)
        obs, info = train_env.reset()
        print("Running a few steps with the trained PPO model:")
        for i in range(20): # Run 20 steps
            action, _states = loaded_ppo_model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = train_env.step(action)
            train_env.render()
            print(f"  Trained Action: {action}, Reward: {reward:.2f}, CGM: {info.get('cgm', 'N/A'):.2f}")
            if done or truncated:
                print("Episode finished during test run.")
                obs, info = train_env.reset() # Reset for next potential loop if testing multiple episodes
    except Exception as e:
        print(f"Error during testing trained agent: {e}")

    train_env.close()
    print("\nPPO Agent training example finished.")

if __name__ == '__main__':
    train_ppo_diabetes_agent()