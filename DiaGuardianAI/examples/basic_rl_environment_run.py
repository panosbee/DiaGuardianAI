# DiaGuardianAI - Basic RL Environment Run Example
# This script demonstrates how to set up and run the DiaGuardianEnv.

import sys
import os
import numpy as np

# Ensure the DiaGuardianAI package is discoverable
if __package__ is None or __package__ == '':
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from DiaGuardianAI.core.environments import DiaGuardianEnv
from DiaGuardianAI.data_generation.synthetic_patient_model import SyntheticPatient
from DiaGuardianAI.agents.decision_agent import RLAgent
from DiaGuardianAI.predictive_models.model_zoo.lstm_predictor import LSTMPredictor

def run_basic_rl_env():
    print("--- Starting Basic RL Environment Example ---")

    # 1. Patient Setup (similar to basic_simulation_run.py)
    patient_params = {
        "name": "Adult#EnvTest001", "base_glucose": 100.0, "isf": 50.0, "cr": 10.0,
        "cir": 10.0, "basal_insulin_rate": 1.0, "weight_kg": 70.0,
        "initial_glucose": 120.0, "time_step_minutes": 5
        # Add other necessary params from SyntheticPatientModel if defaults are not sufficient
    }
    try:
        patient = SyntheticPatient(params=patient_params)
    except Exception as e:
        print(f"Error initializing SyntheticPatient: {e}")
        return

    # 2. Agent Setup (for specs - observation/action space, state definition)
    cgm_hist_len_for_agent = 12
    pred_horizon_for_agent_state = 6
    calculated_agent_state_dim = 1 + cgm_hist_len_for_agent + 1 + 1 + pred_horizon_for_agent_state + 2 # 23

    action_space_definition = {
        "bolus_u": {"low": 0.0, "high": 10.0},
        "basal_rate_u_hr": {"low": 0.0, "high": 3.0}
    }

    # Optional: Setup a predictor if RLAgent's _define_state expects it or for consistency
    # For this env test, RLAgent's _define_state is called with glucose_predictions=None
    # so a live predictor isn't strictly needed for the agent_for_specs if its _define_state handles None.
    # However, RLAgent __init__ might still expect a predictor instance.
    example_predictor = LSTMPredictor(
        input_dim=1, hidden_dim=32, num_layers=2, output_horizon_steps=pred_horizon_for_agent_state
    )

    agent_for_specs = RLAgent(
        state_dim=calculated_agent_state_dim,
        action_space_definition=action_space_definition,
        predictor=example_predictor, # Pass a predictor instance
        cgm_history_len=cgm_hist_len_for_agent,
        prediction_horizon_len=pred_horizon_for_agent_state
    )

    # 3. Environment Configuration
    env_sim_config = {
        "time_step_minutes": 5,
        "max_episode_steps": 50, # Short episode for testing
        "meal_schedule": { # Times are in minutes from start of episode
            30: 30.0,   # 30g carbs at 30 minutes
            90: 40.0    # 40g carbs at 90 minutes (if episode is long enough)
        },
        "exercise_schedule": {
            60: {"duration_minutes": 20, "intensity_factor": 1.0} # Exercise at 60 mins
        },
        "reward_params": {} # Use default reward params in DiaGuardianEnv
    }

    # 4. Create DiaGuardianEnv
    try:
        env = DiaGuardianEnv(
            patient=patient,
            agent_for_specs=agent_for_specs,
            sim_config=env_sim_config
        )
        print(f"\nDiaGuardianEnv created.")
        print(f"Observation Space: {env.observation_space}")
        print(f"Action Space: {env.action_space}")
    except Exception as e:
        print(f"Error creating DiaGuardianEnv: {e}")
        return

    # 5. Test Environment Interaction
    print("\n--- Testing Environment Interaction ---")
    try:
        obs, info = env.reset()
        print(f"Initial Observation shape: {obs.shape}, dtype: {obs.dtype}")
        print(f"Initial Info: {info}")
        env.render()

        for i in range(15): # Run for a few steps
            action = env.action_space.sample() # Sample a random action
            print(f"\nStep {i+1}: Taking action: {action}")
            
            obs, reward, done, truncated, info = env.step(action)
            
            print(f"  Observation shape: {obs.shape}, dtype: {obs.dtype}")
            print(f"  Reward: {reward:.4f}")
            print(f"  Done: {done}, Truncated: {truncated}")
            print(f"  Info: {info}")
            env.render(mode='human') # Render human-readable output

            if done or truncated:
                print(f"Episode finished at step {i+1}.")
                break
        
        env.close()
        print("\nEnvironment interaction test complete.")

    except Exception as e:
        print(f"Error during environment interaction test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    run_basic_rl_env()