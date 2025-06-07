# DiaGuardianAI - Generate Labeled Data for Pattern Advisor Agent Training

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple

# Ensure the DiaGuardianAI package is discoverable
if __package__ is None or __package__ == '':
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from DiaGuardianAI.core.environments import DiaGuardianEnv
from DiaGuardianAI.data_generation.synthetic_patient_model import SyntheticPatient
from DiaGuardianAI.agents.decision_agent import RLAgent # To run simulations
from DiaGuardianAI.agents.pattern_advisor_agent import PatternAdvisorAgent # For feature extraction
from DiaGuardianAI.pattern_repository.repository_manager import RepositoryManager # Added import
from DiaGuardianAI.predictive_models.model_zoo.lstm_predictor import LSTMPredictor
from DiaGuardianAI.utils.patient_sampler import sample_patient_params
from stable_baselines3 import PPO # Assuming we use a trained PPO agent to generate good examples

def generate_data_with_rl_agent(
    patient_params_list: List[Dict[str, Any]],
    rl_agent_model_path: str,
    advisor_agent_for_features: PatternAdvisorAgent, # Used for its _prepare_features_for_internal_model
    num_episodes_per_patient: int = 1,
    max_steps_per_episode: int = 288 * 1, # 1 day
    time_step_minutes: int = 5,
    reward_threshold_for_good_pattern: float = 0.5 # Example: only consider actions that led to reward > 0.5
) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
    """
    Runs simulations using a trained RL agent and collects (state_features, successful_action_dict) pairs.
    """
    collected_features: List[np.ndarray] = []
    collected_successful_actions: List[Dict[str, Any]] = []

    for i, patient_params in enumerate(patient_params_list):
        print(f"\nSimulating with patient profile {i+1}/{len(patient_params_list)}...")
        patient = SyntheticPatient(params=patient_params)
        
        # RLAgent for specs for DiaGuardianEnv
        # This RLAgent's predictor will be used by the Env to generate observations
        # It should match the one used for training the loaded PPO model, if applicable
        # For simplicity, we can load the example LSTM predictor
        # Define project_root locally for this function to ensure it's available
        current_file_dir_func = os.path.dirname(os.path.abspath(__file__))
        project_root_func = os.path.abspath(os.path.join(current_file_dir_func, '..', '..'))

        lstm_predictor_path = os.path.join(project_root_func, "DiaGuardianAI", "models", "lstm_predictor_example_run")
        loaded_lstm_pred = None
        if os.path.exists(lstm_predictor_path):
            try:
                loaded_lstm_pred = LSTMPredictor(input_seq_len=1,output_seq_len=1,n_features=1) #dummy
                loaded_lstm_pred.load(lstm_predictor_path)
                print("Loaded LSTM predictor for RLAgent spec in data generation.")
            except Exception as e:
                print(f"Could not load LSTM predictor for RLAgent spec: {e}")
        
        # Determine state_dim based on RLAgent's _define_state
        # Assuming cgm_hist_len=24.
        # pred_horizon for the RLAgent's state definition should match what PPO was trained on.
        cgm_hist_len = 24 # Changed to 24 to match LSTMPredictor and RLAgent's new default
        # The PPO model was trained with an RLAgent that had prediction_horizon_len=6.
        # The LSTMPredictor itself might output more (e.g., 12), but RLAgent._define_state will truncate.
        pred_horizon_for_rl_agent_state = 6
        # pred_horizon_from_lstm = loaded_lstm_pred.output_seq_len if loaded_lstm_pred and hasattr(loaded_lstm_pred, 'output_seq_len') else 6
        # The RLAgent used for spec should have its state_dim correctly set.
        # Let's use the advisor_agent_for_features.state_dim as it's what we'll train.
        # However, the environment's observation space is defined by the RLAgent running the simulation.
        
        # RLAgent that will run the simulation (using the loaded PPO model)
        # Its state_dim and action_space must match the loaded PPO model.
        # We assume the loaded PPO model was trained with an RLAgent having a specific state_dim.
        # For now, let's use a generic state_dim for the acting RLAgent,
        # as the PPO model itself dictates the observation space it expects.
        
        # Action space for the acting RLAgent
        # This MUST match the action space used when training the PPO model being loaded.
        # The PPO model was trained with bolus_u high=10.0.
        acting_agent_action_space = {
            "bolus_u": {"low": 0.0, "high": 10.0}, # Corrected to match trained PPO model
            "basal_rate_u_hr": {"low": 0.0, "high": 3.0}
        }
        # This acting_agent is just a shell to load the PPO model into.
        # Its state_dim MUST match the observation space of the loaded PPO model.
        # The PPO model will be retrained with state_dim = 51.
        ppo_model_expected_state_dim = 51 # Changed to 51
        acting_rl_agent = RLAgent(
            state_dim=ppo_model_expected_state_dim, # Must match the PPO model (now 51)
            action_space_definition=acting_agent_action_space,
            predictor=loaded_lstm_pred, # LSTMPredictor might have output_seq_len=12
            cgm_history_len=cgm_hist_len, # This is 24
            prediction_horizon_len=pred_horizon_for_rl_agent_state # This is 6, for RLAgent's state construction
            # The RLAgent's internal observation_space_gym will be (39,)
        )
        try:
            acting_rl_agent.load(rl_agent_model_path) # Load the trained PPO model
            print(f"Successfully loaded PPO model from {rl_agent_model_path} into acting_rl_agent.")
        except Exception as e:
            print(f"ERROR: Could not load PPO model from {rl_agent_model_path}: {e}. Skipping this patient.")
            continue

        # Environment setup
        env_sim_config = {
            "time_step_minutes": time_step_minutes,
            "max_episode_steps": max_steps_per_episode,
            "meal_schedule": { # Example dynamic meal schedule for data generation
                np.random.randint(6, 10) * 60 : np.random.uniform(30, 70),
                np.random.randint(12, 16) * 60: np.random.uniform(40, 80),
                np.random.randint(18, 22) * 60: np.random.uniform(30, 60),
            },
            "reward_params": {} # Use default rewards from DiaGuardianEnv
        }
        
        # The agent_for_specs in DiaGuardianEnv defines the observation space structure.
        # This should be the PatternAdvisorAgent instance whose features we want to capture.
        # However, DiaGuardianEnv expects an RLAgent for its agent_for_specs to get observation/action spaces.
        # This is a slight conflict. For now, we'll pass the acting_rl_agent as agent_for_specs,
        # and then separately use advisor_agent_for_features._prepare_features_for_internal_model.
        
        env = DiaGuardianEnv(
            patient=patient,
            agent_for_specs=acting_rl_agent, # Env uses this for obs/action space def
            sim_config=env_sim_config
        )

        for ep in range(num_episodes_per_patient):
            obs_tuple = env.reset()
            obs = obs_tuple[0] if isinstance(obs_tuple, tuple) else obs_tuple # Handle potential tuple from reset
            
            current_episode_states_for_advisor = []
            current_episode_actions_taken = []
            current_episode_rewards = []

            for step in range(max_steps_per_episode):
                if acting_rl_agent.rl_model is None:
                    print("Error: acting_rl_agent.rl_model is None. Cannot predict action. Skipping episode.")
                    break # Skip this episode for this patient
                
                # Get action from the loaded PPO model
                action_np, _ = acting_rl_agent.rl_model.predict(obs, deterministic=True)
                
                # Store state features *before* taking the action
                # The state for the advisor should be based on the observation the PPO agent saw
                # We need to convert 'obs' (the PPO's observation) into the format expected by
                # advisor_agent_for_features._prepare_features_for_internal_model.
                # This requires mapping obs vector back to a state_dict. This is complex.
                
                # Simpler approach for now: Use the state_dict that DiaGuardianEnv
                # would internally use to create 'obs'.
                # DiaGuardianEnv._get_observation() returns the obs vector.
                # We need the dict that *generated* that obs vector.
                # Let's get it from the environment directly if possible, or reconstruct.
                # For now, let's use the state_dict from the *previous* step's info or re-fetch.
                
                # The 'obs' is already the feature vector for the RLAgent.
                # The PatternAdvisorAgent's _prepare_features_for_internal_model expects a state_dict.
                # This means we need the state_dict that led to 'obs'.
                # The `env.step` returns info, which contains some state elements.
                
                # Let's capture the state_dict *before* the step, which led to the current `obs`.
                # This is tricky because `obs` is from `env.reset()` or previous `env.step()`.
                
                # Alternative: The `PatternAdvisorAgent`'s features should be derived from the same
                # fundamental patient state that the `RLAgent` sees.
                # `DiaGuardianEnv._get_observation()` calls `self.agent_for_specs._define_state()`.
                # The `current_state` dict passed to `_define_state` is what we need.
                
                # Let's assume `env.current_patient_state_dict_for_agent` exists or can be added to DiaGuardianEnv
                # For now, we'll reconstruct a simplified one for the advisor.
                # This part needs careful alignment.
                
                # The state for the advisor should be what the RLAgent based its action on.
                # `obs` is that state (already vectorized for RLAgent).
                # If PatternAdvisorAgent's internal model is trained on features derived from a state_dict,
                # we need that state_dict.
                
                # For now, let's assume `advisor_agent_for_features` can extract features from `info` or a simplified dict.
                # This is a placeholder for robust state capture for the advisor.
                # The `info` dict from `env.step` contains `cgm`, `iob`, `cob`.
                # We also need `cgm_history`. `env.cgm_history_buffer` can be used.
                
                # Let's get the state dict that would be used by RLAgent to form `obs`
                # This is available inside env._get_observation() before it calls _define_state.
                # To simplify, we'll use the info from the *previous* step for the advisor's state features.
                # This is an approximation.
                
                # A better way: The advisor agent should be trained on (s, a*) where s is the state
                # the RLAgent saw, and a* is the "good" action the RLAgent took.
                # The features for the advisor's supervised model are `advisor_agent_for_features._prepare_features_for_internal_model(s_dict)`.
                
                # Let's assume `env` can provide the `state_dict` that generated the current `obs`.
                # This would require a small modification to `DiaGuardianEnv` to store/return it.
                # For now, we'll use a simplified approach.
                
                # The state that `acting_rl_agent.rl_model.predict(obs)` used was `obs`.
                # If the advisor is to predict a good action for `obs`, then `obs` (or its dict precursor) is the feature.
                
                # Let's assume the advisor's features are derived from a state dict.
                # We need to get this state_dict.
                # The `info` from the *previous* step is for the state *after* the previous action.
                # The `obs` for the current step *is* the state representation.
                
                # If advisor's internal model is trained on features from `_prepare_features_for_internal_model`,
                # that method expects a `current_state` dict.
                # We need to reconstruct this dict from `obs` or get it from `DiaGuardianEnv`.
                
                # For now, let's assume `obs` is directly usable or can be transformed.
                # This part is the most complex for data generation.
                # Let's simplify: we'll log the `obs` vector itself as features for the advisor,
                # and the advisor's internal model will be trained on these RLAgent state vectors.
                # This means advisor.state_dim should match RLAgent's state_dim.
                
                # Store the observation vector that the PPO agent used.
                state_features_for_advisor = obs.copy() # This is the RLAgent's state vector

                next_obs_tuple, reward, done, truncated, info = env.step(action_np)
                next_obs = next_obs_tuple[0] if isinstance(next_obs_tuple, tuple) else next_obs_tuple

                # Convert numpy action back to dict for storage
                action_taken_dict = {}
                if len(action_np) == len(acting_rl_agent.action_keys_ordered):
                    for i_act, key_act in enumerate(acting_rl_agent.action_keys_ordered):
                        action_taken_dict[key_act] = float(action_np[i_act])
                
                if reward > reward_threshold_for_good_pattern:
                    collected_features.append(state_features_for_advisor)
                    collected_successful_actions.append(action_taken_dict.copy())

                obs = next_obs
                if done or truncated:
                    break
            print(f"Episode {ep+1} finished for patient {i+1}. Collected {len(collected_features)} so far.")
        env.close()
    return collected_features, collected_successful_actions


if __name__ == "__main__":
    print("--- Generating Training Data for Pattern Advisor Agent ---")

    num_different_patients = 2 # How many different patient profiles
    episodes_per_patient = 2   # How many simulation runs for each
    sim_time_step = 5
    
    # Path to a pre-trained RLAgent (PPO model)
    # Ensure this model was trained with an RLAgent whose state_dim and action_space match
    # the acting_rl_agent defined in generate_data_with_rl_agent.
    trained_ppo_model_path = "./trained_ppo_diaguardian_agent.zip" # From previous PPO training

    if not os.path.exists(trained_ppo_model_path):
        print(f"ERROR: Trained PPO model not found at {trained_ppo_model_path}. Please train one first.")
        sys.exit(1)

    patient_params_list_for_data_gen = [sample_patient_params() for _ in range(num_different_patients)]
    for p_idx, p_params in enumerate(patient_params_list_for_data_gen):
        p_params["time_step_minutes"] = sim_time_step
        print(f"Patient {p_idx+1} params: ISF={p_params['ISF']:.2f}, CR={p_params['CR']:.2f}")

    # PatternAdvisorAgent (for its feature extraction logic and state_dim)
    # Its state_dim should match the observation space of the PPO model.
    # The PPO model loaded into acting_rl_agent will have its observation_space (51 features).
    # The advisor_agent_for_features.state_dim should match this.
    advisor_state_dim = 51 # Changed to 51
    advisor_action_space_placeholder = "SuggestPatternID" # Not critical for feature extraction

    # Create a dummy repository manager instance for the advisor agent spec
    # It won't be actively used for retrieval in this script's current logic for the advisor.
    dummy_repo_db_path = "dummy_advisor_data_gen_repo.sqlite"
    if os.path.exists(dummy_repo_db_path): # Clean up if exists from a previous failed run
        try:
            os.remove(dummy_repo_db_path)
        except OSError:
            pass # Ignore if removal fails (e.g. in use)
            
    dummy_repository = RepositoryManager(db_path=dummy_repo_db_path)

    advisor_for_features = PatternAdvisorAgent(
        state_dim=advisor_state_dim, # This should match the RLAgent's state vector length
        action_space=advisor_action_space_placeholder,
        pattern_repository=dummy_repository, # Pass a valid repository instance
        learning_model_type="none" # No learning model needed for this script's use of it
    )

    features, successful_actions = generate_data_with_rl_agent(
        patient_params_list=patient_params_list_for_data_gen,
        rl_agent_model_path=trained_ppo_model_path,
        advisor_agent_for_features=advisor_for_features, # Used for its state_dim
        num_episodes_per_patient=episodes_per_patient,
        time_step_minutes=sim_time_step,
        reward_threshold_for_good_pattern=0.0 # Collect more data initially
    )

    if features:
        features_array = np.array(features)
        # successful_actions is a list of dicts. Convert to DataFrame for easier saving/loading.
        actions_df = pd.DataFrame(successful_actions)

        print(f"\nTotal feature sets for advisor training: {features_array.shape[0]}")
        print(f"Advisor features shape: {features_array.shape}")
        print(f"Successful actions DataFrame shape: {actions_df.shape}")

        output_dir = "DiaGuardianAI/datasets/generated_advisor_training_data"
        os.makedirs(output_dir, exist_ok=True)
        
        features_path = os.path.join(output_dir, "advisor_features.npy")
        actions_path = os.path.join(output_dir, "advisor_successful_actions.csv")

        np.save(features_path, features_array)
        actions_df.to_csv(actions_path, index=False)
        
        print(f"\nGenerated data for Pattern Advisor training saved to:")
        print(f"  Features: {features_path}")
        print(f"  Successful Actions: {actions_path}")
    else:
        print("No suitable data generated for Pattern Advisor training.")

    # Clean up the dummy repository DB file
    if os.path.exists(dummy_repo_db_path):
        try:
            # Check if it's a RepositoryManager instance and if conn exists and is open
            repo_instance = advisor_for_features.pattern_repository
            if isinstance(repo_instance, RepositoryManager) and repo_instance.conn:
                 repo_instance.conn.close()
            os.remove(dummy_repo_db_path)
            print(f"Cleaned up dummy repository DB: {dummy_repo_db_path}")
        except Exception as e:
            print(f"Error cleaning up dummy repository DB {dummy_repo_db_path}: {e}")

    print("\nPattern Advisor training data generation script finished.")