# DiaGuardianAI - Basic Simulation Run Example
# This script demonstrates how to set up and run a simulation
# using the SimulationEngine, SyntheticPatientModel, and RLAgent.

import sys
import os
import numpy as np # For dummy_experience in RLAgent example

# Ensure the DiaGuardianAI package is discoverable
if __package__ is None or __package__ == '':
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from DiaGuardianAI.core.simulation_engine import SimulationEngine
from DiaGuardianAI.data_generation.synthetic_patient_model import SyntheticPatient # Corrected class name
from DiaGuardianAI.agents.decision_agent import RLAgent
from DiaGuardianAI.predictive_models.model_zoo.lstm_predictor import LSTMPredictor

def run_basic_simulation():
    print("--- Starting Basic Simulation Example ---")

    # 1. Patient Setup
    # These parameters should match what SyntheticPatientModel expects.
    # Refer to SyntheticPatientModel.__init__ for required params.
    # Using example parameters from previous work on SyntheticPatientModel.
    patient_params = {
        "name": "Adult#SimEx001",
        "base_glucose": 100.0,
        "isf": 50.0,  # Insulin Sensitivity Factor (mg/dL per U)
        "cr": 10.0,   # Carb Ratio (g/U)
        "cir": 10.0,  # Carb-to-Insulin Ratio (same as CR for simplicity here)
        "basal_insulin_rate": 1.0,  # U/hr (will be converted to U/step by patient model)
        "weight_kg": 70.0,
        # Default PK/PD params for rapid and long-acting insulin (can be overridden)
        "insulin_pk_pd_params": {
            "rapid": {"td": 15, "tp": 60, "gamma": 0.005, "kcl": 0.05, "Vd_I": 0.12, "ka_rapid": 0.02},
            "long": {"td": 120, "tp": 360, "gamma": 0.001, "kcl": 0.01, "Vd_I": 0.12, "ka_long": 0.001}
        },
        # Default meal effect params (can be overridden)
        "meal_effect_params": {
            "protein_glucose_conversion_rate": 0.5, # g glucose per g protein
            "fat_glucose_conversion_rate": 0.1, # g glucose per g fat
            "fat_carb_slowdown_factor": 0.7,
            "fat_isf_reduction_factor": 0.8
        },
        # Default exercise effect params (can be overridden)
        "exercise_effect_params": {
            "glucose_reduction_rate_moderate": 1.0, # mg/dL per minute of moderate exercise
            "glucose_reduction_rate_intense": 2.0,  # mg/dL per minute of intense exercise
            "isf_increase_factor_moderate": 1.2,
            "isf_increase_factor_intense": 1.5,
            "post_exercise_sensitivity_duration_minutes": 120,
            "post_exercise_isf_multiplier": 1.3
        },
        "initial_iob": 0.0,
        "initial_cob": 0.0,
        "initial_gb": 100.0, # Initial blood glucose
        "initial_gp": 100.0, # Initial plasma glucose
        "initial_gt": 100.0, # Initial tissue glucose
        "time_step_minutes": 5 # Should match simulation engine's time_step_minutes
    }
    try:
        patient = SyntheticPatient(params=patient_params) # Corrected class name
    except Exception as e:
        print(f"Error initializing SyntheticPatient: {e}") # Corrected class name
        print("Ensure SyntheticPatient is correctly implemented and params match.") # Corrected class name
        return

    # 2. Agent Setup (RLAgent placeholder)
    # The RLAgent's state_dim needs to be calculated based on its _define_state method.
    # From RLAgent._define_state:
    # 1 (cgm) + cgm_history_len + 1 (iob) + 1 (cob) + prediction_horizon_len + 2 (meal flags) + 1 (time)
    # The time component was added in SimulationEngine._get_current_patient_state_for_agent
    # but RLAgent._define_state doesn't explicitly use it yet.
    # Let's assume RLAgent's state_dim is calculated based on its cgm_history_len and prediction_horizon_len.
    
    cgm_hist_len_for_agent = 12  # e.g., 1 hour of 5-min CGM readings
    pred_horizon_for_agent_state = 6 # e.g., 30 mins of 5-min predictions
    
    # state_dim = 1 (cgm) + cgm_hist_len_for_agent + 1 (iob) + 1 (cob) + pred_horizon_for_agent_state + 2 (meal flags)
    # This calculation should match how RLAgent._define_state actually constructs its state vector.
    # RLAgent._define_state currently has:
    # 1 (cgm) + self.cgm_history_len + 1 (iob) + 1 (cob) + self.prediction_horizon_len + 2 (meal flags)
    # So, state_dim = 1 + 12 + 1 + 1 + 6 + 2 = 23 (if using these default lengths)
    
    # The state_dim passed to RLAgent must match the length of the vector returned by its _define_state.
    # RLAgent's _define_state uses self.cgm_history_len and self.prediction_horizon_len.
    # So, state_dim should be: 1 (cgm) + cgm_hist_len_for_agent + 1 (iob) + 1 (cob) + pred_horizon_for_agent_state + 2 (meal flags)
    calculated_agent_state_dim = 1 + cgm_hist_len_for_agent + 1 + 1 + pred_horizon_for_agent_state + 2 # = 23

    # Define a simple action space for the placeholder RLAgent
    # This structure should be interpretable by RLAgent.decide_action's placeholder logic
    action_space_definition = {
        "bolus_u": {"low": 0.0, "high": 10.0},          # Units
        "basal_rate_u_hr": {"low": 0.0, "high": 3.0}    # Units per hour
    }
    
    # Predictor is optional for RLAgent; pass None if not using one for this basic example.
    # If a predictor were used, it would be instantiated here and passed to RLAgent.
    # For LSTMPredictor, input_dim is the number of features per time step.
    # Since RLAgent reshapes cgm_history to (1, seq_len, 1), input_dim is 1.
    lstm_input_dim = 1
    lstm_hidden_dim = 32 # Example value
    lstm_num_layers = 2  # Example value
    
    # output_horizon_steps for LSTMPredictor should match pred_horizon_for_agent_state
    lstm_output_horizon = pred_horizon_for_agent_state

    example_predictor = LSTMPredictor(
        input_dim=lstm_input_dim,
        hidden_dim=lstm_hidden_dim,
        num_layers=lstm_num_layers,
        output_horizon_steps=lstm_output_horizon
    )
    # example_predictor.load("path_to_trained_lstm_model.pth") # If pre-trained, load here
    
    agent = RLAgent(
        state_dim=calculated_agent_state_dim,
        action_space_definition=action_space_definition,
        predictor=example_predictor, # Pass the instantiated predictor
        cgm_history_len=cgm_hist_len_for_agent,
        prediction_horizon_len=pred_horizon_for_agent_state
    )

    # 3. Simulation Engine Configuration
    sim_config = {
        "max_simulation_steps": 288,  # 288 steps * 5 min/step = 1440 minutes = 1 day
        "time_step_minutes": 5,
        "cgm_history_buffer_size": cgm_hist_len_for_agent, # For SimulationEngine's internal buffer
        "meal_schedule": { # Times are in minutes from start of simulation
            60 * 8: 50.0,   # Breakfast: 50g carbs at 8:00 AM (480 minutes)
            60 * 13: 70.0,  # Lunch: 70g carbs at 1:00 PM (780 minutes)
            60 * 19: 60.0   # Dinner: 60g carbs at 7:00 PM (1140 minutes)
        },
        "exercise_schedule": { # Times are in minutes
            60 * 16: {"duration_minutes": 30, "intensity": "moderate"} # Exercise at 4:00 PM (960 minutes)
        }
        # protein_schedule and fat_schedule can also be added here if desired
    }

    # 4. Initialize and Run SimulationEngine
    # The SimulationEngine does not take a predictor directly; the agent manages its own.
    engine = SimulationEngine(patient=patient, agent=agent, config=sim_config)
    
    print("\n--- Running Simulation ---")
    simulation_results = engine.run()
    print("--- Simulation Complete ---")

    # 5. Process or Plot Results (using placeholder plot)
    if simulation_results:
        print(f"\nSimulation produced {len(simulation_results['cgm_readings'])} CGM readings.")
        engine.plot_glucose_trace() # Calls the placeholder plotting method

        # Example: Print last few actions
        num_actions_to_print = 5
        if len(simulation_results['actions_taken']) >= num_actions_to_print:
            print(f"\nLast {num_actions_to_print} actions taken:")
            for i in range(-num_actions_to_print, 0):
                print(f"  Step {len(simulation_results['actions_taken']) + i}: {simulation_results['actions_taken'][i]}")
        
        # Save the agent's (untrained) model
        agent_save_path = "./basic_sim_rl_agent_ppo"
        print(f"\n--- Saving Agent Model to {agent_save_path} ---")
        agent.save(agent_save_path)

        # Test loading the saved agent
        print(f"\n--- Loading Agent Model from {agent_save_path} ---")
        loaded_agent = RLAgent(
            state_dim=calculated_agent_state_dim,
            action_space_definition=action_space_definition, # Must be same as when saved
            predictor=example_predictor, # Or re-instantiate a predictor if it's not saved with agent
            cgm_history_len=cgm_hist_len_for_agent,
            prediction_horizon_len=pred_horizon_for_agent_state,
            rl_algorithm="PPO" # Ensure this matches the saved model type
        )
        loaded_agent.load(agent_save_path)
        print("Agent model loaded successfully.")

        # Optional: Run a very short simulation with the loaded agent to test action prediction
        print("\n--- Running Short Simulation with Loaded Agent ---")
        short_sim_config = sim_config.copy()
        short_sim_config["max_simulation_steps"] = 5
        
        # Re-initialize patient for a clean run, or use existing patient if desired
        # For simplicity, using the same patient instance that has already run a full sim.
        # This means its internal state is not reset.
        # patient_for_loaded_test = SyntheticPatient(params=patient_params) # For a fresh patient
        
        engine_loaded_agent = SimulationEngine(patient=patient, agent=loaded_agent, config=short_sim_config)
        loaded_agent_results = engine_loaded_agent.run()
        if loaded_agent_results and loaded_agent_results.get('actions_taken'):
            print("Loaded agent produced actions:")
            for i, act in enumerate(loaded_agent_results['actions_taken']):
                print(f"  Step {i}: {act}")
        else:
            print("Loaded agent did not produce actions in short simulation.")

        # --- Example of initiating training ---
        # Note: For actual SB3 PPO training, the RLAgent's internal _MinimalGymEnv
        # would need to be properly connected to the SimulationEngine to run episodes
        # and collect rollouts. This is a conceptual placeholder.
        print("\n--- Example: Initiating Agent Training (Conceptual) ---")
        try:
            # Using the 'loaded_agent' instance for this example.
            # In a real workflow, you might train 'agent' before saving, or train 'loaded_agent' further.
            print("Calling loaded_agent.train_rl_model(total_timesteps=100)...") # Small number for example
            loaded_agent.train_rl_model(total_timesteps=100)
            # SB3's .learn() method (called by train_rl_model) will print its own progress if verbose > 0.
            # The _MinimalGymEnv will just loop with static observations for now.
            
            trained_model_save_path = "./basic_sim_rl_agent_ppo_trained_example"
            print(f"Conceptual training finished. Saving potentially updated model to {trained_model_save_path}")
            loaded_agent.save(trained_model_save_path)
        except Exception as e:
            print(f"Error during conceptual training example: {e}")
            print("This might be expected if the _MinimalGymEnv is not fully interactive for training yet.")


    else:
        print("Simulation did not produce results.")

if __name__ == '__main__':
    run_basic_simulation()