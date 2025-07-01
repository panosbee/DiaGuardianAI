# DiaGuardianAI - Run Pattern Advisor Agent (Regressor) in Simulation

import sys
import os
import numpy as np
import pandas as pd
import datetime
from typing import Dict, Any, Optional, List

# Ensure the DiaGuardianAI package is discoverable
if __package__ is None or __package__ == '':
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from DiaGuardianAI.agents.pattern_advisor_agent import PatternAdvisorAgent
from DiaGuardianAI.pattern_repository.repository_manager import RepositoryManager
from DiaGuardianAI.core.simulation_engine import SimulationEngine
from DiaGuardianAI.data_generation.synthetic_patient_model import SyntheticPatient
from DiaGuardianAI.predictive_models.model_zoo.lstm_predictor import LSTMPredictor

# Configuration
ADVISOR_MODEL_FILENAME = "pattern_advisor_regressor_v1.pkl" # Changed to the new regressor model
SIMULATION_DURATION_HOURS = 24 * 3 # 3 days
PATIENT_ID = "adult#001" # Example patient

# Assumed parameters for the loaded PatternAdvisorAgent (regressor)
# These should ideally match how the agent was trained/saved.
# The load_agent_from_files method will try to load these from metadata first.
ASSUMED_STATE_DIM = 51 # Should match the RLAgent's observation space if mimicking it
ASSUMED_ACTION_DIM = 2
ASSUMED_ACTION_KEYS_ORDERED = ['basal_rate_u_hr', 'bolus_u'] # Example, ensure this matches training

def load_glucose_predictor(model_load_dir: str) -> Optional[LSTMPredictor]:
    """Loads a pre-trained LSTMPredictor."""
    if not os.path.exists(model_load_dir):
        print(f"LSTMPredictor model directory not found: {model_load_dir}. Cannot load predictor.")
        return None
    try:
        # Initialize with dummy values, load will overwrite from config
        predictor = LSTMPredictor(input_seq_len=1, output_seq_len=1, n_features=1)
        predictor.load(model_load_dir)
        print(f"LSTMPredictor loaded successfully from {model_load_dir}")
        return predictor
    except Exception as e:
        print(f"Error loading LSTMPredictor from {model_load_dir}: {e}")
        return None

def create_realistic_meal_schedule(duration_hours: int) -> Dict[int, Dict[str, Any]]:
    """Creates a realistic meal schedule for the simulation.

    Args:
        duration_hours: Total simulation duration in hours

    Returns:
        Dict mapping simulation time (minutes) to meal details
    """
    meal_schedule = {}

    # Define meal patterns (time in minutes from start, carbs, GI factor)
    daily_meals = [
        (480, {"grams": 60, "gi_factor": 1.0, "type": "breakfast"}),    # 8:00 AM - 60g carbs
        (720, {"grams": 45, "gi_factor": 1.0, "type": "lunch"}),       # 12:00 PM - 45g carbs
        (1080, {"grams": 75, "gi_factor": 1.0, "type": "dinner"}),     # 6:00 PM - 75g carbs
        (570, {"grams": 15, "gi_factor": 1.2, "type": "snack"}),       # 9:30 AM - morning snack
        (900, {"grams": 20, "gi_factor": 0.8, "type": "snack"}),       # 3:00 PM - afternoon snack
    ]

    # Add meals for each day
    num_days = max(1, duration_hours // 24 + 1)
    for day in range(num_days):
        day_offset = day * 24 * 60  # Minutes per day

        for meal_time, meal_details in daily_meals:
            total_time = day_offset + meal_time

            # Only add meals within simulation duration
            if total_time < duration_hours * 60:
                meal_schedule[total_time] = meal_details.copy()

    print(f"Created meal schedule with {len(meal_schedule)} meals over {duration_hours} hours:")
    for time_min, details in sorted(meal_schedule.items()):
        hours = time_min // 60
        minutes = time_min % 60
        print(f"  Day {hours//24 + 1}, {hours%24:02d}:{minutes:02d} - {details['type']}: {details['grams']}g carbs (GI: {details['gi_factor']})")

    return meal_schedule

def load_trained_pattern_advisor(
    model_path: str,
    state_dim: int,
    dummy_repo_db_path: str = "dummy_advisor_sim_repo.sqlite"
) -> Optional[PatternAdvisorAgent]:
    """Loads a trained PatternAdvisorAgent."""
    if not os.path.exists(model_path):
        print(f"PatternAdvisorAgent model file not found: {model_path}")
        return None

    if os.path.exists(dummy_repo_db_path):
        os.remove(dummy_repo_db_path)
    dummy_repository = RepositoryManager(db_path=dummy_repo_db_path)

    try:
        advisor = PatternAdvisorAgent.load_agent_from_files(
            model_path=model_path,
            pattern_repository=dummy_repository
        )
        if advisor.model is None and advisor.learning_model_type != "none":
            print(f"Warning: Advisor model object is None after loading, attempting to build for type {advisor.learning_model_type}")
            advisor._build_model()

        if advisor.model is None and advisor.learning_model_type != "none":
            print(f"ERROR: Advisor model is still None after attempting to build. Type: {advisor.learning_model_type}")
            return None

        print(f"PatternAdvisorAgent loaded successfully from {model_path}")
        print(f"  Agent type: {advisor.learning_model_type}")
        print(f"  Is trained: {advisor.is_trained}")
        print(f"  Action dim: {advisor.action_dim}, Action keys: {advisor.action_keys_ordered}")
        return advisor
    except Exception as e:
        print(f"Error loading PatternAdvisorAgent from {model_path}: {e}")
        return None
    finally:
        if dummy_repository and hasattr(dummy_repository, 'conn') and dummy_repository.conn:
            dummy_repository.conn.close()
        if os.path.exists(dummy_repo_db_path):
            try:
                os.remove(dummy_repo_db_path)
            except Exception as e_del:
                print(f"Error deleting dummy repo DB: {e_del}")


def run_advisor_simulation():
    print(f"--- Running Simulation with PatternAdvisorAgent ({ADVISOR_MODEL_FILENAME}) ---")

    # 1. Setup paths and models
    advisor_model_dir = os.path.join(project_root, "DiaGuardianAI", "models", "pattern_advisor_agent_model")
    advisor_model_path = os.path.join(advisor_model_dir, ADVISOR_MODEL_FILENAME)
    lstm_predictor_model_dir = os.path.join(project_root, "DiaGuardianAI", "models", "lstm_predictor_example_run")

    # 2. Load Glucose Predictor (PatternAdvisorAgent might expect a predictor instance)
    glucose_predictor = load_glucose_predictor(lstm_predictor_model_dir)

    # 3. Load Trained PatternAdvisorAgent
    advisor_agent = load_trained_pattern_advisor(
        model_path=advisor_model_path,
        state_dim=ASSUMED_STATE_DIM
    )

    if advisor_agent is None:
        print("Failed to load PatternAdvisorAgent. Aborting simulation.")
        return
    if advisor_agent.learning_model_type not in ["mlp_regressor", "gradient_boosting_regressor"]:
        print(f"ERROR: Loaded PatternAdvisorAgent is not a recognized regressor type ({advisor_agent.learning_model_type}). Aborting.")
        return
    if not advisor_agent.is_trained:
        print("ERROR: Loaded PatternAdvisorAgent model is not marked as trained. Aborting.")
        return
    if (
        advisor_agent.action_dim != ASSUMED_ACTION_DIM or
        advisor_agent.action_keys_ordered is None or
        sorted(advisor_agent.action_keys_ordered) != sorted(ASSUMED_ACTION_KEYS_ORDERED)
    ):
        print(f"Warning: Loaded advisor action spec (dim: {advisor_agent.action_dim}, keys: {advisor_agent.action_keys_ordered}) "
              f"differs from assumed (dim: {ASSUMED_ACTION_DIM}, keys: {ASSUMED_ACTION_KEYS_ORDERED}).")

    # 4. Initialize Patient with REALISTIC diabetes parameters
    patient_params = {
        "initial_glucose": 120.0,
        "ISF": 50.0,  # Insulin Sensitivity Factor: 1U drops glucose by 50 mg/dL
        "CR": 10.0,   # Carb Ratio: 1U covers 10g carbs
        "basal_rate_U_hr": 1.0,  # Baseline insulin need
        "body_weight_kg": 70.0,
        # CRITICAL: Fix insulin action parameters
        "k_u_id": 0.0005,  # Increase insulin-dependent glucose utilization
        "k_egp": 0.02,     # Reduce endogenous glucose production
        "target_glucose": 100.0,  # Target glucose level
    }
    patient = SyntheticPatient(params=patient_params)
    print(f"Patient initialized with realistic diabetes parameters:")
    print(f"  ISF: {patient_params['ISF']} mg/dL per unit")
    print(f"  CR: {patient_params['CR']} g carbs per unit")
    print(f"  Basal: {patient_params['basal_rate_U_hr']} U/hr")

    # 5. Create Realistic Meal Schedule for 3 days
    meal_schedule = create_realistic_meal_schedule(SIMULATION_DURATION_HOURS)

    sim_config = {
        "max_simulation_steps": SIMULATION_DURATION_HOURS * 12,  # 12 steps per hour for 5-min steps
        "time_step_minutes": 5,
        "meal_schedule": meal_schedule,  # Add meal schedule to config
        "glucose_rescue_enabled": True,  # Enable automatic glucose rescue
        "glucose_rescue_threshold": 70.0,  # Rescue when CGM < 70 mg/dL
        "glucose_rescue_carbs": 15.0,  # 15g fast-acting carbs
    }
    engine = SimulationEngine(patient=patient, agent=advisor_agent, config=sim_config)
    print(f"Simulation engine initialized. Duration: {SIMULATION_DURATION_HOURS} hours.")

    # 5. Run Simulation
    simulation_results = engine.run()

    # 6. Analyze Results
    print("\n--- Simulation Results ---")
    cgm_values = simulation_results.get("cgm_readings", [])
    cgm_series = pd.Series(cgm_values)
    tir_ideal = ((cgm_series >= 70) & (cgm_series <= 180)).mean() * 100
    tir_tight = ((cgm_series >= 70) & (cgm_series <= 140)).mean() * 100
    percent_hypo_l1 = (cgm_series < 70).mean() * 100
    percent_hypo_l2 = (cgm_series < 54).mean() * 100
    percent_hyper_l1 = (cgm_series > 180).mean() * 100
    percent_hyper_l2 = (cgm_series > 250).mean() * 100

    print(f"Simulation duration: {SIMULATION_DURATION_HOURS} hours")
    print(f"Average CGM: {cgm_series.mean():.2f} mg/dL")
    print(f"Std Dev CGM: {cgm_series.std():.2f} mg/dL")
    print(f"Min CGM: {cgm_series.min():.2f} mg/dL")
    print(f"Max CGM: {cgm_series.max():.2f} mg/dL")
    print(f"Time in Ideal Range (70-180 mg/dL): {tir_ideal:.2f}%")
    print(f"Time in Tight Range (70-140 mg/dL): {tir_tight:.2f}%")
    print(f"Time Below 70 mg/dL (Hypo L1): {percent_hypo_l1:.2f}%")
    print(f"Time Below 54 mg/dL (Hypo L2): {percent_hypo_l2:.2f}%")
    print(f"Time Above 180 mg/dL (Hyper L1): {percent_hyper_l1:.2f}%")
    print(f"Time Above 250 mg/dL (Hyper L2): {percent_hyper_l2:.2f}%")

if __name__ == "__main__":
    run_advisor_simulation()