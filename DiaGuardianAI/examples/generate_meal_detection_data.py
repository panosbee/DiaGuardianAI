# DiaGuardianAI - Generate Labeled Data for Meal Detection Training

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional # Added Optional

# Ensure the DiaGuardianAI package is discoverable
if __package__ is None or __package__ == '':
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from DiaGuardianAI.data_generation.synthetic_patient_model import SyntheticPatient
from DiaGuardianAI.agents.meal_detector import MealDetector # To use _extract_features_for_ml
from DiaGuardianAI.utils.patient_sampler import sample_patient_params # For diverse patients

def generate_simulation_for_meal_data(
    patient_params: Dict[str, Any],
    duration_days: int = 2,
    time_step_minutes: int = 5,
    meal_events: Optional[List[Dict[str, Any]]] = None
) -> pd.DataFrame:
    """
    Runs a simulation for a single patient and collects data relevant for meal detection.

    Args:
        patient_params (Dict[str, Any]): Parameters for the SyntheticPatient.
        duration_days (int): Duration of the simulation in days.
        time_step_minutes (int): Time step for the simulation.
        meal_events (Optional[List[Dict[str, Any]]]): List of meal events.
            Each dict: {"time_offset_minutes": int, "carbs_g": float, "protein_g": float, "fat_g": float}
            time_offset_minutes is from the start of the simulation.

    Returns:
        pd.DataFrame: DataFrame with columns like 'timestamp', 'cgm', 'actual_carbs_g_at_event', 'is_meal_event_window'.
    """
    patient = SyntheticPatient(params=patient_params)
    total_steps = (duration_days * 24 * 60) // time_step_minutes
    
    simulation_log = []
    current_time = datetime(2024, 1, 1, 6, 0, 0) # Arbitrary start time

    # Convert meal_events time_offset_minutes to step indices
    meal_schedule_steps: Dict[int, Dict[str, float]] = {}
    if meal_events:
        for meal in meal_events:
            step_idx = meal["time_offset_minutes"] // time_step_minutes
            meal_schedule_steps[step_idx] = {
                "carbs_g": meal.get("carbs_g", 0.0),
                "protein_g": meal.get("protein_g", 0.0),
                "fat_g": meal.get("fat_g", 0.0)
            }

    for step in range(total_steps):
        # For data generation, agent actions are not critical, assume basic basal.
        # We are interested in the patient's response to meals.
        basal_for_step = patient_params.get("basal_rate_U_hr", 0.8) * (time_step_minutes / 60.0)
        bolus_for_step = 0.0 # No bolus for this data generation pass

        meal_input_carbs = 0.0
        meal_input_protein = 0.0
        meal_input_fat = 0.0
        
        carbs_details_for_patient_step = None
        if step in meal_schedule_steps:
            meal_data = meal_schedule_steps[step]
            meal_input_carbs = meal_data["carbs_g"]
            meal_input_protein = meal_data["protein_g"]
            meal_input_fat = meal_data["fat_g"]
            if meal_input_carbs > 0:
                 carbs_details_for_patient_step = {"grams": meal_input_carbs, "gi_factor": 1.0}


        patient.step(
            basal_insulin=basal_for_step,
            bolus_insulin=bolus_for_step,
            carbs_details=carbs_details_for_patient_step,
            protein_ingested=meal_input_protein,
            fat_ingested=meal_input_fat,
            exercise_event=None # No exercise for this simple meal data generation
        )
        
        cgm = patient.get_cgm_reading()
        
        simulation_log.append({
            "timestamp": current_time,
            "cgm": cgm,
            "actual_carbs_g_at_event": meal_input_carbs, # Carbs ingested at this specific step
            "actual_protein_g_at_event": meal_input_protein,
            "actual_fat_g_at_event": meal_input_fat
        })
        current_time += timedelta(minutes=time_step_minutes)
        
    return pd.DataFrame(simulation_log)

def create_features_and_labels(
    simulation_df: pd.DataFrame,
    meal_detector_instance: MealDetector,
    history_len: int = 12, # Number of past CGM samples to use for features
    meal_label_window_future_minutes: int = 15, # How far ahead to label a meal event after actual ingestion
    time_step_minutes: int = 5
) -> Tuple[List[np.ndarray], List[int], List[float]]:
    """
    Creates feature vectors and labels from simulation data.

    Args:
        simulation_df (pd.DataFrame): DataFrame from generate_simulation_for_meal_data.
        meal_detector_instance (MealDetector): Instance to use for feature extraction.
        history_len (int): How many past samples for CGM history features.
        meal_label_window_future_minutes (int): Window after carb ingestion to label as "meal event".
        time_step_minutes (int): Simulation time step.

    Returns:
        Tuple[List[np.ndarray], List[int], List[float]]:
            - List of feature arrays.
            - List of binary meal event labels (1 for meal, 0 for no meal).
            - List of carbohydrate amounts (for regression task).
    """
    features_list = []
    is_meal_labels = []
    carb_amount_labels = []

    meal_label_window_steps = meal_label_window_future_minutes // time_step_minutes

    for i in range(len(simulation_df)):
        if i < history_len -1 : # Need enough history to form features
            continue

        # Extract history for feature calculation
        # Features are based on data UP TO AND INCLUDING current point i
        current_cgm_history = simulation_df['cgm'].iloc[max(0, i - history_len + 1) : i + 1].tolist()
        current_timestamps = simulation_df['timestamp'].iloc[max(0, i - history_len + 1) : i + 1].tolist()

        feature_vector = meal_detector_instance._extract_features_for_ml(
            cgm_history=current_cgm_history,
            timestamps=current_timestamps
        )
        features_list.append(feature_vector)

        # Determine label: is there a significant meal event starting "around" now?
        # A meal event is positive if carbs were ingested at step 'i' or within the near future window.
        # This labeling is for detecting the *onset* or *presence* of a meal based on features leading up to it.
        
        is_meal_event_in_window = 0
        actual_carbs_for_label = 0.0

        # Check if a meal was ingested at the current step or in the recent past leading to this window
        # The features at step 'i' are to predict if a meal is happening *now* or started *recently*.
        # Let's define "meal event" if carbs > 0 at current step 'i'.
        # Or, if we want to predict the start of a meal based on CGM rise *after* ingestion,
        # the label needs to be shifted.
        # For now, let's label based on actual ingestion at step 'i'.
        
        # Simpler: if carbs were ingested at step 'i', it's a meal event start.
        if simulation_df['actual_carbs_g_at_event'].iloc[i] > 5: # Threshold for significant carbs
            is_meal_event_in_window = 1
            actual_carbs_for_label = simulation_df['actual_carbs_g_at_event'].iloc[i]
        
        # Alternative: Label based on a window *after* ingestion.
        # This means features at time 't' predict a meal that started at 't' or 't-window'.
        # Let's stick to: features at 't' are to detect if a meal is active/starting at 't'.
        # So, if simulation_df['actual_carbs_g_at_event'].iloc[i] > 0, then it's a meal.

        is_meal_labels.append(is_meal_event_in_window)
        carb_amount_labels.append(actual_carbs_for_label)
        
    return features_list, is_meal_labels, carb_amount_labels


if __name__ == "__main__":
    print("--- Generating Labeled Data for Meal Detection ---")
    
    num_patients_to_simulate = 15 # Increased number of patients
    sim_duration_days_per_patient = 5 # Increased simulation duration
    time_step = 5

    all_features: List[np.ndarray] = []
    all_is_meal_labels: List[int] = []
    all_carb_labels: List[float] = []

    # Initialize a dummy MealDetector just for its _extract_features_for_ml method
    # Parameters for feature extraction itself are not highly dependent on rule-based params here.
    dummy_meal_detector = MealDetector(detection_method="rule_based", params={})

    for i in range(num_patients_to_simulate):
        print(f"\nSimulating patient {i+1}/{num_patients_to_simulate}...")
        # Create diverse patient params (can be expanded)
        patient_config = sample_patient_params() # Removed patient_type argument
        patient_config["time_step_minutes"] = time_step # Ensure patient model uses same time step

        # Define some varied meal events for this patient
        # Times are offsets from simulation start in minutes
        meal_events_for_patient: List[Dict[str, Any]] = []
        # Generate 3-5 meals/snacks per day
        num_meals_per_day = np.random.randint(3, 6)
        for day in range(sim_duration_days_per_patient):
            day_offset_minutes = day * 24 * 60
            # Breakfast (6-9 AM)
            meal_events_for_patient.append({
                "time_offset_minutes": day_offset_minutes + np.random.randint(6*60, 9*60 + 1),
                "carbs_g": np.random.uniform(20, 70),
                "protein_g": np.random.uniform(5,25),
                "fat_g": np.random.uniform(5,20)
            })
            # Lunch (12-3 PM)
            meal_events_for_patient.append({
                "time_offset_minutes": day_offset_minutes + np.random.randint(12*60, 15*60 + 1),
                "carbs_g": np.random.uniform(30, 80),
                "protein_g": np.random.uniform(10,30),
                "fat_g": np.random.uniform(10,25)
            })
            # Dinner (6-9 PM)
            meal_events_for_patient.append({
                "time_offset_minutes": day_offset_minutes + np.random.randint(18*60, 21*60 + 1),
                "carbs_g": np.random.uniform(30, 70),
                "protein_g": np.random.uniform(10,30),
                "fat_g": np.random.uniform(8,20)
            })
            # Add 0-2 random snacks
            for _ in range(np.random.randint(0, 3)):
                 meal_events_for_patient.append({
                     "time_offset_minutes": day_offset_minutes + np.random.randint(7*60, 22*60 + 1), # Snack anytime between 7am-10pm
                     "carbs_g": np.random.uniform(10, 40),
                     "protein_g": np.random.uniform(0,10),
                     "fat_g": np.random.uniform(0,10)
                 })
        
        # Sort meal events by time to ensure correct processing if any overlaps from random generation
        meal_events_for_patient.sort(key=lambda x: x["time_offset_minutes"])

        sim_df = generate_simulation_for_meal_data(
            patient_params=patient_config,
            duration_days=sim_duration_days_per_patient,
            time_step_minutes=time_step,
            meal_events=meal_events_for_patient
        )
        
        print(f"Simulation for patient {i+1} generated {len(sim_df)} data points.")
        
        # Define history length for feature extraction
        # This should match what the ML model will expect during inference.
        # The MealDetector's _extract_features_for_ml currently generates 11 features.
        feature_history_len = 12 # e.g., 1 hour of 5-min CGM data

        features, is_meal, carbs = create_features_and_labels(
            simulation_df=sim_df,
            meal_detector_instance=dummy_meal_detector,
            history_len=feature_history_len,
            time_step_minutes=time_step
        )
        
        all_features.extend(features)
        all_is_meal_labels.extend(is_meal)
        all_carb_labels.extend(carbs)
        print(f"Extracted {len(features)} feature sets for patient {i+1}.")

    if all_features:
        features_array = np.array(all_features)
        is_meal_array = np.array(all_is_meal_labels)
        carbs_array = np.array(all_carb_labels)
        
        print(f"\nTotal feature sets generated: {features_array.shape[0]}")
        print(f"Feature vector shape: {features_array.shape}")
        print(f"Meal labels shape: {is_meal_array.shape}, Positive meal events: {np.sum(is_meal_array)}")
        print(f"Carb labels shape: {carbs_array.shape}, Avg carbs on meal events: {np.mean(carbs_array[is_meal_array==1]) if np.sum(is_meal_array) > 0 else 0 :.2f}g")

        # Save the data
        output_dir = "DiaGuardianAI/datasets/generated_meal_detection_data"
        os.makedirs(output_dir, exist_ok=True)
        
        features_path = os.path.join(output_dir, "meal_features.npy")
        labels_path = os.path.join(output_dir, "meal_labels.npy") # For is_meal
        carbs_path = os.path.join(output_dir, "meal_carbs.npy")   # For carb amounts

        np.save(features_path, features_array)
        np.save(labels_path, is_meal_array)
        np.save(carbs_path, carbs_array)
        
        print(f"\nGenerated data saved to:")
        print(f"  Features: {features_path}")
        print(f"  Labels (is_meal): {labels_path}")
        print(f"  Labels (carbs_g): {carbs_path}")
    else:
        print("No features were generated.")

    print("\nMeal detection data generation script finished.")