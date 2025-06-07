# DiaGuardianAI Dataset Generator
# Generates synthetic patient data for training and testing purposes.

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from DiaGuardianAI.data_generation.synthetic_patient_model import SyntheticPatient
from typing import List, Dict, Any, Optional

# --- Configuration ---
BASE_OUTPUT_DIR = "DiaGuardianAI/datasets/generated_data"
SIMULATION_STEP_MINUTES = 5 # Should match patient model's dt_minutes

# --- Patient Profiles ---
# Define a list of base patient parameter dictionaries
# These can be expanded or loaded from external files later
PATIENT_PROFILES = [
    {
        "profile_name": "adult_moderate_control",
        "params": {
            "initial_glucose": 120.0,
            "ISF": 50.0, "CR": 10.0, "target_glucose": 100.0, "body_weight_kg": 70.0,
            "basal_rate_U_hr": 0.9,
            "carb_absorption_rate_g_min": 0.07, # Base hourly fraction for D1->D2
            "k_d2_to_plasma_rate_per_min": 0.02,
            "p1_glucose_clearance_rate_per_min": 0.003,
            "k_u_id_coeff": 0.0005, "k_egp_feedback_strength": 0.005,
            "glucose_utilization_rate_mg_dl_min": 0.1,
            "protein_glucose_conversion_factor": 0.5, "protein_max_absorption_g_per_min": 0.1,
            "k_prot_absorption_to_plasma_per_min": 0.005,
            "fat_carb_slowdown_factor_per_g": 0.01, "fat_effect_duration_min": 180.0,
            "fat_glucose_effect_mg_dl_per_g_total": 0.3, "fat_is_reduction_per_g_active": 0.002,
            "fat_max_is_reduction_factor": 0.3,
            "exercise_glucose_utilization_increase_factor": 1.5,
            "exercise_insulin_sensitivity_increase_factor": 1.2,
            "exercise_carryover_duration_min": 120.0, "exercise_carryover_initial_effect_fraction": 0.5,
            "cgm_noise_sd": 2.0, "cgm_delay_minutes": 10,
            "stress_insulin_sensitivity_reduction_factor": 0.2, "stress_egp_increase_factor": 0.1,
            "illness_insulin_sensitivity_reduction_factor": 0.3, "illness_egp_increase_factor": 0.2,
            "illness_carb_absorption_reduction_factor": 0.25,
            "ISF_variability_percent": 0.15, "CR_variability_percent": 0.15,
            "basal_rate_variability_percent": 0.10,
            "carb_absorption_rate_variability_percent": 0.20,
            # PK/PD params (can be kept standard or varied too)
            "k_abs1_rapid_per_min": 1/20, "k_abs2_rapid_per_min": 1/30,
            "k_p_decay_rapid_per_min": 1/70, "k_x_prod_rapid_per_min": 1/50,
            "k_x_decay_rapid_per_min": 1/80, "iob_decay_rate_rapid_per_min": 4.6/240,
            "k_abs1_long_per_min": 1/300, "k_abs2_long_per_min": 1/300,
            "k_p_decay_long_per_min": 1/200, "k_x_prod_long_per_min": 1/180,
            "k_x_decay_long_per_min": 1/720, "iob_decay_rate_long_per_min": 4.6/1440,
        }
    },
    {
        "profile_name": "child_high_variability",
        "params": {
            "initial_glucose": 110.0,
            "ISF": 70.0, "CR": 15.0, "target_glucose": 100.0, "body_weight_kg": 40.0, # Higher ISF, higher CR for lower insulin needs per g/carb
            "basal_rate_U_hr": 0.5, # Lower basal
            "carb_absorption_rate_g_min": 0.08,
            "k_d2_to_plasma_rate_per_min": 0.022,
            "p1_glucose_clearance_rate_per_min": 0.0035,
            "k_u_id_coeff": 0.0007, "k_egp_feedback_strength": 0.006,
            "glucose_utilization_rate_mg_dl_min": 0.12,
            "protein_glucose_conversion_factor": 0.5, "protein_max_absorption_g_per_min": 0.08,
            "k_prot_absorption_to_plasma_per_min": 0.006,
            "fat_carb_slowdown_factor_per_g": 0.008, "fat_effect_duration_min": 150.0,
            "fat_glucose_effect_mg_dl_per_g_total": 0.25, "fat_is_reduction_per_g_active": 0.0015,
            "fat_max_is_reduction_factor": 0.25,
            "exercise_glucose_utilization_increase_factor": 1.6,
            "exercise_insulin_sensitivity_increase_factor": 1.3,
            "exercise_carryover_duration_min": 90.0, "exercise_carryover_initial_effect_fraction": 0.6,
            "cgm_noise_sd": 2.5, "cgm_delay_minutes": 8,
            "stress_insulin_sensitivity_reduction_factor": 0.25, "stress_egp_increase_factor": 0.12,
            "illness_insulin_sensitivity_reduction_factor": 0.35, "illness_egp_increase_factor": 0.22,
            "illness_carb_absorption_reduction_factor": 0.30,
            "ISF_variability_percent": 0.20, "CR_variability_percent": 0.20, # Higher variability
            "basal_rate_variability_percent": 0.15,
            "carb_absorption_rate_variability_percent": 0.25,
            "k_abs1_rapid_per_min": 1/15, "k_abs2_rapid_per_min": 1/25, # Faster absorption for some insulins in children
            "k_p_decay_rapid_per_min": 1/60, "k_x_prod_rapid_per_min": 1/40,
            "k_x_decay_rapid_per_min": 1/70, "iob_decay_rate_rapid_per_min": 4.6/210, # Shorter duration
            "k_abs1_long_per_min": 1/280, "k_abs2_long_per_min": 1/280,
            "k_p_decay_long_per_min": 1/180, "k_x_prod_long_per_min": 1/160,
            "k_x_decay_long_per_min": 1/680, "iob_decay_rate_long_per_min": 4.6/1320,
        }
    },
    {
        "profile_name": "adult_insulin_resistant",
        "params": {
            "initial_glucose": 130.0,
            "ISF": 30.0, "CR": 7.0, "target_glucose": 110.0, "body_weight_kg": 90.0, # Lower ISF, lower CR (more insulin needed)
            "basal_rate_U_hr": 1.5, # Higher basal
            "carb_absorption_rate_g_min": 0.06,
            "k_d2_to_plasma_rate_per_min": 0.018,
            "p1_glucose_clearance_rate_per_min": 0.0025,
            "k_u_id_coeff": 0.0004, "k_egp_feedback_strength": 0.0045, # Potentially blunted EGP feedback
            "glucose_utilization_rate_mg_dl_min": 0.08,
            "protein_glucose_conversion_factor": 0.5, "protein_max_absorption_g_per_min": 0.1,
            "k_prot_absorption_to_plasma_per_min": 0.005,
            "fat_carb_slowdown_factor_per_g": 0.012, "fat_effect_duration_min": 200.0,
            "fat_glucose_effect_mg_dl_per_g_total": 0.35, "fat_is_reduction_per_g_active": 0.0025,
            "fat_max_is_reduction_factor": 0.35,
            "exercise_glucose_utilization_increase_factor": 1.3, # Less benefit from exercise
            "exercise_insulin_sensitivity_increase_factor": 1.1, # Less IS benefit
            "exercise_carryover_duration_min": 100.0, "exercise_carryover_initial_effect_fraction": 0.4,
            "cgm_noise_sd": 1.8, "cgm_delay_minutes": 12,
            "stress_insulin_sensitivity_reduction_factor": 0.15, "stress_egp_increase_factor": 0.08,
            "illness_insulin_sensitivity_reduction_factor": 0.25, "illness_egp_increase_factor": 0.18,
            "illness_carb_absorption_reduction_factor": 0.20,
            "ISF_variability_percent": 0.10, "CR_variability_percent": 0.10,
            "basal_rate_variability_percent": 0.08,
            "carb_absorption_rate_variability_percent": 0.15,
            "k_abs1_rapid_per_min": 1/20, "k_abs2_rapid_per_min": 1/30,
            "k_p_decay_rapid_per_min": 1/70, "k_x_prod_rapid_per_min": 1/50,
            "k_x_decay_rapid_per_min": 1/80, "iob_decay_rate_rapid_per_min": 4.6/240,
            "k_abs1_long_per_min": 1/300, "k_abs2_long_per_min": 1/300,
            "k_p_decay_long_per_min": 1/200, "k_x_prod_long_per_min": 1/180,
            "k_x_decay_long_per_min": 1/720, "iob_decay_rate_long_per_min": 4.6/1440,
        }
    }
]

# --- Scenario Generation Helper ---
def generate_daily_schedule(day_number: int, patient: SyntheticPatient) -> List[Dict[str, Any]]:
    """Generates a schedule of events for a single day."""
    schedule = []
    # Simple meal schedule: breakfast, lunch, dinner
    meal_times_hours = [8, 13, 19] # Hours from start of day
    
    for meal_hour in meal_times_hours:
        # Vary meal composition
        carbs = np.random.uniform(30, 80)
        protein = carbs * np.random.uniform(0.2, 0.4)
        fat = carbs * np.random.uniform(0.1, 0.3)
        gi_factor = np.random.uniform(0.6, 1.4) # Vary GI
        
        # Simple bolus calculation (can be improved or agent-driven later)
        # Uses the patient's *current* (potentially day-varied) CR
        bolus = carbs / patient.cr if patient.cr > 0 else 0
        
        schedule.append({
            "time_offset_minutes": meal_hour * 60,
            "type": "meal_bolus",
            "carbs_details": {"grams": round(carbs,1), "gi_factor": round(gi_factor,1)},
            "protein_ingested": round(protein,1),
            "fat_ingested": round(fat,1),
            "bolus_insulin": round(bolus, 2)
        })

    # Occasional exercise
    if np.random.rand() < 0.4: # 40% chance of exercise
        exercise_start_hour = np.random.choice([10, 16, 17])
        duration = np.random.choice([30, 45, 60])
        intensity = np.random.uniform(0.7, 1.5)
        schedule.append({
            "time_offset_minutes": exercise_start_hour * 60,
            "type": "exercise",
            "exercise_event": {"duration_minutes": float(duration), "intensity_factor": round(intensity,1)}
        })

    # Occasional stress/illness (ensure they don't overlap too much for simplicity now)
    current_event_time = 0
    if np.random.rand() < 0.2: # 20% chance of stress
        stress_start_hour = np.random.randint(6, 20)
        stress_duration_hours = np.random.randint(1, 4)
        stress_level = np.random.uniform(0.2, 0.7)
        schedule.append({
            "time_offset_minutes": stress_start_hour * 60,
            "type": "set_stress", "level": round(stress_level,2), "duration_minutes": stress_duration_hours * 60
        })
        current_event_time = (stress_start_hour + stress_duration_hours) * 60
    
    if np.random.rand() < 0.15 and (current_event_time == 0 or (np.random.randint(6,20)*60 > current_event_time) ): # 15% chance of illness, try not to overlap
        illness_start_hour = np.random.randint(6, 18) # Earlier start for potentially longer duration
        illness_duration_hours = np.random.randint(4, 12)
        illness_level = np.random.uniform(0.2, 0.6)
        schedule.append({
            "time_offset_minutes": illness_start_hour * 60,
            "type": "set_illness", "level": round(illness_level,2), "duration_minutes": illness_duration_hours * 60
        })
        
    schedule.sort(key=lambda x: x["time_offset_minutes"])
    return schedule

# --- Simulation Function ---
def run_simulation_for_patient(patient_profile: Dict[str, Any], num_days: int, sim_start_time: datetime) -> pd.DataFrame:
    """Runs a simulation for a given patient profile for a number of days."""
    patient = SyntheticPatient(params=patient_profile["params"])
    
    all_data = []
    current_time = sim_start_time
    
    active_stress_end_time = None
    active_illness_end_time = None

    for day_idx in range(num_days):
        patient.reset() # Resamples ISF, CR etc. for the new "day"
        print(f"  Simulating Day {day_idx+1} for profile {patient_profile['profile_name']}...")
        print(f"    Sampled for day: ISF={patient.isf:.1f}, CR={patient.cr:.1f}, BasalNeed={patient.basal_rate_U_hr:.2f}, CarbAbsRate={patient.carb_absorption_rate_g_min:.3f}")
        
        daily_schedule = generate_daily_schedule(day_idx, patient)
        schedule_idx = 0
        
        num_steps_per_day = (24 * 60) // SIMULATION_STEP_MINUTES
        
        for step_idx in range(num_steps_per_day):
            minutes_into_day = step_idx * SIMULATION_STEP_MINUTES
            
            # Event handling
            bolus_to_give = 0.0
            carbs_to_ingest_details = None
            protein_to_ingest = 0.0
            fat_to_ingest = 0.0
            exercise_event_for_step = None
            
            # Check for scheduled events
            while schedule_idx < len(daily_schedule) and daily_schedule[schedule_idx]["time_offset_minutes"] <= minutes_into_day:
                event = daily_schedule[schedule_idx]
                print(f"    Day {day_idx+1}, {current_time.strftime('%H:%M')}: Processing event: {event['type']}")
                if event["type"] == "meal_bolus":
                    carbs_to_ingest_details = event["carbs_details"]
                    protein_to_ingest = event["protein_ingested"]
                    fat_to_ingest = event["fat_ingested"]
                    bolus_to_give = event["bolus_insulin"]
                elif event["type"] == "exercise":
                    exercise_event_for_step = event["exercise_event"]
                elif event["type"] == "set_stress":
                    patient.set_stress_level(event["level"])
                    active_stress_end_time = current_time + timedelta(minutes=event["duration_minutes"])
                    print(f"      Stress ON: level {event['level']:.2f} until {active_stress_end_time.strftime('%H:%M')}")
                elif event["type"] == "set_illness":
                    patient.set_illness_level(event["level"])
                    active_illness_end_time = current_time + timedelta(minutes=event["duration_minutes"])
                    print(f"      Illness ON: level {event['level']:.2f} until {active_illness_end_time.strftime('%H:%M')}")
                schedule_idx += 1

            # Check for event endings
            if active_stress_end_time and current_time >= active_stress_end_time:
                patient.set_stress_level(0.0)
                print(f"    Day {day_idx+1}, {current_time.strftime('%H:%M')}: Stress OFF")
                active_stress_end_time = None
            if active_illness_end_time and current_time >= active_illness_end_time:
                patient.set_illness_level(0.0)
                print(f"    Day {day_idx+1}, {current_time.strftime('%H:%M')}: Illness OFF")
                active_illness_end_time = None

            # Patient takes a step
            # Basal rate is patient's current physiological need (sampled daily)
            patient.step(
                basal_insulin=patient.basal_rate_U_hr,
                bolus_insulin=bolus_to_give,
                carbs_details=carbs_to_ingest_details,
                protein_ingested=protein_to_ingest,
                fat_ingested=fat_to_ingest,
                exercise_event=exercise_event_for_step
            )
            
            # Collect data
            cgm = patient.get_cgm_reading()
            internal_states = patient.get_internal_states()
            
            record = {
                "timestamp": current_time.isoformat(),
                "profile_name": patient_profile["profile_name"],
                "day": day_idx + 1,
                "minutes_in_day": minutes_into_day,
                "cgm_mg_dl": round(cgm, 2),
                "bolus_U": round(bolus_to_give, 2),
                "basal_U_hr_setting": round(patient.basal_rate_U_hr, 3), # The target basal for the patient this day
                "carbs_g": carbs_to_ingest_details["grams"] if carbs_to_ingest_details else 0.0,
                "carb_gi_factor": carbs_to_ingest_details["gi_factor"] if carbs_to_ingest_details else 1.0,
                "protein_g": round(protein_to_ingest,1),
                "fat_g": round(fat_to_ingest,1),
                "exercise_intensity_factor": exercise_event_for_step["intensity_factor"] if exercise_event_for_step else 0.0,
                "exercise_duration_input_min": exercise_event_for_step["duration_minutes"] if exercise_event_for_step else 0.0,
                "stress_level": round(patient.current_stress_level, 2),
                "illness_level": round(patient.current_illness_level, 2),
                "iob_U": round(internal_states.get("total_iob_U", 0.0), 2),
                "cob_g": round(internal_states.get("carbs_on_board_g", 0.0), 1),
                "current_isf": round(internal_states.get("current_isf", patient.isf),1),
                "current_cr": round(internal_states.get("current_cr", patient.cr),1),
                "current_carb_absorption_rate": round(internal_states.get("current_carb_absorption_rate_g_min", patient.carb_absorption_rate_g_min),3),
                "G_p_mg_dl": round(internal_states.get("G_p_mg_dl", 0.0), 2),
            }
            all_data.append(record)
            
            current_time += timedelta(minutes=SIMULATION_STEP_MINUTES)
            
    return pd.DataFrame(all_data)

# --- Main Execution ---
if __name__ == "__main__":
    num_simulation_days_per_profile = 2 # Generate 2 days of data per profile for now
    
    if not os.path.exists(BASE_OUTPUT_DIR):
        os.makedirs(BASE_OUTPUT_DIR)
        print(f"Created output directory: {BASE_OUTPUT_DIR}")

    simulation_run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for i, profile_config in enumerate(PATIENT_PROFILES):
        profile_name = profile_config["profile_name"]
        print(f"Starting simulation for patient profile: {profile_name} ({i+1}/{len(PATIENT_PROFILES)})")
        
        start_datetime = datetime(2024, 1, 1, 0, 0, 0) # Arbitrary start
        
        patient_data_df = run_simulation_for_patient(profile_config, num_simulation_days_per_profile, start_datetime)
        
        # Save data
        output_filename = f"{profile_name}_data_{simulation_run_timestamp}.csv"
        output_filepath = os.path.join(BASE_OUTPUT_DIR, output_filename)
        patient_data_df.to_csv(output_filepath, index=False)
        print(f"  Data for profile {profile_name} saved to: {output_filepath}")
        
        # Save metadata (patient params used for this run)
        metadata = {
            "profile_name": profile_name,
            "base_parameters": profile_config["params"],
            "simulation_run_timestamp": simulation_run_timestamp,
            "num_days_simulated": num_simulation_days_per_profile,
            "data_file": output_filename
        }
        metadata_filename = f"{profile_name}_metadata_{simulation_run_timestamp}.json"
        metadata_filepath = os.path.join(BASE_OUTPUT_DIR, metadata_filename)
        with open(metadata_filepath, 'w') as f:
            json.dump(metadata, f, indent=4)
        print(f"  Metadata for profile {profile_name} saved to: {metadata_filepath}")

    print("\nDataset generation complete.")