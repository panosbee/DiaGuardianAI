# DiaGuardianAI Research Data Generation Example

import sys
import os
# Add project root to path for direct script execution
if __package__ is None or __package__ == '':
    package_path = os.path.join(os.path.dirname(__file__), '..')
    sys.path.insert(0, os.path.abspath(package_path))

from DiaGuardianAI.data_generation.synthetic_patient_model import SyntheticPatient
from DiaGuardianAI.data_generation.data_formatter import DataFormatter
# from DiaGuardianAI.utils.config import ConfigManager # Optional: for loading patient profiles
import numpy as np
import pandas as pd # For saving data to CSV or other formats

def generate_research_data(num_patients: int = 3, simulation_days_per_patient: int = 10,
                           output_format: str = "csv", output_prefix: str = "synthetic_patient_data"):
    """
    Generates synthetic patient data for research purposes.
    Simulates multiple patients and saves their data.
    Args:
        num_patients (int): Number of synthetic patients to generate.
        simulation_days_per_patient (int): Number of days to simulate for each patient.
        output_format (str): "csv", "json", "pickle" (placeholder for now, only CSV basic implemented).
        output_prefix (str): Prefix for the output filenames.
    """
    print(f"--- Starting Research Data Generation ---")
    print(f"Generating data for {num_patients} patient(s), {simulation_days_per_patient} days each.")

    all_patient_data_dfs = [] # To store DataFrames for each patient if combining later

    for i in range(num_patients):
        print(f"\nSimulating Patient {i+1}/{num_patients}...")

        # 1. Define patient parameters (could be varied for each patient)
        # Example: Randomize ISF and CR slightly for variability
        base_isf = 50
        base_cr = 10
        patient_params = {
            "ISF": base_isf + np.random.randint(-10, 11), # e.g., 40-60
            "CR": base_cr + np.random.randint(-3, 4),    # e.g., 7-13
            "initial_glucose": 120.0 + np.random.randint(-20, 21), # e.g., 100-140
            "weight": 70 + np.random.randint(-10, 11) # e.g., 60-80
        }
        patient = SyntheticPatient(params=patient_params)
        print(f"  Patient {i+1} params: {patient_params}")

        # 2. Simulation loop for this patient
        time_step_minutes = 5 # Standard simulation step
        num_steps = (simulation_days_per_patient * 24 * 60) // time_step_minutes
        
        timestamps = []
        cgm_readings = []
        basal_rates_applied = [] # U/hr
        bolus_doses_applied = [] # U
        carbs_ingested_g = []    # g
        # exercise_events_info = [] # Could be more complex

        current_basal_rate = 0.8 # Example starting basal rate (U/hr)
        start_datetime = pd.Timestamp("2023-01-01T00:00:00") # Ensure it's a fixed start time

        for step_num in range(num_steps):
            current_time = start_datetime + pd.Timedelta(minutes=step_num * time_step_minutes)
            timestamps.append(current_time)

            # Simple placeholder logic for insulin and carbs (not an intelligent agent)
            # This section would be replaced by an actual closed-loop or open-loop therapy regimen.
            bolus_this_step = 0.0
            carbs_this_step = 0.0
            
            # Example: Simulate meals at certain times
            hour_of_day = current_time.hour
            if hour_of_day == 8 and current_time.minute == 0: # Breakfast
                carbs_this_step = 50.0 + np.random.randint(-10, 11)
                bolus_this_step = carbs_this_step / patient_params["CR"] # Simple carb bolus
            elif hour_of_day == 13 and current_time.minute == 0: # Lunch
                carbs_this_step = 70.0 + np.random.randint(-15, 16)
                bolus_this_step = carbs_this_step / patient_params["CR"]
            elif hour_of_day == 19 and current_time.minute == 0: # Dinner
                carbs_this_step = 60.0 + np.random.randint(-10, 11)
                bolus_this_step = carbs_this_step / patient_params["CR"]

            # Example: Simulate small correction bolus if BG is high (very naive)
            current_cgm_for_decision = patient.get_cgm_reading() # Get current CGM before step
            if current_cgm_for_decision > 200 and bolus_this_step == 0: # If no meal bolus
                bolus_this_step += 1.0 # Small correction

            # Apply to patient model
            patient.step(basal_insulin=current_basal_rate,
                         bolus_insulin=bolus_this_step,
                         carbs_ingested=carbs_this_step,
                         exercise_event=None) # Exercise can be added similarly

            # Record data *after* the step
            cgm_readings.append(patient.get_cgm_reading())
            basal_rates_applied.append(current_basal_rate)
            bolus_doses_applied.append(bolus_this_step)
            carbs_ingested_g.append(carbs_this_step)
            
            if (step_num + 1) % (24 * 60 // time_step_minutes) == 0: # Every day
                print(f"  Patient {i+1}: Day { (step_num + 1) // (24 * 60 // time_step_minutes) } simulated. Last CGM: {cgm_readings[-1]:.1f}")


        # 3. Store or save data for this patient
        patient_df = pd.DataFrame({
            "timestamp": timestamps,
            "cgm_mg_dl": cgm_readings,
            "basal_u_hr": basal_rates_applied,
            "bolus_u": bolus_doses_applied,
            "carbs_g": carbs_ingested_g,
            "patient_id": f"synthetic_patient_{i+1}"
        })
        all_patient_data_dfs.append(patient_df)

        if output_format == "csv":
            filename = f"{output_prefix}_patient_{i+1}.csv"
            patient_df.to_csv(filename, index=False)
            print(f"  Data for Patient {i+1} saved to {filename}")
        # Add other formats like JSON, Parquet, etc. later
        # elif output_format == "json":
        #     filename = f"{output_prefix}_patient_{i+1}.json"
        #     patient_df.to_json(filename, orient="records", indent=4, date_format="iso")
        #     print(f"  Data for Patient {i+1} saved to {filename}")


    # Optionally, combine all data into one file (e.g., for easier loading later)
    if len(all_patient_data_dfs) > 1:
        combined_df = pd.concat(all_patient_data_dfs, ignore_index=True)
        combined_filename = f"{output_prefix}_all_patients_combined.{output_format if output_format=='csv' else 'csv'}" # Default to csv for combined
        if output_format == "csv":
            combined_df.to_csv(combined_filename, index=False)
            print(f"\nCombined data for all patients saved to {combined_filename}")
        # Can add combined saving for other formats too.

    # 4. (Optional) Format data using DataFormatter for model training
    # This would typically be a separate step after data generation.
    # Example:
    # if len(all_patient_data_dfs) > 0:
    #     first_patient_df = all_patient_data_dfs[0]
    #     formatter = DataFormatter(
    #         cgm_time_step_minutes=5, # Ensure this matches data
    #         prediction_horizons_minutes=[30, 60, 120],
    #         history_window_minutes=180
    #     )
    #     # Timestamps for create_dataset should be list of datetime objects or DatetimeIndex
    #     # The DataFrame already has 'timestamp' as datetime objects from pd.Timestamp and pd.Timedelta
    #     timestamps_for_formatter = first_patient_df["timestamp"].tolist()
    #     # Or directly: first_patient_df["timestamp"] if it's already a DatetimeIndex and create_dataset handles it

    #     features, targets = formatter.create_dataset(
    #         cgm_data=first_patient_df["cgm_mg_dl"].tolist(),
    #         insulin_data=first_patient_df["bolus_u"].tolist(),
    #         carb_data=first_patient_df["carbs_g"].tolist(),
    #         timestamps=timestamps_for_formatter
    #     )
    #     if features.size > 0:
    #         print(f"\nFormatted features for Patient 1 (example): Shape {features.shape}")
    #         print(f"Formatted targets for Patient 1 (example): Shape {targets.shape}")
    #     else:
    #         print("\nCould not generate formatted features/targets for Patient 1 (likely not enough data for window/horizon).")

    print(f"\n--- Research Data Generation Finished ---")
    return all_patient_data_dfs


if __name__ == "__main__":
    # Ensure imports work when run as a script
    if __package__ is None or __package__ == '':
        package_path = os.path.join(os.path.dirname(__file__), '..')
        sys.path.insert(0, os.path.abspath(package_path))
        # Re-import to ensure they are found after path modification if needed by main execution
        from DiaGuardianAI.data_generation.synthetic_patient_model import SyntheticPatient
        from DiaGuardianAI.data_generation.data_formatter import DataFormatter

    generated_data = generate_research_data(num_patients=2, simulation_days_per_patient=3, output_format="csv")
    
    if generated_data:
        print(f"\nSuccessfully generated data for {len(generated_data)} patient(s).")
        # print("First few rows of the first patient's data:")
        # print(generated_data[0].head())