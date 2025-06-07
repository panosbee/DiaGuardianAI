# DiaGuardianAI Open-Loop Simulation Example

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Ensure the DiaGuardianAI package is discoverable
if __package__ is None or __package__ == '':
    # If run as a script, add project root to path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from DiaGuardianAI.data_generation.synthetic_patient_model import SyntheticPatient
from DiaGuardianAI.utils.patient_sampler import sample_patient_params

def run_open_loop_simulation_example(use_sampled_params: bool = True, simulation_hours: int = 8):
    """
    Runs a simple open-loop simulation with the SyntheticPatient model
    and plots the results, including protein, fat, and exercise effects.

    Args:
        use_sampled_params (bool): If True, uses randomly sampled patient parameters.
                                   Otherwise, uses a fixed default set.
        simulation_hours (int): The duration of the simulation in hours.
    """
    print("--- Starting DiaGuardianAI Open-Loop Simulation Example (with Protein/Fat/Exercise) ---")

    # 1. Initialize Patient
    if use_sampled_params:
        print("\n1. Initializing patient with SAMPLER parameters...")
        patient_params = sample_patient_params()
    else:
        print("\n1. Initializing patient with DEFAULT parameters...")
        patient_params = {
            "initial_glucose": 120.0, "ISF": 50.0, "CR": 10.0, "target_glucose": 100.0,
            "body_weight_kg": 70.0, "basal_rate_U_hr": 1.0,
            "carb_absorption_rate_g_min": 0.05, "k_d2_to_plasma_rate_per_min": 0.02,
            # "iob_decay_rate_per_min": 0.005, "k_ip_decay_rate_per_min": 0.03, # Old params
            # "k_x_prod_rate_per_min": 0.005, "insulin_action_decay_rate_per_min": 0.02, # Old params
            "p1_glucose_clearance_rate_per_min": 0.003, "k_u_id_coeff": 0.0005, # k_u_id_coeff now applies to total X
            "k_egp_feedback_strength": 0.005, "glucose_utilization_rate_mg_dl_min": 0.1,
            # "bolus_absorption_factor": 1.0, "bolus_action_factor": 1.0, # Old params
            "protein_glucose_conversion_factor": 0.5,
            "protein_max_absorption_g_per_min": 0.1, # New protein param
            "k_prot_absorption_to_plasma_per_min": 0.005, # New protein param
            "fat_carb_slowdown_factor_per_g": 0.01,
            "fat_effect_duration_min": 180.0,
            "fat_glucose_effect_mg_dl_per_g_total": 0.3,
            "fat_is_reduction_per_g_active": 0.002, # New fat param
            "fat_max_is_reduction_factor": 0.3, # New fat param
            # Add new PK/PD params for default case if not sampled
            "k_abs1_rapid_per_min": 1/20, "k_abs2_rapid_per_min": 1/30,
            "k_p_decay_rapid_per_min": 1/70, "k_x_prod_rapid_per_min": 1/50,
            "k_x_decay_rapid_per_min": 1/80, "iob_decay_rate_rapid_per_min": 4.6/240,
            "k_abs1_long_per_min": 1/300, "k_abs2_long_per_min": 1/300,
            "k_p_decay_long_per_min": 1/200, "k_x_prod_long_per_min": 1/180,
            "k_x_decay_long_per_min": 1/720, "iob_decay_rate_long_per_min": 4.6/1440,
            "exercise_glucose_utilization_increase_factor": 1.5,
            "exercise_insulin_sensitivity_increase_factor": 1.2,
            "exercise_carryover_duration_min": 120.0, # New exercise param
            "exercise_carryover_initial_effect_fraction": 0.5, # New exercise param
            "cgm_noise_sd": 1.5, "cgm_delay_minutes": 10
        }
    patient = SyntheticPatient(params=patient_params)
    print(f"   Patient Parameters (first few):")
    for k, v in list(patient_params.items())[:8]: # Print a few more
        print(f"     {k}: {v}")
    print(f"   Initial G_p: {patient.G_p:.2f}, Initial CGM: {patient.get_cgm_reading():.2f}")

    # 2. Define Simulation Scenario
    time_step_minutes = patient.dt_minutes
    total_steps = (simulation_hours * 60) // time_step_minutes
    
    basal_rate_U_hr = patient_params.get("basal_rate_U_hr", 1.0)
    carb_ratio = patient_params.get("CR", 10.0)
    if carb_ratio == 0: carb_ratio = 10.0 

    # Scenario events
    meal_time_step = (2 * 60) // time_step_minutes 
    meal_carbs = 50.0 
    meal_protein = 25.0 
    meal_fat = 15.0 
    
    exercise_start_step = (4 * 60) // time_step_minutes # Exercise at Hour 4
    exercise_details_event = {"duration_minutes": 45.0, "intensity_factor": 1.0} # Moderate exercise

    # Data collection lists
    time_points_minutes = []
    cgm_readings = []
    gp_values = []
    iob_rapid_values = []
    iob_long_values = []
    iob_total_values = []
    cob_values = []
    prot_g2_values = []
    active_fat_values = []
    exercise_intensity_values = []
    exercise_duration_values = []
    exercise_carryover_rem_values = [] # New
    exercise_carryover_factor_values = [] # New
    insulin_administered_bolus = []
    carbs_ingested_values = []
    protein_ingested_values = []
    fat_ingested_values = []

    print(f"\n2. Running simulation for {simulation_hours} hours ({total_steps} steps)...")
    for step_num in range(total_steps):
        current_time_minutes = step_num * time_step_minutes
        
        bolus_insulin_U = 0.0
        carbs_g = 0.0
        protein_g = 0.0
        fat_g = 0.0
        current_exercise_event = None

        if step_num == meal_time_step:
            carbs_g = meal_carbs
            protein_g = meal_protein
            fat_g = meal_fat
            bolus_insulin_U = meal_carbs / carb_ratio
            print(f"   Time {current_time_minutes/60:.1f}h: Meal C={carbs_g}g, P={protein_g}g, F={fat_g}g, Bolus {bolus_insulin_U:.2f}U (Rapid)")
        
        if step_num == exercise_start_step:
            current_exercise_event = exercise_details_event
            print(f"   Time {current_time_minutes/60:.1f}h: Exercise Event Triggered: Duration={exercise_details_event['duration_minutes']}min, Intensity={exercise_details_event['intensity_factor']}")
        
        patient.step(
            basal_insulin=basal_rate_U_hr,  # Assumed to be long-acting
            bolus_insulin=bolus_insulin_U, # Assumed to be rapid-acting
            carbs_ingested=carbs_g,
            protein_ingested=protein_g,
            fat_ingested=fat_g,
            exercise_event=current_exercise_event
        )

        time_points_minutes.append(current_time_minutes)
        cgm_readings.append(patient.get_cgm_reading())
        gp_values.append(patient.G_p)
        iob_rapid_values.append(patient.iob_rapid)
        iob_long_values.append(patient.iob_long)
        iob_total_values.append(patient.iob_rapid + patient.iob_long)
        cob_values.append(patient.cob)
        prot_g2_values.append(patient.Prot_G2)
        active_fat_values.append(patient.active_fat_g)
        exercise_intensity_values.append(patient.active_exercise_intensity_factor)
        exercise_duration_values.append(patient.exercise_duration_remaining_min)
        exercise_carryover_rem_values.append(patient.exercise_carryover_remaining_min) # New
        exercise_carryover_factor_values.append(1.0 + patient.current_exercise_carryover_additional_is_factor) # New, plot as multiplier
        insulin_administered_bolus.append(bolus_insulin_U)
        carbs_ingested_values.append(carbs_g)
        protein_ingested_values.append(protein_g)
        fat_ingested_values.append(fat_g)

        if step_num > 0 and step_num % (60 // time_step_minutes) == 0:
             print(f"   Hour {current_time_minutes/60:.0f}: CGM={cgm_readings[-1]:.1f}, G_p={gp_values[-1]:.1f}, IOB_tot={iob_total_values[-1]:.2f} (R:{iob_rapid_values[-1]:.2f} L:{iob_long_values[-1]:.2f}), COB={cob_values[-1]:.1f}, ProtG2={prot_g2_values[-1]:.1f}, ActiveFat={active_fat_values[-1]:.1f}, ExFactor={exercise_intensity_values[-1]:.1f}, ExCarryRem={exercise_carryover_rem_values[-1]:.1f}, ExCarryFactor={exercise_carryover_factor_values[-1]:.3f}")

    # 3. Plot Results
    print("\n3. Plotting results...")
    fig, axs = plt.subplots(7, 1, figsize=(14, 24), sharex=True) 
    time_hours = np.array(time_points_minutes) / 60.0
    bar_width_time_units = 0.8 * (time_step_minutes / 60.0)

    # Glucose Plot
    axs[0].plot(time_hours, cgm_readings, label="CGM (mg/dL)", color="dodgerblue", linewidth=2)
    axs[0].plot(time_hours, gp_values, label="Plasma Glucose (G_p, mg/dL)", color="skyblue", linestyle="--", linewidth=1.5)
    axs[0].set_ylabel("Glucose (mg/dL)")
    axs[0].legend(loc="upper right")
    axs[0].grid(True, linestyle=':', alpha=0.7)
    axs[0].set_title(f"Open-Loop Simulation ({'Sampled' if use_sampled_params else 'Default'} Params) - {simulation_hours} Hours")

    # IOB Plot
    axs[1].plot(time_hours, iob_total_values, label="Total IOB (U)", color="crimson", linewidth=2)
    axs[1].plot(time_hours, iob_rapid_values, label="Rapid IOB (U)", color="salmon", linestyle="--", linewidth=1)
    axs[1].plot(time_hours, iob_long_values, label="Long IOB (U)", color="lightcoral", linestyle=":", linewidth=1)
    axs[1].set_ylabel("IOB (U)")
    axs[1].legend(loc="upper right")
    axs[1].grid(True, linestyle=':', alpha=0.7)

    # COB & Protein On Board (Prot_G2) Plot
    axs[2].plot(time_hours, cob_values, label="Carbs on Board (D1+D2, g)", color="forestgreen", linewidth=2)
    axs[2].plot(time_hours, prot_g2_values, label="Protein (Prot_G2, g-eq)", color="saddlebrown", linestyle="--", linewidth=1.5)
    axs[2].set_ylabel("On Board (g or g-eq)")
    axs[2].legend(loc="upper right")
    axs[2].grid(True, linestyle=':', alpha=0.7)

    # Active Fat Plot
    axs[3].plot(time_hours, active_fat_values, label="Active Fat (g)", color="goldenrod", linewidth=2)
    axs[3].set_ylabel("Active Fat (g)")
    axs[3].legend(loc="upper right")
    axs[3].grid(True, linestyle=':', alpha=0.7)
    
    # Insulin Bolus Plot
    axs[4].bar(time_hours, insulin_administered_bolus, width=bar_width_time_units, label="Bolus Insulin (U)", color="darkorchid", alpha=0.8)
    axs[4].set_ylabel("Bolus (U)")
    axs[4].legend(loc="upper right")
    axs[4].grid(True, linestyle=':', alpha=0.7)

    # Macronutrient Intake Plot
    axs[5].bar(time_hours, carbs_ingested_values, width=bar_width_time_units*0.9, label="Carbs (g)", color="darkorange", alpha=0.7, align='center')
    ax5_twin = axs[5].twinx() 
    ax5_twin.bar(time_hours - bar_width_time_units*0.33, protein_ingested_values, width=bar_width_time_units*0.3, label="Protein (g)", color="deepskyblue", alpha=0.6, align='edge')
    ax5_twin.bar(time_hours + bar_width_time_units*0.33, fat_ingested_values, width=bar_width_time_units*0.3, label="Fat (g)", color="lightcoral", alpha=0.6, align='edge')
    axs[5].set_ylabel("Carbs (g)") 
    ax5_twin.set_ylabel("Protein/Fat (g)")
    lines, labels = axs[5].get_legend_handles_labels()
    lines2, labels2 = ax5_twin.get_legend_handles_labels()
    axs[5].legend(lines + lines2, labels + labels2, loc="upper right")
    axs[5].grid(True, linestyle=':', alpha=0.7)
    
    # Exercise Plot
    axs[6].plot(time_hours, exercise_intensity_values, label="Active Ex. Intensity Factor", color="limegreen", linewidth=2)
    axs[6].plot(time_hours, exercise_carryover_factor_values, label="Ex. Carry-over IS Factor", color="lightgreen", linestyle=":", linewidth=2) # New
    ax6_twin_exercise = axs[6].twinx()
    ax6_twin_exercise.plot(time_hours, exercise_duration_values, label="Active Ex. Duration Rem. (min)", color="teal", linestyle="--", linewidth=1.5)
    ax6_twin_exercise.plot(time_hours, exercise_carryover_rem_values, label="Ex. Carry-over Rem. (min)", color="darkcyan", linestyle="-.", linewidth=1.5) # New
    axs[6].set_ylabel("Intensity/IS Factor", color="limegreen")
    ax6_twin_exercise.set_ylabel("Duration Rem. (min)", color="teal")
    axs[6].tick_params(axis='y', labelcolor="limegreen")
    ax6_twin_exercise.tick_params(axis='y', labelcolor="teal")
    lines_ex, labels_ex = axs[6].get_legend_handles_labels()
    lines2_ex, labels2_ex = ax6_twin_exercise.get_legend_handles_labels()
    axs[6].legend(lines_ex + lines2_ex, labels_ex + labels2_ex, loc="upper right")
    axs[6].grid(True, linestyle=':', alpha=0.7)
    axs[6].set_ylim(bottom=0) # Ensure factors don't go below 0 visually if not expected
    ax6_twin_exercise.set_ylim(bottom=0)


    axs[-1].set_xlabel("Time (hours)")
    fig.suptitle("DiaGuardianAI: Synthetic Patient Open-Loop Simulation (with Protein/Fat/Exercise)", fontsize=16)
    plt.tight_layout(rect=(0, 0, 1, 0.96)) 
    plt.show()

    print("\n--- DiaGuardianAI Open-Loop Simulation Example Finished ---")

if __name__ == "__main__":
    run_open_loop_simulation_example(use_sampled_params=True, simulation_hours=8)
    # To run with default, predictable parameters:
    # run_open_loop_simulation_example(use_sampled_params=False, simulation_hours=8)