import numpy as np
from typing import Dict, Any

def sample_patient_params() -> Dict[str, Any]:
    """Samples a set of synthetic patient parameters from predefined distributions.

    This function defines plausible ranges and distributions for various
    physiological and model parameters to create diverse virtual patient profiles.

    Returns:
        Dict[str, Any]: A dictionary of patient parameters suitable for
                        initializing the SyntheticPatient class.
    """
    params = {}

    # Basic Patient Characteristics
    params["initial_glucose"] = np.random.uniform(90.0, 180.0)  # mg/dL
    params["ISF"] = np.random.normal(loc=50.0, scale=10.0)     # Insulin Sensitivity Factor (mg/dL/U)
    params["CR"] = np.random.normal(loc=12.0, scale=3.0)       # Carb Ratio (g/U)
    params["target_glucose"] = np.random.uniform(90.0, 120.0)  # mg/dL
    params["body_weight_kg"] = np.random.normal(loc=70.0, scale=15.0) # kg
    params["basal_rate_U_hr"] = np.random.uniform(0.5, 1.5)    # U/hr

    # Clamp ISF, CR, body_weight to be positive
    params["ISF"] = max(10.0, params["ISF"])
    params["CR"] = max(3.0, params["CR"])
    params["body_weight_kg"] = max(30.0, params["body_weight_kg"])

    # Carb Dynamics Parameters
    params["carb_absorption_rate_g_min"] = np.random.uniform(0.03, 0.08) # Rate of carb absorption from D1 to D2 (g/min)
    params["k_d2_to_plasma_rate_per_min"] = np.random.uniform(0.015, 0.035) # Rate D2 carbs affect plasma glucose (/min)

    # Insulin Dynamics Parameters (Old generic ones - now handled by specific PK/PD in SyntheticPatient)
    # params["iob_decay_rate_per_min"] = np.random.uniform(0.003, 0.007)
    # params["k_ip_decay_rate_per_min"] = np.random.uniform(0.025, 0.045)
    # params["k_x_prod_rate_per_min"] = np.random.uniform(0.004, 0.008)
    # params["insulin_action_decay_rate_per_min"] = np.random.uniform(0.015, 0.025)
    # Note: Specific PK/PD parameters for rapid/long insulin (e.g., k_abs1_rapid_per_min)
    # are currently using defaults in SyntheticPatient if not provided by sampler.
    # They could be added here for more variability if needed in the future.

    # Glucose Dynamics Parameters
    params["p1_glucose_clearance_rate_per_min"] = np.random.uniform(0.002, 0.004) # Basal glucose clearance rate (/min)
    params["k_u_id_coeff"] = np.random.uniform(0.0004, 0.0008)          # Coefficient for insulin-dependent glucose utilization
    params["k_egp_feedback_strength"] = np.random.uniform(0.004, 0.008) # Strength of EGP feedback to target glucose
    params["glucose_utilization_rate_mg_dl_min"] = np.random.uniform(0.05, 0.2) # Basal insulin-independent glucose utilization (mg/dL/min) - Reduced range

    # Bolus-specific insulin dynamics parameters (Old - now part of rapid-acting PK/PD)
    # params["bolus_absorption_factor"] = np.random.uniform(1.0, 1.5)
    # params["bolus_action_factor"] = np.random.uniform(1.0, 1.3)

    # Protein and Fat effect parameters
    params["protein_glucose_conversion_factor"] = np.random.uniform(0.4, 0.6) # g glucose / g protein
    params["protein_max_absorption_g_per_min"] = np.random.uniform(0.05, 0.15) # g of protein from stomach to gut per min
    params["k_prot_absorption_to_plasma_per_min"] = np.random.uniform(0.003, 0.007) # fraction of gut protein appearing as glucose eq in plasma per min

    params["fat_carb_slowdown_factor_per_g"] = np.random.uniform(0.005, 0.015) # % slowdown per g of fat (e.g., 0.01 = 1%)
    params["fat_effect_duration_min"] = np.random.uniform(120.0, 300.0) # Duration of fat's primary effect
    params["fat_glucose_effect_mg_dl_per_g_total"] = np.random.uniform(0.2, 0.8) # Total mg/dL rise per g fat over duration
    params["fat_is_reduction_per_g_active"] = np.random.uniform(0.001, 0.005) # IS reduction per g of active fat
    params["fat_max_is_reduction_factor"] = np.random.uniform(0.1, 0.4) # Max fractional IS reduction from fat

    # Exercise effect parameters
    params["exercise_glucose_utilization_increase_factor"] = np.random.uniform(1.2, 2.0) # e.g., 20% to 100% increase
    params["exercise_insulin_sensitivity_increase_factor"] = np.random.uniform(1.1, 1.5) # e.g., 10% to 50% increase
    params["exercise_carryover_duration_min"] = np.random.uniform(60.0, 240.0) # minutes
    params["exercise_carryover_initial_effect_fraction"] = np.random.uniform(0.3, 0.7) # fraction of during-exercise IS boost

    # CGM Characteristics
    params["cgm_noise_sd"] = np.random.uniform(1.0, 3.0)       # Standard deviation for CGM noise
    params["cgm_delay_minutes"] = int(np.random.uniform(8, 15))    # Approximate CGM sensor delay

    return params

if __name__ == '__main__':
    for i in range(3):
        sampled_params = sample_patient_params()
        print(f"\n--- Sampled Patient Profile {i+1} ---")
        for key, value in sampled_params.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")