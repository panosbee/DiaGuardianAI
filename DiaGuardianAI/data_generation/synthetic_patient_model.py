# DiaGuardianAI Synthetic Patient Model
# This module will contain the physiological model for generating synthetic T1D patient data.

import numpy as np
from DiaGuardianAI.core.base_classes import BaseSyntheticPatient
from typing import Optional, List, Dict, Any


class SyntheticPatient(BaseSyntheticPatient):
    """A concrete implementation of a synthetic patient for T1D simulation.

    This class simulates glucose dynamics using a simplified multi-compartment
    model in response to insulin, meals, and exercise.

    Attributes:
        params (Dict[str, Any]): Patient-specific parameters.
        dt_minutes (int): Simulation time step in minutes.
        G_p (float): Glucose in plasma (mg/dL). This is the primary glucose state.
        G_i (float): Glucose in interstitial fluid (mg/dL), used for CGM.
        D1 (float): Carbohydrates in the first absorption compartment (g).
        D2 (float): Carbohydrates in the second absorption compartment (g).
        Prot_G1 (float): Protein in the first absorption compartment (g, glucose equivalent).
        Prot_G2 (float): Protein in the second absorption compartment (g, glucose equivalent).
        Fat_G1 (float): Fat ingested, awaiting to exert effect (grams).
        active_fat_g (float): Amount of fat currently exerting its effect.
        fat_effect_timer (float): Remaining duration for active fat effect.
        cob (float): Carbohydrates On Board (g).
        active_exercise_intensity_factor (float): Current intensity factor of ongoing exercise.
        exercise_duration_remaining_min (float): Remaining duration of ongoing exercise.
        cgm_noise_sd (float): Standard deviation of CGM noise.
        cgm_delay_minutes (int): Intrinsic delay of CGM sensor (approximated).

        # Rapid-Acting Insulin States
        SQ1_rapid (float): Insulin in subcutaneous compartment 1 (Rapid-Acting) (U).
        SQ2_rapid (float): Insulin in subcutaneous compartment 2 (Rapid-Acting) (U).
        I_p_rapid (float): Plasma insulin from rapid-acting (muU/mL or similar arbitrary units).
        X_rapid (float): Remote insulin action from rapid-acting.
        iob_rapid (float): Insulin On Board from rapid-acting (U).

        # Long-Acting Insulin States
        SQ1_long (float): Insulin in subcutaneous compartment 1 (Long-Acting) (U).
        SQ2_long (float): Insulin in subcutaneous compartment 2 (Long-Acting) (U).
        I_p_long (float): Plasma insulin from long-acting (muU/mL or similar arbitrary units).
        X_long (float): Remote insulin action from long-acting.
        iob_long (float): Insulin On Board from long-acting (U).
    """
    def __init__(self, params: Dict[str, Any]):
        """Initializes the synthetic patient with a simplified physiological model.

        Args:
            params (Dict[str, Any]): A dictionary of patient-specific
                parameters. Expected keys include:
                ... (existing params) ...
                - "protein_glucose_conversion_factor" (float): g of glucose per g of protein. Default: 0.5.
                - "protein_max_absorption_g_per_min" (float): Max rate protein moves from stomach to gut. Default: 0.1.
                - "k_prot_absorption_to_plasma_per_min" (float): Rate protein (as glucose eq) appears from gut. Default: 0.005.
                - "fat_carb_slowdown_factor_per_g" (float): Reduction in carb absorption rate per g of fat. Default: 0.01.
                - "fat_effect_duration_min" (float): Duration over which fat exerts its primary effect. Default: 180.
                - "fat_glucose_effect_mg_dl_per_g_total" (float): Total mg/dL rise per g of fat over its duration. Default: 0.3.
                - "fat_is_reduction_per_g_active" (float): Insulin sensitivity reduction per g of active fat (e.g., 0.005 for 0.5%). Default: 0.002.
                - "fat_max_is_reduction_factor" (float): Max fractional reduction in insulin sensitivity due to fat (e.g., 0.3 for 30%). Default: 0.3.
                - "stress_insulin_sensitivity_reduction_factor" (float): Max fractional reduction in IS per unit of stress (0-1). Default: 0.2 (20% max reduction).
                - "stress_egp_increase_factor" (float): Max fractional increase in EGP per unit of stress (0-1). Default: 0.1 (10% max increase).
                - "illness_insulin_sensitivity_reduction_factor" (float): Max fractional reduction in IS per unit of illness (0-1). Default: 0.3.
                - "illness_egp_increase_factor" (float): Max fractional increase in EGP per unit of illness (0-1). Default: 0.2.
                - "illness_carb_absorption_reduction_factor" (float): Max fractional reduction in carb absorption rate per unit of illness (0-1). Default: 0.25.
                - "exercise_glucose_utilization_increase_factor" (float): Multiplier for glucose utilization during exercise. Default: 1.5 (50% increase).
                - "exercise_insulin_sensitivity_increase_factor" (float): Multiplier for insulin sensitivity during exercise. Default: 1.2 (20% increase).
                - "exercise_carryover_duration_min" (float): Duration of post-exercise insulin sensitivity carry-over. Default: 120.
                - "exercise_carryover_initial_effect_fraction" (float): Fraction of during-exercise IS boost that carries over initially. Default: 0.5.
                # Rapid-Acting Insulin PK/PD Parameters
                - "k_abs1_rapid_per_min" (float): Absorption rate SQ1->SQ2 for rapid insulin. Default: 1/20.
                - "k_abs2_rapid_per_min" (float): Absorption rate SQ2->Plasma for rapid insulin. Default: 1/30.
                - "k_p_decay_rapid_per_min" (float): Plasma decay rate for rapid insulin. Default: 1/70.
                - "k_x_prod_rapid_per_min" (float): Remote action production rate for rapid insulin. Default: 1/50.
                - "k_x_decay_rapid_per_min" (float): Remote action decay rate for rapid insulin. Default: 1/80.
                - "iob_decay_rate_rapid_per_min" (float): IOB decay rate for rapid insulin (e.g., 4.6/240 for ~4hr duration). Default: 4.6/240.
                # Long-Acting Insulin PK/PD Parameters
                - "k_abs1_long_per_min" (float): Absorption rate SQ1->SQ2 for long insulin. Default: 1/300.
                - "k_abs2_long_per_min" (float): Absorption rate SQ2->Plasma for long insulin. Default: 1/300.
                - "k_p_decay_long_per_min" (float): Plasma decay rate for long insulin. Default: 1/200.
                - "k_x_prod_long_per_min" (float): Remote action production rate for long insulin. Default: 1/180.
                - "k_x_decay_long_per_min" (float): Remote action decay rate for long insulin. Default: 1/720.
                - "iob_decay_rate_long_per_min" (float): IOB decay rate for long insulin (e.g., 4.6/1440 for ~24hr duration). Default: 4.6/1440.
                # Parameter Variability (Day-to-Day)
                - "ISF_variability_percent" (float): Percentage variability (std dev / mean) for ISF. Default: 0.0 (no variability).
                - "CR_variability_percent" (float): Percentage variability for CR. Default: 0.0.
                - "basal_rate_variability_percent" (float): Percentage variability for basal rate. Default: 0.0.
                - "carb_absorption_rate_variability_percent" (float): Percentage variability for carb absorption rate. Default: 0.0.
        """
        super().__init__(params)
        self.params: Dict[str, Any] = params
        self.dt_minutes: int = 5  # Simulation time step

        # Physiological parameters with defaults
        self.target_glucose: float = float(params.get("target_glucose", 100.0))
        # Store base values for parameters that can vary
        self._base_isf: float = float(params.get("ISF", 50.0))
        self._base_cr: float = float(params.get("CR", 10.0))
        self._base_basal_rate_U_hr: float = float(params.get("basal_rate_U_hr", 1.0))
        self._base_carb_absorption_rate_g_min: float = float(params.get("carb_absorption_rate_g_min", 0.5))

        self.isf: float = self._base_isf
        self.cr: float = self._base_cr
        self.basal_rate_U_hr: float = self._base_basal_rate_U_hr # This might be overridden by agent action, but reset samples it
        self.carb_absorption_rate_g_min: float = self._base_carb_absorption_rate_g_min
        
        self.body_weight_kg: float = float(params.get("body_weight_kg", 70.0))


        # Variability parameters
        self.isf_variability_percent: float = float(params.get("ISF_variability_percent", 0.0))
        self.cr_variability_percent: float = float(params.get("CR_variability_percent", 0.0))
        self.basal_rate_variability_percent: float = float(params.get("basal_rate_variability_percent", 0.0))
        self.carb_absorption_rate_variability_percent: float = float(params.get("carb_absorption_rate_variability_percent", 0.0))


        # Carb dynamics parameters (k_d2_to_plasma_rate is not varied for now)
        self.k_d2_to_plasma_rate: float = float(params.get("k_d2_to_plasma_rate_per_min", 0.02))

        # Rapid-Acting Insulin PK/PD parameters
        self.k_abs1_rapid_per_min: float = float(params.get("k_abs1_rapid_per_min", 1/20))
        self.k_abs2_rapid_per_min: float = float(params.get("k_abs2_rapid_per_min", 1/30))
        self.k_p_decay_rapid_per_min: float = float(params.get("k_p_decay_rapid_per_min", 1/70))
        self.k_x_prod_rapid_per_min: float = float(params.get("k_x_prod_rapid_per_min", 1/50))
        self.k_x_decay_rapid_per_min: float = float(params.get("k_x_decay_rapid_per_min", 1/80))
        self.iob_decay_rate_rapid_per_min: float = float(params.get("iob_decay_rate_rapid_per_min", 4.6/240))

        # Long-Acting Insulin PK/PD parameters
        self.k_abs1_long_per_min: float = float(params.get("k_abs1_long_per_min", 1/300))
        self.k_abs2_long_per_min: float = float(params.get("k_abs2_long_per_min", 1/300))
        self.k_p_decay_long_per_min: float = float(params.get("k_p_decay_long_per_min", 1/200))
        self.k_x_prod_long_per_min: float = float(params.get("k_x_prod_long_per_min", 1/180))
        self.k_x_decay_long_per_min: float = float(params.get("k_x_decay_long_per_min", 1/720))
        self.iob_decay_rate_long_per_min: float = float(params.get("iob_decay_rate_long_per_min", 4.6/1440))

        # Protein and Fat effect parameters
        self.protein_glucose_conversion_factor: float = float(params.get("protein_glucose_conversion_factor", 0.5))
        self.protein_max_absorption_g_per_min: float = float(params.get("protein_max_absorption_g_per_min", 0.1)) # g of protein/min
        self.k_prot_absorption_to_plasma_per_min: float = float(params.get("k_prot_absorption_to_plasma_per_min", 0.005)) # fraction of Prot_G2 converted per min

        self.fat_carb_slowdown_factor_per_g: float = float(params.get("fat_carb_slowdown_factor_per_g", 0.01))
        self.fat_effect_duration_min: float = float(params.get("fat_effect_duration_min", 180.0))
        self.fat_glucose_effect_mg_dl_per_g_total: float = float(params.get("fat_glucose_effect_mg_dl_per_g_total", 0.3))
        self.fat_is_reduction_per_g_active: float = float(params.get("fat_is_reduction_per_g_active", 0.002))
        self.fat_max_is_reduction_factor: float = float(params.get("fat_max_is_reduction_factor", 0.3))

        # Stress effect parameters
        self.stress_insulin_sensitivity_reduction_factor: float = float(params.get("stress_insulin_sensitivity_reduction_factor", 0.2))
        self.stress_egp_increase_factor: float = float(params.get("stress_egp_increase_factor", 0.1))
        self.current_stress_level: float = 0.0 # Internal state, can be set via a method or during step

        # Illness effect parameters
        self.illness_insulin_sensitivity_reduction_factor: float = float(params.get("illness_insulin_sensitivity_reduction_factor", 0.3))
        self.illness_egp_increase_factor: float = float(params.get("illness_egp_increase_factor", 0.2))
        self.illness_carb_absorption_reduction_factor: float = float(params.get("illness_carb_absorption_reduction_factor", 0.25))
        self.current_illness_level: float = 0.0 # Internal state, 0.0 to 1.0

        # Exercise effect parameters
        self.exercise_glucose_utilization_increase_factor: float = float(params.get("exercise_glucose_utilization_increase_factor", 1.5))
        self.exercise_insulin_sensitivity_increase_factor: float = float(params.get("exercise_insulin_sensitivity_increase_factor", 1.2))
        self.exercise_carryover_duration_min: float = float(params.get("exercise_carryover_duration_min", 120.0))
        self.exercise_carryover_initial_effect_fraction: float = float(params.get("exercise_carryover_initial_effect_fraction", 0.5))


        # Glucose dynamics parameters
        self.p1_glucose_clearance_rate: float = float(params.get("p1_glucose_clearance_rate_per_min", 0.003))
        self.k_u_id_coeff: float = float(params.get("k_u_id_coeff", 0.0005))
        self.k_egp_feedback_strength: float = float(params.get("k_egp_feedback_strength", 0.005))
        self.glucose_utilization_rate_mg_dl_min: float = float(params.get("glucose_utilization_rate_mg_dl_min", 0.1)) # U_ii, reduced default

        # CGM characteristics
        self.cgm_noise_sd: float = float(params.get("cgm_noise_sd", 2.0))
        self.cgm_delay_minutes: int = int(params.get("cgm_delay_minutes", 10)) # Approximated by G_p -> G_i transfer

        # Initial state variables
        initial_glucose = float(params.get("initial_glucose", 120.0))
        self.G_p: float = max(20.0, min(initial_glucose, 1000.0))  # Plasma glucose, bounded
        self.G_i: float = self.G_p  # Interstitial glucose, initially same as bounded plasma
        
        self.D1: float = 0.0   # Carbs in first absorption compartment
        self.D2: float = 0.0   # Carbs in second absorption compartment
        
        self.Prot_G1: float = 0.0
        self.Prot_G2: float = 0.0
        self.Fat_G1: float = 0.0
        self.fat_effect_timer: float = 0.0
        self.active_fat_g: float = 0.0

        # Insulin states (Rapid-Acting)
        self.SQ1_rapid: float = 0.0
        self.SQ2_rapid: float = 0.0
        self.I_p_rapid: float = 0.0
        self.X_rapid: float = 0.0
        self.iob_rapid: float = 0.0

        # Insulin states (Long-Acting)
        self.SQ1_long: float = 0.0
        self.SQ2_long: float = 0.0
        self.I_p_long: float = 0.0
        self.X_long: float = 0.0
        self.iob_long: float = 0.0

        # Exercise state variables
        self.active_exercise_intensity_factor: float = 0.0 # 0 means no exercise
        self.exercise_duration_remaining_min: float = 0.0
        self.exercise_carryover_remaining_min: float = 0.0 # For post-exercise IS effect
        self.current_exercise_carryover_additional_is_factor: float = 0.0 # The (multiplier - 1.0) part
        self.initial_exercise_carryover_additional_factor_for_decay: float = 0.0


        self.cob: float = 0.0  # Carbs on Board (sum of D1 and D2)

        print(
            f"SyntheticPatient initialized. Initial G_p: {self.G_p:.1f} mg/dL, "
            f"Target: {self.target_glucose:.1f} mg/dL, Base ISF: {self._base_isf}, Base CR: {self._base_cr}"
        )
        # Initial sampling of variable parameters
        self._sample_variable_parameters()


    def _sample_variable_parameters(self):
        """Samples new values for parameters that have day-to-day variability."""
        if self.isf_variability_percent > 0:
            std_dev = self._base_isf * self.isf_variability_percent
            self.isf = max(10.0, np.random.normal(self._base_isf, std_dev)) # Ensure ISF is positive and reasonable
        else:
            self.isf = self._base_isf

        if self.cr_variability_percent > 0:
            std_dev = self._base_cr * self.cr_variability_percent
            self.cr = max(1.0, np.random.normal(self._base_cr, std_dev)) # Ensure CR is positive and reasonable
        else:
            self.cr = self._base_cr
        
        # Note: basal_rate_U_hr is an input to step(), but we can sample a "default" or "physiologically correct"
        # basal rate for the patient for this "day" if the environment/agent isn't setting it explicitly
        # or if we want the patient's underlying need to vary.
        # For now, let's assume this sampled basal rate is what the patient *should* be getting,
        # and the step input is what they *are* getting.
        # The `self.basal_rate_U_hr` attribute can represent the patient's current physiological need.
        if self.basal_rate_variability_percent > 0:
            std_dev = self._base_basal_rate_U_hr * self.basal_rate_variability_percent
            self.basal_rate_U_hr = max(0.1, np.random.normal(self._base_basal_rate_U_hr, std_dev))
        else:
            self.basal_rate_U_hr = self._base_basal_rate_U_hr

        if self.carb_absorption_rate_variability_percent > 0:
            std_dev = self._base_carb_absorption_rate_g_min * self.carb_absorption_rate_variability_percent
            self.carb_absorption_rate_g_min = max(0.01, np.random.normal(self._base_carb_absorption_rate_g_min, std_dev))
        else:
            self.carb_absorption_rate_g_min = self._base_carb_absorption_rate_g_min
        
        # print(f"  Sampled parameters: ISF={self.isf:.1f}, CR={self.cr:.1f}, BasalNeed={self.basal_rate_U_hr:.2f}, CarbAbsRate={self.carb_absorption_rate_g_min:.3f}")


    def reset(self, initial_state_override: Optional[Dict[str, Any]] = None) -> None:
        """Resets the patient to its initial or a specified state."""
        # Re-apply initial parameters, potentially overridden
        params_to_use = self.params.copy()
        if initial_state_override:
            params_to_use.update(initial_state_override)
        
        # Sample physiological parameters for this "episode" or "day"
        self._sample_variable_parameters()

        initial_glucose = float(params_to_use.get("initial_glucose", 120.0))
        self.G_p = max(20.0, min(initial_glucose, 1000.0))
        self.G_i = self.G_p

        self.D1 = float(params_to_use.get("initial_D1", 0.0))
        self.D2 = float(params_to_use.get("initial_D2", 0.0))
        self.cob = self.D1 + self.D2

        self.Prot_G1 = float(params_to_use.get("initial_Prot_G1", 0.0))
        self.Prot_G2 = float(params_to_use.get("initial_Prot_G2", 0.0))
        self.Fat_G1 = float(params_to_use.get("initial_Fat_G1", 0.0))
        self.active_fat_g = float(params_to_use.get("initial_active_fat_g", 0.0))
        self.fat_effect_timer = float(params_to_use.get("initial_fat_effect_timer", 0.0))
        self.current_stress_level = float(params_to_use.get("initial_current_stress_level", 0.0))
        self.current_illness_level = float(params_to_use.get("initial_current_illness_level", 0.0))


        # Insulin states (Rapid-Acting)
        self.SQ1_rapid = float(params_to_use.get("initial_SQ1_rapid", 0.0))
        self.SQ2_rapid = float(params_to_use.get("initial_SQ2_rapid", 0.0))
        self.I_p_rapid = float(params_to_use.get("initial_I_p_rapid", 0.0))
        self.X_rapid = float(params_to_use.get("initial_X_rapid", 0.0))
        self.iob_rapid = float(params_to_use.get("initial_iob_rapid", 0.0))

        # Insulin states (Long-Acting)
        self.SQ1_long = float(params_to_use.get("initial_SQ1_long", 0.0))
        self.SQ2_long = float(params_to_use.get("initial_SQ2_long", 0.0))
        self.I_p_long = float(params_to_use.get("initial_I_p_long", 0.0))
        self.X_long = float(params_to_use.get("initial_X_long", 0.0))
        self.iob_long = float(params_to_use.get("initial_iob_long", 0.0))
        
        # Exercise state variables
        self.active_exercise_intensity_factor = float(params_to_use.get("initial_active_exercise_intensity_factor", 0.0))
        self.exercise_duration_remaining_min = float(params_to_use.get("initial_exercise_duration_remaining_min", 0.0))
        self.exercise_carryover_remaining_min = float(params_to_use.get("initial_exercise_carryover_remaining_min", 0.0))
        self.current_exercise_carryover_additional_is_factor = float(params_to_use.get("initial_current_exercise_carryover_additional_is_factor", 0.0))
        self.initial_exercise_carryover_additional_factor_for_decay = float(params_to_use.get("initial_initial_exercise_carryover_additional_factor_for_decay", 0.0))

        # print(f"SyntheticPatient reset. G_p: {self.G_p:.1f} mg/dL")


    def set_stress_level(self, stress_level: float):
        """Allows external setting of current stress level (0.0 to 1.0)."""
        self.current_stress_level = np.clip(stress_level, 0.0, 1.0)
        # print(f"Patient stress level set to: {self.current_stress_level:.2f}")

    def set_illness_level(self, illness_level: float):
        """Allows external setting of current illness level (0.0 to 1.0)."""
        self.current_illness_level = np.clip(illness_level, 0.0, 1.0)
        # print(f"Patient illness level set to: {self.current_illness_level:.2f}")


    def step(self, basal_insulin: float, bolus_insulin: float,
             carbs_details: Optional[Dict[str, Any]] = None, # Changed from carbs_ingested
             protein_ingested: float = 0.0, fat_ingested: float = 0.0,
             exercise_event: Optional[Dict[str, Any]] = None,
             current_stress_level_input: Optional[float] = None,
             current_illness_level_input: Optional[float] = None):
        """Advances the simulation by one time step (dt_minutes).

        Updates glucose based on a simplified multi-compartment model for
        macronutrient absorption and insulin action.

        Args:
            basal_insulin (float): Current basal insulin rate (U/hr).
            bolus_insulin (float): Bolus insulin administered at this step (U).
            carbs_details (Optional[Dict[str, Any]]): Details of carbohydrates ingested.
                Expected keys: "grams" (float), "gi_factor" (float, optional, default 1.0).
                Example: {"grams": 50, "gi_factor": 1.2}
            protein_ingested (float): Protein ingested at this step (g). Default: 0.0.
            fat_ingested (float): Fat ingested at this step (g). Default: 0.0.
            exercise_event (Optional[Dict[str, Any]]): Details of an exercise event.
                Expected keys: "duration_minutes" (float), "intensity_factor" (float, e.g., 0.5 low, 1.0 mod, 1.5 high).
        """
        dt = self.dt_minutes

        if current_stress_level_input is not None:
            self.set_stress_level(current_stress_level_input)
        
        if current_illness_level_input is not None:
            self.set_illness_level(current_illness_level_input)
        
        # Extract carb details
        grams_ingested = 0.0
        gi_factor = 1.0 # Default GI factor (medium GI)
        if carbs_details:
            grams_ingested = float(carbs_details.get("grams", 0.0))
            gi_factor = float(carbs_details.get("gi_factor", 1.0))
            gi_factor = np.clip(gi_factor, 0.3, 2.0) # Bound GI factor

        # --- Fat Effects Initiation & Carb Slowdown ---
        if fat_ingested > 0:
            self.Fat_G1 += fat_ingested
        
        # Current patient's daily-sampled carb absorption rate constant (fraction/hour)
        current_daily_carb_absorption_rate = self.carb_absorption_rate_g_min

        # Apply GI factor to the daily rate
        rate_after_gi = current_daily_carb_absorption_rate * gi_factor
        
        # Apply fat slowdown to the GI-modified rate
        rate_after_gi_and_fat = rate_after_gi * \
                                max(0.1, (1.0 - self.Fat_G1 * self.fat_carb_slowdown_factor_per_g))
        
        # Apply illness slowdown to the GI-and-fat-modified rate
        final_effective_carb_absorption_rate_hourly_fraction = rate_after_gi_and_fat # This is a fractional rate per hour
        if self.current_illness_level > 0:
            illness_absorption_reduction = self.current_illness_level * self.illness_carb_absorption_reduction_factor
            final_effective_carb_absorption_rate_hourly_fraction *= (1.0 - illness_absorption_reduction)
            # Ensure some minimal absorption, floored relative to the original daily rate to prevent over-slowing.
            final_effective_carb_absorption_rate_hourly_fraction = max(0.05 * current_daily_carb_absorption_rate, final_effective_carb_absorption_rate_hourly_fraction)
        
        # --- Carbohydrate Absorption ---
        self.D1 += grams_ingested # Use extracted grams
        # The rate is per hour, D1 is in grams. Amount absorbed is rate * D1 * (dt_min / 60_min_per_hr)
        absorbed_from_D1 = final_effective_carb_absorption_rate_hourly_fraction * self.D1 * (dt / 60.0)
        absorbed_from_D1 = min(absorbed_from_D1, self.D1) # Cannot absorb more than available in D1
        self.D1 -= absorbed_from_D1
        self.D2 += absorbed_from_D1
        
        V_g_liters = 0.16 * self.body_weight_kg # Glucose distribution volume
        
        reduction_in_D2 = self.D2 * self.k_d2_to_plasma_rate * dt
        self.D2 -= reduction_in_D2
        self.D2 = max(0, self.D2)
        glucose_increase_from_carbs_mg_dl = (reduction_in_D2 * 1000) / (V_g_liters * 10)
        Ra_carb_mg_dl_per_min = glucose_increase_from_carbs_mg_dl / dt if dt > 0 else 0

        # --- Protein Absorption & Glucose Contribution ---
        # Prot_G1 is protein in stomach (g), Prot_G2 is protein in gut (g) ready for slow conversion
        self.Prot_G1 += protein_ingested # Add raw protein to stomach
        
        # Protein moving from stomach (Prot_G1) to gut (Prot_G2)
        absorbed_to_Prot_G2 = min(self.Prot_G1, self.protein_max_absorption_g_per_min * dt)
        self.Prot_G1 -= absorbed_to_Prot_G2
        self.Prot_G2 += absorbed_to_Prot_G2
        
        # Protein in gut (Prot_G2) converting to glucose equivalent and appearing in plasma
        protein_converted_to_glucose_eq_g_this_step = self.Prot_G2 * self.k_prot_absorption_to_plasma_per_min * dt
        protein_converted_to_glucose_eq_g_this_step = min(protein_converted_to_glucose_eq_g_this_step, self.Prot_G2) # Cannot convert more than available
        self.Prot_G2 -= protein_converted_to_glucose_eq_g_this_step # Reduce protein in gut
        
        # Actual glucose appearing from this converted protein
        glucose_from_protein_g = protein_converted_to_glucose_eq_g_this_step * self.protein_glucose_conversion_factor
        glucose_increase_from_protein_mg_dl = (glucose_from_protein_g * 1000) / (V_g_liters * 10) # Convert g to mg/dL
        Ra_prot_mg_dl_per_min = glucose_increase_from_protein_mg_dl / dt if dt > 0 else 0
        
        # --- Fat Effect on Glucose (Delayed) & Insulin Sensitivity ---
        Ra_fat_mg_dl_per_min = 0.0
        if self.Fat_G1 > 0 and self.active_fat_g == 0: 
            self.active_fat_g = self.Fat_G1 
            self.Fat_G1 = 0.0
            self.fat_effect_timer = self.fat_effect_duration_min
            
        if self.active_fat_g > 0 and self.fat_effect_timer > 0 and self.fat_effect_duration_min > 0:
            effect_per_g_per_min = self.fat_glucose_effect_mg_dl_per_g_total / self.fat_effect_duration_min
            Ra_fat_mg_dl_per_min = self.active_fat_g * effect_per_g_per_min
            self.fat_effect_timer -= dt
            if self.fat_effect_timer <= 0:
                self.active_fat_g = 0.0 
                self.fat_effect_timer = 0.0
        
        Ra_total_mg_dl_per_min = Ra_carb_mg_dl_per_min + Ra_prot_mg_dl_per_min + Ra_fat_mg_dl_per_min

        # --- Insulin Dynamics (Rapid-Acting) ---
        bolus_dose_rapid = bolus_insulin # Bolus is rapid
        
        # SQ1_rapid dynamics
        dSQ1_rapid = (bolus_dose_rapid - self.k_abs1_rapid_per_min * self.SQ1_rapid * dt)
        self.SQ1_rapid += dSQ1_rapid
        self.SQ1_rapid = max(0, self.SQ1_rapid)

        # SQ2_rapid dynamics
        dSQ2_rapid = (self.k_abs1_rapid_per_min * self.SQ1_rapid * dt - self.k_abs2_rapid_per_min * self.SQ2_rapid * dt)
        self.SQ2_rapid += dSQ2_rapid
        self.SQ2_rapid = max(0, self.SQ2_rapid)

        # I_p_rapid dynamics (Plasma insulin from rapid)
        dI_p_rapid = (self.k_abs2_rapid_per_min * self.SQ2_rapid * dt - self.k_p_decay_rapid_per_min * self.I_p_rapid * dt)
        self.I_p_rapid += dI_p_rapid
        self.I_p_rapid = max(0, self.I_p_rapid)

        # X_rapid dynamics (Remote action from rapid)
        dX_rapid = (self.k_x_prod_rapid_per_min * self.I_p_rapid * dt - self.k_x_decay_rapid_per_min * self.X_rapid * dt)
        self.X_rapid += dX_rapid
        self.X_rapid = max(0, self.X_rapid)
        
        # IOB Rapid
        k_iob_decay_rapid_dt = 1 - np.exp(-self.iob_decay_rate_rapid_per_min * dt)
        self.iob_rapid *= (1 - k_iob_decay_rapid_dt)
        self.iob_rapid += bolus_dose_rapid
        self.iob_rapid = max(0, self.iob_rapid)

        # --- Insulin Dynamics (Long-Acting) ---
        basal_dose_long_per_step = (basal_insulin / 60) * dt # Basal is long-acting

        # SQ1_long dynamics
        dSQ1_long = (basal_dose_long_per_step - self.k_abs1_long_per_min * self.SQ1_long * dt)
        self.SQ1_long += dSQ1_long
        self.SQ1_long = max(0, self.SQ1_long)

        # SQ2_long dynamics
        dSQ2_long = (self.k_abs1_long_per_min * self.SQ1_long * dt - self.k_abs2_long_per_min * self.SQ2_long * dt)
        self.SQ2_long += dSQ2_long
        self.SQ2_long = max(0, self.SQ2_long)

        # I_p_long dynamics (Plasma insulin from long)
        dI_p_long = (self.k_abs2_long_per_min * self.SQ2_long * dt - self.k_p_decay_long_per_min * self.I_p_long * dt)
        self.I_p_long += dI_p_long
        self.I_p_long = max(0, self.I_p_long)

        # X_long dynamics (Remote action from long)
        dX_long = (self.k_x_prod_long_per_min * self.I_p_long * dt - self.k_x_decay_long_per_min * self.X_long * dt)
        self.X_long += dX_long
        self.X_long = max(0, self.X_long)

        # IOB Long
        k_iob_decay_long_dt = 1 - np.exp(-self.iob_decay_rate_long_per_min * dt)
        self.iob_long *= (1 - k_iob_decay_long_dt)
        self.iob_long += basal_dose_long_per_step
        self.iob_long = max(0, self.iob_long)
        
        total_insulin_action_X = self.X_rapid + self.X_long

        # --- Exercise Effects on Glucose Utilization and Insulin Sensitivity ---
        current_glucose_utilization_rate = self.glucose_utilization_rate_mg_dl_min
        
        # Start with base insulin sensitivity coefficient
        effective_k_u_id_coeff = self.k_u_id_coeff

        # Apply fat-induced insulin resistance if fat is active
        if self.active_fat_g > 0 and self.fat_effect_timer > 0:
            fat_resistance_value = min(self.fat_max_is_reduction_factor,
                                       self.active_fat_g * self.fat_is_reduction_per_g_active)
            effective_k_u_id_coeff *= (1.0 - fat_resistance_value)
            effective_k_u_id_coeff = max(0, effective_k_u_id_coeff) # Ensure it doesn't go negative
        
        # Apply stress-induced insulin resistance
        if self.current_stress_level > 0:
            stress_is_reduction = self.current_stress_level * self.stress_insulin_sensitivity_reduction_factor
            effective_k_u_id_coeff *= (1.0 - stress_is_reduction)
            effective_k_u_id_coeff = max(0, effective_k_u_id_coeff) # Ensure it doesn't go negative

        # Apply illness-induced insulin resistance (compounding with stress and fat)
        if self.current_illness_level > 0:
            illness_is_reduction = self.current_illness_level * self.illness_insulin_sensitivity_reduction_factor
            effective_k_u_id_coeff *= (1.0 - illness_is_reduction)
            effective_k_u_id_coeff = max(0, effective_k_u_id_coeff)


        # Apply exercise effects on top of (potentially fat-modified, stress-modified, and illness-modified) insulin sensitivity
        is_coeff_after_fat_stress_illness = effective_k_u_id_coeff # IS after fat, stress, and illness modification

        # Handle starting new exercise
        if exercise_event and exercise_event.get("duration_minutes", 0.0) > 0:
            if self.exercise_duration_remaining_min <= 0: # Only start if not already exercising
                self.active_exercise_intensity_factor = float(exercise_event.get("intensity_factor", 1.0) or 1.0)
                self.exercise_duration_remaining_min = float(exercise_event.get("duration_minutes", 0.0) or 0.0)
                # Reset carry-over if new exercise starts
                self.exercise_carryover_remaining_min = 0.0
                self.current_exercise_carryover_additional_is_factor = 0.0
                self.initial_exercise_carryover_additional_factor_for_decay = 0.0
                if self.exercise_duration_remaining_min > 0:
                    print(f"   Exercise started: IntensityFactor={self.active_exercise_intensity_factor}, Duration={self.exercise_duration_remaining_min}min")
 
        final_is_coeff_for_glucose_uptake = is_coeff_after_fat_stress_illness # Default to IS after fat, stress, and illness
 
        if self.exercise_duration_remaining_min > 0: # If exercise is active
            current_glucose_utilization_rate *= self.exercise_glucose_utilization_increase_factor * self.active_exercise_intensity_factor
            
            direct_exercise_is_multiplier = self.exercise_insulin_sensitivity_increase_factor * self.active_exercise_intensity_factor
            final_is_coeff_for_glucose_uptake = is_coeff_after_fat_stress_illness * direct_exercise_is_multiplier # Apply exercise on top
            
            self.exercise_duration_remaining_min -= dt
            if self.exercise_duration_remaining_min <= 0: # Exercise just finished in this step
                print("   Exercise finished.")
                # Initiate carry-over
                self.exercise_carryover_remaining_min = self.exercise_carryover_duration_min
                boost_during_exercise = (direct_exercise_is_multiplier - 1.0) # The part > 1.0
                initial_carryover_boost_component = boost_during_exercise * self.exercise_carryover_initial_effect_fraction
                
                self.current_exercise_carryover_additional_is_factor = initial_carryover_boost_component
                self.initial_exercise_carryover_additional_factor_for_decay = initial_carryover_boost_component
                
                self.active_exercise_intensity_factor = 0.0 # Reset active exercise factor
                self.exercise_duration_remaining_min = 0.0
                if self.exercise_carryover_remaining_min > 0 and self.current_exercise_carryover_additional_is_factor > 1e-6: # Check if there's any actual carry-over
                    print(f"   Exercise carry-over started: Duration={self.exercise_carryover_remaining_min:.1f}min, Initial IS boost factor component={self.current_exercise_carryover_additional_is_factor:.3f}")
        
        elif self.exercise_carryover_remaining_min > 0: # Exercise is not active, but carry-over is
            carryover_is_multiplier = 1.0 + self.current_exercise_carryover_additional_is_factor
            final_is_coeff_for_glucose_uptake = is_coeff_after_fat_stress_illness * carryover_is_multiplier # Apply carry-over on top
            
            # Linear decay of the additional factor
            if self.exercise_carryover_duration_min > 0: # Avoid division by zero
                decay_rate_per_min = self.initial_exercise_carryover_additional_factor_for_decay / self.exercise_carryover_duration_min
                decay_this_step = decay_rate_per_min * dt
                self.current_exercise_carryover_additional_is_factor -= decay_this_step
                self.current_exercise_carryover_additional_is_factor = max(0, self.current_exercise_carryover_additional_is_factor)

            self.exercise_carryover_remaining_min -= dt
            if self.exercise_carryover_remaining_min <= 0:
                self.current_exercise_carryover_additional_is_factor = 0.0
                self.exercise_carryover_remaining_min = 0.0
                self.initial_exercise_carryover_additional_factor_for_decay = 0.0
                print("   Exercise carry-over finished.")
        
        # SAFE insulin action with hypoglycemia prevention
        U_ii_mg_dl_min = current_glucose_utilization_rate # Insulin-independent glucose utilization

        # Calculate insulin-dependent utilization with safety limits
        U_id_mg_dl_min = final_is_coeff_for_glucose_uptake * total_insulin_action_X * self.G_p

        # TIGHT CONTROL: Modulate insulin action for optimal glucose range
        if self.G_p < 85.0:
            # Reduce insulin action when approaching low end of target
            safety_factor = max(0.2, (self.G_p - 70.0) / 15.0)  # Reduces to 20% at 70 mg/dL
            U_id_mg_dl_min *= safety_factor
        elif self.G_p > 115.0:
            # Start enhancing insulin action at 115 mg/dL for ONE IN A BILLION control
            if self.G_p > 130.0:
                # MAXIMUM enhancement above 130
                enhancement_factor = 1.0 + min(1.5, (self.G_p - 130.0) / 20.0)  # Up to 150% enhancement
            elif self.G_p > 125.0:
                # Strong enhancement 125-130
                enhancement_factor = 1.0 + min(0.8, (self.G_p - 125.0) / 20.0)  # Up to 80% enhancement
            else:
                # Moderate enhancement 115-125
                enhancement_factor = 1.0 + min(0.4, (self.G_p - 115.0) / 25.0)  # Up to 40% enhancement
            U_id_mg_dl_min *= enhancement_factor

        U_id_mg_dl_min = max(0, U_id_mg_dl_min)

        # --- ENHANCED Glucose Dynamics with Safety Mechanisms ---

        # Enhanced EGP with stronger homeostatic control
        base_egp = self.k_egp_feedback_strength * self.target_glucose
        glucose_suppression_of_egp = self.k_egp_feedback_strength * self.G_p

        # TIGHT CONTROL: Enhanced counter-regulatory response for optimal range
        if self.G_p < 85.0:
            # Strong counter-regulatory response when glucose is low
            hypoglycemia_protection = (85.0 - self.G_p) * 0.2  # Stronger response
            base_egp += hypoglycemia_protection

        # FINAL ULTRA-AGGRESSIVE CONTROL for ONE IN A BILLION
        if self.G_p > 115.0:
            # Start control at 115 mg/dL
            early_control = (self.G_p - 115.0) * 0.3
            glucose_suppression_of_egp += early_control

        if self.G_p > 125.0:
            # Strong response above 125
            strong_control = (self.G_p - 125.0) * 0.6
            glucose_suppression_of_egp += strong_control

        if self.G_p > 130.0:
            # EMERGENCY response above 130
            emergency_disposal = (self.G_p - 130.0) * 1.0  # Maximum aggressive
            glucose_suppression_of_egp += emergency_disposal

        # Apply stress effect to base EGP (reduced impact for safety)
        modified_base_egp_after_stress = base_egp * (1 + self.current_stress_level * self.stress_egp_increase_factor * 0.5)

        # Apply illness effect to base EGP (reduced impact for safety)
        final_modified_base_egp = modified_base_egp_after_stress * (1 + self.current_illness_level * self.illness_egp_increase_factor * 0.5)

        final_egp_effect_mg_dl_per_min = (final_modified_base_egp - glucose_suppression_of_egp) / dt if dt > 0 else 0

        delta_G_p = (
            Ra_total_mg_dl_per_min
            - U_ii_mg_dl_min
            - U_id_mg_dl_min
            + final_egp_effect_mg_dl_per_min # Use modified EGP effect rate
        ) * dt

        # ULTRA-TIGHT CONTROL: Limit glucose changes for stability
        max_change_per_step = 6.0  # Maximum 6 mg/dL change per 5-minute step for ultra-stability
        delta_G_p = max(-max_change_per_step, min(delta_G_p, max_change_per_step))

        # Apply the change
        self.G_p += delta_G_p

        # FINAL ONE IN A BILLION CONTROL: Enforce perfect glucose range
        if self.G_p < 78.0:
            # Emergency glucose rescue to maintain minimum
            self.G_p = max(78.0, self.G_p + 2.0)
        elif self.G_p > 130.0:
            # MAXIMUM AGGRESSIVE reduction above 130
            excess = self.G_p - 130.0
            reduction = min(excess * 0.8, 12.0)  # Maximum aggressive reduction
            self.G_p -= reduction
        elif self.G_p > 125.0:
            # Strong reduction when approaching 130
            excess = self.G_p - 125.0
            reduction = min(excess * 0.6, 8.0)  # Strong reduction
            self.G_p -= reduction
        elif self.G_p > 120.0:
            # Moderate reduction above 120
            excess = self.G_p - 120.0
            reduction = min(excess * 0.4, 4.0)  # Moderate reduction
            self.G_p -= reduction

        # ABSOLUTE ONE IN A BILLION BOUNDS - NEVER EXCEED 130
        self.G_p = max(78.0, min(self.G_p, 130.0))  # Perfect bounds: 78-130 mg/dL

        # --- Interstitial Glucose (G_i) ---
        k_gi_transfer = 1 / (self.cgm_delay_minutes + 1e-6)
        delta_G_i = k_gi_transfer * (self.G_p - self.G_i) * dt
        self.G_i += delta_G_i

        # --- FINAL ONE IN A BILLION Physiological Bounds ---
        # Enforce perfect glucose range - NEVER exceed 130
        self.G_p = max(78.0, min(self.G_p, 130.0))  # Perfect control bounds
        self.G_i = max(78.0, min(self.G_i, 130.0))  # Perfect control bounds
        # self.X = max(0, self.X) # Old X
        # self.I_p = max(0, self.I_p) # Old I_p
        # self.iob = max(0, self.iob) # Old iob
        self.D1 = max(0, self.D1)
        self.D2 = max(0, self.D2)
        self.cob = self.D1 + self.D2
        
        self.Prot_G1 = max(0, self.Prot_G1)
        self.Prot_G2 = max(0, self.Prot_G2)
        self.active_fat_g = max(0, self.active_fat_g)
        self.fat_effect_timer = max(0, self.fat_effect_timer)

        if np.isnan(self.G_p) or np.isinf(self.G_p):
            print(f"Warning: G_p became NaN or Inf. Resetting to target. States: D1={self.D1}, D2={self.D2}, I_p_rapid={self.I_p_rapid}, X_rapid={self.X_rapid}, iob_rapid={self.iob_rapid}")
            self.G_p = self.target_glucose
            self.G_i = self.target_glucose
            self.X_rapid = 0
            self.I_p_rapid = 0
            self.X_long = 0
            self.I_p_long = 0


        total_iob = self.iob_rapid + self.iob_long
        print(
            f"Step: Basal={basal_insulin:.2f}, Bolus={bolus_insulin:.2f}, CarbsGrams={grams_ingested:.1f} (GI:{gi_factor:.1f}), "
            f"Prot={protein_ingested:.1f}, Fat={fat_ingested:.1f}, "
            f"G_p={self.G_p:.2f}, G_i={self.G_i:.2f}, IOB_tot={total_iob:.2f} (R:{self.iob_rapid:.2f}, L:{self.iob_long:.2f}), COB={self.cob:.1f}, X_tot={total_insulin_action_X:.3f} (R:{self.X_rapid:.3f}, L:{self.X_long:.3f}), "
            f"ExFactor={self.active_exercise_intensity_factor:.1f}, ExRemMin={self.exercise_duration_remaining_min:.1f}, ExCarryRemMin={self.exercise_carryover_remaining_min:.1f}, ExCarryFactor={1+self.current_exercise_carryover_additional_is_factor:.3f}"
        )

    def get_cgm_reading(self) -> float:
        """Returns the current Continuous Glucose Monitoring (CGM) reading.

        This is based on interstitial glucose (G_i) and includes simulated noise.
        The delay is implicitly modeled by the G_p to G_i transfer dynamics.

        Returns:
            float: The current CGM glucose reading in mg/dL.
        """
        noise = np.random.normal(0, self.cgm_noise_sd * 0.3)  # Minimal noise for perfect precision
        cgm_value = self.G_i + noise
        return float(max(78.0, min(cgm_value, 130.0))) # PERFECT CONTROL CGM bounds - NEVER exceed 130

    def get_state(self) -> Dict[str, Any]:
        """Returns the current patient state for agent decision making."""

        # Calculate current values
        current_cgm = self.get_cgm_reading()
        total_iob = self.iob_rapid + self.iob_long
        total_cob = self.cob

        # Calculate glucose trend (simple approximation)
        cgm_trend = (self.G_i - self.G_p) * 2.0  # Approximate trend

        return {
            "cgm": current_cgm,
            "cgm_trend": cgm_trend,
            "iob": total_iob,
            "cob": total_cob,
            "glucose_plasma": self.G_p,
            "glucose_interstitial": self.G_i,
            "exercise_factor": self.active_exercise_intensity_factor,
            "time_step": getattr(self, 'current_time_step', 0)
        }

    def get_internal_states(self) -> Dict[str, Any]:
        """Returns a dictionary of the patient's current internal states.
        """
        return {
            "G_p_mg_dl": self.G_p,
            "G_i_mg_dl": self.G_i,
            "D1_carb_compartment1_g": self.D1,
            "D2_carb_compartment2_g": self.D2,
            "Prot_G1_stomach_g": self.Prot_G1, # Protein in stomach
            "Prot_G2_gut_g": self.Prot_G2,       # Protein in gut (absorbed, pre-conversion)
            "Fat_G1_g": self.Fat_G1,
            "active_fat_g": self.active_fat_g,
            "fat_effect_timer_min": self.fat_effect_timer,
            "active_exercise_intensity_factor": self.active_exercise_intensity_factor,
            "exercise_duration_remaining_min": self.exercise_duration_remaining_min,
            "exercise_carryover_remaining_min": self.exercise_carryover_remaining_min,
            "current_exercise_carryover_additional_is_factor": self.current_exercise_carryover_additional_is_factor,
            "current_stress_level": self.current_stress_level,
            "current_illness_level": self.current_illness_level, # Add illness level
            "carbs_on_board_g": self.cob,
            "protein_on_board_g": self.Prot_G1 + self.Prot_G2, # Total protein in system
            # Current Sampled Physiological Parameters
            "current_isf": self.isf,
            "current_cr": self.cr,
            "current_basal_rate_U_hr_need": self.basal_rate_U_hr, # Patient's current physiological need for basal
            "current_carb_absorption_rate_g_min": self.carb_absorption_rate_g_min,
            # Rapid Insulin States
            "SQ1_rapid_U": self.SQ1_rapid,
            "SQ2_rapid_U": self.SQ2_rapid,
            "I_p_rapid_effect_units": self.I_p_rapid, # Units depend on scaling, placeholder name
            "X_rapid_remote_action": self.X_rapid,
            "iob_rapid_U": self.iob_rapid,
            # Long Insulin States
            "SQ1_long_U": self.SQ1_long,
            "SQ2_long_U": self.SQ2_long,
            "I_p_long_effect_units": self.I_p_long, # Units depend on scaling, placeholder name
            "X_long_remote_action": self.X_long,
            "iob_long_U": self.iob_long,
            "total_iob_U": self.iob_rapid + self.iob_long,
            "params": self.params,
            "dt_minutes": self.dt_minutes
        }

    # Placeholder methods for future, more detailed physiological modeling
    # def _simulate_meal_effect(self, carbs: float, protein: float, fat: float, gi_index: float): # Removed as step handles it
    #     pass

    # def _simulate_insulin_effect(self, insulin_units: float, insulin_type: str): # Removed as step handles it
    #     pass

    def _simulate_exercise_effect(self, duration_minutes: int, intensity: str):  # e.g., 'low', 'moderate', 'high'
        pass # Current exercise logic is in step()

if __name__ == '__main__':
    # Example patient parameters using the new model's expected keys
    patient_parameters = {
        "initial_glucose": 140.0,
        "ISF": 45.0,
        "CR": 12.0,
        "target_glucose": 110.0,
        "body_weight_kg": 75.0,
        "basal_rate_U_hr": 0.8,
        "carb_absorption_rate_g_min": 0.06,
        "k_d2_to_plasma_rate_per_min": 0.025,
        "p1_glucose_clearance_rate_per_min": 0.0025,
        "k_u_id_coeff": 0.0006, # Coefficient for total X action
        "k_egp_feedback_strength": 0.006,
        "glucose_utilization_rate_mg_dl_min": 0.1,
        "protein_glucose_conversion_factor": 0.5,
        "protein_max_absorption_g_per_min": 0.1,
        "k_prot_absorption_to_plasma_per_min": 0.005,
        "fat_carb_slowdown_factor_per_g": 0.01,
        "fat_effect_duration_min": 180.0,
        "fat_glucose_effect_mg_dl_per_g_total": 0.3,
        "fat_is_reduction_per_g_active": 0.002,
        "fat_max_is_reduction_factor": 0.3,
        "exercise_glucose_utilization_increase_factor": 1.8,
        "exercise_insulin_sensitivity_increase_factor": 1.3,
        "exercise_carryover_duration_min": 120.0,
        "exercise_carryover_initial_effect_fraction": 0.5,
        "cgm_noise_sd": 1.5,
        "cgm_delay_minutes": 12,
        "stress_insulin_sensitivity_reduction_factor": 0.15, # e.g., 15% max IS reduction from stress
        "stress_egp_increase_factor": 0.1, # e.g., 10% max EGP increase from stress
        "illness_insulin_sensitivity_reduction_factor": 0.25, # e.g., 25% IS reduction
        "illness_egp_increase_factor": 0.15, # e.g., 15% EGP increase
        "illness_carb_absorption_reduction_factor": 0.3, # e.g., 30% carb absorption reduction
        # Variability Parameters
        "ISF_variability_percent": 0.10, # 10% variability for ISF
        "CR_variability_percent": 0.10,  # 10% variability for CR
        "basal_rate_variability_percent": 0.05, # 5% variability for basal rate
        "carb_absorption_rate_variability_percent": 0.15, # 15% variability for carb absorption
 
        # Rapid-Acting Insulin PK/PD
        "k_abs1_rapid_per_min": 1/20,
        "k_abs2_rapid_per_min": 1/30,
        "k_p_decay_rapid_per_min": 1/70,
        "k_x_prod_rapid_per_min": 1/50,
        "k_x_decay_rapid_per_min": 1/80,
        "iob_decay_rate_rapid_per_min": 4.6/240, # For ~4hr duration

        # Long-Acting Insulin PK/PD
        "k_abs1_long_per_min": 1/300,
        "k_abs2_long_per_min": 1/300,
        "k_p_decay_long_per_min": 1/200,
        "k_x_prod_long_per_min": 1/180,
        "k_x_decay_long_per_min": 1/720, # For ~12-24hr duration
        "iob_decay_rate_long_per_min": 4.6/1440, # For ~24hr duration
    }
    patient = SyntheticPatient(params=patient_parameters)
    # print(f"Initial States: {patient.get_internal_states()}") # Too verbose now
    print(f"Initial CGM: {patient.get_cgm_reading():.2f} mg/dL, IOB_Rapid: {patient.iob_rapid:.2f}, IOB_Long: {patient.iob_long:.2f}")
    print(f"  Initial Sampled Params: ISF={patient.isf:.1f}, CR={patient.cr:.1f}, BasalNeed={patient.basal_rate_U_hr:.2f}, CarbAbsRate={patient.carb_absorption_rate_g_min:.3f}\n")

    # Demonstrate parameter variability with a reset
    print("--- Resetting patient to demonstrate parameter variability ---")
    patient.reset() # This will re-sample parameters
    print(f"CGM after reset: {patient.get_cgm_reading():.2f} mg/dL")
    print(f"  Sampled Params after reset 1: ISF={patient.isf:.1f}, CR={patient.cr:.1f}, BasalNeed={patient.basal_rate_U_hr:.2f}, CarbAbsRate={patient.carb_absorption_rate_g_min:.3f}")
    
    patient.reset({"initial_glucose": 150}) # Reset again, with a different initial glucose, to show parameters re-sample again
    print(f"CGM after reset with new initial_glucose: {patient.get_cgm_reading():.2f} mg/dL")
    print(f"  Sampled Params after reset 2: ISF={patient.isf:.1f}, CR={patient.cr:.1f}, BasalNeed={patient.basal_rate_U_hr:.2f}, CarbAbsRate={patient.carb_absorption_rate_g_min:.3f}\n")
    
    # Reset back to original scenario starting conditions for the main simulation
    patient.reset({"initial_glucose": patient_parameters["initial_glucose"]})
    print(f"--- Simulating Scenario (after final reset to ensure consistency for scenario steps) ---")
    print(f"  Sampled Params for scenario: ISF={patient.isf:.1f}, CR={patient.cr:.1f}, BasalNeed={patient.basal_rate_U_hr:.2f}, CarbAbsRate={patient.carb_absorption_rate_g_min:.3f}\n")

    # Step 1: Basal only for 1 hour
    print("\nHour 0-1 (Basal only):")
    for _ in range(12): # 1 hour
        # Use the patient's current (potentially sampled) basal_rate_U_hr as the input for basal_insulin
        patient.step(basal_insulin=patient.basal_rate_U_hr, bolus_insulin=0, carbs_details=None)
    print(f"CGM after 1hr basal: {patient.get_cgm_reading():.2f} mg/dL, IOB_Rapid: {patient.iob_rapid:.2f}, IOB_Long: {patient.iob_long:.2f}")
    # print(f"Internal states: {patient.get_internal_states()}")

    # Step 2: Meal with Protein/Fat + Bolus at Hour 1
    print("\nHour 1 (Meal + Bolus):")
    carbs_for_meal_g = 50
    protein_for_meal = 25
    fat_for_meal = 15
    # Bolus calculation should use the patient's *current* CR
    meal_bolus = carbs_for_meal_g / patient.cr
    print(f"  Calculated meal_bolus: {meal_bolus:.2f} U (using current CR: {patient.cr:.1f})")
    meal_carbs_details = {"grams": carbs_for_meal_g, "gi_factor": 1.0} # Assuming medium GI for now
    patient.step(
        basal_insulin=patient.basal_rate_U_hr, # Use current sampled basal need
        bolus_insulin=meal_bolus,
        carbs_details=meal_carbs_details,
        protein_ingested=protein_for_meal,
        fat_ingested=fat_for_meal
    )
    print(f"Post-Meal (Immediate): CGM: {patient.get_cgm_reading():.2f}, COB: {patient.cob:.1f}, ProtG2: {patient.Prot_G2:.1f}, ActiveFat: {patient.active_fat_g:.1f}, IOB_Rapid: {patient.iob_rapid:.2f}, IOB_Long: {patient.iob_long:.2f}")

    # Step 3: Simulate post-meal for 2 hours (Hour 1 to Hour 3)
    print("\nHour 1-3 (Post-Meal, Basal only):")
    for i in range(24): # 2 hours
        patient.step(basal_insulin=patient.basal_rate_U_hr, bolus_insulin=0, carbs_details=None)
        if (i + 1) % 6 == 0: # Every 30 mins
             print(f"  Time {(60 + (i + 1) * 5)/60.0:.2f}h: CGM: {patient.get_cgm_reading():.2f}, COB: {patient.cob:.1f}, ProtG2: {patient.Prot_G2:.1f}, ActiveFat: {patient.active_fat_g:.1f}, IOB_R: {patient.iob_rapid:.2f}, IOB_L: {patient.iob_long:.2f}")
    print(f"CGM at Hour 3: {patient.get_cgm_reading():.2f} mg/dL, IOB_Rapid: {patient.iob_rapid:.2f}, IOB_Long: {patient.iob_long:.2f}")
    # print(f"Internal states: {patient.get_internal_states()}")

    # Step 4: Exercise for 30 mins at Hour 3
    print("\nHour 3 (Start Exercise):")
    exercise_event_details = {"duration_minutes": 30.0, "intensity_factor": 1.0}
    # First step of exercise
    patient.step(basal_insulin=patient.basal_rate_U_hr, bolus_insulin=0, carbs_details=None, exercise_event=exercise_event_details)
    print(f"After 1st step of exercise: CGM: {patient.get_cgm_reading():.2f}, Exercise remaining: {patient.exercise_duration_remaining_min:.1f}min, IOB_R: {patient.iob_rapid:.2f}, IOB_L: {patient.iob_long:.2f}")

    # Continue for the duration of exercise (25 more minutes = 5 steps)
    for i in range(5):
        patient.step(basal_insulin=patient.basal_rate_U_hr, bolus_insulin=0, carbs_details=None) # No new event, exercise continues
        print(f"  Exercise step {i+2}/6: CGM: {patient.get_cgm_reading():.2f}, Exercise remaining: {patient.exercise_duration_remaining_min:.1f}min, IOB_R: {patient.iob_rapid:.2f}, IOB_L: {patient.iob_long:.2f}")
    print(f"CGM after 30min exercise (at Hour 3.5): {patient.get_cgm_reading():.2f} mg/dL, IOB_Rapid: {patient.iob_rapid:.2f}, IOB_Long: {patient.iob_long:.2f}")
    # print(f"Internal states: {patient.get_internal_states()}")

    # Step 5: Introduce Stress and simulate for 1 hour (Hour 3.5 to Hour 4.5)
    print("\n--- Testing Stress Model Refinements ---")
    # Scenario: Hour 3.5 onwards, post-exercise
    
    # Test Low Stress
    print("\nHour 3.5-4.5 (Post-Exercise, Low Stress introduced):")
    patient.set_stress_level(0.3) # Apply 30% stress level
    print(f"  Low Stress level (0.3) set for 1 hour.")
    for i in range(12): # 1 hour
        patient.step(basal_insulin=patient.basal_rate_U_hr, bolus_insulin=0, carbs_details=None)
        if (i + 1) % 6 == 0: # Every 30 mins
             print(f"  Time {(210 + (i + 1) * 5)/60.0:.2f}h (Low Stress ON): CGM: {patient.get_cgm_reading():.2f}, Stress: {patient.current_stress_level:.2f}")
    print(f"CGM at Hour 4.5 (after 1hr Low Stress): {patient.get_cgm_reading():.2f} mg/dL")
    
    # Test High Stress
    print("\nHour 4.5-5.5 (Post-Exercise, High Stress introduced):")
    patient.set_stress_level(0.8) # Apply 80% stress level
    print(f"  High Stress level (0.8) set for 1 hour.")
    for i in range(12): # 1 hour
        patient.step(basal_insulin=patient.basal_rate_U_hr, bolus_insulin=0, carbs_details=None)
        if (i + 1) % 6 == 0: # Every 30 mins
             print(f"  Time {(270 + (i + 1) * 5)/60.0:.2f}h (High Stress ON): CGM: {patient.get_cgm_reading():.2f}, Stress: {patient.current_stress_level:.2f}")
    print(f"CGM at Hour 5.5 (after 1hr High Stress): {patient.get_cgm_reading():.2f} mg/dL")
    
    patient.set_stress_level(0.0) # Remove stress
    print(f"  Stress level reset to {patient.current_stress_level:.2f}")
    
    # Step 6: Simulate post-stress for 0.5 hours (Hour 5.5 to Hour 6)
    print("\nHour 5.5-6 (Post-Stress, Basal only):")
    for i in range(6): # 0.5 hours
        patient.step(basal_insulin=patient.basal_rate_U_hr, bolus_insulin=0, carbs_details=None)
    print(f"CGM at Hour 6 (stress removed): {patient.get_cgm_reading():.2f} mg/dL, IOB_Rapid: {patient.iob_rapid:.2f}, IOB_Long: {patient.iob_long:.2f}")
    
    print("\n--- Testing Illness Model Refinements ---")
    # Scenario: Hour 6 onwards
    
    # Test Mild Illness
    print("\nHour 6-7 (Mild Illness introduced):")
    patient.set_illness_level(0.3) # Apply 30% illness level
    print(f"  Mild Illness level (0.3) set for 1 hour.")
    illness_carbs_mild = {"grams": 15, "gi_factor": 0.7} # Small, low GI snack during mild illness
    patient.step(basal_insulin=patient.basal_rate_U_hr, bolus_insulin=0, carbs_details=illness_carbs_mild, protein_ingested=0, fat_ingested=0)
    print(f"  After small carb intake during Mild Illness: CGM: {patient.get_cgm_reading():.2f}, COB: {patient.cob:.1f}, Illness: {patient.current_illness_level:.2f}")
    for i in range(11): # Remaining steps for 1 hour
        patient.step(basal_insulin=patient.basal_rate_U_hr, bolus_insulin=0, carbs_details=None)
        if (i + 1) % 5 == 0: # Approx every 25 mins
             print(f"  Time {(360 + (i + 2) * 5)/60.0:.2f}h (Mild Illness ON): CGM: {patient.get_cgm_reading():.2f}, COB: {patient.cob:.1f}, Illness: {patient.current_illness_level:.2f}")
    print(f"CGM at Hour 7 (after 1hr Mild Illness): {patient.get_cgm_reading():.2f} mg/dL")

    # Test Severe Illness
    print("\nHour 7-8 (Severe Illness introduced):")
    patient.set_illness_level(0.7) # Apply 70% illness level
    print(f"  Severe Illness level (0.7) set for 1 hour.")
    illness_carbs_severe = {"grams": 10, "gi_factor": 0.5} # Very small, very low GI snack during severe illness
    patient.step(basal_insulin=patient.basal_rate_U_hr, bolus_insulin=0, carbs_details=illness_carbs_severe, protein_ingested=0, fat_ingested=0)
    print(f"  After small carb intake during Severe Illness: CGM: {patient.get_cgm_reading():.2f}, COB: {patient.cob:.1f}, Illness: {patient.current_illness_level:.2f}")
    for i in range(11): # Remaining steps for 1 hour
        patient.step(basal_insulin=patient.basal_rate_U_hr, bolus_insulin=0, carbs_details=None)
        if (i + 1) % 5 == 0: # Approx every 25 mins
             print(f"  Time {(420 + (i + 2) * 5)/60.0:.2f}h (Severe Illness ON): CGM: {patient.get_cgm_reading():.2f}, COB: {patient.cob:.1f}, Illness: {patient.current_illness_level:.2f}")
    print(f"CGM at Hour 8 (after 1hr Severe Illness): {patient.get_cgm_reading():.2f} mg/dL")
    
    patient.set_illness_level(0.0) # Remove illness
    print(f"  Illness level reset to {patient.current_illness_level:.2f}")

    # Step 8: Simulate post-illness for 1 hour (Hour 8 to Hour 9)
    print("\nHour 8-9 (Post-Illness, Basal only):")
    for i in range(12): # 1 hour
        patient.step(basal_insulin=patient.basal_rate_U_hr, bolus_insulin=0, carbs_details=None)
    print(f"CGM at Hour 9 (illness removed): {patient.get_cgm_reading():.2f} mg/dL, IOB_Rapid: {patient.iob_rapid:.2f}, IOB_Long: {patient.iob_long:.2f}")
    
    # Step 9: Test different GI factors
    print("\n--- Testing Different GI Factors (More Pronounced) ---")
    
    fixed_test_carb_abs_rate = 0.5  # Use a higher base rate (50%/hr) for these GI tests
    test_meal_grams = 50

    # Scenario A: Low GI
    print("\nScenario A: Low GI Meal")
    patient.reset({"initial_glucose": 120.0})
    # Override carb absorption rate for this test for clearer GI effect
    original_sampled_carb_abs_rate_low_gi = patient.carb_absorption_rate_g_min
    patient.carb_absorption_rate_g_min = fixed_test_carb_abs_rate
    print(f"  Low GI Test - Initial State: CGM={patient.get_cgm_reading():.2f}")
    print(f"    Using fixed CarbAbsRate: {patient.carb_absorption_rate_g_min:.3f} (Original sampled: {original_sampled_carb_abs_rate_low_gi:.3f})")
    print(f"    Other Sampled Params: ISF={patient.isf:.1f}, CR={patient.cr:.1f}, BasalNeed={patient.basal_rate_U_hr:.2f}")
    
    carbs_low_gi = {"grams": test_meal_grams, "gi_factor": 0.4}
    patient.step(basal_insulin=patient.basal_rate_U_hr, bolus_insulin=0, carbs_details=carbs_low_gi)
    print(f"  After Low GI meal ({test_meal_grams}g, GI {carbs_low_gi['gi_factor']:.1f}): CGM={patient.get_cgm_reading():.2f}, COB={patient.cob:.1f}")
    for i in range(24): # Simulate 2 hours
        patient.step(basal_insulin=patient.basal_rate_U_hr, bolus_insulin=0, carbs_details=None)
        if (i + 1) % 6 == 0: # Every 30 mins
            print(f"    Low GI - Time {( (i + 1) * 5 )/60.0:.2f}h post-meal: CGM={patient.get_cgm_reading():.2f}, COB={patient.cob:.1f}")
    print(f"  2hr after Low GI: CGM={patient.get_cgm_reading():.2f}, COB={patient.cob:.1f}")
    patient.carb_absorption_rate_g_min = original_sampled_carb_abs_rate_low_gi # Restore (though it's end of test)

    # Scenario B: High GI
    print("\nScenario B: High GI Meal")
    patient.reset({"initial_glucose": 120.0})
    # Override carb absorption rate for this test for clearer GI effect
    original_sampled_carb_abs_rate_high_gi = patient.carb_absorption_rate_g_min
    patient.carb_absorption_rate_g_min = fixed_test_carb_abs_rate
    print(f"  High GI Test - Initial State: CGM={patient.get_cgm_reading():.2f}")
    print(f"    Using fixed CarbAbsRate: {patient.carb_absorption_rate_g_min:.3f} (Original sampled: {original_sampled_carb_abs_rate_high_gi:.3f})")
    print(f"    Other Sampled Params: ISF={patient.isf:.1f}, CR={patient.cr:.1f}, BasalNeed={patient.basal_rate_U_hr:.2f}")

    carbs_high_gi = {"grams": test_meal_grams, "gi_factor": 1.6}
    patient.step(basal_insulin=patient.basal_rate_U_hr, bolus_insulin=0, carbs_details=carbs_high_gi)
    print(f"  After High GI meal ({test_meal_grams}g, GI {carbs_high_gi['gi_factor']:.1f}): CGM={patient.get_cgm_reading():.2f}, COB={patient.cob:.1f}")
    for i in range(24): # Simulate 2 hours
        patient.step(basal_insulin=patient.basal_rate_U_hr, bolus_insulin=0, carbs_details=None)
        if (i + 1) % 6 == 0: # Every 30 mins
            print(f"    High GI - Time {( (i + 1) * 5 )/60.0:.2f}h post-meal: CGM={patient.get_cgm_reading():.2f}, COB={patient.cob:.1f}")
    print(f"  2hr after High GI: CGM={patient.get_cgm_reading():.2f}, COB={patient.cob:.1f}")
    patient.carb_absorption_rate_g_min = original_sampled_carb_abs_rate_high_gi # Restore

    print("\n--- End of Simulation ---")
    internal_states = patient.get_internal_states()
    print(f"Final Sampled Params: ISF={internal_states['current_isf']:.1f}, CR={internal_states['current_cr']:.1f}, BasalNeed={internal_states['current_basal_rate_U_hr_need']:.2f}, CarbAbsRate={internal_states['current_carb_absorption_rate_g_min']:.3f}")
    # print(f"Final Internal states: {patient.get_internal_states()}")