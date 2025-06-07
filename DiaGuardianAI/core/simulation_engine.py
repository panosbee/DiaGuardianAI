# DiaGuardianAI Simulation Engine
# This module will manage the main simulation loop and coordinate interactions.

import sys
import os
from typing import Dict, Any, Optional, List # Added List

# Ensure the DiaGuardianAI package is discoverable
if __package__ is None or __package__ == '':
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from DiaGuardianAI.core.base_classes import BaseSyntheticPatient, BaseAgent # BasePredictiveModel removed as it's not directly used by engine
# Removed unused BasePredictiveModel from this file's direct imports as agent handles its predictor.
# If SimulationEngine were to directly use a predictor, it would be needed.

class SimulationEngine:
    """Orchestrates the closed-loop simulation of diabetes management.
    (Docstring remains the same)
    """
    def __init__(self, patient: BaseSyntheticPatient, agent: BaseAgent, config: Dict[str, Any]):
        """Initializes the SimulationEngine.
        (Docstring remains the same)
        """
        self.patient = patient
        self.agent = agent # Agent now manages its own predictor
        self.config = config
        
        self.simulation_data: Dict[str, List[Any]] = {
            "time_steps_minutes": [], # Simulation time in minutes
            "cgm_readings": [],
            "iob": [], 
            "cob": [], 
            "actions_taken": [],
            "rewards": [],
            "meals_ingested_g": [], # For carbs
            "protein_ingested_g": [],
            "fat_ingested_g": [],
            "exercise_events_details": [],
            # Other patient-specific states can be added if patient.get_internal_states() provides more
        }
        self.current_step_index = 0 # Renamed from current_step for clarity
        self.max_simulation_steps = self.config.get("max_simulation_steps", 288) # Default: 1 day if 5-min steps
        self.time_step_minutes = self.config.get("time_step_minutes", 5)
        
        agent_cgm_history_len = getattr(self.agent, 'cgm_history_len', 12) 
        self.cgm_history_buffer_size = self.config.get("cgm_history_buffer_size", agent_cgm_history_len)
        self.cgm_history_buffer: List[float] = []
        
        print(f"SimulationEngine initialized. Max steps: {self.max_simulation_steps}, Time step: {self.time_step_minutes} min.")

    def _get_current_patient_state_for_agent(self) -> Dict[str, Any]:
        """Helper to construct the state dictionary for the agent."""
        current_cgm = self.patient.get_cgm_reading()
        internal_states = self.patient.get_internal_states() 
        
        self.cgm_history_buffer.append(current_cgm)
        if len(self.cgm_history_buffer) > self.cgm_history_buffer_size:
            self.cgm_history_buffer.pop(0)
        
        padded_cgm_history = list(self.cgm_history_buffer)
        if len(padded_cgm_history) < self.cgm_history_buffer_size:
            padding = [current_cgm] * (self.cgm_history_buffer_size - len(padded_cgm_history))
            padded_cgm_history = padding + padded_cgm_history

        return {
            "cgm": current_cgm,
            "iob": internal_states.get("iob", internal_states.get("total_iob_U", 0.0)), # Check for total_iob_U as well
            "cob": internal_states.get("cob", internal_states.get("carbs_on_board_g", 0.0)), # Check for carbs_on_board_g
            "cgm_history": padded_cgm_history, 
            "meal_announced": internal_states.get("meal_announced_this_step", False),
            "announced_carbs": internal_states.get("announced_carbs_this_step", 0.0),
            "current_simulation_time_minutes": self.current_step_index * self.time_step_minutes,
            # It might be beneficial to pass the patient params directly if the agent needs them
            # and cannot get them from the patient object itself.
            # "patient_params": self.patient.params # If patient object has a .params attribute
        }

    def run(self) -> Dict[str, Any]:
        """Runs the diabetes management simulation."""
        print(f"Simulation engine run started for {self.max_simulation_steps} steps.")
        for key in self.simulation_data: self.simulation_data[key] = []
        self.cgm_history_buffer = [] 
        self.current_step_index = 0

        for step_idx in range(self.max_simulation_steps):
            self.current_step_index = step_idx
            current_sim_time_minutes = step_idx * self.time_step_minutes
            self.simulation_data["time_steps_minutes"].append(current_sim_time_minutes)

            patient_state_for_agent = self._get_current_patient_state_for_agent()
            
            self.simulation_data["cgm_readings"].append(patient_state_for_agent["cgm"])
            self.simulation_data["iob"].append(patient_state_for_agent["iob"])
            self.simulation_data["cob"].append(patient_state_for_agent["cob"])

            # Pass the patient object to decide_action
            action = self.agent.decide_action(patient_state_for_agent, patient=self.patient)
            self.simulation_data["actions_taken"].append(action.copy() if action else {})

            # Ensure action is not None before trying to get items
            bolus_insulin = 0.0
            basal_rate_u_hr = None # Default to None if action is None or keys are missing

            if action:
                # CRITICAL FIX: Handle both flat and nested action formats
                # PatternAdvisorAgent returns nested format: {"actions": {"bolus_u": X, "basal_rate_u_hr": Y}}
                # RLAgent returns flat format: {"bolus_u": X, "basal_rate_u_hr": Y}

                if "actions" in action and isinstance(action["actions"], dict):
                    # Nested format from PatternAdvisorAgent
                    action_values = action["actions"]
                    bolus_insulin = action_values.get("bolus_u", 0.0)
                    basal_rate_u_hr = action_values.get("basal_rate_u_hr")
                    print(f"SimulationEngine: Using nested action format. Basal: {basal_rate_u_hr}, Bolus: {bolus_insulin}")
                else:
                    # Flat format from RLAgent or other agents
                    bolus_insulin = action.get("bolus_u", 0.0)
                    basal_rate_u_hr = action.get("basal_rate_u_hr")
                    print(f"SimulationEngine: Using flat action format. Basal: {basal_rate_u_hr}, Bolus: {bolus_insulin}")
            else:
                # Handle case where agent returns None (e.g. no suggestion)
                # Default to some safe action or continue patient's current basal
                # For now, this means no bolus and basal_rate_u_hr remains None (patient continues current basal)
                print(f"SimulationEngine: Warning - Agent returned no action at step {step_idx}. Using defaults.")

            basal_for_step = 0.0
            if basal_rate_u_hr is not None:
                basal_for_step = basal_rate_u_hr * (self.time_step_minutes / 60.0)

            # ENHANCED MEAL HANDLING: Support both old format (float) and new format (dict)
            meal_schedule = self.config.get("meal_schedule", {})
            scheduled_meal = meal_schedule.get(current_sim_time_minutes, None)

            carbs_ingested = 0.0
            gi_factor = 1.0
            meal_type = "none"

            if scheduled_meal is not None:
                if isinstance(scheduled_meal, dict):
                    # New format: {"grams": 60, "gi_factor": 1.0, "type": "breakfast"}
                    carbs_ingested = scheduled_meal.get("grams", 0.0)
                    gi_factor = scheduled_meal.get("gi_factor", 1.0)
                    meal_type = scheduled_meal.get("type", "meal")
                    print(f"SimulationEngine: Scheduled {meal_type} - {carbs_ingested}g carbs (GI: {gi_factor})")
                else:
                    # Old format: simple float value
                    carbs_ingested = float(scheduled_meal)
                    print(f"SimulationEngine: Scheduled meal - {carbs_ingested}g carbs")

            # GLUCOSE RESCUE MECHANISM: Automatic carb delivery when glucose is dangerously low
            current_cgm = patient_state_for_agent["cgm"]
            glucose_rescue_enabled = self.config.get("glucose_rescue_enabled", False)
            glucose_rescue_threshold = self.config.get("glucose_rescue_threshold", 70.0)
            glucose_rescue_carbs = self.config.get("glucose_rescue_carbs", 15.0)

            if glucose_rescue_enabled and current_cgm < glucose_rescue_threshold:
                # Add emergency glucose rescue
                carbs_ingested += glucose_rescue_carbs
                gi_factor = 2.0  # Fast-acting glucose
                meal_type = "glucose_rescue"
                print(f"🚑 GLUCOSE RESCUE: CGM {current_cgm:.1f} mg/dL < {glucose_rescue_threshold} mg/dL, giving {glucose_rescue_carbs}g fast carbs!")

            protein_ingested = self.config.get("protein_schedule", {}).get(current_sim_time_minutes, 0.0)
            fat_ingested = self.config.get("fat_schedule", {}).get(current_sim_time_minutes, 0.0)
            exercise_event = self.config.get("exercise_schedule", {}).get(current_sim_time_minutes, None)

            self.simulation_data["meals_ingested_g"].append(carbs_ingested)
            self.simulation_data["protein_ingested_g"].append(protein_ingested)
            self.simulation_data["fat_ingested_g"].append(fat_ingested)
            self.simulation_data["exercise_events_details"].append(exercise_event)

            carbs_details_for_step: Optional[Dict[str, Any]] = None
            if carbs_ingested > 0: # Only create details if carbs are ingested
                carbs_details_for_step = {
                    "grams": carbs_ingested,
                    "gi_factor": gi_factor,
                    "meal_type": meal_type
                }

            self.patient.step(
                basal_insulin=basal_for_step,
                bolus_insulin=bolus_insulin,
                carbs_details=carbs_details_for_step, # Changed from carbs_ingested
                protein_ingested=protein_ingested,
                fat_ingested=fat_ingested,
                exercise_event=exercise_event
            )

            # Reward calculation
            reward_config = self.config.get("reward_function_config", {})
            TARGET_RANGE_LOW = float(reward_config.get("target_range_low", 70.0))
            TARGET_RANGE_HIGH = float(reward_config.get("target_range_high", 180.0))
            HYPO_THRESHOLD = float(reward_config.get("hypo_threshold", 70.0))
            SEVERE_HYPO_THRESHOLD = float(reward_config.get("severe_hypo_threshold", 54.0))
            HYPER_THRESHOLD = float(reward_config.get("hyper_threshold", 180.0))
            SEVERE_HYPER_THRESHOLD = float(reward_config.get("severe_hyper_threshold", 250.0))
            
            REWARD_IN_RANGE = float(reward_config.get("reward_in_range", 1.0))
            PENALTY_HYPO = float(reward_config.get("penalty_hypo", -2.0))
            PENALTY_SEVERE_HYPO = float(reward_config.get("penalty_severe_hypo", -10.0))
            PENALTY_HYPER = float(reward_config.get("penalty_hyper", -1.0))
            PENALTY_SEVERE_HYPER = float(reward_config.get("penalty_severe_hyper", -5.0))

            # Ensure scales are not zero to avoid division by zero
            MAX_HYPO_DEVIATION_SCALE = max(1e-6, HYPO_THRESHOLD - SEVERE_HYPO_THRESHOLD)
            MAX_HYPER_DEVIATION_SCALE = max(1e-6, SEVERE_HYPER_THRESHOLD - HYPER_THRESHOLD)

            cgm_after_action = self.patient.get_cgm_reading()
            reward = 0.0
            
            # TIR component
            if TARGET_RANGE_LOW <= cgm_after_action <= TARGET_RANGE_HIGH:
                reward += REWARD_IN_RANGE
            
            # Hypoglycemia penalties
            if cgm_after_action < HYPO_THRESHOLD:
                if cgm_after_action < SEVERE_HYPO_THRESHOLD:
                    reward += PENALTY_SEVERE_HYPO
                    # Additional penalty for how deep into severe hypo
                    deviation_severe = SEVERE_HYPO_THRESHOLD - cgm_after_action
                    reward += (PENALTY_HYPO / 2) * (deviation_severe / MAX_HYPO_DEVIATION_SCALE) # Scaled additional penalty
                else:
                    # Penalty for mild/moderate hypo
                    deviation_mild = HYPO_THRESHOLD - cgm_after_action
                    reward += PENALTY_HYPO * (deviation_mild / (HYPO_THRESHOLD - SEVERE_HYPO_THRESHOLD if HYPO_THRESHOLD > SEVERE_HYPO_THRESHOLD else 1e-6) )

            # Hyperglycemia penalties
            elif cgm_after_action > HYPER_THRESHOLD:
                if cgm_after_action > SEVERE_HYPER_THRESHOLD:
                    reward += PENALTY_SEVERE_HYPER
                    # Additional penalty for how deep into severe hyper
                    deviation_severe = cgm_after_action - SEVERE_HYPER_THRESHOLD
                    reward += (PENALTY_HYPER / 2) * (deviation_severe / MAX_HYPER_DEVIATION_SCALE) # Scaled additional penalty
                else:
                    # Penalty for mild/moderate hyper
                    deviation_mild = cgm_after_action - HYPER_THRESHOLD
                    reward += PENALTY_HYPER * (deviation_mild / (SEVERE_HYPER_THRESHOLD - HYPER_THRESHOLD if SEVERE_HYPER_THRESHOLD > HYPER_THRESHOLD else 1e-6))
            
            # Glycemic Variability Penalty
            variability_penalty_weight = float(reward_config.get("variability_penalty_weight", -0.05)) # Small negative weight
            max_cgm_change_per_step = float(reward_config.get("max_expected_cgm_change_per_step", 20.0)) # e.g., 20 mg/dL in 5 mins

            if len(self.simulation_data["cgm_readings"]) > 0: # Need at least one previous reading
                # The current cgm_after_action is for the current step.
                # The last entry in self.simulation_data["cgm_readings"] is from the *beginning* of the current step (before action).
                # Or, if cgm_history_buffer is up-to-date with pre-action CGM:
                # prev_cgm = self.cgm_history_buffer[-1] if self.cgm_history_buffer else cgm_after_action
                # For simplicity, let's use the last recorded CGM if available, otherwise no variability penalty this step.
                # The cgm_readings list stores the CGM *before* the current patient.step() call.
                # So, patient_state_for_agent["cgm"] was the CGM at the start of this step.
                
                cgm_before_action_this_step = patient_state_for_agent["cgm"] # CGM at the start of this step
                cgm_change = abs(cgm_after_action - cgm_before_action_this_step)
                
                # Normalize change and apply penalty
                # Penalize more for larger changes, up to max_expected_cgm_change_per_step
                normalized_change_penalty = min(1.0, cgm_change / max_cgm_change_per_step)
                reward += variability_penalty_weight * normalized_change_penalty
                
            self.simulation_data["rewards"].append(reward)

            if (step_idx + 1) % (self.max_simulation_steps // 20 if self.max_simulation_steps >=20 else 1) == 0:
                 print(f"  SimEngine Step {step_idx + 1}/{self.max_simulation_steps}. CGM: {patient_state_for_agent['cgm']:.2f}, Action: {action}")
        
        print("Simulation engine run finished.")
        return self.simulation_data

    def plot_glucose_trace(self):
        """Plots the glucose trace from the simulation results."""
        if self.simulation_data and self.simulation_data.get("cgm_readings"):
            print(f"Plotting glucose trace... (Trace length: {len(self.simulation_data['cgm_readings'])})")
            # Actual plotting logic using matplotlib would go here
        else:
            print("No glucose trace data to plot or results are not available.")

if __name__ == '__main__':
    from DiaGuardianAI.data_generation.synthetic_patient_model import SyntheticPatient 
    from DiaGuardianAI.agents.decision_agent import RLAgent 
    
    print("--- SimulationEngine Standalone Example ---")
    print("Setting up a simulation...")
    patient_params = {
        "initial_glucose": 120.0, "ISF": 50.0, "CR": 10.0,
        "basal_rate_U_hr": 1.0, "body_weight_kg": 70.0,
    }
    try:
        dummy_patient = SyntheticPatient(params=patient_params)
    except Exception as e:
        print(f"Error initializing SyntheticPatient: {e}.")
        dummy_patient = None

    cgm_hist_len_ex = 12
    pred_hor_len_ex = 6
    calculated_s_dim = 1 + cgm_hist_len_ex + 1 + 1 + pred_hor_len_ex + pred_hor_len_ex + 2 # = 29
    
    action_space_def_ex = { 
        "bolus_u": {"low": 0.0, "high": 10.0}, 
        "basal_rate_u_hr": {"low": 0.0, "high": 2.0} 
    }
    dummy_agent = RLAgent(
        state_dim=calculated_s_dim, 
        action_space_definition=action_space_def_ex,
        cgm_history_len=cgm_hist_len_ex,
        prediction_horizon_len=pred_hor_len_ex,
        predictor=None 
    )
    
    sim_config_ex = {
        "max_simulation_steps": 144, 
        "time_step_minutes": 5,
        "cgm_history_buffer_size": cgm_hist_len_ex,
        "meal_schedule": { 
            60: 50.0, 240: 75.0, 480: 30.0 
        },
        "protein_schedule": { 60: 20.0, 240: 10.0 },
        "fat_schedule": { 60: 10.0 },
        "exercise_schedule": { 
            180: {"duration_minutes": 30.0, "intensity_factor": 1.0} 
        }
    }
    
    if dummy_patient:
        print("Patient and Agent initialized. Starting SimulationEngine run...")
        engine = SimulationEngine(patient=dummy_patient, agent=dummy_agent, config=sim_config_ex)
        simulation_results = engine.run()
        
        print("\n--- Simulation Results Summary ---")
        print(f"Total steps run: {len(simulation_results['time_steps_minutes'])}")
        if simulation_results['cgm_readings']:
            print(f"Final CGM: {simulation_results['cgm_readings'][-1]:.2f}")
        
        print("\nLogged Meals (Carbs g):")
        for i, carbs in enumerate(simulation_results['meals_ingested_g']):
            if carbs > 0:
                print(f"  Step {i} (Time: {simulation_results['time_steps_minutes'][i]} min): {carbs}g")
        
        print("\nLogged Protein (g):")
        for i, protein in enumerate(simulation_results['protein_ingested_g']):
            if protein > 0:
                print(f"  Step {i} (Time: {simulation_results['time_steps_minutes'][i]} min): {protein}g")

        print("\nLogged Fat (g):")
        for i, fat in enumerate(simulation_results['fat_ingested_g']):
            if fat > 0:
                print(f"  Step {i} (Time: {simulation_results['time_steps_minutes'][i]} min): {fat}g")

        print("\nLogged Exercise Events:")
        for i, event in enumerate(simulation_results['exercise_events_details']):
            if event: 
                print(f"  Step {i} (Time: {simulation_results['time_steps_minutes'][i]} min): {event}")
    else:
        print("Dummy patient could not be initialized. Skipping simulation run.")
    
    print("\nSimulationEngine example run complete.")