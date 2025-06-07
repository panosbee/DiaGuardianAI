#!/usr/bin/env python3
"""
DiaGuardianAI Intelligent Meal Detection and Training Simulation

This script demonstrates the advanced meal detection and random injection system
for training the DiaGuardianAI library to automatically detect meals from CGM patterns.

PHASE 1: Random meal injection for diverse training data
PHASE 2: CGM spike detection for meal identification
PHASE 3: Pattern learning for meal prediction
"""

import os
import sys
import numpy as np
from typing import Dict, Any, Optional

# Ensure the DiaGuardianAI package is discoverable
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from DiaGuardianAI.data_generation.synthetic_patient_model import SyntheticPatient
from DiaGuardianAI.agents.pattern_advisor_agent import PatternAdvisorAgent
from DiaGuardianAI.core.simulation_engine import SimulationEngine
from DiaGuardianAI.core.intelligent_meal_system import IntelligentMealSystem

# Try to import LSTMPredictor, but don't fail if not available
try:
    from DiaGuardianAI.predictive_models.lstm_predictor import LSTMPredictor
except ImportError:
    try:
        from DiaGuardianAI.models.lstm_predictor import LSTMPredictor
    except ImportError:
        print("Warning: LSTMPredictor not found, continuing without it")
        LSTMPredictor = None

def load_glucose_predictor(model_dir: str):
    """Loads a trained LSTMPredictor."""
    if LSTMPredictor is None:
        print(f"LSTMPredictor not available, skipping glucose predictor loading")
        return None

    try:
        predictor = LSTMPredictor.load_from_directory(model_dir)
        print(f"LSTMPredictor loaded successfully from {model_dir}")
        return predictor
    except Exception as e:
        print(f"Failed to load LSTMPredictor from {model_dir}: {e}")
        return None

def load_trained_pattern_advisor(
    model_path: str,
    dummy_repo_db_path: str = "dummy_advisor_sim_repo.sqlite"
) -> Optional[PatternAdvisorAgent]:
    """Loads a trained PatternAdvisorAgent."""
    try:
        advisor_agent = PatternAdvisorAgent.load_from_file(
            model_path, 
            dummy_repo_db_path=dummy_repo_db_path
        )
        print(f"PatternAdvisorAgent loaded successfully from {model_path}")
        print(f"  Agent type: {advisor_agent.learning_model_type}")
        print(f"  Is trained: {advisor_agent.is_trained}")
        print(f"  Action dim: {advisor_agent.action_dim}, Action keys: {advisor_agent.action_keys_ordered}")
        return advisor_agent
    except Exception as e:
        print(f"Failed to load PatternAdvisorAgent from {model_path}: {e}")
        return None

class IntelligentSimulationEngine(SimulationEngine):
    """Enhanced simulation engine with intelligent meal system."""
    
    def __init__(self, patient, agent, config: Dict[str, Any]):
        super().__init__(patient, agent, config)
        
        # Initialize intelligent meal system
        meal_system_config = {
            "random_injection_enabled": config.get("random_injection_enabled", True),
            "injection_probability_per_hour": config.get("injection_probability_per_hour", 0.2),
            "min_meal_interval_minutes": config.get("min_meal_interval_minutes", 90),
            "spike_detection_enabled": config.get("spike_detection_enabled", True),
            "spike_threshold_mg_dl": config.get("spike_threshold_mg_dl", 25.0),
            "spike_detection_window_minutes": config.get("spike_detection_window_minutes", 30),
        }
        
        self.intelligent_meal_system = IntelligentMealSystem(meal_system_config)
        
        # Enhanced data tracking
        self.simulation_data["intelligent_meals"] = []
        self.simulation_data["meal_detection_events"] = []
        
        print(f"IntelligentSimulationEngine initialized with intelligent meal system.")
    
    def run(self) -> Dict[str, Any]:
        """Enhanced simulation run with intelligent meal detection."""
        print(f"Intelligent simulation started for {self.max_simulation_steps} steps.")
        
        # Initialize data structures
        for key in self.simulation_data: 
            self.simulation_data[key] = []
        self.cgm_history_buffer = [] 
        self.current_step_index = 0

        for step_idx in range(self.max_simulation_steps):
            self.current_step_index = step_idx
            current_sim_time_minutes = step_idx * self.time_step_minutes
            self.simulation_data["time_steps_minutes"].append(current_sim_time_minutes)

            # Get current patient state
            patient_state_for_agent = self._get_current_patient_state_for_agent()
            current_cgm = patient_state_for_agent["cgm"]
            
            self.simulation_data["cgm_readings"].append(current_cgm)
            self.simulation_data["iob"].append(patient_state_for_agent["iob"])
            self.simulation_data["cob"].append(patient_state_for_agent["cob"])

            # INTELLIGENT MEAL PROCESSING
            intelligent_meal = self.intelligent_meal_system.process_simulation_step(
                current_sim_time_minutes, current_cgm
            )
            
            # Get agent decision
            action = self.agent.decide_action(patient_state_for_agent, patient=self.patient)
            self.simulation_data["actions_taken"].append(action.copy() if action else {})

            # Process insulin action (same as before)
            bolus_insulin = 0.0
            basal_rate_u_hr = None

            if action:
                if "actions" in action and isinstance(action["actions"], dict):
                    action_values = action["actions"]
                    bolus_insulin = action_values.get("bolus_u", 0.0)
                    basal_rate_u_hr = action_values.get("basal_rate_u_hr")
                else:
                    bolus_insulin = action.get("bolus_u", 0.0)
                    basal_rate_u_hr = action.get("basal_rate_u_hr")

            basal_for_step = 0.0
            if basal_rate_u_hr is not None:
                basal_for_step = basal_rate_u_hr * (self.time_step_minutes / 60.0)
            
            # ENHANCED MEAL HANDLING: Combine scheduled + intelligent meals
            scheduled_meal = self.config.get("meal_schedule", {}).get(current_sim_time_minutes, None)
            
            carbs_ingested = 0.0
            gi_factor = 1.0
            meal_type = "none"
            meal_source = "none"
            
            # Process scheduled meal (if any)
            if scheduled_meal is not None:
                if isinstance(scheduled_meal, dict):
                    carbs_ingested += scheduled_meal.get("grams", 0.0)
                    gi_factor = scheduled_meal.get("gi_factor", 1.0)
                    meal_type = scheduled_meal.get("type", "scheduled")
                    meal_source = "scheduled"
                else:
                    carbs_ingested += float(scheduled_meal)
                    meal_source = "scheduled"
            
            # Process intelligent meal (if any)
            if intelligent_meal:
                carbs_ingested += intelligent_meal["grams"]
                if meal_source == "none":
                    gi_factor = intelligent_meal["gi_factor"]
                    meal_type = intelligent_meal["type"]
                    meal_source = intelligent_meal["detection_method"]
                else:
                    # Combine with scheduled meal
                    meal_type = f"{meal_type}+{intelligent_meal['type']}"
                    meal_source = f"{meal_source}+{intelligent_meal['detection_method']}"
                
                # Log intelligent meal
                self.simulation_data["intelligent_meals"].append({
                    "time": current_sim_time_minutes,
                    "carbs": intelligent_meal["grams"],
                    "type": intelligent_meal["type"],
                    "method": intelligent_meal["detection_method"],
                    "confidence": intelligent_meal["confidence"]
                })
            
            # Glucose rescue (same as before)
            glucose_rescue_enabled = self.config.get("glucose_rescue_enabled", False)
            glucose_rescue_threshold = self.config.get("glucose_rescue_threshold", 70.0)
            glucose_rescue_carbs = self.config.get("glucose_rescue_carbs", 15.0)
            
            if glucose_rescue_enabled and current_cgm < glucose_rescue_threshold:
                carbs_ingested += glucose_rescue_carbs
                gi_factor = 2.0
                meal_type = f"{meal_type}+rescue" if meal_type != "none" else "glucose_rescue"
                meal_source = f"{meal_source}+rescue" if meal_source != "none" else "rescue"
                print(f"🚑 GLUCOSE RESCUE: CGM {current_cgm:.1f} mg/dL, adding {glucose_rescue_carbs}g carbs")

            # Log meal information
            self.simulation_data["meals_ingested_g"].append(carbs_ingested)
            self.simulation_data["protein_ingested_g"].append(0.0)
            self.simulation_data["fat_ingested_g"].append(0.0)
            self.simulation_data["exercise_events_details"].append(None)

            # Create carbs details for patient
            carbs_details_for_step = None
            if carbs_ingested > 0:
                carbs_details_for_step = {
                    "grams": carbs_ingested,
                    "gi_factor": gi_factor,
                    "meal_type": meal_type,
                    "meal_source": meal_source
                }

            # Step patient simulation
            self.patient.step(
                basal_insulin=basal_for_step,
                bolus_insulin=bolus_insulin,
                carbs_details=carbs_details_for_step,
                protein_ingested=0.0,
                fat_ingested=0.0,
                exercise_event=None
            )

            # Calculate reward (same as before)
            reward = self._calculate_reward(current_cgm)
            self.simulation_data["rewards"].append(reward)

            # Progress reporting
            if (step_idx + 1) % (self.max_simulation_steps // 20 if self.max_simulation_steps >= 20 else 1) == 0:
                meal_info = f", Meal: {meal_type} ({carbs_ingested:.1f}g)" if carbs_ingested > 0 else ""
                print(f"  Step {step_idx + 1}/{self.max_simulation_steps}. CGM: {current_cgm:.2f}{meal_info}")
        
        # Add intelligent meal statistics
        self.simulation_data["intelligent_meal_stats"] = self.intelligent_meal_system.get_statistics()
        
        print("Intelligent simulation finished.")
        return self.simulation_data
    
    def _calculate_reward(self, cgm_value: float) -> float:
        """Calculate reward based on CGM value (simplified version)."""
        if 70 <= cgm_value <= 180:
            return 1.0
        elif cgm_value < 70:
            return -5.0 * (70 - cgm_value) / 70
        else:  # cgm_value > 180
            return -2.0 * (cgm_value - 180) / 180

def main():
    """Main function to run intelligent meal simulation."""
    print("=== DiaGuardianAI Intelligent Meal Detection Simulation ===")
    
    # Configuration
    SIMULATION_DURATION_HOURS = 24  # 1 day for testing
    
    # 1. Set up paths
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    lstm_predictor_model_dir = os.path.join(project_root, "DiaGuardianAI", "models", "lstm_predictor_example_run")
    pattern_advisor_model_path = os.path.join(project_root, "DiaGuardianAI", "models", "pattern_advisor_agent_model", "pattern_advisor_regressor_v1.pkl")

    # 2. Load models
    glucose_predictor = load_glucose_predictor(lstm_predictor_model_dir)
    advisor_agent = load_trained_pattern_advisor(pattern_advisor_model_path)

    if not advisor_agent:
        print("Failed to load PatternAdvisorAgent. Exiting.")
        return

    # 3. Initialize patient with better parameters
    patient_params = {
        "initial_glucose": 120.0,
        "ISF": 40.0,  # More sensitive to insulin
        "CR": 12.0,   # Better carb ratio
        "basal_rate_U_hr": 0.8,  # Lower basal rate
    }
    patient = SyntheticPatient(params=patient_params)

    # 4. Configure intelligent simulation
    sim_config = {
        "max_simulation_steps": SIMULATION_DURATION_HOURS * 12,
        "time_step_minutes": 5,
        
        # Intelligent meal system
        "random_injection_enabled": True,
        "injection_probability_per_hour": 0.25,  # 25% chance per hour
        "min_meal_interval_minutes": 90,  # 1.5 hours between meals
        "spike_detection_enabled": True,
        "spike_threshold_mg_dl": 20.0,
        
        # Safety systems
        "glucose_rescue_enabled": True,
        "glucose_rescue_threshold": 70.0,
        "glucose_rescue_carbs": 15.0,
        
        # Optional: Add some scheduled meals for comparison
        "meal_schedule": {
            480: {"grams": 50, "gi_factor": 1.0, "type": "scheduled_breakfast"},  # 8:00 AM
            720: {"grams": 40, "gi_factor": 1.0, "type": "scheduled_lunch"},     # 12:00 PM
        }
    }

    # 5. Run intelligent simulation
    engine = IntelligentSimulationEngine(patient=patient, agent=advisor_agent, config=sim_config)
    results = engine.run()

    # 6. Analyze results
    print("\n=== INTELLIGENT MEAL SIMULATION RESULTS ===")
    
    cgm_readings = results["cgm_readings"]
    if cgm_readings:
        avg_cgm = np.mean(cgm_readings)
        std_cgm = np.std(cgm_readings)
        min_cgm = np.min(cgm_readings)
        max_cgm = np.max(cgm_readings)
        
        # Time in ranges
        tir_ideal = np.mean([(70 <= cgm <= 180) for cgm in cgm_readings]) * 100
        tir_tight = np.mean([(70 <= cgm <= 140) for cgm in cgm_readings]) * 100
        time_hypo_l1 = np.mean([cgm < 70 for cgm in cgm_readings]) * 100
        time_hypo_l2 = np.mean([cgm < 54 for cgm in cgm_readings]) * 100
        time_hyper_l1 = np.mean([cgm > 180 for cgm in cgm_readings]) * 100
        time_hyper_l2 = np.mean([cgm > 250 for cgm in cgm_readings]) * 100
        
        print(f"Simulation duration: {SIMULATION_DURATION_HOURS} hours")
        print(f"Average CGM: {avg_cgm:.2f} mg/dL")
        print(f"Std Dev CGM: {std_cgm:.2f} mg/dL")
        print(f"Min CGM: {min_cgm:.2f} mg/dL")
        print(f"Max CGM: {max_cgm:.2f} mg/dL")
        print(f"Time in Ideal Range (70-180 mg/dL): {tir_ideal:.2f}%")
        print(f"Time in Tight Range (70-140 mg/dL): {tir_tight:.2f}%")
        print(f"Time Below 70 mg/dL (Hypo L1): {time_hypo_l1:.2f}%")
        print(f"Time Below 54 mg/dL (Hypo L2): {time_hypo_l2:.2f}%")
        print(f"Time Above 180 mg/dL (Hyper L1): {time_hyper_l1:.2f}%")
        print(f"Time Above 250 mg/dL (Hyper L2): {time_hyper_l2:.2f}%")
    
    # Intelligent meal statistics
    meal_stats = results.get("intelligent_meal_stats", {})
    print(f"\n=== INTELLIGENT MEAL SYSTEM STATISTICS ===")
    print(f"Total detected meals: {meal_stats.get('total_detected_meals', 0)}")
    print(f"Total injected meals: {meal_stats.get('total_injected_meals', 0)}")
    print(f"Detection methods: {meal_stats.get('detection_methods', {})}")
    print(f"Meal types: {meal_stats.get('meal_types', {})}")
    
    # Show intelligent meals
    intelligent_meals = results.get("intelligent_meals", [])
    if intelligent_meals:
        print(f"\n=== INTELLIGENT MEALS DETECTED/INJECTED ===")
        for meal in intelligent_meals:
            hours = meal["time"] // 60
            minutes = meal["time"] % 60
            print(f"  {hours:02d}:{minutes:02d} - {meal['type']}: {meal['carbs']:.1f}g (method: {meal['method']}, confidence: {meal['confidence']:.2f})")

if __name__ == "__main__":
    main()
