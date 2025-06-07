# DiaGuardianAI Synthetic Data Generator
# Generates high-quality training data using diverse human models

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import random
from datetime import datetime, timedelta
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import pickle

from .human_model_factory import HumanModelFactory, HumanProfile
from .synthetic_patient_model import SyntheticPatient
from ..core.intelligent_meal_system import IntelligentMealSystem
from ..agents.pattern_advisor_agent import PatternAdvisorAgent

class SyntheticDataGenerator:
    """
    High-performance synthetic data generator for training DiaGuardianAI models.
    
    Uses diverse human models to create realistic, varied training scenarios.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.human_factory = HumanModelFactory()
        
        # Data generation parameters
        self.simulation_duration_hours = self.config.get("simulation_duration_hours", 168)  # 1 week
        self.time_step_minutes = self.config.get("time_step_minutes", 5)
        self.num_parallel_workers = self.config.get("num_parallel_workers", mp.cpu_count() - 1)
        
        # Data quality parameters
        self.target_time_in_range = self.config.get("target_time_in_range", 0.70)  # 70%
        self.max_hypoglycemia_time = self.config.get("max_hypoglycemia_time", 0.04)  # 4%
        self.data_quality_threshold = self.config.get("data_quality_threshold", 0.8)
        
        print(f"SyntheticDataGenerator initialized")
        print(f"  Simulation duration: {self.simulation_duration_hours} hours")
        print(f"  Time step: {self.time_step_minutes} minutes")
        print(f"  Parallel workers: {self.num_parallel_workers}")
        print(f"  Target TIR: {self.target_time_in_range:.1%}")
    
    def generate_training_dataset(
        self, 
        num_patients: int,
        scenarios_per_patient: int = 3,
        save_path: str = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive training dataset with diverse scenarios.
        
        Args:
            num_patients: Number of different human models to create
            scenarios_per_patient: Number of different scenarios per patient
            save_path: Path to save the dataset
            
        Returns:
            Dictionary containing training data and metadata
        """
        
        print(f"\n=== GENERATING COMPREHENSIVE TRAINING DATASET ===")
        print(f"Target: {num_patients} patients × {scenarios_per_patient} scenarios = {num_patients * scenarios_per_patient} simulations")
        
        # Step 1: Generate diverse human population
        print(f"\n📊 Step 1: Generating {num_patients} diverse human models...")
        population = self.human_factory.generate_population(num_patients)
        
        # Step 2: Create simulation scenarios
        print(f"\n🎯 Step 2: Creating {scenarios_per_patient} scenarios per patient...")
        simulation_tasks = []
        
        for patient_idx, profile in enumerate(population):
            for scenario_idx in range(scenarios_per_patient):
                scenario_config = self._create_scenario_config(profile, scenario_idx)
                
                task = {
                    "patient_id": profile.patient_id,
                    "patient_idx": patient_idx,
                    "scenario_idx": scenario_idx,
                    "profile": profile,
                    "scenario_config": scenario_config,
                    "simulation_id": f"{profile.patient_id}_scenario_{scenario_idx}"
                }
                simulation_tasks.append(task)
        
        print(f"  Created {len(simulation_tasks)} simulation tasks")
        
        # Step 3: Run simulations in parallel
        print(f"\n⚡ Step 3: Running simulations with {self.num_parallel_workers} workers...")
        
        all_simulation_data = []
        successful_simulations = 0
        failed_simulations = 0
        
        # Process in batches to manage memory
        batch_size = max(1, self.num_parallel_workers * 2)
        
        for batch_start in range(0, len(simulation_tasks), batch_size):
            batch_end = min(batch_start + batch_size, len(simulation_tasks))
            batch_tasks = simulation_tasks[batch_start:batch_end]
            
            print(f"  Processing batch {batch_start//batch_size + 1}/{(len(simulation_tasks)-1)//batch_size + 1} ({len(batch_tasks)} simulations)")
            
            with ProcessPoolExecutor(max_workers=self.num_parallel_workers) as executor:
                # Submit batch tasks
                future_to_task = {
                    executor.submit(self._run_single_simulation, task): task 
                    for task in batch_tasks
                }
                
                # Collect results
                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    try:
                        result = future.result()
                        if result and self._validate_simulation_quality(result):
                            all_simulation_data.append(result)
                            successful_simulations += 1
                        else:
                            failed_simulations += 1
                            print(f"    ❌ Failed quality check: {task['simulation_id']}")
                    except Exception as e:
                        failed_simulations += 1
                        print(f"    ❌ Simulation error: {task['simulation_id']} - {str(e)}")
            
            print(f"    Batch complete: {successful_simulations} successful, {failed_simulations} failed")
        
        # Step 4: Process and structure the data
        print(f"\n📈 Step 4: Processing {len(all_simulation_data)} successful simulations...")
        
        dataset = self._structure_training_data(all_simulation_data, population)
        
        # Step 5: Generate data quality report
        print(f"\n📋 Step 5: Generating data quality report...")
        quality_report = self._generate_quality_report(dataset, population)
        
        # Step 6: Save dataset if requested
        if save_path:
            print(f"\n💾 Step 6: Saving dataset to {save_path}...")
            self._save_dataset(dataset, save_path)
        
        print(f"\n✅ DATASET GENERATION COMPLETE")
        print(f"  Successful simulations: {successful_simulations}")
        print(f"  Failed simulations: {failed_simulations}")
        print(f"  Success rate: {successful_simulations/(successful_simulations+failed_simulations):.1%}")
        print(f"  Total data points: {dataset['metadata']['total_data_points']}")
        print(f"  Average TIR: {quality_report['average_time_in_range']:.1%}")
        
        return dataset
    
    def _create_scenario_config(self, profile: HumanProfile, scenario_idx: int) -> Dict[str, Any]:
        """Create different scenario configurations for the same patient."""
        
        base_config = {
            "simulation_duration_hours": self.simulation_duration_hours,
            "time_step_minutes": self.time_step_minutes,
            "random_injection_enabled": True,
            "spike_detection_enabled": True,
            "glucose_rescue_enabled": True,
            "glucose_rescue_threshold": 70.0,
            "glucose_rescue_carbs": 15.0,
        }
        
        # Scenario variations
        if scenario_idx == 0:
            # Standard scenario - regular meals
            base_config.update({
                "injection_probability_per_hour": 0.15,
                "min_meal_interval_minutes": 180,  # 3 hours
                "stress_multiplier": 1.0,
                "exercise_enabled": False,
                "scenario_name": "standard"
            })
        
        elif scenario_idx == 1:
            # High variability scenario - irregular meals, stress
            base_config.update({
                "injection_probability_per_hour": 0.25,
                "min_meal_interval_minutes": 90,   # 1.5 hours
                "stress_multiplier": 1.3,
                "exercise_enabled": True,
                "exercise_probability": 0.1,
                "scenario_name": "high_variability"
            })
        
        elif scenario_idx == 2:
            # Low carb scenario - fewer, smaller meals
            base_config.update({
                "injection_probability_per_hour": 0.08,
                "min_meal_interval_minutes": 240,  # 4 hours
                "carb_reduction_factor": 0.6,
                "stress_multiplier": 0.9,
                "exercise_enabled": True,
                "exercise_probability": 0.15,
                "scenario_name": "low_carb"
            })
        
        else:
            # Random scenario - completely random parameters
            base_config.update({
                "injection_probability_per_hour": random.uniform(0.05, 0.3),
                "min_meal_interval_minutes": random.randint(60, 300),
                "stress_multiplier": random.uniform(0.8, 1.5),
                "exercise_enabled": random.choice([True, False]),
                "exercise_probability": random.uniform(0.05, 0.2),
                "scenario_name": "random"
            })
        
        return base_config
    
    def _run_single_simulation(self, task: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Run a single simulation (designed for parallel execution)."""
        
        try:
            profile = task["profile"]
            scenario_config = task["scenario_config"]
            
            # Create patient with profile parameters
            patient_params = profile.simulation_params.copy()
            patient = SyntheticPatient(params=patient_params)
            
            # Create intelligent meal system
            meal_system = IntelligentMealSystem(scenario_config)
            
            # Run simulation
            simulation_data = self._simulate_patient_scenario(
                patient, meal_system, scenario_config, profile
            )
            
            # Add metadata
            simulation_data.update({
                "patient_id": profile.patient_id,
                "scenario_name": scenario_config["scenario_name"],
                "simulation_id": task["simulation_id"],
                "profile_summary": self._summarize_profile(profile)
            })
            
            return simulation_data
            
        except Exception as e:
            print(f"Simulation error for {task['simulation_id']}: {str(e)}")
            return None
    
    def _simulate_patient_scenario(
        self, 
        patient: SyntheticPatient,
        meal_system: IntelligentMealSystem,
        config: Dict[str, Any],
        profile: HumanProfile
    ) -> Dict[str, Any]:
        """Simulate a single patient scenario."""
        
        duration_hours = config["simulation_duration_hours"]
        time_step_minutes = config["time_step_minutes"]
        max_steps = int(duration_hours * 60 / time_step_minutes)
        
        # Data storage
        simulation_data = {
            "time_steps": [],
            "cgm_readings": [],
            "insulin_basal": [],
            "insulin_bolus": [],
            "carbs_ingested": [],
            "meals_detected": [],
            "iob": [],
            "cob": [],
            "glucose_trends": [],
            "meal_events": [],
            "exercise_events": [],
            "stress_events": []
        }
        
        # Simulation loop
        for step in range(max_steps):
            current_time_minutes = step * time_step_minutes
            
            # Get current patient state
            patient_state = patient.get_state()
            current_cgm = patient_state.get("cgm", 100.0)
            
            # Process intelligent meal system
            meal_event = meal_system.process_simulation_step(current_time_minutes, current_cgm)
            
            # Determine insulin dosing (simplified for data generation)
            basal_insulin, bolus_insulin = self._calculate_insulin_dose(
                current_cgm, meal_event, profile, config
            )
            
            # Apply scenario-specific modifications
            basal_insulin *= config.get("stress_multiplier", 1.0)
            
            # Handle exercise events
            exercise_event = None
            if config.get("exercise_enabled", False):
                if random.random() < config.get("exercise_probability", 0.1) / 12:  # Per 5-min step
                    exercise_event = {"type": "moderate", "duration": 30}
            
            # Step patient simulation
            carbs_details = None
            carbs_amount = 0.0
            
            if meal_event:
                carbs_amount = meal_event["grams"] * config.get("carb_reduction_factor", 1.0)
                carbs_details = {
                    "grams": carbs_amount,
                    "gi_factor": meal_event["gi_factor"],
                    "meal_type": meal_event["type"]
                }
            
            # Glucose rescue
            if current_cgm < config.get("glucose_rescue_threshold", 70.0):
                rescue_carbs = config.get("glucose_rescue_carbs", 15.0)
                if carbs_details:
                    carbs_details["grams"] += rescue_carbs
                else:
                    carbs_details = {"grams": rescue_carbs, "gi_factor": 2.0, "meal_type": "rescue"}
                carbs_amount += rescue_carbs
            
            # Step the patient
            patient.step(
                basal_insulin=basal_insulin * time_step_minutes / 60.0,
                bolus_insulin=bolus_insulin,
                carbs_details=carbs_details,
                exercise_event=exercise_event
            )
            
            # Record data
            new_state = patient.get_state()
            simulation_data["time_steps"].append(current_time_minutes)
            simulation_data["cgm_readings"].append(new_state.get("cgm", 100.0))
            simulation_data["insulin_basal"].append(basal_insulin)
            simulation_data["insulin_bolus"].append(bolus_insulin)
            simulation_data["carbs_ingested"].append(carbs_amount)
            simulation_data["iob"].append(new_state.get("iob", 0.0))
            simulation_data["cob"].append(new_state.get("cob", 0.0))
            
            # Calculate glucose trend
            if len(simulation_data["cgm_readings"]) >= 3:
                recent_cgm = simulation_data["cgm_readings"][-3:]
                trend = (recent_cgm[-1] - recent_cgm[0]) / 2  # mg/dL per 10 minutes
                simulation_data["glucose_trends"].append(trend)
            else:
                simulation_data["glucose_trends"].append(0.0)
            
            # Record events
            simulation_data["meals_detected"].append(1 if meal_event else 0)
            simulation_data["meal_events"].append(meal_event)
            simulation_data["exercise_events"].append(exercise_event)
            simulation_data["stress_events"].append(config.get("stress_multiplier", 1.0))
        
        return simulation_data
    
    def _calculate_insulin_dose(
        self, 
        current_cgm: float, 
        meal_event: Optional[Dict], 
        profile: HumanProfile,
        config: Dict[str, Any]
    ) -> Tuple[float, float]:
        """Calculate insulin dose based on current state and meal."""
        
        # Basal insulin
        basal_rate = profile.basal_rate_u_hr
        
        # Bolus insulin
        bolus_insulin = 0.0
        
        # Meal bolus
        if meal_event:
            carb_grams = meal_event["grams"] * config.get("carb_reduction_factor", 1.0)
            meal_bolus = carb_grams / profile.cr
            bolus_insulin += meal_bolus * profile.carb_counting_accuracy
        
        # Correction bolus
        if current_cgm > 150:
            correction_needed = (current_cgm - 120) / profile.isf
            bolus_insulin += correction_needed * profile.compliance_rate
        
        # Safety limits
        basal_rate = max(0.0, min(3.0, basal_rate))
        bolus_insulin = max(0.0, min(10.0, bolus_insulin))
        
        return basal_rate, bolus_insulin
    
    def _validate_simulation_quality(self, simulation_data: Dict[str, Any]) -> bool:
        """Validate that simulation data meets quality standards."""
        
        cgm_readings = simulation_data.get("cgm_readings", [])
        if not cgm_readings:
            return False
        
        # Calculate time in range
        tir = np.mean([(70 <= cgm <= 180) for cgm in cgm_readings])
        
        # Calculate hypoglycemia time
        hypo_time = np.mean([cgm < 70 for cgm in cgm_readings])
        
        # Quality checks
        quality_score = 0.0
        
        # Time in range should be reasonable
        if 0.3 <= tir <= 0.9:
            quality_score += 0.4
        
        # Hypoglycemia should be minimal
        if hypo_time <= self.max_hypoglycemia_time:
            quality_score += 0.3
        
        # CGM variability should be realistic
        cgm_std = np.std(cgm_readings)
        if 20 <= cgm_std <= 80:
            quality_score += 0.3
        
        return quality_score >= self.data_quality_threshold
    
    def _structure_training_data(
        self, 
        all_simulation_data: List[Dict[str, Any]], 
        population: List[HumanProfile]
    ) -> Dict[str, Any]:
        """Structure the simulation data for training."""
        
        # Combine all simulation data
        combined_data = {
            "cgm_readings": [],
            "insulin_basal": [],
            "insulin_bolus": [],
            "carbs_ingested": [],
            "iob": [],
            "cob": [],
            "glucose_trends": [],
            "patient_features": [],
            "scenario_features": [],
            "time_features": [],
            "simulation_ids": []
        }
        
        total_data_points = 0
        
        for sim_data in all_simulation_data:
            n_points = len(sim_data["cgm_readings"])
            total_data_points += n_points
            
            # Add simulation data
            combined_data["cgm_readings"].extend(sim_data["cgm_readings"])
            combined_data["insulin_basal"].extend(sim_data["insulin_basal"])
            combined_data["insulin_bolus"].extend(sim_data["insulin_bolus"])
            combined_data["carbs_ingested"].extend(sim_data["carbs_ingested"])
            combined_data["iob"].extend(sim_data["iob"])
            combined_data["cob"].extend(sim_data["cob"])
            combined_data["glucose_trends"].extend(sim_data["glucose_trends"])
            
            # Add features for each data point
            profile_summary = sim_data["profile_summary"]
            scenario_name = sim_data["scenario_name"]
            
            for i in range(n_points):
                # Patient features
                combined_data["patient_features"].append([
                    profile_summary["age"],
                    profile_summary["weight_kg"],
                    profile_summary["bmi"],
                    profile_summary["isf"],
                    profile_summary["cr"],
                    profile_summary["basal_rate_u_hr"],
                    profile_summary["diabetes_type_numeric"]
                ])
                
                # Scenario features
                scenario_numeric = {"standard": 0, "high_variability": 1, "low_carb": 2, "random": 3}
                combined_data["scenario_features"].append([
                    scenario_numeric.get(scenario_name, 3)
                ])
                
                # Time features
                time_minutes = sim_data["time_steps"][i]
                hour_of_day = (time_minutes / 60) % 24
                day_of_week = ((time_minutes / 60) // 24) % 7
                combined_data["time_features"].append([
                    hour_of_day,
                    day_of_week,
                    np.sin(2 * np.pi * hour_of_day / 24),
                    np.cos(2 * np.pi * hour_of_day / 24)
                ])
                
                combined_data["simulation_ids"].append(sim_data["simulation_id"])
        
        # Convert to numpy arrays
        for key in combined_data:
            if key != "simulation_ids":
                combined_data[key] = np.array(combined_data[key])
        
        # Create final dataset structure
        dataset = {
            "data": combined_data,
            "metadata": {
                "total_simulations": len(all_simulation_data),
                "total_data_points": total_data_points,
                "population_size": len(population),
                "simulation_duration_hours": self.simulation_duration_hours,
                "time_step_minutes": self.time_step_minutes,
                "generation_timestamp": datetime.now().isoformat()
            },
            "population_stats": self.human_factory.get_population_statistics(population)
        }
        
        return dataset
    
    def _summarize_profile(self, profile: HumanProfile) -> Dict[str, Any]:
        """Create a summary of patient profile for features."""
        return {
            "age": profile.age,
            "weight_kg": profile.weight_kg,
            "bmi": profile.bmi,
            "isf": profile.isf,
            "cr": profile.cr,
            "basal_rate_u_hr": profile.basal_rate_u_hr,
            "diabetes_type_numeric": 1 if profile.diabetes_type.value == "type_1" else 2
        }
    
    def _generate_quality_report(
        self, 
        dataset: Dict[str, Any], 
        population: List[HumanProfile]
    ) -> Dict[str, Any]:
        """Generate comprehensive quality report."""
        
        cgm_readings = dataset["data"]["cgm_readings"]
        
        # Calculate metrics
        tir = np.mean([(70 <= cgm <= 180) for cgm in cgm_readings])
        hypo_l1 = np.mean([cgm < 70 for cgm in cgm_readings])
        hypo_l2 = np.mean([cgm < 54 for cgm in cgm_readings])
        hyper_l1 = np.mean([cgm > 180 for cgm in cgm_readings])
        hyper_l2 = np.mean([cgm > 250 for cgm in cgm_readings])
        
        report = {
            "average_time_in_range": tir,
            "average_hypo_l1": hypo_l1,
            "average_hypo_l2": hypo_l2,
            "average_hyper_l1": hyper_l1,
            "average_hyper_l2": hyper_l2,
            "cgm_statistics": {
                "mean": np.mean(cgm_readings),
                "std": np.std(cgm_readings),
                "min": np.min(cgm_readings),
                "max": np.max(cgm_readings)
            }
        }
        
        return report
    
    def _save_dataset(self, dataset: Dict[str, Any], save_path: str):
        """Save dataset to file."""
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save as pickle for efficiency
        with open(save_path, 'wb') as f:
            pickle.dump(dataset, f)
        
        print(f"Dataset saved to {save_path}")
        print(f"  Size: {os.path.getsize(save_path) / 1024 / 1024:.1f} MB")
