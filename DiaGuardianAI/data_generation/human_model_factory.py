# DiaGuardianAI Human Model Factory
# Creates diverse, realistic human diabetes models for high-quality synthetic data generation

import numpy as np
import random
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import json
import os

class DiabetesType(Enum):
    TYPE_1 = "type_1"
    TYPE_2 = "type_2"
    GESTATIONAL = "gestational"
    MODY = "mody"

class ActivityLevel(Enum):
    SEDENTARY = "sedentary"
    LIGHTLY_ACTIVE = "lightly_active"
    MODERATELY_ACTIVE = "moderately_active"
    VERY_ACTIVE = "very_active"
    EXTREMELY_ACTIVE = "extremely_active"

class DietType(Enum):
    STANDARD = "standard"
    LOW_CARB = "low_carb"
    KETO = "keto"
    MEDITERRANEAN = "mediterranean"
    VEGETARIAN = "vegetarian"
    VEGAN = "vegan"

@dataclass
class HumanProfile:
    """Comprehensive human diabetes profile for realistic simulation."""
    
    # Basic Demographics
    patient_id: str
    age: int
    gender: str  # "male", "female", "other"
    weight_kg: float
    height_cm: float
    bmi: float
    
    # Diabetes Characteristics
    diabetes_type: DiabetesType
    years_with_diabetes: int
    hba1c_percent: float  # Target HbA1c
    
    # Insulin Parameters (Core Physiology)
    isf: float  # Insulin Sensitivity Factor (mg/dL per unit)
    cr: float   # Carb Ratio (grams per unit)
    basal_rate_u_hr: float  # Baseline insulin need
    
    # Advanced Physiological Parameters
    insulin_absorption_rate: float  # How fast insulin is absorbed
    glucose_absorption_rate: float  # How fast carbs are absorbed
    dawn_phenomenon_mg_dl: float   # Morning glucose rise
    somogyi_effect_strength: float # Rebound hyperglycemia strength
    
    # Lifestyle Factors
    activity_level: ActivityLevel
    diet_type: DietType
    stress_level: float  # 0.0-1.0
    sleep_quality: float  # 0.0-1.0
    
    # Meal Patterns
    meals_per_day: int
    snacks_per_day: int
    meal_timing_variability: float  # How much meal times vary
    carb_counting_accuracy: float   # How accurate carb counting is
    
    # Behavioral Patterns
    compliance_rate: float  # How often follows treatment plan
    bg_check_frequency: int  # Times per day checking glucose
    exercise_frequency: int  # Times per week
    
    # Physiological Variability
    glucose_variability: float  # Natural glucose fluctuation
    insulin_variability: float  # Insulin absorption variability
    
    # Advanced Parameters for Simulation
    simulation_params: Dict[str, float] = field(default_factory=dict)

class HumanModelFactory:
    """Factory for creating diverse, realistic human diabetes models."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.profiles_database: List[HumanProfile] = []
        self.population_statistics = self._load_population_statistics()
        
        print("HumanModelFactory initialized")
        print(f"  Target: Generate diverse human models for accurate synthetic data")
        print(f"  Goal: >90% accuracy through realistic physiological diversity")
    
    def _load_population_statistics(self) -> Dict[str, Any]:
        """Load real-world diabetes population statistics."""
        return {
            "age_distribution": {
                "type_1": {"mean": 25, "std": 15, "min": 5, "max": 80},
                "type_2": {"mean": 55, "std": 12, "min": 30, "max": 85}
            },
            "weight_distribution": {
                "male": {"mean": 85, "std": 15, "min": 50, "max": 150},
                "female": {"mean": 70, "std": 12, "min": 45, "max": 120}
            },
            "isf_distribution": {
                "type_1": {"mean": 50, "std": 20, "min": 20, "max": 100},
                "type_2": {"mean": 30, "std": 15, "min": 15, "max": 80}
            },
            "cr_distribution": {
                "type_1": {"mean": 12, "std": 4, "min": 6, "max": 25},
                "type_2": {"mean": 8, "std": 3, "min": 4, "max": 18}
            },
            "basal_distribution": {
                "type_1": {"mean": 1.0, "std": 0.4, "min": 0.3, "max": 2.5},
                "type_2": {"mean": 0.6, "std": 0.3, "min": 0.2, "max": 1.8}
            }
        }
    
    def generate_realistic_profile(self, diabetes_type: DiabetesType = None) -> HumanProfile:
        """Generate a single realistic human profile."""
        
        # Random diabetes type if not specified
        if diabetes_type is None:
            diabetes_type = random.choice([DiabetesType.TYPE_1, DiabetesType.TYPE_2])
        
        # Generate basic demographics
        age = self._sample_from_distribution(
            self.population_statistics["age_distribution"][diabetes_type.value]
        )
        gender = random.choice(["male", "female"])
        
        weight_kg = self._sample_from_distribution(
            self.population_statistics["weight_distribution"][gender]
        )
        height_cm = random.uniform(150, 190) if gender == "male" else random.uniform(145, 180)
        bmi = weight_kg / ((height_cm / 100) ** 2)
        
        # Generate insulin parameters based on diabetes type
        isf = self._sample_from_distribution(
            self.population_statistics["isf_distribution"][diabetes_type.value]
        )
        cr = self._sample_from_distribution(
            self.population_statistics["cr_distribution"][diabetes_type.value]
        )
        basal_rate = self._sample_from_distribution(
            self.population_statistics["basal_distribution"][diabetes_type.value]
        )
        
        # Generate lifestyle factors
        activity_level = random.choice(list(ActivityLevel))
        diet_type = random.choice(list(DietType))
        
        # Generate meal patterns based on lifestyle
        meals_per_day = random.choice([2, 3, 4, 5])
        snacks_per_day = random.choice([0, 1, 2, 3])
        
        # Generate physiological parameters
        years_with_diabetes = max(1, int(np.random.exponential(8)))  # Most recent diagnosis
        hba1c = random.uniform(6.5, 9.5)  # Realistic HbA1c range
        
        # Advanced physiological parameters
        insulin_absorption_rate = random.uniform(0.8, 1.2)  # Relative to standard
        glucose_absorption_rate = random.uniform(0.7, 1.3)
        dawn_phenomenon = random.uniform(0, 40)  # mg/dL morning rise
        somogyi_strength = random.uniform(0.0, 0.3)
        
        # Behavioral factors
        compliance_rate = np.random.beta(8, 2)  # Most people are fairly compliant
        carb_counting_accuracy = np.random.beta(6, 3)  # Moderate accuracy
        
        # Variability factors
        glucose_variability = random.uniform(0.1, 0.4)
        insulin_variability = random.uniform(0.05, 0.25)
        
        # Create simulation parameters
        simulation_params = self._generate_simulation_parameters(
            diabetes_type, isf, cr, basal_rate, weight_kg, activity_level
        )
        
        profile = HumanProfile(
            patient_id=f"patient_{len(self.profiles_database):06d}",
            age=int(age),
            gender=gender,
            weight_kg=round(weight_kg, 1),
            height_cm=round(height_cm, 1),
            bmi=round(bmi, 1),
            diabetes_type=diabetes_type,
            years_with_diabetes=years_with_diabetes,
            hba1c_percent=round(hba1c, 1),
            isf=round(isf, 1),
            cr=round(cr, 1),
            basal_rate_u_hr=round(basal_rate, 2),
            insulin_absorption_rate=round(insulin_absorption_rate, 2),
            glucose_absorption_rate=round(glucose_absorption_rate, 2),
            dawn_phenomenon_mg_dl=round(dawn_phenomenon, 1),
            somogyi_effect_strength=round(somogyi_strength, 2),
            activity_level=activity_level,
            diet_type=diet_type,
            stress_level=round(random.uniform(0.1, 0.8), 2),
            sleep_quality=round(random.uniform(0.4, 1.0), 2),
            meals_per_day=meals_per_day,
            snacks_per_day=snacks_per_day,
            meal_timing_variability=round(random.uniform(0.1, 0.5), 2),
            carb_counting_accuracy=round(carb_counting_accuracy, 2),
            compliance_rate=round(compliance_rate, 2),
            bg_check_frequency=random.choice([2, 3, 4, 6, 8, 12]),
            exercise_frequency=random.choice([0, 1, 2, 3, 4, 5, 7]),
            glucose_variability=round(glucose_variability, 2),
            insulin_variability=round(insulin_variability, 2),
            simulation_params=simulation_params
        )
        
        return profile
    
    def _sample_from_distribution(self, dist_params: Dict[str, float]) -> float:
        """Sample from a normal distribution with bounds."""
        value = np.random.normal(dist_params["mean"], dist_params["std"])
        return max(dist_params["min"], min(dist_params["max"], value))
    
    def _generate_simulation_parameters(
        self, 
        diabetes_type: DiabetesType,
        isf: float,
        cr: float, 
        basal_rate: float,
        weight_kg: float,
        activity_level: ActivityLevel
    ) -> Dict[str, float]:
        """Generate detailed simulation parameters for the patient model."""
        
        # Base parameters
        params = {
            "initial_glucose": random.uniform(90, 140),
            "target_glucose": random.uniform(95, 110),
            "ISF": isf,
            "CR": cr,
            "basal_rate_U_hr": basal_rate,
            "body_weight_kg": weight_kg,
        }
        
        # Insulin action parameters (critical for realistic behavior)
        if diabetes_type == DiabetesType.TYPE_1:
            # Type 1: More insulin sensitive, faster action
            params.update({
                "k_u_id": random.uniform(0.0008, 0.0015),  # Insulin-dependent utilization
                "k_egp": random.uniform(0.015, 0.025),     # Endogenous glucose production
                "tau_insulin": random.uniform(45, 75),     # Insulin action time
                "tau_carbs": random.uniform(15, 30),       # Carb absorption time
            })
        else:
            # Type 2: More insulin resistant, slower action
            params.update({
                "k_u_id": random.uniform(0.0003, 0.0008),  # Lower insulin sensitivity
                "k_egp": random.uniform(0.020, 0.035),     # Higher glucose production
                "tau_insulin": random.uniform(60, 100),    # Slower insulin action
                "tau_carbs": random.uniform(20, 45),       # Slower carb absorption
            })
        
        # Activity level adjustments
        activity_multipliers = {
            ActivityLevel.SEDENTARY: 0.8,
            ActivityLevel.LIGHTLY_ACTIVE: 0.9,
            ActivityLevel.MODERATELY_ACTIVE: 1.0,
            ActivityLevel.VERY_ACTIVE: 1.2,
            ActivityLevel.EXTREMELY_ACTIVE: 1.4
        }
        
        multiplier = activity_multipliers[activity_level]
        params["k_u_id"] *= multiplier  # More active = more insulin sensitive
        params["basal_rate_U_hr"] *= (2.0 - multiplier)  # More active = less basal needed
        
        return params
    
    def generate_population(self, size: int, type_1_ratio: float = 0.3) -> List[HumanProfile]:
        """Generate a diverse population of human models."""
        
        print(f"\n=== GENERATING DIVERSE HUMAN POPULATION ===")
        print(f"Target size: {size} patients")
        print(f"Type 1 ratio: {type_1_ratio:.1%}")
        
        population = []
        
        # Calculate how many of each type
        num_type_1 = int(size * type_1_ratio)
        num_type_2 = size - num_type_1
        
        # Generate Type 1 patients
        for i in range(num_type_1):
            profile = self.generate_realistic_profile(DiabetesType.TYPE_1)
            population.append(profile)
            if (i + 1) % 10 == 0:
                print(f"  Generated {i + 1}/{num_type_1} Type 1 patients")
        
        # Generate Type 2 patients
        for i in range(num_type_2):
            profile = self.generate_realistic_profile(DiabetesType.TYPE_2)
            population.append(profile)
            if (i + 1) % 10 == 0:
                print(f"  Generated {i + 1}/{num_type_2} Type 2 patients")
        
        # Shuffle to mix types
        random.shuffle(population)
        
        # Store in database
        self.profiles_database.extend(population)
        
        print(f"\n✅ POPULATION GENERATION COMPLETE")
        print(f"  Total patients: {len(population)}")
        print(f"  Type 1: {num_type_1} ({type_1_ratio:.1%})")
        print(f"  Type 2: {num_type_2} ({1-type_1_ratio:.1%})")
        
        return population
    
    def get_population_statistics(self, population: List[HumanProfile]) -> Dict[str, Any]:
        """Analyze population statistics for quality assurance."""
        
        if not population:
            return {}
        
        # Basic demographics
        ages = [p.age for p in population]
        weights = [p.weight_kg for p in population]
        bmis = [p.bmi for p in population]
        
        # Diabetes parameters
        isfs = [p.isf for p in population]
        crs = [p.cr for p in population]
        basals = [p.basal_rate_u_hr for p in population]
        hba1cs = [p.hba1c_percent for p in population]
        
        # Type distribution
        type_1_count = sum(1 for p in population if p.diabetes_type == DiabetesType.TYPE_1)
        type_2_count = len(population) - type_1_count
        
        stats = {
            "population_size": len(population),
            "demographics": {
                "age": {"mean": np.mean(ages), "std": np.std(ages), "range": [min(ages), max(ages)]},
                "weight": {"mean": np.mean(weights), "std": np.std(weights), "range": [min(weights), max(weights)]},
                "bmi": {"mean": np.mean(bmis), "std": np.std(bmis), "range": [min(bmis), max(bmis)]},
            },
            "diabetes_parameters": {
                "isf": {"mean": np.mean(isfs), "std": np.std(isfs), "range": [min(isfs), max(isfs)]},
                "cr": {"mean": np.mean(crs), "std": np.std(crs), "range": [min(crs), max(crs)]},
                "basal": {"mean": np.mean(basals), "std": np.std(basals), "range": [min(basals), max(basals)]},
                "hba1c": {"mean": np.mean(hba1cs), "std": np.std(hba1cs), "range": [min(hba1cs), max(hba1cs)]},
            },
            "type_distribution": {
                "type_1": {"count": type_1_count, "percentage": type_1_count / len(population) * 100},
                "type_2": {"count": type_2_count, "percentage": type_2_count / len(population) * 100},
            }
        }
        
        return stats
    
    def save_population(self, population: List[HumanProfile], filepath: str):
        """Save population to JSON file."""
        
        # Convert to serializable format
        serializable_population = []
        for profile in population:
            profile_dict = {
                "patient_id": profile.patient_id,
                "age": profile.age,
                "gender": profile.gender,
                "weight_kg": profile.weight_kg,
                "height_cm": profile.height_cm,
                "bmi": profile.bmi,
                "diabetes_type": profile.diabetes_type.value,
                "years_with_diabetes": profile.years_with_diabetes,
                "hba1c_percent": profile.hba1c_percent,
                "isf": profile.isf,
                "cr": profile.cr,
                "basal_rate_u_hr": profile.basal_rate_u_hr,
                "insulin_absorption_rate": profile.insulin_absorption_rate,
                "glucose_absorption_rate": profile.glucose_absorption_rate,
                "dawn_phenomenon_mg_dl": profile.dawn_phenomenon_mg_dl,
                "somogyi_effect_strength": profile.somogyi_effect_strength,
                "activity_level": profile.activity_level.value,
                "diet_type": profile.diet_type.value,
                "stress_level": profile.stress_level,
                "sleep_quality": profile.sleep_quality,
                "meals_per_day": profile.meals_per_day,
                "snacks_per_day": profile.snacks_per_day,
                "meal_timing_variability": profile.meal_timing_variability,
                "carb_counting_accuracy": profile.carb_counting_accuracy,
                "compliance_rate": profile.compliance_rate,
                "bg_check_frequency": profile.bg_check_frequency,
                "exercise_frequency": profile.exercise_frequency,
                "glucose_variability": profile.glucose_variability,
                "insulin_variability": profile.insulin_variability,
                "simulation_params": profile.simulation_params
            }
            serializable_population.append(profile_dict)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_population, f, indent=2)
        
        print(f"Population saved to {filepath}")
    
    def load_population(self, filepath: str) -> List[HumanProfile]:
        """Load population from JSON file."""
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        population = []
        for profile_dict in data:
            profile = HumanProfile(
                patient_id=profile_dict["patient_id"],
                age=profile_dict["age"],
                gender=profile_dict["gender"],
                weight_kg=profile_dict["weight_kg"],
                height_cm=profile_dict["height_cm"],
                bmi=profile_dict["bmi"],
                diabetes_type=DiabetesType(profile_dict["diabetes_type"]),
                years_with_diabetes=profile_dict["years_with_diabetes"],
                hba1c_percent=profile_dict["hba1c_percent"],
                isf=profile_dict["isf"],
                cr=profile_dict["cr"],
                basal_rate_u_hr=profile_dict["basal_rate_u_hr"],
                insulin_absorption_rate=profile_dict["insulin_absorption_rate"],
                glucose_absorption_rate=profile_dict["glucose_absorption_rate"],
                dawn_phenomenon_mg_dl=profile_dict["dawn_phenomenon_mg_dl"],
                somogyi_effect_strength=profile_dict["somogyi_effect_strength"],
                activity_level=ActivityLevel(profile_dict["activity_level"]),
                diet_type=DietType(profile_dict["diet_type"]),
                stress_level=profile_dict["stress_level"],
                sleep_quality=profile_dict["sleep_quality"],
                meals_per_day=profile_dict["meals_per_day"],
                snacks_per_day=profile_dict["snacks_per_day"],
                meal_timing_variability=profile_dict["meal_timing_variability"],
                carb_counting_accuracy=profile_dict["carb_counting_accuracy"],
                compliance_rate=profile_dict["compliance_rate"],
                bg_check_frequency=profile_dict["bg_check_frequency"],
                exercise_frequency=profile_dict["exercise_frequency"],
                glucose_variability=profile_dict["glucose_variability"],
                insulin_variability=profile_dict["insulin_variability"],
                simulation_params=profile_dict["simulation_params"]
            )
            population.append(profile)
        
        print(f"Population loaded from {filepath}: {len(population)} patients")
        return population
