# DiaGuardianAI Intelligent Meal Detection and Injection System
# This module implements the vision of automatic meal detection and random meal injection for training

import numpy as np
import random
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class MealPattern:
    """Represents a detected or predicted meal pattern."""
    time_minutes: int
    carbs_grams: float
    gi_factor: float
    meal_type: str
    confidence: float
    detection_method: str  # "cgm_spike", "pattern_recognition", "time_based", "random_injection"

class IntelligentMealSystem:
    """
    Advanced meal detection and injection system for DiaGuardianAI.
    
    PHASE 1: Random meal injection for training data generation
    PHASE 2: CGM spike detection for meal identification  
    PHASE 3: Pattern recognition for meal prediction
    PHASE 4: Behavioral learning for personalized meal timing
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Random meal injection parameters
        self.random_injection_enabled = config.get("random_injection_enabled", True)
        self.injection_probability_per_hour = config.get("injection_probability_per_hour", 0.15)  # 15% chance per hour
        self.min_meal_interval_minutes = config.get("min_meal_interval_minutes", 120)  # 2 hours between meals
        
        # Meal characteristics
        self.meal_types = {
            "breakfast": {"carbs_range": (40, 80), "gi_range": (0.8, 1.2), "time_range": (360, 600)},    # 6-10 AM
            "lunch": {"carbs_range": (30, 60), "gi_range": (0.7, 1.1), "time_range": (660, 840)},       # 11-2 PM  
            "dinner": {"carbs_range": (50, 100), "gi_range": (0.8, 1.3), "time_range": (1020, 1260)},   # 5-9 PM
            "snack": {"carbs_range": (10, 30), "gi_range": (0.6, 1.5), "time_range": (0, 1440)},        # Any time
        }
        
        # CGM spike detection parameters
        self.spike_detection_enabled = config.get("spike_detection_enabled", True)
        self.spike_threshold_mg_dl = config.get("spike_threshold_mg_dl", 30.0)  # 30 mg/dL rise
        self.spike_detection_window_minutes = config.get("spike_detection_window_minutes", 30)
        
        # Pattern recognition parameters
        self.pattern_recognition_enabled = config.get("pattern_recognition_enabled", False)  # Future feature
        self.learned_meal_patterns: List[MealPattern] = []
        
        # State tracking
        self.last_meal_time = -999999  # Very early time to allow first meal
        self.cgm_history: List[Tuple[int, float]] = []  # (time_minutes, cgm_value)
        self.detected_meals: List[MealPattern] = []
        self.injected_meals: List[MealPattern] = []
        
        print(f"IntelligentMealSystem initialized:")
        print(f"  Random injection: {self.random_injection_enabled}")
        print(f"  Spike detection: {self.spike_detection_enabled}")
        print(f"  Pattern recognition: {self.pattern_recognition_enabled}")
    
    def update_cgm_history(self, time_minutes: int, cgm_value: float):
        """Update CGM history for spike detection."""
        self.cgm_history.append((time_minutes, cgm_value))
        
        # Keep only recent history (last 2 hours)
        cutoff_time = time_minutes - 120
        self.cgm_history = [(t, cgm) for t, cgm in self.cgm_history if t >= cutoff_time]
    
    def detect_cgm_spike(self, current_time: int, current_cgm: float) -> Optional[MealPattern]:
        """Detect potential meal from CGM spike pattern."""
        if not self.spike_detection_enabled or len(self.cgm_history) < 6:
            return None
        
        # Look for rapid glucose rise in last 30 minutes
        window_start = current_time - self.spike_detection_window_minutes
        recent_cgm = [(t, cgm) for t, cgm in self.cgm_history if t >= window_start]
        
        if len(recent_cgm) < 3:
            return None
        
        # Calculate glucose rise rate
        start_cgm = recent_cgm[0][1]
        max_rise = current_cgm - start_cgm
        
        if max_rise >= self.spike_threshold_mg_dl:
            # Estimate meal size based on glucose rise
            estimated_carbs = max(10.0, min(100.0, max_rise * 1.5))  # Rough estimation
            
            # Determine meal type based on time of day
            time_of_day = current_time % 1440  # Minutes in day
            meal_type = self._determine_meal_type_by_time(time_of_day)
            
            detected_meal = MealPattern(
                time_minutes=current_time - 15,  # Assume meal was 15 min ago
                carbs_grams=estimated_carbs,
                gi_factor=1.0,
                meal_type=f"detected_{meal_type}",
                confidence=min(0.9, max_rise / 50.0),  # Higher rise = higher confidence
                detection_method="cgm_spike"
            )
            
            self.detected_meals.append(detected_meal)
            print(f"🔍 MEAL DETECTED: CGM spike {max_rise:.1f} mg/dL → {estimated_carbs:.1f}g {meal_type}")
            return detected_meal
        
        return None
    
    def inject_random_meal(self, current_time: int) -> Optional[MealPattern]:
        """Inject random meal for training data generation."""
        if not self.random_injection_enabled:
            return None
        
        # Check if enough time has passed since last meal
        if current_time - self.last_meal_time < self.min_meal_interval_minutes:
            return None
        
        # Random chance to inject meal (per 5-minute step)
        step_probability = self.injection_probability_per_hour / 12.0  # 12 steps per hour
        if random.random() > step_probability:
            return None
        
        # Determine meal type based on time of day with some randomness
        time_of_day = current_time % 1440
        meal_type = self._determine_meal_type_by_time(time_of_day, random_factor=0.3)
        
        # Generate meal characteristics
        meal_config = self.meal_types[meal_type]
        carbs = random.uniform(*meal_config["carbs_range"])
        gi_factor = random.uniform(*meal_config["gi_range"])
        
        injected_meal = MealPattern(
            time_minutes=current_time,
            carbs_grams=carbs,
            gi_factor=gi_factor,
            meal_type=f"random_{meal_type}",
            confidence=1.0,
            detection_method="random_injection"
        )
        
        self.injected_meals.append(injected_meal)
        self.last_meal_time = current_time
        
        print(f"🎲 RANDOM MEAL INJECTED: {carbs:.1f}g {meal_type} (GI: {gi_factor:.2f})")
        return injected_meal
    
    def _determine_meal_type_by_time(self, time_of_day: int, random_factor: float = 0.0) -> str:
        """Determine most likely meal type based on time of day."""
        # Calculate probabilities for each meal type
        probabilities = {}
        
        for meal_type, config in self.meal_types.items():
            if meal_type == "snack":
                probabilities[meal_type] = 0.2  # Base snack probability
                continue
            
            time_range = config["time_range"]
            if time_range[0] <= time_of_day <= time_range[1]:
                # Peak probability in the middle of time range
                range_center = (time_range[0] + time_range[1]) / 2
                distance_from_center = abs(time_of_day - range_center)
                range_width = time_range[1] - time_range[0]
                probability = max(0.1, 1.0 - (distance_from_center / (range_width / 2)))
            else:
                probability = 0.05  # Small chance outside normal time
            
            probabilities[meal_type] = probability
        
        # Add randomness if requested
        if random_factor > 0:
            for meal_type in probabilities:
                probabilities[meal_type] += random.uniform(-random_factor, random_factor)
                probabilities[meal_type] = max(0.01, probabilities[meal_type])
        
        # Select meal type based on probabilities
        total_prob = sum(probabilities.values())
        rand_val = random.uniform(0, total_prob)
        
        cumulative = 0
        for meal_type, prob in probabilities.items():
            cumulative += prob
            if rand_val <= cumulative:
                return meal_type
        
        return "snack"  # Fallback
    
    def process_simulation_step(self, current_time: int, current_cgm: float) -> Optional[Dict[str, Any]]:
        """
        Main processing function called each simulation step.
        
        Returns meal details if a meal should be delivered, None otherwise.
        """
        # Update CGM history
        self.update_cgm_history(current_time, current_cgm)
        
        # Try to detect meal from CGM spike
        detected_meal = self.detect_cgm_spike(current_time, current_cgm)
        
        # Try to inject random meal for training
        injected_meal = self.inject_random_meal(current_time)
        
        # Return meal details if any meal was detected/injected
        meal = detected_meal or injected_meal
        if meal:
            return {
                "grams": meal.carbs_grams,
                "gi_factor": meal.gi_factor,
                "type": meal.meal_type,
                "confidence": meal.confidence,
                "detection_method": meal.detection_method,
                "time_minutes": meal.time_minutes
            }
        
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about meal detection and injection."""
        return {
            "total_detected_meals": len(self.detected_meals),
            "total_injected_meals": len(self.injected_meals),
            "detection_methods": {
                "cgm_spike": len([m for m in self.detected_meals if m.detection_method == "cgm_spike"]),
                "random_injection": len([m for m in self.injected_meals if m.detection_method == "random_injection"]),
            },
            "meal_types": {
                meal_type: len([m for m in self.detected_meals + self.injected_meals if meal_type in m.meal_type])
                for meal_type in ["breakfast", "lunch", "dinner", "snack"]
            }
        }
