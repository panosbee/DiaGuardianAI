#!/usr/bin/env python3
"""
Smart Insulin Controller
Intelligent insulin delivery system that prevents lows and controls highs
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass
from .low_glucose_prevention_agent import LowGlucosePreventionAgent, InsulinAdjustment

@dataclass
class InsulinRecommendation:
    """Complete insulin delivery recommendation."""
    basal_rate: float  # U/hr
    bolus_amount: float  # U
    explanation: str
    safety_level: str  # 'safe', 'caution', 'warning', 'critical'
    predicted_glucose_30min: float
    confidence: float

class SmartInsulinController:
    """
    Smart insulin controller that:
    1. Prevents hypoglycemia by adjusting insulin delivery
    2. Controls hyperglycemia with appropriate insulin
    3. Uses predictive algorithms for safety
    4. Never relies on rescue carbs for low treatment
    """
    
    def __init__(self, patient_profile):
        """
        Initialize the smart insulin controller.
        
        Args:
            patient_profile: Patient profile with ISF, CR, basal rates, etc.
        """
        self.patient_profile = patient_profile
        self.low_prevention_agent = LowGlucosePreventionAgent()
        
        # Controller parameters
        self.target_glucose = 105.0  # mg/dL - ONE IN A BILLION target
        self.target_range_low = 80.0
        self.target_range_high = 130.0
        
        # Insulin parameters from patient profile
        self.base_basal_rate = patient_profile.basal_rate_u_hr
        self.isf = patient_profile.isf  # mg/dL per unit
        self.cr = patient_profile.cr    # grams per unit
        
        # Safety parameters
        self.max_bolus_per_hour = 8.0  # Maximum bolus in any hour
        self.max_basal_multiplier = 2.0  # Maximum basal increase
        self.min_glucose_for_correction = 90.0
        
        # History tracking
        self.bolus_history = []  # Track recent boluses
        self.correction_history = []  # Track recent corrections
        
    def update_bolus_history(self, bolus_amount: float, timestamp_minutes: float):
        """Track bolus history for safety."""
        self.bolus_history.append((timestamp_minutes, bolus_amount))
        
        # Keep only last hour
        cutoff_time = timestamp_minutes - 60
        self.bolus_history = [(t, b) for t, b in self.bolus_history if t > cutoff_time]
    
    def get_recent_bolus_total(self) -> float:
        """Get total bolus in the last hour."""
        return sum(bolus for _, bolus in self.bolus_history)
    
    def calculate_correction_bolus(self, current_glucose: float, target_glucose: float,
                                 current_iob: float) -> float:
        """
        Calculate correction bolus for high glucose.
        
        Args:
            current_glucose: Current glucose reading
            target_glucose: Target glucose level
            current_iob: Current insulin on board
            
        Returns:
            Correction bolus amount (units)
        """
        if current_glucose <= target_glucose:
            return 0.0
        
        # Calculate raw correction
        glucose_excess = current_glucose - target_glucose
        raw_correction = glucose_excess / self.isf
        
        # Adjust for IOB (insulin already working)
        # Assume IOB will lower glucose by ISF amount
        iob_effect = current_iob * self.isf
        adjusted_excess = glucose_excess - (iob_effect * 0.7)  # 70% of IOB effect
        
        if adjusted_excess <= 0:
            return 0.0  # IOB will handle the correction
        
        correction_bolus = adjusted_excess / self.isf
        
        # Safety limits
        correction_bolus = max(0.0, min(correction_bolus, 5.0))  # Max 5U correction
        
        return correction_bolus
    
    def calculate_meal_bolus(self, carbs_grams: float, current_glucose: float) -> float:
        """
        Calculate meal bolus with glucose-dependent adjustments.
        
        Args:
            carbs_grams: Carbohydrates in grams
            current_glucose: Current glucose level
            
        Returns:
            Meal bolus amount (units)
        """
        if carbs_grams <= 0:
            return 0.0
        
        # Base meal bolus
        meal_bolus = carbs_grams / self.cr
        
        # Adjust based on current glucose
        if current_glucose < 80:
            # Reduce bolus if glucose is low
            meal_bolus *= 0.7
        elif current_glucose < 90:
            meal_bolus *= 0.85
        elif current_glucose > 150:
            # Increase bolus if glucose is high
            meal_bolus *= 1.15
        elif current_glucose > 130:
            meal_bolus *= 1.1
        
        return meal_bolus
    
    def calculate_dynamic_basal(self, current_glucose: float, glucose_trend: str,
                              current_iob: float) -> float:
        """
        Calculate dynamic basal rate based on glucose level and trend.
        
        Args:
            current_glucose: Current glucose reading
            glucose_trend: 'rising', 'falling', 'stable'
            current_iob: Current insulin on board
            
        Returns:
            Basal rate (U/hr)
        """
        base_rate = self.base_basal_rate
        
        # Adjust based on glucose level
        if current_glucose > 140:
            # Increase basal for high glucose
            multiplier = 1.0 + min(0.5, (current_glucose - 140) / 100)
        elif current_glucose < 90:
            # Decrease basal for low glucose
            multiplier = max(0.3, (current_glucose - 60) / 30)
        else:
            multiplier = 1.0
        
        # Adjust based on trend
        if glucose_trend == 'rising' and current_glucose > 120:
            multiplier *= 1.2  # Increase for rising glucose
        elif glucose_trend == 'falling' and current_glucose < 100:
            multiplier *= 0.7  # Decrease for falling glucose
        
        # Safety limits
        multiplier = max(0.0, min(multiplier, self.max_basal_multiplier))
        
        return base_rate * multiplier
    
    def get_insulin_recommendation(self, patient_state: Dict[str, Any],
                                 meal_carbs: float = 0.0,
                                 timestamp_minutes: float = 0.0) -> InsulinRecommendation:
        """
        Get complete insulin delivery recommendation.
        
        Args:
            patient_state: Current patient state
            meal_carbs: Carbohydrates being consumed (grams)
            timestamp_minutes: Current timestamp
            
        Returns:
            InsulinRecommendation with safe insulin delivery
        """
        current_glucose = patient_state.get('cgm', 100)
        current_iob = patient_state.get('iob', 0)
        current_cob = patient_state.get('cob', 0)
        
        # Get glucose prediction and trend
        prediction = self.low_prevention_agent.predict_glucose(
            current_glucose, current_iob, current_cob
        )
        
        # Calculate proposed insulin delivery
        
        # 1. Calculate meal bolus
        meal_bolus = self.calculate_meal_bolus(meal_carbs, current_glucose)
        
        # 2. Calculate correction bolus
        correction_bolus = self.calculate_correction_bolus(
            current_glucose, self.target_glucose, current_iob
        )
        
        # 3. Total proposed bolus
        total_proposed_bolus = meal_bolus + correction_bolus
        
        # 4. Calculate dynamic basal
        proposed_basal = self.calculate_dynamic_basal(
            current_glucose, prediction.trend, current_iob
        )
        
        # 5. Apply safety checks through low prevention agent
        safe_basal, safe_bolus, safety_explanation = self.low_prevention_agent.get_safe_insulin_delivery(
            patient_state, proposed_basal, total_proposed_bolus, timestamp_minutes
        )
        
        # 6. Additional safety checks
        recent_bolus_total = self.get_recent_bolus_total()
        if recent_bolus_total + safe_bolus > self.max_bolus_per_hour:
            safe_bolus = max(0, self.max_bolus_per_hour - recent_bolus_total)
            safety_explanation += f" | Limited by max hourly bolus ({self.max_bolus_per_hour}U)"
        
        # Update history
        if safe_bolus > 0:
            self.update_bolus_history(safe_bolus, timestamp_minutes)
        
        # Determine safety level
        if current_glucose < 70 or prediction.predicted_glucose < 70:
            safety_level = 'critical'
        elif current_glucose < 80 or prediction.predicted_glucose < 80:
            safety_level = 'warning'
        elif current_glucose > 200 or prediction.predicted_glucose > 200:
            safety_level = 'caution'
        else:
            safety_level = 'safe'
        
        # Create detailed explanation
        explanation_parts = []
        
        if meal_carbs > 0:
            explanation_parts.append(f"Meal: {meal_carbs}g carbs")
        
        if correction_bolus > 0.1:
            explanation_parts.append(f"Correction: {correction_bolus:.1f}U for {current_glucose:.0f} mg/dL")
        
        if abs(proposed_basal - self.base_basal_rate) > 0.1:
            basal_change = ((proposed_basal / self.base_basal_rate) - 1) * 100
            explanation_parts.append(f"Basal: {basal_change:+.0f}% ({proposed_basal:.2f} U/hr)")
        
        explanation_parts.append(safety_explanation)
        
        full_explanation = " | ".join(explanation_parts)
        
        return InsulinRecommendation(
            basal_rate=safe_basal,
            bolus_amount=safe_bolus,
            explanation=full_explanation,
            safety_level=safety_level,
            predicted_glucose_30min=prediction.predicted_glucose,
            confidence=prediction.confidence
        )

# Example usage
if __name__ == "__main__":
    from DiaGuardianAI.data_generation.human_model_factory import HumanModelFactory
    
    print("🧠 SMART INSULIN CONTROLLER TEST")
    print("=" * 50)
    
    # Create a test patient
    factory = HumanModelFactory()
    patients = factory.generate_population(size=1, type_1_ratio=1.0)
    patient_profile = patients[0]
    
    # Create controller
    controller = SmartInsulinController(patient_profile)
    
    print(f"Patient: ISF={patient_profile.isf:.0f}, CR={patient_profile.cr:.0f}, Basal={patient_profile.basal_rate_u_hr:.2f}")
    
    # Test scenarios
    test_scenarios = [
        {"glucose": 85, "iob": 1.0, "cob": 0, "meal": 0, "scenario": "Low glucose"},
        {"glucose": 160, "iob": 0.2, "cob": 0, "meal": 0, "scenario": "High glucose"},
        {"glucose": 110, "iob": 0.5, "cob": 0, "meal": 45, "scenario": "Normal with meal"},
        {"glucose": 75, "iob": 2.0, "cob": 0, "meal": 30, "scenario": "Low with high IOB + meal"},
    ]
    
    for i, scenario in enumerate(test_scenarios):
        print(f"\n--- Scenario {i+1}: {scenario['scenario']} ---")
        
        patient_state = {
            'cgm': scenario['glucose'],
            'iob': scenario['iob'],
            'cob': scenario['cob']
        }
        
        recommendation = controller.get_insulin_recommendation(
            patient_state, scenario['meal'], i * 5
        )
        
        print(f"Current: {scenario['glucose']} mg/dL, IOB: {scenario['iob']}U")
        if scenario['meal'] > 0:
            print(f"Meal: {scenario['meal']}g carbs")
        
        print(f"Recommendation:")
        print(f"  Basal: {recommendation.basal_rate:.2f} U/hr")
        print(f"  Bolus: {recommendation.bolus_amount:.1f} U")
        print(f"  Predicted 30min: {recommendation.predicted_glucose_30min:.0f} mg/dL")
        print(f"  Safety: {recommendation.safety_level.upper()}")
        print(f"  Explanation: {recommendation.explanation}")
