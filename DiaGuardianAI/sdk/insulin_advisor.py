"""
DiaGuardianAI SDK - Insulin Advisor
Intelligent insulin delivery recommendations
"""

from typing import Optional, Dict, Any
from datetime import datetime
import logging

from .data_types import (
    PatientProfile, GlucoseReading, InsulinRecommendation, 
    SafetyLevel, SafetyAlert
)

class InsulinAdvisor:
    """
    Professional insulin recommendation system.
    
    Provides intelligent insulin delivery recommendations based on:
    - Current glucose levels
    - Meal carbohydrates
    - Insulin on board (IOB)
    - Patient-specific parameters
    - Safety constraints
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the insulin advisor."""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Default safety thresholds
        self.safety_thresholds = self.config.get('safety_thresholds', {
            'critical_low': 70,
            'warning_low': 80,
            'warning_high': 180,
            'critical_high': 250
        })
        
        # Insulin limits
        self.insulin_limits = self.config.get('insulin_limits', {
            'max_bolus_per_hour': 10.0,
            'max_basal_multiplier': 3.0,
            'min_glucose_for_bolus': 85
        })
    
    def get_recommendation(
        self,
        profile: PatientProfile,
        glucose_reading: GlucoseReading,
        meal_carbs: float = 0,
        current_iob: float = 0,
        safety_alert: Optional[SafetyAlert] = None
    ) -> InsulinRecommendation:
        """
        Get intelligent insulin delivery recommendation.
        
        Args:
            profile: Patient profile with insulin parameters
            glucose_reading: Current glucose measurement
            meal_carbs: Carbohydrates being consumed (grams)
            current_iob: Current insulin on board (units)
            safety_alert: Optional safety alert context
            
        Returns:
            InsulinRecommendation: Complete insulin delivery recommendation
        """
        current_glucose = glucose_reading.glucose_mg_dl
        timestamp = glucose_reading.timestamp
        
        # Calculate meal bolus
        meal_bolus = 0.0
        if meal_carbs > 0:
            meal_bolus = meal_carbs / profile.cr
        
        # Calculate correction bolus
        correction_bolus = 0.0
        target_glucose = getattr(profile, 'target_glucose', 105.0)
        
        if current_glucose > target_glucose:
            glucose_excess = current_glucose - target_glucose
            correction_bolus = glucose_excess / profile.isf
        
        # Total bolus calculation
        total_bolus = meal_bolus + correction_bolus
        
        # IOB adjustment
        if current_iob > 0:
            # Reduce bolus based on active insulin
            iob_reduction = min(current_iob * 0.5, total_bolus * 0.7)
            total_bolus = max(0, total_bolus - iob_reduction)
        
        # Basal rate calculation
        basal_rate = profile.basal_rate_u_hr
        
        # Safety adjustments
        safety_level = SafetyLevel.SAFE
        explanation = "Normal insulin delivery"
        
        # Check for low glucose
        if current_glucose < self.safety_thresholds['warning_low']:
            if current_glucose < self.safety_thresholds['critical_low']:
                # Critical low - suspend all insulin
                total_bolus = 0.0
                basal_rate = 0.0
                safety_level = SafetyLevel.CRITICAL
                explanation = "CRITICAL LOW: All insulin suspended"
            else:
                # Warning low - reduce insulin significantly
                total_bolus *= 0.3  # Reduce bolus by 70%
                basal_rate *= 0.5   # Reduce basal by 50%
                safety_level = SafetyLevel.WARNING
                explanation = "LOW GLUCOSE: Reduced insulin delivery"
        
        # Check for high glucose with IOB
        elif current_glucose > self.safety_thresholds['warning_high']:
            if current_iob > 2.0:
                # High glucose but significant IOB - be cautious
                total_bolus *= 0.8
                safety_level = SafetyLevel.CAUTION
                explanation = "HIGH GLUCOSE with IOB: Cautious insulin delivery"
            elif current_glucose > self.safety_thresholds['critical_high']:
                # Critical high - increase insulin
                basal_rate *= 1.5
                safety_level = SafetyLevel.WARNING
                explanation = "CRITICAL HIGH: Increased insulin delivery"
        
        # Apply insulin limits
        total_bolus = min(total_bolus, self.insulin_limits['max_bolus_per_hour'])
        basal_rate = min(basal_rate, profile.basal_rate_u_hr * self.insulin_limits['max_basal_multiplier'])
        
        # Don't bolus if glucose too low
        if current_glucose < self.insulin_limits['min_glucose_for_bolus']:
            total_bolus = 0.0
            if safety_level == SafetyLevel.SAFE:
                safety_level = SafetyLevel.CAUTION
                explanation = "LOW GLUCOSE: Bolus suspended"
        
        # Calculate confidence based on data quality
        confidence = 1.0
        if hasattr(glucose_reading, 'confidence') and glucose_reading.confidence:
            confidence *= glucose_reading.confidence
        
        # Reduce confidence if safety alert present
        if safety_alert and safety_alert.level in [SafetyLevel.WARNING, SafetyLevel.CRITICAL]:
            confidence *= 0.8
        
        # Create recommendation
        recommendation = InsulinRecommendation(
            patient_id=profile.patient_id,
            timestamp=timestamp,
            basal_rate_u_hr=round(basal_rate, 2),
            bolus_amount_u=round(total_bolus, 1),
            meal_bolus_u=round(meal_bolus, 1),
            correction_bolus_u=round(correction_bolus, 1),
            safety_level=safety_level,
            explanation=explanation,
            confidence=confidence,
            current_glucose=current_glucose,
            current_iob=current_iob,
            meal_carbs=meal_carbs if meal_carbs > 0 else None
        )
        
        # Add predictions (simplified)
        recommendation.predicted_glucose_30min = self._predict_glucose_30min(
            current_glucose, total_bolus, meal_carbs, profile
        )
        
        self.logger.info(
            f"Insulin recommendation: Bolus {total_bolus:.1f}U, "
            f"Basal {basal_rate:.2f}U/hr, Safety: {safety_level.value}"
        )
        
        return recommendation
    
    def _predict_glucose_30min(
        self,
        current_glucose: float,
        bolus_amount: float,
        meal_carbs: float,
        profile: PatientProfile
    ) -> float:
        """
        Simple 30-minute glucose prediction.
        
        Args:
            current_glucose: Current glucose level
            bolus_amount: Bolus insulin amount
            meal_carbs: Meal carbohydrates
            profile: Patient profile
            
        Returns:
            float: Predicted glucose in 30 minutes
        """
        # Simplified prediction model
        predicted_glucose = current_glucose
        
        # Effect of meal carbs (peak around 30 minutes)
        if meal_carbs > 0:
            carb_effect = meal_carbs * 3.0  # Rough estimate: 3 mg/dL per gram
            predicted_glucose += carb_effect * 0.7  # 70% effect at 30 min
        
        # Effect of insulin (starting to act at 30 minutes)
        if bolus_amount > 0:
            insulin_effect = bolus_amount * profile.isf
            predicted_glucose -= insulin_effect * 0.3  # 30% effect at 30 min
        
        # Natural glucose trend (simplified)
        if current_glucose > 120:
            predicted_glucose -= 5  # Slight downward trend
        elif current_glucose < 100:
            predicted_glucose += 3  # Slight upward trend
        
        return round(predicted_glucose, 1)
    
    def validate_recommendation(
        self,
        recommendation: InsulinRecommendation,
        profile: PatientProfile
    ) -> bool:
        """
        Validate insulin recommendation for safety.
        
        Args:
            recommendation: Insulin recommendation to validate
            profile: Patient profile
            
        Returns:
            bool: True if recommendation is safe
        """
        # Check bolus limits
        if recommendation.bolus_amount_u > self.insulin_limits['max_bolus_per_hour']:
            return False
        
        # Check basal limits
        max_basal = profile.basal_rate_u_hr * self.insulin_limits['max_basal_multiplier']
        if recommendation.basal_rate_u_hr > max_basal:
            return False
        
        # Check glucose constraints
        if (recommendation.current_glucose and 
            recommendation.current_glucose < self.insulin_limits['min_glucose_for_bolus'] and
            recommendation.bolus_amount_u > 0):
            return False
        
        return True
