#!/usr/bin/env python3
"""
Low Glucose Prevention Agent
Intelligent insulin delivery control to prevent hypoglycemia
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass

@dataclass
class GlucosePrediction:
    """Glucose prediction result."""
    predicted_glucose: float
    confidence: float
    time_horizon_minutes: int
    trend: str  # 'rising', 'falling', 'stable'

@dataclass
class InsulinAdjustment:
    """Insulin delivery adjustment recommendation."""
    basal_multiplier: float  # 0.0 = suspend, 1.0 = normal, 0.5 = 50% reduction
    bolus_multiplier: float  # 0.0 = skip, 1.0 = normal, 0.5 = 50% reduction
    reason: str
    urgency: str  # 'low', 'medium', 'high', 'critical'

class LowGlucosePreventionAgent:
    """
    Agent that prevents hypoglycemia by intelligently adjusting insulin delivery.
    
    This agent:
    1. Predicts glucose trends
    2. Adjusts basal insulin delivery
    3. Modifies bolus recommendations
    4. Prevents dangerous lows without using rescue carbs
    """
    
    def __init__(self):
        # Glucose thresholds (mg/dL)
        self.critical_low_threshold = 70.0
        self.low_threshold = 80.0
        self.target_low = 90.0
        self.target_high = 130.0
        
        # Prediction parameters
        self.prediction_horizon_minutes = 30
        self.trend_sensitivity = 2.0  # mg/dL/min for trend detection
        
        # Safety parameters
        self.max_iob_for_bolus = 2.0  # Maximum IOB to allow full bolus
        self.min_glucose_for_bolus = 85.0  # Minimum glucose for any bolus
        
        # History for trend calculation
        self.glucose_history = []
        self.max_history_length = 6  # 30 minutes of 5-minute readings
    
    def update_glucose_history(self, glucose: float, timestamp_minutes: float):
        """Update glucose history for trend analysis."""
        self.glucose_history.append((timestamp_minutes, glucose))
        
        # Keep only recent history
        if len(self.glucose_history) > self.max_history_length:
            self.glucose_history.pop(0)
    
    def calculate_glucose_trend(self) -> Tuple[float, str]:
        """
        Calculate glucose trend from recent history.
        
        Returns:
            Tuple of (trend_mg_dl_per_min, trend_description)
        """
        if len(self.glucose_history) < 3:
            return 0.0, 'stable'
        
        # Use linear regression on recent points
        times = np.array([point[0] for point in self.glucose_history])
        glucoses = np.array([point[1] for point in self.glucose_history])
        
        # Calculate slope (mg/dL per minute)
        if len(times) > 1:
            slope = np.polyfit(times, glucoses, 1)[0]
        else:
            slope = 0.0
        
        # Classify trend
        if slope > self.trend_sensitivity:
            trend_desc = 'rising'
        elif slope < -self.trend_sensitivity:
            trend_desc = 'falling'
        else:
            trend_desc = 'stable'
        
        return slope, trend_desc
    
    def predict_glucose(self, current_glucose: float, current_iob: float, 
                       current_cob: float) -> GlucosePrediction:
        """
        Predict glucose level in the near future.
        
        Args:
            current_glucose: Current glucose reading (mg/dL)
            current_iob: Current insulin on board (units)
            current_cob: Current carbs on board (grams)
            
        Returns:
            GlucosePrediction object
        """
        trend_slope, trend_desc = self.calculate_glucose_trend()
        
        # Simple prediction model
        # In a real system, this would be much more sophisticated
        
        # Base prediction from trend
        predicted_glucose = current_glucose + (trend_slope * self.prediction_horizon_minutes)
        
        # Adjust for IOB effect (insulin will lower glucose)
        # Assume 1 unit of IOB will lower glucose by ~40 mg/dL over time
        iob_effect = -current_iob * 40 * 0.3  # 30% of effect in prediction window
        predicted_glucose += iob_effect
        
        # Adjust for COB effect (carbs will raise glucose)
        # Assume 1g carbs raises glucose by ~3 mg/dL
        cob_effect = current_cob * 3 * 0.5  # 50% of effect in prediction window
        predicted_glucose += cob_effect
        
        # Calculate confidence based on trend stability
        if len(self.glucose_history) >= 4:
            recent_glucoses = [point[1] for point in self.glucose_history[-4:]]
            glucose_variability = np.std(recent_glucoses)
            confidence = max(0.3, 1.0 - (glucose_variability / 20.0))  # Lower confidence with high variability
        else:
            confidence = 0.5
        
        return GlucosePrediction(
            predicted_glucose=predicted_glucose,
            confidence=confidence,
            time_horizon_minutes=self.prediction_horizon_minutes,
            trend=trend_desc
        )
    
    def assess_low_risk(self, current_glucose: float, prediction: GlucosePrediction,
                       current_iob: float) -> str:
        """
        Assess the risk of hypoglycemia.
        
        Returns:
            Risk level: 'none', 'low', 'medium', 'high', 'critical'
        """
        # Current glucose risk
        if current_glucose < self.critical_low_threshold:
            return 'critical'
        elif current_glucose < self.low_threshold:
            return 'high'
        elif current_glucose < self.target_low:
            return 'medium'
        
        # Predicted glucose risk
        if prediction.predicted_glucose < self.critical_low_threshold:
            return 'critical'
        elif prediction.predicted_glucose < self.low_threshold:
            return 'high'
        elif prediction.predicted_glucose < self.target_low:
            return 'medium'
        
        # IOB risk (high IOB with falling glucose)
        if current_iob > 1.5 and prediction.trend == 'falling':
            return 'medium'
        
        return 'low' if current_glucose < 100 else 'none'
    
    def calculate_insulin_adjustments(self, current_glucose: float, prediction: GlucosePrediction,
                                    current_iob: float, current_cob: float,
                                    proposed_bolus: float = 0.0) -> InsulinAdjustment:
        """
        Calculate insulin delivery adjustments to prevent lows.
        
        Args:
            current_glucose: Current glucose (mg/dL)
            prediction: Glucose prediction
            current_iob: Current IOB (units)
            current_cob: Current COB (grams)
            proposed_bolus: Proposed bolus amount (units)
            
        Returns:
            InsulinAdjustment recommendation
        """
        risk_level = self.assess_low_risk(current_glucose, prediction, current_iob)
        
        # Default: no adjustment
        basal_multiplier = 1.0
        bolus_multiplier = 1.0
        reason = "Normal insulin delivery"
        
        # Adjust based on risk level
        if risk_level == 'critical':
            # CRITICAL: Suspend all insulin
            basal_multiplier = 0.0
            bolus_multiplier = 0.0
            reason = f"CRITICAL LOW RISK: Current {current_glucose:.0f}, Predicted {prediction.predicted_glucose:.0f} mg/dL"
        
        elif risk_level == 'high':
            # HIGH: Suspend basal, reduce/skip bolus
            basal_multiplier = 0.0
            if current_glucose < 75:
                bolus_multiplier = 0.0
                reason = f"HIGH LOW RISK: Suspended all insulin. Current {current_glucose:.0f} mg/dL"
            else:
                bolus_multiplier = 0.3
                reason = f"HIGH LOW RISK: Suspended basal, reduced bolus 70%. Current {current_glucose:.0f} mg/dL"
        
        elif risk_level == 'medium':
            # MEDIUM: Reduce basal and bolus
            if prediction.trend == 'falling':
                basal_multiplier = 0.3
                bolus_multiplier = 0.5
                reason = f"MEDIUM LOW RISK: Falling trend. Reduced insulin 50-70%"
            else:
                basal_multiplier = 0.6
                bolus_multiplier = 0.7
                reason = f"MEDIUM LOW RISK: Reduced insulin 30-40%"
        
        elif risk_level == 'low':
            # LOW: Minor adjustments
            if current_iob > self.max_iob_for_bolus:
                bolus_multiplier = 0.8
                reason = f"LOW RISK: High IOB ({current_iob:.1f}U), reduced bolus 20%"
            elif prediction.trend == 'falling':
                basal_multiplier = 0.8
                reason = f"LOW RISK: Falling trend, reduced basal 20%"
        
        # Additional bolus safety checks
        if proposed_bolus > 0:
            # Never bolus if glucose too low
            if current_glucose < self.min_glucose_for_bolus:
                bolus_multiplier = 0.0
                reason = f"SAFETY: No bolus below {self.min_glucose_for_bolus} mg/dL"
            
            # Reduce bolus if high IOB
            elif current_iob > self.max_iob_for_bolus:
                iob_reduction = min(0.8, self.max_iob_for_bolus / current_iob)
                bolus_multiplier = min(bolus_multiplier, iob_reduction)
                reason = f"SAFETY: High IOB ({current_iob:.1f}U), reduced bolus"
        
        return InsulinAdjustment(
            basal_multiplier=basal_multiplier,
            bolus_multiplier=bolus_multiplier,
            reason=reason,
            urgency=risk_level
        )
    
    def get_safe_insulin_delivery(self, patient_state: Dict[str, Any], 
                                 proposed_basal: float, proposed_bolus: float,
                                 timestamp_minutes: float) -> Tuple[float, float, str]:
        """
        Get safe insulin delivery recommendations.
        
        Args:
            patient_state: Current patient state from SyntheticPatient
            proposed_basal: Proposed basal rate (U/hr)
            proposed_bolus: Proposed bolus amount (U)
            timestamp_minutes: Current time in minutes
            
        Returns:
            Tuple of (safe_basal, safe_bolus, explanation)
        """
        current_glucose = patient_state.get('cgm', 100)
        current_iob = patient_state.get('iob', 0)
        current_cob = patient_state.get('cob', 0)
        
        # Update glucose history
        self.update_glucose_history(current_glucose, timestamp_minutes)
        
        # Get prediction
        prediction = self.predict_glucose(current_glucose, current_iob, current_cob)
        
        # Calculate adjustments
        adjustment = self.calculate_insulin_adjustments(
            current_glucose, prediction, current_iob, current_cob, proposed_bolus
        )
        
        # Apply adjustments
        safe_basal = proposed_basal * adjustment.basal_multiplier
        safe_bolus = proposed_bolus * adjustment.bolus_multiplier
        
        # Create explanation
        explanation = f"{adjustment.reason}"
        if adjustment.urgency in ['high', 'critical']:
            explanation = f"🚨 {explanation}"
        elif adjustment.urgency == 'medium':
            explanation = f"⚠️ {explanation}"
        
        return safe_basal, safe_bolus, explanation

# Example usage and testing
if __name__ == "__main__":
    # Test the low glucose prevention agent
    agent = LowGlucosePreventionAgent()
    
    print("🛡️ LOW GLUCOSE PREVENTION AGENT TEST")
    print("=" * 50)
    
    # Simulate various scenarios
    test_scenarios = [
        {"glucose": 85, "iob": 1.5, "cob": 0, "scenario": "Mild low with IOB"},
        {"glucose": 70, "iob": 0.5, "cob": 0, "scenario": "Critical low"},
        {"glucose": 95, "iob": 3.0, "cob": 0, "scenario": "High IOB risk"},
        {"glucose": 110, "iob": 0.2, "cob": 20, "scenario": "Normal with COB"},
    ]
    
    for i, scenario in enumerate(test_scenarios):
        print(f"\nScenario {i+1}: {scenario['scenario']}")
        
        # Simulate patient state
        patient_state = {
            'cgm': scenario['glucose'],
            'iob': scenario['iob'],
            'cob': scenario['cob']
        }
        
        # Test with proposed insulin
        proposed_basal = 1.0  # U/hr
        proposed_bolus = 3.0  # U
        
        safe_basal, safe_bolus, explanation = agent.get_safe_insulin_delivery(
            patient_state, proposed_basal, proposed_bolus, i * 5
        )
        
        print(f"  Glucose: {scenario['glucose']} mg/dL, IOB: {scenario['iob']}U, COB: {scenario['cob']}g")
        print(f"  Proposed: Basal {proposed_basal:.1f} U/hr, Bolus {proposed_bolus:.1f}U")
        print(f"  Safe: Basal {safe_basal:.1f} U/hr, Bolus {safe_bolus:.1f}U")
        print(f"  {explanation}")
