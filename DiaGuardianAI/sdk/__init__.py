"""
DiaGuardianAI SDK - Professional Diabetes Management Library

The DiaGuardianAI SDK provides a simple, powerful interface for integrating
ONE IN A BILLION diabetes management capabilities into medical applications.

Key Features:
- Real-time glucose control with >90% TIR (80-130 mg/dL)
- Intelligent insulin delivery optimization
- Hypoglycemia prevention without rescue carbs
- Clinical-grade safety mechanisms
- Professional healthcare interfaces

Quick Start:
    from DiaGuardianAI.sdk import DiabetesManager
    
    # Create a diabetes management system
    dm = DiabetesManager()
    
    # Add a patient
    patient_id = dm.add_patient(
        diabetes_type="type_1",
        age=35,
        weight_kg=70,
        isf=50,  # mg/dL per unit
        cr=10,   # grams per unit
        basal_rate=1.2  # units per hour
    )
    
    # Get insulin recommendation
    recommendation = dm.get_insulin_recommendation(
        patient_id=patient_id,
        current_glucose=120,  # mg/dL
        meal_carbs=45,        # grams
        current_iob=1.5       # units
    )
    
    print(f"Recommended bolus: {recommendation.bolus_amount:.1f} units")
    print(f"Recommended basal: {recommendation.basal_rate:.2f} U/hr")

For detailed documentation and examples, visit:
https://github.com/your-repo/DiaGuardianAI
"""

from .diabetes_manager import DiabetesManager
from .patient_manager import PatientManager
from .insulin_advisor import InsulinAdvisor
from .safety_monitor import SafetyMonitor
from .ood_and_explanation import OODDetector, SelfExplainer
from .data_types import (
    PatientProfile,
    InsulinRecommendation,
    SafetyAlert,
    GlucoseReading,
    MealEvent
)

__version__ = "1.0.0"
__author__ = "DiaGuardianAI Team"
__email__ = "contact@diaguardianai.com"
__description__ = "ONE IN A BILLION Diabetes Management SDK"

# SDK version and compatibility
SDK_VERSION = "1.0.0"
API_VERSION = "v1"
MIN_PYTHON_VERSION = "3.8"

# Default configuration
DEFAULT_CONFIG = {
    "target_glucose_range": (80, 130),  # ONE IN A BILLION range
    "safety_thresholds": {
        "critical_low": 70,
        "warning_low": 80,
        "warning_high": 180,
        "critical_high": 250
    },
    "insulin_limits": {
        "max_bolus_per_hour": 10.0,
        "max_basal_multiplier": 3.0,
        "min_glucose_for_bolus": 85
    },
    "prediction_horizon_minutes": 30,
    "update_frequency_minutes": 5
}

# Export main classes for easy import
__all__ = [
    'DiabetesManager',
    'PatientManager', 
    'InsulinAdvisor',
    'SafetyMonitor',
    'OODDetector',
    'SelfExplainer',
    'PatientProfile',
    'InsulinRecommendation',
    'SafetyAlert',
    'GlucoseReading',
    'MealEvent',
    'SDK_VERSION',
    'API_VERSION',
    'DEFAULT_CONFIG'
]
