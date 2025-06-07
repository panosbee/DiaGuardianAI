"""
DiaGuardianAI SDK Data Types
Professional data structures for diabetes management
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum

class DiabetesType(Enum):
    """Diabetes type classification."""
    TYPE_1 = "type_1"
    TYPE_2 = "type_2"
    GESTATIONAL = "gestational"
    MODY = "mody"

class SafetyLevel(Enum):
    """Safety alert levels."""
    SAFE = "safe"
    CAUTION = "caution"
    WARNING = "warning"
    CRITICAL = "critical"

class InsulinType(Enum):
    """Insulin delivery types."""
    BASAL = "basal"
    BOLUS = "bolus"
    CORRECTION = "correction"

@dataclass
class PatientProfile:
    """Complete patient profile for diabetes management."""
    patient_id: str
    diabetes_type: DiabetesType
    age: int
    weight_kg: float
    
    # Insulin parameters
    isf: float  # Insulin sensitivity factor (mg/dL per unit)
    cr: float   # Carbohydrate ratio (grams per unit)
    basal_rate_u_hr: float  # Basal insulin rate (units per hour)
    
    # Optional parameters
    target_glucose: Optional[float] = 105.0  # mg/dL
    active_insulin_time: Optional[int] = 180  # minutes
    
    # Medical history
    diagnosis_date: Optional[datetime] = None
    hba1c: Optional[float] = None
    
    # Metadata
    created_at: datetime = datetime.now()
    updated_at: datetime = datetime.now()

@dataclass
class GlucoseReading:
    """Glucose measurement data."""
    patient_id: str
    glucose_mg_dl: float
    timestamp: datetime
    source: str = "cgm"  # cgm, bgm, lab
    confidence: Optional[float] = None
    
    def is_in_target_range(self, low: float = 80, high: float = 130) -> bool:
        """Check if glucose is in target range."""
        return low <= self.glucose_mg_dl <= high
    
    def get_safety_level(self) -> SafetyLevel:
        """Determine safety level based on glucose value."""
        if self.glucose_mg_dl < 54:
            return SafetyLevel.CRITICAL
        elif self.glucose_mg_dl < 70:
            return SafetyLevel.WARNING
        elif self.glucose_mg_dl > 250:
            return SafetyLevel.CRITICAL
        elif self.glucose_mg_dl > 180:
            return SafetyLevel.WARNING
        elif self.glucose_mg_dl < 80 or self.glucose_mg_dl > 130:
            return SafetyLevel.CAUTION
        else:
            return SafetyLevel.SAFE

@dataclass
class MealEvent:
    """Meal consumption data."""
    patient_id: str
    carbs_grams: float
    timestamp: datetime
    
    # Optional meal details
    protein_grams: Optional[float] = None
    fat_grams: Optional[float] = None
    gi_factor: Optional[float] = 1.0  # Glycemic index factor
    meal_type: Optional[str] = None  # breakfast, lunch, dinner, snack
    
    # Metadata
    confidence: Optional[float] = None
    source: str = "user_input"  # user_input, photo_analysis, etc.

@dataclass
class InsulinRecommendation:
    """Insulin delivery recommendation."""
    patient_id: str
    timestamp: datetime
    
    # Insulin recommendations
    basal_rate_u_hr: float
    bolus_amount_u: float
    
    # Recommendation details
    meal_bolus_u: float = 0.0
    correction_bolus_u: float = 0.0
    
    # Safety information
    safety_level: SafetyLevel = SafetyLevel.SAFE
    explanation: str = ""
    confidence: float = 1.0
    
    # Predictions
    predicted_glucose_30min: Optional[float] = None
    predicted_glucose_60min: Optional[float] = None
    
    # Context
    current_glucose: Optional[float] = None
    current_iob: Optional[float] = None
    meal_carbs: Optional[float] = None

@dataclass
class SafetyAlert:
    """Safety monitoring alert."""
    patient_id: str
    alert_id: str
    timestamp: datetime
    
    # Alert details
    level: SafetyLevel
    title: str
    message: str
    
    # Context
    current_glucose: Optional[float] = None
    predicted_glucose: Optional[float] = None
    current_iob: Optional[float] = None
    
    # Actions taken
    actions_taken: List[str] = None
    requires_intervention: bool = False
    
    # Resolution
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.actions_taken is None:
            self.actions_taken = []

@dataclass
class SystemStatus:
    """Overall system status."""
    timestamp: datetime
    
    # System health
    is_operational: bool = True
    last_update: Optional[datetime] = None
    
    # Patient statistics
    total_patients: int = 0
    active_patients: int = 0
    
    # Performance metrics
    average_tir_80_130: Optional[float] = None
    safety_events_24h: int = 0
    
    # System metrics
    uptime_hours: Optional[float] = None
    api_response_time_ms: Optional[float] = None

@dataclass
class PerformanceMetrics:
    """Performance tracking metrics."""
    patient_id: str
    period_start: datetime
    period_end: datetime
    
    # Time in range metrics
    tir_70_180_percent: float
    tir_80_130_percent: float  # ONE IN A BILLION target
    
    # Glucose statistics
    mean_glucose_mg_dl: float
    glucose_std_mg_dl: float
    min_glucose_mg_dl: float
    max_glucose_mg_dl: float
    
    # Safety metrics
    time_below_70_percent: float
    time_below_54_percent: float
    time_above_180_percent: float
    time_above_250_percent: float
    
    # Insulin metrics
    total_insulin_units: float
    basal_insulin_percent: float
    bolus_insulin_percent: float
    
    # Events
    safety_events_count: int
    meal_events_count: int
    
    # Assessment
    one_in_billion_achieved: bool = False
    
    def __post_init__(self):
        """Calculate derived metrics."""
        self.one_in_billion_achieved = (
            self.tir_80_130_percent >= 90 and
            self.time_below_70_percent < 1 and
            self.time_above_250_percent < 1
        )
