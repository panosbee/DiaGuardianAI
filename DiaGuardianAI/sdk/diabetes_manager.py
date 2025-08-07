"""
DiaGuardianAI SDK - Main Diabetes Manager
Professional interface for diabetes management integration
"""

import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import logging

from .data_types import (
    PatientProfile, DiabetesType, GlucoseReading, MealEvent,
    InsulinRecommendation, SafetyAlert, SafetyLevel, PerformanceMetrics
)
from .patient_manager import PatientManager
from .insulin_advisor import InsulinAdvisor
from .safety_monitor import SafetyMonitor

class DiabetesManager:
    """
    Main DiaGuardianAI SDK interface for diabetes management.
    
    This class provides a high-level API for integrating the advanced
    AI-driven diabetes management capabilities into applications. It orchestrates
    patient management, safety monitoring, and the powerful multi-agent
    system for intelligent insulin recommendations.
    
    Example:
        dm = DiabetesManager()
        
        # Add a patient
        patient_id = dm.add_patient(
            diabetes_type="type_1",
            age=35,
            weight_kg=70,
            isf=50,
            cr=10,
            basal_rate=1.2
        )
        
        # Get an AI-driven insulin recommendation
        recommendation = dm.get_insulin_recommendation(
            patient_id=patient_id,
            current_glucose=120,
            meal_carbs=45
        )
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the DiaGuardianAI diabetes management system.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize core components
        self.patient_manager = PatientManager()
        # The InsulinAdvisor now contains the full AI decision-making system.
        self.insulin_advisor = InsulinAdvisor(config=self.config)
        self.safety_monitor = SafetyMonitor(config=self.config)
        
        # System state
        self.is_initialized = True
        self.start_time = datetime.now()
        
        self.logger.info("DiaGuardianAI SDK initialized successfully")
    
    def add_patient(
        self,
        diabetes_type: Union[str, DiabetesType],
        age: int,
        weight_kg: float,
        isf: float,
        cr: float,
        basal_rate: float,
        patient_id: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Add a new patient to the diabetes management system.
        
        Args:
            diabetes_type: Type of diabetes (type_1, type_2, etc.)
            age: Patient age in years
            weight_kg: Patient weight in kilograms
            isf: Insulin sensitivity factor (mg/dL per unit)
            cr: Carbohydrate ratio (grams per unit)
            basal_rate: Basal insulin rate (units per hour)
            patient_id: Optional custom patient ID
            **kwargs: Additional patient parameters
            
        Returns:
            str: Unique patient identifier
        """
        if patient_id is None:
            patient_id = str(uuid.uuid4())
        
        if isinstance(diabetes_type, str):
            diabetes_type = DiabetesType(diabetes_type)
        
        profile = PatientProfile(
            patient_id=patient_id,
            diabetes_type=diabetes_type,
            age=age,
            weight_kg=weight_kg,
            isf=isf,
            cr=cr,
            basal_rate_u_hr=basal_rate,
            **kwargs
        )
        
        self.patient_manager.add_patient(profile)
        self.safety_monitor.add_patient(patient_id)
        
        self.logger.info(f"Added patient {patient_id} ({diabetes_type.value})")
        return patient_id
    
    def get_insulin_recommendation(
        self,
        patient_id: str,
        current_glucose: float,
        meal_carbs: float = 0,
        current_iob: float = 0,
        timestamp: Optional[datetime] = None
    ) -> InsulinRecommendation:
        """
        Get an AI-driven insulin delivery recommendation.

        This method queries the underlying multi-agent system, which uses
        advanced glucose prediction and a continuous learning loop to determine
        the optimal insulin dose.
        
        Args:
            patient_id: Patient identifier
            current_glucose: Current glucose reading (mg/dL)
            meal_carbs: Carbohydrates being consumed (grams)
            current_iob: Current insulin on board (units)
            timestamp: Optional timestamp (defaults to now)
            
        Returns:
            InsulinRecommendation: The complete, AI-generated insulin recommendation.
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        profile = self.patient_manager.get_patient(patient_id)
        if not profile:
            raise ValueError(f"Patient {patient_id} not found")
        
        glucose_reading = GlucoseReading(
            patient_id=patient_id,
            glucose_mg_dl=current_glucose,
            timestamp=timestamp
        )
        
        safety_alert = self.safety_monitor.check_safety(
            patient_id, glucose_reading, current_iob
        )
        
        # Delegate the core decision to the AI-powered InsulinAdvisor
        recommendation = self.insulin_advisor.get_recommendation(
            profile=profile,
            glucose_reading=glucose_reading,
            meal_carbs=meal_carbs,
            current_iob=current_iob,
            safety_alert=safety_alert
        )
        
        self.logger.info(
            f"Insulin recommendation for {patient_id}: "
            f"Bolus {recommendation.bolus_amount_u:.1f}U, "
            f"Basal {recommendation.basal_rate_u_hr:.2f}U/hr"
        )
        
        return recommendation
    
    def add_glucose_reading(
        self,
        patient_id: str,
        glucose_mg_dl: float,
        timestamp: Optional[datetime] = None,
        source: str = "cgm"
    ) -> Optional[SafetyAlert]:
        """
        Add a glucose reading and check for safety alerts.
        
        Args:
            patient_id: Patient identifier
            glucose_mg_dl: Glucose value (mg/dL)
            timestamp: Optional timestamp (defaults to now)
            source: Source of reading (cgm, bgm, lab)
            
        Returns:
            Optional[SafetyAlert]: Safety alert if triggered
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        glucose_reading = GlucoseReading(
            patient_id=patient_id,
            glucose_mg_dl=glucose_mg_dl,
            timestamp=timestamp,
            source=source
        )
        
        self.patient_manager.add_glucose_reading(glucose_reading)
        safety_alert = self.safety_monitor.check_safety(patient_id, glucose_reading)
        
        return safety_alert
    
    def add_meal_event(
        self,
        patient_id: str,
        carbs_grams: float,
        timestamp: Optional[datetime] = None,
        **kwargs
    ) -> str:
        """
        Record a meal event.
        
        Args:
            patient_id: Patient identifier
            carbs_grams: Carbohydrates consumed (grams)
            timestamp: Optional timestamp (defaults to now)
            **kwargs: Additional meal parameters
            
        Returns:
            str: Meal event ID
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        meal_event = MealEvent(
            patient_id=patient_id,
            carbs_grams=carbs_grams,
            timestamp=timestamp,
            **kwargs
        )
        
        meal_id = self.patient_manager.add_meal_event(meal_event)
        
        self.logger.info(f"Recorded meal for {patient_id}: {carbs_grams}g carbs")
        return meal_id
    
    def get_performance_metrics(
        self,
        patient_id: str,
        days: int = 7
    ) -> PerformanceMetrics:
        """
        Get performance metrics for a patient.
        
        Args:
            patient_id: Patient identifier
            days: Number of days to analyze
            
        Returns:
            PerformanceMetrics: Comprehensive performance analysis
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        return self.patient_manager.calculate_performance_metrics(
            patient_id, start_time, end_time
        )
    
    def get_patient_list(self) -> List[str]:
        """Get list of all patient IDs."""
        return self.patient_manager.get_patient_list()
    
    def get_patient_profile(self, patient_id: str) -> Optional[PatientProfile]:
        """Get patient profile by ID."""
        return self.patient_manager.get_patient(patient_id)
    
    def get_safety_alerts(
        self,
        patient_id: Optional[str] = None,
        hours: int = 24
    ) -> List[SafetyAlert]:
        """
        Get recent safety alerts.
        
        Args:
            patient_id: Optional patient filter
            hours: Hours to look back
            
        Returns:
            List[SafetyAlert]: Recent safety alerts
        """
        return self.safety_monitor.get_recent_alerts(patient_id, hours)
    
    def get_system_status(self) -> Dict:
        """Get overall system status and health."""
        uptime = datetime.now() - self.start_time
        
        status = {
            "is_operational": self.is_initialized,
            "uptime_hours": uptime.total_seconds() / 3600,
            "total_patients": len(self.get_patient_list()),
            "sdk_version": "1.1.0", # Version bump to reflect significant changes
            "last_update": datetime.now().isoformat()
        }

        # Add AI system status if available
        if self.insulin_advisor and self.insulin_advisor.decision_system:
            status['ai_system_status'] = self.insulin_advisor.decision_system.get_complete_system_status()

        return status
