"""
DiaGuardianAI SDK - Patient Manager
Professional patient lifecycle management
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import uuid
import logging

from .data_types import (
    PatientProfile, GlucoseReading, MealEvent, PerformanceMetrics
)

class PatientManager:
    """
    Professional patient management system for DiaGuardianAI.
    
    Handles patient profiles, glucose readings, meal events, and
    performance analytics with clinical-grade data management.
    """
    
    def __init__(self):
        """Initialize the patient management system."""
        self.patients: Dict[str, PatientProfile] = {}
        self.glucose_readings: Dict[str, List[GlucoseReading]] = {}
        self.meal_events: Dict[str, List[MealEvent]] = {}
        self.logger = logging.getLogger(__name__)
    
    def add_patient(self, profile: PatientProfile) -> str:
        """
        Add a new patient to the management system.
        
        Args:
            profile: Complete patient profile
            
        Returns:
            str: Patient ID
        """
        patient_id = profile.patient_id
        
        if patient_id in self.patients:
            raise ValueError(f"Patient {patient_id} already exists")
        
        self.patients[patient_id] = profile
        self.glucose_readings[patient_id] = []
        self.meal_events[patient_id] = []
        
        self.logger.info(f"Added patient {patient_id}")
        return patient_id
    
    def get_patient(self, patient_id: str) -> Optional[PatientProfile]:
        """Get patient profile by ID."""
        return self.patients.get(patient_id)
    
    def update_patient(self, patient_id: str, **updates) -> bool:
        """
        Update patient profile parameters.
        
        Args:
            patient_id: Patient identifier
            **updates: Fields to update
            
        Returns:
            bool: Success status
        """
        if patient_id not in self.patients:
            return False
        
        profile = self.patients[patient_id]
        
        for field, value in updates.items():
            if hasattr(profile, field):
                setattr(profile, field, value)
        
        profile.updated_at = datetime.now()
        
        self.logger.info(f"Updated patient {patient_id}")
        return True
    
    def add_glucose_reading(self, reading: GlucoseReading) -> str:
        """
        Add a glucose reading for a patient.
        
        Args:
            reading: Glucose reading data
            
        Returns:
            str: Reading ID
        """
        patient_id = reading.patient_id
        
        if patient_id not in self.patients:
            raise ValueError(f"Patient {patient_id} not found")
        
        self.glucose_readings[patient_id].append(reading)
        
        # Keep only last 30 days of readings for performance
        cutoff_date = datetime.now() - timedelta(days=30)
        self.glucose_readings[patient_id] = [
            r for r in self.glucose_readings[patient_id]
            if r.timestamp >= cutoff_date
        ]
        
        reading_id = str(uuid.uuid4())
        return reading_id
    
    def add_meal_event(self, meal: MealEvent) -> str:
        """
        Add a meal event for a patient.
        
        Args:
            meal: Meal event data
            
        Returns:
            str: Meal event ID
        """
        patient_id = meal.patient_id
        
        if patient_id not in self.patients:
            raise ValueError(f"Patient {patient_id} not found")
        
        self.meal_events[patient_id].append(meal)
        
        # Keep only last 30 days of meals
        cutoff_date = datetime.now() - timedelta(days=30)
        self.meal_events[patient_id] = [
            m for m in self.meal_events[patient_id]
            if m.timestamp >= cutoff_date
        ]
        
        meal_id = str(uuid.uuid4())
        return meal_id
    
    def get_glucose_readings(
        self,
        patient_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[GlucoseReading]:
        """
        Get glucose readings for a patient within a time range.
        
        Args:
            patient_id: Patient identifier
            start_time: Optional start time filter
            end_time: Optional end time filter
            
        Returns:
            List[GlucoseReading]: Filtered glucose readings
        """
        if patient_id not in self.glucose_readings:
            return []
        
        readings = self.glucose_readings[patient_id]
        
        if start_time:
            readings = [r for r in readings if r.timestamp >= start_time]
        
        if end_time:
            readings = [r for r in readings if r.timestamp <= end_time]
        
        return sorted(readings, key=lambda r: r.timestamp)
    
    def get_meal_events(
        self,
        patient_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[MealEvent]:
        """
        Get meal events for a patient within a time range.
        
        Args:
            patient_id: Patient identifier
            start_time: Optional start time filter
            end_time: Optional end time filter
            
        Returns:
            List[MealEvent]: Filtered meal events
        """
        if patient_id not in self.meal_events:
            return []
        
        meals = self.meal_events[patient_id]
        
        if start_time:
            meals = [m for m in meals if m.timestamp >= start_time]
        
        if end_time:
            meals = [m for m in meals if m.timestamp <= end_time]
        
        return sorted(meals, key=lambda m: m.timestamp)
    
    def calculate_performance_metrics(
        self,
        patient_id: str,
        start_time: datetime,
        end_time: datetime
    ) -> PerformanceMetrics:
        """
        Calculate comprehensive performance metrics for a patient.
        
        Args:
            patient_id: Patient identifier
            start_time: Analysis start time
            end_time: Analysis end time
            
        Returns:
            PerformanceMetrics: Complete performance analysis
        """
        readings = self.get_glucose_readings(patient_id, start_time, end_time)
        meals = self.get_meal_events(patient_id, start_time, end_time)
        
        if not readings:
            raise ValueError(f"No glucose readings found for patient {patient_id}")
        
        # Extract glucose values
        glucose_values = [r.glucose_mg_dl for r in readings]
        
        # Calculate time in range metrics
        total_readings = len(glucose_values)
        
        tir_70_180 = sum(1 for g in glucose_values if 70 <= g <= 180) / total_readings * 100
        tir_80_130 = sum(1 for g in glucose_values if 80 <= g <= 130) / total_readings * 100
        
        time_below_70 = sum(1 for g in glucose_values if g < 70) / total_readings * 100
        time_below_54 = sum(1 for g in glucose_values if g < 54) / total_readings * 100
        time_above_180 = sum(1 for g in glucose_values if g > 180) / total_readings * 100
        time_above_250 = sum(1 for g in glucose_values if g > 250) / total_readings * 100
        
        # Calculate glucose statistics
        mean_glucose = sum(glucose_values) / len(glucose_values)
        glucose_std = (sum((g - mean_glucose) ** 2 for g in glucose_values) / len(glucose_values)) ** 0.5
        min_glucose = min(glucose_values)
        max_glucose = max(glucose_values)
        
        # Count events
        safety_events_count = sum(1 for r in readings if r.get_safety_level().value in ['warning', 'critical'])
        meal_events_count = len(meals)
        
        # Estimate insulin metrics (simplified)
        total_insulin_units = meal_events_count * 5.0  # Rough estimate
        basal_insulin_percent = 50.0  # Typical split
        bolus_insulin_percent = 50.0
        
        return PerformanceMetrics(
            patient_id=patient_id,
            period_start=start_time,
            period_end=end_time,
            tir_70_180_percent=tir_70_180,
            tir_80_130_percent=tir_80_130,
            mean_glucose_mg_dl=mean_glucose,
            glucose_std_mg_dl=glucose_std,
            min_glucose_mg_dl=min_glucose,
            max_glucose_mg_dl=max_glucose,
            time_below_70_percent=time_below_70,
            time_below_54_percent=time_below_54,
            time_above_180_percent=time_above_180,
            time_above_250_percent=time_above_250,
            total_insulin_units=total_insulin_units,
            basal_insulin_percent=basal_insulin_percent,
            bolus_insulin_percent=bolus_insulin_percent,
            safety_events_count=safety_events_count,
            meal_events_count=meal_events_count
        )
    
    def get_patient_list(self) -> List[str]:
        """Get list of all patient IDs."""
        return list(self.patients.keys())
    
    def get_patient_count(self) -> int:
        """Get total number of patients."""
        return len(self.patients)
    
    def remove_patient(self, patient_id: str) -> bool:
        """
        Remove a patient and all associated data.
        
        Args:
            patient_id: Patient identifier
            
        Returns:
            bool: Success status
        """
        if patient_id not in self.patients:
            return False
        
        del self.patients[patient_id]
        del self.glucose_readings[patient_id]
        del self.meal_events[patient_id]
        
        self.logger.info(f"Removed patient {patient_id}")
        return True
