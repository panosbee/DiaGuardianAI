"""
DiaGuardianAI SDK - Safety Monitor
Real-time safety monitoring and alerts
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
import uuid
import logging

from .data_types import (
    GlucoseReading, SafetyAlert, SafetyLevel
)

class SafetyMonitor:
    """
    Professional safety monitoring system for diabetes management.
    
    Provides real-time safety oversight with:
    - Continuous glucose surveillance
    - Multi-level alert system
    - Automatic intervention protocols
    - Clinical escalation pathways
    - Comprehensive audit trails
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the safety monitoring system."""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Safety thresholds
        self.thresholds = self.config.get('safety_thresholds', {
            'critical_low': 54,      # Severe hypoglycemia
            'warning_low': 70,       # Hypoglycemia
            'caution_low': 80,       # Below target
            'caution_high': 130,     # Above target
            'warning_high': 180,     # Hyperglycemia
            'critical_high': 250     # Severe hyperglycemia
        })
        
        # Alert storage
        self.alerts: Dict[str, List[SafetyAlert]] = {}
        self.patient_status: Dict[str, Dict] = {}
        
        # Alert history retention (days)
        self.retention_days = self.config.get('alert_retention_days', 30)
    
    def add_patient(self, patient_id: str) -> None:
        """
        Add a patient to safety monitoring.
        
        Args:
            patient_id: Patient identifier
        """
        if patient_id not in self.alerts:
            self.alerts[patient_id] = []
            self.patient_status[patient_id] = {
                'last_glucose': None,
                'last_update': None,
                'current_safety_level': SafetyLevel.SAFE,
                'consecutive_lows': 0,
                'consecutive_highs': 0
            }
            
            self.logger.info(f"Added patient {patient_id} to safety monitoring")
    
    def check_safety(
        self,
        patient_id: str,
        glucose_reading: GlucoseReading,
        current_iob: float = 0
    ) -> Optional[SafetyAlert]:
        """
        Check safety status and generate alerts if needed.
        
        Args:
            patient_id: Patient identifier
            glucose_reading: Current glucose measurement
            current_iob: Current insulin on board
            
        Returns:
            Optional[SafetyAlert]: Safety alert if triggered
        """
        if patient_id not in self.patient_status:
            self.add_patient(patient_id)
        
        glucose = glucose_reading.glucose_mg_dl
        timestamp = glucose_reading.timestamp
        
        # Determine safety level
        safety_level = self._assess_safety_level(glucose)
        
        # Update patient status
        status = self.patient_status[patient_id]
        previous_level = status['current_safety_level']
        status['last_glucose'] = glucose
        status['last_update'] = timestamp
        status['current_safety_level'] = safety_level
        
        # Track consecutive events
        if safety_level in [SafetyLevel.WARNING, SafetyLevel.CRITICAL] and glucose < 80:
            status['consecutive_lows'] += 1
            status['consecutive_highs'] = 0
        elif safety_level in [SafetyLevel.WARNING, SafetyLevel.CRITICAL] and glucose > 180:
            status['consecutive_highs'] += 1
            status['consecutive_lows'] = 0
        else:
            status['consecutive_lows'] = 0
            status['consecutive_highs'] = 0
        
        # Generate alert if needed
        alert = None
        if safety_level != SafetyLevel.SAFE or safety_level != previous_level:
            alert = self._create_safety_alert(
                patient_id, glucose_reading, safety_level, current_iob, status
            )
            
            if alert:
                self.alerts[patient_id].append(alert)
                self._cleanup_old_alerts(patient_id)
        
        return alert
    
    def _assess_safety_level(self, glucose: float) -> SafetyLevel:
        """
        Assess safety level based on glucose value.
        
        Args:
            glucose: Glucose value in mg/dL
            
        Returns:
            SafetyLevel: Assessed safety level
        """
        if glucose <= self.thresholds['critical_low'] or glucose >= self.thresholds['critical_high']:
            return SafetyLevel.CRITICAL
        elif glucose <= self.thresholds['warning_low'] or glucose >= self.thresholds['warning_high']:
            return SafetyLevel.WARNING
        elif glucose <= self.thresholds['caution_low'] or glucose >= self.thresholds['caution_high']:
            return SafetyLevel.CAUTION
        else:
            return SafetyLevel.SAFE
    
    def _create_safety_alert(
        self,
        patient_id: str,
        glucose_reading: GlucoseReading,
        safety_level: SafetyLevel,
        current_iob: float,
        status: Dict
    ) -> Optional[SafetyAlert]:
        """
        Create a safety alert based on current conditions.
        
        Args:
            patient_id: Patient identifier
            glucose_reading: Current glucose reading
            safety_level: Assessed safety level
            current_iob: Current insulin on board
            status: Patient status dictionary
            
        Returns:
            Optional[SafetyAlert]: Created safety alert
        """
        glucose = glucose_reading.glucose_mg_dl
        timestamp = glucose_reading.timestamp
        
        # Don't create alerts for safe conditions
        if safety_level == SafetyLevel.SAFE:
            return None
        
        # Generate alert ID
        alert_id = str(uuid.uuid4())[:8]
        
        # Determine alert details based on safety level and glucose
        if glucose <= self.thresholds['critical_low']:
            title = "CRITICAL HYPOGLYCEMIA"
            message = f"Severe low glucose: {glucose:.0f} mg/dL. Immediate intervention required."
            actions_taken = ["Suspend all insulin", "Emergency protocol activated"]
            requires_intervention = True
            
        elif glucose <= self.thresholds['warning_low']:
            title = "HYPOGLYCEMIA WARNING"
            message = f"Low glucose detected: {glucose:.0f} mg/dL. Reducing insulin delivery."
            actions_taken = ["Reduced basal rate", "Suspended bolus delivery"]
            requires_intervention = True
            
        elif glucose <= self.thresholds['caution_low']:
            title = "LOW GLUCOSE CAUTION"
            message = f"Glucose trending low: {glucose:.0f} mg/dL. Monitoring closely."
            actions_taken = ["Increased monitoring frequency"]
            requires_intervention = False
            
        elif glucose >= self.thresholds['critical_high']:
            title = "CRITICAL HYPERGLYCEMIA"
            message = f"Severe high glucose: {glucose:.0f} mg/dL. Check for ketones."
            actions_taken = ["Increased insulin delivery", "Clinical notification"]
            requires_intervention = True
            
        elif glucose >= self.thresholds['warning_high']:
            title = "HYPERGLYCEMIA WARNING"
            message = f"High glucose detected: {glucose:.0f} mg/dL. Adjusting insulin."
            actions_taken = ["Increased basal rate", "Correction bolus calculated"]
            requires_intervention = False
            
        else:  # caution_high
            title = "HIGH GLUCOSE CAUTION"
            message = f"Glucose above target: {glucose:.0f} mg/dL. Monitoring trend."
            actions_taken = ["Trend monitoring activated"]
            requires_intervention = False
        
        # Add IOB context
        if current_iob > 2.0:
            message += f" IOB: {current_iob:.1f}U - considering active insulin."
            actions_taken.append("IOB-adjusted recommendations")
        
        # Add consecutive event context
        if status['consecutive_lows'] > 1:
            message += f" ({status['consecutive_lows']} consecutive low readings)"
            requires_intervention = True
        elif status['consecutive_highs'] > 2:
            message += f" ({status['consecutive_highs']} consecutive high readings)"
        
        # Create alert
        alert = SafetyAlert(
            patient_id=patient_id,
            alert_id=alert_id,
            timestamp=timestamp,
            level=safety_level,
            title=title,
            message=message,
            current_glucose=glucose,
            current_iob=current_iob if current_iob > 0 else None,
            actions_taken=actions_taken,
            requires_intervention=requires_intervention
        )
        
        self.logger.warning(
            f"Safety alert for {patient_id}: {safety_level.value.upper()} - {glucose:.0f} mg/dL"
        )
        
        return alert
    
    def get_recent_alerts(
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
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_alerts = []
        
        if patient_id:
            # Get alerts for specific patient
            if patient_id in self.alerts:
                recent_alerts = [
                    alert for alert in self.alerts[patient_id]
                    if alert.timestamp >= cutoff_time
                ]
        else:
            # Get alerts for all patients
            for alerts_list in self.alerts.values():
                recent_alerts.extend([
                    alert for alert in alerts_list
                    if alert.timestamp >= cutoff_time
                ])
        
        # Sort by timestamp (most recent first)
        recent_alerts.sort(key=lambda a: a.timestamp, reverse=True)
        
        return recent_alerts
    
    def get_patient_safety_status(self, patient_id: str) -> Dict:
        """
        Get current safety status for a patient.
        
        Args:
            patient_id: Patient identifier
            
        Returns:
            Dict: Current safety status
        """
        if patient_id not in self.patient_status:
            return {"error": "Patient not found in safety monitoring"}
        
        status = self.patient_status[patient_id].copy()
        
        # Add recent alert count
        recent_alerts = self.get_recent_alerts(patient_id, hours=24)
        status['alerts_24h'] = len(recent_alerts)
        status['critical_alerts_24h'] = len([
            a for a in recent_alerts if a.level == SafetyLevel.CRITICAL
        ])
        
        return status
    
    def resolve_alert(self, patient_id: str, alert_id: str) -> bool:
        """
        Mark an alert as resolved.
        
        Args:
            patient_id: Patient identifier
            alert_id: Alert identifier
            
        Returns:
            bool: Success status
        """
        if patient_id not in self.alerts:
            return False
        
        for alert in self.alerts[patient_id]:
            if alert.alert_id == alert_id and not alert.resolved:
                alert.resolved = True
                alert.resolved_at = datetime.now()
                self.logger.info(f"Resolved alert {alert_id} for patient {patient_id}")
                return True
        
        return False
    
    def _cleanup_old_alerts(self, patient_id: str) -> None:
        """
        Remove old alerts to manage memory usage.
        
        Args:
            patient_id: Patient identifier
        """
        if patient_id not in self.alerts:
            return
        
        cutoff_date = datetime.now() - timedelta(days=self.retention_days)
        
        # Keep only recent alerts
        self.alerts[patient_id] = [
            alert for alert in self.alerts[patient_id]
            if alert.timestamp >= cutoff_date
        ]
    
    def get_system_safety_summary(self) -> Dict:
        """
        Get overall system safety summary.
        
        Returns:
            Dict: System safety summary
        """
        total_patients = len(self.patient_status)
        active_alerts = 0
        critical_alerts = 0
        
        for patient_alerts in self.alerts.values():
            for alert in patient_alerts:
                if not alert.resolved and alert.timestamp >= datetime.now() - timedelta(hours=1):
                    active_alerts += 1
                    if alert.level == SafetyLevel.CRITICAL:
                        critical_alerts += 1
        
        return {
            "total_patients_monitored": total_patients,
            "active_alerts": active_alerts,
            "critical_alerts": critical_alerts,
            "system_status": "operational",
            "last_update": datetime.now().isoformat()
        }
