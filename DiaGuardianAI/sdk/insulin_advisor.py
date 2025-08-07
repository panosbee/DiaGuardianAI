"""
DiaGuardianAI SDK - Insulin Advisor
This module contains the core AI engine for DiaGuardianAI.
"""

from typing import Optional, Dict, Any, List
from datetime import datetime
import logging
import numpy as np
import pandas as pd

from .data_types import (
    PatientProfile, GlucoseReading, InsulinRecommendation,
    SafetyLevel, SafetyAlert
)

# --- AI Model and Data Imports ---
from DiaGuardianAI.models.transformer_zoo import TransformerZoo
from DiaGuardianAI.examples.research_data_generation import generate_research_data
from DiaGuardianAI.data_generation.data_formatter import DataFormatter
from DiaGuardianAI.agents.advanced_multi_agent_system import ContinuousLearningLoop
from DiaGuardianAI.pattern_repository.repository_manager import RepositoryManager

class InsulinAdvisor:
    """
    Hosts and orchestrates the DiaGuardianAI continuous learning system.

    This class is the heart of the AI engine. On initialization, it sets up
    a complete, self-contained AI ecosystem including:
    - A multi-model predictive zoo (`TransformerZoo`) for glucose forecasting.
    - A multi-agent system (`ContinuousLearningLoop`) for decision-making.
    - A pattern repository (`RepositoryManager`) for storing and learning from
      past decisions.
    
    It is responsible for training the initial models on synthetic data and
    then delegating all insulin recommendation requests to the learning loop.
    """
    
    def __init__(self, config: Optional[Dict] = None, train_on_init: bool = True):
        """
        Initializes the full AI decision-making system.

        Args:
            config (Optional[Dict]): Configuration dictionary for potential future use.
            train_on_init (bool): If True, orchestrates the entire AI system setup,
                                  including data generation and model training.
                                  Set to False only for specific testing scenarios.
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        self.zoo = None
        self.repository = None
        self.decision_system = None
        self.formatter = None
        # { patient_id: { 'timestamps': [], 'cgm': [], 'bolus': [], 'carbs': [] } }
        self.patient_history: Dict[str, Dict[str, List]] = {}

        if train_on_init:
            self._initialize_ai_system()

    def _initialize_ai_system(self):
        """
        Orchestrates the full setup of the AI system, including data generation,
        model training, and agent initialization.
        """
        self.logger.info("Initializing full AI system...")
        try:
            # --- Step 1: Train Prediction Models ---
            self.logger.info("Phase 1: Training predictive models...")
            raw_data_dfs = generate_research_data(num_patients=2, simulation_days_per_patient=5)
            if not raw_data_dfs:
                raise RuntimeError("AI Init Failed: Synthetic data generation returned no data.")

            combined_data = pd.concat(raw_data_dfs, ignore_index=True)

            self.formatter = DataFormatter(history_window_minutes=120, prediction_horizons_minutes=[10, 20, 30, 40, 50, 60, 90, 120])

            features, targets = self.formatter.create_dataset(
                cgm_data=combined_data['cgm_mg_dl'].tolist(),
                timestamps=pd.to_datetime(combined_data['timestamp']).tolist(),
                insulin_bolus_data=combined_data['bolus_u'].tolist(),
                carb_data=combined_data['carbs_g'].tolist()
            )

            if features.size == 0:
                raise RuntimeError(f"AI Init Failed: Data formatting produced no samples.")

            self.zoo = TransformerZoo(input_dim=features.shape[1])
            self.zoo.train_all_models(features, targets)

            if not self.zoo.models.get('transformer') or not self.zoo.models['transformer'].is_trained:
                raise RuntimeError("AI Init Failed: Transformer model training was unsuccessful.")

            self.logger.info("Phase 1 Complete: Predictive models trained successfully.")

            # --- Step 2: Initialize Pattern Repository and Learning Loop ---
            self.logger.info("Phase 2: Initializing continuous learning system...")
            self.repository = RepositoryManager(db_path="DiaGuardianAI_patterns.sqlite")
            self.decision_system = ContinuousLearningLoop(self.zoo, self.repository)
            self.decision_system.start_learning_loop()
            self.logger.info("Phase 2 Complete: Continuous learning system is active.")
            self.logger.info("AI System Initialized Successfully.")

        except Exception as e:
            self.logger.exception(f"A critical error occurred during AI system initialization: {e}")
            self.decision_system = None
            self.zoo = None

    def _update_patient_history(self, patient_id: str, glucose_reading: GlucoseReading, bolus: float, meal_carbs: float):
        """Updates the internal buffer of recent patient data."""
        if patient_id not in self.patient_history:
            self.patient_history[patient_id] = {'timestamps': [], 'cgm': [], 'bolus': [], 'carbs': []}
        
        history = self.patient_history[patient_id]
        history['timestamps'].append(glucose_reading.timestamp)
        history['cgm'].append(glucose_reading.glucose_mg_dl)
        history['bolus'].append(bolus)
        history['carbs'].append(meal_carbs)

        max_history_length = 288 # 24 hours of 5-min data
        for key in history:
            if len(history[key]) > max_history_length:
                history[key] = history[key][-max_history_length:]
    
    def _prepare_state_for_agent(self, profile: PatientProfile, glucose_reading: GlucoseReading, meal_carbs: float, current_iob: float) -> Dict[str, Any]:
        """Prepares the comprehensive state dictionary for the multi-agent system."""
        history = self.patient_history.get(profile.patient_id, {})

        cgm_trend = 0.0
        if len(history.get('cgm', [])) >= 4:
            cgm_trend = history['cgm'][-1] - history['cgm'][-4]

        time_since_meal = 180.0
        if meal_carbs > 0:
            time_since_meal = 0.0
        elif history.get('carbs'):
            try:
                last_meal_idx = max(loc for loc, val in enumerate(history['carbs']) if val > 0)
                time_since_meal = (len(history['carbs']) - 1 - last_meal_idx) * 5
            except ValueError:
                pass

        state = {
            "cgm": glucose_reading.glucose_mg_dl,
            "cgm_trend": cgm_trend,
            "iob": current_iob,
            "cob": meal_carbs,
            "time_of_day": glucose_reading.timestamp.hour,
            "time_since_meal": time_since_meal,
            "time_since_bolus": 0,
            "stress_level": 1.0,
            "exercise_recent": 0.0,
            "isf": profile.isf,
            "cr": profile.cr,
            "basal_rate": profile.basal_rate_u_hr,
            "glucose_variability": np.std(history.get('cgm', [glucose_reading.glucose_mg_dl])) if history.get('cgm') else 0,
            "day_of_week": glucose_reading.timestamp.weekday()
        }
        return state

    def get_recommendation(
        self,
        profile: PatientProfile,
        glucose_reading: GlucoseReading,
        meal_carbs: float = 0,
        current_iob: float = 0,
        safety_alert: Optional[SafetyAlert] = None
    ) -> InsulinRecommendation:
        """
        Gets an insulin recommendation by delegating to the continuous learning loop.
        
        This method constructs the current patient state and passes it to the
        multi-agent system for a holistic, AI-driven decision.

        Args:
            profile: The patient's profile.
            glucose_reading: The most recent glucose reading.
            meal_carbs: Any meal carbohydrates being consumed.
            current_iob: The current estimate for insulin on board.
            safety_alert: Any pre-existing safety alerts.

        Returns:
            An `InsulinRecommendation` object containing the AI-generated advice.
        """
        self._update_patient_history(profile.patient_id, glucose_reading, 0.0, meal_carbs)

        if not self.decision_system or not self.decision_system.loop_active:
            self.logger.error("Decision system is not active. Returning a safe, default recommendation.")
            return InsulinRecommendation(
                patient_id=profile.patient_id,
                timestamp=datetime.now(),
                basal_rate_u_hr=profile.basal_rate_u_hr * 0.5,
                bolus_amount_u=0.0,
                safety_level=SafetyLevel.CRITICAL,
                explanation="AI system is offline. Using safe fallback.",
                confidence=0.1
            )

        current_state = self._prepare_state_for_agent(profile, glucose_reading, meal_carbs, current_iob)
        agent_result = self.decision_system.process_patient_state(current_state)
        
        if "error" in agent_result:
            self.logger.error(f"Agent system returned an error: {agent_result['error']}. Returning safe fallback.")
            return InsulinRecommendation(
                patient_id=profile.patient_id,
                timestamp=datetime.now(),
                basal_rate_u_hr=profile.basal_rate_u_hr * 0.5,
                bolus_amount_u=0.0,
                safety_level=SafetyLevel.CRITICAL,
                explanation="AI system error. Using safe fallback.",
                confidence=0.1
            )

        decision = agent_result['insulin_decision']
        
        final_safety_level = SafetyLevel.SAFE
        if current_glucose < 70:
            final_safety_level = SafetyLevel.CRITICAL
        elif current_glucose < 80:
            final_safety_level = SafetyLevel.WARNING
        
        recommendation = InsulinRecommendation(
            patient_id=profile.patient_id,
            timestamp=datetime.now(),
            basal_rate_u_hr=round(decision['basal_rate_u_hr'], 2),
            bolus_amount_u=round(decision['bolus_u'], 1),
            safety_level=final_safety_level,
            explanation=decision['reasoning'],
            confidence=agent_result['system_performance'].get('average_confidence', 0.8),
            current_glucose=current_glucose,
            current_iob=current_iob,
            meal_carbs=meal_carbs if meal_carbs > 0 else None,
            predicted_glucose_horizons={decision['chosen_horizon']: decision['predicted_glucose']}
        )
        
        self.logger.info(
            f"Agent-driven recommendation: Bolus {recommendation.bolus_amount_u:.1f}U, "
            f"Basal {recommendation.basal_rate_u_hr:.2f}U/hr. "
            f"Reason: {recommendation.explanation}"
        )
        
        return recommendation
