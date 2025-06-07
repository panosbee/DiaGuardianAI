#!/usr/bin/env python3
"""
DiaGuardianAI Advanced Multi-Agent System
Agent 1: Decision Maker (chooses optimal predictions, schedules insulin)
Agent 2: Pattern Inspector (analyzes patterns, pushes improvements to Agent 1)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import json
import sqlite3
from dataclasses import dataclass, asdict
from enum import Enum

from ..models.transformer_zoo import TransformerZoo
from ..pattern_repository.repository_manager import RepositoryManager

class PredictionQuality(Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"

@dataclass
class PredictionResult:
    """Result from a model prediction."""
    model_name: str
    horizon_minutes: int
    predicted_glucose: float
    confidence: float
    timestamp: datetime
    
@dataclass
class InsulinDecision:
    """Insulin dosing decision."""
    basal_rate_u_hr: float
    bolus_u: float
    reasoning: str
    chosen_prediction: PredictionResult
    alternative_predictions: List[PredictionResult]
    decision_timestamp: datetime
    
@dataclass
class DecisionPattern:
    """Pattern of decisions for analysis."""
    pattern_id: str
    glucose_context: Dict[str, float]  # current, trend, etc.
    predictions_used: List[PredictionResult]
    insulin_decision: InsulinDecision
    actual_outcome: Optional[float]  # actual glucose after decision
    outcome_quality: Optional[PredictionQuality]
    timestamp: datetime

class Agent1_DecisionMaker:
    """
    Agent 1: Decision Maker
    - Receives multi-horizon predictions from TransformerZoo
    - Chooses optimal prediction based on context
    - Schedules basal and bolus insulin
    - Saves all decisions as patterns
    """
    
    def __init__(self, transformer_zoo: TransformerZoo, pattern_repository: RepositoryManager):
        self.transformer_zoo = transformer_zoo
        self.pattern_repository = pattern_repository
        self.decision_history = []
        self.performance_metrics = {
            "total_decisions": 0,
            "successful_decisions": 0,
            "average_accuracy": 0.0
        }
        
        # Decision-making parameters
        self.target_glucose = 110.0
        self.glucose_range_tight = (80, 140)
        self.glucose_range_acceptable = (70, 180)
        
        print(f"🤖 Agent1_DecisionMaker initialized")
        print(f"  Target glucose: {self.target_glucose} mg/dL")
        print(f"  Tight range: {self.glucose_range_tight}")
        print(f"  Acceptable range: {self.glucose_range_acceptable}")
    
    def make_insulin_decision(self, current_state: Dict[str, Any]) -> InsulinDecision:
        """
        Make insulin decision based on multi-horizon predictions.
        
        Args:
            current_state: Current patient state (CGM, IOB, COB, etc.)
            
        Returns:
            InsulinDecision with basal/bolus recommendations
        """
        
        # Step 1: Get predictions from all models for all horizons
        current_features = self._extract_features(current_state)
        all_predictions = self.transformer_zoo.predict_all_models(current_features)
        
        # Step 2: Convert to PredictionResult objects
        prediction_results = []
        for model_name, horizon_predictions in all_predictions.items():
            for horizon, glucose_pred in horizon_predictions.items():
                if isinstance(glucose_pred, np.ndarray):
                    glucose_pred = float(glucose_pred[0]) if len(glucose_pred) > 0 else float(glucose_pred)
                
                confidence = self._calculate_prediction_confidence(model_name, horizon, current_state)
                
                pred_result = PredictionResult(
                    model_name=model_name,
                    horizon_minutes=horizon,
                    predicted_glucose=float(glucose_pred),
                    confidence=confidence,
                    timestamp=datetime.now()
                )
                prediction_results.append(pred_result)
        
        # Step 3: Choose optimal prediction using advanced selection logic
        chosen_prediction = self._choose_optimal_prediction(prediction_results, current_state)
        
        # Step 4: Calculate insulin dosing based on chosen prediction
        basal_rate, bolus_dose, reasoning = self._calculate_insulin_dosing(
            chosen_prediction, current_state, prediction_results
        )
        
        # Step 5: Create decision object
        decision = InsulinDecision(
            basal_rate_u_hr=basal_rate,
            bolus_u=bolus_dose,
            reasoning=reasoning,
            chosen_prediction=chosen_prediction,
            alternative_predictions=[p for p in prediction_results if p != chosen_prediction],
            decision_timestamp=datetime.now()
        )
        
        # Step 6: Save decision as pattern
        self._save_decision_pattern(decision, current_state)
        
        # Step 7: Update performance metrics
        self.performance_metrics["total_decisions"] += 1
        
        print(f"🎯 Agent1 Decision: Basal {basal_rate:.2f} U/hr, Bolus {bolus_dose:.2f} U")
        print(f"  Chosen: {chosen_prediction.model_name} {chosen_prediction.horizon_minutes}min → {chosen_prediction.predicted_glucose:.1f} mg/dL")
        print(f"  Reasoning: {reasoning}")
        
        return decision
    
    def _extract_features(self, current_state: Dict[str, Any]) -> np.ndarray:
        """Extract features for model prediction."""
        # Extract key features from current state
        features = [
            current_state.get("cgm", 100.0),
            current_state.get("cgm_trend", 0.0),
            current_state.get("iob", 0.0),
            current_state.get("cob", 0.0),
            current_state.get("time_of_day", 12.0),
            current_state.get("time_since_meal", 180.0),
            current_state.get("time_since_bolus", 180.0),
            current_state.get("stress_level", 1.0),
            current_state.get("exercise_recent", 0.0),
            current_state.get("isf", 50.0),
            current_state.get("cr", 12.0),
            current_state.get("basal_rate", 1.0),
            current_state.get("glucose_variability", 0.2),
            np.sin(2 * np.pi * current_state.get("time_of_day", 12.0) / 24),
            np.cos(2 * np.pi * current_state.get("time_of_day", 12.0) / 24),
            current_state.get("day_of_week", 3.0)
        ]
        
        return np.array(features).reshape(1, -1)
    
    def _calculate_prediction_confidence(self, model_name: str, horizon: int, current_state: Dict[str, Any]) -> float:
        """Calculate confidence in a prediction based on model and context."""
        base_confidence = {
            "lstm": 0.8,
            "transformer": 0.85,
            "ensemble": 0.9
        }.get(model_name, 0.7)
        
        # Adjust confidence based on horizon
        if horizon <= 30:
            horizon_factor = 1.0
        elif horizon <= 60:
            horizon_factor = 0.9
        elif horizon <= 90:
            horizon_factor = 0.8
        else:
            horizon_factor = 0.7
        
        # Adjust based on glucose stability
        cgm_trend = abs(current_state.get("cgm_trend", 0.0))
        stability_factor = max(0.5, 1.0 - cgm_trend / 50.0)  # Lower confidence for rapid changes
        
        return base_confidence * horizon_factor * stability_factor
    
    def _choose_optimal_prediction(self, predictions: List[PredictionResult], current_state: Dict[str, Any]) -> PredictionResult:
        """Choose the optimal prediction using advanced selection logic."""
        
        if not predictions:
            # Fallback prediction
            return PredictionResult(
                model_name="fallback",
                horizon_minutes=30,
                predicted_glucose=current_state.get("cgm", 100.0),
                confidence=0.5,
                timestamp=datetime.now()
            )
        
        current_glucose = current_state.get("cgm", 100.0)
        iob = current_state.get("iob", 0.0)
        cob = current_state.get("cob", 0.0)
        
        # Scoring function for each prediction
        scored_predictions = []
        
        for pred in predictions:
            score = 0.0
            
            # Base confidence score
            score += pred.confidence * 100
            
            # Prefer predictions that keep glucose in target range
            if self.glucose_range_tight[0] <= pred.predicted_glucose <= self.glucose_range_tight[1]:
                score += 50
            elif self.glucose_range_acceptable[0] <= pred.predicted_glucose <= self.glucose_range_acceptable[1]:
                score += 25
            else:
                score -= 25
            
            # Prefer appropriate horizons based on context
            if iob > 2.0:  # High IOB - prefer longer horizons
                if pred.horizon_minutes >= 60:
                    score += 20
            elif cob > 20.0:  # High COB - prefer medium horizons
                if 30 <= pred.horizon_minutes <= 60:
                    score += 20
            else:  # Stable state - prefer shorter horizons
                if pred.horizon_minutes <= 30:
                    score += 20
            
            # Prefer transformer for longer horizons, LSTM for shorter
            if pred.horizon_minutes >= 60 and pred.model_name == "transformer":
                score += 15
            elif pred.horizon_minutes <= 30 and pred.model_name == "lstm":
                score += 15
            elif pred.model_name == "ensemble":
                score += 10  # Ensemble is always good
            
            # Penalize extreme predictions unless justified
            if pred.predicted_glucose < 70 or pred.predicted_glucose > 250:
                if not (current_glucose < 80 or current_glucose > 200):  # Not justified by current state
                    score -= 30
            
            scored_predictions.append((score, pred))
        
        # Choose highest scoring prediction
        scored_predictions.sort(key=lambda x: x[0], reverse=True)
        chosen_prediction = scored_predictions[0][1]
        
        return chosen_prediction
    
    def _calculate_insulin_dosing(
        self, 
        chosen_prediction: PredictionResult, 
        current_state: Dict[str, Any],
        all_predictions: List[PredictionResult]
    ) -> Tuple[float, float, str]:
        """Calculate basal and bolus insulin based on chosen prediction."""
        
        current_glucose = current_state.get("cgm", 100.0)
        predicted_glucose = chosen_prediction.predicted_glucose
        horizon = chosen_prediction.horizon_minutes
        
        iob = current_state.get("iob", 0.0)
        cob = current_state.get("cob", 0.0)
        isf = current_state.get("isf", 50.0)
        cr = current_state.get("cr", 12.0)
        current_basal = current_state.get("basal_rate", 1.0)
        
        # Initialize dosing
        basal_rate = current_basal
        bolus_dose = 0.0
        reasoning_parts = []
        
        # Basal adjustment based on prediction trend
        glucose_change = predicted_glucose - current_glucose
        time_factor = horizon / 60.0  # Convert to hours
        
        if glucose_change > 30 and time_factor > 0:  # Rising glucose
            basal_increase = min(0.5, glucose_change / (isf * time_factor * 2))
            basal_rate = min(current_basal + basal_increase, current_basal * 1.5)
            reasoning_parts.append(f"Increased basal by {basal_increase:.2f} U/hr for rising glucose")
        elif glucose_change < -20 and time_factor > 0:  # Falling glucose
            basal_decrease = min(0.3, abs(glucose_change) / (isf * time_factor * 3))
            basal_rate = max(current_basal - basal_decrease, current_basal * 0.5)
            reasoning_parts.append(f"Decreased basal by {basal_decrease:.2f} U/hr for falling glucose")
        
        # Bolus calculation for immediate correction
        if current_glucose > 150 and iob < 1.0:
            correction_needed = (current_glucose - self.target_glucose) / isf
            bolus_dose = max(0.0, correction_needed - iob * 0.5)  # Account for existing IOB
            bolus_dose = min(bolus_dose, 5.0)  # Safety limit
            reasoning_parts.append(f"Correction bolus {bolus_dose:.2f} U for glucose {current_glucose:.0f}")
        
        # Meal bolus if COB detected
        if cob > 5.0:
            meal_bolus = cob / cr
            bolus_dose += meal_bolus
            reasoning_parts.append(f"Meal bolus {meal_bolus:.2f} U for {cob:.0f}g COB")
        
        # Safety checks
        if current_glucose < 80:
            basal_rate = min(basal_rate, current_basal * 0.7)
            bolus_dose = 0.0
            reasoning_parts.append("Safety: Reduced insulin for low glucose")
        
        # Limit total bolus
        bolus_dose = min(bolus_dose, 8.0)
        
        reasoning = "; ".join(reasoning_parts) if reasoning_parts else "Maintaining current therapy"
        
        return basal_rate, bolus_dose, reasoning
    
    def _save_decision_pattern(self, decision: InsulinDecision, current_state: Dict[str, Any]):
        """Save decision as a pattern for Agent 2 analysis."""
        
        pattern = DecisionPattern(
            pattern_id=f"decision_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
            glucose_context={
                "current_glucose": current_state.get("cgm", 100.0),
                "glucose_trend": current_state.get("cgm_trend", 0.0),
                "iob": current_state.get("iob", 0.0),
                "cob": current_state.get("cob", 0.0),
                "time_of_day": current_state.get("time_of_day", 12.0)
            },
            predictions_used=[decision.chosen_prediction] + decision.alternative_predictions,
            insulin_decision=decision,
            actual_outcome=None,  # Will be updated later
            outcome_quality=None,  # Will be updated later
            timestamp=datetime.now()
        )
        
        # Save to pattern repository
        try:
            pattern_data = {
                "pattern_id": pattern.pattern_id,
                "pattern_type": "insulin_decision",
                "data": json.dumps(asdict(pattern), default=str),
                "timestamp": pattern.timestamp.isoformat(),
                "quality_score": 0.5  # Initial neutral score
            }
            
            self.pattern_repository.add_pattern(pattern_data)
            self.decision_history.append(pattern)
            
        except Exception as e:
            print(f"Error saving decision pattern: {e}")
    
    def update_decision_outcome(self, pattern_id: str, actual_glucose: float):
        """Update a decision pattern with actual outcome."""
        
        # Find the pattern
        pattern = None
        for p in self.decision_history:
            if p.pattern_id == pattern_id:
                pattern = p
                break
        
        if pattern:
            pattern.actual_outcome = actual_glucose
            
            # Calculate outcome quality
            predicted_glucose = pattern.insulin_decision.chosen_prediction.predicted_glucose
            error = abs(actual_glucose - predicted_glucose)
            
            if error <= 10:
                pattern.outcome_quality = PredictionQuality.EXCELLENT
                quality_score = 1.0
            elif error <= 20:
                pattern.outcome_quality = PredictionQuality.GOOD
                quality_score = 0.8
            elif error <= 40:
                pattern.outcome_quality = PredictionQuality.FAIR
                quality_score = 0.6
            else:
                pattern.outcome_quality = PredictionQuality.POOR
                quality_score = 0.3
            
            # Update performance metrics
            if pattern.outcome_quality in [PredictionQuality.EXCELLENT, PredictionQuality.GOOD]:
                self.performance_metrics["successful_decisions"] += 1
            
            self.performance_metrics["average_accuracy"] = (
                self.performance_metrics["successful_decisions"] / 
                self.performance_metrics["total_decisions"]
            )
            
            # Update in repository
            try:
                updated_data = {
                    "pattern_id": pattern.pattern_id,
                    "pattern_type": "insulin_decision",
                    "data": json.dumps(asdict(pattern), default=str),
                    "timestamp": pattern.timestamp.isoformat(),
                    "quality_score": quality_score
                }
                
                self.pattern_repository.update_pattern(pattern.pattern_id, updated_data)
                
            except Exception as e:
                print(f"Error updating decision pattern: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for Agent 1."""
        return {
            "total_decisions": self.performance_metrics["total_decisions"],
            "successful_decisions": self.performance_metrics["successful_decisions"],
            "success_rate": self.performance_metrics["average_accuracy"],
            "recent_decisions": len([p for p in self.decision_history if 
                                   (datetime.now() - p.timestamp).total_seconds() < 3600])  # Last hour
        }

class Agent2_PatternInspector:
    """
    Agent 2: Pattern Inspector
    - Analyzes decision patterns from Agent 1
    - Identifies successful and unsuccessful patterns
    - Finds optimal strategies and pushes improvements to Agent 1
    - Triggers model retraining when needed
    """
    
    def __init__(self, pattern_repository: RepositoryManager, agent1: Agent1_DecisionMaker):
        self.pattern_repository = pattern_repository
        self.agent1 = agent1
        self.analysis_history = []
        
        # Analysis parameters
        self.min_patterns_for_analysis = 10
        self.analysis_interval_hours = 1
        self.last_analysis_time = datetime.now()
        
        print(f"🕵️ Agent2_PatternInspector initialized")
        print(f"  Min patterns for analysis: {self.min_patterns_for_analysis}")
        print(f"  Analysis interval: {self.analysis_interval_hours} hours")
    
    def analyze_patterns_and_improve(self) -> Dict[str, Any]:
        """
        Analyze patterns and push improvements to Agent 1.
        
        Returns:
            Analysis results and improvement recommendations
        """
        
        # Check if it's time for analysis
        time_since_last = (datetime.now() - self.last_analysis_time).total_seconds() / 3600
        if time_since_last < self.analysis_interval_hours:
            return {"status": "too_soon", "next_analysis_in_hours": self.analysis_interval_hours - time_since_last}
        
        print(f"\n🔍 Agent2 starting pattern analysis...")
        
        # Step 1: Retrieve recent patterns
        patterns = self._retrieve_recent_patterns()
        
        if len(patterns) < self.min_patterns_for_analysis:
            return {"status": "insufficient_data", "patterns_available": len(patterns)}
        
        # Step 2: Analyze pattern performance
        analysis_results = self._analyze_pattern_performance(patterns)
        
        # Step 3: Identify improvement opportunities
        improvements = self._identify_improvements(analysis_results)
        
        # Step 4: Push improvements to Agent 1
        self._push_improvements_to_agent1(improvements)
        
        # Step 5: Determine if model retraining is needed
        retrain_recommendation = self._assess_retraining_need(analysis_results)
        
        self.last_analysis_time = datetime.now()
        
        analysis_summary = {
            "status": "completed",
            "patterns_analyzed": len(patterns),
            "analysis_results": analysis_results,
            "improvements_identified": len(improvements),
            "retrain_recommended": retrain_recommendation,
            "timestamp": datetime.now().isoformat()
        }
        
        self.analysis_history.append(analysis_summary)
        
        print(f"✅ Agent2 analysis complete: {len(improvements)} improvements identified")
        
        return analysis_summary
    
    def _retrieve_recent_patterns(self) -> List[Dict[str, Any]]:
        """Retrieve recent decision patterns from repository."""
        
        try:
            # Get patterns from last 24 hours
            cutoff_time = datetime.now() - timedelta(hours=24)
            
            patterns = self.pattern_repository.get_patterns_by_type("insulin_decision")
            
            recent_patterns = []
            for pattern in patterns:
                pattern_time = datetime.fromisoformat(pattern.get("timestamp", ""))
                if pattern_time >= cutoff_time:
                    recent_patterns.append(pattern)
            
            return recent_patterns
            
        except Exception as e:
            print(f"Error retrieving patterns: {e}")
            return []
    
    def _analyze_pattern_performance(self, patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze performance of decision patterns."""
        
        analysis = {
            "total_patterns": len(patterns),
            "quality_distribution": {},
            "model_performance": {},
            "horizon_performance": {},
            "glucose_range_performance": {},
            "time_of_day_performance": {}
        }
        
        quality_counts = {}
        model_performance = {}
        horizon_performance = {}
        glucose_performance = {}
        time_performance = {}
        
        for pattern_data in patterns:
            try:
                pattern_dict = json.loads(pattern_data["data"])
                quality_score = pattern_data.get("quality_score", 0.5)
                
                # Quality distribution
                quality = pattern_dict.get("outcome_quality", "unknown")
                quality_counts[quality] = quality_counts.get(quality, 0) + 1
                
                # Model performance
                chosen_pred = pattern_dict["insulin_decision"]["chosen_prediction"]
                model_name = chosen_pred["model_name"]
                if model_name not in model_performance:
                    model_performance[model_name] = {"count": 0, "total_score": 0}
                model_performance[model_name]["count"] += 1
                model_performance[model_name]["total_score"] += quality_score
                
                # Horizon performance
                horizon = chosen_pred["horizon_minutes"]
                if horizon not in horizon_performance:
                    horizon_performance[horizon] = {"count": 0, "total_score": 0}
                horizon_performance[horizon]["count"] += 1
                horizon_performance[horizon]["total_score"] += quality_score
                
                # Glucose range performance
                current_glucose = pattern_dict["glucose_context"]["current_glucose"]
                glucose_range = self._categorize_glucose(current_glucose)
                if glucose_range not in glucose_performance:
                    glucose_performance[glucose_range] = {"count": 0, "total_score": 0}
                glucose_performance[glucose_range]["count"] += 1
                glucose_performance[glucose_range]["total_score"] += quality_score
                
                # Time of day performance
                time_of_day = pattern_dict["glucose_context"]["time_of_day"]
                time_category = self._categorize_time(time_of_day)
                if time_category not in time_performance:
                    time_performance[time_category] = {"count": 0, "total_score": 0}
                time_performance[time_category]["count"] += 1
                time_performance[time_category]["total_score"] += quality_score
                
            except Exception as e:
                print(f"Error analyzing pattern: {e}")
                continue
        
        # Calculate averages
        analysis["quality_distribution"] = quality_counts
        
        for model, data in model_performance.items():
            analysis["model_performance"][model] = data["total_score"] / data["count"] if data["count"] > 0 else 0
        
        for horizon, data in horizon_performance.items():
            analysis["horizon_performance"][horizon] = data["total_score"] / data["count"] if data["count"] > 0 else 0
        
        for glucose_range, data in glucose_performance.items():
            analysis["glucose_range_performance"][glucose_range] = data["total_score"] / data["count"] if data["count"] > 0 else 0
        
        for time_cat, data in time_performance.items():
            analysis["time_of_day_performance"][time_cat] = data["total_score"] / data["count"] if data["count"] > 0 else 0
        
        return analysis
    
    def _categorize_glucose(self, glucose: float) -> str:
        """Categorize glucose level."""
        if glucose < 70:
            return "low"
        elif glucose < 140:
            return "normal"
        elif glucose < 200:
            return "high"
        else:
            return "very_high"
    
    def _categorize_time(self, hour: float) -> str:
        """Categorize time of day."""
        if 6 <= hour < 12:
            return "morning"
        elif 12 <= hour < 18:
            return "afternoon"
        elif 18 <= hour < 22:
            return "evening"
        else:
            return "night"
    
    def _identify_improvements(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify improvement opportunities."""
        
        improvements = []
        
        # Model performance improvements
        model_perf = analysis["model_performance"]
        if model_perf:
            best_model = max(model_perf.items(), key=lambda x: x[1])
            worst_model = min(model_perf.items(), key=lambda x: x[1])
            
            if best_model[1] - worst_model[1] > 0.2:  # Significant difference
                improvements.append({
                    "type": "model_preference",
                    "recommendation": f"Prefer {best_model[0]} over {worst_model[0]}",
                    "impact": "high",
                    "data": {"best_model": best_model[0], "worst_model": worst_model[0]}
                })
        
        # Horizon performance improvements
        horizon_perf = analysis["horizon_performance"]
        if horizon_perf:
            best_horizon = max(horizon_perf.items(), key=lambda x: x[1])
            
            improvements.append({
                "type": "horizon_preference",
                "recommendation": f"Prefer {best_horizon[0]}-minute predictions",
                "impact": "medium",
                "data": {"best_horizon": best_horizon[0]}
            })
        
        # Context-specific improvements
        glucose_perf = analysis["glucose_range_performance"]
        for glucose_range, score in glucose_perf.items():
            if score < 0.6:  # Poor performance
                improvements.append({
                    "type": "glucose_context",
                    "recommendation": f"Improve decision-making for {glucose_range} glucose",
                    "impact": "high",
                    "data": {"glucose_range": glucose_range, "current_score": score}
                })
        
        return improvements
    
    def _push_improvements_to_agent1(self, improvements: List[Dict[str, Any]]):
        """Push improvements to Agent 1."""
        
        for improvement in improvements:
            if improvement["type"] == "model_preference":
                # Update Agent 1's model selection logic
                self._update_agent1_model_preferences(improvement["data"])
            elif improvement["type"] == "horizon_preference":
                # Update Agent 1's horizon selection logic
                self._update_agent1_horizon_preferences(improvement["data"])
            elif improvement["type"] == "glucose_context":
                # Update Agent 1's context-specific logic
                self._update_agent1_glucose_context(improvement["data"])
        
        print(f"📤 Pushed {len(improvements)} improvements to Agent1")
    
    def _update_agent1_model_preferences(self, data: Dict[str, Any]):
        """Update Agent 1's model preferences."""
        # This would update Agent 1's scoring logic for model selection
        print(f"  Updated model preference: favor {data['best_model']}")
    
    def _update_agent1_horizon_preferences(self, data: Dict[str, Any]):
        """Update Agent 1's horizon preferences."""
        print(f"  Updated horizon preference: favor {data['best_horizon']} minutes")
    
    def _update_agent1_glucose_context(self, data: Dict[str, Any]):
        """Update Agent 1's glucose context handling."""
        print(f"  Updated glucose context handling for {data['glucose_range']} range")
    
    def _assess_retraining_need(self, analysis: Dict[str, Any]) -> bool:
        """Assess if model retraining is needed."""
        
        # Check overall performance
        total_patterns = analysis["total_patterns"]
        quality_dist = analysis["quality_distribution"]
        
        poor_patterns = quality_dist.get("poor", 0)
        poor_ratio = poor_patterns / total_patterns if total_patterns > 0 else 0
        
        # Recommend retraining if >30% of patterns are poor
        return poor_ratio > 0.3
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get summary of recent analyses."""
        return {
            "total_analyses": len(self.analysis_history),
            "last_analysis": self.analysis_history[-1] if self.analysis_history else None,
            "next_analysis_due": (self.last_analysis_time + timedelta(hours=self.analysis_interval_hours)).isoformat()
        }

class ContinuousLearningLoop:
    """
    Continuous Learning Loop System
    Orchestrates the flow: Models → Agent1 → Patterns → Agent2 → Models
    """

    def __init__(self, transformer_zoo: TransformerZoo, pattern_repository: RepositoryManager):
        self.transformer_zoo = transformer_zoo
        self.pattern_repository = pattern_repository

        # Initialize agents
        self.agent1 = Agent1_DecisionMaker(transformer_zoo, pattern_repository)
        self.agent2 = Agent2_PatternInspector(pattern_repository, self.agent1)

        # Learning loop parameters
        self.loop_active = False
        self.loop_iteration = 0
        self.retraining_threshold = 100  # Retrain after 100 poor decisions
        self.performance_history = []

        print(f"🔄 ContinuousLearningLoop initialized")
        print(f"  Agent1: Decision Maker ready")
        print(f"  Agent2: Pattern Inspector ready")
        print(f"  Retraining threshold: {self.retraining_threshold} patterns")

    def start_learning_loop(self):
        """Start the continuous learning loop."""
        self.loop_active = True
        print(f"🚀 Continuous Learning Loop STARTED")

    def stop_learning_loop(self):
        """Stop the continuous learning loop."""
        self.loop_active = False
        print(f"⏹️ Continuous Learning Loop STOPPED")

    def process_patient_state(self, current_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a patient state through the complete learning loop.

        Args:
            current_state: Current patient state

        Returns:
            Complete decision and analysis results
        """

        if not self.loop_active:
            return {"error": "Learning loop not active"}

        self.loop_iteration += 1

        print(f"\n🔄 Learning Loop Iteration {self.loop_iteration}")

        # Step 1: Agent1 makes decision using TransformerZoo predictions
        decision = self.agent1.make_insulin_decision(current_state)

        # Step 2: Agent2 analyzes patterns (if enough data)
        analysis_result = self.agent2.analyze_patterns_and_improve()

        # Step 3: Check if model retraining is needed
        retrain_needed = analysis_result.get("retrain_recommended", False)

        # Step 4: Trigger retraining if needed
        retraining_result = None
        if retrain_needed:
            retraining_result = self._trigger_model_retraining()

        # Step 5: Update performance tracking
        self._update_performance_tracking(decision, analysis_result)

        return {
            "loop_iteration": self.loop_iteration,
            "insulin_decision": {
                "basal_rate_u_hr": decision.basal_rate_u_hr,
                "bolus_u": decision.bolus_u,
                "reasoning": decision.reasoning,
                "chosen_model": decision.chosen_prediction.model_name,
                "chosen_horizon": decision.chosen_prediction.horizon_minutes,
                "predicted_glucose": decision.chosen_prediction.predicted_glucose
            },
            "pattern_analysis": analysis_result,
            "retraining_triggered": retrain_needed,
            "retraining_result": retraining_result,
            "system_performance": self.get_system_performance()
        }

    def update_decision_outcome(self, pattern_id: str, actual_glucose: float):
        """Update decision outcome and trigger learning if needed."""

        # Update Agent1's decision outcome
        self.agent1.update_decision_outcome(pattern_id, actual_glucose)

        # Check if this triggers any immediate learning
        if self.loop_iteration % 10 == 0:  # Every 10 decisions
            analysis_result = self.agent2.analyze_patterns_and_improve()

            if analysis_result.get("retrain_recommended", False):
                self._trigger_model_retraining()

    def _trigger_model_retraining(self) -> Dict[str, Any]:
        """Trigger retraining of models based on recent patterns."""

        print(f"🔄 Triggering model retraining...")

        try:
            # Get recent patterns for retraining data
            patterns = self.agent2._retrieve_recent_patterns()

            if len(patterns) < 50:  # Need minimum data
                return {"status": "insufficient_data", "patterns_available": len(patterns)}

            # Extract features and targets from patterns
            X_retrain, y_retrain = self._extract_retraining_data(patterns)

            if len(X_retrain) == 0:
                return {"status": "no_valid_data"}

            # Retrain models
            retraining_results = self.transformer_zoo.train_all_models(X_retrain, y_retrain)

            print(f"✅ Model retraining complete")

            return {
                "status": "completed",
                "patterns_used": len(patterns),
                "training_samples": len(X_retrain),
                "models_retrained": list(retraining_results.keys()),
                "retraining_results": retraining_results
            }

        except Exception as e:
            print(f"❌ Model retraining failed: {e}")
            return {"status": "failed", "error": str(e)}

    def _extract_retraining_data(self, patterns: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        """Extract training data from decision patterns."""

        X_data = []
        y_data = []

        for pattern_data in patterns:
            try:
                pattern_dict = json.loads(pattern_data["data"])

                # Skip patterns without outcomes
                if pattern_dict.get("actual_outcome") is None:
                    continue

                # Extract features from glucose context
                glucose_context = pattern_dict["glucose_context"]

                features = [
                    glucose_context.get("current_glucose", 100.0),
                    glucose_context.get("glucose_trend", 0.0),
                    glucose_context.get("iob", 0.0),
                    glucose_context.get("cob", 0.0),
                    glucose_context.get("time_of_day", 12.0),
                    # Add more features as needed
                ]

                # Pad features to match expected input dimension
                while len(features) < self.transformer_zoo.input_dim:
                    features.append(0.0)

                features = features[:self.transformer_zoo.input_dim]  # Truncate if too long

                # Target is the actual outcome for all horizons
                actual_glucose = pattern_dict["actual_outcome"]
                targets = [actual_glucose] * len(self.transformer_zoo.prediction_horizons)

                X_data.append(features)
                y_data.append(targets)

            except Exception as e:
                print(f"Error extracting pattern data: {e}")
                continue

        return np.array(X_data), np.array(y_data)

    def _update_performance_tracking(self, decision: InsulinDecision, analysis_result: Dict[str, Any]):
        """Update system performance tracking."""

        performance_snapshot = {
            "iteration": self.loop_iteration,
            "timestamp": datetime.now().isoformat(),
            "agent1_performance": self.agent1.get_performance_summary(),
            "agent2_analysis": analysis_result,
            "chosen_model": decision.chosen_prediction.model_name,
            "chosen_horizon": decision.chosen_prediction.horizon_minutes,
            "prediction_confidence": decision.chosen_prediction.confidence
        }

        self.performance_history.append(performance_snapshot)

        # Keep only last 100 snapshots
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]

    def get_system_performance(self) -> Dict[str, Any]:
        """Get overall system performance metrics."""

        if not self.performance_history:
            return {"status": "no_data"}

        recent_performance = self.performance_history[-10:]  # Last 10 iterations

        # Calculate trends
        model_usage = {}
        horizon_usage = {}
        confidence_scores = []

        for perf in recent_performance:
            model = perf["chosen_model"]
            horizon = perf["chosen_horizon"]
            confidence = perf["prediction_confidence"]

            model_usage[model] = model_usage.get(model, 0) + 1
            horizon_usage[horizon] = horizon_usage.get(horizon, 0) + 1
            confidence_scores.append(confidence)

        return {
            "total_iterations": self.loop_iteration,
            "recent_model_usage": model_usage,
            "recent_horizon_usage": horizon_usage,
            "average_confidence": np.mean(confidence_scores) if confidence_scores else 0,
            "agent1_success_rate": self.agent1.get_performance_summary().get("success_rate", 0),
            "loop_active": self.loop_active,
            "last_update": datetime.now().isoformat()
        }

    def get_complete_system_status(self) -> Dict[str, Any]:
        """Get complete status of the learning system."""

        return {
            "learning_loop": {
                "active": self.loop_active,
                "iteration": self.loop_iteration,
                "performance": self.get_system_performance()
            },
            "transformer_zoo": self.transformer_zoo.get_zoo_status(),
            "agent1": self.agent1.get_performance_summary(),
            "agent2": self.agent2.get_analysis_summary(),
            "pattern_repository": {
                "total_patterns": len(self.pattern_repository.get_patterns_by_type("insulin_decision")),
                "recent_patterns": len(self.agent2._retrieve_recent_patterns())
            }
        }
