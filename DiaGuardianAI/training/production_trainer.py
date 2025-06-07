#!/usr/bin/env python3
"""
DiaGuardianAI Production Training Pipeline
Train PatternAdvisorAgent on diverse synthetic data for >90% accuracy
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Tuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import glob

# Ensure the DiaGuardianAI package is discoverable
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from DiaGuardianAI.agents.pattern_advisor_agent import PatternAdvisorAgent
from DiaGuardianAI.data_generation.human_model_factory import HumanModelFactory

class ProductionTrainer:
    """Production training pipeline for DiaGuardianAI PatternAdvisorAgent."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        
        # Setup directories
        self.training_data_dir = self.config.get("training_data_dir", "training_data")
        self.models_output_dir = self.config.get("models_output_dir", "trained_models")
        self.datasets_dir = os.path.join(self.training_data_dir, "datasets")
        
        os.makedirs(self.models_output_dir, exist_ok=True)
        
        print(f"🎯 ProductionTrainer initialized")
        print(f"  Training data directory: {self.training_data_dir}")
        print(f"  Models output directory: {self.models_output_dir}")
        print(f"  Target accuracy: >{self.config['target_accuracy']:.0%}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default training configuration."""
        return {
            # Data parameters
            "training_data_dir": "training_data",
            "models_output_dir": "trained_models",
            
            # Training parameters
            "test_size": 0.2,  # 20% for testing
            "validation_size": 0.1,  # 10% for validation
            "random_state": 42,
            
            # Model parameters
            "model_type": "mlp_regressor",
            "model_params": {
                "hidden_layer_sizes": [256, 128, 64],
                "max_iter": 1000,
                "random_state": 42,
                "early_stopping": True,
                "n_iter_no_change": 20,
                "solver": "adam",
                "activation": "relu",
                "learning_rate": "adaptive"
            },
            
            # Performance targets
            "target_accuracy": 0.90,  # >90% accuracy
            "target_mae": 10.0,  # Mean Absolute Error < 10 mg/dL
            "target_r2": 0.85,   # R² > 0.85
            
            # Feature engineering
            "feature_window_size": 12,  # 1 hour of history (12 * 5min)
            "prediction_horizon": 6,    # 30 minutes ahead (6 * 5min)
            "include_patient_features": True,
            "include_time_features": True,
        }
    
    def train_production_model(self) -> Dict[str, Any]:
        """Train production model on synthetic data."""
        
        print(f"\n" + "="*80)
        print(f"🚀 STARTING PRODUCTION MODEL TRAINING")
        print(f"Target: >{self.config['target_accuracy']:.0%} accuracy")
        print(f"="*80)
        
        # Step 1: Load training data
        print(f"\n📊 STEP 1: Loading training data...")
        dataset = self._load_latest_dataset()
        
        # Step 2: Prepare features and targets
        print(f"\n🔧 STEP 2: Preparing features and targets...")
        X, y, feature_info = self._prepare_training_data(dataset)
        
        # Step 3: Split data
        print(f"\n📈 STEP 3: Splitting data...")
        X_train, X_test, y_train, y_test = self._split_data(X, y)
        
        # Step 4: Train model
        print(f"\n🎯 STEP 4: Training PatternAdvisorAgent...")
        agent, training_history = self._train_agent(X_train, y_train, X_test, y_test)
        
        # Step 5: Evaluate model
        print(f"\n✅ STEP 5: Evaluating model performance...")
        evaluation_results = self._evaluate_model(agent, X_test, y_test, feature_info)
        
        # Step 6: Save trained model
        print(f"\n💾 STEP 6: Saving trained model...")
        model_info = self._save_trained_model(agent, evaluation_results, feature_info)
        
        # Step 7: Generate training report
        print(f"\n📋 STEP 7: Generating training report...")
        training_report = self._generate_training_report(
            dataset, evaluation_results, training_history, model_info
        )
        
        print(f"\n" + "="*80)
        print(f"🎉 PRODUCTION MODEL TRAINING COMPLETE!")
        print(f"  Model accuracy: {evaluation_results['accuracy']:.1%}")
        print(f"  Target achieved: {'✅ YES' if evaluation_results['accuracy'] >= self.config['target_accuracy'] else '❌ NO'}")
        print(f"  Model saved to: {model_info['model_path']}")
        print(f"="*80)
        
        return training_report
    
    def _load_latest_dataset(self) -> Dict[str, Any]:
        """Load the latest production dataset."""
        
        # Find latest dataset file
        dataset_pattern = os.path.join(self.datasets_dir, "production_dataset_*.pkl")
        dataset_files = glob.glob(dataset_pattern)
        
        if not dataset_files:
            raise FileNotFoundError(f"No production datasets found in {self.datasets_dir}")
        
        latest_file = max(dataset_files, key=os.path.getctime)
        
        print(f"  Loading dataset: {latest_file}")
        
        with open(latest_file, 'rb') as f:
            dataset = pickle.load(f)
        
        print(f"  ✅ Dataset loaded successfully")
        print(f"    Total simulations: {dataset['metadata']['total_simulations']}")
        print(f"    Total data points: {dataset['metadata']['total_data_points']:,}")
        print(f"    Average TIR: {dataset['quality_metrics']['average_time_in_range']:.1%}")
        
        return dataset
    
    def _prepare_training_data(self, dataset: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Prepare features and targets for training."""
        
        simulation_data = dataset["data"]
        
        print(f"  Processing {len(simulation_data)} simulations...")
        
        # Collect all features and targets
        all_features = []
        all_targets = []
        feature_names = []
        
        for sim_idx, sim in enumerate(simulation_data):
            if sim_idx % 50 == 0:
                print(f"    Processing simulation {sim_idx + 1}/{len(simulation_data)}")
            
            # Extract time series data
            cgm_readings = np.array(sim["cgm_readings"])
            insulin_basal = np.array(sim["insulin_basal"])
            insulin_bolus = np.array(sim["insulin_bolus"])
            carbs_ingested = np.array(sim["carbs_ingested"])
            
            # Patient features
            profile = sim["profile_summary"]
            patient_features = [
                profile["isf"],
                profile["cr"],
                profile["basal_rate"],
                1.0 if profile["diabetes_type"] == "type_1" else 0.0
            ]
            
            # Create sliding windows
            window_size = self.config["feature_window_size"]
            horizon = self.config["prediction_horizon"]
            
            for i in range(window_size, len(cgm_readings) - horizon):
                # Historical features
                cgm_history = cgm_readings[i-window_size:i]
                insulin_basal_history = insulin_basal[i-window_size:i]
                insulin_bolus_history = insulin_bolus[i-window_size:i]
                carbs_history = carbs_ingested[i-window_size:i]
                
                # Current state
                current_cgm = cgm_readings[i]
                current_iob = np.sum(insulin_basal_history[-6:]) + np.sum(insulin_bolus_history[-6:])  # Last 30 min
                current_cob = np.sum(carbs_history[-6:])  # Last 30 min
                
                # Time features
                time_of_day = (i * 5 / 60) % 24  # Hour of day
                
                # Combine features
                features = []
                
                # CGM features
                features.extend([
                    current_cgm,
                    np.mean(cgm_history),
                    np.std(cgm_history),
                    cgm_history[-1] - cgm_history[-2] if len(cgm_history) >= 2 else 0,  # Trend
                ])
                
                # Insulin features
                features.extend([
                    current_iob,
                    np.sum(insulin_basal_history),
                    np.sum(insulin_bolus_history),
                ])
                
                # Carb features
                features.extend([
                    current_cob,
                    np.sum(carbs_history),
                ])
                
                # Patient features
                features.extend(patient_features)
                
                # Time features
                features.extend([
                    time_of_day,
                    np.sin(2 * np.pi * time_of_day / 24),
                    np.cos(2 * np.pi * time_of_day / 24),
                ])
                
                # Target: insulin needed for next period
                future_cgm = cgm_readings[i + horizon]
                
                # Calculate ideal insulin dose
                if future_cgm > 150:
                    correction_bolus = (future_cgm - 120) / profile["isf"]
                else:
                    correction_bolus = 0.0
                
                # Estimate meal bolus from carbs
                future_carbs = np.sum(carbs_ingested[i:i+horizon])
                meal_bolus = future_carbs / profile["cr"] if future_carbs > 0 else 0.0
                
                # Target: [basal_rate, bolus_dose]
                target_basal = profile["basal_rate"]
                target_bolus = correction_bolus + meal_bolus
                
                # Clip targets to reasonable ranges
                target_basal = np.clip(target_basal, 0.0, 3.0)
                target_bolus = np.clip(target_bolus, 0.0, 10.0)
                
                all_features.append(features)
                all_targets.append([target_basal, target_bolus])
        
        # Convert to numpy arrays
        X = np.array(all_features)
        y = np.array(all_targets)
        
        # Feature names for interpretability
        feature_names = [
            "current_cgm", "cgm_mean", "cgm_std", "cgm_trend",
            "current_iob", "basal_sum", "bolus_sum",
            "current_cob", "carbs_sum",
            "isf", "cr", "basal_rate", "is_type1",
            "time_of_day", "time_sin", "time_cos"
        ]
        
        feature_info = {
            "feature_names": feature_names,
            "num_features": X.shape[1],
            "num_samples": X.shape[0],
            "target_names": ["basal_rate_u_hr", "bolus_u"]
        }
        
        print(f"  ✅ Features prepared successfully")
        print(f"    Feature matrix shape: {X.shape}")
        print(f"    Target matrix shape: {y.shape}")
        print(f"    Number of features: {len(feature_names)}")
        
        return X, y, feature_info
    
    def _split_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into training and testing sets."""
        
        test_size = self.config["test_size"]
        random_state = self.config["random_state"]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=True
        )
        
        print(f"  ✅ Data split successfully")
        print(f"    Training samples: {X_train.shape[0]:,}")
        print(f"    Testing samples: {X_test.shape[0]:,}")
        print(f"    Test ratio: {test_size:.1%}")
        
        return X_train, X_test, y_train, y_test
    
    def _train_agent(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Tuple[PatternAdvisorAgent, Dict[str, Any]]:
        """Train the PatternAdvisorAgent."""
        
        # Initialize agent
        state_dim = X_train.shape[1]
        action_dim = y_train.shape[1]

        # Create a dummy repository for training
        from DiaGuardianAI.pattern_repository.repository_manager import RepositoryManager
        dummy_repo = RepositoryManager(db_path="dummy_training_repo.sqlite")

        agent = PatternAdvisorAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            action_keys_ordered=["basal_rate_u_hr", "bolus_u"],
            pattern_repository=dummy_repo,
            learning_model_type=self.config["model_type"],
            model_params=self.config["model_params"]
        )
        
        print(f"  Training {self.config['model_type']} model...")
        print(f"    State dimension: {state_dim}")
        print(f"    Action dimension: {action_dim}")
        print(f"    Training samples: {X_train.shape[0]:,}")
        
        # Train the agent
        training_start = datetime.now()

        # Train the agent using the correct method signature
        agent.train(features=X_train, actions=y_train)
        
        training_time = (datetime.now() - training_start).total_seconds()
        
        print(f"  ✅ Training completed in {training_time:.1f} seconds")
        
        training_history = {
            "training_time_seconds": training_time,
            "training_samples": len(X_train),
            "model_type": self.config["model_type"],
            "model_params": self.config["model_params"]
        }
        
        return agent, training_history
    
    def _evaluate_model(
        self, 
        agent: PatternAdvisorAgent, 
        X_test: np.ndarray, 
        y_test: np.ndarray,
        feature_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate model performance."""
        
        print(f"  Evaluating on {len(X_test):,} test samples...")
        
        # Make predictions directly using the trained model
        predictions = []
        for i in range(len(X_test)):
            # Use the model directly to avoid feature reconstruction issues
            try:
                # Direct prediction using the trained model
                pred = agent.model.predict(X_test[i].reshape(1, -1))[0]
                predictions.append(pred)
            except Exception as e:
                print(f"    Error in prediction {i}: {e}")
                predictions.append([0.0, 0.0])
        
        predictions = np.array(predictions)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        # Calculate accuracy (within 20% of target)
        relative_errors = np.abs(predictions - y_test) / (y_test + 1e-8)
        accuracy = np.mean(relative_errors < 0.2)  # Within 20%
        
        evaluation_results = {
            "mse": mse,
            "mae": mae,
            "r2_score": r2,
            "accuracy": accuracy,
            "test_samples": len(X_test),
            "target_achieved": accuracy >= self.config["target_accuracy"]
        }
        
        print(f"  ✅ Evaluation completed")
        print(f"    Accuracy: {accuracy:.1%}")
        print(f"    MAE: {mae:.2f}")
        print(f"    R² Score: {r2:.3f}")
        print(f"    Target achieved: {'✅ YES' if evaluation_results['target_achieved'] else '❌ NO'}")
        
        return evaluation_results
    
    def _save_trained_model(
        self, 
        agent: PatternAdvisorAgent, 
        evaluation_results: Dict[str, Any],
        feature_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Save the trained model."""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        accuracy_str = f"{evaluation_results['accuracy']:.1%}".replace(".", "p")
        
        model_name = f"pattern_advisor_production_{accuracy_str}_{timestamp}"
        model_path = os.path.join(self.models_output_dir, f"{model_name}.pkl")
        
        # Save the agent model (avoiding sqlite connection pickle issue)
        import pickle
        import joblib

        # Save just the trained model
        model_save_path = model_path.replace('.pkl', '_model.joblib')
        joblib.dump(agent.model, model_save_path)

        # Save agent configuration
        config_save_path = model_path.replace('.pkl', '_config.json')
        import json
        agent_config = {
            "state_dim": agent.state_dim,
            "action_dim": agent.action_dim,
            "action_keys_ordered": agent.action_keys_ordered,
            "learning_model_type": agent.learning_model_type,
            "model_params": agent.model_params
        }
        with open(config_save_path, 'w') as f:
            json.dump(agent_config, f, indent=2)
        
        model_info = {
            "model_name": model_name,
            "model_path": model_path,
            "timestamp": timestamp,
            "accuracy": evaluation_results["accuracy"],
            "feature_info": feature_info
        }
        
        print(f"  ✅ Model saved to {model_save_path}")
        print(f"  ✅ Config saved to {config_save_path}")
        
        return model_info
    
    def _generate_training_report(
        self,
        dataset: Dict[str, Any],
        evaluation_results: Dict[str, Any],
        training_history: Dict[str, Any],
        model_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate comprehensive training report."""
        
        report = {
            "training_summary": {
                "timestamp": datetime.now().isoformat(),
                "model_name": model_info["model_name"],
                "accuracy_achieved": evaluation_results["accuracy"],
                "target_accuracy": self.config["target_accuracy"],
                "target_achieved": evaluation_results["target_achieved"]
            },
            "dataset_info": {
                "total_simulations": dataset["metadata"]["total_simulations"],
                "total_data_points": dataset["metadata"]["total_data_points"],
                "population_size": dataset["metadata"]["population_size"],
                "average_tir": dataset["quality_metrics"]["average_time_in_range"]
            },
            "model_performance": evaluation_results,
            "training_details": training_history,
            "model_info": model_info
        }
        
        # Save report
        report_path = os.path.join(self.models_output_dir, f"{model_info['model_name']}_report.json")
        import json
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"  ✅ Training report saved to {report_path}")
        
        return report

def main():
    """Main production training function."""
    
    print("🎯 DiaGuardianAI Production Training Pipeline")
    print("Target: >90% accuracy through diverse synthetic data training")
    
    # Initialize trainer
    trainer = ProductionTrainer()
    
    # Train production model
    report = trainer.train_production_model()
    
    print(f"\n🚀 PRODUCTION MODEL TRAINING COMPLETE!")
    if report["training_summary"]["target_achieved"]:
        print(f"🎉 SUCCESS: {report['training_summary']['accuracy_achieved']:.1%} accuracy achieved!")
        print(f"🎯 DiaGuardianAI is ready for deployment!")
    else:
        print(f"⚠️  Target not reached: {report['training_summary']['accuracy_achieved']:.1%}")
        print(f"💡 Consider: More data, different model, or hyperparameter tuning")

if __name__ == "__main__":
    main()
