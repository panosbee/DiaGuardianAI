# DiaGuardianAI - Train ML-based Meal Detector Example

import sys
import os
import numpy as np
from typing import Dict, Any, Optional

# Ensure the DiaGuardianAI package is discoverable
if __package__ is None or __package__ == '':
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from DiaGuardianAI.agents.meal_detector import MealDetector
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

def train_rf_meal_detector():
    print("--- Training ML-based Meal Detector (RandomForestClassifier) ---")

    # 1. Load the generated data
    data_load_dir = "DiaGuardianAI/datasets/generated_meal_detection_data"
    features_path = os.path.join(data_load_dir, "meal_features.npy")
    labels_path = os.path.join(data_load_dir, "meal_labels.npy") # Using is_meal labels for classification
    # carbs_path = os.path.join(data_load_dir, "meal_carbs.npy") # For future regression task

    if not (os.path.exists(features_path) and os.path.exists(labels_path)):
        print(f"Error: Data files not found in {data_load_dir}. Please run generate_meal_detection_data.py first.")
        return

    try:
        X_all = np.load(features_path)
        y_all_is_meal = np.load(labels_path)
        # y_all_carbs = np.load(carbs_path) # Load if doing carb regression later
        print(f"Loaded data: Features shape {X_all.shape}, Meal labels shape {y_all_is_meal.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    if X_all.shape[0] != y_all_is_meal.shape[0]:
        print("Error: Mismatch between number of feature samples and meal labels.")
        return
    
    if X_all.shape[0] == 0:
        print("Error: No data loaded to train the model.")
        return

    # 2. Split data into training and testing sets
    X_train, X_test, y_train_is_meal, y_test_is_meal = train_test_split(
        X_all, y_all_is_meal, test_size=0.2, random_state=42, stratify=y_all_is_meal if np.sum(y_all_is_meal) > 1 else None
    )
    print(f"Training set: {X_train.shape[0]} samples. Test set: {X_test.shape[0]} samples.")
    print(f"Positive meal events in training set: {np.sum(y_train_is_meal)}")
    print(f"Positive meal events in test set: {np.sum(y_test_is_meal)}")

    if np.sum(y_train_is_meal) == 0 or len(np.unique(y_train_is_meal)) < 2 :
        print("Warning: Training set contains only one class for meal detection. Model may not train effectively.")
        # Optionally, could skip training if only one class, but let's proceed for demonstration.
        if np.sum(y_train_is_meal) == 0:
             print("No positive meal samples in training data. Training will likely be trivial.")


    # 3. Initialize and Train the MealDetector
    # Define parameters for the RandomForestClassifier if desired
    # These can be tuned.
    rf_params = {
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": 42,
        "class_weight": "balanced" # Useful for imbalanced datasets
    }
    
    # The MealDetector's __init__ for "ml_based" can take model_config
    # which it passes to RandomForestClassifier(**model_config)
    meal_detector_ml = MealDetector(
        detection_method="ml_based",
        params={"model_config": rf_params} # Pass RF params here
    )

    print("\nTraining the RandomForestClassifier model...")
    meal_detector_ml.train_model(X_train, y_train_is_meal)

    # 4. Evaluate the Model (Basic)
    if meal_detector_ml.ml_model is not None and \
       hasattr(meal_detector_ml.ml_model, 'predict') and \
       X_test.shape[0] > 0:
        print("\nEvaluating the trained model on the test set...")
        try:
            y_pred_is_meal = meal_detector_ml.ml_model.predict(X_test)
            
            print("\nClassification Report:")
            print(classification_report(y_test_is_meal, y_pred_is_meal, zero_division=0))
            
            print("Confusion Matrix:")
            print(confusion_matrix(y_test_is_meal, y_pred_is_meal))
        except Exception as e:
            print(f"Error during model evaluation: {e}")
    else:
        print("Skipping evaluation as model does not have predict method or no test data.")


    # 5. Save the Trained Model
    model_save_dir = "DiaGuardianAI/models/meal_detector_model"
    os.makedirs(model_save_dir, exist_ok=True)
    model_save_path = os.path.join(model_save_dir, "rf_meal_detector.joblib")
    
    meal_detector_ml.save_model(model_save_path)

    # 6. Example of loading and using the saved model (conceptual)
    print(f"\n--- Conceptual: Loading and using the saved model from {model_save_path} ---")
    loaded_meal_detector = MealDetector(detection_method="ml_based", model_path=model_save_path)
    
    if loaded_meal_detector.ml_model and X_test.shape[0] > 0:
        # Test with the first sample from the test set
        sample_features = X_test[0]
        # The detect_meal_event method in MealDetector handles reshaping internally for predict_proba
        prob, time, carbs = loaded_meal_detector.detect_meal_event(
            cgm_history=[], # _extract_features_for_ml is called internally, but we pass features directly for predict
                            # This part needs refinement: detect_meal_event expects raw cgm, not features.
                            # For now, let's call predict_proba directly on the loaded model for this test.
        )
        # Corrected test:
        try:
            sample_pred_proba = loaded_meal_detector.ml_model.predict_proba(sample_features.reshape(1, -1))[0]
            print(f"Prediction for first test sample (Prob meal): {sample_pred_proba[1]:.2f} (Actual: {y_test_is_meal[0]})")
        except Exception as e:
            print(f"Error predicting with loaded model: {e}")
    else:
        print("Could not test loaded model (model not loaded or no test data).")

    print("\nML-based Meal Detector training script finished.")

if __name__ == "__main__":
    train_rf_meal_detector()