# DiaGuardianAI - Train Pattern Advisor Agent's Regressor Model

import sys
import os
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple, cast

# Ensure the DiaGuardianAI package is discoverable
if __package__ is None or __package__ == '':
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from DiaGuardianAI.agents.pattern_advisor_agent import PatternAdvisorAgent
from DiaGuardianAI.pattern_repository.repository_manager import RepositoryManager # For dummy repo
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score # Changed for regression

def train_advisor_regressor_model(): # Renamed function
    print("--- Training Pattern Advisor's Regressor Model (MLPRegressor) ---") # Updated title

    # 1. Load the generated data
    data_load_dir = os.path.join(project_root, "DiaGuardianAI", "datasets", "generated_advisor_training_data")
    features_path = os.path.join(data_load_dir, "advisor_features.npy")
    actions_path = os.path.join(data_load_dir, "advisor_successful_actions.csv")

    if not (os.path.exists(features_path) and os.path.exists(actions_path)):
        print(f"Error: Data files not found in {data_load_dir}. "
              "Please run generate_advisor_training_data.py first.")
        return

    try:
        X_all = np.load(features_path)
        actions_df = pd.read_csv(actions_path)
        print(f"Loaded data: Features shape {X_all.shape}, Actions DF shape {actions_df.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    if X_all.shape[0] != actions_df.shape[0]:
        print("Error: Mismatch between number of feature samples and action samples.")
        return
    
    if X_all.shape[0] == 0:
        print("Error: No data loaded to train the model.")
        return

    # 2. Prepare target variables for regression
    ACTION_KEYS = ['basal_rate_u_hr', 'bolus_u'] # Define action keys for regression
    
    # Check if all action keys are present in the actions_df
    missing_keys = [key for key in ACTION_KEYS if key not in actions_df.columns]
    if missing_keys:
        print(f"Error: The following action keys are missing in 'advisor_successful_actions.csv': {missing_keys}")
        return
        
    y_all_continuous = actions_df[ACTION_KEYS].values # Extract continuous actions
    
    print(f"Target actions prepared for regression. Shape: {y_all_continuous.shape}")

    # 3. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all_continuous, test_size=0.2, random_state=42
    )
    print(f"Training set: {X_train.shape[0]} samples. Test set: {X_test.shape[0]} samples.")

    if X_train.shape[0] == 0 or X_test.shape[0] == 0:
        print("Error: Not enough data to create training and testing sets after split.")
        return
    
    # 4. Initialize PatternAdvisorAgent for Regression
    advisor_state_dim = X_all.shape[1]
    action_dim = len(ACTION_KEYS)
    
    model_output_base_dir = os.path.join(project_root, "DiaGuardianAI", "models", "pattern_advisor_agent_model")
    os.makedirs(model_output_base_dir, exist_ok=True)
    dummy_repo_path = os.path.join(model_output_base_dir, "dummy_advisor_train_repo_regressor.sqlite") # Updated dummy repo name
    if os.path.exists(dummy_repo_path): os.remove(dummy_repo_path)
    dummy_repository = RepositoryManager(db_path=dummy_repo_path)

    # Parameters for MLPRegressor (can be customized or left to agent's defaults)
    mlp_params_advisor = {
        'hidden_layer_sizes': (128, 64), # Example: larger layers
        'max_iter': 500, # Default in agent
        'random_state': 42,
        'early_stopping': True, # Default in agent
        'n_iter_no_change': 10, # Default in agent
        'solver': 'adam',
        'activation': 'relu'
    }
    
    advisor_agent = PatternAdvisorAgent(
        state_dim=advisor_state_dim,
        pattern_repository=dummy_repository,
        learning_model_type="mlp_regressor", # Changed to regressor
        action_dim=action_dim,               # Added for regressor
        action_keys_ordered=ACTION_KEYS,     # Added for regressor
        model_params=mlp_params_advisor,
        action_space=None ,
        cgm_history_len_for_features=25,  # Adjusted to match state_dim 51
        prediction_horizon_for_features=6, # Adjusted to match state_dim 51
    )

    print(f"\\nTraining the Pattern Advisor's internal regressor model ({advisor_agent.learning_model_type})...")
    advisor_agent.train(X_train, y_train)

    # 5. Evaluate the Model (Regression Metrics)
    if advisor_agent.model is not None and \
       hasattr(advisor_agent.model, 'predict') and \
       X_test.shape[0] > 0:
        print("\\nEvaluating the trained advisor regressor model on the test set...")
        try:
            y_pred = advisor_agent.model.predict(X_test)
            
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            print(f"\\nRegressor Model Evaluation:")
            print(f"  Mean Squared Error (MSE): {mse:.4f}")
            print(f"  R-squared (R2 Score): {r2:.4f}")

            # Optionally, print MSE per action dimension if y_pred is multi-output
            if y_pred.ndim > 1 and y_pred.shape[1] > 1:
                print("  MSE per action dimension:")
                for i, key in enumerate(ACTION_KEYS):
                    dim_mse = mean_squared_error(y_test[:, i], y_pred[:, i])
                    print(f"    {key}: {dim_mse:.4f}")

        except Exception as e:
            print(f"Error during advisor regressor model evaluation: {e}")
    else:
        print("Skipping evaluation as advisor's regressor model is not available/trained or no test data.")

    # 6. Save the Trained Advisor Agent (which saves its internal model and metadata)
    # Using the filename that run_advisor_simulation.py expects, or a new one.
    # Let's use "pattern_advisor_regressor_v1.pkl" as per earlier discussions.
    advisor_model_save_filename = "pattern_advisor_regressor_v1.pkl" 
    advisor_model_save_path = os.path.join(model_output_base_dir, advisor_model_save_filename)
    
    if advisor_agent.is_trained:
        advisor_agent.save(path=advisor_model_save_path)
        print(f"Advisor regressor agent saved to {advisor_model_save_path} (and .meta.json)")
    else:
        print("Advisor regressor agent was not trained. Skipping save.")

    # 7. Conceptual: Load and test regressor
    print(f"\\n--- Conceptual: Loading and using the saved PatternAdvisorAgent (Regressor) from {advisor_model_save_path} ---")
    if not advisor_agent.is_trained:
        print("Skipping load test as regressor agent was not saved.")
    else:
        try:
            loaded_advisor = PatternAdvisorAgent.load_agent_from_files(
                model_path=advisor_model_save_path,
                pattern_repository=dummy_repository 
            )

            if loaded_advisor.model and X_test.shape[0] > 0:
                sample_features_for_advisor = X_test[0]
                try:
                    predicted_actions_loaded = loaded_advisor.predict(sample_features_for_advisor) # predict method handles reshape
                    actual_actions = {key: val for key, val in zip(ACTION_KEYS, y_test[0])}
                    
                    print(f"Prediction from LOADED regressor for first test sample:")
                    print(f"  Predicted Actions: {predicted_actions_loaded}")
                    print(f"  Actual Actions:    {actual_actions}")

                except Exception as e:
                    print(f"Error predicting with loaded advisor regressor model: {e}")
            else:
                print("Loaded advisor's regressor model not available or no test data for conceptual test.")
        except Exception as e:
            print(f"Error loading or testing loaded PatternAdvisorAgent (Regressor): {e}")

    if os.path.exists(dummy_repo_path): # Clean up dummy db
        if dummy_repository and hasattr(dummy_repository, 'conn') and dummy_repository.conn:
             dummy_repository.conn.close()
        try:
            os.remove(dummy_repo_path)
            print(f"Cleaned up dummy repository DB: {dummy_repo_path}")
        except Exception as e:
            print(f"Could not remove dummy_repo_path {dummy_repo_path}: {e}")


    print("\\nPattern Advisor regressor model training script finished.") # Updated message

if __name__ == "__main__":
    train_advisor_regressor_model() # Updated function call