\
import sys
import os
import joblib

# Ensure the DiaGuardianAI package is discoverable
if __package__ is None or __package__ == \'\':
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), \'..\', \'..\'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from DiaGuardianAI.agents.pattern_advisor_agent import PatternAdvisorAgent
from DiaGuardianAI.pattern_repository.repository_manager import RepositoryManager

# --- Configuration for the existing model ---
MODEL_FILENAME = "pattern_advisor_supervised.joblib"
MODEL_DIR = os.path.join(project_root, "DiaGuardianAI", "models", "pattern_advisor_agent_model")
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

# These should match the properties of your pre-trained pattern_advisor_supervised.joblib
# Taken from ASSUMED_ values in run_advisor_simulation.py
STATE_DIM = 51
ACTION_DIM = 2
ACTION_KEYS_ORDERED = [\'basal_rate_u_hr\', \'bolus_u\']
LEARNING_MODEL_TYPE = "mlp_regressor"  # IMPORTANT: Change if your model is GradientBoostingRegressor or other
MODEL_PARAMS = {} # Add any specific params used to initialize the raw model if known, otherwise defaults will be used by agent

def wrap_and_resave_model():
    print(f"--- Wrapping and re-saving model: {MODEL_PATH} ---")

    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file not found at {MODEL_PATH}. Cannot proceed.")
        return

    # 1. Load the raw scikit-learn model
    try:
        raw_sklearn_model = joblib.load(MODEL_PATH)
        print(f"Successfully loaded raw model from {MODEL_PATH}")
        print(f"Raw model type: {type(raw_sklearn_model)}")
    except Exception as e:
        print(f"ERROR: Failed to load raw model from {MODEL_PATH}: {e}")
        return

    # 2. Create a dummy pattern repository (required by PatternAdvisorAgent constructor)
    dummy_repo_db_path = os.path.join(MODEL_DIR, "dummy_wrapper_repo.sqlite")
    if os.path.exists(dummy_repo_db_path):
        os.remove(dummy_repo_db_path)
    dummy_repository = RepositoryManager(db_path=dummy_repo_db_path)
    print(f"Initialized dummy repository at {dummy_repo_db_path}")

    # 3. Instantiate PatternAdvisorAgent
    try:
        advisor_agent = PatternAdvisorAgent(
            state_dim=STATE_DIM,
            action_dim=ACTION_DIM,
            action_keys_ordered=ACTION_KEYS_ORDERED,
            pattern_repository=dummy_repository,
            learning_model_type=LEARNING_MODEL_TYPE,
            model_params=MODEL_PARAMS  # Pass model_params here
        )
        print(f"PatternAdvisorAgent instantiated with type: {LEARNING_MODEL_TYPE}")
    except Exception as e:
        print(f"ERROR: Failed to instantiate PatternAdvisorAgent: {e}")
        if os.path.exists(dummy_repo_db_path): os.remove(dummy_repo_db_path)
        return

    # 4. Assign the loaded raw model to the agent and set as trained
    # The agent's _build_model() will be called internally if model is None,
    # but we are replacing it with our pre-trained one.
    # If _build_model() was already called and created a new model, this overwrites it.
    advisor_agent.model = raw_sklearn_model
    advisor_agent.is_trained = True # Mark as trained
    print("Assigned raw model to agent and marked as trained.")

    # 5. Save the agent using its save method (this will create the .meta.json file)
    try:
        # Saving to the same path will overwrite the .joblib and create the .meta.json
        advisor_agent.save(path=MODEL_PATH)
        print(f"Agent saved successfully to {MODEL_PATH} (and .meta.json should now exist).")
    except Exception as e:
        print(f"ERROR: Failed to save agent: {e}")
    finally:
        if dummy_repository and hasattr(dummy_repository, \'conn\') and dummy_repository.conn:
            dummy_repository.conn.close()
        if os.path.exists(dummy_repo_db_path):
            try:
                os.remove(dummy_repo_db_path)
            except Exception as e_del:
                print(f"Error deleting dummy repo DB: {e_del}")

if __name__ == "__main__":
    wrap_and_resave_model()
