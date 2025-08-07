import pytest
import os
from DiaGuardianAI.sdk.diabetes_manager import DiabetesManager
from DiaGuardianAI.sdk.data_types import InsulinRecommendation, SafetyLevel

@pytest.fixture(scope="module")
def diabetes_manager_instance():
    """
    Fixture to create a single instance of the DiabetesManager for the
    entire test module. This is crucial because the initialization
    process is computationally expensive (trains AI models).
    """
    print("\n--- Initializing DiabetesManager for integration test ---")
    # Set train_on_init=True to run the full pipeline
    manager = DiabetesManager(config={}, train_on_init=True)
    # Ensure the pattern DB file is cleaned up after tests
    yield manager
    print("\n--- Tearing down DiabetesManager instance ---")
    db_path = "DiaGuardianAI_patterns.sqlite"
    if os.path.exists(db_path):
        # This part might fail if the DB is still in use.
        # A more robust solution would be to close the DB connection in the manager.
        # For now, we'll try to remove it.
        try:
            # A graceful close method would be better.
            # For now, we assume the object is destroyed and the file is free.
            del manager
            os.remove(db_path)
            print(f"Cleaned up database file: {db_path}")
        except Exception as e:
            print(f"Could not clean up database file {db_path}: {e}")

def test_full_system_initialization(diabetes_manager_instance):
    """
    Tests if the DiabetesManager and its underlying AI systems initialize
    without errors.
    """
    print("--- Test: Full System Initialization ---")
    manager = diabetes_manager_instance
    assert manager is not None, "DiabetesManager instance should not be None"
    assert manager.insulin_advisor is not None, "InsulinAdvisor should be initialized"
    assert manager.insulin_advisor.decision_system is not None, "Decision system should be initialized"
    assert manager.insulin_advisor.decision_system.loop_active, "Learning loop should be active"
    print("Assertions passed: Manager and AI subsystems initialized successfully.")

def test_get_insulin_recommendation_end_to_end(diabetes_manager_instance):
    """
    Tests the full end-to-end process of getting an insulin recommendation.
    """
    print("--- Test: End-to-End Insulin Recommendation ---")
    manager = diabetes_manager_instance

    # 1. Add a patient
    print("Step 1: Adding a new patient")
    patient_id = manager.add_patient(
        diabetes_type="type_1",
        age=30,
        weight_kg=75,
        isf=45,
        cr=10,
        basal_rate=1.1
    )
    assert patient_id is not None
    print(f"Patient added with ID: {patient_id}")

    # 2. Get an insulin recommendation
    print("Step 2: Requesting insulin recommendation")
    recommendation = manager.get_insulin_recommendation(
        patient_id=patient_id,
        current_glucose=150,
        meal_carbs=50,
        current_iob=0.5
    )
    print(f"Recommendation received: {recommendation}")

    # 3. Assert the recommendation is valid
    print("Step 3: Validating the recommendation object")
    assert isinstance(recommendation, InsulinRecommendation), "Should return an InsulinRecommendation object"
    assert recommendation.patient_id == patient_id, "Patient ID should match"
    assert isinstance(recommendation.basal_rate_u_hr, float), "Basal rate should be a float"
    assert isinstance(recommendation.bolus_amount_u, float), "Bolus amount should be a float"
    assert recommendation.basal_rate_u_hr >= 0, "Basal rate should be non-negative"
    assert recommendation.bolus_amount_u >= 0, "Bolus amount should be non-negative"
    assert isinstance(recommendation.explanation, str) and recommendation.explanation, "Explanation should be a non-empty string"
    assert isinstance(recommendation.safety_level, SafetyLevel), "Safety level should be a SafetyLevel enum"
    assert recommendation.confidence >= 0 and recommendation.confidence <= 1, "Confidence should be between 0 and 1"
    print("Assertions passed: Recommendation object is valid and well-formed.")

def test_system_status_endpoint(diabetes_manager_instance):
    """
    Tests the get_system_status method to ensure it returns a comprehensive status.
    """
    print("--- Test: System Status Endpoint ---")
    manager = diabetes_manager_instance
    status = manager.get_system_status()
    print(f"System status: {status}")

    assert isinstance(status, dict)
    assert status['is_operational'] is True
    assert 'ai_system_status' in status
    ai_status = status['ai_system_status']
    assert 'learning_loop' in ai_status
    assert 'transformer_zoo' in ai_status
    assert 'agent1' in ai_status
    assert 'agent2' in ai_status
    assert ai_status['learning_loop']['active'] is True
    print("Assertions passed: System status is comprehensive and reports AI system health.")
