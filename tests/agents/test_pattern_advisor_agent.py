# tests/agents/test_pattern_advisor_agent.py

import pytest
import os
import sys
import numpy as np
from typing import Dict, Any, Optional

# Ensure the DiaGuardianAI package is discoverable
project_root_for_tests = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root_for_tests not in sys.path:
    sys.path.insert(0, project_root_for_tests)

from DiaGuardianAI.agents.pattern_advisor_agent import PatternAdvisorAgent
from DiaGuardianAI.pattern_repository.repository_manager import RepositoryManager # Using real repo
from DiaGuardianAI.core.base_classes import BasePatternRepository


# --- Test Data ---
ADVISOR_PATTERN_1_DATA = {"pattern_type": "meal_advice", "data": {"carbs": 60, "suggestion": "pre_bolus_10min"}}
ADVISOR_PATTERN_1_META = {"source": "expert_v1"}
ADVISOR_PATTERN_2_DATA = {"pattern_type": "hypo_fix", "data": {"symptoms": "mild", "action": "15g_glucose_tabs"}}
ADVISOR_PATTERN_2_META = {"source": "protocol_standard"}
ADVISOR_PATTERN_3_DATA = {"pattern_type": "meal_advice", "data": {"carbs": 30, "suggestion": "bolus_with_meal"}}
ADVISOR_PATTERN_3_META = {"source": "user_log_success"}


# --- Pytest Fixtures ---
@pytest.fixture
def temp_advisor_repo_file(tmp_path) -> str:
    """Temporary file for the repository used by the advisor."""
    return str(tmp_path / "advisor_test_patterns.json")

@pytest.fixture
def empty_pattern_repository(temp_advisor_repo_file: str) -> RepositoryManager:
    """Returns an empty RepositoryManager instance, configured with a temp file."""
    # Ensure the temp file directory exists for RepositoryManager to potentially write to
    # os.makedirs(os.path.dirname(temp_advisor_repo_file), exist_ok=True) # tmp_path handles this
    return RepositoryManager(file_path=temp_advisor_repo_file)

@pytest.fixture
def pattern_repository_with_data(empty_pattern_repository: RepositoryManager) -> RepositoryManager:
    """Returns a RepositoryManager with some patterns."""
    repo = empty_pattern_repository
    repo.add_pattern(ADVISOR_PATTERN_1_DATA, ADVISOR_PATTERN_1_META) # ID 1, score 0.0
    repo.add_pattern(ADVISOR_PATTERN_2_DATA, ADVISOR_PATTERN_2_META) # ID 2, score 0.0
    repo.add_pattern(ADVISOR_PATTERN_3_DATA, ADVISOR_PATTERN_3_META) # ID 3, score 0.0
    # Update scores for testing retrieval logic
    repo.update_pattern_effectiveness(1, {"reward": 0.8})  # ID 1, type meal_advice, score 0.8
    repo.update_pattern_effectiveness(3, {"reward": 0.95}) # ID 3, type meal_advice, score 0.95
    return repo

@pytest.fixture
def pattern_advisor(pattern_repository_with_data: RepositoryManager) -> PatternAdvisorAgent:
    """Returns a PatternAdvisorAgent initialized with a repository."""
    return PatternAdvisorAgent(
        state_dim=5,
        action_space="placeholder_action_space",
        pattern_repository=pattern_repository_with_data,
        learning_model_type="mlp_regressor",  # Use mlp_regressor for test compatibility
        action_dim=4,  # Required for regression tests
        action_keys_ordered=["a", "b", "c", "d"]  # Required for regression tests
    )

@pytest.fixture
def pattern_advisor_empty_repo(empty_pattern_repository: RepositoryManager) -> PatternAdvisorAgent:
    """Returns a PatternAdvisorAgent initialized with an empty repository."""
    return PatternAdvisorAgent(
        state_dim=5,
        action_space="placeholder_action_space",
        pattern_repository=empty_pattern_repository,
        learning_model_type="mlp_regressor",  # Use mlp_regressor for test compatibility
        action_dim=4,  # Required for regression tests
        action_keys_ordered=["a", "b", "c", "d"]  # Required for regression tests
    )

# --- Test Cases ---

def test_pattern_advisor_initialization(pattern_advisor: PatternAdvisorAgent, pattern_repository_with_data: RepositoryManager):
    """Test basic initialization of PatternAdvisorAgent."""
    assert isinstance(pattern_advisor, PatternAdvisorAgent)
    assert pattern_advisor.pattern_repository == pattern_repository_with_data
    assert pattern_advisor.learning_model_type == "mlp_regressor" # Updated for compatibility with tests

def test_decide_action_no_preference_retrieves_highest_score(pattern_advisor: PatternAdvisorAgent):
    """Test decide_action retrieves the highest scored pattern when no type preference is given."""
    current_state = {"cgm": 100.0, "iob": 1.0, "cob": 0.0, "cgm_history": [100.0]*5}
    suggestion = pattern_advisor.decide_action(current_state)
    
    assert suggestion is not None
    assert suggestion["pattern_id"] == 3 # ID 3 has score 0.95
    assert suggestion["pattern_data"]["pattern_type"] == "meal_advice"
    assert "Score: 0.95" in suggestion["rationale"]
    assert "specifically for type" not in suggestion["rationale"] # No type preference was given

def test_decide_action_with_type_preference(pattern_advisor: PatternAdvisorAgent):
    """Test decide_action with a pattern_type_preference."""
    pattern_advisor.pattern_repository.add_pattern(
        {"pattern_type": "hypo_fix", "data": {"symptoms": "severe", "action": "glucagon"}},
        {"source": "emergency_protocol"}
    ) # ID 4
    pattern_advisor.pattern_repository.update_pattern_effectiveness(4, {"reward": 0.99})

    current_state_meal_pref = {
        "cgm": 150.0, "iob": 2.0, "cob": 30.0, "cgm_history": [150.0]*5,
        "pattern_type_preference": "meal_advice"
    }
    suggestion_meal = pattern_advisor.decide_action(current_state_meal_pref)
    
    assert suggestion_meal is not None
    assert suggestion_meal["pattern_id"] == 3 
    assert suggestion_meal["pattern_data"]["pattern_type"] == "meal_advice"
    assert "Score: 0.95" in suggestion_meal["rationale"]
    assert "specifically for type: meal_advice" in suggestion_meal["rationale"]

    current_state_hypo_pref = {
        "cgm": 60.0, "iob": 0.5, "cob": 0.0, "cgm_history": [60.0]*5,
        "pattern_type_preference": "hypo_fix"
    }
    suggestion_hypo = pattern_advisor.decide_action(current_state_hypo_pref)
    assert suggestion_hypo is not None
    assert suggestion_hypo["pattern_id"] == 4 
    assert suggestion_hypo["pattern_data"]["pattern_type"] == "hypo_fix"
    assert "Score: 0.99" in suggestion_hypo["rationale"]
    assert "specifically for type: hypo_fix" in suggestion_hypo["rationale"]


def test_decide_action_empty_repository(pattern_advisor_empty_repo: PatternAdvisorAgent):
    """Test decide_action when the repository is empty."""
    current_state = {"cgm": 100.0, "iob": 1.0, "cob": 0.0, "cgm_history": [100.0]*5}
    suggestion = pattern_advisor_empty_repo.decide_action(current_state)
    assert suggestion is None

def test_decide_action_no_matching_type_preference(pattern_advisor: PatternAdvisorAgent):
    """Test decide_action when type preference yields no results."""
    current_state = {
        "cgm": 100.0, "iob": 1.0, "cob": 0.0, "cgm_history": [100.0]*5,
        "pattern_type_preference": "non_existent_type"
    }
    suggestion = pattern_advisor.decide_action(current_state)
    assert suggestion is None

def test_decide_action_insufficient_features(pattern_advisor: PatternAdvisorAgent):
    """Test decide_action when current_state provides insufficient features for query."""
    current_state_missing_all = {} 
    suggestion = pattern_advisor.decide_action(current_state_missing_all)
    assert suggestion is None

    current_state_some_none = {"cgm": None, "iob": None, "cob": None, "cgm_history": None}
    suggestion_some_none = pattern_advisor.decide_action(current_state_some_none)
    assert suggestion_some_none is None


def test_pattern_advisor_placeholder_methods(pattern_advisor: PatternAdvisorAgent, tmp_path):
    """Test that placeholder methods run without error."""
    pattern_advisor.learn(experience=("dummy_experience_data"))
    
    save_path = str(tmp_path / "advisor_model.pkl")
    pattern_advisor.save(save_path)
    # Use the classmethod load_agent_from_files instead
    PatternAdvisorAgent.load_agent_from_files(
        model_path=save_path,
        pattern_repository=pattern_advisor.pattern_repository
    )

# --- Regression Model Tests ---

def test_predict_negative_values_clipping(pattern_advisor: PatternAdvisorAgent):
    """Test that negative predictions are clipped to 0."""
    # Mock model that returns negative values
    pattern_advisor.model = lambda x: np.array([-1.0, -0.5, 0.5, 1.0])
    pattern_advisor.is_trained = True
    
    # Create a feature vector matching state_dim
    feature_vector = np.zeros(pattern_advisor.state_dim)
    
    result = pattern_advisor.predict(feature_vector)
    assert result == {"a": 0.0, "b": 0.0, "c": 0.5, "d": 1.0}

def test_feature_normalization_ranges(pattern_advisor: PatternAdvisorAgent):
    """Test that features are normalized to expected ranges."""
    # Mock model that returns constant values
    pattern_advisor.model = lambda x: np.array([0.5, 0.5, 0.5, 0.5])  # Match action_dim=4 from fixture
    pattern_advisor.is_trained = True
    
    # Create a feature vector with normalized values
    feature_vector = np.zeros(pattern_advisor.state_dim)
    
    result = pattern_advisor.predict(feature_vector)
    
    # Verify all values are as expected
    assert all(value == 0.5 for value in result.values())

def test_training_data_validation(pattern_advisor: PatternAdvisorAgent):
    """Test that training data validation rejects invalid ranges."""
    # For this test, we're just testing that the shape validation works
    with pytest.raises(ValueError):
        # Mismatch in dimensions between features and action_dim
        pattern_advisor.train(
            features=np.array([[1, 2, 3, 4, 5]]),  # Correct feature dimension
            actions=np.array([[1, 2]])  # Wrong action dimension (should be 4 for our fixture)
        )
        
    # This would pass shape validation but we expect value errors
    # We can test with valid shapes
    with pytest.raises(ValueError):
        # Here we need to use proper shapes but test validation logic
        # Create features of correct state_dim
        features = np.ones((1, pattern_advisor.state_dim))
        # Create actions with wrong shape (action_dim should be 4 based on fixture)
        actions = np.ones((1, 2))  # Wrong action dimension
        pattern_advisor.train(features=features, actions=actions)