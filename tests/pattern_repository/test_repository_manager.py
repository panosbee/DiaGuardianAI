# tests/pattern_repository/test_repository_manager.py

import pytest
import os
import json
from typing import Dict, Any, List

# Ensure the DiaGuardianAI package is discoverable for imports from the main library
import sys
project_root_for_tests = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root_for_tests not in sys.path:
    sys.path.insert(0, project_root_for_tests)

from DiaGuardianAI.pattern_repository.repository_manager import RepositoryManager

# --- Test Data ---
PATTERN_1_DATA = {"pattern_type": "meal_bolus", "data": {"carbs": 50, "bolus_calc": "standard_cr"}}
PATTERN_1_META = {"source": "initial_rules", "meal_type": "breakfast"}

PATTERN_2_DATA = {"pattern_type": "correction_bolus", "data": {"bg_target": 100, "isf": 50}}
PATTERN_2_META = {"source": "expert_system_v1"}

PATTERN_3_DATA = {"pattern_type": "meal_bolus", "data": {"carbs": 75, "bolus_calc": "aggressive_cr"}}
PATTERN_3_META = {"source": "RL_agent_run_123", "meal_type": "lunch"}


# --- Pytest Fixtures ---
@pytest.fixture
def temp_repo_file(tmp_path) -> str:
    """Creates a temporary file path for saving/loading repository data."""
    return str(tmp_path / "test_patterns.json")

@pytest.fixture
def empty_repo() -> RepositoryManager:
    """Returns an empty RepositoryManager instance."""
    return RepositoryManager()

@pytest.fixture
def repo_with_patterns(empty_repo: RepositoryManager) -> RepositoryManager:
    """Returns a RepositoryManager instance with some pre-added patterns."""
    empty_repo.add_pattern(PATTERN_1_DATA, PATTERN_1_META) # ID 1
    empty_repo.add_pattern(PATTERN_2_DATA, PATTERN_2_META) # ID 2
    empty_repo.add_pattern(PATTERN_3_DATA, PATTERN_3_META) # ID 3
    return empty_repo

# --- Test Cases ---

def test_repository_manager_initialization_empty(empty_repo: RepositoryManager):
    """Test basic initialization of an empty repository."""
    assert isinstance(empty_repo, RepositoryManager)
    assert empty_repo.patterns == []
    assert empty_repo.next_pattern_id == 1
    assert empty_repo.file_path is None

def test_repository_manager_initialization_with_non_existent_file(temp_repo_file: str):
    """Test initialization with a non-existent file path."""
    repo = RepositoryManager(file_path=temp_repo_file)
    assert repo.patterns == []
    assert repo.next_pattern_id == 1
    assert repo.file_path == temp_repo_file

def test_add_pattern(empty_repo: RepositoryManager):
    """Test adding patterns to the repository."""
    empty_repo.add_pattern(PATTERN_1_DATA, PATTERN_1_META)
    assert len(empty_repo.patterns) == 1
    assert empty_repo.patterns[0]["id"] == 1
    assert empty_repo.patterns[0]["pattern_type"] == PATTERN_1_DATA["pattern_type"]
    assert empty_repo.patterns[0]["data"] == PATTERN_1_DATA["data"]
    assert empty_repo.patterns[0]["metadata"] == PATTERN_1_META
    assert empty_repo.patterns[0]["effectiveness_score"] == 0.0
    assert empty_repo.patterns[0]["usage_count"] == 0
    assert empty_repo.next_pattern_id == 2

    empty_repo.add_pattern(PATTERN_2_DATA, PATTERN_2_META)
    assert len(empty_repo.patterns) == 2
    assert empty_repo.patterns[1]["id"] == 2
    assert empty_repo.next_pattern_id == 3

def test_retrieve_relevant_patterns_empty_repo(empty_repo: RepositoryManager):
    """Test retrieving from an empty repository."""
    retrieved = empty_repo.retrieve_relevant_patterns({"pattern_type": "meal_bolus"})
    assert retrieved == []

def test_retrieve_relevant_patterns_no_filter(repo_with_patterns: RepositoryManager):
    """Test retrieving without specific filters (should sort by score then ID desc)."""
    # Add a pattern with a higher score to test sorting
    repo_with_patterns.add_pattern({"pattern_type": "test_type", "data": {}}, {"source": "test"}) # ID 4
    repo_with_patterns.update_pattern_effectiveness(pattern_id=1, new_outcome_data={"reward": 0.8})
    repo_with_patterns.update_pattern_effectiveness(pattern_id=4, new_outcome_data={"reward": 0.9})


    retrieved = repo_with_patterns.retrieve_relevant_patterns({}, n_top_patterns=2)
    assert len(retrieved) == 2
    assert retrieved[0]["id"] == 4 # Highest score
    assert retrieved[1]["id"] == 1 # Next highest score

    # Test default retrieval (most recent if scores are same)
    repo_fresh = RepositoryManager()
    repo_fresh.add_pattern(PATTERN_1_DATA) # ID 1
    repo_fresh.add_pattern(PATTERN_2_DATA) # ID 2
    retrieved_recent = repo_fresh.retrieve_relevant_patterns({}, n_top_patterns=1)
    assert retrieved_recent[0]["id"] == 2 # ID 2 is more recent than ID 1

def test_retrieve_relevant_patterns_by_type(repo_with_patterns: RepositoryManager):
    """Test retrieving patterns filtered by type."""
    retrieved_meal = repo_with_patterns.retrieve_relevant_patterns({"pattern_type": "meal_bolus"}, n_top_patterns=2)
    assert len(retrieved_meal) == 2
    assert all(p["pattern_type"] == "meal_bolus" for p in retrieved_meal)
    # Default sort is by score (0.0 for all here), then ID descending
    assert retrieved_meal[0]["id"] == 3
    assert retrieved_meal[1]["id"] == 1

    retrieved_correction = repo_with_patterns.retrieve_relevant_patterns({"pattern_type": "correction_bolus"})
    assert len(retrieved_correction) == 1
    assert retrieved_correction[0]["pattern_type"] == "correction_bolus"
    assert retrieved_correction[0]["id"] == 2

    retrieved_non_existent = repo_with_patterns.retrieve_relevant_patterns({"pattern_type": "non_existent_type"})
    assert retrieved_non_existent == []

def test_update_pattern_effectiveness(repo_with_patterns: RepositoryManager):
    """Test updating the effectiveness score and usage count of a pattern."""
    pattern_id_to_update = 1
    initial_pattern = next(p for p in repo_with_patterns.patterns if p["id"] == pattern_id_to_update)
    assert initial_pattern["effectiveness_score"] == 0.0
    assert initial_pattern["usage_count"] == 0

    repo_with_patterns.update_pattern_effectiveness(pattern_id_to_update, {"reward": 0.9, "notes": "Good"})
    updated_pattern = next(p for p in repo_with_patterns.patterns if p["id"] == pattern_id_to_update)
    assert updated_pattern["effectiveness_score"] == pytest.approx(0.9)
    assert updated_pattern["usage_count"] == 1
    assert updated_pattern["metadata"]["last_outcome_notes"] == "Good"

    # Second update to test averaging
    repo_with_patterns.update_pattern_effectiveness(pattern_id_to_update, {"effectiveness_score": 0.7})
    updated_pattern_2 = next(p for p in repo_with_patterns.patterns if p["id"] == pattern_id_to_update)
    # (0.9 * 1 + 0.7) / 2 = 0.8
    assert updated_pattern_2["effectiveness_score"] == pytest.approx(0.8)
    assert updated_pattern_2["usage_count"] == 2

    # Test updating non-existent pattern (should not error, just print)
    repo_with_patterns.update_pattern_effectiveness(999, {"reward": 0.5}) # No assertion, just ensure no crash

def test_save_and_load_patterns(repo_with_patterns: RepositoryManager, temp_repo_file: str):
    """Test saving patterns to a file and loading them back."""
    repo_with_patterns.update_pattern_effectiveness(1, {"reward": 0.85}) # Change a score
    original_patterns = [dict(p) for p in repo_with_patterns.patterns] # Deep copy for comparison
    original_next_id = repo_with_patterns.next_pattern_id

    repo_with_patterns.save_patterns(temp_repo_file)
    assert os.path.exists(temp_repo_file)

    # Load into a new repository instance
    new_repo = RepositoryManager(file_path=temp_repo_file) # Should auto-load
    
    assert len(new_repo.patterns) == len(original_patterns)
    assert new_repo.next_pattern_id == original_next_id
    
    # Check if patterns are identical (order might change if not sorted before save, but IDs are key)
    # For simplicity, let's check one pattern's details
    loaded_pattern_1 = next(p for p in new_repo.patterns if p["id"] == 1)
    original_pattern_1 = next(p for p in original_patterns if p["id"] == 1)
    assert loaded_pattern_1["pattern_type"] == original_pattern_1["pattern_type"]
    assert loaded_pattern_1["data"] == original_pattern_1["data"]
    assert loaded_pattern_1["metadata"] == original_pattern_1["metadata"]
    assert loaded_pattern_1["effectiveness_score"] == pytest.approx(original_pattern_1["effectiveness_score"])

    # Test explicit load method
    repo_explicit_load = RepositoryManager()
    repo_explicit_load.load_patterns(temp_repo_file)
    assert len(repo_explicit_load.patterns) == len(original_patterns)
    assert repo_explicit_load.next_pattern_id == original_next_id


def test_load_patterns_non_existent_file(empty_repo: RepositoryManager, temp_repo_file: str):
    """Test loading from a non-existent file (should result in empty repo, no error)."""
    # Ensure file does not exist (though tmp_path fixture usually handles this)
    if os.path.exists(temp_repo_file):
        os.remove(temp_repo_file)
    
    empty_repo.load_patterns(temp_repo_file) # Should print warning but not raise FileNotFoundError
    assert empty_repo.patterns == []
    assert empty_repo.next_pattern_id == 1

def test_save_patterns_no_path_at_init_or_call(empty_repo: RepositoryManager):
    """Test save_patterns when no path was given at init and none at call."""
    # This should print an error and not raise, as per current implementation
    empty_repo.save_patterns() 
    # No assertion needed other than it doesn't crash. Output can be checked manually if needed.

def test_load_patterns_no_path_at_init_or_call(empty_repo: RepositoryManager):
    """Test load_patterns when no path was given at init and none at call."""
    # This should print an error and not raise, as per current implementation
    empty_repo.load_patterns()
    # No assertion needed other than it doesn't crash.

def test_next_pattern_id_management_after_load(repo_with_patterns: RepositoryManager, temp_repo_file: str):
    """Test that next_pattern_id is correctly managed after loading and adding new patterns."""
    # repo_with_patterns has 3 patterns, next_id is 4
    assert repo_with_patterns.next_pattern_id == 4
    
    repo_with_patterns.save_patterns(temp_repo_file)
    
    loaded_repo = RepositoryManager(file_path=temp_repo_file)
    assert loaded_repo.next_pattern_id == 4 # Should be restored
    
    # Add a new pattern to the loaded repository
    loaded_repo.add_pattern({"pattern_type": "new_type", "data": {}}, {})
    assert len(loaded_repo.patterns) == 4
    assert loaded_repo.patterns[-1]["id"] == 4 # ID of the new pattern
    assert loaded_repo.next_pattern_id == 5 # next_id should increment

    # Test scenario where loaded next_pattern_id from file is lower than max_id in patterns + 1
    # (e.g., if file was manually edited or from an older version)
    corrupted_data = {
        "next_pattern_id": 1, # Intentionally low
        "patterns": [
            {"id": 1, "pattern_type": "typeA", "data": {}},
            {"id": 5, "pattern_type": "typeB", "data": {}} # Max ID is 5
        ]
    }
    with open(temp_repo_file, 'w') as f:
        json.dump(corrupted_data, f)
    
    repo_corrupt_load = RepositoryManager(file_path=temp_repo_file)
    # next_pattern_id should be max(loaded_next_id, max_id_in_patterns + 1, 1)
    # max(1, 5 + 1, 1) = 6
    assert repo_corrupt_load.next_pattern_id == 6
    repo_corrupt_load.add_pattern({"pattern_type": "typeC", "data": {}}, {})
    assert repo_corrupt_load.patterns[-1]["id"] == 6
    assert repo_corrupt_load.next_pattern_id == 7

def test_load_empty_json_file(temp_repo_file: str):
    """Test loading an empty or malformed JSON file."""
    # Empty JSON array
    with open(temp_repo_file, 'w') as f:
        f.write("[]") # Not the expected structure
    repo1 = RepositoryManager(file_path=temp_repo_file)
    assert repo1.patterns == [] # Should default to empty
    assert repo1.next_pattern_id == 1

    # Malformed JSON
    with open(temp_repo_file, 'w') as f:
        f.write("{,}")
    repo2 = RepositoryManager(file_path=temp_repo_file)
    assert repo2.patterns == [] # Should default to empty on JSONDecodeError
    assert repo2.next_pattern_id == 1

    # JSON with missing keys
    with open(temp_repo_file, 'w') as f:
        json.dump({"patterns": [{"id":1, "data":{}}]}, f) # Missing next_pattern_id
    repo3 = RepositoryManager(file_path=temp_repo_file)
    assert len(repo3.patterns) == 1
    assert repo3.next_pattern_id == 2 # Should be 1 (from pattern) + 1

    with open(temp_repo_file, 'w') as f:
        json.dump({"next_pattern_id": 5}, f) # Missing patterns list
    repo4 = RepositoryManager(file_path=temp_repo_file)
    assert repo4.patterns == []
    assert repo4.next_pattern_id == 5 # Should use the one from file if patterns list is missing/empty