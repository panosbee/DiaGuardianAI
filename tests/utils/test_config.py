# Tests for DiaGuardianAI.utils.config

import pytest
import os
import yaml
import json
from DiaGuardianAI.utils.config import load_config, get_config_value, ConfigManager, DEFAULT_CONFIG_FILENAME

@pytest.fixture
def dummy_yaml_config_file(tmp_path):
    content = {
        "simulation": {
            "duration_days": 7,
            "patient": {"ISF": 50, "CR": 10}
        },
        "agent": {"type": "RLAgent", "lr": 0.001}
    }
    file_path = tmp_path / "test_config.yaml"
    with open(file_path, "w") as f:
        yaml.dump(content, f)
    return file_path

@pytest.fixture
def dummy_json_config_file(tmp_path):
    content = {
        "simulation": {
            "duration_days": 14,
            "patient": {"ISF": 45}
        },
        "agent": {"type": "RuleBased"}
    }
    file_path = tmp_path / "test_config.json"
    with open(file_path, "w") as f:
        json.dump(content, f)
    return file_path

def test_load_config_yaml(dummy_yaml_config_file):
    """Test loading from a YAML file."""
    config = load_config(str(dummy_yaml_config_file))
    assert config["simulation"]["duration_days"] == 7
    assert config["agent"]["lr"] == 0.001
    print("test_load_config_yaml: PASSED")

def test_load_config_json(dummy_json_config_file):
    """Test loading from a JSON file."""
    config = load_config(str(dummy_json_config_file))
    assert config["simulation"]["duration_days"] == 14
    assert config["agent"]["type"] == "RuleBased"
    print("test_load_config_json: PASSED")

def test_load_config_non_existent_file():
    """Test loading a non-existent file returns empty dict."""
    config = load_config("non_existent_config_file.yaml")
    assert config == {}
    print("test_load_config_non_existent_file: PASSED")

def test_load_config_unknown_format(tmp_path):
    """Test loading an unknown file format."""
    file_path = tmp_path / "test_config.txt"
    with open(file_path, "w") as f:
        f.write("some_setting = value")
    config = load_config(str(file_path))
    assert config == {}
    print("test_load_config_unknown_format: PASSED")

def test_load_config_default_file(dummy_yaml_config_file, tmp_path, monkeypatch):
    """Test loading default config file if no path is provided."""
    # Create a default config file in a temporary current directory
    default_file_path = tmp_path / DEFAULT_CONFIG_FILENAME
    content = {"default_setting": True}
    with open(default_file_path, "w") as f:
        yaml.dump(content, f)

    # Monkeypatch current working directory to tmp_path
    monkeypatch.chdir(tmp_path)
    config = load_config() # No path, should find default
    assert config.get("default_setting") is True
    print("test_load_config_default_file: PASSED")


def test_get_config_value():
    """Test retrieving values using dot-separated keys."""
    config = {
        "level1": {
            "level2_item": "value1",
            "level2_dict": {"level3_item": "value2"}
        },
        "top_item": "value3"
    }
    assert get_config_value(config, "top_item") == "value3"
    assert get_config_value(config, "level1.level2_item") == "value1"
    assert get_config_value(config, "level1.level2_dict.level3_item") == "value2"
    assert get_config_value(config, "level1.non_existent", "default") == "default"
    assert get_config_value(config, "non.existent.path", "another_default") == "another_default"
    assert get_config_value(config, "level1.level2_dict.missing", None) is None
    print("test_get_config_value: PASSED")

def test_config_manager_initialization(dummy_yaml_config_file):
    """Test ConfigManager initialization."""
    manager = ConfigManager(str(dummy_yaml_config_file))
    assert manager.config_data["simulation"]["duration_days"] == 7
    
    manager_no_file = ConfigManager() # No file provided
    assert manager_no_file.config_data == {} # Expects empty if default not found
    print("test_config_manager_initialization: PASSED")

def test_config_manager_get(dummy_yaml_config_file):
    """Test ConfigManager.get() method."""
    manager = ConfigManager(str(dummy_yaml_config_file))
    assert manager.get("simulation.duration_days") == 7
    assert manager.get("simulation.patient.ISF") == 50
    assert manager.get("agent.type") == "RLAgent"
    assert manager.get("non.existent.key", "default_val") == "default_val"
    print("test_config_manager_get: PASSED")

def test_config_manager_get_section(dummy_yaml_config_file):
    """Test ConfigManager.get_section() method."""
    manager = ConfigManager(str(dummy_yaml_config_file))
    sim_patient_section = manager.get_section("simulation.patient")
    assert isinstance(sim_patient_section, dict)
    assert sim_patient_section["ISF"] == 50
    assert sim_patient_section["CR"] == 10

    agent_section = manager.get_section("agent")
    assert agent_section["type"] == "RLAgent"
    
    non_existent_section = manager.get_section("non.existent")
    assert non_existent_section == {}
    
    # Test getting a non-dict item as section
    not_a_section = manager.get_section("agent.type")
    assert not_a_section == {}
    print("test_config_manager_get_section: PASSED")

def test_config_manager_reload(dummy_yaml_config_file, dummy_json_config_file, tmp_path, monkeypatch):
    """Test ConfigManager.reload() method."""
    manager = ConfigManager(str(dummy_yaml_config_file))
    assert manager.get("simulation.duration_days") == 7

    # Reload with a new JSON file
    manager.reload(str(dummy_json_config_file))
    assert manager.get("simulation.duration_days") == 14 # From JSON
    assert manager.get("simulation.patient.ISF") == 45   # From JSON
    assert manager.get("agent.type") == "RuleBased"      # From JSON

    # Test reload without path (should try default logic, which might be empty if default not set up)
    # To test this properly, set up a default file, then reload without path
    default_file_path = tmp_path / DEFAULT_CONFIG_FILENAME
    default_content = {"reloaded_default": True}
    with open(default_file_path, "w") as f:
        yaml.dump(default_content, f)
    
    monkeypatch.chdir(tmp_path) # Change CWD to where default file is
    manager.reload(str(default_file_path)) # Explicitly reload the default file by path
    assert manager.get("reloaded_default") is True
    print("test_config_manager_reload: PASSED")