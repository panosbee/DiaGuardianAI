# DiaGuardianAI Configuration Management
# Handles loading and accessing configuration parameters for the library.

import yaml
import json
from typing import Dict, Any, Optional
import os

DEFAULT_CONFIG_FILENAME = "diaguardian_config.yaml"  # Default config filename to look for


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Loads configuration parameters from a YAML or JSON file.

    If `config_path` is not provided, this function will attempt to load
    from a file named `DEFAULT_CONFIG_FILENAME` in the current working
    directory. If the specified file (or default) is not found, or if
    an error occurs during loading (e.g., malformed file), an empty
    dictionary is returned along with a warning.

    Args:
        config_path (Optional[str]): The full path to the configuration
            file. Supports `.yaml`, `.yml`, and `.json` extensions. If
            None, attempts to load `DEFAULT_CONFIG_FILENAME` from the
            current directory.

    Returns:
        Dict[str, Any]: A dictionary containing the loaded configuration
            parameters. Returns an empty dictionary if loading fails or
            no file is found.
    """
    resolved_path = config_path
    if resolved_path is None:
        if os.path.exists(DEFAULT_CONFIG_FILENAME):
            resolved_path = DEFAULT_CONFIG_FILENAME
            print(
                f"No config path provided, using default: "
                f"'{DEFAULT_CONFIG_FILENAME}' in CWD."
            )
        else:
            print(
                f"No config path provided and default "
                f"'{DEFAULT_CONFIG_FILENAME}' not found in CWD. "
                f"Returning empty config."
            )
            return {}

    if not os.path.exists(resolved_path):
        print(
            f"Warning: Configuration file not found at '{resolved_path}'. "
            f"Returning empty config."
        )
        return {}

    try:
        with open(resolved_path, 'r', encoding='utf-8') as f:
            if resolved_path.endswith((".yaml", ".yml")):
                config_data = yaml.safe_load(f)
            elif resolved_path.endswith(".json"):
                config_data = json.load(f)
            else:
                print(
                    f"Warning: Unknown config file format for '{resolved_path}'. "
                    f"Supported: .yaml, .yml, .json. Returning empty config."
                )
                return {}
        print(f"Configuration loaded successfully from '{resolved_path}'.")
        return config_data if config_data is not None else {}
    except yaml.YAMLError as ye:
        print(
            f"Error parsing YAML configuration from '{resolved_path}': {ye}. "
            f"Returning empty config."
        )
        return {}
    except json.JSONDecodeError as je:
        print(
            f"Error parsing JSON configuration from '{resolved_path}': {je}. "
            f"Returning empty config."
        )
        return {}
    except Exception as e:
        print(
            f"An unexpected error occurred while loading configuration "
            f"from '{resolved_path}': {e}. Returning empty config."
        )
        return {}

def get_config_value(config: Dict[str, Any], key_path: str,
                     default: Optional[Any] = None) -> Any:
    """Retrieves a value from a nested config dict using a dot-separated key.

    Example:
        `get_config_value(config, "simulation.patient.isf", 50)`
        This would look for `config['simulation']['patient']['isf']`.

    Args:
        config (Dict[str, Any]): The configuration dictionary to search
            within.
        key_path (str): A dot-separated string representing the path to
            the desired key (e.g., "agent.learning_rate",
            "simulation.patient_params.CR").
        default (Optional[Any]): The default value to return if the key
            path is not found or if an intermediate key does not lead to
            a dictionary. Defaults to None.

    Returns:
        Any: The configuration value found at the `key_path`, or the
            `default` value if the path is not found or invalid.
    """
    keys = key_path.split('.')
    current_level = config
    for key in keys:
        if isinstance(current_level, dict) and key in current_level:
            current_level = current_level[key]
        else:
            return default  # Key not found or path is invalid
    return current_level


class ConfigManager:
    """A manager class for handling library configurations.

    This class provides a convenient way to load configuration settings
    from a file (YAML or JSON) and access them throughout the library
    using dot-separated key paths.

    Attributes:
        config_data (Dict[str, Any]): The dictionary holding all loaded
            configuration parameters.
        _config_file_path (Optional[str]): The path to the configuration
            file that was last successfully loaded. Stored for
            reloading.
    """
    def __init__(self, config_file_path: Optional[str] = None):
        """Initializes the ConfigManager and loads configuration.

        If `config_file_path` is None, it attempts to load from a
        default configuration file name (`DEFAULT_CONFIG_FILENAME`) in
        the current working directory.

        Args:
            config_file_path (Optional[str]): The path to the
                configuration file (YAML or JSON). Defaults to None.
        """
        self._config_file_path: Optional[str] = config_file_path  # Store initial path for reload
        self.config_data: Dict[str, Any] = load_config(config_file_path)
        
        if self.config_data:
            effective_path = config_file_path if config_file_path else \
                             (DEFAULT_CONFIG_FILENAME if os.path.exists(DEFAULT_CONFIG_FILENAME) \
                              else "no specific file (used defaults or empty)")
            print(
                f"ConfigManager initialized successfully with config from: "
                f"'{effective_path}'."
            )
        elif config_file_path:  # Attempted to load a specific file but failed
            print(
                f"ConfigManager: Warning - Could not load configuration from "
                f"'{config_file_path}'. Using empty/default configuration."
            )
        else:  # No specific file given, and default not found
             print(
                f"ConfigManager: Initialized with no specific config file and "
                f"default not found. Using empty/default configuration."
            )


    def get(self, key_path: str, default: Optional[Any] = None) -> Any:
        """Retrieves a configuration value using a dot-separated key path.

        Args:
            key_path (str): The dot-separated path to the desired
                configuration value (e.g., "simulation.duration_days",
                "agent.params.learning_rate").
            default (Optional[Any]): The default value to return if the
                key path is not found. Defaults to None.

        Returns:
            Any: The retrieved configuration value, or the default if not
                found.
        """
        return get_config_value(self.config_data, key_path, default)

    def get_section(self, section_key_path: str) -> Dict[str, Any]:
        """Retrieves an entire section of the configuration as a dictionary.

        Args:
            section_key_path (str): The dot-separated path to the desired
                configuration section (e.g.,
                "simulation.patient_params").

        Returns:
            Dict[str, Any]: The configuration section as a dictionary.
                Returns an empty dictionary if the section is not found
                or if the item at the path is not a dictionary.
        """
        section = self.get(section_key_path, default={})
        return section if isinstance(section, dict) else {}

    def reload(self, new_config_file_path: Optional[str] = None):
        """Reloads the configuration.

        If `new_config_file_path` is provided, it attempts to load from
        this new path and updates the internal path for future reloads.
        If no path is provided, it attempts to reload from the path used
        during the last successful load (or the default file if
        initially loaded that way).

        Args:
            new_config_file_path (Optional[str]): The path to a new
                configuration file to load. If None, reloads from the
                last known path.
        """
        path_to_load: Optional[str]
        if new_config_file_path is not None:
            path_to_load = new_config_file_path
            self._config_file_path = new_config_file_path  # Update stored path
            print(
                f"ConfigManager: Reloading configuration from new path: "
                f"'{path_to_load}'."
            )
        elif self._config_file_path is not None:
            path_to_load = self._config_file_path
            print(
                f"ConfigManager: Reloading configuration from previously "
                f"used path: '{path_to_load}'."
            )
        else:
            # If no path was ever stored (e.g., initialized with None and
            # default also not found), load_config will try default again.
            path_to_load = None 
            print(
                f"ConfigManager: Reloading configuration (attempting default: "
                f"'{DEFAULT_CONFIG_FILENAME}')."
            )
        
        self.config_data = load_config(path_to_load)
        if not self.config_data:
             print(
                f"ConfigManager: Warning - Reload attempt failed or resulted "
                f"in empty config for path: "
                f"'{path_to_load if path_to_load else 'default'}'."
            )


if __name__ == '__main__':
    # This block demonstrates usage of the configuration loading utilities.
    # It creates temporary dummy YAML and JSON config files for testing.

    print("--- Configuration Module Standalone Example ---")

    # Create dummy config files in a temporary directory for the example
    temp_dir = "temp_config_test_dir"
    os.makedirs(temp_dir, exist_ok=True)
    
    dummy_yaml_path = os.path.join(temp_dir, "test_config.yaml")
    dummy_json_path = os.path.join(temp_dir, "test_config.json")
    default_config_in_temp_path = os.path.join(temp_dir, DEFAULT_CONFIG_FILENAME)

    dummy_yaml_content = """
simulation:
  duration_days: 7
  time_step_minutes: 5
  patient_params:
    ISF: 50
    CR: 10
    initial_glucose: 120

predictive_model:
  type: "LSTM"
  lstm_params:
    hidden_dim: 64
    num_layers: 2

agent:
  type: "RLAgent"
  rl_algorithm: "PPO"
  learning_rate: 0.0003
  reward_params:
    tir_target_min: 70
    tir_target_max: 180
"""
    dummy_json_content = """
{
  "simulation": {
    "duration_days": 14,
    "time_step_minutes": 5,
    "patient_params": {
      "ISF": 45,
      "CR": 12
    }
  },
  "agent": {
    "type": "RuleBasedAgent",
    "rules": {
      "hypo_threshold": 65
    }
  }
}
"""
    default_dummy_content = {"default_setting_example": "Hello from default!"}

    with open(dummy_yaml_path, "w", encoding='utf-8') as f:
        f.write(dummy_yaml_content)
    with open(dummy_json_path, "w", encoding='utf-8') as f:
        f.write(dummy_json_content)
    with open(default_config_in_temp_path, "w", encoding='utf-8') as f:
        yaml.dump(default_dummy_content, f)

    print("\n--- Testing load_config function ---")
    config_yaml = load_config(dummy_yaml_path)
    print(
        f"Duration from YAML: "
        f"{get_config_value(config_yaml, 'simulation.duration_days', 1)}"
    )
    assert get_config_value(config_yaml, 'simulation.duration_days') == 7
    print(
        f"Agent LR from YAML: "
        f"{get_config_value(config_yaml, 'agent.learning_rate', 0.001)}"
    )
    assert get_config_value(config_yaml, 'agent.learning_rate') == 0.0003
    print(
        f"Missing key from YAML: "
        f"{get_config_value(config_yaml, 'agent.missing_key', 'default_val')}"
    )
    assert get_config_value(config_yaml, 'agent.missing_key', 'default_val') == 'default_val'

    config_json = load_config(dummy_json_path)
    print(
        f"\nISF from JSON: "
        f"{get_config_value(config_json, 'simulation.patient_params.ISF', 50)}"
    )
    assert get_config_value(config_json, 'simulation.patient_params.ISF') == 45
    print(
        f"Agent type from JSON: "
        f"{get_config_value(config_json, 'agent.type', 'Unknown')}"
    )
    assert get_config_value(config_json, 'agent.type') == 'RuleBasedAgent'

    # Test loading default when CWD is changed
    original_cwd = os.getcwd()
    os.chdir(temp_dir)  # Change CWD to where DEFAULT_CONFIG_FILENAME is
    print(f"\nChanged CWD to: {os.getcwd()}")
    config_default = load_config()  # Should load DEFAULT_CONFIG_FILENAME
    assert config_default.get("default_setting_example") == "Hello from default!"
    print(
        f"Default setting from '{DEFAULT_CONFIG_FILENAME}': "
        f"{config_default.get('default_setting_example')}"
    )
    os.chdir(original_cwd)  # Change back CWD
    print(f"Restored CWD to: {os.getcwd()}")


    print("\n--- Testing ConfigManager class ---")
    manager = ConfigManager(dummy_yaml_path)
    print(
        f"Manager - Patient ISF: "
        f"{manager.get('simulation.patient_params.ISF', 55)}"
    )
    assert manager.get('simulation.patient_params.ISF') == 50
    print(
        f"Manager - Agent Type: "
        f"{manager.get('agent.type', 'DefaultAgent')}"
    )
    assert manager.get('agent.type') == 'RLAgent'
    
    agent_config_section = manager.get_section('agent')
    print(f"Manager - Agent Section: {agent_config_section}")
    assert agent_config_section.get('learning_rate') == 0.0003

    print("\n--- Testing ConfigManager reload ---")
    manager.reload(dummy_json_path)
    print(
        f"Manager (reloaded to JSON) - Duration: "
        f"{manager.get('simulation.duration_days')}"
    )
    assert manager.get('simulation.duration_days') == 14
    print(
        f"Manager (reloaded to JSON) - Agent Type: "
        f"{manager.get('agent.type')}"
    )
    assert manager.get('agent.type') == 'RuleBasedAgent'

    # Test reload to default
    os.chdir(temp_dir)
    # When CWD is temp_dir, ConfigManager should be initialized with the base filename
    # so it's resolved correctly relative to the new CWD.
    manager_for_default_reload = ConfigManager(os.path.basename(dummy_json_path))  # Start with JSON
    assert manager_for_default_reload.get('simulation.duration_days') == 14
    manager_for_default_reload.reload()  # Reload from last known path (which is now the basename)
    assert manager_for_default_reload.get('simulation.duration_days') == 14  # Still JSON
    # Now, make it "forget" its path by re-init without path, then reload
    manager_for_default_reload_no_path = ConfigManager()  # Should load default from CWD (temp_dir)
    assert manager_for_default_reload_no_path.get("default_setting_example") == "Hello from default!"
    # If we call reload on an instance that loaded default, it should try default again
    manager_for_default_reload_no_path.reload()
    assert manager_for_default_reload_no_path.get("default_setting_example") == "Hello from default!"
    os.chdir(original_cwd)


    # Clean up dummy files and directory
    try:
        os.remove(dummy_yaml_path)
        os.remove(dummy_json_path)
        os.remove(default_config_in_temp_path)
        os.rmdir(temp_dir)
        print(f"\nCleaned up temporary config files and directory: {temp_dir}")
    except OSError as e:
        print(f"Error during cleanup: {e}")

    print("\nConfig module example run complete.")