"""Utility module for the DiaGuardianAI library.

This package provides various helper functions, common tools, and utility
classes used across the library. This includes functions for calculating
performance metrics, managing configuration parameters, and potentially other
shared functionalities like plotting or data manipulation helpers in the future.

Key Contents:
    - `metrics.py`: Contains functions for calculating diabetes-specific
      evaluation metrics such as MARD, RMSE, TIR, Clarke Error Grid Analysis,
      LBGI, and HBGI.
    - `config.py`: Provides tools for loading and accessing configuration
      parameters from YAML or JSON files, facilitating easy management of
      settings for simulations, models, and agents.
"""

# from .config import ConfigManager, load_config, get_config_value
# from . import metrics # Or specific functions like from .metrics import calculate_mard