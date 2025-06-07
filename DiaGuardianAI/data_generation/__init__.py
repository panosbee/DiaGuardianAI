"""Module for synthetic data generation in DiaGuardianAI.

This package contains components responsible for creating realistic synthetic
patient data for Type 1 Diabetes (T1D) simulations. This includes models for
simulating patient physiology and formatters to prepare the data for use by
predictive models and agents.

Key Contents:
    - `SyntheticPatient`: A model to simulate T1D patient glucose dynamics
      in response to insulin, meals, and exercise.
    - `DataFormatter`: A utility to process raw time-series data (CGM, insulin,
      carbs) into structured formats (e.g., sliding windows of features and
      targets) suitable for machine learning models.
"""

# Optionally, make key classes from this module available.
# from .synthetic_patient_model import SyntheticPatient
# from .data_formatter import DataFormatter