"""Core components for the DiaGuardianAI simulation environment.

This module provides the central mechanisms for running simulations, including
the simulation engine itself and the abstract base classes that define the
interfaces for all major components of the library (patients, predictive models,
agents, and pattern repositories). These base classes are crucial for ensuring
modularity and extensibility.

Key Contents:
    - `SimulationEngine`: Orchestrates the simulation loop and interactions.
    - `BaseSyntheticPatient`: ABC for synthetic patient models.
    - `BasePredictiveModel`: ABC for glucose prediction models.
    - `BaseAgent`: ABC for decision-making agents.
    - `BasePatternRepository`: ABC for pattern storage and retrieval systems.
    - `DiaGuardianEnv`: Gymnasium-compatible environment for RL training.
"""

# Make key classes from this module available when `core` is imported.
from .simulation_engine import SimulationEngine
from .base_classes import (
    BaseSyntheticPatient,
    BasePredictiveModel,
    BaseAgent,
    BasePatternRepository
)
from .environments import DiaGuardianEnv

# Define __all__ for explicit public API if desired
__all__ = [
    "SimulationEngine",
    "BaseSyntheticPatient",
    "BasePredictiveModel",
    "BaseAgent",
    "BasePatternRepository",
    "DiaGuardianEnv"
]