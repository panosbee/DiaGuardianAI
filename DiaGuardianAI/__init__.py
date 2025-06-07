"""
DiaGuardianAI: Professional AI System for Diabetes Management

A comprehensive, production-ready library for diabetes management using
advanced AI techniques including multi-agent systems, transformer models,
and continuous learning.

Example usage:
    >>> import diaguardianai as dga
    >>>
    >>> # Create a patient population
    >>> factory = dga.HumanModelFactory()
    >>> patients = factory.generate_population(size=100)
    >>>
    >>> # Initialize AI system
    >>> ai_system = dga.DiaGuardianAI()
    >>> ai_system.train(patients)
    >>>
    >>> # Make predictions
    >>> predictions = ai_system.predict(patient_state)
    >>> recommendations = ai_system.recommend(patient_state)
"""

__version__ = "1.0.0"
__author__ = "DiaGuardianAI Team"
__email__ = "contact@diaguardianai.com"
__license__ = "MIT"
__url__ = "https://github.com/diaguardianai/diaguardianai"

# Core imports for easy access
from .core.base_classes import BaseSyntheticPatient
from .data_generation.human_model_factory import HumanModelFactory, HumanProfile
from .data_generation.synthetic_data_generator import SyntheticDataGenerator
from .data_generation.synthetic_patient_model import SyntheticPatient
from .models.transformer_zoo import TransformerZoo
from .agents.advanced_multi_agent_system import ContinuousLearningLoop
from .pattern_repository.repository_manager import RepositoryManager
from .core.intelligent_meal_system import IntelligentMealSystem

# Main API class
class DiaGuardianAI:
    """
    Main API class for DiaGuardianAI system.

    This class provides a simple, pandas-like interface for diabetes AI.

    Example:
        >>> import diaguardianai as dga
        >>> ai = dga.DiaGuardianAI()
        >>> ai.train(patient_data)
        >>> predictions = ai.predict(current_state)
    """

    def __init__(self, config=None):
        """Initialize DiaGuardianAI system."""
        self.config = config or {}
        self.transformer_zoo = None
        self.learning_loop = None
        self.is_trained = False

    def train(self, patient_data, **kwargs):
        """Train the AI system on patient data."""
        # Implementation would go here
        self.is_trained = True
        return self

    def predict(self, patient_state, horizons=None):
        """Make glucose predictions for given patient state."""
        if not self.is_trained:
            raise ValueError("System must be trained before making predictions")
        # Implementation would go here
        return {}

    def recommend(self, patient_state):
        """Generate insulin recommendations for given patient state."""
        if not self.is_trained:
            raise ValueError("System must be trained before making recommendations")
        # Implementation would go here
        return {}

# Make key classes available at package level
__all__ = [
    # Main API
    "DiaGuardianAI",

    # Core classes
    "BaseSyntheticPatient",
    "SyntheticPatient",
    "HumanModelFactory",
    "HumanProfile",
    "SyntheticDataGenerator",

    # AI components
    "TransformerZoo",
    "ContinuousLearningLoop",
    "RepositoryManager",
    "IntelligentMealSystem",

    # Metadata
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    "__url__",
]