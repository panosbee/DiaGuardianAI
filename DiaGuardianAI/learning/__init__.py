"""Utility submodule for advanced learning features."""

from .meta_learning import MetaLearner
from .federated_learning import FederatedClient
from .ood_detection import SimpleOODDetector

__all__ = ["MetaLearner", "FederatedClient", "SimpleOODDetector"]
