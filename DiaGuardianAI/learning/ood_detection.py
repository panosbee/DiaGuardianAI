import numpy as np

class SimpleOODDetector:
    """Basic out-of-distribution detector using vector magnitude."""

    def __init__(self, threshold: float = 3.0):
        self.threshold = threshold

    def score(self, obs: np.ndarray) -> float:
        return float(np.linalg.norm(obs))

    def is_ood(self, obs: np.ndarray) -> bool:
        return self.score(obs) > self.threshold
