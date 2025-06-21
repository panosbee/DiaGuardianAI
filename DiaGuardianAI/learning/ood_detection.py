import numpy as np

class SimpleOODDetector:
    """Basic out-of-distribution detector using vector magnitude."""

    def __init__(self, threshold: float = 3.0, scale: float = 1.0):
        self.threshold = threshold
        self.scale = scale

    def score(self, obs: np.ndarray) -> float:
        return float(np.linalg.norm(obs))

    def probability_ood(self, obs: np.ndarray) -> float:
        """Return a probability-like score that the observation is OOD."""
        score = self.score(obs)
        return float(1.0 / (1.0 + np.exp(-(score - self.threshold) / self.scale)))

    def is_ood(self, obs: np.ndarray) -> bool:
        return self.score(obs) > self.threshold
