from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any

@dataclass
class TrainingParams:
    """Configuration parameters for model training."""
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 0.001
    random_state: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert parameters to a dictionary."""
        return asdict(self)
