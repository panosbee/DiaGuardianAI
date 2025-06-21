import random
from typing import Any, List

class ReplayBuffer:
    """Simple replay buffer for continual learning."""

    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.buffer: List[Any] = []

    def add(self, experience: Any) -> None:
        self.buffer.append(experience)
        if len(self.buffer) > self.capacity:
            # Drop the oldest experience to maintain capacity
            self.buffer.pop(0)

    def sample(self, batch_size: int) -> List[Any]:
        if batch_size <= 0 or len(self.buffer) == 0:
            return []
        if len(self.buffer) <= batch_size:
            return list(self.buffer)
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)

