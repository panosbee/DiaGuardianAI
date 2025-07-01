"""Basic Federated Learning client utilities."""

from typing import Iterable, Tuple, Any, Callable, Optional

from .replay_buffer import ReplayBuffer

class FederatedClient:
    """Simplified federated learning client placeholder."""

    def __init__(self, model: object, client_id: str, server_callback=None, buffer_capacity: int = 1000):
        self.model = model
        self.client_id = client_id
        self.server_callback = server_callback
        self.replay_buffer = ReplayBuffer(buffer_capacity)

    def _update_model(self, batch: Iterable[Tuple[Any, Any]]) -> None:
        """Apply a training step on the underlying model."""
        if not batch:
            return

        features, targets = zip(*batch)
        X = list(features)
        y = list(targets)

        if hasattr(self.model, "partial_fit"):
            self.model.partial_fit(X, y)
        elif hasattr(self.model, "fit"):
            self.model.fit(X, y)
        elif hasattr(self.model, "update"):
            self.model.update(list(batch))
        elif hasattr(self.model, "train_step"):
            self.model.train_step(list(batch))
        else:
            print(f"Client {self.client_id}: Model does not support training methods")

    def train_local(self, data: Iterable[Tuple[Any, Any]]) -> None:
        """Perform local training on provided data and store in the buffer."""
        for sample in data:
            self.replay_buffer.add(sample)

        self._update_model(list(data))
        print(
            f"Client {self.client_id}: replay buffer size {len(self.replay_buffer)}"
        )

    def continual_update(self, batch_size: int = 32):
        """Train the local model using a sample from the replay buffer."""
        batch = self.replay_buffer.sample(batch_size)
        if not batch:
            return
        self._update_model(batch)
        print(
            f"Client {self.client_id}: training on batch of {len(batch)} samples"
        )

    def share_updates(self):
        """Send model updates back to the server."""
        print(f"Client {self.client_id}: sharing updates with server")
        if self.server_callback:
            updated_model = self.server_callback(self.model)
            if updated_model is not None:
                self.model = updated_model

