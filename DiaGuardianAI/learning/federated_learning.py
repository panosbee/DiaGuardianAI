from .replay_buffer import ReplayBuffer

class FederatedClient:
    """Simplified federated learning client placeholder."""

    def __init__(self, model: object, client_id: str, server_callback=None, buffer_capacity: int = 1000):
        self.model = model
        self.client_id = client_id
        self.server_callback = server_callback
        self.replay_buffer = ReplayBuffer(buffer_capacity)

    def train_local(self, data):
        """Simulate local training on the client's data."""
        for sample in data:
            self.replay_buffer.add(sample)
        print(f"Client {self.client_id}: replay buffer size {len(self.replay_buffer)}")
        # Actual gradient updates would normally occur here.

    def continual_update(self, batch_size: int = 32):
        """Train the local model using a sample from the replay buffer."""
        batch = self.replay_buffer.sample(batch_size)
        if not batch:
            return
        print(f"Client {self.client_id}: training on batch of {len(batch)} samples")
        # Placeholder: the model would be updated using `batch` here.

    def share_updates(self):
        """Send model updates back to the server."""
        print(f"Client {self.client_id}: sharing updates with server")
        if self.server_callback:
            self.server_callback(self.model)