class FederatedClient:
    """Simplified federated learning client placeholder."""

    def __init__(self, model: object, client_id: str, server_callback=None):
        self.model = model
        self.client_id = client_id
        self.server_callback = server_callback

    def train_local(self, data):
        """Simulate local training on the client's data."""
        print(f"Client {self.client_id}: training on {len(data)} records")
        # Normally gradients or model weights would be updated here.

    def share_updates(self):
        """Send model updates back to the server."""
        print(f"Client {self.client_id}: sharing updates with server")
        if self.server_callback:
            self.server_callback(self.model)
