import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from DiaGuardianAI.learning.federated_learning import FederatedClient

class SimpleModel:
    def __init__(self):
        self.updated = False


def dummy_server_callback(model):
    model.updated = True


def test_federated_client_workflow():
    model = SimpleModel()
    client = FederatedClient(model=model, client_id="c1", server_callback=dummy_server_callback, buffer_capacity=10)
    data = [1, 2, 3]
    client.train_local(data)
    assert len(client.replay_buffer) == 3
    client.continual_update(batch_size=2)
    client.share_updates()
    assert model.updated is True
