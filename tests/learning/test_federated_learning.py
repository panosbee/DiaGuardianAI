import importlib.util
import types
from pathlib import Path
import sys
import pytest

ROOT = Path(__file__).resolve().parents[2]
pkg_root = types.ModuleType("DiaGuardianAI")
pkg_root.__path__ = [str(ROOT / "DiaGuardianAI")]
sys.modules.setdefault("DiaGuardianAI", pkg_root)

learning_pkg = types.ModuleType("DiaGuardianAI.learning")
learning_pkg.__path__ = [str(ROOT / "DiaGuardianAI" / "learning")]
sys.modules.setdefault("DiaGuardianAI.learning", learning_pkg)

spec = importlib.util.spec_from_file_location(
    "DiaGuardianAI.learning.federated_learning",
    str(ROOT / "DiaGuardianAI/learning/federated_learning.py"),
)
fed_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = fed_module
spec.loader.exec_module(fed_module)
FederatedClient = fed_module.FederatedClient


class SimpleModel:
    """Very small model used to test federated training."""

    def __init__(self):
        self.weight = 0.0

    def update(self, batch):
        for x, y in batch:
            self.weight += x * y


def test_train_local_updates_model_and_buffer():
    model = SimpleModel()
    client = FederatedClient(model, client_id="c1")
    data = [(1, 2), (2, 3)]
    client.train_local(data)
    assert len(client.replay_buffer) == 2
    assert pytest.approx(model.weight) == 1 * 2 + 2 * 3


def test_continual_update_uses_buffer_samples():
    model = SimpleModel()
    client = FederatedClient(model, client_id="c2")
    # all samples identical for deterministic weight change
    client.train_local([(1, 2)] * 5)
    model.weight = 0.0
    client.continual_update(batch_size=3)
    # each update adds x*y=2; 3 samples => 6
    assert pytest.approx(model.weight) == 6.0


def test_share_updates_applies_server_response():
    model = SimpleModel()
    # pre-populate weight
    model.weight = 5.0

    def server_cb(m):
        assert m.weight == 5.0
        new_model = SimpleModel()
        new_model.weight = 7.0
        return new_model

    client = FederatedClient(model, client_id="c3", server_callback=server_cb)
    client.share_updates()
    assert isinstance(client.model, SimpleModel)
    assert client.model.weight == 7.0




