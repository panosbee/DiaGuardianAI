import numpy as np
import pandas as pd
import pytest
import sys
import types

sys.modules.setdefault("optuna", types.ModuleType("optuna"))

from DiaGuardianAI.predictive_models.model_zoo.lstm_predictor import LSTMPredictor
from DiaGuardianAI.core.base_classes import BasePredictiveModel


def _create_dummy_df(rows: int) -> pd.DataFrame:
    return pd.DataFrame({
        "cgm_mg_dl": np.random.rand(rows).astype(np.float32) * 100 + 80,
        "bolus_U": np.random.rand(rows).astype(np.float32),
        "carbs_g": np.random.rand(rows).astype(np.float32) * 10,
    })


@pytest.fixture
def lstm_predictor_instance() -> LSTMPredictor:
    return LSTMPredictor(input_seq_len=4, output_seq_len=2, n_features=3,
                         hidden_units=8, num_layers=1, dropout_rate=0.1,
                         learning_rate=0.01)


def test_lstm_predictor_initialization(lstm_predictor_instance: LSTMPredictor):
    assert isinstance(lstm_predictor_instance, BasePredictiveModel)
    assert lstm_predictor_instance.input_seq_len == 4
    assert lstm_predictor_instance.output_seq_len == 2
    assert lstm_predictor_instance.n_features == 3
    assert lstm_predictor_instance.hidden_units == 8


def test_lstm_predictor_training_with_validation(lstm_predictor_instance: LSTMPredictor, capsys):
    train_df = _create_dummy_df(30)
    val_df = _create_dummy_df(30)

    lstm_predictor_instance.train(data=[train_df], epochs=1, batch_size=8, validation_data=[val_df])
    captured = capsys.readouterr()
    assert "Train Loss" in captured.out
    assert "Val Loss" in captured.out
    assert lstm_predictor_instance.is_trained


def test_lstm_predictor_predict(lstm_predictor_instance: LSTMPredictor):
    train_df = _create_dummy_df(30)
    lstm_predictor_instance.train(data=[train_df], epochs=1, batch_size=8)

    input_df = _create_dummy_df(4)
    preds = lstm_predictor_instance.predict(input_df)
    assert "mean" in preds
    assert len(preds["mean"]) == lstm_predictor_instance.output_seq_len


def test_lstm_predictor_save_load(tmp_path):
    model_dir = tmp_path / "lstm_model"
    model_dir.mkdir()

    train_df = _create_dummy_df(30)
    predictor = LSTMPredictor(input_seq_len=4, output_seq_len=2, n_features=3,
                              hidden_units=8, num_layers=1, dropout_rate=0.1,
                              learning_rate=0.01)
    predictor.train(data=[train_df], epochs=1, batch_size=8)
    predictor.save(str(model_dir))

    new_predictor = LSTMPredictor()
    new_predictor.load(str(model_dir))
    assert new_predictor.is_trained
