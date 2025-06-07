"""Module for blood glucose (BG) predictive models in DiaGuardianAI.

This package houses various models designed for forecasting future BG levels.
It includes a `model_zoo` for different architectures (e.g., LSTM, Transformer,
Ensemble) and a `ModelTrainer` class to manage the training and optimization
of these models.

Key Contents:
    - `model_zoo/`: A sub-package containing implementations of various
      predictive model architectures.
        - `LSTMPredictor`: LSTM-based model.
        - `TransformerPredictor`: Transformer-based model.
        - `EnsemblePredictor`: Model for combining multiple predictors.
    - `ModelTrainer`: A class to handle the training, fine-tuning, and
      hyperparameter optimization of the predictive models.
"""

# Optionally, make key classes available.
from .model_trainer import ModelTrainer
from .model_zoo import LSTMPredictor # model_zoo now exports these
from .model_zoo import TransformerPredictor
from .model_zoo import EnsemblePredictor