"""Sub-package for various glucose predictive model architectures.

The `model_zoo` contains implementations of different types of machine learning
models that can be used for blood glucose prediction within the DiaGuardianAI
framework. Each model is expected to inherit from `BasePredictiveModel`.

Current Models (Placeholders):
    - `LSTMPredictor`: A Long Short-Term Memory (LSTM) based recurrent neural network.
    - `TransformerPredictor`: A Transformer-based model, typically using an encoder
      architecture for time-series forecasting.
    - `EnsemblePredictor`: A model that combines predictions from multiple other
      models to potentially improve performance and robustness.

Future models (as per plan2.txt) could include N-BEATS, TSMixer, etc.
"""

from .lstm_predictor import LSTMPredictor
from .transformer_predictor import TransformerPredictor
from .ensemble_predictor import EnsemblePredictor
from .nbeats_predictor import NBEATSPredictor
from .tsmixer_predictor import TSMixerPredictor
# Import other models as they are added.