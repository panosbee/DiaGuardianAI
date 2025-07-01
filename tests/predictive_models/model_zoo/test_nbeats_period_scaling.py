import torch
import numpy as np
from DiaGuardianAI.predictive_models.model_zoo.nbeats_predictor import (
    SeasonalityBasisLayer,
    NBEATSPredictor,
)


def test_seasonality_basis_period_scaling_effect():
    layer_default = SeasonalityBasisLayer(num_fourier_terms=1, out_features=4)
    layer_scaled = SeasonalityBasisLayer(num_fourier_terms=1, out_features=4, period_scaling=2.0)
    theta = torch.ones(1, 2)
    out_default = layer_default(theta)
    out_scaled = layer_scaled(theta)
    assert not torch.allclose(out_default, out_scaled)


def test_nbeats_predictor_period_scaling_initialization():
    predictor = NBEATSPredictor(
        input_chunk_length=5,
        output_horizon_steps=3,
        stack_types=["seasonality"],
        seasonality_fourier_terms=1,
        period_scaling=2.0,
    )
    forecast_layer = predictor.model.forecast_basis_layers[0]
    backcast_layer = predictor.model.backcast_basis_layers[0]
    assert isinstance(forecast_layer, SeasonalityBasisLayer)
    assert forecast_layer.period_scaling == 2.0
    assert backcast_layer.period_scaling == 2.0

