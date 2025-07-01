# DiaGuardianAI - N-BEATS Predictor
# Neural Basis Expansion Analysis for Time Series Forecasting

import sys
import os
from typing import Dict, Any, List, Optional, Tuple, Union, cast

import torch
import torch.nn as nn
import numpy as np

# Ensure the DiaGuardianAI package is discoverable
if __package__ is None or __package__ == '':
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from DiaGuardianAI.core.base_classes import BasePredictiveModel
from DiaGuardianAI.predictive_models.model_trainer import ModelTrainer # For the train method

import math # For seasonality

# --- N-BEATS Core Components ---

# --- Basis Function Layers ---
class TrendBasisLayer(nn.Module):
    def __init__(self, degree: int, out_features: int):
        super().__init__()
        self.degree = degree
        self.out_features = out_features
        # Time vector t is created in forward pass, normalized for the output length

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        # theta shape: (batch_size, degree + 1)
        # out_features: H (forecast horizon) or P (lookback period for backcast)
        batch_size = theta.shape[0]
        
        # Create time vector t for the output length (normalized from 0 to 1 or similar)
        # For simplicity, let's use 0 to 1 for forecast, and -1 to 0 for backcast if needed,
        # or just always 0 to 1 and let the coefficients adapt.
        # The original paper uses t = (0, 1/H, 2/H, ..., (H-1)/H) for forecast
        # and t = (0, 1/P, ..., (P-1)/P) for backcast, but often P=H.
        # Let's use linspace from 0 to 1 for now.
        t = torch.linspace(0, 1, steps=self.out_features, device=theta.device).unsqueeze(0) # (1, out_features)
        
        powers = torch.arange(self.degree + 1, device=theta.device).float() # (degree + 1)
        t_powered = t.unsqueeze(-1) ** powers # (1, out_features, degree + 1)
        
        # theta is (batch_size, degree + 1) -> needs to be (batch_size, 1, degree+1) for broadcasting
        trend = torch.einsum('btd,bd->bt', t_powered, theta) # (batch_size, out_features)
        return trend

class SeasonalityBasisLayer(nn.Module):
    """Seasonality basis using Fourier terms.

    Parameters
    ----------
    num_fourier_terms : int
        Number of Fourier harmonics to generate.
    out_features : int
        Length of the generated sequence.
    period_scaling : float, optional
        Factor to scale the base period. ``period_scaling=2`` doubles the
        period of the fundamental harmonic, effectively dividing the
        frequency by two. Default is ``1.0`` which keeps the standard
        period tied to ``out_features``.
    """

    def __init__(self, num_fourier_terms: int, out_features: int, period_scaling: float = 1.0):
        super().__init__()
        self.num_fourier_terms = num_fourier_terms
        self.out_features = out_features
        self.period_scaling = period_scaling

        # Coefficients for cos_1, sin_1, cos_2, sin_2, ...
        if num_fourier_terms <= 0:
            raise ValueError("num_fourier_terms must be positive for SeasonalityBasisLayer")

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        # theta shape: (batch_size, 2 * num_fourier_terms)
        batch_size = theta.shape[0]
        
        # Time vector t (e.g., 0 to 1 for forecast horizon)
        t = torch.linspace(0, 1, steps=self.out_features, device=theta.device).unsqueeze(0) # (1, out_features)
        
        seasonality_components = []
        for k in range(1, self.num_fourier_terms + 1):
            # Get cos_k and sin_k coefficients from theta
            # theta[:, 2*(k-1)] is cos_k coeff, theta[:, 2*(k-1)+1] is sin_k coeff
            cos_coeff = theta[:, 2*(k-1)].unsqueeze(1) # (batch_size, 1)
            sin_coeff = theta[:, 2*(k-1)+1].unsqueeze(1) # (batch_size, 1)
            
            # 2 * pi * k * t
            # The 'k' here is the harmonic number.
            # The 'period' in the original paper is often the forecast horizon H for forecast seasonality
            # or lookback P for backcast seasonality.
            # So, 2*pi*k*t_h/H where t_h = 0,1,...,H-1.
            # Our t is already 0..1, so effectively t_h/H.
            angle = 2 * math.pi * k * t / self.period_scaling
            
            seasonality_components.append(cos_coeff * torch.cos(angle))
            seasonality_components.append(sin_coeff * torch.sin(angle))
            
        seasonality = torch.sum(torch.stack(seasonality_components, dim=-1), dim=-1) # (batch_size, out_features)
        return seasonality

class GenericBasisLayer(nn.Module):
    """Generic basis layer using a simple linear projection."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        return self.linear(theta)


class NBEATSBlock(nn.Module):
    """
    A single block in the N-BEATS architecture.
    It consists of a few fully connected layers with ReLU activations.
    The block produces a forecast (backcast_output) for its input
    and a representation (forecast_output) for the next block.
    """
    def __init__(self,
                 input_chunk_length: int,
                 theta_b_dim: int, # Dimension of backcast coefficients
                 theta_f_dim: int, # Dimension of forecast coefficients
                 num_hidden_layers: int = 4,
                 hidden_layer_units: int = 256,
                 share_weights_in_stack: bool = False): # Not directly used here but common in NBEATS variants
        super().__init__()
        self.input_chunk_length = input_chunk_length
        self.share_weights_in_stack = share_weights_in_stack

        layers = []
        current_dim = input_chunk_length
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(current_dim, hidden_layer_units))
            layers.append(nn.ReLU())
            current_dim = hidden_layer_units
        
        self.fc_stack = nn.Sequential(*layers)
        
        # Output layers for backcast (theta_b) and forecast (theta_f) coefficients
        self.theta_b_fc = nn.Linear(hidden_layer_units, theta_b_dim)
        self.theta_f_fc = nn.Linear(hidden_layer_units, theta_f_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_chunk_length)
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - backcast_coeffs (torch.Tensor): Coefficients for the backcast, shape (batch_size, theta_dims)
                - forecast_coeffs (torch.Tensor): Coefficients for the forecast, shape (batch_size, theta_dims)
        """
        hidden_output = self.fc_stack(x)
        backcast_coeffs = self.theta_b_fc(hidden_output)
        forecast_coeffs = self.theta_f_fc(hidden_output)
        return backcast_coeffs, forecast_coeffs


class PyTorchNBEATSModel(nn.Module):
    """
    The N-BEATS model architecture.
    It consists of multiple stacks, each containing multiple blocks.
    Supports generic, trend, and seasonality stacks. The ``period_scaling``
    argument controls the base period of seasonality stacks, allowing the
    Fourier basis to model longer or shorter seasonal cycles.
    """
    def __init__(self,
                 input_chunk_length: int,
                 output_chunk_length: int, # Forecast horizon
                 stack_types: List[str], # e.g., ['trend', 'seasonality', 'generic']
                 num_blocks_per_stack: Union[int, List[int]],
                 num_hidden_layers_per_block: Union[int, List[int]],
                 hidden_layer_units: Union[int, List[int]],
                 trend_polynomial_degree: int = 2, # Default for trend stacks
                 seasonality_fourier_terms: int = 5, # Default for seasonality stacks
                 period_scaling: float = 1.0,
                 generic_theta_dims: int = 256, # Default for generic stacks
                 share_weights_in_stack: bool = False,
                 dropout_prob: float = 0.0):
        super().__init__()
        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length
        self.num_stacks = len(stack_types)
        self.stack_types = stack_types
        self.period_scaling = period_scaling

        # Helper to get per-stack config or use default
        def _get_stack_config(param_name: str, param_value: Union[int, List[int]], stack_idx: int, default_val_if_int: int):
            if isinstance(param_value, list):
                if len(param_value) != self.num_stacks:
                    raise ValueError(f"Config list length mismatch for '{param_name}' ({len(param_value)}) and num_stacks ({self.num_stacks}).")
                return param_value[stack_idx]
            return param_value # If int, it applies to all stacks

        self.stacks = nn.ModuleList()
        self.forecast_basis_layers = nn.ModuleList()
        self.backcast_basis_layers = nn.ModuleList()

        for i in range(self.num_stacks):
            stack_type = self.stack_types[i]
            
            current_num_blocks = _get_stack_config("num_blocks_per_stack", num_blocks_per_stack, i, 1)
            current_num_hidden_layers = _get_stack_config("num_hidden_layers_per_block", num_hidden_layers_per_block, i, 4)
            current_hidden_units = _get_stack_config("hidden_layer_units", hidden_layer_units, i, 256)
            
            theta_b_dim_for_stack: int
            theta_f_dim_for_stack: int

            if stack_type == 'trend':
                theta_dim = trend_polynomial_degree + 1
                self.forecast_basis_layers.append(TrendBasisLayer(degree=trend_polynomial_degree, out_features=output_chunk_length))
                self.backcast_basis_layers.append(TrendBasisLayer(degree=trend_polynomial_degree, out_features=input_chunk_length))
                theta_b_dim_for_stack = theta_f_dim_for_stack = theta_dim
            elif stack_type == 'seasonality':
                if seasonality_fourier_terms <= 0: # Ensure positive for layer
                     raise ValueError("seasonality_fourier_terms must be positive for seasonality stack type.")
                theta_dim = 2 * seasonality_fourier_terms
                self.forecast_basis_layers.append(
                    SeasonalityBasisLayer(
                        num_fourier_terms=seasonality_fourier_terms,
                        out_features=output_chunk_length,
                        period_scaling=self.period_scaling,
                    )
                )
                self.backcast_basis_layers.append(
                    SeasonalityBasisLayer(
                        num_fourier_terms=seasonality_fourier_terms,
                        out_features=input_chunk_length,
                        period_scaling=self.period_scaling,
                    )
                )
                theta_b_dim_for_stack = theta_f_dim_for_stack = theta_dim
            elif stack_type == 'generic':
                self.forecast_basis_layers.append(GenericBasisLayer(in_features=generic_theta_dims, out_features=output_chunk_length))
                self.backcast_basis_layers.append(GenericBasisLayer(in_features=generic_theta_dims, out_features=input_chunk_length))
                theta_b_dim_for_stack = theta_f_dim_for_stack = generic_theta_dims
            else:
                raise ValueError(f"Unknown stack_type: {stack_type}")

            stack_blocks_list = nn.ModuleList() # Correctly define as ModuleList for this stack
            for _ in range(current_num_blocks):
                stack_blocks_list.append(
                    NBEATSBlock(
                        input_chunk_length=input_chunk_length,
                        theta_b_dim=theta_b_dim_for_stack,
                        theta_f_dim=theta_f_dim_for_stack,
                        num_hidden_layers=current_num_hidden_layers,
                        hidden_layer_units=current_hidden_units,
                        share_weights_in_stack=share_weights_in_stack
                    )
                )
            self.stacks.append(stack_blocks_list) # Append the ModuleList of blocks for this stack
        
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x_input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_input (torch.Tensor): Input tensor of shape (batch_size, input_chunk_length, num_features).
                                   N-BEATS is typically univariate, so num_features=1.
                                   We'll assume input is (batch_size, input_chunk_length) for now.
        Returns:
            torch.Tensor: Forecast tensor of shape (batch_size, output_chunk_length)
        """
        if x_input.ndim == 3 and x_input.shape[2] == 1:
            x = x_input.squeeze(-1) # (batch_size, input_chunk_length)
        elif x_input.ndim == 2:
            x = x_input # Already (batch_size, input_chunk_length)
        else:
            raise ValueError(f"NBEATS expects input shape (batch, seq_len, 1) or (batch, seq_len). Got {x_input.shape}")

        batch_size = x.shape[0]
        
        # Initialize overall forecast
        forecast = torch.zeros(batch_size, self.output_chunk_length, device=x.device)
        
        # Residual connection: input to the first block of the first stack
        current_backcast_residual = x
        
        for stack_idx, _stack_module_from_list in enumerate(self.stacks):
            # Explicitly cast to nn.ModuleList to help Pylance understand it's iterable
            stack = cast(nn.ModuleList, _stack_module_from_list)
            stack_forecast_sum = torch.zeros(batch_size, self.output_chunk_length, device=x.device)
            
            for block_idx, block in enumerate(stack):
                # Apply dropout to the input of the block's FC stack
                block_input_after_dropout = self.dropout(current_backcast_residual)
                
                backcast_coeffs, forecast_coeffs = block(block_input_after_dropout)
                
                # Apply stack-specific basis functions
                block_backcast = self.backcast_basis_layers[stack_idx](backcast_coeffs)
                block_forecast = self.forecast_basis_layers[stack_idx](forecast_coeffs)
                
                # Update residual for the next block/stack
                current_backcast_residual = current_backcast_residual - block_backcast
                
                # Accumulate forecast from this block
                stack_forecast_sum = stack_forecast_sum + block_forecast
            
            # Add this stack's forecast to the overall forecast
            forecast = forecast + stack_forecast_sum
            
        return forecast


class NBEATSPredictor(BasePredictiveModel):
    """
    Wrapper for the :class:`PyTorchNBEATSModel`, conforming to
    :class:`BasePredictiveModel` interface. The ``period_scaling`` parameter
    allows tuning of the seasonality frequency when seasonality stacks are
    present.
    """
    def __init__(self,
                 input_chunk_length: int,
                 output_horizon_steps: int,
                 stack_types: List[str], # Must be provided, e.g., ['trend', 'seasonality', 'generic']
                 num_blocks_per_stack: Union[int, List[int]] = 1,
                 num_hidden_layers_per_block: Union[int, List[int]] = 4,
                 hidden_layer_units: Union[int, List[int]] = 256,
                 trend_polynomial_degree: int = 2,
                 seasonality_fourier_terms: int = 5,
                 period_scaling: float = 1.0,
                 generic_theta_dims: int = 256,
                 share_weights_in_stack: bool = False,
                 dropout_prob: float = 0.1,
                 mc_dropout_samples: int = 50,
                 id_val: int = 0):
        super().__init__()
        self.input_chunk_length = input_chunk_length
        self.output_horizon_steps = output_horizon_steps
        # Store all parameters needed to reconstruct PyTorchNBEATSModel
        self.stack_types_config = stack_types
        self.num_blocks_per_stack_config = num_blocks_per_stack
        self.num_hidden_layers_per_block_config = num_hidden_layers_per_block
        self.hidden_layer_units_config = hidden_layer_units
        self.trend_polynomial_degree_config = trend_polynomial_degree
        self.seasonality_fourier_terms_config = seasonality_fourier_terms
        self.period_scaling_config = period_scaling
        self.generic_theta_dims_config = generic_theta_dims
        self.share_weights_in_stack_config = share_weights_in_stack
        self.dropout_prob_config = dropout_prob
        
        self.mc_dropout_samples = mc_dropout_samples # For predict method
        self.id_val = id_val

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = PyTorchNBEATSModel(
            input_chunk_length=self.input_chunk_length,
            output_chunk_length=self.output_horizon_steps,
            stack_types=self.stack_types_config,
            num_blocks_per_stack=self.num_blocks_per_stack_config,
            num_hidden_layers_per_block=self.num_hidden_layers_per_block_config,
            hidden_layer_units=self.hidden_layer_units_config,
            trend_polynomial_degree=self.trend_polynomial_degree_config,
            seasonality_fourier_terms=self.seasonality_fourier_terms_config,
            period_scaling=self.period_scaling_config,
            generic_theta_dims=self.generic_theta_dims_config,
            share_weights_in_stack=self.share_weights_in_stack_config,
            dropout_prob=self.dropout_prob_config
        ).to(self.device)

        self.trained = False
        print(f"NBEATSPredictor {self.id_val} initialized on {self.device}.")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")


    def train(self,
              X_train: Union[np.ndarray, torch.Tensor],
              y_train: Union[np.ndarray, torch.Tensor],
              X_val: Optional[Union[np.ndarray, torch.Tensor]] = None,
              y_val: Optional[Union[np.ndarray, torch.Tensor]] = None,
              training_params: Optional[Dict[str, Any]] = None):
        """Trains the N-BEATS model using ModelTrainer."""
        print(f"NBEATSPredictor {self.id_val} train called.")
        
        default_training_params = {
            "epochs": 50,
            "batch_size": 32,
            "learning_rate": 0.001
        }
        if training_params:
            default_training_params.update(training_params)

        trainer = ModelTrainer(model=self, training_params=default_training_params)
        # ModelTrainer expects X_train to be (batch, seq_len, features) or (batch, features)
        # and y_train to be (batch, output_horizon)
        # NBEATS forward expects (batch, seq_len) if univariate.
        # Ensure X_train is (batch, input_chunk_length) for the NBEATS model if it's univariate.
        # If X_train comes from DataFormatter, it might be (samples, lookback, features).
        # We need to ensure it's reshaped/selected appropriately before passing to PyTorchNBEATSModel.
        # For now, assume ModelTrainer handles tensor conversion and device placement.
        # The PyTorchNBEATSModel's forward method will handle squeezing the last dim if it's 1.

        # The y_train should be (batch_size, output_horizon_steps)
        # The NBEATS model output is (batch_size, output_horizon_steps)
        
        trainer.train_model(X_train, y_train, X_val, y_val)
        self.trained = True
        print(f"NBEATSPredictor {self.id_val} training complete.")

    def predict(self, X_current_state: Union[np.ndarray, torch.Tensor]) -> Dict[str, List[float]]:
        """
        Makes predictions using the trained N-BEATS model.
        Includes Monte Carlo Dropout for uncertainty quantification if dropout_prob > 0.
        Args:
            X_current_state (Union[np.ndarray, torch.Tensor]):
                Input data of shape (num_samples, input_chunk_length, num_features)
                or (num_samples, input_chunk_length) if univariate.
                If a single sample, shape (input_chunk_length, num_features) or (input_chunk_length).
        Returns:
            Dict[str, List[float]]: Dictionary with "mean" and "std_dev" of predictions.
                                     Each list contains `output_horizon_steps` values.
                                     If multiple samples are passed, this returns predictions for the first sample.
        """
        if not self.trained:
            print("Warning: NBEATSPredictor model has not been trained. Predictions might be random.")

        if not isinstance(X_current_state, torch.Tensor):
            X_current_state = torch.tensor(X_current_state, dtype=torch.float32)
        
        X_current_state = X_current_state.to(self.device)

        if X_current_state.ndim == 1: # (input_chunk_length)
            X_current_state = X_current_state.unsqueeze(0) # (1, input_chunk_length)
        if X_current_state.ndim == 2 and X_current_state.shape[0] != 1 : # (num_samples, input_chunk_length)
             print(f"NBEATSPredictor: Predicting for the first sample out of {X_current_state.shape[0]} provided samples.")
             X_current_state = X_current_state[0].unsqueeze(0) # Take first sample: (1, input_chunk_length)
        elif X_current_state.ndim == 2 and X_current_state.shape[0] == 1: # (1, input_chunk_length)
            pass # Already in correct shape
        elif X_current_state.ndim == 3 and X_current_state.shape[0] !=1 : # (num_samples, input_chunk_length, features)
            print(f"NBEATSPredictor: Predicting for the first sample out of {X_current_state.shape[0]} provided samples.")
            X_current_state = X_current_state[0].unsqueeze(0) # (1, input_chunk_length, features)
        elif X_current_state.ndim == 3 and X_current_state.shape[0] == 1: # (1, input_chunk_length, features)
            pass # Already in correct shape for model's forward
        else:
            raise ValueError(f"Unsupported input shape for NBEATS predict: {X_current_state.shape}")

        self.model.eval() # Set to evaluation mode for standard prediction

        if self.dropout_prob_config > 0 and self.mc_dropout_samples > 0:
            # Enable dropout layers for MC samples
            def activate_dropout(m):
                if type(m) == nn.Dropout:
                    m.train()
            self.model.apply(activate_dropout)
            
            mc_predictions = []
            with torch.no_grad(): # Still no grad for inference part of MC
                for _ in range(self.mc_dropout_samples):
                    pred = self.model(X_current_state) # (1, output_horizon_steps)
                    mc_predictions.append(pred.squeeze(0).cpu().numpy())
            
            mc_predictions_np = np.array(mc_predictions) # (mc_samples, output_horizon_steps)
            mean_preds = np.mean(mc_predictions_np, axis=0).tolist()
            std_dev_preds = np.std(mc_predictions_np, axis=0).tolist()
            
            self.model.eval() # Revert to standard eval mode
        else:
            # Standard prediction without MC Dropout
            with torch.no_grad():
                predictions = self.model(X_current_state) # (1, output_horizon_steps)
            mean_preds = predictions.squeeze(0).cpu().tolist()
            std_dev_preds = [0.0] * len(mean_preds) # No uncertainty estimate

        return {"mean": mean_preds, "std_dev": std_dev_preds}

    def save(self, path: str):
        """Saves the model's state_dict."""
        dir_name = os.path.dirname(path)
        if dir_name: # Only create directories if a path is specified
            if not os.path.exists(dir_name):
                os.makedirs(dir_name, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"NBEATSPredictor {self.id_val} model saved to {path}")

    def load(self, path: str):
        """Loads the model's state_dict."""
        if not os.path.exists(path):
            print(f"Error: Model file not found at {path}")
            return
        try:
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            self.model.to(self.device) # Ensure model is on the correct device
            self.trained = True # Assume a loaded model is trained
            print(f"NBEATSPredictor {self.id_val} model loaded from {path}")
        except Exception as e:
            print(f"Error loading NBEATSPredictor model from {path}: {e}")


if __name__ == '__main__':
    print("--- N-BEATS Predictor Example ---")
    
    # Parameters
    input_len = 24  # Lookback window (e.g., 24 hours of CGM data)
    output_len = 6 # Forecast horizon (e.g., 3 hours, if 1 step = 30 mins)
    num_features_input = 1 # Univariate for classic N-BEATS

    # Instantiate the NBEATSPredictor
    # Example: 1 trend stack, 1 seasonality stack, 1 generic stack
    example_stack_types = ['trend', 'seasonality', 'generic']
    nbeats_model = NBEATSPredictor(
        input_chunk_length=input_len,
        output_horizon_steps=output_len,
        stack_types=example_stack_types,
        num_blocks_per_stack=[1, 1, 2],  # Can be a list matching stack_types, or int
        hidden_layer_units=[128, 128, 64], # Example: different units per stack type, or int
        num_hidden_layers_per_block = 2, # Example: same for all stacks
        trend_polynomial_degree=2,
        seasonality_fourier_terms=3,
        generic_theta_dims=32,
        dropout_prob=0.1
    )

    # Create dummy data
    # X shape: (num_samples, input_chunk_length, num_features_input)
    # y shape: (num_samples, output_horizon_steps)
    num_samples = 100
    # NBEATS typically expects univariate input, so X_train_dummy will be (num_samples, input_len)
    # The model's forward pass will handle if it's (num_samples, input_len, 1)
    X_train_dummy = np.random.rand(num_samples, input_len).astype(np.float32)
    y_train_dummy = np.random.rand(num_samples, output_len).astype(np.float32)
    
    X_val_dummy = np.random.rand(num_samples // 2, input_len).astype(np.float32)
    y_val_dummy = np.random.rand(num_samples // 2, output_len).astype(np.float32)

    print(f"X_train_dummy shape: {X_train_dummy.shape}, y_train_dummy shape: {y_train_dummy.shape}")

    # Train the model
    train_params = {"epochs": 3, "batch_size": 16, "learning_rate": 0.005} # Quick test
    nbeats_model.train(X_train_dummy, y_train_dummy, X_val_dummy, y_val_dummy, training_params=train_params)

    # Make a prediction
    # X_pred_dummy shape: (1, input_len) or (1, input_len, 1)
    X_pred_dummy = np.random.rand(1, input_len).astype(np.float32)
    predictions = nbeats_model.predict(X_pred_dummy)
    print(f"\nPrediction for a dummy sample:")
    print(f"  Mean: {predictions['mean']}")
    print(f"  Std Dev: {predictions['std_dev']}")

    # Save and load
    model_path = "temp_nbeats_model.pth"
    nbeats_model.save(model_path)
    
    # For loading, ensure parameters match the saved model's architecture
    nbeats_model_loaded = NBEATSPredictor(
        input_chunk_length=input_len,
        output_horizon_steps=output_len,
        stack_types=example_stack_types,
        num_blocks_per_stack=[1, 1, 2],
        hidden_layer_units=[128, 128, 64],
        num_hidden_layers_per_block = 2,
        trend_polynomial_degree=2,
        seasonality_fourier_terms=3,
        generic_theta_dims=32,
        dropout_prob=0.1
    )
    nbeats_model_loaded.load(model_path)
    
    predictions_loaded = nbeats_model_loaded.predict(X_pred_dummy)
    print(f"\nPrediction from loaded model:")
    print(f"  Mean: {predictions_loaded['mean']}")
    print(f"  Std Dev: {predictions_loaded['std_dev']}")

    # Clean up
    if os.path.exists(model_path):
        os.remove(model_path)
    
    print("\n--- N-BEATS Predictor Example Run Complete ---")