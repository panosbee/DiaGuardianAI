# DiaGuardianAI - TSMixer Predictor
# An All-MLP Architecture for Time Series Forecasting

import sys
import os
from typing import Dict, Any, List, Optional, Union

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

# --- TSMixer Core Components ---

class TimeMixingMLP(nn.Module):
    """MLP for mixing information across the time dimension."""
    def __init__(self, input_dim: int, hidden_dim: int, dropout_prob: float):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, input_dim), # Project back to original time dimension size
            nn.Dropout(dropout_prob)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, num_features, seq_len)
        # We want to mix across seq_len for each feature independently.
        # So, transpose to (batch_size, num_features, seq_len) -> (batch_size * num_features, seq_len)
        # then apply MLP, then transpose back.
        # However, TSMixer applies MLP directly on transposed input.
        # Input x: (batch_size, seq_len, num_features)
        # Transpose for time mixing: (batch_size, num_features, seq_len)
        x_transposed = x.transpose(1, 2) # (batch_size, num_features, seq_len)
        out = self.mlp(x_transposed)     # MLP applied on last dim (seq_len)
        return out.transpose(1, 2)       # (batch_size, seq_len, num_features)

class FeatureMixingMLP(nn.Module):
    """MLP for mixing information across the feature dimension."""
    def __init__(self, input_dim: int, hidden_dim: int, dropout_prob: float):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, input_dim), # Project back to original feature dimension size
            nn.Dropout(dropout_prob)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, num_features)
        # MLP applied on last dim (num_features)
        return self.mlp(x)

class TSMixerLayer(nn.Module):
    """A single layer of TSMixer, containing a TimeMixingMLP and a FeatureMixingMLP."""
    def __init__(self, seq_len: int, num_features: int, time_mlp_hidden_dim: int, feature_mlp_hidden_dim: int, dropout_prob: float):
        super().__init__()
        # LayerNorm is applied on the last dimension (num_features)
        self.norm_time = nn.LayerNorm(num_features)
        self.time_mixer = TimeMixingMLP(seq_len, time_mlp_hidden_dim, dropout_prob)
        
        self.norm_feature = nn.LayerNorm(num_features)
        self.feature_mixer = FeatureMixingMLP(num_features, feature_mlp_hidden_dim, dropout_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, num_features)
        
        # Time Mixing Block
        residual = x
        x_norm_time = self.norm_time(x)
        x_time_mixed = self.time_mixer(x_norm_time) # Operates on transposed data internally
        x = x_time_mixed + residual # First residual connection
        
        # Feature Mixing Block
        residual = x
        x_norm_feature = self.norm_feature(x)
        x_feature_mixed = self.feature_mixer(x_norm_feature)
        x = x_feature_mixed + residual # Second residual connection
        
        return x

class PyTorchTSMixerModel(nn.Module):
    """
    The TSMixer model architecture.
    """
    def __init__(self,
                 seq_len: int,          # Input sequence length (lookback window)
                 pred_len: int,         # Prediction length (forecast horizon)
                 num_features: int,     # Number of input features
                 num_mixer_layers: int,
                 time_mlp_hidden_dim: int,  # Hidden dim for TimeMixingMLP (operates on seq_len)
                 feature_mlp_hidden_dim: int, # Hidden dim for FeatureMixingMLP (operates on num_features)
                 dropout_prob: float):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_features = num_features

        self.mixer_layers = nn.ModuleList([
            TSMixerLayer(seq_len, num_features, time_mlp_hidden_dim, feature_mlp_hidden_dim, dropout_prob)
            for _ in range(num_mixer_layers)
        ])
        
        # Final projection layer to map from num_features to pred_len * num_output_features
        # For univariate forecast (predicting one target variable), num_output_features = 1
        # The TSMixer paper often flattens the seq_len * num_features for this projection.
        # Or projects features to a new dimension then flattens time.
        # Let's project the features from the last time step.
        # Or, more commonly, project from all time steps then average or use a specific one.
        # The paper uses a Linear layer that maps (seq_len * num_features) to (pred_len * num_target_features)
        # For simplicity, let's try projecting the features of the *last* time step.
        # self.projection = nn.Linear(num_features, pred_len) # Predicts pred_len steps for one feature

        # Alternative: Flatten the time dimension and features, then project
        # This is more aligned with some TSMixer implementations.
        self.flatten_projection = nn.Linear(seq_len * num_features, pred_len)


    def forward(self, x_input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_input (torch.Tensor): Input tensor of shape (batch_size, seq_len, num_features).
        Returns:
            torch.Tensor: Forecast tensor of shape (batch_size, pred_len)
        """
        if x_input.ndim != 3 or x_input.shape[1] != self.seq_len or x_input.shape[2] != self.num_features:
            raise ValueError(f"TSMixer expects input shape (batch, {self.seq_len}, {self.num_features}). Got {x_input.shape}")

        x = x_input
        for layer in self.mixer_layers:
            x = layer(x)
        
        # Projection to prediction length
        # Option 1: Use only the last time step's features
        # x_last_step = x[:, -1, :] # (batch_size, num_features)
        # forecast = self.projection(x_last_step) # (batch_size, pred_len)

        # Option 2: Flatten and project (more common in TSMixer literature)
        x_flat = x.reshape(x.shape[0], -1) # (batch_size, seq_len * num_features)
        forecast = self.flatten_projection(x_flat) # (batch_size, pred_len)
            
        return forecast


class TSMixerPredictor(BasePredictiveModel):
    """
    Wrapper for the PyTorchTSMixerModel, conforming to BasePredictiveModel interface.
    """
    def __init__(self,
                 input_chunk_length: int, # seq_len
                 output_horizon_steps: int, # pred_len
                 num_input_features: int, # num_features
                 num_mixer_layers: int = 2, # Default from some TSMixer papers
                 time_mlp_hidden_dim_ratio: float = 1.0, # Ratio to seq_len
                 feature_mlp_hidden_dim_ratio: float = 1.0, # Ratio to num_features
                 dropout_prob: float = 0.1,
                 mc_dropout_samples: int = 50,
                 id_val: int = 0):
        super().__init__()
        self.input_chunk_length = input_chunk_length
        self.output_horizon_steps = output_horizon_steps
        self.num_input_features = num_input_features
        self.num_mixer_layers = num_mixer_layers
        self.time_mlp_hidden_dim_ratio = time_mlp_hidden_dim_ratio
        self.feature_mlp_hidden_dim_ratio = feature_mlp_hidden_dim_ratio
        self.dropout_prob = dropout_prob
        self.mc_dropout_samples = mc_dropout_samples
        self.id_val = id_val

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        time_mlp_hidden_dim = int(input_chunk_length * time_mlp_hidden_dim_ratio)
        feature_mlp_hidden_dim = int(num_input_features * feature_mlp_hidden_dim_ratio)

        self.model = PyTorchTSMixerModel(
            seq_len=self.input_chunk_length,
            pred_len=self.output_horizon_steps,
            num_features=self.num_input_features,
            num_mixer_layers=self.num_mixer_layers,
            time_mlp_hidden_dim=time_mlp_hidden_dim,
            feature_mlp_hidden_dim=feature_mlp_hidden_dim,
            dropout_prob=self.dropout_prob
        ).to(self.device)

        self.trained = False
        print(f"TSMixerPredictor {self.id_val} initialized on {self.device}.")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")

    def train(self,
              X_train: Union[np.ndarray, torch.Tensor],
              y_train: Union[np.ndarray, torch.Tensor],
              X_val: Optional[Union[np.ndarray, torch.Tensor]] = None,
              y_val: Optional[Union[np.ndarray, torch.Tensor]] = None,
              training_params: Optional[Dict[str, Any]] = None):
        print(f"TSMixerPredictor {self.id_val} train called.")
        default_training_params = {"epochs": 50, "batch_size": 32, "learning_rate": 0.001}
        if training_params: default_training_params.update(training_params)

        trainer = ModelTrainer(model=self, training_params=default_training_params)
        # TSMixer expects X_train: (batch, seq_len, features), y_train: (batch, pred_len)
        trainer.train_model(X_train, y_train, X_val, y_val)
        self.trained = True
        print(f"TSMixerPredictor {self.id_val} training complete.")

    def predict(self, X_current_state: Union[np.ndarray, torch.Tensor]) -> Dict[str, List[float]]:
        if not self.trained:
            print("Warning: TSMixerPredictor model has not been trained.")

        if not isinstance(X_current_state, torch.Tensor):
            X_current_state = torch.tensor(X_current_state, dtype=torch.float32)
        
        X_current_state = X_current_state.to(self.device)

        # Ensure X_current_state is (batch_size, seq_len, num_features)
        if X_current_state.ndim == 2: # (seq_len, num_features) for a single sample
            X_current_state = X_current_state.unsqueeze(0) # (1, seq_len, num_features)
        
        if X_current_state.shape[0] > 1:
             print(f"TSMixerPredictor: Predicting for the first sample out of {X_current_state.shape[0]} provided samples.")
             X_current_state = X_current_state[0].unsqueeze(0)

        self.model.eval()
        if self.dropout_prob > 0 and self.mc_dropout_samples > 0:
            def activate_dropout(m):
                if isinstance(m, nn.Dropout): m.train()
            self.model.apply(activate_dropout)
            
            mc_predictions = []
            with torch.no_grad():
                for _ in range(self.mc_dropout_samples):
                    pred = self.model(X_current_state)
                    mc_predictions.append(pred.squeeze(0).cpu().numpy())
            
            mc_predictions_np = np.array(mc_predictions)
            mean_preds = np.mean(mc_predictions_np, axis=0).tolist()
            std_dev_preds = np.std(mc_predictions_np, axis=0).tolist()
            self.model.eval() # Revert to standard eval mode
        else:
            with torch.no_grad():
                predictions = self.model(X_current_state)
            mean_preds = predictions.squeeze(0).cpu().tolist()
            std_dev_preds = [0.0] * len(mean_preds)

        if not isinstance(mean_preds, list): mean_preds = [mean_preds] # Ensure list for single step horizon
        if not isinstance(std_dev_preds, list): std_dev_preds = [std_dev_preds]

        return {"mean": mean_preds, "std_dev": std_dev_preds}

    def save(self, path: str):
        dir_name = os.path.dirname(path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"TSMixerPredictor {self.id_val} model saved to {path}")

    def load(self, path: str):
        if not os.path.exists(path):
            print(f"Error: Model file not found at {path}")
            return
        try:
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            self.model.to(self.device)
            self.trained = True
            print(f"TSMixerPredictor {self.id_val} model loaded from {path}")
        except Exception as e:
            print(f"Error loading TSMixerPredictor model from {path}: {e}")


if __name__ == '__main__':
    print("--- TSMixer Predictor Example ---")
    
    seq_len = 24
    pred_len = 6
    num_features = 5 # Example: CGM, IOB, COB, exercise, meal carbs

    tsmixer_model = TSMixerPredictor(
        input_chunk_length=seq_len,
        output_horizon_steps=pred_len,
        num_input_features=num_features,
        num_mixer_layers=2,
        time_mlp_hidden_dim_ratio=0.5, # Smaller for test
        feature_mlp_hidden_dim_ratio=0.5, # Smaller for test
        dropout_prob=0.1
    )

    num_samples = 100
    X_train_dummy = np.random.rand(num_samples, seq_len, num_features).astype(np.float32)
    y_train_dummy = np.random.rand(num_samples, pred_len).astype(np.float32)
    X_val_dummy = np.random.rand(num_samples // 2, seq_len, num_features).astype(np.float32)
    y_val_dummy = np.random.rand(num_samples // 2, pred_len).astype(np.float32)

    print(f"X_train_dummy shape: {X_train_dummy.shape}, y_train_dummy shape: {y_train_dummy.shape}")

    train_params = {"epochs": 3, "batch_size": 16, "learning_rate": 0.005}
    tsmixer_model.train(X_train_dummy, y_train_dummy, X_val_dummy, y_val_dummy, training_params=train_params)

    X_pred_dummy = np.random.rand(1, seq_len, num_features).astype(np.float32)
    predictions = tsmixer_model.predict(X_pred_dummy)
    print(f"\nPrediction for a dummy sample:")
    print(f"  Mean: {predictions['mean']}")
    print(f"  Std Dev: {predictions['std_dev']}")

    model_path = "temp_tsmixer_model.pth"
    tsmixer_model.save(model_path)
    
    tsmixer_model_loaded = TSMixerPredictor(
        input_chunk_length=seq_len, output_horizon_steps=pred_len, num_input_features=num_features,
        num_mixer_layers=2, time_mlp_hidden_dim_ratio=0.5, feature_mlp_hidden_dim_ratio=0.5, dropout_prob=0.1
    )
    tsmixer_model_loaded.load(model_path)
    
    predictions_loaded = tsmixer_model_loaded.predict(X_pred_dummy)
    print(f"\nPrediction from loaded model:")
    print(f"  Mean: {predictions_loaded['mean']}")
    print(f"  Std Dev: {predictions_loaded['std_dev']}")

    if os.path.exists(model_path):
        os.remove(model_path)
    
    print("\n--- TSMixer Predictor Example Run Complete ---")