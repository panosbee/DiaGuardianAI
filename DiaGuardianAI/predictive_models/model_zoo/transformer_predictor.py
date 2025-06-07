# DiaGuardianAI Transformer Predictor
# A Transformer-based model for blood glucose prediction.

import sys
import os
import math
from typing import List, Any, Dict, Optional
import torch
import torch.nn as nn
import numpy as np

# Ensure the DiaGuardianAI package is discoverable when run as a script
if __package__ is None or __package__ == '':
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from DiaGuardianAI.core.base_classes import BasePredictiveModel
from DiaGuardianAI.predictive_models.model_trainer import ModelTrainer # Import ModelTrainer

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model) # Batch dimension first for register_buffer
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe) # pe is [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim] (batch_first=True)
        """
        # x is [batch_size, seq_len, d_model]
        # self.pe is [1, max_len, d_model]
        # We need to add pe[:, :x.size(1)] to x, broadcasting over batch
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class PyTorchTransformerModel(nn.Module):
    def __init__(self, input_dim: int, model_dim: int, num_heads: int,
                 num_encoder_layers: int, output_dim: int,
                 dim_feedforward: int, dropout_prob: float, max_seq_len: int = 500): # Added max_seq_len
        super().__init__()
        self.model_dim = model_dim
        self.input_fc = nn.Linear(input_dim, model_dim)
        self.pos_encoder = PositionalEncoding(model_dim, dropout_prob, max_len=max_seq_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=num_heads,
            dim_feedforward=dim_feedforward, dropout=dropout_prob, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )
        self.output_fc = nn.Linear(model_dim, output_dim)
        self.dropout = nn.Dropout(dropout_prob) # For MC Dropout on output

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        src = self.input_fc(src) * math.sqrt(self.model_dim) # Scale input
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        # Use the output of the last time step for prediction
        output = output[:, -1, :]
        output = self.dropout(output) # Apply dropout for MC
        output = self.output_fc(output)
        return output

class TransformerPredictor(BasePredictiveModel):
    """Transformer-based predictor with MC Dropout for UQ."""
    def __init__(self, input_dim: int, model_dim: int, num_heads: int,
                 num_encoder_layers: int, output_horizon_steps: int,
                 dim_feedforward: int = 2048, dropout_prob: float = 0.1,
                 mc_dropout_samples: int = 50, max_seq_len: int = 500): # Added max_seq_len
        super().__init__()
        self.input_dim = input_dim
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers
        self.output_horizon_steps = output_horizon_steps
        self.dim_feedforward = dim_feedforward
        self.dropout_prob = dropout_prob
        self.mc_dropout_samples = mc_dropout_samples
        self.max_seq_len = max_seq_len
        self.training_history: List[Dict[str, Any]] = [] # To store training history

        self.model = PyTorchTransformerModel(
            input_dim=self.input_dim, model_dim=self.model_dim, num_heads=self.num_heads,
            num_encoder_layers=self.num_encoder_layers, output_dim=self.output_horizon_steps,
            dim_feedforward=self.dim_feedforward, dropout_prob=self.dropout_prob,
            max_seq_len=self.max_seq_len
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(
            f"TransformerPredictor initialized: input_dim={self.input_dim}, "
            f"model_dim={self.model_dim}, heads={self.num_heads}, "
            f"layers={self.num_encoder_layers}, output_steps={self.output_horizon_steps}, "
            f"dropout={self.dropout_prob}, mc_samples={self.mc_dropout_samples}, device={self.device}"
        )

    def train(self, X_train: Any, y_train: Any,
              X_val: Optional[Any] = None, y_val: Optional[Any] = None,
              training_params: Optional[Dict[str, Any]] = None):
        """Trains the Transformer model using ModelTrainer."""
        print(f"TransformerPredictor train method called. X_train shape: {getattr(X_train, 'shape', 'N/A')}, y_train shape: {getattr(y_train, 'shape', 'N/A')}")
        
        default_trainer_params = {"epochs": 10, "batch_size": 32, "learning_rate": 0.001}
        current_training_params = training_params if training_params is not None else default_trainer_params

        trainer = ModelTrainer(model=self, training_params=current_training_params)
        
        history = trainer.train_model(X_train, y_train, X_val, y_val)
        self.training_history = history.get("history", [])
        print(f"TransformerPredictor training finished. History: {self.training_history[-1] if self.training_history else 'No history'}")


    def predict(self, X_current_state: Any) -> Dict[str, List[float]]:
        """Makes predictions with MC Dropout for uncertainty."""
        self.model.train() # Enable dropout layers for MC Dropout
        
        if not isinstance(X_current_state, torch.Tensor):
            X_current_state = torch.tensor(X_current_state, dtype=torch.float32)
        
        X_current_state = X_current_state.to(self.device)

        if X_current_state.ndim == 2: # (sequence_length, n_features)
            X_current_state = X_current_state.unsqueeze(0) # Add batch dimension
        
        if X_current_state.shape[0] != 1:
            print("Warning: MC Dropout predict called with batch size > 1. Processing first sample only.")
            X_current_state = X_current_state[0].unsqueeze(0)
        
        # Transformer typically doesn't need src_mask for encoder-only if not padding
        # src_key_padding_mask might be needed if sequences have padding. Assuming no padding for now.
        with torch.no_grad():
            predictions_mc = []
            for _ in range(self.mc_dropout_samples):
                pred = self.model(X_current_state) # src_mask and src_key_padding_mask are None by default
                predictions_mc.append(pred.squeeze().cpu().numpy())
        
        predictions_mc_np = np.array(predictions_mc)
        
        mean_predictions = np.mean(predictions_mc_np, axis=0).tolist()
        std_dev_predictions = np.std(predictions_mc_np, axis=0).tolist()
        
        return {"mean": mean_predictions, "std_dev": std_dev_predictions}

    def save(self, path: str):
        """Saves the model's state dictionary."""
        torch.save(self.model.state_dict(), path)
        print(f"TransformerPredictor model saved to {path}")

    def load(self, path: str):
        """Loads the model's state dictionary."""
        if self.model is None:
            print("Error: Model not initialized before loading. Re-initializing.")
            self.model = PyTorchTransformerModel(
                input_dim=self.input_dim, model_dim=self.model_dim, num_heads=self.num_heads,
                num_encoder_layers=self.num_encoder_layers, output_dim=self.output_horizon_steps,
                dim_feedforward=self.dim_feedforward, dropout_prob=self.dropout_prob,
                max_seq_len=self.max_seq_len
            )
            self.model.to(self.device)
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
        print(f"TransformerPredictor model loaded from {path} and set to eval mode.")


if __name__ == '__main__':
    predictor = TransformerPredictor(
        input_dim=7, model_dim=64, num_heads=4, num_encoder_layers=2, # Smaller for faster dummy run
        output_horizon_steps=12, dim_feedforward=256, dropout_prob=0.1,
        max_seq_len=36
    )

    dummy_input_dim_example = predictor.input_dim
    dummy_sequence_length_example = 36
    dummy_output_horizon_example = predictor.output_horizon_steps

    # Dummy training data
    X_train_dummy = torch.randn(100, dummy_sequence_length_example, dummy_input_dim_example)
    y_train_dummy = torch.randn(100, dummy_output_horizon_example)
    
    # Dummy validation data (optional)
    X_val_dummy = torch.randn(20, dummy_sequence_length_example, dummy_input_dim_example)
    y_val_dummy = torch.randn(20, dummy_output_horizon_example)

    print(f"\n--- Training TransformerPredictor (dummy data) ---")
    trainer_params_example = {"epochs": 2, "batch_size": 16, "learning_rate": 0.005} # Slightly different params
    predictor.train(X_train_dummy, y_train_dummy, X_val_dummy, y_val_dummy, training_params=trainer_params_example)
    
    # Dummy prediction data
    X_current_dummy = torch.randn(1, dummy_sequence_length_example, dummy_input_dim_example)
    
    print(f"\n--- Predicting with TransformerPredictor (dummy data) ---")
    print(f"Predicting with dummy input of shape: {X_current_dummy.shape}")
    predictions_dict = predictor.predict(X_current_dummy)
    
    print(f"\nDummy predictions (mean): {predictions_dict['mean']}")
    print(f"Dummy predictions (std_dev): {predictions_dict['std_dev']}")

    print("\n--- Saving and Loading TransformerPredictor (dummy) ---")
    model_path = "dummy_transformer_model.pth"
    predictor.save(model_path)
    
    new_predictor = TransformerPredictor(
        input_dim=dummy_input_dim_example, model_dim=64, num_heads=4, num_encoder_layers=2,
        output_horizon_steps=dummy_output_horizon_example, dim_feedforward=256, dropout_prob=0.1,
        max_seq_len=dummy_sequence_length_example
    )
    new_predictor.load(model_path)
    
    print("\n--- Predicting with loaded TransformerPredictor (dummy data) ---")
    loaded_predictions_dict = new_predictor.predict(X_current_dummy)
    print(f"Loaded model predictions (mean): {loaded_predictions_dict['mean']}")
    print(f"Loaded model predictions (std_dev): {loaded_predictions_dict['std_dev']}")

    if os.path.exists(model_path):
        os.remove(model_path)
        
    print("\nTransformerPredictor example run complete.")