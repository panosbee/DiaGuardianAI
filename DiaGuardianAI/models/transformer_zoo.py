#!/usr/bin/env python3
"""
DiaGuardianAI Transformer Zoo
Multiple advanced models for multi-horizon glucose prediction
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Any
from abc import ABC, abstractmethod
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
import warnings
warnings.filterwarnings('ignore')

from ..utils.metrics import calculate_rmse

class BasePredictor(ABC):
    """Base class for all prediction models in the zoo."""
    
    def __init__(self, name: str, input_dim: int, prediction_horizons: List[int]):
        self.name = name
        self.input_dim = input_dim
        self.prediction_horizons = prediction_horizons  # [10, 20, 30, 40, 50, 60, 90, 120] minutes
        self.is_trained = False
        self.scaler = StandardScaler()
        
    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train the model."""
        pass
    
    @abstractmethod
    def predict_multi_horizon(self, X: np.ndarray) -> Dict[int, np.ndarray]:
        """Predict glucose for multiple time horizons."""
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "name": self.name,
            "input_dim": self.input_dim,
            "prediction_horizons": self.prediction_horizons,
            "is_trained": self.is_trained
        }

class LSTMPredictor(BasePredictor):
    """LSTM-based glucose predictor."""
    
    def __init__(self, input_dim: int, prediction_horizons: List[int], hidden_dim: int = 128):
        super().__init__("LSTM_Predictor", input_dim, prediction_horizons)
        self.hidden_dim = hidden_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        
    def _build_model(self):
        """Build LSTM model."""
        class LSTMModel(nn.Module):
            def __init__(self, input_dim, hidden_dim, num_horizons):
                super().__init__()
                self.hidden_dim = hidden_dim
                self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, num_layers=2, dropout=0.2)
                self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
                self.fc_layers = nn.ModuleList([
                    nn.Linear(hidden_dim, 64),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(64, num_horizons)
                ])
                
            def forward(self, x):
                # LSTM processing
                lstm_out, _ = self.lstm(x)
                
                # Self-attention
                attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
                
                # Take last timestep
                last_hidden = attn_out[:, -1, :]
                
                # Feed through FC layers
                out = last_hidden
                for layer in self.fc_layers:
                    out = layer(out)
                
                return out
        
        self.model = LSTMModel(self.input_dim, self.hidden_dim, len(self.prediction_horizons))
        self.model.to(self.device)
        
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train LSTM model."""
        print(f"Training {self.name}...")
        
        # Normalize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Reshape for LSTM (batch, sequence, features)
        sequence_length = 12  # 1 hour of 5-minute intervals
        X_sequences = []
        y_sequences = []
        
        for i in range(sequence_length, len(X_scaled)):
            X_sequences.append(X_scaled[i-sequence_length:i])
            y_sequences.append(y[i])
        
        X_sequences = np.array(X_sequences)
        y_sequences = np.array(y_sequences)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_sequences).to(self.device)
        y_tensor = torch.FloatTensor(y_sequences).to(self.device)
        
        # Build model
        self._build_model()
        
        # Training setup
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        # Training loop
        self.model.train()
        losses = []
        
        batch_size = 256
        num_epochs = 100
        
        for epoch in range(num_epochs):
            epoch_loss = 0
            num_batches = 0
            
            for i in range(0, len(X_tensor), batch_size):
                batch_X = X_tensor[i:i+batch_size]
                batch_y = y_tensor[i:i+batch_size]
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            losses.append(avg_loss)
            scheduler.step(avg_loss)
            
            if epoch % 20 == 0:
                print(f"  Epoch {epoch}, Loss: {avg_loss:.4f}")
        
        self.is_trained = True
        
        return {
            "training_loss": losses[-1],
            "epochs": num_epochs,
            "final_lr": optimizer.param_groups[0]['lr']
        }
    
    def predict_multi_horizon(self, X: np.ndarray) -> Dict[int, np.ndarray]:
        """Predict glucose for multiple horizons."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Normalize input
        X_scaled = self.scaler.transform(X)
        
        # Create sequences
        sequence_length = 12
        if len(X_scaled) < sequence_length:
            # Pad if necessary
            padding = np.tile(X_scaled[0], (sequence_length - len(X_scaled), 1))
            X_scaled = np.vstack([padding, X_scaled])
        
        X_sequence = X_scaled[-sequence_length:].reshape(1, sequence_length, -1)
        X_tensor = torch.FloatTensor(X_sequence).to(self.device)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_tensor).cpu().numpy()[0]
        
        # Map to horizons
        horizon_predictions = {}
        for i, horizon in enumerate(self.prediction_horizons):
            horizon_predictions[horizon] = predictions[i]
        
        return horizon_predictions

class TransformerPredictor(BasePredictor):
    """Transformer-based glucose predictor."""
    
    def __init__(self, input_dim: int, prediction_horizons: List[int], d_model: int = 128):
        super().__init__("Transformer_Predictor", input_dim, prediction_horizons)
        self.d_model = d_model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        
    def _build_model(self):
        """Build Transformer model."""
        class TransformerModel(nn.Module):
            def __init__(self, input_dim, d_model, num_horizons):
                super().__init__()
                self.input_projection = nn.Linear(input_dim, d_model)
                self.positional_encoding = nn.Parameter(torch.randn(100, d_model))
                
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=8,
                    dim_feedforward=512,
                    dropout=0.1,
                    batch_first=True
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)
                
                self.output_projection = nn.Sequential(
                    nn.Linear(d_model, 256),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(128, num_horizons)
                )
                
            def forward(self, x):
                seq_len = x.size(1)
                
                # Project input
                x = self.input_projection(x)
                
                # Add positional encoding
                x = x + self.positional_encoding[:seq_len].unsqueeze(0)
                
                # Transformer encoding
                x = self.transformer(x)
                
                # Global average pooling
                x = x.mean(dim=1)
                
                # Output projection
                return self.output_projection(x)
        
        self.model = TransformerModel(self.input_dim, self.d_model, len(self.prediction_horizons))
        self.model.to(self.device)
    
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train Transformer model."""
        print(f"Training {self.name}...")
        
        # Similar training logic to LSTM but with Transformer
        X_scaled = self.scaler.fit_transform(X)
        
        sequence_length = 24  # 2 hours for transformer
        X_sequences = []
        y_sequences = []
        
        for i in range(sequence_length, len(X_scaled)):
            X_sequences.append(X_scaled[i-sequence_length:i])
            y_sequences.append(y[i])
        
        X_sequences = np.array(X_sequences)
        y_sequences = np.array(y_sequences)
        
        X_tensor = torch.FloatTensor(X_sequences).to(self.device)
        y_tensor = torch.FloatTensor(y_sequences).to(self.device)
        
        self._build_model()
        
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=0.0001, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        
        self.model.train()
        losses = []
        
        batch_size = 128
        num_epochs = 100
        
        for epoch in range(num_epochs):
            epoch_loss = 0
            num_batches = 0
            
            for i in range(0, len(X_tensor), batch_size):
                batch_X = X_tensor[i:i+batch_size]
                batch_y = y_tensor[i:i+batch_size]
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            losses.append(avg_loss)
            scheduler.step()
            
            if epoch % 20 == 0:
                print(f"  Epoch {epoch}, Loss: {avg_loss:.4f}")
        
        self.is_trained = True
        
        return {
            "training_loss": losses[-1],
            "epochs": num_epochs,
            "final_lr": optimizer.param_groups[0]['lr']
        }
    
    def predict_multi_horizon(self, X: np.ndarray) -> Dict[int, np.ndarray]:
        """Predict glucose for multiple horizons."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        X_scaled = self.scaler.transform(X)
        
        sequence_length = 24
        if len(X_scaled) < sequence_length:
            padding = np.tile(X_scaled[0], (sequence_length - len(X_scaled), 1))
            X_scaled = np.vstack([padding, X_scaled])
        
        X_sequence = X_scaled[-sequence_length:].reshape(1, sequence_length, -1)
        X_tensor = torch.FloatTensor(X_sequence).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_tensor).cpu().numpy()[0]
        
        horizon_predictions = {}
        for i, horizon in enumerate(self.prediction_horizons):
            horizon_predictions[horizon] = predictions[i]
        
        return horizon_predictions

class EnsemblePredictor(BasePredictor):
    """Ensemble of multiple models for robust predictions."""
    
    def __init__(self, input_dim: int, prediction_horizons: List[int]):
        super().__init__("Ensemble_Predictor", input_dim, prediction_horizons)
        
        # Simple models for ensemble
        self.models = {
            "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "ridge": Ridge(alpha=1.0),
        }
        self.model_weights = {}
        
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train ensemble models."""
        print(f"Training {self.name}...")
        
        X_scaled = self.scaler.fit_transform(X)
        
        # Train each model for each horizon
        training_results = {}
        
        for i, horizon in enumerate(self.prediction_horizons):
            y_horizon = y[:, i] if y.ndim > 1 else y  # Handle single vs multi-output
            
            horizon_models = {}
            horizon_scores = {}
            
            for name, model in self.models.items():
                model_copy = type(model)(**model.get_params())
                model_copy.fit(X_scaled, y_horizon)
                
                # Simple validation score
                score = model_copy.score(X_scaled, y_horizon)
                
                horizon_models[name] = model_copy
                horizon_scores[name] = score
                
                print(f"  {name} for {horizon}min: R² = {score:.3f}")
            
            # Calculate weights based on performance
            total_score = sum(horizon_scores.values())
            weights = {name: score/total_score for name, score in horizon_scores.items()}
            
            training_results[horizon] = {
                "models": horizon_models,
                "weights": weights,
                "scores": horizon_scores
            }
        
        self.training_results = training_results
        self.is_trained = True
        
        return {"ensemble_results": training_results}
    
    def predict_multi_horizon(self, X: np.ndarray) -> Dict[int, np.ndarray]:
        """Predict using ensemble."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        X_scaled = self.scaler.transform(X)
        
        horizon_predictions = {}
        
        for horizon in self.prediction_horizons:
            horizon_data = self.training_results[horizon]
            models = horizon_data["models"]
            weights = horizon_data["weights"]
            
            # Weighted ensemble prediction
            ensemble_pred = 0
            for name, model in models.items():
                pred = model.predict(X_scaled)
                ensemble_pred += weights[name] * pred
            
            horizon_predictions[horizon] = ensemble_pred
        
        return horizon_predictions

class TransformerZoo:
    """Zoo of transformer and ML models for glucose prediction."""
    
    def __init__(self, input_dim: int = 16):
        self.input_dim = input_dim
        self.prediction_horizons = [10, 20, 30, 40, 50, 60, 90, 120]  # minutes
        
        # Initialize model zoo
        self.models = {
            "lstm": LSTMPredictor(input_dim, self.prediction_horizons),
            "transformer": TransformerPredictor(input_dim, self.prediction_horizons),
            "ensemble": EnsemblePredictor(input_dim, self.prediction_horizons)
        }
        
        self.performance_history = {}
        
        print(f"🦁 TransformerZoo initialized with {len(self.models)} models")
        print(f"  Prediction horizons: {self.prediction_horizons} minutes")
        print(f"  Models: {list(self.models.keys())}")
    
    def train_all_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train all models in the zoo."""
        print(f"\n🎯 Training all models in TransformerZoo...")
        
        training_results = {}
        
        for name, model in self.models.items():
            try:
                print(f"\n📈 Training {name}...")
                result = model.train(X, y)
                training_results[name] = result
                print(f"  ✅ {name} training complete")
            except Exception as e:
                print(f"  ❌ {name} training failed: {str(e)}")
                training_results[name] = {"error": str(e)}
        
        print(f"\n🎉 TransformerZoo training complete!")
        return training_results
    
    def predict_all_models(self, X: np.ndarray) -> Dict[str, Dict[int, np.ndarray]]:
        """Get predictions from all trained models."""
        all_predictions = {}
        
        for name, model in self.models.items():
            if model.is_trained:
                try:
                    predictions = model.predict_multi_horizon(X)
                    all_predictions[name] = predictions
                except Exception as e:
                    print(f"Prediction error for {name}: {str(e)}")
                    all_predictions[name] = {}
        
        return all_predictions
    
    def get_best_model_for_horizon(self, horizon: int) -> Optional[str]:
        """Get the best performing model for a specific horizon."""
        if horizon not in self.prediction_horizons:
            return None
        
        # Simple heuristic: return transformer for longer horizons, LSTM for shorter
        if horizon >= 60:
            return "transformer"
        elif horizon >= 30:
            return "lstm"
        else:
            return "ensemble"
    
    def get_zoo_status(self) -> Dict[str, Any]:
        """Get status of all models in the zoo."""
        status = {
            "total_models": len(self.models),
            "trained_models": sum(1 for model in self.models.values() if model.is_trained),
            "prediction_horizons": self.prediction_horizons,
            "model_status": {}
        }
        
        for name, model in self.models.items():
            status["model_status"][name] = model.get_model_info()

        return status

    def evaluate_all_models(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Dict[str, Dict[int, float]]:
        """Evaluate all trained models using RMSE for each horizon."""

        results: Dict[str, Dict[int, float]] = {}

        for name, model in self.models.items():
            if not model.is_trained:
                continue

            try:
                preds = []
                for i in range(len(X)):
                    horizon_preds = model.predict_multi_horizon(X[i].reshape(1, -1))
                    preds.append([horizon_preds[h] for h in self.prediction_horizons])

                preds_arr = np.array(preds)
                model_results: Dict[int, float] = {}
                for idx, horizon in enumerate(self.prediction_horizons):
                    model_results[horizon] = calculate_rmse(y[:, idx], preds_arr[:, idx])

                results[name] = model_results

            except Exception as e:
                print(f"Evaluation error for {name}: {e}")
                results[name] = {}

        return results
