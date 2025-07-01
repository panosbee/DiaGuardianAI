# DiaGuardianAI - LSTM Predictive Model

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler # For data scaling
import joblib # For saving/loading scalers
import json # For saving/loading config
import os

from DiaGuardianAI.core.base_classes import BasePredictiveModel

# Define the PyTorch LSTM Model class
class LSTMNetwork(nn.Module):
    def __init__(self, input_size, hidden_units, num_layers, output_size, dropout_rate):
        super(LSTMNetwork, self).__init__()
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_units,
            num_layers=num_layers,
            batch_first=True, # Input tensor format: (batch, seq_len, features)
            dropout=dropout_rate if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_units, output_size)

    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_units).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_units).to(x.device)
        
        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_units)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :]) # Take output of the last time step
        return out

class LSTMPredictor(BasePredictiveModel):
    """
    LSTM-based model for glucose prediction.
    """
    def __init__(self,
                 input_seq_len: int = 12, # e.g., 12 * 5 min = 1 hour of history
                 output_seq_len: int = 6, # e.g., 6 * 5 min = 30 min prediction horizon
                 n_features: int = 3,     # e.g., CGM, Bolus Insulin, Carbs
                 hidden_units: int = 50,
                 num_layers: int = 1,
                 dropout_rate: float = 0.2,
                 learning_rate: float = 0.001,
                 **kwargs):
        """
        Initializes the LSTM Predictive Model.

        Args:
            input_seq_len (int): Length of the input sequence (e.g., number of past time steps).
            output_seq_len (int): Length of the output sequence (prediction horizon).
            n_features (int): Number of features in the input data (e.g., CGM, insulin, carbs).
            hidden_units (int): Number of units in LSTM hidden layers.
            num_layers (int): Number of LSTM layers.
            dropout_rate (float): Dropout rate for regularization.
            learning_rate (float): Learning rate for the optimizer.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len
        self.n_features = n_features
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        
        self.model: Optional[LSTMNetwork] = None
        self.scaler_X: Optional[MinMaxScaler] = None # Scaler for input features
        self.scaler_y: Optional[MinMaxScaler] = None # Scaler for target (CGM)
        self.is_trained: bool = False
        self.training_history: List[Dict[str, float]] = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self._build_model()
        print(f"LSTMPredictor initialized on {self.device}: input_seq_len={input_seq_len}, output_seq_len={output_seq_len}, n_features={n_features}")

    def _build_model(self):
        """
        Defines and builds the LSTM model architecture using PyTorch.
        """
        self.model = LSTMNetwork(
            input_size=self.n_features,
            hidden_units=self.hidden_units,
            num_layers=self.num_layers,
            output_size=self.output_seq_len, # Predicting a sequence of future CGM values
            dropout_rate=self.dropout_rate
        ).to(self.device)
        print(f"LSTMNetwork built and moved to {self.device}.")

    def _prepare_data_for_lstm(self, data_df: pd.DataFrame, fit_scalers: bool = False) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Prepares raw data from a DataFrame into sequences for LSTM.

        Args:
            data_df (pd.DataFrame): DataFrame containing time-series data.
                                 Expected columns: 'cgm_mg_dl', 'bolus_U', 'carbs_g'.
            fit_scalers (bool): If True, fit the scalers. This should only be done
                                with the training dataset.

        Returns:
            tuple[Optional[np.ndarray], Optional[np.ndarray]]:
                (X_sequences, y_sequences) or (None, None) if not enough data.
                X_sequences shape: (n_samples, input_seq_len, n_features)
                y_sequences shape: (n_samples, output_seq_len)
        """
        features_to_use = ['cgm_mg_dl', 'bolus_U', 'carbs_g'] # Ensure n_features matches this
        if self.n_features != len(features_to_use):
            raise ValueError(f"n_features ({self.n_features}) in constructor does not match selected features ({len(features_to_use)}).")

        if not all(col in data_df.columns for col in features_to_use):
            missing_cols = [col for col in features_to_use if col not in data_df.columns]
            print(f"Warning: Missing expected columns in data: {missing_cols}. Skipping this DataFrame.")
            return None, None
            
        data_values = data_df[features_to_use].values.astype(np.float32)
        cgm_values = data_df['cgm_mg_dl'].values.astype(np.float32).reshape(-1, 1)

        if fit_scalers:
            self.scaler_X = MinMaxScaler(feature_range=(0, 1))
            self.scaler_y = MinMaxScaler(feature_range=(0, 1))
            # Fit scalers only on the first batch of training data or a representative sample
            # For simplicity here, we'll fit if it's the first time and fit_scalers is true.
            # In a more robust setup, you'd aggregate all training data before fitting scalers.
            self.scaler_X.fit(data_values)
            self.scaler_y.fit(cgm_values)
            print("Scalers fitted.")

        if not self.scaler_X or not self.scaler_y:
            print("Warning: Scalers not fitted. Call with fit_scalers=True on training data first. Skipping.")
            return None, None

        scaled_X_data = self.scaler_X.transform(data_values)
        scaled_y_data = self.scaler_y.transform(cgm_values)

        X_sequences, y_sequences = [], []
        
        # Total length needed for one sample: input_seq_len for features, output_seq_len for target lookahead
        total_len_needed = self.input_seq_len + self.output_seq_len
        
        if len(scaled_X_data) < total_len_needed:
            # print(f"Not enough data in this segment to create sequences (need {total_len_needed}, got {len(scaled_X_data)}).")
            return None, None

        for i in range(len(scaled_X_data) - total_len_needed + 1):
            X_sequences.append(scaled_X_data[i : i + self.input_seq_len])
            y_sequences.append(scaled_y_data[i + self.input_seq_len : i + self.input_seq_len + self.output_seq_len].flatten())
            
        if not X_sequences: # Should be redundant due to earlier check, but good practice
            return None, None
            
        return np.array(X_sequences), np.array(y_sequences)


    def train(self, data: List[pd.DataFrame], targets: Optional[Any] = None, **kwargs):
        """
        Trains the LSTM model.

        Args:
            data (List[pd.DataFrame]): List of DataFrames, each from a patient simulation run,
                                       containing features and implicitly targets for future steps.
            targets (Optional[Any]): Not directly used here as targets are derived from `data`.
                                     Kept for compatibility with BasePredictiveModel.
            **kwargs: Additional training parameters like 'epochs', 'batch_size', 'validation_data'.
        """
        epochs = kwargs.get('epochs', 10)
        batch_size = kwargs.get('batch_size', 32)
        validation_data = kwargs.get('validation_data', None)

        if not self.model:
            self._build_model() # Ensure model is built

        print(f"Starting LSTM training for {epochs} epochs with batch_size {batch_size}.")
        
        # Aggregate and preprocess all training data
        all_X_train_list = []
        all_y_train_list = []
        
        features_to_use = ['cgm_mg_dl', 'bolus_U', 'carbs_g'] # Consistent with _prepare_data_for_lstm

        # --- Step 1: Aggregate all data for fitting scalers ---
        if not self.scaler_X or not self.scaler_y: # Fit scalers if not already done (e.g., from a loaded model)
            print("Preparing to fit scalers on the entire training dataset.")
            all_feature_data_for_scaling = []
            all_cgm_data_for_scaling = []
            
            for df in data:
                if not all(col in df.columns for col in features_to_use):
                    print(f"Warning: DataFrame missing required columns for scaling. Skipping this segment for scaler fitting.")
                    continue
                all_feature_data_for_scaling.append(df[features_to_use].values.astype(np.float32))
                all_cgm_data_for_scaling.append(df['cgm_mg_dl'].values.astype(np.float32).reshape(-1, 1))

            if not all_feature_data_for_scaling or not all_cgm_data_for_scaling:
                print("Not enough valid data to fit scalers. Aborting training.")
                return

            concatenated_features = np.concatenate(all_feature_data_for_scaling, axis=0)
            concatenated_cgm = np.concatenate(all_cgm_data_for_scaling, axis=0)

            self.scaler_X = MinMaxScaler(feature_range=(0, 1))
            self.scaler_y = MinMaxScaler(feature_range=(0, 1))
            
            self.scaler_X.fit(concatenated_features)
            self.scaler_y.fit(concatenated_cgm)
            print("Scalers fitted on the aggregated training data.")
        else:
            print("Scalers already fitted (possibly from a loaded model).")

        # --- Step 2: Create sequences using the fitted scalers ---
        for df in data:
            X_seq, y_seq = self._prepare_data_for_lstm(df, fit_scalers=False) # fit_scalers is now always False here
            if X_seq is not None and X_seq.size > 0 and y_seq is not None and y_seq.size > 0:
                all_X_train_list.append(X_seq)
                all_y_train_list.append(y_seq)
        
        if not all_X_train_list:
            print("No training data sequences generated after processing all data. Skipping training.")
            return

        X_train_all = np.concatenate(all_X_train_list, axis=0)
        y_train_all = np.concatenate(all_y_train_list, axis=0)

        print(f"Total training sequences: {X_train_all.shape[0]}")

        # Convert to PyTorch Tensors
        X_train_tensor = torch.from_numpy(X_train_all).float().to(self.device)
        y_train_tensor = torch.from_numpy(y_train_all).float().to(self.device)

        # Create DataLoader for training
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Prepare validation DataLoader if validation data is provided
        val_loader = None
        if validation_data:
            all_X_val_list = []
            all_y_val_list = []
            for df in validation_data:
                X_val_seq, y_val_seq = self._prepare_data_for_lstm(df, fit_scalers=False)
                if X_val_seq is not None and X_val_seq.size > 0 and y_val_seq is not None and y_val_seq.size > 0:
                    all_X_val_list.append(X_val_seq)
                    all_y_val_list.append(y_val_seq)
            if all_X_val_list:
                X_val_all = np.concatenate(all_X_val_list, axis=0)
                y_val_all = np.concatenate(all_y_val_list, axis=0)
                X_val_tensor = torch.from_numpy(X_val_all).float().to(self.device)
                y_val_tensor = torch.from_numpy(y_val_all).float().to(self.device)
                val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Optimizer and Loss Function
        if self.model is None: # Should have been built by now
            self._build_model()
            if self.model is None: # Still None after build attempt
                print("Error: Model could not be built. Aborting training.")
                return

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        self.model.train() # Set model to training mode

        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()

                # Forward pass
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)

                # Backward pass and optimize
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_epoch_loss = epoch_loss / len(train_loader)

            log_message = f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_epoch_loss:.4f}"
            avg_val_loss = None
            if val_loader is not None:
                self.model.eval()
                val_loss_total = 0.0
                with torch.no_grad():
                    for batch_X_val, batch_y_val in val_loader:
                        val_outputs = self.model(batch_X_val)
                        v_loss = criterion(val_outputs, batch_y_val)
                        val_loss_total += v_loss.item()
                avg_val_loss = val_loss_total / len(val_loader)
                log_message += f", Val Loss: {avg_val_loss:.4f}"
                self.model.train()

            self.training_history.append({"epoch": epoch + 1, "train_loss": avg_epoch_loss, "val_loss": avg_val_loss})

            print(log_message)

        self.is_trained = True
        print("LSTM training finished.")


    def predict(self, current_input: pd.DataFrame, **kwargs) -> Dict[str, List[float]]:
        """
        Makes glucose predictions based on the current history.

        Args:
            current_input (pd.DataFrame): DataFrame containing the recent history
                of features (CGM, insulin, carbs) up to the current point.
                Must contain at least `input_seq_len` rows.
            **kwargs: Additional prediction parameters (not used currently).

        Returns:
            Dict[str, List[float]]: Predictions, e.g., {"mean": [pred1, pred2,...]}
        """
        if not self.is_trained or not self.model or not self.scaler_X or not self.scaler_y:
            print("Warning: LSTMPredictor not trained, model not built, or scalers not fitted. Returning naive persistence.")
            last_cgm = current_input['cgm_mg_dl'].iloc[-1] if not current_input.empty else 100.0
            return {"mean": [last_cgm] * self.output_seq_len}

        if len(current_input) < self.input_seq_len:
            print(f"Warning: Not enough history ({len(current_input)} points) for input_seq_len {self.input_seq_len}. Returning naive persistence.")
            last_cgm = current_input['cgm_mg_dl'].iloc[-1] if not current_input.empty else 100.0
            return {"mean": [last_cgm] * self.output_seq_len}
            
        # Prepare the most recent sequence
        features_to_use = ['cgm_mg_dl', 'bolus_U', 'carbs_g'] # Should match n_features
        
        # Get the last input_seq_len rows for prediction
        history_for_prediction = current_input[features_to_use].tail(self.input_seq_len).values.astype(np.float32)
        
        # Scale the features
        scaled_history = self.scaler_X.transform(history_for_prediction)
        
        # Convert to PyTorch Tensor and reshape for LSTM: (batch_size=1, seq_len, n_features)
        input_tensor = torch.from_numpy(scaled_history).float().unsqueeze(0).to(self.device)
        
        self.model.eval() # Set model to evaluation mode
        with torch.no_grad():
            raw_predictions = self.model(input_tensor) # Shape: (1, output_seq_len)
        
        # Inverse transform the predictions to original CGM scale
        # Predictions are for CGM, so use scaler_y
        # Reshape raw_predictions to (output_seq_len, 1) for inverse_transform if it expects 2D array
        predicted_scaled_cgm = raw_predictions.cpu().numpy().reshape(-1, 1)
        
        # If output_seq_len is 1, scaler_y might expect (n_samples, 1).
        # If output_seq_len > 1, and fc layer outputs directly output_seq_len features,
        # then predicted_scaled_cgm is already (1, output_seq_len).
        # We need to ensure the shape matches what scaler_y.inverse_transform expects for multiple target values.
        # The LSTMNetwork's fc layer outputs `output_size` which is `self.output_seq_len`.
        # So `raw_predictions` is (1, self.output_seq_len).
        # `scaler_y` was fit on CGM data of shape (n_samples, 1).
        # To inverse transform each predicted CGM value, we need to treat them as separate samples for the scaler.
        
        # If output_seq_len > 1, we need to reshape for scaler_y.
        # The current LSTMNetwork outputs a flat vector of size output_seq_len.
        # We assume each element of this vector is a predicted CGM value.
        
        # Let's reshape the output of the model if it's not already (N,1) for scaler.
        # The current LSTMNetwork's fc layer outputs (batch_size, output_seq_len).
        # For a single prediction, batch_size is 1. So raw_predictions is (1, output_seq_len).
        # We need to make it (output_seq_len, 1) for scaler_y.inverse_transform.
        
        predictions_for_inverse_transform = raw_predictions.cpu().numpy().T # Transpose to (output_seq_len, 1)
        
        # If scaler_y was fit on single CGM values, it expects (n_samples, 1)
        # Our LSTM output is (1, output_seq_len). We need to treat each of the output_seq_len values
        # as an independent value to be inverse-scaled.
        # A common way is to have the LSTM output (batch, seq_len, 1) for CGM prediction
        # or ensure the final linear layer outputs (batch, output_seq_len) and then inverse transform.
        # Given our LSTMNetwork.fc outputs (hidden_units, output_size=output_seq_len),
        # and then out = self.fc(out[:, -1, :]) makes it (batch_size, output_seq_len).
        # So, raw_predictions is (1, output_seq_len).
        
        # Reshape for scaler:
        # Each of the `output_seq_len` values is a scaled CGM prediction.
        # We need to apply inverse_transform to each.
        # `scaler_y.inverse_transform` expects an array of shape (n_samples, n_features=1)
        
        # Correct reshaping for inverse transform:
        scaled_predictions = raw_predictions.cpu().numpy().reshape(self.output_seq_len, 1)
        actual_predictions = self.scaler_y.inverse_transform(scaled_predictions).flatten().tolist()
        
        return {"mean": actual_predictions}

    def save(self, path: str): # Changed directory_path to path
        """Saves the trained LSTM model, its configuration, and scalers to the given directory path."""
        if not self.is_trained or not self.model:
            print("Model not trained. Nothing to save.")
            return

        if not os.path.exists(path): # path is treated as a directory
            os.makedirs(path)
            print(f"Created directory for saving model: {path}")

        model_path = os.path.join(path, "lstm_model.pth")
        config_path = os.path.join(path, "lstm_config.json")
        scaler_x_path = os.path.join(path, "scaler_x.joblib")
        scaler_y_path = os.path.join(path, "scaler_y.joblib")

        # Save model state dictionary
        torch.save(self.model.state_dict(), model_path)
        
        # Save configuration
        config = {
            "input_seq_len": self.input_seq_len,
            "output_seq_len": self.output_seq_len,
            "n_features": self.n_features,
            "hidden_units": self.hidden_units,
            "num_layers": self.num_layers,
            "dropout_rate": self.dropout_rate,
            "learning_rate": self.learning_rate # Added learning_rate to config
        }
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
            
        # Save scalers
        if self.scaler_X:
            joblib.dump(self.scaler_X, scaler_x_path)
        if self.scaler_y:
            joblib.dump(self.scaler_y, scaler_y_path)
            
        print(f"LSTMPredictor saved to directory: {path}")

    def load(self, path: str): # Changed directory_path to path
        """Loads a pre-trained LSTM model, its configuration, and scalers from the given directory path."""
        model_path = os.path.join(path, "lstm_model.pth")
        config_path = os.path.join(path, "lstm_config.json")
        scaler_x_path = os.path.join(path, "scaler_x.joblib")
        scaler_y_path = os.path.join(path, "scaler_y.joblib")

        if not all(os.path.exists(p) for p in [model_path, config_path, scaler_x_path, scaler_y_path]):
            print(f"Error: Not all required files found in directory {path}. Cannot load model.")
            return

        # Load configuration
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Re-initialize with loaded parameters (important if loading into a fresh instance)
        self.input_seq_len = config["input_seq_len"]
        self.output_seq_len = config["output_seq_len"]
        self.n_features = config["n_features"]
        self.hidden_units = config["hidden_units"]
        self.num_layers = config["num_layers"]
        self.dropout_rate = config["dropout_rate"]
        self.learning_rate = config.get("learning_rate", 0.001) # Handle older configs

        self._build_model() # Build model structure with loaded params
        
        if self.model:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device) # Ensure model is on the correct device
            self.is_trained = True
            print(f"LSTM model state loaded from {model_path}")
        else:
            print("Error: Model could not be built during load. Cannot load state dict.")
            return

        # Load scalers
        self.scaler_X = joblib.load(scaler_x_path)
        self.scaler_y = joblib.load(scaler_y_path)
        print(f"Scalers loaded from {path}")
        print(f"LSTMPredictor loaded successfully from directory: {path}")


if __name__ == '__main__':
    print("LSTMPredictor Example Usage:")
    sample_history_df = None # Initialize to handle cases where data loading might fail
    
    # --- Configuration ---
    INPUT_SEQ_LEN = 24  # e.g., 24 * 5 min = 2 hours of history
    OUTPUT_SEQ_LEN = 12 # e.g., 12 * 5 min = 1 hour prediction horizon
    N_FEATURES = 3      # CGM, Bolus, Carbs
    EPOCHS = 5 # Small number for example
    BATCH_SIZE = 64
    MODEL_SAVE_DIR = "DiaGuardianAI/models/lstm_predictor_example_run"

    # --- 1. Initialize the predictor ---
    predictor = LSTMPredictor(
        input_seq_len=INPUT_SEQ_LEN,
        output_seq_len=OUTPUT_SEQ_LEN,
        n_features=N_FEATURES,
        hidden_units=64,
        num_layers=2,
        dropout_rate=0.1,
        learning_rate=0.001
    )

    # --- 2. Load training data ---
    data_load_dir = "DiaGuardianAI/datasets/generated_data"
    profile_to_load = "adult_moderate_control"
    
    loaded_training_dfs = []
    try:
        profile_files = [f for f in os.listdir(data_load_dir) if f.startswith(profile_to_load) and f.endswith(".csv")]
        if not profile_files:
            print(f"No CSV data found for profile '{profile_to_load}' in '{data_load_dir}'. Skipping training.")
        else:
            latest_profile_file = max(profile_files, key=lambda f: os.path.getmtime(os.path.join(data_load_dir, f)))
            data_path = os.path.join(data_load_dir, latest_profile_file)
            print(f"Loading data from: {data_path}")
            df = pd.read_csv(data_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            loaded_training_dfs.append(df)
            
    except Exception as e:
        print(f"Error loading data: {e}. Skipping training.")

    if not loaded_training_dfs:
        print("No training data loaded. Exiting example.")
    else:
        # --- 3. Train the model ---
        print(f"\nStarting training with {len(loaded_training_dfs)} DataFrame(s)...")
        predictor.train(data=loaded_training_dfs, epochs=EPOCHS, batch_size=BATCH_SIZE)

        # --- 4. Simulate making a prediction ---
        if predictor.is_trained:
            print("\nSimulating prediction...")
            sample_history_df = loaded_training_dfs[0]
            if len(sample_history_df) >= predictor.input_seq_len:
                pred_start_idx = len(sample_history_df) // 2
                pred_end_idx = pred_start_idx + predictor.input_seq_len
                if pred_end_idx > len(sample_history_df):
                    pred_end_idx = len(sample_history_df)
                    pred_start_idx = pred_end_idx - predictor.input_seq_len

                if pred_start_idx >= 0 :
                    current_history_for_pred = sample_history_df.iloc[pred_start_idx:pred_end_idx].copy()
                    
                    print(f"  Using history of {len(current_history_for_pred)} points for prediction.")
                    predictions = predictor.predict(current_history_for_pred)
                    print(f"  Predictions based on history: {[round(p,2) for p in predictions.get('mean', [])]}")

                    last_actual_cgm = current_history_for_pred['cgm_mg_dl'].iloc[-1]
                    print(f"  Last actual CGM in history: {last_actual_cgm:.2f}")
                    
                    actual_future_start_idx = pred_end_idx
                    actual_future_end_idx = pred_end_idx + predictor.output_seq_len
                    if actual_future_end_idx <= len(sample_history_df):
                        actual_future_cgm = sample_history_df['cgm_mg_dl'].iloc[actual_future_start_idx:actual_future_end_idx].values.tolist()
                        print(f"  Actual future CGM values: {[round(x,2) for x in actual_future_cgm]}")
                    else:
                        print("  Not enough subsequent data to show all actual future CGM values for comparison.")
                else:
                    print(f"  Not enough data in sample_history_df to form a prediction input sequence.")
            else:
                print(f"  Not enough data in the loaded DataFrame ({len(sample_history_df)} points) to form an input sequence of length {predictor.input_seq_len}.")

        # --- 5. Save/Load ---
        if predictor.is_trained:
            print(f"\nSaving model to: {MODEL_SAVE_DIR}")
            predictor.save(MODEL_SAVE_DIR)
            
            print(f"\nCreating new predictor instance and loading from: {MODEL_SAVE_DIR}")
            loaded_predictor = LSTMPredictor(
                input_seq_len=1, output_seq_len=1, n_features=1 # Dummy init, load will overwrite
            )
            loaded_predictor.load(MODEL_SAVE_DIR)
            print(f"  Loaded predictor config: input_seq_len={loaded_predictor.input_seq_len}, output_seq_len={loaded_predictor.output_seq_len}")

            if sample_history_df is not None and isinstance(sample_history_df, pd.DataFrame):
                if len(sample_history_df) >= loaded_predictor.input_seq_len:
                    pred_start_idx = len(sample_history_df) // 2
                    pred_end_idx = pred_start_idx + loaded_predictor.input_seq_len
                    if pred_end_idx > len(sample_history_df):
                        pred_end_idx = len(sample_history_df)
                        pred_start_idx = pred_end_idx - loaded_predictor.input_seq_len
                    
                    if pred_start_idx >=0:
                        current_history_for_loaded_pred = sample_history_df.iloc[pred_start_idx:pred_end_idx].copy()
                        loaded_predictions = loaded_predictor.predict(current_history_for_loaded_pred)
                        print(f"  Predictions from loaded model: {[round(p,2) for p in loaded_predictions.get('mean', [])]}")
                    else:
                        print("  Not enough data for loaded model prediction after slicing (within valid DataFrame).")
                else:
                    print(f"  Not enough data in sample_history_df ({len(sample_history_df)}) for loaded model prediction (need {loaded_predictor.input_seq_len}).")
            else:
                print("  sample_history_df is not available or not a DataFrame. Skipping loaded model prediction test.")
        else:
            print("\nSkipping save/load example as model was not trained.")