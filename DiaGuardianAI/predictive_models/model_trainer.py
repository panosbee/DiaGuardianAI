# DiaGuardianAI Model Trainer
# Logic for training and fine-tuning predictive models.

import sys
import os
from typing import Dict, Any, Tuple, Optional, List, cast # Ensure cast is here
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.model_selection import KFold # For cross-validation
import optuna # For hyperparameter tuning

# Ensure the DiaGuardianAI package is discoverable when run as a script
if __package__ is None or __package__ == '':
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')) # Corrected path for this file location
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from DiaGuardianAI.core.base_classes import BasePredictiveModel
from DiaGuardianAI.data_generation.data_formatter import DataFormatter
# Removed duplicated imports that are now at the top


class ModelTrainer:
    """Handles training, cross-validation, and hyperparameter tuning for PyTorch models."""
    def __init__(self, model: BasePredictiveModel, # Expects model to have a .model attribute which is nn.Module
                 data_formatter: Optional[DataFormatter] = None,
                 training_params: Optional[Dict[str, Any]] = None):
        """Initializes the ModelTrainer.
        Args:
            model (BasePredictiveModel): The predictive model instance.
                                       Must have a `model` attribute that is an `nn.Module`
                                       and a `device` attribute.
            data_formatter (Optional[DataFormatter]): DataFormatter instance.
            training_params (Optional[Dict[str, Any]]): Training parameters like
                `{"epochs": 100, "batch_size": 32, "learning_rate": 0.001}`.
        """
        # Runtime checks for essential attributes expected from concrete PyTorch-based models
        if not hasattr(model, 'model') or not isinstance(getattr(model, 'model'), nn.Module):
            raise ValueError("The 'model' argument (BasePredictiveModel instance) must have a 'model' "
                             "attribute that is an instance of torch.nn.Module.")
        if not hasattr(model, 'device') or not isinstance(getattr(model, 'device'), torch.device):
            raise ValueError("The 'model' argument (BasePredictiveModel instance) must have a 'device' "
                             "attribute that is an instance of torch.device.")

        self.model_wrapper: BasePredictiveModel = model
        self.pytorch_model: nn.Module = getattr(model, 'model')
        self.device: torch.device = getattr(model, 'device')
        self.data_formatter: Optional[DataFormatter] = data_formatter
        self.training_params: Dict[str, Any] = training_params if training_params else {}
        self.training_history: List[Dict[str, Any]] = []

        print(
            f"ModelTrainer initialized for model: {self.model_wrapper.__class__.__name__} "
            f"on device: {self.device} with params: {self.training_params}"
        )

    def train_model(self, X_train: Any, y_train: Any,
                    X_val: Optional[Any] = None, y_val: Optional[Any] = None) -> Dict[str, Any]:
        """Trains the PyTorch model.
        Args:
            X_train, y_train: Training features and targets (NumPy arrays or PyTorch tensors).
            X_val, y_val: Validation features and targets (Optional).
        Returns:
            Dict[str, Any]: Training status and history.
        """
        epochs = self.training_params.get("epochs", 10)
        batch_size = self.training_params.get("batch_size", 32)
        learning_rate = self.training_params.get("learning_rate", 0.001)

        if not isinstance(X_train, torch.Tensor): X_train = torch.tensor(X_train, dtype=torch.float32)
        if not isinstance(y_train, torch.Tensor): y_train = torch.tensor(y_train, dtype=torch.float32)
        
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_loader = None
        if X_val is not None and y_val is not None:
            if not isinstance(X_val, torch.Tensor): X_val = torch.tensor(X_val, dtype=torch.float32)
            if not isinstance(y_val, torch.Tensor): y_val = torch.tensor(y_val, dtype=torch.float32)
            val_dataset = TensorDataset(X_val, y_val)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.pytorch_model.parameters(), lr=learning_rate)

        print(f"Starting training for {self.model_wrapper.__class__.__name__} for {epochs} epochs...")
        self.training_history = []

        for epoch in range(epochs):
            self.pytorch_model.train() # Set model to training mode
            epoch_train_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                # The model's forward pass should handle the input shape
                # For LSTM/Transformer, input is (batch, seq_len, features)
                # Output is (batch, output_horizon_steps)
                predictions = self.pytorch_model(batch_X)
                loss = criterion(predictions, batch_y)
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item() * batch_X.size(0)
            
            epoch_train_loss /= len(train_loader.dataset) # type: ignore[arg-type]
            
            epoch_log = {"epoch": epoch + 1, "train_loss": epoch_train_loss}

            if val_loader:
                self.pytorch_model.eval() # Set model to evaluation mode
                epoch_val_loss = 0.0
                with torch.no_grad():
                    for batch_X_val, batch_y_val in val_loader:
                        batch_X_val, batch_y_val = batch_X_val.to(self.device), batch_y_val.to(self.device)
                        val_predictions = self.pytorch_model(batch_X_val)
                        val_loss = criterion(val_predictions, batch_y_val)
                        epoch_val_loss += val_loss.item() * batch_X_val.size(0)
                epoch_val_loss /= len(val_loader.dataset) # type: ignore[arg-type]
                epoch_log["val_loss"] = epoch_val_loss
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {epoch_train_loss:.4f} - Val Loss: {epoch_val_loss:.4f}")
            else:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {epoch_train_loss:.4f}")
            
            self.training_history.append(epoch_log)

        return {"status": "Training complete", "history": self.training_history}

    def cross_validate(self, X: Any, y: Any, n_splits: int = 5) -> Dict[str, Any]:
        """Performs k-fold cross-validation on the model. (Placeholder)

        This method would split the data into `n_splits` folds,
        training the model on `n_splits - 1` folds and validating on
        the remaining fold, iterating through all folds.

        Args:
            X (np.ndarray): The full dataset features.
            y (np.ndarray): The full dataset targets.
            n_splits (int): The number of cross-validation folds. Defaults to 5.

        Returns:
            Dict[str, Any]: A dictionary containing cross-validation
                results, such as a list of scores for each fold.
                Example: `{"cv_scores_mse": [10.5, 11.2, ...], "mean_cv_mse": 10.8}`.
        """
        print(f"Starting {n_splits}-fold cross-validation...")

        if not isinstance(X, np.ndarray): X = np.array(X)
        if not isinstance(y, np.ndarray): y = np.array(y)

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.training_params.get("random_state", None))
        fold_scores_mse = []

        # Store original model's init parameters to re-instantiate for each fold
        # This assumes the model wrapper (e.g., LSTMPredictor) stores its init args.
        # For DummyTrainableModel, we'll need to ensure its args are accessible or passed.
        model_class = self.model_wrapper.__class__
        
        # Try to get init args from the model_wrapper instance.
        # This is a bit heuristic; a more robust way would be for BasePredictiveModel
        # to enforce a method that returns its init args, or for wrappers to store them.
        init_args = {}
        # Suppress Pylance errors for these getattr calls as hasattr is used for runtime safety.
        if hasattr(self.model_wrapper, 'input_dim'): init_args['input_dim'] = self.model_wrapper.input_dim # type: ignore
        if hasattr(self.model_wrapper, 'hidden_dim'): init_args['hidden_dim'] = self.model_wrapper.hidden_dim # type: ignore # For LSTM
        if hasattr(self.model_wrapper, 'num_layers'): init_args['num_layers'] = self.model_wrapper.num_layers # type: ignore # For LSTM
        if hasattr(self.model_wrapper, 'output_horizon_steps'): init_args['output_horizon_steps'] = self.model_wrapper.output_horizon_steps # type: ignore
        if hasattr(self.model_wrapper, 'dropout_prob'): init_args['dropout_prob'] = self.model_wrapper.dropout_prob # type: ignore
        if hasattr(self.model_wrapper, 'mc_dropout_samples'): init_args['mc_dropout_samples'] = self.model_wrapper.mc_dropout_samples # type: ignore
        # For Transformer
        if hasattr(self.model_wrapper, 'model_dim'): init_args['model_dim'] = self.model_wrapper.model_dim # type: ignore
        if hasattr(self.model_wrapper, 'num_heads'): init_args['num_heads'] = self.model_wrapper.num_heads # type: ignore
        if hasattr(self.model_wrapper, 'num_encoder_layers'): init_args['num_encoder_layers'] = self.model_wrapper.num_encoder_layers # type: ignore
        if hasattr(self.model_wrapper, 'dim_feedforward'): init_args['dim_feedforward'] = self.model_wrapper.dim_feedforward # type: ignore
        if hasattr(self.model_wrapper, 'max_seq_len'): init_args['max_seq_len'] = self.model_wrapper.max_seq_len # type: ignore
        
        # For DummyTrainableModel specific args
        if isinstance(self.model_wrapper, DummyTrainableModel): # Check if it's the dummy
            # These attributes are specific to DummyTrainableModel and defined in its __init__
            init_args['input_features'] = self.model_wrapper.input_features # type: ignore
            init_args['output_horizon_steps'] = self.model_wrapper.output_horizon_steps # type: ignore
            # id_val is also part of DummyTrainableModel __init__, but defaults to 0.
            # If it were important to preserve, it would need to be stored and retrieved too.
            # For now, relying on its default for re-instantiation.

        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            print(f"  Fold {fold+1}/{n_splits}")
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]

            # Create a new model instance for this fold
            try:
                fold_model_wrapper = model_class(**init_args)
            except TypeError as e:
                 print(f"Error re-instantiating model {model_class.__name__} for CV fold: {e}")
                 print(f"Attempted with init_args: {init_args}")
                 # Fallback for DummyTrainableModel if specific args are missing from init_args dict
                 if model_class == DummyTrainableModel:
                     fold_model_wrapper = DummyTrainableModel(
                         input_features=init_args.get('input_features', 10),
                         output_horizon_steps=init_args.get('output_horizon_steps', 1)
                     )
                 else:
                     raise # Re-raise if not the dummy model or fallback failed

            fold_trainer = ModelTrainer(model=fold_model_wrapper, training_params=self.training_params)
            fold_trainer.train_model(X_train_fold, y_train_fold) # No validation set passed to inner train_model here

            # Evaluate on the validation fold
            # Assuming predict method handles batching or we predict one by one
            y_pred_fold_list = []
            if X_val_fold.shape[0] > 0:
                for i in range(X_val_fold.shape[0]):
                    sample_to_predict = X_val_fold[i] # This is (features_per_sample) or (seq_len, features_per_step)
                    
                    # The model's predict method should handle the exact input shape it expects.
                    # For a single sample prediction, we typically add a batch dimension.
                    if sample_to_predict.ndim == 1: # e.g., (features_per_sample) for MLP
                        sample_to_predict_batched = np.expand_dims(sample_to_predict, axis=0) # (1, features_per_sample)
                    elif sample_to_predict.ndim == 2: # e.g., (seq_len, features_per_step) for RNN/Transformer
                        sample_to_predict_batched = np.expand_dims(sample_to_predict, axis=0) # (1, seq_len, features_per_step)
                    else: # Should not happen if X_val_fold is (num_samples, ...)
                        sample_to_predict_batched = sample_to_predict # Or raise error

                    prediction_dict = fold_model_wrapper.predict(sample_to_predict_batched)
                    y_pred_fold_list.append(prediction_dict["mean"])
            
            if y_pred_fold_list:
                y_pred_fold_np = np.array(y_pred_fold_list)
                # Ensure y_val_fold and y_pred_fold_np have compatible shapes for MSE
                # y_val_fold is (n_val_samples, output_horizon_steps)
                # y_pred_fold_np should also be (n_val_samples, output_horizon_steps)
                if y_pred_fold_np.shape == y_val_fold.shape:
                    mse = np.mean((y_pred_fold_np - y_val_fold)**2)
                    fold_scores_mse.append(mse)
                    print(f"    Fold {fold+1} MSE: {mse:.4f}")
                else:
                    print(f"    Fold {fold+1} MSE: Shape mismatch. y_pred: {y_pred_fold_np.shape}, y_true: {y_val_fold.shape}")
                    fold_scores_mse.append(np.nan) # Or handle error
            else:
                print(f"    Fold {fold+1} MSE: No predictions made (empty validation set or issue).")
                fold_scores_mse.append(np.nan)


        mean_mse = np.nanmean(fold_scores_mse) if fold_scores_mse else np.nan
        print(f"Cross-validation finished. Mean MSE: {mean_mse:.4f}")
        return {"cv_scores_mse": fold_scores_mse, "mean_cv_mse": mean_mse}

    def hyperparameter_tune(self, X_train: np.ndarray, y_train: np.ndarray,
                              search_space: Dict[str, Any],
                              n_trials: int = 10,
                              cv_folds: int = 3) -> Dict[str, Any]:
        """Performs hyperparameter tuning. (Placeholder)

        This method would use a library like Optuna or Ray Tune to
        search for the best hyperparameters for the model based on
        performance on a validation set or cross-validation.

        Args:
            X_train (np.ndarray): Training features.
            y_train (np.ndarray): Training targets.
            search_space (Dict[str, Any]): A dictionary defining the
                hyperparameter search space. The format depends on the
                tuning library (e.g., Optuna distributions).
            n_trials (int): The number of hyperparameter combinations to
                try. Defaults to 10.
            cv_folds (int): The number of cross-validation folds to use
                for evaluating each hyperparameter set. Defaults to 3.

        Returns:
            Dict[str, Any]: A dictionary containing the best found
                hyperparameters and the corresponding performance score.
                Example: `{"best_params": {"lr": 0.001, ...},
                "best_score": 0.95}`.
        """
        print(f"Starting hyperparameter tuning with {n_trials} trials using Optuna...")

        # --- Objective function for Optuna ---
        def objective(trial: optuna.Trial) -> float:
            # --- Suggest Hyperparameters ---
            # These hyperparameters can be for the model architecture or training process.
            # The search_space dict should guide what to suggest.
            # Example: search_space = {
            #     "model_params": {
            #         "hidden_dim": {"type": "int", "low": 32, "high": 128, "step": 16},
            #         "num_layers": {"type": "int", "low": 1, "high": 3}
            #     },
            #     "training_params": {
            #         "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            #         "batch_size": {"type": "categorical", "choices": [16, 32, 64]}
            #     }
            # }
            
            trial_model_init_args = {} # For model re-instantiation
            trial_training_params = self.training_params.copy() # Start with base training params

            # Populate model init args from search_space["model_params"]
            if "model_params" in search_space:
                for param_name, config in search_space["model_params"].items():
                    if config["type"] == "int":
                        trial_model_init_args[param_name] = trial.suggest_int(
                            param_name, config["low"], config["high"], step=config.get("step", 1), log=config.get("log", False)
                        )
                    elif config["type"] == "float":
                        trial_model_init_args[param_name] = trial.suggest_float(
                            param_name, config["low"], config["high"], step=config.get("step"), log=config.get("log", False)
                        )
                    elif config["type"] == "categorical":
                        trial_model_init_args[param_name] = trial.suggest_categorical(param_name, config["choices"])
            
            # Populate training params from search_space["training_params"]
            if "training_params" in search_space:
                 for param_name, config in search_space["training_params"].items():
                    if config["type"] == "int":
                        trial_training_params[param_name] = trial.suggest_int(
                            f"tp_{param_name}", config["low"], config["high"], step=config.get("step", 1), log=config.get("log", False)
                        )
                    elif config["type"] == "float":
                        trial_training_params[param_name] = trial.suggest_float(
                            f"tp_{param_name}", config["low"], config["high"], step=config.get("step"), log=config.get("log", False)
                        )
                    elif config["type"] == "categorical":
                        trial_training_params[param_name] = trial.suggest_categorical(f"tp_{param_name}", config["choices"])


            # --- Instantiate Model with Trial Hyperparameters ---
            # This assumes the model wrapper class can be re-instantiated with these args.
            # We need to gather the *original* non-tuned init args of the model_wrapper first.
            base_model_init_args = {}
            # Heuristically gather known args (similar to cross_validate)
            # A more robust solution would be for BasePredictiveModel to have a get_init_args method.
            potential_args = [
                'input_dim', 'hidden_dim', 'num_layers', 'output_horizon_steps',
                'dropout_prob', 'mc_dropout_samples', 'model_dim', 'num_heads',
                'num_encoder_layers', 'dim_feedforward', 'max_seq_len',
                'input_features' # For DummyTrainableModel
            ]
            for arg_name in potential_args:
                if hasattr(self.model_wrapper, arg_name):
                    base_model_init_args[arg_name] = getattr(self.model_wrapper, arg_name)
            
            # Override with tuned model parameters
            final_model_init_args = {**base_model_init_args, **trial_model_init_args}
            
            # Ensure essential args like output_horizon_steps and input_features are present if not tuned
            if 'output_horizon_steps' not in final_model_init_args and hasattr(self.model_wrapper, 'output_horizon_steps'):
                final_model_init_args['output_horizon_steps'] = self.model_wrapper.output_horizon_steps # type: ignore
            if 'input_features' not in final_model_init_args and hasattr(self.model_wrapper, 'input_features'):
                 final_model_init_args['input_features'] = self.model_wrapper.input_features # type: ignore


            try:
                trial_model_wrapper = self.model_wrapper.__class__(**final_model_init_args)
            except TypeError as e:
                print(f"Optuna objective: Error re-instantiating model {self.model_wrapper.__class__.__name__}: {e}")
                print(f"Attempted with final_model_init_args: {final_model_init_args}")
                # Fallback for DummyTrainableModel if specific args are missing
                if self.model_wrapper.__class__ == DummyTrainableModel:
                    trial_model_wrapper = DummyTrainableModel(
                        input_features=final_model_init_args.get('input_features', 10), # Default if not found
                        output_horizon_steps=final_model_init_args.get('output_horizon_steps', 1) # Default
                    )
                else:
                    raise # Re-raise for other models

            # --- Train and Evaluate ---
            trial_trainer = ModelTrainer(model=trial_model_wrapper, training_params=trial_training_params)
            
            # Use cross-validation for robust evaluation
            cv_results = trial_trainer.cross_validate(X_train, y_train, n_splits=cv_folds)
            mean_cv_mse = cv_results.get("mean_cv_mse", float('inf')) # Default to infinity if not found

            if np.isnan(mean_cv_mse): # Handle NaN cases, Optuna doesn't like them
                return float('inf')

            return mean_cv_mse
        # --- End of Objective Function ---

        study = optuna.create_study(direction="minimize") # We want to minimize MSE
        study.optimize(objective, n_trials=n_trials,
                       # Catch exceptions during trials to prevent Optuna from crashing
                       # and allow it to continue with other trials.
                       catch=(TypeError, ValueError, RuntimeError))

        print(f"Hyperparameter tuning complete.")
        print(f"Best hyperparameters: {study.best_params}")
        print(f"Best mean CV MSE: {study.best_value}")
        
        return {"best_params": study.best_params, "best_score_mse": study.best_value}

        return {"best_params": study.best_params, "best_score_mse": study.best_value}

    def continuous_fine_tuning(self, new_X: Any, new_y: Any,
                               fine_tuning_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Fine-tunes the existing model with new incoming data.

        This method allows for updating the model with new data.
        It uses the existing PyTorch model and its current state.

        Args:
            new_X (Any): New features for fine-tuning (NumPy array or PyTorch tensor).
            new_y (Any): New targets for fine-tuning (NumPy array or PyTorch tensor).
            fine_tuning_params (Optional[Dict[str, Any]]): Fine-tuning parameters like
                `{"epochs": 10, "batch_size": 16, "learning_rate": 0.0001}`.
                If None, uses a subset of existing training_params or defaults.
        Returns:
            Dict[str, Any]: Fine-tuning status and history.
        """
        if not self.training_history: # Ensure model has been trained at least once
            print("Warning: Model has not been trained yet. Consider initial training first.")
            # Or, could call self.train_model here if that's desired behavior.

        effective_params = self.training_params.copy() # Start with original params
        if fine_tuning_params:
            effective_params.update(fine_tuning_params)
        
        # Typically, fine-tuning uses a smaller learning rate and fewer epochs.
        epochs = effective_params.get("epochs", 5) # Default to fewer epochs for fine-tuning
        batch_size = effective_params.get("batch_size", self.training_params.get("batch_size", 32))
        learning_rate = effective_params.get("learning_rate", self.training_params.get("learning_rate", 0.001) / 10) # Default to smaller LR

        print(
            f"Starting continuous fine-tuning with new data "
            f"(X shape: {getattr(new_X, 'shape', 'N/A')}, Y shape: {getattr(new_y, 'shape', 'N/A')}) "
            f"for {epochs} epochs, LR: {learning_rate}, Batch: {batch_size}."
        )

        if not isinstance(new_X, torch.Tensor): new_X = torch.tensor(new_X, dtype=torch.float32)
        if not isinstance(new_y, torch.Tensor): new_y = torch.tensor(new_y, dtype=torch.float32)
        
        fine_tune_dataset = TensorDataset(new_X, new_y)
        fine_tune_loader = DataLoader(fine_tune_dataset, batch_size=batch_size, shuffle=True)

        criterion = nn.MSELoss()
        # Use the existing optimizer or re-initialize if needed (e.g., to change LR for specific param groups)
        # For simplicity, we'll re-initialize with the potentially new fine-tuning learning rate.
        optimizer = optim.Adam(self.pytorch_model.parameters(), lr=learning_rate)

        fine_tuning_history = []

        for epoch in range(epochs):
            self.pytorch_model.train() # Set model to training mode
            epoch_loss = 0.0
            for batch_X, batch_y in fine_tune_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                predictions = self.pytorch_model(batch_X)
                loss = criterion(predictions, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch_X.size(0)
            
            epoch_loss /= len(fine_tune_loader.dataset) # type: ignore[arg-type]
            fine_tuning_history.append({"epoch": epoch + 1, "fine_tune_loss": epoch_loss})
            print(f"Fine-tuning Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")
            
        # Append to the main training history
        self.training_history.extend(fine_tuning_history)
        print("Continuous fine-tuning complete.")
        return {"status": "Fine-tuning complete", "history": fine_tuning_history}


if __name__ == '__main__':
    # Dummy model for testing trainer
    class DummyTrainableModel(BasePredictiveModel):
        def __init__(self, id_val: int = 0, output_horizon_steps: int = 1, input_features: int = 10): # Added input_features
            super().__init__() # Call BasePredictiveModel's __init__
            self.trained = False
            self.id_val = id_val
            # Define model and device attributes directly here
            self.input_features = input_features # Store for CV re-instantiation
            self.output_horizon_steps = output_horizon_steps # Store for CV and predict
            self.model: nn.Module = nn.Linear(self.input_features, self.output_horizon_steps) # Dummy nn.Module
            self.device: torch.device = torch.device("cpu")
            # self.output_horizon_steps_dummy = output_horizon_steps # For dummy predict # Renamed
            self.model.to(self.device) # Move model to device
            print(f"DummyTrainableModel {id_val} initialized with input_features={self.input_features}, output_horizon_steps={self.output_horizon_steps} on {self.device}.")

        def train(self, X_train: Any, y_train: Any): # X_train, y_train are expected to be tensors by ModelTrainer
            print(
                f"DummyTrainableModel {self.id_val} train called with "
                f"X shape {X_train.shape}, y shape {y_train.shape}."
            )
            # Minimal training simulation
            optimizer = optim.SGD(self.model.parameters(), lr=0.01)
            criterion = nn.MSELoss()
            self.model.train()
            for _ in range(2): # Simulate a few steps
                optimizer.zero_grad()
                # Assuming X_train is (batch, features) for Linear layer
                # If X_train is (batch, seq, features), need to adjust or use a sequential dummy model
                if X_train.ndim == 3: # (batch, seq, features) -> take last step for Linear
                    dummy_input = X_train[:, -1, :]
                else: # (batch, features)
                    dummy_input = X_train

                if dummy_input.shape[1] != cast(nn.Linear, self.model).in_features:
                     print(f"Warning: DummyTrainableModel input feature mismatch. Expected {cast(nn.Linear, self.model).in_features}, got {dummy_input.shape[1]}")
                     # Fallback to avoid error, real model would handle this better or error earlier
                     dummy_input = torch.randn(dummy_input.shape[0], cast(nn.Linear, self.model).in_features).to(self.device)


                predictions = self.model(dummy_input)
                # Ensure y_train matches prediction shape if it's multi-output
                if predictions.shape != y_train.shape and y_train.ndim == predictions.ndim:
                     current_y_train = y_train[:, :predictions.shape[1]] # Adjust if y_train has more targets than model outputs
                else:
                     current_y_train = y_train

                loss = criterion(predictions, current_y_train)
                loss.backward()
                optimizer.step()
            self.trained = True
            print(f"DummyTrainableModel {self.id_val} finished pseudo-training.")


        def predict(self, X_current_state: Any) -> Dict[str, List[float]]:
            print(f"DummyTrainableModel {self.id_val} predict called.")
            self.model.eval() # Set to eval mode
            mean_preds: List[float] = []
            std_dev_preds: List[float] = []

            if not isinstance(X_current_state, torch.Tensor):
                X_current_state = torch.tensor(X_current_state, dtype=torch.float32).to(self.device)
            
            if X_current_state.ndim == 3: # (batch, seq, features)
                 X_current_state = X_current_state[:, -1, :] # Use last time step for Linear layer
            
            if X_current_state.shape[0] == 0: # Handle empty input case
                return {"mean": [], "std_dev": []}

            # Assuming predict is called for a single sample by the trainer for this dummy
            # or that X_current_state is already shaped as (1, features) or (features)
            if X_current_state.ndim == 1: # (features)
                X_current_state = X_current_state.unsqueeze(0) # (1, features)

            if X_current_state.shape[1] != cast(nn.Linear, self.model).in_features:
                print(f"Warning: DummyTrainableModel predict input feature mismatch. Expected {cast(nn.Linear, self.model).in_features}, got {X_current_state.shape[1]}")
                # Create dummy output of correct size for a single sample
                dummy_output_tensor = torch.rand(1, self.output_horizon_steps)
            else:
                with torch.no_grad():
                    dummy_output_tensor = self.model(X_current_state) # Shape (1, self.output_horizon_steps)

            mean_preds = dummy_output_tensor.squeeze().tolist()
            if not isinstance(mean_preds, list): # If self.output_horizon_steps is 1, tolist() might return a float
                mean_preds = [mean_preds]
            
            std_dev_preds = [0.1] * len(mean_preds)
            
            return {"mean": mean_preds, "std_dev": std_dev_preds}

        def save(self, path: str): print(f"DummyTrainableModel {self.id_val} save to {path}.")
        def load(self, path: str): print(f"DummyTrainableModel {self.id_val} load from {path}.")

    dummy_input_features = 10
    dummy_output_horizon = 1 # For simplicity in dummy model's predict
    dummy_model_instance = DummyTrainableModel(input_features=dummy_input_features, output_horizon_steps=dummy_output_horizon)
    trainer = ModelTrainer(model=dummy_model_instance, training_params={"epochs": 2, "lr": 0.01})

    # Dummy data
    X_dummy = np.random.rand(100, 10)  # 100 samples, 10 features
    y_dummy = np.random.rand(100, 1)   # 100 samples, 1 target

    trainer.train_model(X_dummy, y_dummy)
    cv_results_main = trainer.cross_validate(X_dummy, y_dummy, n_splits=3)
    print(f"Main CV results: {cv_results_main}")

    # Define a search space for hyperparameter tuning
    # For DummyTrainableModel, we'll tune training parameters of ModelTrainer
    # as the model structure itself (input_features, output_horizon_steps) is fixed for this test.
    dummy_search_space = {
        "training_params": {
            "learning_rate": {"type": "float", "low": 1e-4, "high": 1e-1, "log": True},
            "epochs": {"type": "categorical", "choices": [2, 3]} # Keep epochs low for quick test
        }
        # "model_params" could be added here if DummyTrainableModel had tunable structural params
        # e.g., "hidden_layers_dummy": {"type": "int", "low": 1, "high": 2}
    }
    
    print(f"\nStarting hyperparameter tuning for DummyTrainableModel...")
    tune_results = trainer.hyperparameter_tune(
        X_dummy, y_dummy,
        search_space=dummy_search_space,
        n_trials=4, # Small number of trials for quick test
        cv_folds=2  # Small number of folds for quick test
    )
    print(f"Hyperparameter tuning results: {tune_results}")

    # Demonstrate continuous fine-tuning
    print("\nStarting continuous fine-tuning demonstration...")
    X_new_dummy = np.random.rand(50, dummy_input_features) # 50 new samples
    y_new_dummy = np.random.rand(50, dummy_output_horizon)
    
    fine_tune_params = {
        "epochs": 3, # Fewer epochs for fine-tuning
        "learning_rate": trainer.training_params.get("learning_rate", 0.001) / 20, # Much smaller LR
        "batch_size": 16
    }
    fine_tune_results_history = trainer.continuous_fine_tuning(X_new_dummy, y_new_dummy, fine_tuning_params=fine_tune_params)
    print(f"Fine-tuning history from results: {fine_tune_results_history['history']}")
    
    # Optionally, evaluate after fine-tuning on a holdout set if available
    # X_test_dummy = np.random.rand(30, dummy_input_features)
    # y_test_dummy = np.random.rand(30, dummy_output_horizon)
    # test_preds_after_finetune = dummy_model_instance.predict(X_test_dummy)
    # This would require predict to handle batch, or loop.
    # For now, just showing the training part.

    print("\nModelTrainer example run complete.")