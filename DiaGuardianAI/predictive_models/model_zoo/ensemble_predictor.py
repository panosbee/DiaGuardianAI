# DiaGuardianAI Ensemble Predictor
# Combines predictions from multiple models.

import sys  # Added sys
import os  # Added os

# Ensure the DiaGuardianAI package is discoverable
if __package__ is None or __package__ == "":
    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")
    )  # Adjusted path
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from DiaGuardianAI.core.base_classes import BasePredictiveModel
from typing import List, Optional, Any, Dict
import numpy as np
from sklearn.linear_model import Ridge
import joblib


class EnsemblePredictor(BasePredictiveModel):
    """Combines predictions from multiple models for improved accuracy.

    This class supports various ensembling strategies such as
    averaging, weighted averaging, and (as a future extension)
    stacking. It takes a list of pre-initialized predictive models and
    combines their outputs.

    Attributes:
        models (List[BasePredictiveModel]): A list of the predictive
            model instances that form the ensemble.
        strategy (str): The ensembling strategy being used
            (e.g., "average", "weighted_average").
        weights (Optional[np.ndarray]): An array of weights used for
            the "weighted_average" strategy. None for other strategies.
        meta_learner (Optional[Any]): The meta-learner model used for
            the "stacking" strategy.
    """

    def __init__(
        self,
        models: List[BasePredictiveModel],
        strategy: str = "average",
        weights: Optional[List[float]] = None,
        meta_learner: Optional[Any] = None,
    ):
        """Initializes the EnsemblePredictor.

        Args:
            models (List[BasePredictiveModel]): A list of instantiated
                predictive models that will be part of the ensemble.
            strategy (str): The ensembling strategy to use. Supported
                options: "average", "weighted_average". "stacking" is
                planned for future implementation. Defaults to "average".
            weights (Optional[List[float]]): A list of weights
                corresponding to each model in `models`. Required if
                `strategy` is "weighted_average". Weights must sum to
                1.0. Defaults to None.
            meta_learner (Optional[Any]): Meta learner to use when
                `strategy` is "stacking". If None, a default ``Ridge``
                model will be created.

        Raises:
            ValueError: If no models are provided, or if weights are
                invalid for the "weighted_average" strategy.
            NotImplementedError: If an unsupported strategy is specified.
        """
        super().__init__()  # Calls __init__ of BasePredictiveModel
        if not models:
            raise ValueError("At least one model must be provided for the ensemble.")
        self.models: List[BasePredictiveModel] = models
        self.strategy: str = strategy.lower()
        self.weights: Optional[np.ndarray] = None
        self.meta_learner: Optional[Any] = None

        if self.strategy == "weighted_average":
            if weights is None or len(weights) != len(self.models):
                raise ValueError(
                    "Weights must be provided for 'weighted_average' strategy "
                    "and match the number of models."
                )
            if not np.isclose(sum(weights), 1.0):
                raise ValueError("Weights for 'weighted_average' must sum to 1.0.")
            self.weights = np.array(weights)
        elif self.strategy == "stacking":
            self.meta_learner = meta_learner if meta_learner is not None else Ridge()
        elif self.strategy not in ["average"]:
            raise NotImplementedError(
                f"Strategy '{strategy}' is not yet implemented. "
                f"Available: 'average', 'weighted_average', 'stacking'."
            )

        print(
            f"EnsemblePredictor initialized with {len(self.models)} models "
            f"and strategy '{self.strategy}'."
        )

    def train(self, X_train: Any, y_train: Any):
        """Trains the ensemble predictor. (Placeholder)

        For "average" or "weighted_average" strategies, this method
        typically ensures that all constituent sub-models are already
        trained. It might optionally offer a way to re-train sub-models.
        For a "stacking" strategy, this method would involve training
        the sub-models (often on cross-validation folds) and then
        training the meta-learner on the out-of-fold predictions from
        the sub-models.

        Args:
            X_train (Any): Training features.
            y_train (Any): Training targets.
        """
        print(f"EnsemblePredictor train called for strategy '{self.strategy}'.")
        # Sub-models are generally expected to be pre-trained or trained externally
        # before being passed to the EnsemblePredictor.
        # This method could be extended to facilitate re-training or fine-tuning.
        # for i, model in enumerate(self.models):
        #     print(
        #         f"Ensuring sub-model {i+1}/{len(self.models)} "
        #         f"({model.__class__.__name__}) is trained..."
        #     )
        #     # model.train(X_train, y_train)  # This line would re-train. Use with caution.

        if self.strategy == "stacking" and self.meta_learner is not None:
            print("Training meta-learner for stacking strategy...")
            base_preds = []
            for model in self.models:
                if hasattr(model, "predict"):
                    pred_dict = model.predict(X_train)
                    base_preds.append(np.array(pred_dict.get("mean", []), dtype=float))
            if base_preds:
                meta_features = np.stack(base_preds, axis=1)
                self.meta_learner.fit(meta_features, y_train)
        print("EnsemblePredictor training placeholder complete.")
        print("EnsemblePredictor training placeholder complete.")

    def predict(self, X_current_state: Any) -> Dict[str, List[float]]:
        """Makes predictions by combining outputs from all sub-models.

        Args:
            X_current_state (Any): Input features for the current state,
                formatted as expected by the sub-models.

        Returns:
            Dict[str, List[float]]: A dictionary containing "mean" and "std_dev"
                of combined predicted future glucose values.

        Raises:
            RuntimeError: If any sub-model fails to produce a prediction.
            ValueError: If sub-models return predictions of
                inconsistent lengths, invalid format, or if an unknown
                ensemble strategy is encountered.
        """
        all_mean_predictions_np: List[np.ndarray] = []
        all_std_dev_predictions_np: List[np.ndarray] = []  # For UQ

        for model_idx, model in enumerate(self.models):
            try:
                # Expect sub-model.predict to return Dict[str, List[float]]
                model_pred_dict = model.predict(X_current_state)

                if (
                    not isinstance(model_pred_dict, dict)
                    or "mean" not in model_pred_dict
                    or not isinstance(model_pred_dict["mean"], list)
                    or not all(
                        isinstance(p, (int, float)) for p in model_pred_dict["mean"]
                    )
                ):
                    raise ValueError(
                        f"Model {model.__class__.__name__} (index {model_idx}) "
                        f"returned an invalid format for 'mean' predictions. "
                        f"Expected Dict with 'mean': List[float]."
                    )
                all_mean_predictions_np.append(
                    np.array(model_pred_dict["mean"], dtype=float)
                )

                # Handle std_dev, optional for now or default to zeros if not present
                if (
                    "std_dev" in model_pred_dict
                    and isinstance(model_pred_dict["std_dev"], list)
                    and all(
                        isinstance(s, (int, float)) for s in model_pred_dict["std_dev"]
                    )
                    and len(model_pred_dict["std_dev"]) == len(model_pred_dict["mean"])
                ):
                    all_std_dev_predictions_np.append(
                        np.array(model_pred_dict["std_dev"], dtype=float)
                    )
                else:
                    # If std_dev is missing or malformed, append zeros of the same length as mean
                    print(
                        f"Warning: Model {model.__class__.__name__} (index {model_idx}) "
                        f"did not provide valid 'std_dev'. Using zeros."
                    )
                    all_std_dev_predictions_np.append(
                        np.zeros_like(all_mean_predictions_np[-1])
                    )

            except Exception as e:
                print(
                    f"Error predicting with sub-model {model_idx} "
                    f"({model.__class__.__name__}): {e}"
                )
                raise RuntimeError(
                    f"Failed to get prediction from sub-model {model_idx} "
                    f"({model.__class__.__name__})"
                ) from e

        if not all_mean_predictions_np:
            print(
                "Warning: No 'mean' predictions received from sub-models to ensemble."
            )
            return {"mean": [], "std_dev": []}

        # Ensure all prediction arrays have the same length
        first_pred_len = len(all_mean_predictions_np[0])
        if not all(
            len(p) == first_pred_len for p in all_mean_predictions_np
        ) or not all(len(s) == first_pred_len for s in all_std_dev_predictions_np):
            error_msg = "Sub-models returned 'mean' or 'std_dev' predictions of inconsistent lengths."
            # Further details could be logged here.
            raise ValueError(error_msg)

        mean_predictions_array = np.stack(
            all_mean_predictions_np, axis=0
        )  # Shape: (num_models, num_horizons)
        std_dev_predictions_array = np.stack(
            all_std_dev_predictions_np, axis=0
        )  # Shape: (num_models, num_horizons)

        ensembled_mean_prediction: np.ndarray
        ensembled_std_dev_prediction: np.ndarray

        if self.strategy == "average":
            ensembled_mean_prediction = np.mean(mean_predictions_array, axis=0)
            # Simple averaging of std_devs (a basic approach for UQ combination)
            ensembled_std_dev_prediction = np.sqrt(
                np.mean(std_dev_predictions_array**2, axis=0)
            )  # Root Mean Square of std_devs

        elif self.strategy == "weighted_average":
            if self.weights is None:
                raise ValueError("Weights are not set for 'weighted_average' strategy.")
            ensembled_mean_prediction = np.average(
                mean_predictions_array, axis=0, weights=self.weights
            )
            # Weighted average of variances (std_dev**2), then sqrt
            ensembled_std_dev_prediction = np.sqrt(
                np.average(std_dev_predictions_array**2, axis=0, weights=self.weights)
            )
        elif self.strategy == "stacking" and self.meta_learner is not None:
            meta_features = mean_predictions_array.T
            ensembled_mean_prediction = self.meta_learner.predict(meta_features)
            ensembled_std_dev_prediction = np.std(mean_predictions_array, axis=0)
        else:
            raise ValueError(
                f"Unknown or unsupported ensemble strategy: {self.strategy}"
            )

        return {
            "mean": ensembled_mean_prediction.tolist(),
            "std_dev": ensembled_std_dev_prediction.tolist(),
        }

    def save(self, path: str):
        """Saves the ensemble predictor's configuration. (Placeholder)

        Saving an ensemble typically involves saving its configuration
        (like strategy and weights) and potentially references to or
        configurations of its sub-models. Sub-models themselves might
        need to be saved separately.

        Args:
            path (str): The base path for saving ensemble information.
        """
        ensemble_config_data = {
            "strategy": self.strategy,
            "weights": self.weights.tolist() if self.weights is not None else None,
            "num_models": len(self.models),
        }
        joblib.dump(ensemble_config_data, f"{path}_ensemble_config.joblib")
        if self.strategy == "stacking" and self.meta_learner is not None:
            joblib.dump(self.meta_learner, f"{path}_meta_learner.joblib")

    def load(self, path: str):
        """Loads the ensemble predictor's configuration. (Placeholder)

        Loading an ensemble requires reconstructing its state, including
        its strategy, weights, and potentially reloading or
        re-instantiating its sub-models and meta-learner.

        Args:
            path (str): The base path from which to load ensemble information.
        """
        ensemble_config_data = joblib.load(f"{path}_ensemble_config.joblib")
        self.strategy = ensemble_config_data.get("strategy", "average")
        weights = ensemble_config_data.get("weights")
        self.weights = np.array(weights) if weights is not None else None
        if self.strategy == "stacking":
            try:
                self.meta_learner = joblib.load(f"{path}_meta_learner.joblib")
            except FileNotFoundError:
                print(
                    f"Warning: Meta-learner file not found at {path}_meta_learner.joblib"
                )
                self.meta_learner = None


if __name__ == "__main__":
    # Dummy models for testing ensemble
    class DummyModel(BasePredictiveModel):
        def __init__(self, model_id: str, prediction_value: float, output_len: int = 3):
            super().__init__()
            self.model_id = model_id
            self.prediction_value = prediction_value
            self.output_len = output_len
            print(f"DummyModel {model_id} initialized to predict {prediction_value}")

        def train(self, X_train, y_train):
            print(f"DummyModel {self.model_id} train called.")

        def predict(
            self, X_current_state
        ) -> Dict[str, List[float]]:  # Changed return type
            print(f"DummyModel {self.model_id} predict called.")
            mean_preds = [self.prediction_value + i for i in range(self.output_len)]
            # Simple std_dev for dummy model, e.g., 5% of the mean value
            std_dev_preds = [p * 0.05 for p in mean_preds]
            return {"mean": mean_preds, "std_dev": std_dev_preds}

        def save(self, path: str):
            print(f"DummyModel {self.model_id} save to {path}.")

        def load(self, path: str):
            print(f"DummyModel {self.model_id} load from {path}.")

    model1 = DummyModel("M1", 100.0, output_len=5)
    model2 = DummyModel("M2", 110.0, output_len=5)
    model3 = DummyModel("M3", 105.0, output_len=5)

    dummy_X = np.random.rand(1, 10, 1)  # Dummy input state

    # Average ensemble
    avg_ensemble = EnsemblePredictor(
        models=[model1, model2, model3], strategy="average"
    )
    avg_preds_dict = avg_ensemble.predict(dummy_X)
    print(
        f"Average Ensemble Predictions: Mean={avg_preds_dict['mean']}, StdDev={avg_preds_dict['std_dev']}"
    )

    # Weighted average ensemble
    try:
        weighted_ensemble = EnsemblePredictor(
            models=[model1, model2, model3],
            strategy="weighted_average",
            weights=[0.5, 0.3, 0.2],
        )
        weighted_preds_dict = weighted_ensemble.predict(dummy_X)
        print(
            f"Weighted Ensemble Predictions: Mean={weighted_preds_dict['mean']}, StdDev={weighted_preds_dict['std_dev']}"
        )
    except Exception as e:
        print(f"Error creating weighted ensemble: {e}")

    # Example of inconsistent prediction length (should raise error)
    # model4_short = DummyModel("M4_short", 120.0, output_len=3)
    # try:
    #     print("\nTesting inconsistent prediction length:")
    #     error_ensemble = EnsemblePredictor(models=[model1, model4_short], strategy="average")
    #     error_ensemble.predict(dummy_X)
    # except ValueError as e:
    #     print(f"Caught expected error: {e}")

    print("\nEnsemblePredictor example run complete.")
