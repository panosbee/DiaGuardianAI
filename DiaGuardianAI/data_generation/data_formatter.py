# DiaGuardianAI Data Formatter
# This module will be responsible for preparing synthetic (and later real) data
# into the format required by predictive models.

import numpy as np
import pandas as pd
from typing import Optional, List, Union, Tuple, Dict, Any
# from sklearn.preprocessing import StandardScaler, MinMaxScaler  # Potential future use


class DataFormatter:
    """Formats time-series data into features and targets for predictive models.

    This class takes raw time-series data such as CGM readings,
    insulin doses, and carbohydrate intake, and processes it into a
    format suitable for training machine learning models, particularly
    for glucose prediction. This involves creating sliding windows of
    historical data as features and identifying corresponding future
    glucose values as targets. It also provides a basis for future
    feature engineering and data normalization.

    Attributes:
        cgm_time_step_minutes (int): The time interval in minutes between CGM readings.
        prediction_horizons_steps (List[int]): List of future time steps
            (in terms of number of CGM readings) for which predictions
            are to be made.
        history_window_steps (int): The number of past CGM readings to
            include in each feature window.
        include_cgm_raw (bool): Flag to include raw CGM history as features.
        include_insulin_raw (bool): Flag to include raw insulin history as features.
        include_carbs_raw (bool): Flag to include raw carbohydrate intake history as features.
        include_cgm_roc (bool): Flag to include CGM Rate of Change as features.
        include_cgm_stats (bool): Flag to include CGM window statistics (mean, std, min, max) as features.
        include_cgm_ema (bool): Flag to include CGM Exponential Moving Averages as features.
        ema_spans_minutes (List[int]): Spans for EMA calculation in minutes.
        include_iob (bool): Flag to include current IOB (rapid, long, total) as features.
        include_cob (bool): Flag to include current COB as features.
        include_time_features (bool): Flag to include cyclically encoded time features (hour, day of week).
    """
    def __init__(self, cgm_time_step_minutes: int = 5,
                 prediction_horizons_minutes: List[int] = [
                     10, 20, 30, 40, 50, 60, 120
                 ],
                 history_window_minutes: int = 180,
                 include_cgm_raw: bool = True,
                 include_insulin_raw: bool = True, # Raw historical insulin doses
                 include_carbs_raw: bool = True,   # Raw historical carb intakes
                 include_cgm_roc: bool = True,
                 include_cgm_stats: bool = True,
                 include_cgm_ema: bool = True,
                 ema_spans_minutes: List[int] = [30, 60, 120], # Spans for EMA
                 include_iob: bool = True, # Current IOB values
                 include_cob: bool = True, # Current COB value
                 include_time_features: bool = True
                 ):
        """Initializes the DataFormatter.

        Args:
            cgm_time_step_minutes (int): Interval in minutes between CGM readings.
            prediction_horizons_minutes (List[int]): Future time points (minutes) for predictions.
            history_window_minutes (int): Duration of historical data (minutes) for features.
            include_cgm_raw (bool): Include raw historical CGM readings.
            include_insulin_raw (bool): Include raw historical insulin doses.
            include_carbs_raw (bool): Include raw historical carbohydrate intake.
            include_cgm_roc (bool): Include CGM Rate of Change.
            include_cgm_stats (bool): Include CGM window statistics.
            include_cgm_ema (bool): Include CGM Exponential Moving Averages.
            ema_spans_minutes (List[int]): Spans in minutes for EMA calculations.
            include_iob (bool): Include current Insulin On Board (rapid, long, total).
            include_cob (bool): Include current Carbohydrates On Board.
            include_time_features (bool): Include cyclically encoded time features.
        """
        self.cgm_time_step_minutes: int = cgm_time_step_minutes
        if not self.cgm_time_step_minutes > 0:
            raise ValueError("cgm_time_step_minutes must be positive.")
        self.prediction_horizons_steps: List[int] = sorted([
            int(h / self.cgm_time_step_minutes)
            for h in prediction_horizons_minutes if h > 0
        ])
        if not self.prediction_horizons_steps:
            raise ValueError(
                "prediction_horizons_minutes must contain positive values "
                "that result in at least one valid step."
            )
        self.history_window_steps: int = int(history_window_minutes / self.cgm_time_step_minutes)
        if not self.history_window_steps > 0:
            raise ValueError("history_window_minutes must be positive and result in at least one step.")

        self.include_cgm_raw: bool = include_cgm_raw
        self.include_insulin_raw: bool = include_insulin_raw
        self.include_carbs_raw: bool = include_carbs_raw
        self.include_cgm_roc: bool = include_cgm_roc
        self.include_cgm_stats: bool = include_cgm_stats
        self.include_cgm_ema: bool = include_cgm_ema
        self.ema_spans_steps: List[int] = sorted([
            int(span / self.cgm_time_step_minutes) for span in ema_spans_minutes if span > 0
        ])
        self.include_iob: bool = include_iob
        self.include_cob: bool = include_cob
        self.include_time_features: bool = include_time_features

        # TODO: Initialize scalers (e.g., StandardScaler) for normalization if needed.
        # self.scaler_cgm = StandardScaler()
        # self.scaler_insulin = StandardScaler()
        # self.scaler_carbs = StandardScaler()
        print(
            f"DataFormatter initialized: History steps={self.history_window_steps}, "
            f"Prediction steps={self.prediction_horizons_steps}"
        )

    def create_sliding_windows(self,
                               cgm_series: pd.Series,
                               insulin_bolus_series: Optional[pd.Series] = None, # Raw historical bolus
                               # insulin_basal_series: Optional[pd.Series] = None, # Raw historical basal
                               carb_series: Optional[pd.Series] = None,          # Raw historical carbs
                               iob_rapid_series: Optional[pd.Series] = None,     # Current IOB rapid
                               iob_long_series: Optional[pd.Series] = None,      # Current IOB long
                               cob_series: Optional[pd.Series] = None            # Current COB
                               ) -> Tuple[np.ndarray, np.ndarray]:
        """Creates sliding windows of features and corresponding future target CGM values.

        Args:
            cgm_series (pd.Series): CGM values, indexed by datetime.
            insulin_bolus_series (Optional[pd.Series]): Historical insulin bolus doses.
            carb_series (Optional[pd.Series]): Historical carbohydrate intake.
            iob_rapid_series (Optional[pd.Series]): Current rapid IOB values.
            iob_long_series (Optional[pd.Series]): Current long-acting IOB values.
            cob_series (Optional[pd.Series]): Current COB values.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (features, targets)
                features: 2D array (n_samples, total_features_per_sample)
                targets: 2D array (n_samples, n_prediction_horizons)
        """
        features_list = []
        targets_list = []
        max_horizon_step = self.prediction_horizons_steps[-1]

        for i in range(self.history_window_steps - 1, len(cgm_series) - max_horizon_step):
            current_timestamp = pd.Timestamp(cgm_series.index[i]) # Explicit cast
            
            # --- Sequential Features ---
            sequential_features_for_sample = []
            
            # Raw CGM
            cgm_window_raw = cgm_series.iloc[i - self.history_window_steps + 1 : i + 1].values
            if self.include_cgm_raw:
                sequential_features_for_sample.append(cgm_window_raw)

            # Raw Insulin (Bolus)
            if self.include_insulin_raw and insulin_bolus_series is not None:
                insulin_window_raw = insulin_bolus_series.iloc[i - self.history_window_steps + 1 : i + 1].values
                sequential_features_for_sample.append(insulin_window_raw)
            
            # Raw Carbs
            if self.include_carbs_raw and carb_series is not None:
                carb_window_raw = carb_series.iloc[i - self.history_window_steps + 1 : i + 1].values
                sequential_features_for_sample.append(carb_window_raw)

            # CGM Rate of Change
            if self.include_cgm_roc:
                # Ensure cgm_window_raw is available
                if not self.include_cgm_raw and not any(s is cgm_window_raw for s in sequential_features_for_sample):
                    temp_cgm_window_np = np.asarray(cgm_series.iloc[i - self.history_window_steps + 1 : i + 1].values)
                    cgm_roc = np.diff(temp_cgm_window_np, prepend=temp_cgm_window_np[0])
                else:
                    cgm_window_raw_np = np.asarray(cgm_window_raw)
                    cgm_roc = np.diff(cgm_window_raw_np, prepend=cgm_window_raw_np[0])
                sequential_features_for_sample.append(cgm_roc)

            # CGM EMAs
            if self.include_cgm_ema and self.ema_spans_steps:
                current_cgm_for_ema_np = np.asarray(cgm_window_raw if self.include_cgm_raw or any(s is cgm_window_raw for s in sequential_features_for_sample) else cgm_series.iloc[i - self.history_window_steps + 1 : i + 1].values)
                
                temp_cgm_series_for_ema = pd.Series(current_cgm_for_ema_np) # EMA needs Series
                for span_steps in self.ema_spans_steps:
                    if span_steps > 0 and span_steps < len(temp_cgm_series_for_ema):
                        ema = temp_cgm_series_for_ema.ewm(span=span_steps, adjust=False).mean().values
                        sequential_features_for_sample.append(ema)
            
            # --- Static Features (for the current point 'i') ---
            static_features_for_sample = []

            # CGM Stats
            if self.include_cgm_stats:
                current_cgm_for_stats_np = np.asarray(cgm_window_raw if self.include_cgm_raw or any(s is cgm_window_raw for s in sequential_features_for_sample) else cgm_series.iloc[i - self.history_window_steps + 1 : i + 1].values)
                static_features_for_sample.extend([
                    np.mean(current_cgm_for_stats_np),
                    np.std(current_cgm_for_stats_np),
                    np.min(current_cgm_for_stats_np),
                    np.max(current_cgm_for_stats_np)
                ])
            
            # Current IOB
            if self.include_iob:
                iob_r = iob_rapid_series.iloc[i] if iob_rapid_series is not None else 0.0
                iob_l = iob_long_series.iloc[i] if iob_long_series is not None else 0.0
                static_features_for_sample.extend([iob_r, iob_l, iob_r + iob_l])

            # Current COB
            if self.include_cob and cob_series is not None:
                static_features_for_sample.append(cob_series.iloc[i])
            
            # Time Features
            if self.include_time_features:
                hour = current_timestamp.hour
                day_of_week = current_timestamp.dayofweek # Monday=0, Sunday=6
                static_features_for_sample.extend([
                    np.sin(2 * np.pi * hour / 24.0),
                    np.cos(2 * np.pi * hour / 24.0),
                    np.sin(2 * np.pi * day_of_week / 7.0),
                    np.cos(2 * np.pi * day_of_week / 7.0)
                ])

            # --- Assemble final feature vector for the sample ---
            if not sequential_features_for_sample and not static_features_for_sample:
                continue

            final_sample_features = []
            if sequential_features_for_sample:
                # Stack sequential features: (history_window_steps, num_sequential_feature_types)
                # Then flatten: (history_window_steps * num_sequential_feature_types)
                stacked_sequential = np.stack(sequential_features_for_sample, axis=-1)
                final_sample_features.extend(stacked_sequential.flatten())
            
            if static_features_for_sample:
                final_sample_features.extend(static_features_for_sample)
            
            if not final_sample_features:
                continue
                
            features_list.append(np.array(final_sample_features))
            
            # Target CGM values
            target_values = cgm_series.iloc[np.array([i + step for step in self.prediction_horizons_steps])].values
            targets_list.append(target_values)

        if not features_list or not targets_list:
            return np.array([]), np.array([])

        return np.array(features_list), np.array(targets_list)

    def normalize_features(self, features, fit_scaler=False):
        """
        Normalizes/Standardizes features.
        (Placeholder - actual implementation would use scikit-learn scalers)
        Args:
            features (np.ndarray): Feature array.
            fit_scaler (bool): If true, fit the scaler. Otherwise, use existing.
        Returns:
            np.ndarray: Normalized features.
        """
        # Example:
        # if features.ndim == 3: # (n_samples, n_timesteps, n_features)
        #     # Reshape for scaler, scale, then reshape back
        #     original_shape = features.shape
        #     reshaped_features = features.reshape(-1, original_shape[-1])
        #     if fit_scaler:
        #         self.scaler_cgm.fit(reshaped_features) # Assuming first feature is CGM for this example
        #     scaled_features = self.scaler_cgm.transform(reshaped_features)
        #     return scaled_features.reshape(original_shape)
        # elif features.ndim == 2: # (n_samples, n_features)
        #     if fit_scaler:
        #         self.scaler_cgm.fit(features)
        #     return self.scaler_cgm.transform(features)
        print("Feature normalization placeholder.")
        return features

    def create_dataset(self,
                       cgm_data: list,
                       timestamps: Optional[Union[List, pd.DatetimeIndex]] = None,
                       insulin_bolus_data: Optional[list] = None, # Historical bolus
                       # insulin_basal_data: Optional[list] = None, # Historical basal
                       carb_data: Optional[list] = None,          # Historical carbs
                       iob_rapid_data: Optional[list] = None,     # Current IOB rapid
                       iob_long_data: Optional[list] = None,      # Current IOB long
                       cob_data: Optional[list] = None            # Current COB
                       ) -> Tuple[np.ndarray, np.ndarray]:
        """
        High-level function to process raw data lists into a feature/target dataset.
        """
        processed_timestamps: Optional[pd.DatetimeIndex] = None
        if timestamps is None:
            if not cgm_data: return np.array([]), np.array([])
            processed_timestamps = pd.date_range(
                start="2000-01-01", periods=len(cgm_data),
                freq=f"{self.cgm_time_step_minutes}min"
            )
        elif isinstance(timestamps, list):
            processed_timestamps = pd.to_datetime(timestamps)
        else:
            processed_timestamps = timestamps

        cgm_series = pd.Series(cgm_data, index=processed_timestamps)
        
        def _create_series_if_needed(data: Optional[list], name: str) -> Optional[pd.Series]:
            if data is not None:
                if len(data) != len(cgm_data):
                    raise ValueError(f"{name} data must have the same length as CGM data.")
                return pd.Series(data, index=cgm_series.index)
            return None

        insulin_bolus_series_pd = _create_series_if_needed(insulin_bolus_data, "Insulin bolus")
        carb_series_pd = _create_series_if_needed(carb_data, "Carb")
        iob_rapid_series_pd = _create_series_if_needed(iob_rapid_data, "IOB rapid")
        iob_long_series_pd = _create_series_if_needed(iob_long_data, "IOB long")
        cob_series_pd = _create_series_if_needed(cob_data, "COB")

        features, targets = self.create_sliding_windows(
            cgm_series,
            insulin_bolus_series=insulin_bolus_series_pd,
            carb_series=carb_series_pd,
            iob_rapid_series=iob_rapid_series_pd,
            iob_long_series=iob_long_series_pd,
            cob_series=cob_series_pd
        )
        
        # features = self.normalize_features(features, fit_scaler=True)
        return features, targets

if __name__ == '__main__':
    # This block provides a basic example of how to use the DataFormatter.
    # It demonstrates creating features and targets from sample CGM, insulin, and carb data.

    print("--- DataFormatter Standalone Example ---")
    try:
        formatter_example = DataFormatter(
            cgm_time_step_minutes=5,
            prediction_horizons_minutes=[30, 60, 120],
            history_window_minutes=180,
            include_cgm_raw=True,
            include_insulin_raw=True, # Example will use bolus data for this
            include_carbs_raw=True,
            include_cgm_roc=True,
            include_cgm_stats=True,
            include_cgm_ema=True,
            ema_spans_minutes=[15, 30, 60], # Shorter EMAs for example
            include_iob=True,
            include_cob=True,
            include_time_features=True
        )

        # Generate more sample data
        # Total duration needed: 3hr history + 2hr (max_horizon) = 5 hours
        # 5 hours * (60 min/hr) / (5 min/sample) = 60 samples
        num_total_samples = 60
        example_cgm_main = np.linspace(100, 200, num_total_samples).tolist()
        example_insulin_bolus_main = (np.sin(np.linspace(0, 10, num_total_samples)) * 2).tolist()
        example_insulin_bolus_main = [max(0, x) for x in example_insulin_bolus_main] # No negative bolus
        example_carbs_main = ([0]*20 + [50] + [0]*10 + [30] + [0]*(num_total_samples - 32))
        
        # Mock IOB and COB data (in a real scenario, this would come from the patient model or logs)
        example_iob_rapid_main = (np.cos(np.linspace(0, 5, num_total_samples)) * 1 + 1.5).tolist()
        example_iob_long_main = ([1.0] * num_total_samples) # Constant basal IOB for example
        example_cob_main = ([0]*21 + np.linspace(50,0,10).tolist() + [0]*5 + np.linspace(30,0,5).tolist() + [0]*(num_total_samples-41))
        example_cob_main = [max(0,c) for c in example_cob_main][:num_total_samples]


        example_timestamps_main = pd.date_range(start="2023-01-01T00:00:00", periods=num_total_samples, freq="5min")

        print(f"\nUsing {num_total_samples} data points for the example.")
        # print(f"CGM data points: {len(example_cgm_main)}")
        # print(f"Insulin bolus data points: {len(example_insulin_bolus_main)}")
        # print(f"Carb data points: {len(example_carbs_main)}")
        # print(f"IOB rapid data points: {len(example_iob_rapid_main)}")
        # print(f"COB data points: {len(example_cob_main)}")


        features_main, targets_main = formatter_example.create_dataset(
            cgm_data=example_cgm_main,
            timestamps=example_timestamps_main,
            insulin_bolus_data=example_insulin_bolus_main,
            carb_data=example_carbs_main,
            iob_rapid_data=example_iob_rapid_main,
            iob_long_data=example_iob_long_main,
            cob_data=example_cob_main
        )

        print("\n--- Full DataFormatter Example Results ---")
        if features_main.size > 0 and targets_main.size > 0:
            print(f"Generated features shape: {features_main.shape}")
            print(f"Generated targets shape: {targets_main.shape}")
            print(f"\nNumber of features per sample: {features_main.shape[1]}")
            print("\nFirst feature vector (example):")
            print(features_main[0])
            print("\nFirst target window (example):")
            print(targets_main[0])
        else:
            print("Not enough data to generate features and targets with current settings.")
            print(
                f"Total samples: {num_total_samples}, "
                f"History steps: {formatter_example.history_window_steps}, "
                f"Max Horizon: {formatter_example.prediction_horizons_steps[-1] if formatter_example.prediction_horizons_steps else 'N/A'}"
            )


        # Minimal example for CGM only to test basic functionality
        print("\n--- DataFormatter CGM Raw Only Example ---")
        formatter_cgm_only_example = DataFormatter(
            cgm_time_step_minutes=5,
            prediction_horizons_minutes=[30, 60],
            history_window_minutes=60,
            include_cgm_raw=True,
            include_insulin_raw=False,
            include_carbs_raw=False,
            include_cgm_roc=False,
            include_cgm_stats=False,
            include_cgm_ema=False,
            include_iob=False,
            include_cob=False,
            include_time_features=False
        )
        num_cgm_only_samples = 24 # 1hr history (12 steps) + 1hr max prediction (12 steps)
        example_cgm_cgm_only = np.linspace(120, 160, num_cgm_only_samples).tolist()
        example_timestamps_cgm_only = pd.date_range(start="2023-01-02", periods=num_cgm_only_samples, freq="5min")

        features_cgm, targets_cgm = formatter_cgm_only_example.create_dataset(
            cgm_data=example_cgm_cgm_only,
            timestamps=example_timestamps_cgm_only
        )
        
        if features_cgm.size > 0 and targets_cgm.size > 0:
            print(f"Generated CGM-raw-only features shape: {features_cgm.shape}")
            print(f"Generated CGM-raw-only targets shape: {targets_cgm.shape}")
            # print("\nFirst CGM-raw-only feature vector (example):")
            # print(features_cgm[0])
            # print("\nFirst CGM-raw-only target window (example):")
            # print(targets_cgm[0])
        else:
            print("Not enough data for CGM-raw-only features and targets.")
            print(
                f"Total samples: {num_cgm_only_samples}, "
                f"History steps: {formatter_cgm_only_example.history_window_steps}, "
                f"Max Horizon: {formatter_cgm_only_example.prediction_horizons_steps[-1] if formatter_cgm_only_example.prediction_horizons_steps else 'N/A'}"
            )
    except ValueError as ve:
        print(f"Error during DataFormatter example: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")