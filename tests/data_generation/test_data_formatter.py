# Tests for DiaGuardianAI.data_generation.data_formatter

import pytest
from DiaGuardianAI.data_generation.data_formatter import DataFormatter
import numpy as np
import pandas as pd
from typing import List

@pytest.fixture
def default_formatter_params():
    return {
        "cgm_time_step_minutes": 5,
        "prediction_horizons_minutes": [30, 60], # Predict 30 and 60 minutes ahead
        "history_window_minutes": 60, # Use 1 hour of history
        "include_cgm": True,
        "include_insulin": False, # Keep it simple for initial tests
        "include_carbs": False
    }

@pytest.fixture
def data_formatter(default_formatter_params):
    return DataFormatter(**default_formatter_params)

@pytest.fixture
def sample_cgm_data():
    # 2 hours of data, 5 min intervals = 24 samples
    return pd.Series(np.linspace(100, 150, 24).tolist(),
                     index=pd.date_range(start="2023-01-01", periods=24, freq="5min"))

@pytest.fixture
def sample_cgm_data_list():
    return np.linspace(100,150,24).tolist()

@pytest.fixture
def sample_timestamps_list():
    return pd.date_range(start="2023-01-01", periods=24, freq="5min").tolist()


def test_data_formatter_initialization(data_formatter, default_formatter_params):
    """Test if DataFormatter initializes correctly."""
    assert data_formatter.cgm_time_step_minutes == default_formatter_params["cgm_time_step_minutes"]
    # prediction_horizons_steps = [h / 5 for h in [30, 60]] = [6, 12]
    assert data_formatter.prediction_horizons_steps == [6, 12]
    # history_window_steps = 60 / 5 = 12
    assert data_formatter.history_window_steps == 12
    print("test_data_formatter_initialization: PASSED")

def test_create_sliding_windows_cgm_only(data_formatter: DataFormatter, sample_cgm_data: pd.Series):
    """Test create_sliding_windows with only CGM data."""
    features, targets = data_formatter.create_sliding_windows(cgm_series=sample_cgm_data)

    # Expected number of samples:
    # len(cgm_series) = 24
    # history_window_steps = 12
    # max_horizon_step = 12 (for 60 min prediction)
    # Loop range: range(12-1, 24 - 12) = range(11, 12) -> only i = 11
    # So, 1 sample expected.
    # Let's check calculation: num_samples = total_points - history_window_steps - max_horizon_step + 1
    # num_samples = 24 - 12 - 12 + 1 = 1
    assert features.shape[0] == 1
    assert targets.shape[0] == 1

    # Feature shape: (n_samples, history_window_steps, n_feature_types)
    # n_feature_types = 1 (only CGM)
    assert features.shape == (1, 12, 1)
    # Target shape: (n_samples, n_prediction_horizons)
    # n_prediction_horizons = 2 (for 30 and 60 min)
    assert targets.shape == (1, 2)

    # Check content of the first (and only) sample
    # History for i=11: cgm_series.iloc[11-12+1 : 11+1] = cgm_series.iloc[0:12]
    expected_feature_sample = np.array(sample_cgm_data.iloc[0:12].values).reshape(12, 1)
    assert np.array_equal(features[0], expected_feature_sample)

    # Targets for i=11: cgm_series.iloc[[11+6, 11+12]] = cgm_series.iloc[[17, 23]]
    expected_target_sample = np.array(sample_cgm_data.iloc[np.array([17, 23])].values)
    assert np.array_equal(targets[0], expected_target_sample)
    print("test_create_sliding_windows_cgm_only: PASSED")

def test_create_dataset_cgm_only(data_formatter: DataFormatter, sample_cgm_data_list: list, sample_timestamps_list: list):
    """Test the high-level create_dataset method with CGM only."""
    features, targets = data_formatter.create_dataset(
        cgm_data=sample_cgm_data_list,
        timestamps=sample_timestamps_list
    )
    assert features.shape == (1, 12, 1)
    assert targets.shape == (1, 2)
    print("test_create_dataset_cgm_only: PASSED")


def test_create_sliding_windows_with_insulin_carbs(sample_cgm_data: pd.Series):
    """Test create_sliding_windows with CGM, insulin, and carb data."""
    formatter = DataFormatter(
        cgm_time_step_minutes=5,
        prediction_horizons_minutes=[30, 60],
        history_window_minutes=60,
        include_cgm=True,
        include_insulin=True,
        include_carbs=True
    )
    # Create dummy insulin and carb series aligned with sample_cgm_data
    insulin_series = pd.Series(np.random.rand(len(sample_cgm_data)) * 2, index=sample_cgm_data.index) # 0-2 U
    carb_series = pd.Series(np.random.randint(0, 30, len(sample_cgm_data)), index=sample_cgm_data.index) # 0-30g

    features, targets = formatter.create_sliding_windows(
        cgm_series=sample_cgm_data,
        insulin_series=insulin_series,
        carb_series=carb_series
    )
    # Expected shapes are the same for samples and targets, but feature depth changes
    assert features.shape[0] == 1
    assert targets.shape[0] == 1
    # Feature shape: (n_samples, history_window_steps, n_feature_types)
    # n_feature_types = 3 (CGM, Insulin, Carbs)
    assert features.shape == (1, 12, 3)
    assert targets.shape == (1, 2)

    # Check content (first feature type should be CGM)
    expected_cgm_feature = np.array(sample_cgm_data.iloc[0:12].values)
    assert np.array_equal(features[0, :, 0], expected_cgm_feature)
    # Check insulin feature
    expected_insulin_feature = np.array(insulin_series.iloc[0:12].values)
    assert np.array_equal(features[0, :, 1], expected_insulin_feature)
    # Check carb feature
    expected_carb_feature = np.array(carb_series.iloc[0:12].values)
    assert np.array_equal(features[0, :, 2], expected_carb_feature)
    print("test_create_sliding_windows_with_insulin_carbs: PASSED")


def test_data_formatter_not_enough_data(data_formatter: DataFormatter):
    """Test behavior when not enough data is provided to form any windows."""
    short_cgm_data = pd.Series([100, 105, 110], index=pd.date_range(start="2023-01-01", periods=3, freq="5min"))
    features, targets = data_formatter.create_sliding_windows(short_cgm_data)
    assert features.shape[0] == 0
    assert targets.shape[0] == 0
    assert features.size == 0 # Check it's truly empty
    assert targets.size == 0
    print("test_data_formatter_not_enough_data: PASSED")

def test_data_formatter_normalization_placeholder(data_formatter: DataFormatter, sample_cgm_data_list: list, sample_timestamps_list: list):
    """Test the placeholder normalization function."""
    features, _ = data_formatter.create_dataset(
        cgm_data=sample_cgm_data_list,
        timestamps=sample_timestamps_list
    )
    if features.size > 0:
        normalized_features = data_formatter.normalize_features(features, fit_scaler=True)
        # Placeholder just prints and returns original, so they should be equal
        assert np.array_equal(features, normalized_features)
    print("test_data_formatter_normalization_placeholder: PASSED")

# Future tests:
# - Test with different history_window_minutes and prediction_horizons_minutes
# - Test feature engineering aspects once implemented
# - Test normalization/scaling with actual scalers
# - Test handling of NaNs or missing data in input series
# - Test create_dataset with insulin and carb data lists