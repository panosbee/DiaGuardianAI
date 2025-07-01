# DiaGuardianAI Metrics
# Functions for calculating various performance and evaluation metrics
# for diabetes management.

import numpy as np
from typing import List, Tuple, Dict

# General helper for safe division
def _safe_divide(numerator: float, denominator: float,
                 default_val: float = 0.0) -> float:
    """Safely divides two numbers. Returns `default_val` if denominator is zero."""
    return numerator / denominator if denominator != 0 else default_val

def calculate_mard(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates Mean Absolute Relative Difference (MARD).

    MARD is a common metric for evaluating the accuracy of CGM systems
    or glucose prediction models. It represents the average relative
    error with respect to the true values.

    Formula: MARD = mean(|(y_pred - y_true) / y_true|) * 100%

    Args:
        y_true (np.ndarray): Array of true glucose values (reference
            values).
        y_pred (np.ndarray): Array of predicted glucose values. Must be
            the same length as `y_true`.

    Returns:
        float: MARD value in percentage. Returns 0.0 if `y_true` is
            empty or if all valid `y_true` values (non-zero) result in
            no calculable relative differences.

    Raises:
        ValueError: If `y_true` and `y_pred` have different lengths.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(
            "Input arrays y_true and y_pred must have the same length."
        )
    if len(y_true) == 0:
        return 0.0
    
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    
    # Filter out cases where y_true is zero to avoid division by zero.
    # Glucose values are typically positive.
    valid_indices = y_true != 0
    if not np.any(valid_indices):
        # This case means all true values were zero, or became zero
        # after filtering. MARD is undefined or can be considered 0
        # if no valid points.
        return 0.0

    abs_relative_diff = np.abs(
        (y_pred[valid_indices] - y_true[valid_indices]) / y_true[valid_indices]
    )
    mard = np.mean(abs_relative_diff) * 100
    return float(mard)

def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates Root Mean Squared Error (RMSE).

    RMSE is a standard way to measure the error of a model in
    predicting quantitative data. It represents the standard deviation
    of the residuals (prediction errors).

    Formula: RMSE = sqrt(mean((y_pred - y_true)^2))

    Args:
        y_true (np.ndarray): Array of true glucose values.
        y_pred (np.ndarray): Array of predicted glucose values. Must be
            the same length as `y_true`.

    Returns:
        float: RMSE value. Returns 0.0 if `y_true` is empty.

    Raises:
        ValueError: If `y_true` and `y_pred` have different lengths.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(
            "Input arrays y_true and y_pred must have the same length."
        )
    if len(y_true) == 0:
        return 0.0
        
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    
    mse = np.mean((y_pred - y_true)**2)
    rmse = np.sqrt(mse)
    return float(rmse)

def calculate_tir(y_true: np.ndarray, lower_bound: float = 70.0,
                  upper_bound: float = 180.0) -> float:
    """Calculates Time In Range (TIR).

    TIR is the percentage of time that glucose values fall within a
    specified target range (e.g., 70-180 mg/dL for euglycemia).

    Args:
        y_true (np.ndarray): Array of true glucose values.
        lower_bound (float): The lower bound of the target glucose range
            (inclusive). Defaults to 70.0 mg/dL.
        upper_bound (float): The upper bound of the target glucose range
            (inclusive). Defaults to 180.0 mg/dL.

    Returns:
        float: TIR value in percentage. Returns 0.0 if `y_true` is empty.
    """
    if len(y_true) == 0:
        return 0.0
    
    y_true = np.asarray(y_true, dtype=float)
    in_range_count = np.sum((y_true >= lower_bound) & (y_true <= upper_bound))
    tir = _safe_divide(float(in_range_count), float(len(y_true))) * 100
    return float(tir)

def calculate_time_below_range(y_true: np.ndarray,
                               threshold: float = 70.0) -> float:
    """Calculates Time Below Range (TBR).

    TBR is the percentage of time glucose values are below a specified
    hypoglycemic threshold.

    Args:
        y_true (np.ndarray): Array of true glucose values.
        threshold (float): The hypoglycemic threshold (exclusive, i.e.,
            values < threshold). Defaults to 70.0 mg/dL.

    Returns:
        float: TBR value in percentage. Returns 0.0 if `y_true` is empty.
    """
    if len(y_true) == 0:
        return 0.0
    y_true = np.asarray(y_true, dtype=float)
    below_range_count = np.sum(y_true < threshold)
    return float(_safe_divide(float(below_range_count), float(len(y_true))) * 100)

def calculate_time_above_range(y_true: np.ndarray,
                               threshold: float = 180.0) -> float:
    """Calculates Time Above Range (TAR).

    TAR is the percentage of time glucose values are above a specified
    hyperglycemic threshold.

    Args:
        y_true (np.ndarray): Array of true glucose values.
        threshold (float): The hyperglycemic threshold (exclusive, i.e.,
            values > threshold). Defaults to 180.0 mg/dL.

    Returns:
        float: TAR value in percentage. Returns 0.0 if `y_true` is empty.
    """
    if len(y_true) == 0:
        return 0.0
    y_true = np.asarray(y_true, dtype=float)
    above_range_count = np.sum(y_true > threshold)
    return float(_safe_divide(float(above_range_count), float(len(y_true))) * 100)


def clarke_error_grid_analysis(y_true: np.ndarray,
                               y_pred: np.ndarray) -> Dict[str, float]:
    """Performs Clarke Error Grid Analysis (CEGA).

    This implementation follows the standard Clarke error grid
    definitions. Points are classified into zones A--E based on
    reference (``y_true``) and predicted (``y_pred``) glucose values.
    Zone boundaries are determined by the rules published in the
    original Clarke error grid paper.

    Args:
        y_true (np.ndarray): Array of true glucose values (reference).
        y_pred (np.ndarray): Array of predicted glucose values (test).
            Must be the same length as `y_true`.

    Returns:
        Dict[str, float]: Percentage of points in zones ``A`` through ``E``.

    Raises:
        ValueError: If `y_true` and `y_pred` have different lengths.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(
            "Input arrays y_true and y_pred must have the same length."
        )
    if len(y_true) == 0:
        return {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0, "E": 0.0}

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    points_in_zone: Dict[str, int] = {
        "A": 0, "B": 0, "C": 0, "D": 0, "E": 0
    }

    for ref, est in zip(y_true, y_pred):
        if ref < 70:
            if est < 70:
                zone = 'A' if abs(ref - est) <= 20 else 'B'
            elif est <= 180:
                zone = 'B'
            else:  # est > 180
                zone = 'E'
        elif ref <= 180:
            if est < 70:
                zone = 'C'
            elif est <= 180:
                zone = 'A' if abs(ref - est) <= 0.20 * ref else 'B'
            else:  # est > 180
                zone = 'C'
        else:  # ref > 180
            if est < 70:
                zone = 'E'
            elif est <= 180:
                zone = 'D'
            else:  # est > 180
                zone = 'A' if abs(ref - est) <= 0.20 * ref else 'B'

        points_in_zone[zone] += 1

    total_points = len(y_true)
    return {
        zone: _safe_divide(float(count), float(total_points)) * 100
        for zone, count in points_in_zone.items()
    }


def calculate_lbgi_hbgi(y_true: np.ndarray) -> Tuple[float, float]:
    """Calculates Low Blood Glucose Index (LBGI) and High Blood Glucose Index (HBGI).

    LBGI and HBGI are risk indices for hypo- and hyperglycemia,
    respectively, based on the work of Kovatchev et al. They transform
    glucose values into a risk score, and LBGI/HBGI are derived from
    the mean of these squared risk scores for low/high glucose readings.

    The formula for the risk transformation `f(bg)` is:
    `f(bg) = 1.509 * ( (ln(bg))^1.084 - 5.381 )`
    Risk components: `rl = max(0, -f(bg))` and `rh = max(0, f(bg))`.
    LBGI = mean of `10 * rl^2` for glucose values where `f(bg) < 0`
           (approx. bg < 112.5 mg/dL).
    HBGI = mean of `10 * rh^2` for glucose values where `f(bg) > 0`
           (approx. bg > 112.5 mg/dL).

    Args:
        y_true (np.ndarray): Array of true glucose values (mg/dL).

    Returns:
        Tuple[float, float]: A tuple containing (LBGI, HBGI). Returns
            (0.0, 0.0) if `y_true` is empty or contains no values
            suitable for calculation (e.g., all values are exactly at
            the f(bg)=0 threshold, or non-positive).

    Warns:
        RuntimeWarning: If `y_true` contains non-positive values, as
            `np.log` will produce -inf or NaN, leading to NaN in results.
    """
    if len(y_true) == 0:
        return 0.0, 0.0
    
    y_true_positive = np.asarray(y_true[y_true > 0], dtype=float)  # Filter out non-positive values for log
    if len(y_true_positive) == 0:
        return 0.0, 0.0  # No valid glucose values for calculation
    
    # Kovatchev et al. formula for risk function f(bg)
    # Using np.seterr to handle potential issues with log on edge cases
    # if any slip through, though y_true_positive should prevent log(0) or log(<0).
    with np.errstate(divide='ignore', invalid='ignore'):  # Suppress warnings for log if needed, though filtered
        f_bg = 1.509 * ( (np.log(y_true_positive)**1.084) - 5.381 )  # Element-wise
    
    rl = np.maximum(0, -f_bg)  # Risk for low glucose
    rh = np.maximum(0,  f_bg)  # Risk for high glucose
    
    # Determine components for LBGI and HBGI based on where f(bg) is
    # negative or positive. The threshold where f(bg) = 0 is
    # approximately 112.508 mg/dL.
    # (ln(bg))^1.084 = 5.381  => ln(bg) = 5.381^(1/1.084)
    # => ln(bg) = 4.723 => bg = exp(4.723) = 112.508
    bg_threshold_f_zero = 112.508 

    lbgi_components = rl[y_true_positive < bg_threshold_f_zero]
    hbgi_components = rh[y_true_positive >= bg_threshold_f_zero]

    lbgi_raw = np.mean(10 * lbgi_components**2) if len(lbgi_components) > 0 else 0.0
    hbgi_raw = np.mean(10 * hbgi_components**2) if len(hbgi_components) > 0 else 0.0
    
    # Handle potential NaN results if f_bg calculation resulted in NaN
    # due to extreme inputs
    lbgi = float(lbgi_raw) if not np.isnan(lbgi_raw) else 0.0
    hbgi = float(hbgi_raw) if not np.isnan(hbgi_raw) else 0.0
    
    return lbgi, hbgi


if __name__ == '__main__':
    # Example usage of the metrics functions
    true_values_example = np.array(
        [70, 80, 100, 150, 180, 200, 60, 250, 120, 90]
    )
    pred_values_example = np.array(
        [75, 85, 95, 160, 170, 190, 70, 230, 110, 95]
    )

    print("--- Metrics Calculation Examples ---")
    mard_val = calculate_mard(true_values_example, pred_values_example)
    print(f"MARD: {mard_val:.2f}%")

    rmse_val = calculate_rmse(true_values_example, pred_values_example)
    print(f"RMSE: {rmse_val:.2f} mg/dL")

    tir_val = calculate_tir(true_values_example, lower_bound=70, upper_bound=180)
    print(f"TIR (70-180 mg/dL): {tir_val:.2f}%")
    
    tbr_val = calculate_time_below_range(true_values_example, threshold=70)
    print(f"Time Below Range (<70 mg/dL): {tbr_val:.2f}%")
    
    tar_val = calculate_time_above_range(true_values_example, threshold=180)
    print(f"Time Above Range (>180 mg/dL): {tar_val:.2f}%")

    ceg_zones_example = clarke_error_grid_analysis(
        true_values_example, pred_values_example
    )
    print("Clarke Error Grid Zones:")
    for zone, perc in ceg_zones_example.items():
        print(f"  Zone {zone}: {perc:.2f}%")

    lbgi_val, hbgi_val = calculate_lbgi_hbgi(true_values_example)
    print(f"LBGI: {lbgi_val:.2f}")
    print(f"HBGI: {hbgi_val:.2f}")

    print("\n--- Edge Case Tests ---")
    print(f"MARD (empty): {calculate_mard(np.array([]), np.array([]))}")
    print(f"RMSE (empty): {calculate_rmse(np.array([]), np.array([]))}")
    print(f"TIR (empty): {calculate_tir(np.array([]))}")
    
    # Test MARD with all true values being zero
    # (should return 0.0 as per implementation)
    print(
        f"MARD (all y_true are zero): "
        f"{calculate_mard(np.array([0,0,0]), np.array([10,20,30]))}"
    )
    
    # CEG with identical values (should be 100% in Zone A)
    ceg_identical_example = clarke_error_grid_analysis(
        true_values_example, true_values_example
    )
    print(f"CEG (identical values, Zone A): {ceg_identical_example['A']:.2f}%")
    
    # LBGI/HBGI with potentially problematic values
    print(
        "LBGI/HBGI with non-positive values "
        "(expect 0.0 or NaN handling to 0.0):"
    )
    lbgi_prob_val, hbgi_prob_val = calculate_lbgi_hbgi(
        np.array([0.0, 50.0, -10.0, 100.0])
    )
    print(f"  LBGI: {lbgi_prob_val:.2f}, HBGI: {hbgi_prob_val:.2f}")
    
    lbgi_all_high, hbgi_all_high = calculate_lbgi_hbgi(
        np.array([150, 200, 250])
    )
    print(
        f"LBGI/HBGI (all high values): LBGI={lbgi_all_high:.2f}, "
        f"HBGI={hbgi_all_high:.2f}"
    )

    lbgi_all_low, hbgi_all_low = calculate_lbgi_hbgi(np.array([50, 60, 70]))
    print(
        f"LBGI/HBGI (all low values): LBGI={lbgi_all_low:.2f}, "
        f"HBGI={hbgi_all_low:.2f}"
    )