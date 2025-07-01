# Tests for DiaGuardianAI.data_generation.synthetic_patient_model

import importlib.util
import pathlib
import numpy as np
import pytest

# Import SyntheticPatient and BaseSyntheticPatient directly from their source
# files to avoid importing the entire package hierarchy during test collection.
_ROOT = pathlib.Path(__file__).resolve().parents[2] / "DiaGuardianAI"

spec_sp = importlib.util.spec_from_file_location(
    "synthetic_patient_model", _ROOT / "data_generation" / "synthetic_patient_model.py"
)
sp_module = importlib.util.module_from_spec(spec_sp)
try:
    spec_sp.loader.exec_module(sp_module)
    SyntheticPatient = sp_module.SyntheticPatient
except ModuleNotFoundError as exc:
    pytest.skip(f"SyntheticPatient tests skipped: {exc}", allow_module_level=True)

spec_bc = importlib.util.spec_from_file_location(
    "base_classes", _ROOT / "core" / "base_classes.py"
)
bc_module = importlib.util.module_from_spec(spec_bc)
spec_bc.loader.exec_module(bc_module)
BaseSyntheticPatient = bc_module.BaseSyntheticPatient

@pytest.fixture
def default_patient_params():
    # These are default parameters for testing, matching the SyntheticPatient defaults where possible
    return {
        "initial_glucose": 120.0,
        "ISF": 50.0,
        "CR": 10.0,
        "target_glucose": 100.0,
        "body_weight_kg": 70.0,
        "basal_rate_U_hr": 1.0,
        "carb_absorption_rate_g_min": 0.05, # Default in SyntheticPatient is 0.5, using a more common value for testing
        "k_d2_to_plasma_rate_per_min": 0.02,
        "iob_decay_rate_per_min": 0.005,
        "k_ip_decay_rate_per_min": 0.03,
        "k_x_prod_rate_per_min": 0.005,
        "insulin_action_decay_rate_per_min": 0.02,
        "p1_glucose_clearance_rate_per_min": 0.003,
        "k_u_id_coeff": 0.0005,
        "k_egp_feedback_strength": 0.005,
        "glucose_utilization_rate_mg_dl_min": 0.1, # Matching new default in SyntheticPatient
        "bolus_absorption_factor": 1.0,
        "bolus_action_factor": 1.0,
        # Protein and Fat defaults
        "protein_glucose_conversion_factor": 0.5,
        "protein_absorption_rate_g_min": 0.01,
        "fat_carb_slowdown_factor_per_g": 0.01,
        "fat_effect_duration_min": 180.0,
        "fat_glucose_effect_mg_dl_per_g_total": 0.3, # Matching new default in SyntheticPatient
        # Exercise defaults
        "exercise_glucose_utilization_increase_factor": 1.5,
        "exercise_insulin_sensitivity_increase_factor": 1.2,
        "cgm_noise_sd": 0.0,  # Set to 0 for predictable CGM in some tests
        "cgm_delay_minutes": 0 # Set to 0 for predictable CGM in some tests
    }

@pytest.fixture
def synthetic_patient(default_patient_params):
    return SyntheticPatient(params=default_patient_params)

def test_synthetic_patient_initialization(synthetic_patient: SyntheticPatient, default_patient_params: dict):
    """Test if SyntheticPatient initializes correctly."""
    assert isinstance(synthetic_patient, BaseSyntheticPatient)
    assert synthetic_patient.G_p == default_patient_params["initial_glucose"]
    assert synthetic_patient.isf == default_patient_params["ISF"] # Check one of the direct params
    assert synthetic_patient.iob == 0.0 # Initial IOB should be 0
    assert synthetic_patient.cob == 0.0 # Initial COB should be 0
    print("test_synthetic_patient_initialization: PASSED")

def test_synthetic_patient_get_cgm_reading(synthetic_patient: SyntheticPatient, default_patient_params: dict):
    """Test the get_cgm_reading method."""
    # With cgm_noise_sd=0 and cgm_delay_minutes=0 in fixture, G_i should equal G_p initially
    # and CGM reading should be G_i (which is G_p)
    assert synthetic_patient.G_i == default_patient_params["initial_glucose"]
    assert synthetic_patient.get_cgm_reading() == default_patient_params["initial_glucose"]
    
    # Simulate a step to see if CGM changes
    synthetic_patient.step(basal_insulin=1.0, bolus_insulin=0, carbs_ingested=0)
    cgm_reading = synthetic_patient.get_cgm_reading()
    assert isinstance(cgm_reading, float)
    assert 39 <= cgm_reading <= 401 # Check within typical CGM bounds
    print("test_synthetic_patient_get_cgm_reading: PASSED")

def test_synthetic_patient_step_basic_effects(synthetic_patient: SyntheticPatient):
    """Test that step changes some key internal states. Detailed dynamics are complex."""
    initial_G_p = synthetic_patient.G_p
    initial_iob = synthetic_patient.iob
    initial_cob = synthetic_patient.cob
    initial_X = synthetic_patient.X

    # Test bolus effect
    synthetic_patient.step(basal_insulin=0, bolus_insulin=2.0, carbs_ingested=0)
    assert synthetic_patient.iob > initial_iob # IOB should increase with bolus

    # Test carb effect
    synthetic_patient.step(basal_insulin=0, bolus_insulin=0, carbs_ingested=50)
    assert synthetic_patient.cob > initial_cob # COB should increase with carbs

    # Test basal effect (X should change over time with insulin presence)
    # Run a few steps with basal to ensure X can build up or change
    for _ in range(5):
        synthetic_patient.step(basal_insulin=1.0, bolus_insulin=0, carbs_ingested=0)
    
    # G_p should change after several steps, X should likely be non-zero if insulin was active
    assert synthetic_patient.G_p != initial_G_p
    # X might be small but should change from initial 0 if insulin is active
    assert synthetic_patient.X != initial_X or synthetic_patient.iob > 0


    print("test_synthetic_patient_step_basic_effects: Simplified checks PASSED. Detailed dynamics require specific scenario tests.")


def test_synthetic_patient_glucose_bounds(default_patient_params: dict):
    """Test if glucose stays within the hard-coded model bounds G_p (20-1000) and CGM (39-401)."""
    
    # Test low bound
    low_params = default_patient_params.copy()
    low_params["initial_glucose"] = 10.0 # Start below model's internal lower bound for G_p
    patient_low = SyntheticPatient(params=low_params)
    assert patient_low.G_p >= 20.0 # Model should enforce its internal G_p floor
    # CGM reading also has its own floor
    assert patient_low.get_cgm_reading() >= 39.0

    # Test high bound
    high_params = default_patient_params.copy()
    high_params["initial_glucose"] = 1200.0 # Start above model's internal upper bound for G_p
    patient_high = SyntheticPatient(params=high_params)
    assert patient_high.G_p <= 1000.0 # Model should enforce its internal G_p ceiling
    # CGM reading also has its own ceiling
    assert patient_high.get_cgm_reading() <= 401.0
    
    # Simulate extreme carb intake to test dynamic ceiling
    extreme_carb_params = default_patient_params.copy()
    extreme_carb_params["initial_glucose"] = 150.0
    patient_extreme_carbs = SyntheticPatient(params=extreme_carb_params)
    for _ in range(10): # Large carb intake over multiple steps
        patient_extreme_carbs.step(basal_insulin=0, bolus_insulin=0, carbs_ingested=200)
    assert patient_extreme_carbs.G_p <= 1000.0
    assert patient_extreme_carbs.get_cgm_reading() <= 401.0

    print("test_synthetic_patient_glucose_bounds: PASSED")

def test_synthetic_patient_get_internal_states(synthetic_patient: SyntheticPatient):
    """Test the get_internal_states method."""
    states = synthetic_patient.get_internal_states()
    assert isinstance(states, dict)
    expected_keys = [
        "G_p_mg_dl", "G_i_mg_dl", "I_p_active_insulin_effect",
        "X_remote_insulin_action", "D1_carb_compartment1_g",
        "D2_carb_compartment2_g", "insulin_on_board_U",
        "carbs_on_board_g", "params", "dt_minutes"
    ]
    for key in expected_keys:
        assert key in states
    assert states["G_p_mg_dl"] == synthetic_patient.G_p
    print("test_synthetic_patient_get_internal_states: PASSED")

# Future tests:
# - Test with exercise events once implemented (properly)
# - Test specific meal models (GI, composition) once implemented
# - Test against known scenarios or other simulators if possible (validation)
# - Test CGM noise and delay once implemented