# Tests for DiaGuardianAI.core.simulation_engine

import pytest
from DiaGuardianAI.core.simulation_engine import SimulationEngine
from DiaGuardianAI.core.base_classes import BaseSyntheticPatient, BasePredictiveModel, BaseAgent
from typing import List, Dict, Any, Optional

# --- Mock/Dummy Classes for Testing ---

class MockPatient(BaseSyntheticPatient):
    """A simplified mock patient for testing the SimulationEngine.

    This mock patient simulates basic CGM changes in response to insulin and carbs,
    allowing for controlled testing of the simulation loop.
    """
    def __init__(self, params: dict):
        super().__init__(params)
        self.params = params
        self.cgm = params.get("initial_glucose", 120.0)
        self.step_count = 0
        print(f"MockPatient initialized with CGM: {self.cgm}")

    def step(self,
             basal_insulin: float,
             bolus_insulin: float,
             carbs_ingested: float,
             protein_ingested: float = 0.0, # Added to match base
             fat_ingested: float = 0.0,     # Added to match base
             exercise_event: Optional[dict] = None):
        self.step_count += 1
        # Simulate some CGM change (ignoring protein/fat for this mock)
        if carbs_ingested > 0:
            self.cgm += 10
        if bolus_insulin > 0:
            self.cgm -= 5 * bolus_insulin
        self.cgm += basal_insulin * 0.1 # Small basal effect
        if self.cgm < 40: self.cgm = 40
        if self.cgm > 400: self.cgm = 400
        print(f"MockPatient step {self.step_count}: CGM={self.cgm:.1f}, Basal={basal_insulin}, Bolus={bolus_insulin}, Carbs={carbs_ingested}")


    def get_cgm_reading(self) -> float:
        return self.cgm

class MockPredictor(BasePredictiveModel):
    """A simplified mock predictor for testing the SimulationEngine.

    This mock predictor returns a fixed sequence of dummy glucose predictions.
    """
    def __init__(self, output_len: int = 12):
        super().__init__()
        self.output_len = output_len
        print("MockPredictor initialized.")
    def train(self, X_train, y_train): pass
    def predict(self, X_current_state) -> list:
        # print(f"MockPredictor predict called with state: {X_current_state}")
        return [150.0 + i for i in range(self.output_len)] # Dummy predictions
    def save(self, path: str): pass
    def load(self, path: str): pass

class MockAgent(BaseAgent):
    """A simplified mock agent for testing the SimulationEngine.

    This mock agent returns a fixed dummy action when `decide_action` is called.
    """
    def __init__(self, state_dim: int, action_space: Any, predictor: Optional[BasePredictiveModel] = None):
        super().__init__(state_dim, action_space, predictor)
        self.action_count = 0
        print("MockAgent initialized.")

    def decide_action(self, current_state: Any, **kwargs) -> Any:
        self.action_count += 1
        # print(f"MockAgent decide_action called with state: {current_state}, kwargs: {kwargs}")
        # Action structure depends on how SimulationEngine uses it.
        # Let's assume it expects a dict that patient.step can interpret or engine unpacks.
        return {"basal_rate_change_percent": 0.0, "bolus_u": 0.5, "temp_basal_u_hr": 0.0, "temp_basal_duration_minutes": 0} # Dummy action
    def learn(self, experience: Any): pass
    def save(self, path: str): pass
    def load(self, path: str): pass

# --- Pytest Fixtures ---

@pytest.fixture
def mock_patient_instance():
    return MockPatient(params={"initial_glucose": 120.0, "ISF": 50, "CR": 10})

@pytest.fixture
def mock_predictor_instance():
    return MockPredictor(output_len=6) # Predict 6 steps ahead

@pytest.fixture
def mock_agent_instance(mock_predictor_instance):
    # Action space can be a simple placeholder for mock testing
    return MockAgent(state_dim=10, action_space="mock_action_space", predictor=mock_predictor_instance)

@pytest.fixture
def simulation_engine_instance(mock_patient_instance, mock_predictor_instance, mock_agent_instance):
    config = {"duration_days": 1, "time_step_minutes": 5, "max_simulation_steps": 10} # Short simulation for test
    return SimulationEngine(mock_patient_instance, mock_predictor_instance, mock_agent_instance, config)

# --- Test Cases ---

def test_simulation_engine_initialization(simulation_engine_instance):
    """Test if the SimulationEngine initializes correctly."""
    assert simulation_engine_instance is not None
    assert isinstance(simulation_engine_instance.patient, BaseSyntheticPatient)
    assert isinstance(simulation_engine_instance.predictor, BasePredictiveModel)
    assert isinstance(simulation_engine_instance.agent, BaseAgent)
    print("test_simulation_engine_initialization: PASSED")

def test_simulation_engine_run_placeholder(simulation_engine_instance, capsys):
    """Test the placeholder run method of SimulationEngine."""
    # The current run() is a placeholder and just prints.
    # This test will capture stdout to verify the print statements.
    # More detailed tests will be needed when run() is implemented.
    simulation_engine_instance.run()
    captured = capsys.readouterr()
    assert "Simulation engine run started." in captured.out
    assert "Simulation engine run finished." in captured.out
    # assert simulation_engine_instance.mock_patient_instance.step_count > 0 # This would fail as run() is empty
    print("test_simulation_engine_run_placeholder: PASSED (verified prints)")


def test_simulation_engine_plot_glucose_trace_placeholder(simulation_engine_instance, capsys):
    """Test the placeholder plot_glucose_trace method."""
    simulation_engine_instance.plot_glucose_trace()
    captured = capsys.readouterr()
    assert "No glucose trace data to plot or results are not available." in captured.out # Since results is None initially

    # Simulate having some results (though run() doesn't produce them yet)
    simulation_engine_instance.results = {"metrics": {}, "glucose_trace": [100, 110, 120]}
    simulation_engine_instance.plot_glucose_trace()
    captured_with_data = capsys.readouterr()
    assert "Plotting glucose trace..." in captured_with_data.out
    print("test_simulation_engine_plot_glucose_trace_placeholder: PASSED")

# More tests to be added once SimulationEngine.run() has actual logic:
# - test_simulation_flow_one_step (verify components are called)
# - test_simulation_multiple_steps (verify state changes over time)
# - test_simulation_with_meal_event
# - test_simulation_with_agent_learning (if applicable)
# - test_simulation_results_format