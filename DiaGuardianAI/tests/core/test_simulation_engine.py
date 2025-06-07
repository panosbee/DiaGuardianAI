import unittest
import os
import sys
import pandas as pd
from unittest.mock import MagicMock, patch

# Ensure the DiaGuardianAI module can be found
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from DiaGuardianAI.core.simulation_engine import SimulationEngine
from DiaGuardianAI.data_generation.synthetic_patient_model import SyntheticPatient
from DiaGuardianAI.core.base_classes import BaseAgent
from DiaGuardianAI.utils.patient_sampler import sample_patient_params

class TestSimulationEngine(unittest.TestCase):

    def setUp(self):
        """Set up for test methods."""
        # Use sample_patient_params and then override specific values for testing
        base_params = sample_patient_params()
        base_params['initial_glucose'] = 120.0 # Ensure 'initial_glucose' is used as per SyntheticPatient
        self.patient_params = base_params

        self.patient = SyntheticPatient(params=self.patient_params) # Use 'params', remove 'verbose'
        
        # Mock agent
        self.agent = MagicMock(spec=BaseAgent)
        self.agent.name = "TestAgent"
        self.agent.decide_action.return_value = {"bolus_u": 0.0, "basal_rate_u_hr": self.patient_params.get("basal_rate_U_hr")} # Match action keys
        self.agent.learn = MagicMock()
        self.agent.log_state_action = MagicMock()
        # Removed get_hyperparameters as it's not in BaseAgent


        self.time_step_minutes = 5
        self.max_steps = 1 * (60 // self.time_step_minutes) # 1 hour simulation
        self.cgm_hist_len_for_agent = 12 # Corresponds to 60 minutes if time_step is 5

        self.config = {
            "max_simulation_steps": self.max_steps,
            "time_step_minutes": self.time_step_minutes,
            # "log_simulation_data": True, # Engine always logs to self.simulation_data dict
            "cgm_history_buffer_size": self.cgm_hist_len_for_agent, # Used by engine for its internal buffer
            # "prediction_horizon_minutes": 120 # This is an agent concern, not engine
        }
        self.engine = SimulationEngine(
            patient=self.patient,
            agent=self.agent,
            config=self.config # Removed verbose
        )

    def test_simulation_engine_initialization(self):
        """Test the initialization of the SimulationEngine."""
        self.assertIsNotNone(self.engine.patient)
        self.assertIsNotNone(self.engine.agent)
        self.assertEqual(self.engine.max_simulation_steps, self.max_steps)
        self.assertIsInstance(self.engine.simulation_data, dict) # Check it's a dict
        self.assertEqual(len(self.engine.cgm_history_buffer), 0) # Initially empty, populated during run

    def test_get_current_patient_state_for_agent(self):
        """Test the state gathering for the agent."""
        # Manually step patient and populate engine's cgm_history_buffer
        # to simulate conditions before _get_current_patient_state_for_agent is called by run()
        
        # The engine's _get_current_patient_state_for_agent itself appends to cgm_history_buffer.
        # So, we call it multiple times to build up history.
        # The patient must be stepped externally before each call to get a new CGM.
        
        current_state = {} # Initialize to ensure it's bound
        num_history_points = 3
        if num_history_points == 0: # Handle edge case, though test implies > 0
            # If no history points, _get_current_patient_state_for_agent might not be called
            # or might behave differently. For this test, we assume num_history_points > 0.
            # If it could be 0, the assertions below would need to account for an empty current_state.
            pass

        for i in range(num_history_points):
            # Simulate patient advancing one step
            self.patient.step(
                basal_insulin=self.patient_params.get("basal_rate_U_hr", 1.0), # Use actual basal rate
                bolus_insulin=0.0,
                carbs_ingested=0.0
            )
            # Call the method that internally updates the buffer and returns state
            if i < num_history_points -1: # Don't store the state for intermediate calls, only the last one
                 _ = self.engine._get_current_patient_state_for_agent()
            else:
                current_state = self.engine._get_current_patient_state_for_agent()


        self.assertIn("cgm", current_state)
        self.assertIn("iob", current_state)
        self.assertIn("cob", current_state)
        self.assertIn("cgm_history", current_state)
        self.assertIn("current_simulation_time_minutes", current_state)
        
        # cgm_history in agent state should be padded to cgm_history_buffer_size
        self.assertEqual(len(current_state["cgm_history"]), self.engine.cgm_history_buffer_size)
        # The cgm_history_buffer inside the engine will have num_history_points
        self.assertEqual(len(self.engine.cgm_history_buffer), num_history_points)
        # current_state["cgm"] includes noise. We check it's close to the underlying G_i.
        # A 3-sigma deviation should cover most cases.
        self.assertAlmostEqual(current_state["cgm"], self.patient.G_i, delta=self.patient.cgm_noise_sd * 3)


    def test_simulation_run_completes(self):
        """Test that a short simulation run completes without errors."""
        try:
            self.engine.run()
        except Exception as e:
            self.fail(f"Simulation run failed with exception: {e}")
        
        self.assertEqual(self.engine.current_step_index, self.engine.max_simulation_steps -1) # current_step_index is 0-based
        self.agent.decide_action.assert_called()
        self.assertEqual(self.agent.decide_action.call_count, self.engine.max_simulation_steps)
        if hasattr(self.agent, 'learn') and callable(getattr(self.agent, 'learn')):
            self.agent.learn.assert_called()
            self.assertEqual(self.agent.learn.call_count, self.engine.max_simulation_steps)

    def test_simulation_data_logging(self):
        """Test that simulation data is logged correctly into the dictionary structure."""
        self.engine.run()
        
        self.assertIsInstance(self.engine.simulation_data, dict)
        self.assertTrue(len(self.engine.simulation_data["cgm_readings"]) > 0)
        
        num_steps_logged = self.engine.max_simulation_steps
        
        self.assertEqual(len(self.engine.simulation_data["time_steps_minutes"]), num_steps_logged)
        self.assertEqual(len(self.engine.simulation_data["cgm_readings"]), num_steps_logged)
        self.assertEqual(len(self.engine.simulation_data["iob"]), num_steps_logged)
        self.assertEqual(len(self.engine.simulation_data["cob"]), num_steps_logged)
        self.assertEqual(len(self.engine.simulation_data["actions_taken"]), num_steps_logged)
        self.assertEqual(len(self.engine.simulation_data["rewards"]), num_steps_logged)

        # Check content of the first logged step (index 0)
        self.assertIsNotNone(self.engine.simulation_data["cgm_readings"][0])
        self.assertIsNotNone(self.engine.simulation_data["actions_taken"][0])
        self.assertIsNotNone(self.engine.simulation_data["rewards"][0])
        self.assertEqual(self.engine.simulation_data["time_steps_minutes"][0], 0) # First step is time 0

        # Check content of the last logged step
        last_idx = num_steps_logged - 1
        self.assertIsNotNone(self.engine.simulation_data["cgm_readings"][last_idx])
        self.assertIsNotNone(self.engine.simulation_data["actions_taken"][last_idx])
        self.assertIsNotNone(self.engine.simulation_data["rewards"][last_idx])
        expected_last_time = (self.engine.max_simulation_steps - 1) * self.engine.time_step_minutes
        self.assertEqual(self.engine.simulation_data["time_steps_minutes"][last_idx], expected_last_time)


    def test_cgm_history_buffer_management(self):
        """Test the CGM history buffer is correctly managed by the engine."""
        # Configure for a small buffer and enough steps to fill and roll it
        test_buffer_size = 2
        test_max_steps = 6
        
        local_config = {
            "max_simulation_steps": test_max_steps,
            "time_step_minutes": self.time_step_minutes,
            "cgm_history_buffer_size": test_buffer_size
        }
        
        engine = SimulationEngine(
            patient=self.patient,
            agent=self.agent,
            config=local_config # Removed verbose
        )
        engine.run()
        
        # The engine's internal cgm_history_buffer should not exceed its configured size
        self.assertTrue(len(engine.cgm_history_buffer) <= test_buffer_size)
        if test_max_steps >= test_buffer_size: # If enough steps were run to fill it
             self.assertEqual(len(engine.cgm_history_buffer), test_buffer_size)

# Removed test_save_results_called as SimulationEngine does not have this method

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)