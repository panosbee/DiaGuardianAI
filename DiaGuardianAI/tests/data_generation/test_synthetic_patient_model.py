import unittest
import sys
import os
import numpy as np
from typing import Dict, Any

# Ensure the DiaGuardianAI package is discoverable
if __package__ is None or __package__ == '' or __package__ == 'tests.data_generation':
    # If run from tests/data_generation folder or similar relative path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    # If run from the project root
    elif os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')) not in sys.path :
         project_root_alt = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
         if project_root_alt not in sys.path:
            sys.path.insert(0, project_root_alt)


from DiaGuardianAI.data_generation.synthetic_patient_model import SyntheticPatient

class TestSyntheticPatientModel(unittest.TestCase):

    def get_default_patient_params(self) -> Dict[str, Any]:
        return {
            "name": "TestPatient#001", "base_glucose": 100.0, "isf": 50.0, "cr": 10.0,
            "basal_rate_U_hr": 1.0, "weight_kg": 70.0, "time_step_minutes": 5, # Corrected key name
            "initial_glucose": 120.0,
            # Using simplified PK/PD for easier testing if needed, or rely on defaults
            "k_abs1_rapid_per_min": 1/20, "k_abs2_rapid_per_min": 1/30,
            "k_p_decay_rapid_per_min": 1/70, "k_x_prod_rapid_per_min": 1/50,
            "k_x_decay_rapid_per_min": 1/80, "iob_decay_rate_rapid_per_min": 4.6/240,
            "k_abs1_long_per_min": 1/300, "k_abs2_long_per_min": 1/300,
            "k_p_decay_long_per_min": 1/200, "k_x_prod_long_per_min": 1/180,
            "k_x_decay_long_per_min": 1/720, "iob_decay_rate_long_per_min": 4.6/1440,
            "cgm_noise_sd": 0.0 # Disable noise for predictable tests
        }

    def test_patient_initialization(self):
        params = self.get_default_patient_params()
        patient = SyntheticPatient(params=params)
        self.assertIsNotNone(patient)
        self.assertEqual(patient.G_p, params["initial_glucose"])
        self.assertEqual(patient.basal_rate_U_hr, params["basal_rate_U_hr"])
        print("TestSyntheticPatientModel: test_patient_initialization PASSED")

    def test_step_no_inputs(self):
        params = self.get_default_patient_params()
        patient = SyntheticPatient(params=params)
        initial_cgm = patient.get_cgm_reading()
        
        # Step with only basal insulin (from patient params)
        patient.step(basal_insulin=patient.basal_rate_U_hr, bolus_insulin=0, carbs_ingested=0)
        
        cgm_after_step = patient.get_cgm_reading()
        # Expect some change due to basal insulin action or EGP, could be small
        # This test mainly ensures the step runs without error.
        self.assertNotEqual(initial_cgm, cgm_after_step, "CGM should change after a step with basal.")
        print("TestSyntheticPatientModel: test_step_no_inputs PASSED")

    def test_carb_ingestion_response(self):
        params = self.get_default_patient_params()
        patient = SyntheticPatient(params=params)
        initial_cgm = patient.get_cgm_reading()
        initial_cob = patient.cob

        carbs = 50.0
        patient.step(basal_insulin=patient.basal_rate_U_hr, bolus_insulin=0, carbs_ingested=carbs)
        
        self.assertGreater(patient.cob, initial_cob, "COB should increase after carb ingestion.")
        
        # Simulate a few steps to see glucose rise
        cgm_after_meal_step1 = patient.get_cgm_reading()
        for _ in range(12): # 1 hour
            patient.step(basal_insulin=patient.basal_rate_U_hr, bolus_insulin=0, carbs_ingested=0)
        
        cgm_after_1hr = patient.get_cgm_reading()
        self.assertGreater(cgm_after_1hr, initial_cgm, "CGM should rise significantly after carb ingestion over time.")
        print(f"TestSyntheticPatientModel: test_carb_ingestion_response PASSED (Initial: {initial_cgm:.2f}, After 1 step: {cgm_after_meal_step1:.2f} After 1hr: {cgm_after_1hr:.2f})")

    def test_bolus_insulin_response(self):
        params = self.get_default_patient_params()
        # Start with a slightly elevated glucose to see insulin effect clearly
        params["initial_glucose"] = 180.0
        patient = SyntheticPatient(params=params)
        initial_cgm = patient.get_cgm_reading()
        initial_iob_rapid = patient.iob_rapid

        bolus = 5.0 # 5 Units
        patient.step(basal_insulin=patient.basal_rate_U_hr, bolus_insulin=bolus, carbs_ingested=0)
        
        self.assertGreater(patient.iob_rapid, initial_iob_rapid, "Rapid IOB should increase after bolus.")
        
        cgm_after_bolus_step1 = patient.get_cgm_reading()
        # Simulate a few steps to see glucose fall
        for _ in range(12 * 2): # 2 hours
            patient.step(basal_insulin=patient.basal_rate_U_hr, bolus_insulin=0, carbs_ingested=0)
        
        cgm_after_2hr = patient.get_cgm_reading()
        self.assertLess(cgm_after_2hr, initial_cgm, "CGM should fall after bolus insulin over time.")
        print(f"TestSyntheticPatientModel: test_bolus_insulin_response PASSED (Initial: {initial_cgm:.2f}, After 1 step: {cgm_after_bolus_step1:.2f}, After 2hr: {cgm_after_2hr:.2f})")

    def test_exercise_effect(self):
        params = self.get_default_patient_params()
        params["initial_glucose"] = 150.0
        patient = SyntheticPatient(params=params)
        initial_cgm = patient.get_cgm_reading()

        # Simulate some steps before exercise
        for _ in range(6): # 30 mins
            patient.step(basal_insulin=patient.basal_rate_U_hr, bolus_insulin=0, carbs_ingested=0)
        cgm_before_exercise = patient.get_cgm_reading()

        # Start exercise
        exercise = {"duration_minutes": 30, "intensity_factor": 1.0} # Moderate
        patient.step(basal_insulin=patient.basal_rate_U_hr, bolus_insulin=0, carbs_ingested=0, exercise_event=exercise)
        
        # Simulate during exercise
        for _ in range(5): # 25 more minutes of exercise
             patient.step(basal_insulin=patient.basal_rate_U_hr, bolus_insulin=0, carbs_ingested=0)
        cgm_after_exercise = patient.get_cgm_reading()
        self.assertLess(cgm_after_exercise, cgm_before_exercise, "CGM should decrease during exercise.")
        
        # Test carry-over effect (IS should be higher)
        self.assertGreater(patient.exercise_carryover_remaining_min, 0, "Exercise carry-over should be active.")
        initial_is_factor_for_carryover = patient.current_exercise_carryover_additional_is_factor
        self.assertGreater(initial_is_factor_for_carryover, 0, "Carry-over IS factor should be elevated.")

        # Simulate post-exercise
        for _ in range(12): # 1 hour post exercise
            patient.step(basal_insulin=patient.basal_rate_U_hr, bolus_insulin=0, carbs_ingested=0)
        
        self.assertTrue(patient.exercise_carryover_remaining_min <= params.get("exercise_carryover_duration_min", 120) - 60, 
                        "Carry-over duration should decrease.")
        print(f"TestSyntheticPatientModel: test_exercise_effect PASSED (Before Ex: {cgm_before_exercise:.2f}, After Ex: {cgm_after_exercise:.2f})")

    def test_protein_fat_effect(self):
        params = self.get_default_patient_params()
        params["initial_glucose"] = 100.0
        patient = SyntheticPatient(params=params)
        initial_cgm = patient.get_cgm_reading()

        # Ingest protein and fat, no carbs
        protein_g = 30.0
        fat_g = 15.0
        patient.step(basal_insulin=patient.basal_rate_U_hr, bolus_insulin=0, carbs_ingested=0, protein_ingested=protein_g, fat_ingested=fat_g)
        
        self.assertGreater(patient.Prot_G1, 0, "Protein should be in stomach compartment.")
        # Fat_G1 is moved to active_fat_g in the same step if active_fat_g was 0.
        # So, we should check active_fat_g or that the timer has started.
        self.assertGreater(patient.active_fat_g, 0, "Fat should become active after ingestion.")
        self.assertGreater(patient.fat_effect_timer, 0, "Fat effect timer should start after fat ingestion.")


        cgm_trace = [initial_cgm]
        # Simulate for 4 hours to observe delayed rise
        for _ in range(12 * 4): # 4 hours
            patient.step(basal_insulin=patient.basal_rate_U_hr, bolus_insulin=0, carbs_ingested=0)
            cgm_trace.append(patient.get_cgm_reading())
        
        # Expect a rise, but it might be modest and delayed compared to pure carbs
        # Check if peak is later and higher than initial
        peak_cgm_after_initial = max(cgm_trace[12:]) # Check after first hour
        self.assertGreater(peak_cgm_after_initial, initial_cgm + 5, 
                           "CGM should show a delayed rise from protein/fat.")
        print(f"TestSyntheticPatientModel: test_protein_fat_effect PASSED (Initial: {initial_cgm:.2f}, Peak after 1hr: {peak_cgm_after_initial:.2f})")

    def test_iob_cob_tracking(self):
        params = self.get_default_patient_params()
        patient = SyntheticPatient(params=params)

        # Test IOB
        initial_iob_rapid = patient.iob_rapid
        initial_iob_long = patient.iob_long
        patient.step(basal_insulin=patient.basal_rate_U_hr, bolus_insulin=2.0, carbs_ingested=0)
        self.assertGreater(patient.iob_rapid, initial_iob_rapid)
        self.assertGreater(patient.iob_long, initial_iob_long) # Basal contributes to long IOB

        # Test COB
        initial_cob = patient.cob
        patient.step(basal_insulin=patient.basal_rate_U_hr, bolus_insulin=0, carbs_ingested=30.0)
        self.assertGreater(patient.cob, initial_cob)
        
        # Let COB absorb
        for _ in range(12 * 2): # 2 hours
             patient.step(basal_insulin=patient.basal_rate_U_hr, bolus_insulin=0, carbs_ingested=0)
        self.assertLess(patient.cob, initial_cob + 30.0, "COB should decrease over time after absorption.")
        print("TestSyntheticPatientModel: test_iob_cob_tracking PASSED")

if __name__ == '__main__':
    # Add a more verbose runner if desired
    # runner = unittest.TextTestRunner(verbosity=2)
    # unittest.main(testRunner=runner)
    unittest.main()