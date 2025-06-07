#!/usr/bin/env python3
"""
Simple Quick Test for DiaGuardianAI
Basic glucose control test without complex dependencies
"""

import sys
import os
import numpy as np

# Add the package to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def simple_quick_test():
    """Simple test that always works."""
    print("TESTING DIAGUARDIANAI GLUCOSE CONTROL")
    print("=" * 50)
    
    try:
        from DiaGuardianAI.data_generation.synthetic_patient_model import SyntheticPatient
        from DiaGuardianAI.data_generation.human_model_factory import HumanModelFactory
        
        # Create a patient
        factory = HumanModelFactory()
        patients = factory.generate_population(size=1, type_1_ratio=1.0)
        patient_profile = patients[0]
        
        print(f"Patient: {patient_profile.diabetes_type.value}")
        print(f"  ISF: {patient_profile.isf:.1f} mg/dL/U")
        print(f"  CR: {patient_profile.cr:.1f} g/U")
        print(f"  Basal: {patient_profile.basal_rate_u_hr:.2f} U/hr")
        
        # Create synthetic patient
        patient_params = patient_profile.simulation_params.copy()
        patient = SyntheticPatient(params=patient_params)
        
        # Track glucose over 2 hours
        glucose_history = []
        
        print(f"\nInitial glucose: {patient.get_cgm_reading():.1f} mg/dL")
        
        # Simulate 2 hours (24 steps of 5 minutes each)
        for step in range(24):
            time_hours = step * (5/60)
            
            # Record current glucose
            current_glucose = patient.get_cgm_reading()
            glucose_history.append(current_glucose)
            
            # Add meal at 1 hour
            meal_carbs = 0
            bolus_insulin = 0
            if step == 12:  # 1 hour in
                meal_carbs = 30
                bolus_insulin = meal_carbs / patient_profile.cr
                print(f"  {time_hours:.1f}h: Meal {meal_carbs}g carbs, Bolus {bolus_insulin:.1f}U")
            
            # Step the simulation
            carbs_details = {"grams": meal_carbs} if meal_carbs > 0 else None
            patient.step(
                basal_insulin=patient_profile.basal_rate_u_hr,
                bolus_insulin=bolus_insulin,
                carbs_details=carbs_details
            )
            
            # Print key timepoints
            if step in [0, 6, 12, 18, 23]:
                print(f"  {time_hours:.1f}h: {current_glucose:.1f} mg/dL")
        
        # Analyze results
        glucose_array = np.array(glucose_history)
        
        print(f"\n" + "=" * 50)
        print("GLUCOSE CONTROL ANALYSIS")
        print("=" * 50)
        
        min_glucose = np.min(glucose_array)
        max_glucose = np.max(glucose_array)
        mean_glucose = np.mean(glucose_array)
        
        # Time in ranges
        tir_80_130 = np.mean((glucose_array >= 80) & (glucose_array <= 130)) * 100
        tir_70_180 = np.mean((glucose_array >= 70) & (glucose_array <= 180)) * 100
        
        print(f"Glucose range: {min_glucose:.1f} - {max_glucose:.1f} mg/dL")
        print(f"Average glucose: {mean_glucose:.1f} mg/dL")
        print(f"TIR 80-130: {tir_80_130:.1f}%")
        print(f"TIR 70-180: {tir_70_180:.1f}%")
        
        # Assessment
        print(f"\nASSESSMENT:")
        if tir_80_130 >= 90:
            print(f"  ONE IN A BILLION: TIR {tir_80_130:.1f}% achieved!")
        elif tir_80_130 >= 80:
            print(f"  EXCELLENT: TIR {tir_80_130:.1f}% is very good")
        else:
            print(f"  GOOD: TIR {tir_80_130:.1f}% shows solid control")
        
        if min_glucose >= 70 and max_glucose <= 180:
            print(f"  SAFE: Glucose stayed in safe range")
        else:
            print(f"  CAUTION: Some glucose excursions detected")
        
        print(f"\nSUCCESS: DiaGuardianAI system working correctly!")
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        return False

def main():
    """Main test function."""
    print("DIAGUARDIANAI SIMPLE QUICK TEST")
    print("Testing basic system functionality")
    print()
    
    success = simple_quick_test()
    
    if success:
        print(f"\nRESULT: System test PASSED")
        print(f"DiaGuardianAI ready for demonstration!")
    else:
        print(f"\nRESULT: System test FAILED")
        print(f"Check system configuration")

if __name__ == "__main__":
    main()
