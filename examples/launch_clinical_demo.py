#!/usr/bin/env python3
"""
Quick launcher for DiaGuardianAI Clinical Demonstration Panel
"""

import sys
import os

# Add the package to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    """Launch the clinical demonstration panel."""
    print("🏆 DiaGuardianAI - Clinical Demonstration Panel")
    print("=" * 60)
    print("Launching ONE IN A BILLION diabetes management demonstration...")
    print()
    
    try:
        from DiaGuardianAI.demo.interactive_clinical_panel import ClinicalDemoPanel
        
        print("✅ Loading clinical demonstration system...")
        demo = ClinicalDemoPanel()
        
        print("🚀 Starting interactive clinical panel...")
        print("Ready for healthcare provider demonstration!")
        print()
        
        demo.run()
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please ensure all dependencies are installed:")
        print("  - tkinter (usually included with Python)")
        print("  - matplotlib")
        print("  - numpy")
        
    except Exception as e:
        print(f"❌ Error launching clinical demo: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
