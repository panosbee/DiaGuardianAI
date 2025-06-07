"""
DiaGuardianAI Interactive Clinical Simulator
Real-time diabetes simulation with AI decision making - Like Loop Insight for T1D
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import sys
import os
import threading
from typing import Dict, List, Any

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from DiaGuardianAI.data_generation.human_model_factory import HumanModelFactory
from DiaGuardianAI.data_generation.synthetic_patient_model import SyntheticPatient
from DiaGuardianAI.core.intelligent_meal_system import IntelligentMealSystem
from DiaGuardianAI.models.transformer_zoo import TransformerZoo
from DiaGuardianAI.agents.advanced_multi_agent_system import ContinuousLearningLoop
from DiaGuardianAI.pattern_repository.repository_manager import RepositoryManager

class InteractiveClinicalSimulator:
    """Interactive real-time diabetes simulation for clinical demonstration."""
    
    def __init__(self):
        self.setup_page_config()
        self.initialize_simulation_state()
        self.initialize_ai_system()
    
    def setup_page_config(self):
        """Configure Streamlit for real-time simulation."""
        st.set_page_config(
            page_title="DiaGuardianAI Interactive Simulator",
            page_icon="🔬",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS for simulation interface
        st.markdown("""
        <style>
        .simulation-header {
            font-size: 2.5rem;
            font-weight: bold;
            color: #1f4e79;
            text-align: center;
            margin-bottom: 1rem;
        }
        .status-running {
            background-color: #d4edda;
            color: #155724;
            padding: 0.5rem;
            border-radius: 0.25rem;
            border: 1px solid #c3e6cb;
        }
        .status-paused {
            background-color: #fff3cd;
            color: #856404;
            padding: 0.5rem;
            border-radius: 0.25rem;
            border: 1px solid #ffeaa7;
        }
        .ai-decision {
            background-color: #e8f4fd;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #007bff;
            margin: 0.5rem 0;
        }
        .meal-event {
            background-color: #fff2e6;
            padding: 0.5rem;
            border-radius: 0.25rem;
            border-left: 3px solid #ff8c00;
        }
        .glucose-alert {
            background-color: #f8d7da;
            color: #721c24;
            padding: 0.5rem;
            border-radius: 0.25rem;
            border: 1px solid #f5c6cb;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def initialize_simulation_state(self):
        """Initialize simulation state variables."""
        if 'simulation_running' not in st.session_state:
            st.session_state.simulation_running = False
            st.session_state.simulation_time = 0  # Minutes since start
            st.session_state.glucose_history = []
            st.session_state.insulin_history = []
            st.session_state.meal_history = []
            st.session_state.ai_decisions = []
            st.session_state.current_patient = None
            st.session_state.synthetic_patient = None
            st.session_state.meal_system = None
            st.session_state.last_update = time.time()
    
    def initialize_ai_system(self):
        """Initialize AI system components."""
        if 'ai_system_ready' not in st.session_state:
            with st.spinner('Initializing AI System...'):
                # Create patient population
                factory = HumanModelFactory()
                patients = factory.generate_population(size=5, type_1_ratio=0.6)

                # Initialize AI components
                transformer_zoo = TransformerZoo(input_dim=16)
                repository = RepositoryManager(db_path=":memory:")
                learning_loop = ContinuousLearningLoop(transformer_zoo, repository)
                learning_loop.start_learning_loop()

                # Store in session state
                st.session_state.transformer_zoo = transformer_zoo
                st.session_state.repository = repository
                st.session_state.learning_loop = learning_loop
                st.session_state.ai_system_ready = True
                st.session_state.available_patients = patients

        # Set instance attributes from session state
        if 'learning_loop' in st.session_state:
            self.learning_loop = st.session_state.learning_loop
            self.transformer_zoo = st.session_state.transformer_zoo
            self.repository = st.session_state.repository
    
    def render_simulation_header(self):
        """Render the simulation header with controls."""
        st.markdown('<h1 class="simulation-header">DiaGuardianAI Interactive Clinical Simulator</h1>', unsafe_allow_html=True)
        st.markdown("**Real-time diabetes simulation with AI decision making**")
        
        # Simulation status
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.session_state.simulation_running:
                st.markdown('<div class="status-running">🟢 SIMULATION RUNNING</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="status-paused">⏸️ SIMULATION PAUSED</div>', unsafe_allow_html=True)
        
        with col2:
            sim_hours = st.session_state.simulation_time / 60
            st.metric("Simulation Time", f"{sim_hours:.1f} hours")
        
        with col3:
            if st.session_state.glucose_history:
                current_glucose = st.session_state.glucose_history[-1]['glucose']
                st.metric("Current Glucose", f"{current_glucose:.1f} mg/dL")
            else:
                st.metric("Current Glucose", "-- mg/dL")
        
        with col4:
            ai_decisions_count = len(st.session_state.ai_decisions)
            st.metric("AI Decisions", ai_decisions_count)
    
    def render_simulation_controls(self):
        """Render simulation control panel."""
        st.sidebar.header("Simulation Controls")
        
        # Patient selection
        if 'available_patients' in st.session_state:
            patient_options = [f"Patient {i+1}: {p.diabetes_type.value.title()}, Age {p.age}" 
                              for i, p in enumerate(st.session_state.available_patients)]
            
            selected_idx = st.sidebar.selectbox(
                "Select Patient",
                range(len(patient_options)),
                format_func=lambda x: patient_options[x],
                disabled=st.session_state.simulation_running
            )
            
            if st.session_state.current_patient != selected_idx:
                st.session_state.current_patient = selected_idx
                self.initialize_patient_simulation()
        
        # Simulation controls
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            if st.button("▶️ Start", disabled=st.session_state.simulation_running):
                st.session_state.simulation_running = True
                st.rerun()
        
        with col2:
            if st.button("⏸️ Pause", disabled=not st.session_state.simulation_running):
                st.session_state.simulation_running = False
                st.rerun()
        
        if st.sidebar.button("🔄 Reset Simulation"):
            self.reset_simulation()
            st.rerun()
        
        # Simulation speed
        speed = st.sidebar.slider("Simulation Speed", 1, 10, 5, help="Minutes per second")
        st.session_state.simulation_speed = speed
        
        # Manual interventions
        st.sidebar.subheader("Manual Interventions")
        
        if st.sidebar.button("🍽️ Add Meal"):
            self.add_manual_meal()
        
        if st.sidebar.button("💉 Manual Bolus"):
            self.add_manual_bolus()
        
        # Patient details
        if st.session_state.current_patient is not None:
            patient = st.session_state.available_patients[st.session_state.current_patient]
            st.sidebar.subheader("Patient Details")
            st.sidebar.write(f"**Type:** {patient.diabetes_type.value.title()}")
            st.sidebar.write(f"**Age:** {patient.age} years")
            st.sidebar.write(f"**ISF:** {patient.isf:.1f} mg/dL/U")
            st.sidebar.write(f"**CR:** {patient.cr:.1f} g/U")
            st.sidebar.write(f"**Basal:** {patient.basal_rate_u_hr:.2f} U/hr")
    
    def initialize_patient_simulation(self):
        """Initialize simulation for selected patient."""
        if st.session_state.current_patient is not None:
            patient = st.session_state.available_patients[st.session_state.current_patient]
            
            # Create synthetic patient
            patient_params = patient.simulation_params.copy()
            st.session_state.synthetic_patient = SyntheticPatient(params=patient_params)
            
            # Create meal system
            meal_config = {
                "simulation_duration_hours": 24,
                "time_step_minutes": 5,
                "random_injection_enabled": True,
                "injection_probability_per_hour": 0.1,
                "min_meal_interval_minutes": 180
            }
            st.session_state.meal_system = IntelligentMealSystem(meal_config)
            
            # Reset simulation data
            self.reset_simulation()
    
    def reset_simulation(self):
        """Reset simulation to initial state."""
        st.session_state.simulation_time = 0
        st.session_state.glucose_history = []
        st.session_state.insulin_history = []
        st.session_state.meal_history = []
        st.session_state.ai_decisions = []
        st.session_state.simulation_running = False
        st.session_state.last_update = time.time()
    
    def update_simulation(self):
        """Update simulation by one time step."""
        if not st.session_state.simulation_running or st.session_state.synthetic_patient is None:
            return
        
        current_time = time.time()
        if current_time - st.session_state.last_update < (1.0 / st.session_state.simulation_speed):
            return
        
        st.session_state.last_update = current_time
        st.session_state.simulation_time += 5  # 5-minute steps
        
        # Get current patient state
        patient_state = st.session_state.synthetic_patient.get_state()
        current_glucose = patient_state['cgm']
        
        # Process meal system
        meal_event = st.session_state.meal_system.process_simulation_step(
            st.session_state.simulation_time, current_glucose
        )
        
        # AI decision making every 15 minutes
        if st.session_state.simulation_time % 15 == 0:
            ai_decision = self.make_ai_decision(patient_state, meal_event)
            st.session_state.ai_decisions.append(ai_decision)
        else:
            ai_decision = None
        
        # Apply insulin and meals
        basal_insulin = st.session_state.available_patients[st.session_state.current_patient].basal_rate_u_hr * 5 / 60
        bolus_insulin = ai_decision['bolus_u'] if ai_decision else 0.0
        
        carbs_details = None
        if meal_event:
            carbs_details = {
                "grams": meal_event["grams"],
                "gi_factor": meal_event["gi_factor"],
                "meal_type": meal_event["type"]
            }
            st.session_state.meal_history.append({
                "time": st.session_state.simulation_time,
                "carbs": meal_event["grams"],
                "type": meal_event["type"]
            })
        
        # Step patient simulation
        st.session_state.synthetic_patient.step(
            basal_insulin=basal_insulin,
            bolus_insulin=bolus_insulin,
            carbs_details=carbs_details
        )
        
        # Record data
        st.session_state.glucose_history.append({
            "time": st.session_state.simulation_time,
            "glucose": current_glucose,
            "trend": patient_state.get('cgm_trend', 0)
        })
        
        st.session_state.insulin_history.append({
            "time": st.session_state.simulation_time,
            "basal": basal_insulin * 12,  # Convert to U/hr
            "bolus": bolus_insulin
        })
    
    def make_ai_decision(self, patient_state, meal_event):
        """Make AI decision for current patient state."""
        # Ensure AI system is available
        if not hasattr(self, 'learning_loop') or self.learning_loop is None:
            self.initialize_ai_system()

        # Create state for AI system
        ai_state = {
            "cgm": patient_state['cgm'],
            "cgm_trend": patient_state.get('cgm_trend', 0),
            "iob": patient_state.get('iob', 0),
            "cob": patient_state.get('cob', 0),
            "time_of_day": (st.session_state.simulation_time / 60) % 24,
            "time_since_meal": 120,
            "time_since_bolus": 180,
            "stress_level": 1.0,
            "exercise_recent": 0,
            "isf": st.session_state.available_patients[st.session_state.current_patient].isf,
            "cr": st.session_state.available_patients[st.session_state.current_patient].cr,
            "basal_rate": st.session_state.available_patients[st.session_state.current_patient].basal_rate_u_hr,
            "glucose_variability": 0.2
        }

        # Process through AI system
        try:
            # Use session state directly if instance attribute not available
            learning_loop = getattr(self, 'learning_loop', None) or st.session_state.get('learning_loop')
            if learning_loop:
                result = learning_loop.process_patient_state(ai_state)
                decision = result["insulin_decision"]
            else:
                raise Exception("Learning loop not available")
        except Exception as e:
            # Fallback decision if AI fails
            decision = {
                'basal_rate_u_hr': st.session_state.available_patients[st.session_state.current_patient].basal_rate_u_hr,
                'bolus_u': 0.0,
                'chosen_model': 'fallback',
                'chosen_horizon': 30,
                'predicted_glucose': patient_state['cgm'],
                'reasoning': f'Fallback decision due to AI error: {str(e)}'
            }
        
        return {
            "time": st.session_state.simulation_time,
            "glucose": patient_state['cgm'],
            "basal_rate_u_hr": decision['basal_rate_u_hr'],
            "bolus_u": decision['bolus_u'],
            "chosen_model": decision['chosen_model'],
            "chosen_horizon": decision['chosen_horizon'],
            "predicted_glucose": decision['predicted_glucose'],
            "reasoning": decision['reasoning']
        }
    
    def add_manual_meal(self):
        """Add manual meal intervention."""
        # This would open a dialog for meal input
        # For demo, add a random meal
        meal_carbs = np.random.uniform(20, 60)
        st.session_state.meal_history.append({
            "time": st.session_state.simulation_time,
            "carbs": meal_carbs,
            "type": "manual"
        })
    
    def add_manual_bolus(self):
        """Add manual bolus intervention."""
        # This would open a dialog for bolus input
        # For demo, add a small correction bolus
        bolus_amount = np.random.uniform(0.5, 2.0)
        st.session_state.insulin_history.append({
            "time": st.session_state.simulation_time,
            "basal": 0,
            "bolus": bolus_amount
        })
    
    def render_real_time_charts(self):
        """Render real-time glucose and insulin charts."""
        if not st.session_state.glucose_history:
            st.info("Start simulation to see real-time data")
            return
        
        # Create glucose chart
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Glucose Level', 'Insulin Delivery', 'AI Decisions'),
            vertical_spacing=0.08,
            row_heights=[0.5, 0.25, 0.25]
        )
        
        # Glucose data
        glucose_df = pd.DataFrame(st.session_state.glucose_history)
        
        fig.add_trace(
            go.Scatter(
                x=glucose_df['time'],
                y=glucose_df['glucose'],
                mode='lines+markers',
                name='Glucose',
                line=dict(color='blue', width=2),
                marker=dict(size=4)
            ),
            row=1, col=1
        )
        
        # Target range
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=1)
        fig.add_hline(y=180, line_dash="dash", line_color="red", row=1, col=1)
        fig.add_hrect(y0=70, y1=180, fillcolor="green", opacity=0.1, row=1, col=1)
        
        # Insulin data
        if st.session_state.insulin_history:
            insulin_df = pd.DataFrame(st.session_state.insulin_history)
            
            fig.add_trace(
                go.Bar(
                    x=insulin_df['time'],
                    y=insulin_df['bolus'],
                    name='Bolus',
                    marker_color='orange'
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=insulin_df['time'],
                    y=insulin_df['basal'],
                    mode='lines',
                    name='Basal',
                    line=dict(color='purple', width=2)
                ),
                row=2, col=1
            )
        
        # AI decisions
        if st.session_state.ai_decisions:
            ai_df = pd.DataFrame(st.session_state.ai_decisions)
            
            fig.add_trace(
                go.Scatter(
                    x=ai_df['time'],
                    y=ai_df['predicted_glucose'],
                    mode='markers',
                    name='AI Predictions',
                    marker=dict(color='red', size=8, symbol='diamond')
                ),
                row=3, col=1
            )
        
        # Meal events
        for meal in st.session_state.meal_history:
            fig.add_vline(
                x=meal['time'],
                line_dash="dot",
                line_color="orange",
                annotation_text=f"Meal: {meal['carbs']:.0f}g"
            )
        
        fig.update_layout(
            height=800,
            title="Real-Time Diabetes Simulation",
            showlegend=True
        )
        
        fig.update_xaxes(title_text="Time (minutes)")
        fig.update_yaxes(title_text="Glucose (mg/dL)", row=1, col=1)
        fig.update_yaxes(title_text="Insulin (U)", row=2, col=1)
        fig.update_yaxes(title_text="Predicted Glucose (mg/dL)", row=3, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_ai_decisions_log(self):
        """Render AI decisions log."""
        st.subheader("AI Decision Log")
        
        if not st.session_state.ai_decisions:
            st.info("No AI decisions yet. Start simulation to see AI in action.")
            return
        
        # Show recent decisions
        for decision in st.session_state.ai_decisions[-5:]:  # Last 5 decisions
            time_hours = decision['time'] / 60
            
            st.markdown(f"""
            <div class="ai-decision">
                <strong>Time: {time_hours:.1f}h</strong> | 
                <strong>Glucose: {decision['glucose']:.1f} mg/dL</strong><br>
                <strong>AI Decision:</strong> {decision['chosen_model']} ({decision['chosen_horizon']}min) → 
                {decision['predicted_glucose']:.1f} mg/dL<br>
                <strong>Insulin:</strong> Basal {decision['basal_rate_u_hr']:.2f} U/hr, 
                Bolus {decision['bolus_u']:.2f} U<br>
                <strong>Reasoning:</strong> {decision['reasoning'][:100]}...
            </div>
            """, unsafe_allow_html=True)
    
    def render_simulation_metrics(self):
        """Render simulation performance metrics."""
        if not st.session_state.glucose_history:
            return
        
        glucose_values = [g['glucose'] for g in st.session_state.glucose_history]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_glucose = np.mean(glucose_values)
            st.metric("Average Glucose", f"{avg_glucose:.1f} mg/dL")
        
        with col2:
            tir = np.mean([(70 <= g <= 180) for g in glucose_values]) * 100
            st.metric("Time in Range", f"{tir:.1f}%")
        
        with col3:
            glucose_cv = (np.std(glucose_values) / np.mean(glucose_values)) * 100
            st.metric("Glucose CV", f"{glucose_cv:.1f}%")
        
        with col4:
            hypo_time = np.mean([g < 70 for g in glucose_values]) * 100
            st.metric("Time Below 70", f"{hypo_time:.1f}%")
    
    def run(self):
        """Run the interactive clinical simulator."""
        # Ensure AI system is initialized
        self.initialize_ai_system()

        self.render_simulation_header()
        self.render_simulation_controls()
        
        # Auto-update simulation
        if st.session_state.simulation_running:
            self.update_simulation()
            time.sleep(0.1)  # Small delay
            st.rerun()
        
        # Main content
        tab1, tab2, tab3 = st.tabs(["Real-Time Simulation", "AI Decisions", "Performance Metrics"])
        
        with tab1:
            self.render_real_time_charts()
        
        with tab2:
            self.render_ai_decisions_log()
        
        with tab3:
            self.render_simulation_metrics()

def main():
    """Main function to run the interactive simulator."""
    simulator = InteractiveClinicalSimulator()
    simulator.run()

if __name__ == "__main__":
    main()
