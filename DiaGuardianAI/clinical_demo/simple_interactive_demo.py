"""
DiaGuardianAI Simple Interactive Demo
Streamlined real-time diabetes simulation for clinical demonstration
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time

class SimpleInteractiveDemo:
    """Simple interactive diabetes simulation for clinical demonstration."""
    
    def __init__(self):
        self.setup_page_config()
        self.initialize_simulation_state()
    
    def setup_page_config(self):
        """Configure Streamlit for real-time simulation."""
        st.set_page_config(
            page_title="DiaGuardianAI Interactive Demo",
            page_icon="🔬",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS
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
        .ai-decision {
            background-color: #e8f4fd;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #007bff;
            margin: 0.5rem 0;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def initialize_simulation_state(self):
        """Initialize simulation state variables."""
        if 'sim_running' not in st.session_state:
            st.session_state.sim_running = False
            st.session_state.sim_time = 0
            st.session_state.glucose_data = []
            st.session_state.insulin_data = []
            st.session_state.meal_data = []
            st.session_state.ai_data = []
            st.session_state.current_glucose = 120.0
            st.session_state.current_iob = 0.0
            st.session_state.current_cob = 0.0
            st.session_state.last_update = time.time()
    
    def render_header(self):
        """Render the simulation header."""
        st.markdown('<h1 class="simulation-header">DiaGuardianAI Interactive Demo</h1>', unsafe_allow_html=True)
        st.markdown("**Real-time diabetes simulation with AI decision making**")
        
        # Status indicators
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.session_state.sim_running:
                st.markdown('<div class="status-running">🟢 SIMULATION RUNNING</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="status-running">⏸️ SIMULATION PAUSED</div>', unsafe_allow_html=True)
        
        with col2:
            sim_hours = st.session_state.sim_time / 60
            st.metric("Simulation Time", f"{sim_hours:.1f} hours")
        
        with col3:
            glucose = st.session_state.current_glucose
            if glucose < 70:
                st.metric("Current Glucose", f"{glucose:.1f} mg/dL", delta="🚨 CRITICAL LOW")
            elif glucose < 80:
                st.metric("Current Glucose", f"{glucose:.1f} mg/dL", delta="⚠️ LOW")
            elif glucose > 250:
                st.metric("Current Glucose", f"{glucose:.1f} mg/dL", delta="⚠️ HIGH")
            else:
                st.metric("Current Glucose", f"{glucose:.1f} mg/dL", delta="✅ SAFE")

        with col4:
            st.metric("AI Decisions", len(st.session_state.ai_data))

        # Safety alerts
        if st.session_state.current_glucose < 70:
            st.error(f"🚨 CRITICAL HYPOGLYCEMIA: {st.session_state.current_glucose:.1f} mg/dL - EMERGENCY TREATMENT NEEDED!")
        elif st.session_state.current_glucose < 80:
            st.warning(f"⚠️ HYPOGLYCEMIA: {st.session_state.current_glucose:.1f} mg/dL - Give 15g fast carbs")
    
    def render_controls(self):
        """Render simulation controls."""
        st.sidebar.header("Simulation Controls")
        
        # Patient selection
        patient_type = st.sidebar.selectbox(
            "Patient Type",
            ["Type 1 Diabetes", "Type 2 Diabetes"],
            disabled=st.session_state.sim_running
        )
        
        # Simulation controls
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            if st.button("▶️ Start", disabled=st.session_state.sim_running):
                st.session_state.sim_running = True
                st.rerun()
        
        with col2:
            if st.button("⏸️ Pause", disabled=not st.session_state.sim_running):
                st.session_state.sim_running = False
                st.rerun()
        
        if st.sidebar.button("🔄 Reset"):
            self.reset_simulation()
            st.rerun()
        
        # Speed control
        speed = st.sidebar.slider("Speed (min/sec)", 1, 10, 5)
        st.session_state.sim_speed = speed
        
        # Manual interventions
        st.sidebar.subheader("Manual Interventions")
        
        if st.sidebar.button("🍽️ Add Meal (30g)"):
            self.add_meal(30)
        
        if st.sidebar.button("💉 Bolus (2U)"):
            self.add_bolus(2.0)
        
        # Patient parameters
        st.sidebar.subheader("Patient Parameters")
        if patient_type == "Type 1 Diabetes":
            st.sidebar.write("**ISF:** 45 mg/dL/U")
            st.sidebar.write("**CR:** 12 g/U")
            st.sidebar.write("**Basal:** 1.2 U/hr")
        else:
            st.sidebar.write("**ISF:** 25 mg/dL/U")
            st.sidebar.write("**CR:** 8 g/U")
            st.sidebar.write("**Basal:** 0.8 U/hr")
    
    def reset_simulation(self):
        """Reset simulation to initial state."""
        st.session_state.sim_time = 0
        st.session_state.glucose_data = []
        st.session_state.insulin_data = []
        st.session_state.meal_data = []
        st.session_state.ai_data = []
        st.session_state.current_glucose = 120.0
        st.session_state.current_iob = 0.0
        st.session_state.current_cob = 0.0
        st.session_state.sim_running = False
        st.session_state.last_update = time.time()
    
    def add_meal(self, carbs):
        """Add a meal to the simulation."""
        st.session_state.meal_data.append({
            "time": st.session_state.sim_time,
            "carbs": carbs,
            "type": "manual"
        })
        st.session_state.current_cob += carbs
    
    def add_bolus(self, units):
        """Add a bolus to the simulation."""
        st.session_state.insulin_data.append({
            "time": st.session_state.sim_time,
            "basal": 0,
            "bolus": units
        })
        st.session_state.current_iob += units
    
    def update_simulation(self):
        """Update simulation by one time step with SAFE glucose dynamics."""
        if not st.session_state.sim_running:
            return

        current_time = time.time()
        if current_time - st.session_state.last_update < (1.0 / st.session_state.sim_speed):
            return

        st.session_state.last_update = current_time
        st.session_state.sim_time += 5  # 5-minute steps

        # SAFE glucose dynamics with hypoglycemia prevention
        glucose_change = 0

        # Baseline hepatic glucose production (prevents dangerous lows)
        baseline_production = 2.0  # mg/dL per 5-min step
        glucose_change += baseline_production

        # COB effect (raises glucose)
        if st.session_state.current_cob > 0:
            cob_effect = min(st.session_state.current_cob * 0.2, 10)  # Reduced effect
            glucose_change += cob_effect
            st.session_state.current_cob = max(0, st.session_state.current_cob - 1.5)  # Slower absorption

        # IOB effect (lowers glucose) - SAFE limits
        if st.session_state.current_iob > 0:
            # Reduce insulin effect if glucose is low
            if st.session_state.current_glucose < 80:
                iob_effect = min(st.session_state.current_iob * 5, 8)  # Much reduced when low
            elif st.session_state.current_glucose < 100:
                iob_effect = min(st.session_state.current_iob * 10, 12)  # Reduced when approaching low
            else:
                iob_effect = min(st.session_state.current_iob * 15, 20)  # Normal effect

            glucose_change -= iob_effect
            st.session_state.current_iob = max(0, st.session_state.current_iob - 0.05)  # Slower decay

        # Counter-regulatory response (prevents severe hypoglycemia)
        if st.session_state.current_glucose < 70:
            counter_reg = (70 - st.session_state.current_glucose) * 0.5  # Stronger response when lower
            glucose_change += counter_reg

        # Random variation (smaller)
        glucose_change += np.random.normal(0, 1.5)

        # Apply change with SAFE limits
        st.session_state.current_glucose += glucose_change
        st.session_state.current_glucose = np.clip(st.session_state.current_glucose, 65, 350)  # SAFE minimum
        
        # Record data
        st.session_state.glucose_data.append({
            "time": st.session_state.sim_time,
            "glucose": st.session_state.current_glucose
        })
        
        # AI decision every 15 minutes
        if st.session_state.sim_time % 15 == 0:
            self.make_ai_decision()
        
        # Random meals
        if np.random.random() < 0.02:  # 2% chance per step
            meal_size = np.random.uniform(15, 45)
            self.add_meal(meal_size)
    
    def make_ai_decision(self):
        """Make a SAFE AI decision with hypoglycemia prevention."""
        glucose = st.session_state.current_glucose
        iob = st.session_state.current_iob

        # SAFETY-FIRST AI logic
        if glucose < 70:
            # CRITICAL: Severe hypoglycemia - emergency carbs
            reasoning = f"🚨 SEVERE HYPOGLYCEMIA ({glucose:.1f} mg/dL) - EMERGENCY: Give 20g fast carbs immediately!"
            self.add_meal(20)  # Emergency carbs
        elif glucose < 80:
            # Mild hypoglycemia - preventive carbs
            reasoning = f"⚠️ Mild hypoglycemia ({glucose:.1f} mg/dL) - Give 15g carbs to prevent severe low"
            self.add_meal(15)  # Preventive carbs
        elif glucose < 100 and iob > 1.0:
            # Trending low with IOB - preventive action
            reasoning = f"⚠️ Glucose {glucose:.1f} mg/dL with IOB {iob:.1f}U - Risk of hypoglycemia, consider 10g carbs"
        elif glucose > 250:
            # Very high glucose - conservative correction
            bolus = min((glucose - 180) / 60, 1.5)  # Very conservative
            reasoning = f"High glucose ({glucose:.1f} mg/dL) - Small correction bolus {bolus:.1f}U"
            self.add_bolus(bolus)
        elif glucose > 200:
            # Moderately high - small correction
            bolus = min((glucose - 160) / 80, 1.0)  # Conservative
            reasoning = f"Elevated glucose ({glucose:.1f} mg/dL) - Small correction {bolus:.1f}U"
            self.add_bolus(bolus)
        else:
            # Safe range
            reasoning = f"✅ Glucose in safe range ({glucose:.1f} mg/dL) - No action needed"

        # SAFE predictions with realistic trends
        trend = np.random.normal(0, 3)  # Smaller variation
        predicted_30min = max(65, glucose + trend)  # Never predict dangerous lows
        predicted_60min = max(65, glucose + trend * 1.2)
        
        st.session_state.ai_data.append({
            "time": st.session_state.sim_time,
            "glucose": glucose,
            "prediction_30min": predicted_30min,
            "prediction_60min": predicted_60min,
            "reasoning": reasoning,
            "model": "SimpleAI"
        })
    
    def render_charts(self):
        """Render real-time charts."""
        if not st.session_state.glucose_data:
            st.info("Start simulation to see real-time data")
            return
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Glucose Level', 'AI Predictions'),
            vertical_spacing=0.1,
            row_heights=[0.7, 0.3]
        )
        
        # Glucose data
        glucose_df = pd.DataFrame(st.session_state.glucose_data)
        
        fig.add_trace(
            go.Scatter(
                x=glucose_df['time'],
                y=glucose_df['glucose'],
                mode='lines+markers',
                name='Glucose',
                line=dict(color='blue', width=3),
                marker=dict(size=5)
            ),
            row=1, col=1
        )
        
        # Target range
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=1)
        fig.add_hline(y=180, line_dash="dash", line_color="red", row=1, col=1)
        fig.add_hrect(y0=70, y1=180, fillcolor="green", opacity=0.1, row=1, col=1)
        
        # AI predictions
        if st.session_state.ai_data:
            ai_df = pd.DataFrame(st.session_state.ai_data)
            
            fig.add_trace(
                go.Scatter(
                    x=ai_df['time'],
                    y=ai_df['prediction_30min'],
                    mode='markers',
                    name='30min Prediction',
                    marker=dict(color='orange', size=8, symbol='diamond')
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=ai_df['time'],
                    y=ai_df['prediction_60min'],
                    mode='markers',
                    name='60min Prediction',
                    marker=dict(color='red', size=8, symbol='square')
                ),
                row=2, col=1
            )
        
        # Meal events
        for meal in st.session_state.meal_data:
            fig.add_vline(
                x=meal['time'],
                line_dash="dot",
                line_color="orange",
                annotation_text=f"🍽️ {meal['carbs']:.0f}g"
            )
        
        fig.update_layout(
            height=600,
            title="Real-Time Diabetes Simulation",
            showlegend=True
        )
        
        fig.update_xaxes(title_text="Time (minutes)")
        fig.update_yaxes(title_text="Glucose (mg/dL)", row=1, col=1)
        fig.update_yaxes(title_text="Predicted Glucose (mg/dL)", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_ai_log(self):
        """Render AI decision log."""
        st.subheader("AI Decision Log")
        
        if not st.session_state.ai_data:
            st.info("No AI decisions yet. Start simulation to see AI in action.")
            return
        
        # Show recent decisions
        for decision in st.session_state.ai_data[-3:]:  # Last 3 decisions
            time_hours = decision['time'] / 60
            
            st.markdown(f"""
            <div class="ai-decision">
                <strong>Time: {time_hours:.1f}h</strong> | 
                <strong>Glucose: {decision['glucose']:.1f} mg/dL</strong><br>
                <strong>Predictions:</strong> 30min: {decision['prediction_30min']:.1f} mg/dL, 
                60min: {decision['prediction_60min']:.1f} mg/dL<br>
                <strong>AI Reasoning:</strong> {decision['reasoning']}
            </div>
            """, unsafe_allow_html=True)
    
    def render_metrics(self):
        """Render performance metrics."""
        if not st.session_state.glucose_data:
            return
        
        glucose_values = [g['glucose'] for g in st.session_state.glucose_data]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_glucose = np.mean(glucose_values)
            st.metric("Average Glucose", f"{avg_glucose:.1f} mg/dL")
        
        with col2:
            tir = np.mean([(70 <= g <= 180) for g in glucose_values]) * 100
            st.metric("Time in Range", f"{tir:.1f}%")
        
        with col3:
            st.metric("Current IOB", f"{st.session_state.current_iob:.1f} U")
        
        with col4:
            st.metric("Current COB", f"{st.session_state.current_cob:.1f} g")
    
    def run(self):
        """Run the simple interactive demo."""
        self.render_header()
        self.render_controls()
        
        # Auto-update simulation
        if st.session_state.sim_running:
            self.update_simulation()
            time.sleep(0.1)
            st.rerun()
        
        # Main content
        tab1, tab2, tab3 = st.tabs(["Real-Time Simulation", "AI Decisions", "Metrics"])
        
        with tab1:
            self.render_charts()
        
        with tab2:
            self.render_ai_log()
        
        with tab3:
            self.render_metrics()

def main():
    """Main function."""
    demo = SimpleInteractiveDemo()
    demo.run()

if __name__ == "__main__":
    main()
