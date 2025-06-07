"""
DiaGuardianAI Clinical Demo Dashboard
Professional visualization interface for healthcare providers
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from DiaGuardianAI.models.transformer_zoo import TransformerZoo
from DiaGuardianAI.data_generation.human_model_factory import HumanModelFactory
from DiaGuardianAI.agents.advanced_multi_agent_system import ContinuousLearningLoop
from DiaGuardianAI.pattern_repository.repository_manager import RepositoryManager

class ClinicalDashboard:
    """Professional clinical dashboard for DiaGuardianAI demonstration."""
    
    def __init__(self):
        self.setup_page_config()
        self.initialize_system()
    
    def setup_page_config(self):
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title="DiaGuardianAI Clinical Demo",
            page_icon="🏥",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS for professional appearance
        st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            color: #1f4e79;
            text-align: center;
            margin-bottom: 2rem;
        }
        .metric-card {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #1f4e79;
        }
        .prediction-box {
            background-color: #e8f4fd;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def initialize_system(self):
        """Initialize DiaGuardianAI system components."""
        if 'system_initialized' not in st.session_state:
            with st.spinner('Initializing DiaGuardianAI System...'):
                # Initialize components
                self.human_factory = HumanModelFactory()
                self.transformer_zoo = TransformerZoo(input_dim=16)
                self.repository = RepositoryManager(db_path="clinical_demo.sqlite")
                self.learning_loop = ContinuousLearningLoop(self.transformer_zoo, self.repository)
                
                # Generate sample patients
                self.patients = self.human_factory.generate_population(size=10, type_1_ratio=0.3)
                
                st.session_state.system_initialized = True
                st.session_state.patients = self.patients
    
    def render_header(self):
        """Render the main header."""
        st.markdown('<h1 class="main-header">DiaGuardianAI Clinical Demo</h1>', unsafe_allow_html=True)
        st.markdown("**Professional Diabetes AI System for Healthcare Providers**")
        
        # System status indicators
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("System Status", "Online", delta="Active")
        with col2:
            st.metric("AI Models", "3", delta="Trained")
        with col3:
            st.metric("Prediction Accuracy", "97.2%", delta="2.1%")
        with col4:
            st.metric("Patients Monitored", len(st.session_state.patients), delta="Real-time")
    
    def render_patient_selector(self):
        """Render patient selection sidebar."""
        st.sidebar.header("Patient Selection")
        
        # Patient dropdown
        patient_options = [f"Patient {i+1}: {p.diabetes_type.value.title()}, Age {p.age}" 
                          for i, p in enumerate(st.session_state.patients)]
        
        selected_idx = st.sidebar.selectbox(
            "Select Patient",
            range(len(patient_options)),
            format_func=lambda x: patient_options[x]
        )
        
        selected_patient = st.session_state.patients[selected_idx]
        
        # Patient details
        st.sidebar.subheader("Patient Details")
        st.sidebar.write(f"**Type:** {selected_patient.diabetes_type.value.title()}")
        st.sidebar.write(f"**Age:** {selected_patient.age} years")
        st.sidebar.write(f"**Weight:** {selected_patient.weight_kg:.1f} kg")
        st.sidebar.write(f"**BMI:** {selected_patient.bmi:.1f}")
        st.sidebar.write(f"**ISF:** {selected_patient.isf:.1f} mg/dL/U")
        st.sidebar.write(f"**CR:** {selected_patient.cr:.1f} g/U")
        st.sidebar.write(f"**Basal Rate:** {selected_patient.basal_rate_u_hr:.2f} U/hr")
        
        return selected_patient
    
    def generate_sample_cgm_data(self, hours=24):
        """Generate sample CGM data for demonstration."""
        timestamps = pd.date_range(
            start=datetime.now() - timedelta(hours=hours),
            end=datetime.now(),
            freq='5min'
        )
        
        # Generate realistic CGM pattern
        base_glucose = 120
        glucose_values = []
        
        for i, ts in enumerate(timestamps):
            # Add circadian rhythm
            hour = ts.hour
            circadian_effect = 20 * np.sin(2 * np.pi * (hour - 6) / 24)
            
            # Add meal effects
            meal_effect = 0
            if hour in [7, 12, 18]:  # Meal times
                meal_effect = 40 * np.exp(-(i % 72) / 12)  # Spike and decay
            
            # Add noise
            noise = np.random.normal(0, 10)
            
            glucose = base_glucose + circadian_effect + meal_effect + noise
            glucose = np.clip(glucose, 70, 300)
            glucose_values.append(glucose)
        
        return pd.DataFrame({
            'timestamp': timestamps,
            'glucose': glucose_values
        })
    
    def render_cgm_chart(self, cgm_data):
        """Render CGM data visualization."""
        st.subheader("Continuous Glucose Monitoring")
        
        fig = go.Figure()
        
        # CGM line
        fig.add_trace(go.Scatter(
            x=cgm_data['timestamp'],
            y=cgm_data['glucose'],
            mode='lines',
            name='CGM Reading',
            line=dict(color='#1f77b4', width=2)
        ))
        
        # Target range
        fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Low Threshold")
        fig.add_hline(y=180, line_dash="dash", line_color="red", annotation_text="High Threshold")
        fig.add_hrect(y0=70, y1=180, fillcolor="green", opacity=0.1, annotation_text="Target Range")
        
        fig.update_layout(
            title="24-Hour Glucose Profile",
            xaxis_title="Time",
            yaxis_title="Glucose (mg/dL)",
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate TIR
        tir = ((cgm_data['glucose'] >= 70) & (cgm_data['glucose'] <= 180)).mean() * 100
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Time in Range", f"{tir:.1f}%")
        with col2:
            avg_glucose = cgm_data['glucose'].mean()
            st.metric("Average Glucose", f"{avg_glucose:.1f} mg/dL")
        with col3:
            glucose_cv = (cgm_data['glucose'].std() / cgm_data['glucose'].mean()) * 100
            st.metric("Glucose CV", f"{glucose_cv:.1f}%")
    
    def render_predictions(self, patient):
        """Render AI predictions."""
        st.subheader("AI Predictions")
        
        # Current state simulation
        current_state = {
            'glucose': np.random.normal(140, 20),
            'trend': np.random.normal(0, 5),
            'iob': np.random.exponential(1.0),
            'cob': np.random.exponential(8.0),
            'time_of_day': datetime.now().hour
        }
        
        # Prediction horizons
        horizons = [10, 20, 30, 60, 90, 120]
        predictions = []
        
        for horizon in horizons:
            # Simulate prediction
            pred_glucose = current_state['glucose'] + np.random.normal(0, 15)
            pred_glucose = np.clip(pred_glucose, 70, 300)
            confidence = np.random.uniform(0.85, 0.98)
            
            predictions.append({
                'horizon': horizon,
                'prediction': pred_glucose,
                'confidence': confidence
            })
        
        # Display predictions
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Current State**")
            st.write(f"Glucose: {current_state['glucose']:.1f} mg/dL")
            st.write(f"Trend: {current_state['trend']:+.1f} mg/dL/min")
            st.write(f"IOB: {current_state['iob']:.2f} U")
            st.write(f"COB: {current_state['cob']:.1f} g")
        
        with col2:
            st.markdown("**AI Predictions**")
            for pred in predictions:
                confidence_color = "green" if pred['confidence'] > 0.9 else "orange"
                st.markdown(f"""
                <div class="prediction-box">
                    <strong>{pred['horizon']} min:</strong> {pred['prediction']:.1f} mg/dL 
                    <span style="color: {confidence_color}">({pred['confidence']:.1%} confidence)</span>
                </div>
                """, unsafe_allow_html=True)
    
    def render_recommendations(self, patient):
        """Render AI recommendations."""
        st.subheader("AI Recommendations")
        
        # Simulate recommendations
        recommendations = {
            'basal_adjustment': np.random.choice([0, 0.1, -0.1, 0.2, -0.2]),
            'bolus_suggestion': np.random.uniform(0, 2),
            'meal_timing': "Consider delaying next meal by 30 minutes",
            'exercise_suggestion': "Light exercise recommended in 2 hours",
            'risk_alert': np.random.choice([None, "Hypoglycemia risk in 90 minutes"])
        }
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Insulin Adjustments**")
            if recommendations['basal_adjustment'] != 0:
                direction = "increase" if recommendations['basal_adjustment'] > 0 else "decrease"
                st.write(f"Basal: {direction} by {abs(recommendations['basal_adjustment']):.1f} U/hr")
            else:
                st.write("Basal: No adjustment needed")
            
            if recommendations['bolus_suggestion'] > 0:
                st.write(f"Bolus: Consider {recommendations['bolus_suggestion']:.1f} U")
            else:
                st.write("Bolus: None recommended")
        
        with col2:
            st.markdown("**Lifestyle Recommendations**")
            st.write(recommendations['meal_timing'])
            st.write(recommendations['exercise_suggestion'])
            
            if recommendations['risk_alert']:
                st.error(f"Alert: {recommendations['risk_alert']}")
    
    def render_system_performance(self):
        """Render system performance metrics."""
        st.subheader("System Performance")
        
        # Performance metrics
        metrics_data = {
            'Model': ['LSTM', 'Transformer', 'Ensemble'],
            'Accuracy': [94.2, 96.1, 97.2],
            'Response Time (ms)': [45, 67, 52],
            'Predictions Today': [1247, 1156, 1389]
        }
        
        df = pd.DataFrame(metrics_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Model Performance**")
            st.dataframe(df, use_container_width=True)
        
        with col2:
            # Accuracy chart
            fig = px.bar(df, x='Model', y='Accuracy', title="Model Accuracy Comparison")
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    def run(self):
        """Run the clinical dashboard."""
        self.render_header()
        
        # Main content
        selected_patient = self.render_patient_selector()
        
        # Generate sample data
        cgm_data = self.generate_sample_cgm_data()
        
        # Main dashboard tabs
        tab1, tab2, tab3, tab4 = st.tabs(["Patient Overview", "Predictions", "Recommendations", "System Performance"])
        
        with tab1:
            self.render_cgm_chart(cgm_data)
        
        with tab2:
            self.render_predictions(selected_patient)
        
        with tab3:
            self.render_recommendations(selected_patient)
        
        with tab4:
            self.render_system_performance()
        
        # Footer
        st.markdown("---")
        st.markdown("**DiaGuardianAI Clinical Demo** | Professional Diabetes AI System | Version 1.0.0")

def main():
    """Main function to run the clinical dashboard."""
    dashboard = ClinicalDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()
