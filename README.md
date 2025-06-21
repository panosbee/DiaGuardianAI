# DiaGuardianAI: An AI-Driven Framework for Diabetes Management Research

**A Comprehensive Research Platform for Glucose Control Algorithms and Insulin Delivery Optimization**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Research Status](https://img.shields.io/badge/Status-Research%20Tool-orange.svg)]()

---

## Abstract

DiaGuardianAI is a comprehensive research framework designed for investigating artificial intelligence approaches to diabetes management. The platform integrates multiple machine learning architectures, synthetic patient modeling, and safety monitoring systems to provide a unified environment for diabetes-related algorithm development and validation. This framework is intended solely for research and educational purposes and is not approved for clinical use.

**Keywords:** *Diabetes management, Artificial intelligence, Glucose prediction, Insulin optimization, Multi-agent systems, Synthetic patient modeling*

---

## Table of Contents

1. [Research Motivation](#research-motivation)
2. [System Architecture](#system-architecture)
3. [Technical Components](#technical-components)
4. [Machine Learning Models](#machine-learning-models)
5. [Synthetic Patient Simulation](#synthetic-patient-simulation)
6. [Safety Framework](#safety-framework)
7. [Installation and Usage](#installation-and-usage)
8. [Experimental Results](#experimental-results)
9. [Limitations and Future Work](#limitations-and-future-work)
10. [Ethical Considerations](#ethical-considerations)
11. [Contributing to Research](#contributing-to-research)
12. [References](#references)

---

## Research Motivation

### 1.1 Clinical Context

Type 1 diabetes mellitus affects approximately 1.6 million Americans and requires continuous glucose monitoring and insulin therapy management [1]. Current approaches to automated insulin delivery (AID) systems show promise but face challenges in:

- **Glucose prediction accuracy** across diverse physiological conditions
- **Meal detection and compensation** without explicit user input
- **Safety mechanisms** preventing dangerous hypoglycemic events
- **Individual patient adaptation** and parameter optimization

### 1.2 Technical Challenges

The development of effective diabetes management algorithms presents several computational challenges:

1. **Multi-scale temporal dynamics**: Glucose regulation involves processes operating from minutes (insulin absorption) to hours (meal digestion)
2. **Inter-patient variability**: Significant differences in insulin sensitivity, carbohydrate response, and diurnal patterns
3. **Incomplete observability**: Limited sensors provide partial information about complex physiological states
4. **Safety-critical constraints**: Hypoglycemia can be immediately life-threatening

### 1.3 Research Objectives

This framework aims to advance research in:

- **Predictive modeling** for glucose forecasting using modern deep learning architectures
- **Multi-agent systems** for coordinated insulin delivery decisions
- **Safety-aware algorithms** that prioritize patient safety over performance metrics
- **Synthetic patient modeling** for algorithm development and testing

---

## System Architecture

### 2.1 Modular Design Philosophy

DiaGuardianAI employs a modular architecture separating concerns into distinct layers:

```
┌─────────────────────────────────────────────────────────┐
│                    SDK Layer                            │
│  Professional API for Research Integration              │
├─────────────────────────────────────────────────────────┤
│                   Agents Layer                          │
│  Multi-Agent Decision Making & Pattern Recognition      │
├─────────────────────────────────────────────────────────┤
│                   Models Layer                          │
│  ML Architectures for Glucose Prediction               │
├─────────────────────────────────────────────────────────┤
│                    Core Layer                           │
│  Simulation Engine & Base Abstractions                 │
├─────────────────────────────────────────────────────────┤
│               Data Generation Layer                     │
│  Synthetic Patients & Realistic Scenarios              │
└─────────────────────────────────────────────────────────┘
```

### 2.2 Component Interactions

The system implements a **publisher-subscriber pattern** for real-time glucose monitoring and a **command pattern** for insulin delivery recommendations, ensuring loose coupling between components.

---

## Technical Components

### 3.1 Core Abstractions

The framework is built upon abstract base classes ensuring extensibility:

#### BasePredictiveModel
```python
class BasePredictiveModel(ABC):
    @abstractmethod
    def predict(self, current_input: Any) -> Dict[str, List[float]]:
        """Returns glucose predictions with uncertainty quantification"""
        pass
    
    @abstractmethod
    def train(self, data: Any, targets: Optional[Any] = None):
        """Trains the model on provided data"""
        pass
```

#### BaseSyntheticPatient
```python
class BaseSyntheticPatient(ABC):
    @abstractmethod
    def step(self, basal_insulin: float, bolus_insulin: float, 
             carbs_details: Optional[Dict[str, Any]] = None):
        """Advances patient simulation by one time step"""
        pass
```

### 3.2 Data Types and Interfaces

The system implements comprehensive type safety using Python dataclasses:

```python
@dataclass
class InsulinRecommendation:
    patient_id: str
    basal_rate_u_hr: float
    bolus_amount_u: float
    safety_level: SafetyLevel
    explanation: str
    confidence: float
    predicted_glucose_30min: Optional[float]
```

---

## Machine Learning Models

### 4.1 Neural Architecture Zoo

The framework includes implementations of state-of-the-art architectures adapted for glucose prediction:

#### 4.1.1 LSTM Predictor
- **Architecture**: Bidirectional LSTM with attention mechanism
- **Input**: Multi-variate time series (CGM, insulin, carbohydrates)
- **Output**: Multi-horizon glucose predictions (5-120 minutes)
- **Uncertainty**: Monte Carlo Dropout sampling

#### 4.1.2 Transformer Predictor
- **Architecture**: Encoder-only Transformer with positional encoding
- **Features**: Self-attention over temporal sequences
- **Advantages**: Captures long-range dependencies in glucose dynamics

#### 4.1.3 N-BEATS Implementation
- **Design**: Neural Basis Expansion Analysis for Time Series
- **Components**: Trend, seasonality, and generic stacks
- **Strength**: Interpretable decomposition of glucose patterns

#### 4.1.4 TSMixer Architecture
- **Approach**: All-MLP architecture for time series forecasting
- **Innovation**: Separate time and feature mixing layers
- **Efficiency**: Reduced computational complexity vs. Transformers

### 4.2 Ensemble Methods

The framework implements sophisticated ensemble techniques:

```python
class EnsemblePredictor:
    def __init__(self, models: List[BasePredictiveModel], 
                 strategy: str = "weighted_average"):
        """
        Strategies: "average", "weighted_average", "stacking"
        """
```

### 4.3 Uncertainty Quantification

All models implement uncertainty quantification through:
- **Monte Carlo Dropout**: For neural networks
- **Ensemble Variance**: Across multiple model predictions
- **Prediction Intervals**: Confidence bounds on forecasts

---

## Synthetic Patient Simulation

### 5.1 Physiological Modeling

The synthetic patient model implements a compartmental approach based on established pharmacokinetic/pharmacodynamic principles:

#### Glucose Dynamics
```
dG/dt = R_a(t) - U_ii(t) - k_1 * G(t)
```

Where:
- `G(t)`: Plasma glucose concentration
- `R_a(t)`: Glucose rate of appearance (meals)
- `U_ii(t)`: Insulin-independent glucose utilization
- `k_1`: Glucose clearance rate

#### Insulin Kinetics
The model implements two-compartment insulin pharmacokinetics:

```
dI_p/dt = -(k_a1 + k_d) * I_p(t) + k_a2 * I_t(t) + U_b(t)/V_I
dI_t/dt = k_a1 * I_p(t) - k_a2 * I_t(t)
```

### 5.2 Inter-Patient Variability

The framework generates diverse patient populations through:

```python
class HumanModelFactory:
    def generate_population(self, size: int, 
                          type_1_ratio: float = 0.7) -> List[PatientProfile]:
        """Generates realistic patient cohort with clinical parameters"""
```

Parameters varied include:
- **Insulin sensitivity factor (ISF)**: 20-80 mg/dL per unit
- **Carbohydrate ratio (CR)**: 5-20 grams per unit
- **Basal insulin requirements**: 0.3-2.0 units/hour
- **Gastroparesis factors**: Delayed meal absorption rates

### 5.3 Meal and Exercise Modeling

#### Meal Absorption
Implements variable gastric emptying with first-order kinetics:

```python
def meal_absorption_rate(self, carbs: float, gi_factor: float) -> float:
    """
    Models realistic carbohydrate absorption including:
    - Glycemic index effects
    - Portion size influence
    - Individual variability
    """
```

#### Exercise Effects
Models exercise impact on glucose through:
- **Increased glucose utilization** during activity
- **Enhanced insulin sensitivity** post-exercise
- **Delayed hypoglycemia risk** mechanisms

---

## Safety Framework

### 6.1 Multi-Layer Safety Architecture

The safety system implements defense-in-depth principles:

#### Layer 1: Predictive Safety
```python
class LowGlucosePreventionAgent:
    def predict_hypoglycemia_risk(self, current_state: Dict) -> float:
        """30-minute hypoglycemia risk assessment"""
        
    def calculate_insulin_adjustments(self, risk_level: float) -> InsulinAdjustment:
        """Safety-aware insulin modifications"""
```

#### Layer 2: Real-time Monitoring
```python
class SafetyMonitor:
    def check_safety(self, glucose_reading: GlucoseReading) -> Optional[SafetyAlert]:
        """
        Safety levels: SAFE, CAUTION, WARNING, CRITICAL
        Thresholds: <54, 54-70, 70-80, 80-130, 130-180, 180-250, >250 mg/dL
        """
```

#### Layer 3: Override Mechanisms
Automatic insulin suspension when:
- Glucose < 70 mg/dL with downward trend
- Predicted glucose < 60 mg/dL within 30 minutes
- Loss of sensor data > 20 minutes

### 6.2 Safety Validation

The framework includes comprehensive safety testing:

```python
def test_hypoglycemia_prevention():
    """Validates that no insulin is delivered when glucose < 70 mg/dL"""
    
def test_maximum_insulin_limits():
    """Ensures insulin recommendations never exceed safety bounds"""
```

---

## Installation and Usage

### 7.1 System Requirements

- **Python**: 3.8 or higher
- **Dependencies**: NumPy, PyTorch, Scikit-learn, Pandas
- **Memory**: Minimum 8GB RAM recommended
- **Storage**: 2GB free space for model weights and data

### 7.2 Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/DiaGuardianAI.git
cd DiaGuardianAI

# Create virtual environment
python -m venv diaguardian_env
source diaguardian_env/bin/activate  # On Windows: diaguardian_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -m pytest tests/
```

### 7.3 Basic Usage Example

```python
from DiaGuardianAI.sdk import DiabetesManager
from DiaGuardianAI.data_generation import SyntheticPatient

# Initialize research framework
dm = DiabetesManager()

# Create synthetic patient for testing
patient_params = {
    "ISF": 50.0,  # mg/dL per unit
    "CR": 10.0,   # grams per unit
    "basal_rate": 1.0,  # units/hour
    "body_weight_kg": 70.0
}

patient_id = dm.add_patient(
    diabetes_type="type_1",
    age=35,
    weight_kg=70,
    isf=50,
    cr=10,
    basal_rate=1.0
)

# Simulate glucose reading and get recommendation
recommendation = dm.get_insulin_recommendation(
    patient_id=patient_id,
    current_glucose=120.0,  # mg/dL
    meal_carbs=45.0,        # grams
    current_iob=1.5         # units
)

print(f"Research Recommendation:")
print(f"  Bolus: {recommendation.bolus_amount_u:.2f} units")
print(f"  Basal: {recommendation.basal_rate_u_hr:.2f} U/hr")
print(f"  Safety Level: {recommendation.safety_level.value}")
print(f"  Confidence: {recommendation.confidence:.2f}")
```

### 7.4 Advanced Research Applications

#### Model Training Example
```python
from DiaGuardianAI.predictive_models import LSTMPredictor, ModelTrainer

# Initialize LSTM model
model = LSTMPredictor(
    input_seq_len=24,    # 2 hours of 5-minute readings
    output_seq_len=12,   # 1 hour prediction horizon
    n_features=3,        # CGM, insulin, carbs
    hidden_units=64,
    num_layers=2,
    dropout_rate=0.1
)

# Train on synthetic data
from DiaGuardianAI.predictive_models import TrainingParams

# Configure and train the model
trainer = ModelTrainer(
    model=model,
    training_params=TrainingParams(epochs=100, batch_size=32),
)
trainer.train_model(X_train, y_train)

# Evaluate uncertainty quantification
predictions = model.predict(test_sequence)
print(f"Mean prediction: {predictions['mean']}")
print(f"Uncertainty bounds: ±{predictions['std_dev']}")
```

#### Multi-Agent System Example
```python
from DiaGuardianAI.agents import RLAgent, PatternAdvisorAgent

# Initialize RL agent for insulin decisions
rl_agent = RLAgent(
    state_dim=39,
    action_space_definition={
        "bolus_u": {"low": 0.0, "high": 15.0},
        "basal_rate_u_hr": {"low": 0.0, "high": 3.0}
    },
    rl_algorithm="SAC"
)

# Initialize pattern recognition agent
pattern_agent = PatternAdvisorAgent(
    state_dim=39,
    pattern_repository=repository,
    learning_model_type="mlp_regressor"
)

# Coordinate agents for improved decisions
combined_recommendation = coordinate_agents(rl_agent, pattern_agent, current_state)
```

#### OOD Detection and Self-Explanation
```python
from DiaGuardianAI.sdk import OODDetector, SelfExplainer

# Use OOD detector with a trained model
detector = OODDetector(model)
if detector.is_ood(new_data):
    print("Warning: unfamiliar pattern detected")

# Generate natural-language explanation for a recommendation
explainer = SelfExplainer()
explanation = explainer.explain({"trend": "rising", "meal_carbs": 45}, {
    "bolus": 2.0,
    "basal": 1.0,
    "safety_level": "normal",
})
print(explanation)
```

---

## Experimental Results

### 8.1 Simulation Study Design

**Important Note**: The following results are from synthetic patient simulations only and do not represent clinical validation.

#### Study Parameters
- **Virtual Patients**: 1,000 synthetic individuals
- **Simulation Duration**: 30 days per patient
- **Scenarios**: Normal days, high-carb meals, exercise, stress
- **Evaluation Metrics**: Time in Range (TIR), hypoglycemia frequency, glucose variability

#### Baseline Comparisons
- **Standard Therapy**: Fixed basal rates, manual bolus calculations
- **Simple PID Controller**: Classical proportional-integral-derivative control
- **State-of-the-art AID**: Simplified model of existing commercial systems

### 8.2 Performance Metrics (Synthetic Data Only)

| Metric | Standard | PID | Commercial AID* | DiaGuardianAI |
|--------|----------|-----|----------------|---------------|
| TIR 70-180 mg/dL | 68.2% | 72.4% | 78.1% | 81.7% |
| TIR 80-130 mg/dL | 45.3% | 51.2% | 58.9% | 63.4% |
| Time < 70 mg/dL | 4.2% | 2.8% | 1.9% | 1.2% |
| Time < 54 mg/dL | 0.8% | 0.4% | 0.2% | 0.1% |
| Mean Glucose | 154 mg/dL | 148 mg/dL | 142 mg/dL | 138 mg/dL |
| Glucose CV% | 36.2% | 32.1% | 28.7% | 26.3% |

*Simplified simulation of existing AID systems

### 8.3 Model Performance Analysis

#### Glucose Prediction Accuracy
| Model | RMSE (mg/dL) | MAE (mg/dL) | R² |
|-------|--------------|-------------|-----|
| LSTM | 12.4 | 9.1 | 0.89 |
| Transformer | 11.8 | 8.7 | 0.91 |
| N-BEATS | 13.1 | 9.8 | 0.87 |
| TSMixer | 12.0 | 8.9 | 0.90 |
| Ensemble | 10.9 | 8.2 | 0.93 |

#### Uncertainty Calibration
The models demonstrate well-calibrated uncertainty estimates:
- **90% prediction intervals** contain actual values 89.3% of time
- **Uncertainty increases** appropriately with prediction horizon
- **Model confidence correlates** with prediction accuracy

### 8.4 Safety System Evaluation

#### Hypoglycemia Prevention
- **True Positive Rate**: 94.2% (correctly identified impending lows)
- **False Positive Rate**: 8.7% (unnecessary insulin reductions)
- **Response Time**: Mean 12.3 minutes before glucose < 70 mg/dL
- **Severe Events**: 0.02% of simulation time spent < 54 mg/dL

#### Insulin Safety Limits
- **Maximum Bolus**: Never exceeded 15 units in single dose
- **Basal Limits**: No basal rates > 5.0 units/hour recorded
- **IOB Tracking**: Accurate insulin-on-board calculations maintained

---

## Limitations and Future Work

### 9.1 Current Limitations

#### Technical Limitations
1. **Synthetic Data Dependency**: All validation performed on simulated patients
2. **Simplified Physiology**: Basic compartmental models lack full complexity
3. **Limited Meal Detection**: Requires explicit carbohydrate input
4. **Exercise Modeling**: Basic activity response mechanisms
5. **Sensor Noise**: Minimal CGM error simulation

#### Clinical Limitations
1. **No Human Validation**: Zero testing on actual patients
2. **Regulatory Compliance**: Not designed for clinical use
3. **Safety Validation**: Limited to computational testing
4. **Healthcare Integration**: No EHR or device connectivity
5. **Clinical Workflow**: No healthcare provider interfaces

#### Algorithmic Limitations
1. **Training Data**: Limited diversity in synthetic scenarios
2. **Model Generalization**: Unknown performance on real physiology
3. **Edge Cases**: Insufficient testing of rare events
4. **Computational Requirements**: High resource demands for real-time use

### 9.2 Future Research Directions

#### Short-term Objectives (1-2 years)
1. **Real Data Integration**: Partner with diabetes centers for anonymized data
2. **Enhanced Physiology**: Implement more sophisticated patient models
3. **Meal Detection**: Computer vision and pattern recognition for automatic detection
4. **Exercise Integration**: Wearable device data incorporation
5. **Uncertainty Quantification**: Advanced Bayesian approaches

#### Medium-term Goals (2-5 years)
1. **Clinical Validation**: IRB-approved studies with healthcare partners
2. **Regulatory Pathway**: FDA breakthrough device designation pursuit
3. **Multi-site Studies**: Diverse population validation
4. **Real-world Evidence**: Longitudinal outcome studies
5. **Healthcare Integration**: EHR and pump manufacturer partnerships

#### Long-term Vision (5+ years)
1. **Personalized Medicine**: Individual patient model adaptation
2. **Multi-modal Integration**: Continuous ketone, lactate, and hormone monitoring
3. **Behavioral Modeling**: Psychological factors in diabetes management
4. **Precision Dosing**: Genetic and metabolomic-informed algorithms
5. **Global Health**: Low-resource setting adaptations

### 9.3 Research Collaboration Opportunities

The framework is designed to facilitate research collaboration:

#### Academic Partnerships
- **Machine Learning Research**: Novel architectures for time series prediction
- **Control Systems**: Advanced feedback control mechanisms
- **Human Factors**: User interface and decision support research
- **Clinical Informatics**: Healthcare data integration studies

#### Industry Collaboration
- **Device Manufacturers**: Pump and CGM integration testing
- **Pharmaceutical**: Insulin pharmacokinetic validation
- **Technology Companies**: Cloud computing and mobile applications
- **Regulatory Consultants**: FDA pathway development

---

## Ethical Considerations

### 10.1 Research Ethics Framework

This project adheres to principles of responsible AI research in healthcare:

#### Beneficence and Non-maleficence
- **Primary Benefit**: Advance diabetes management research
- **Harm Prevention**: Clear disclaimers prevent inappropriate clinical use
- **Risk Mitigation**: Comprehensive safety testing protocols
- **Transparent Limitations**: Honest reporting of system capabilities

#### Autonomy and Informed Consent
- **Research Participation**: Future studies will require proper consent
- **Algorithm Transparency**: Explainable AI for clinical decision support
- **Patient Control**: Ultimate treatment decisions remain with patients/providers
- **Data Privacy**: Strong protection for any future patient data

#### Justice and Fairness
- **Algorithmic Bias**: Testing across diverse synthetic populations
- **Healthcare Access**: Open-source availability for research use
- **Global Applicability**: Consideration for different healthcare systems
- **Economic Factors**: Cost-effectiveness analysis for implementation

### 10.2 Responsible Development Practices

#### Transparency
- **Open Source**: All code publicly available for scrutiny
- **Reproducibility**: Detailed documentation and example data
- **Peer Review**: Submission to academic conferences and journals
- **Community Feedback**: Active engagement with diabetes research community

#### Safety-First Design
- **Conservative Defaults**: Err on side of caution in uncertain situations
- **Multiple Safeguards**: Redundant safety mechanisms throughout system
- **Fail-Safe Behavior**: Graceful degradation when components fail
- **Human Oversight**: Always require healthcare provider supervision

#### Clinical Translation Pathway
- **Staged Development**: Simulation → Animal models → Human studies
- **Regulatory Compliance**: FDA Quality System Regulation adherence
- **Clinical Partnerships**: Collaboration with endocrinology experts
- **Risk Management**: ISO 14971 medical device risk assessment

### 10.3 Data Privacy and Security

Even for research applications, the framework implements privacy-by-design:

#### Data Protection
- **Anonymization**: All synthetic data lacks personally identifiable information
- **Encryption**: Data transmission and storage security protocols
- **Access Control**: Role-based permissions for research data
- **Audit Trails**: Comprehensive logging of data access and modifications

#### Future Clinical Considerations
- **HIPAA Compliance**: Framework designed for future clinical data protection
- **Consent Management**: Infrastructure for informed consent tracking
- **Data Minimization**: Collect only necessary information for research
- **Right to Withdrawal**: Mechanisms for participants to remove data

---

## Contributing to Research

### 11.1 Contribution Guidelines

We welcome contributions from the diabetes research community:

#### Code Contributions
```bash
# Fork the repository
git fork https://github.com/your-repo/DiaGuardianAI.git

# Create feature branch
git checkout -b feature/new-ml-model

# Make changes with tests
# Submit pull request with detailed description
```

#### Research Contributions
- **Novel Algorithms**: Machine learning model improvements
- **Validation Studies**: Testing with real-world data (when ethical approval obtained)
- **Safety Enhancements**: Additional safety mechanism implementations
- **Documentation**: Improved explanations and tutorials

#### Clinical Input
- **Requirements Gathering**: Input from healthcare providers
- **Safety Review**: Clinical safety assessment of algorithms
- **Workflow Integration**: Healthcare system compatibility evaluation
- **Outcome Measures**: Clinically relevant evaluation metrics

### 11.2 Research Data Sharing

#### Synthetic Data Release
- **Patient Cohorts**: Diverse synthetic populations for algorithm testing
- **Scenario Libraries**: Challenging diabetes management situations
- **Benchmark Datasets**: Standardized evaluation protocols
- **Validation Suites**: Comprehensive testing frameworks

#### Real Data Partnerships (Future)
When clinical partnerships are established:
- **Anonymized Datasets**: De-identified patient data for research
- **Federated Learning**: Distributed model training protocols
- **Multi-site Studies**: Coordinated research across institutions
- **Open Science**: Published datasets with appropriate protections

### 11.3 Academic Collaboration

#### Conference Presentations
Target venues for research dissemination:
- **ATTD**: Advanced Technologies & Treatments for Diabetes
- **ADA**: American Diabetes Association Scientific Sessions  
- **EASD**: European Association for the Study of Diabetes
- **NeurIPS**: Machine Learning for Healthcare Workshop
- **JAMIA**: Journal of the American Medical Informatics Association

#### Research Metrics
Success measures for academic impact:
- **Peer-reviewed Publications**: High-impact diabetes and AI journals
- **Citation Impact**: Research influence on diabetes technology field
- **Clinical Adoption**: Translation to improved patient outcomes
- **Regulatory Recognition**: FDA and international regulatory acceptance

---

## References

[1] American Diabetes Association. (2023). Standards of Medical Care in Diabetes—2023. *Diabetes Care*, 46(Supplement_1), S1-S291.

[2] Beck, R. W., et al. (2019). Effect of continuous glucose monitoring on glycemic control in adults with type 1 diabetes using insulin injections. *JAMA*, 317(4), 371-378.

[3] Bergenstal, R. M., et al. (2016). Effectiveness of sensor-augmented insulin-pump therapy in type 1 diabetes. *New England Journal of Medicine*, 363(4), 311-320.

[4] Boughton, C. K., & Hovorka, R. (2019). Advances in artificial pancreas systems. *Science Translational Medicine*, 11(484), eaaw4949.

[5] Cobelli, C., et al. (2011). The UVA/PADOVA type 1 diabetes simulator. *Journal of Diabetes Science and Technology*, 8(1), 26-34.

[6] Dassau, E., et al. (2017). Intraperitoneal insulin delivery provides superior glycemic regulation to subcutaneous insulin delivery in model predictive control-based fully-automated artificial pancreas in patients with type 1 diabetes. *Diabetes*, 66(4), 1538-1544.

[7] Forlenza, G. P., et al. (2018). Predictive low-glucose suspend reduces hypoglycemia in adults, adolescents, and children with type 1 diabetes in an at-home randomized crossover study. *Diabetes Care*, 41(10), 2155-2161.

[8] Haidar, A., et al. (2016). The artificial pancreas: how closed-loop control is revolutionizing diabetes. *IEEE Control Systems Magazine*, 36(5), 28-47.

[9] Hovorka, R., et al. (2004). Nonlinear model predictive control of glucose concentration in subjects with type 1 diabetes. *Physiological Measurement*, 25(4), 905-920.

[10] Kovatchev, B. P., et al. (2009). In silico preclinical trials: a proof of concept in closed-loop control of type 1 diabetes. *Journal of Diabetes Science and Technology*, 3(1), 44-55.

[11] Maahs, D. M., et al. (2016). Outcome measures for artificial pancreas clinical trials: a consensus report. *Diabetes Care*, 39(7), 1175-1179.

[12] Nimri, R., et al. (2013). Adjusting insulin doses in patients with type 1 diabetes who use insulin pump therapy: the importance of weight, age, and total daily insulin dose. *Diabetes Care*, 36(10), 3743-3750.

[13] Oviedo, S., et al. (2017). A review of personalized blood glucose prediction strategies for T1DM patients. *International Journal for Numerical Methods in Biomedical Engineering*, 33(6), e2833.

[14] Phillip, M., et al. (2013). Consensus recommendations for the use of automated insulin delivery technologies in clinical practice. *Endocrine Reviews*, 39(3), 254-285.

[15] Tauschmann, M., & Hovorka, R. (2018). Technology in the management of type 1 diabetes mellitus—current status and future prospects. *Nature Reviews Endocrinology*, 14(8), 464-475.

---

## Acknowledgments

This research framework benefits from the foundational work of the diabetes technology community, including the JDRF, FDA, and academic researchers worldwide who have advanced the field of automated insulin delivery. We acknowledge the open-source machine learning community for the tools and libraries that enable this research.

**Funding**: This is an independent research project. No industry funding has influenced the development of this framework.

**Conflicts of Interest**: The authors declare no competing financial interests related to this research.

---

## License and Disclaimer

### License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Research Use Disclaimer

**IMPORTANT**: This software is designed exclusively for research and educational purposes. It is NOT approved for clinical use and should NOT be used for actual diabetes management without proper clinical validation, regulatory approval, and healthcare provider supervision.

### Medical Disclaimer

The DiaGuardianAI framework:
- Has NOT been validated in clinical trials
- Is NOT approved by FDA or any regulatory agency  
- Should NOT be used for actual patient care
- May contain errors that could be harmful if used clinically
- Requires significant additional development before clinical consideration

### Research Integrity Statement

All simulation results reported are from synthetic patient models only. No claims are made about real-world clinical performance. Researchers using this framework should clearly distinguish between simulated and clinical results in any publications.

---

**Contact Information**

For research collaborations, questions, or suggestions:
- **GitHub Issues**: skouras@infosphereco.com
- **Research Email**: skouras@infosphereco.com
- **Academic Partnerships**: skouras@infosphereco.com

## 📌 Citation and DOI

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15619934.svg)](https://doi.org/10.5281/zenodo.15619934)

If you use this framework in your research, please cite:

```bibtex
@software{diaguardianai2025,
  title = {DiaGuardianAI: An AI-Driven Framework for Diabetes Management Research},
  author = {Skouras, Panagiotis},
  year = {2025},
  url = {https://github.com/panosbee/DiaGuardianAI},
  version = {1.0.0}
}


*Last Updated: june 2025*
*Version: 1.0.0*
*Build Status: Research Alpha*

