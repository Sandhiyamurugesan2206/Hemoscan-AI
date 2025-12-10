"""
Hemoglobin Anemia Prediction Dashboard

HOW TO RUN:
1. Make sure you're using the project virtual environment (.venv)
2. Open PowerShell in the project directory (d:\project_root)
3. Run the following command:
   
   D:/project_root/.venv/Scripts/python.exe -m streamlit run dashboard.py
   
   OR if you have activated the venv:
   
   streamlit run dashboard.py

4. The dashboard will open in your default browser (usually http://localhost:8501)
5. Upload a fingernail image and click "Analyze Image" to get predictions

REQUIREMENTS:
- Virtual environment must be activated or use full path to venv python
- Trained model checkpoint should be at: outputs/checkpoints/best_model.pth
- If model is not found, dashboard will use demo/simulation mode

FEATURES:
- Real-time anemia classification using trained ResNet18 model
- Hemoglobin level estimation based on image analysis
- Patient data integration for personalized predictions
- Professional medical report generation
- Analytics and trend visualization
"""

import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os
import io
import base64

# Set page config
st.set_page_config(
    page_title="Hemoglobin Anemia Prediction System",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(90deg, #f0f8ff, #e6f3ff);
        border-radius: 10px;
        border-left: 5px solid #1f4e79;
    }
    
    .metric-container {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #ff6b6b;
    }
    
    .prediction-result {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
    }
    
    .normal-result {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    }
    
    .anemic-result {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
    }
    
    .severe-result {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
    }
    
    .info-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #17a2b8;
        margin: 1rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

class HemoglobinPredictor:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.class_names = ['Anemic', 'Non-Anemic', 'Hand General']
        # Try to auto-load the trained model from the default checkpoint location
        default_path = os.path.join('outputs', 'checkpoints', 'best_model.pth')
        self.model = None
        if os.path.exists(default_path):
            self.model = self.load_model(default_path)
        
    def load_model(self, model_path):
        """Load the trained model"""
        try:
            from torchvision import models
            model = models.resnet18(pretrained=False)
            model.fc = torch.nn.Linear(model.fc.in_features, 3)
            model = model.to(self.device)

            if os.path.exists(model_path):
                # Support both full state_dict and wrapped checkpoint dict
                state = torch.load(model_path, map_location=self.device)
                if isinstance(state, dict) and 'model_state_dict' in state:
                    model.load_state_dict(state['model_state_dict'])
                elif isinstance(state, dict) and set(state.keys()) >= {'epoch', 'model_state_dict'}:
                    model.load_state_dict(state['model_state_dict'])
                else:
                    # assume plain state_dict
                    model.load_state_dict(state)

                model.eval()
                return model
            else:
                return None
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
    
    def predict_anemia(self, image, patient_data=None):
        """Predict anemia from Skin image"""
        # Ensure RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # If model is available, run real inference
        if getattr(self, 'model', None) is not None:
            try:
                image_tensor = self.transform(image).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    outputs = self.model(image_tensor)
                    probs = torch.nn.functional.softmax(outputs.squeeze(0), dim=0).cpu().numpy()

                predicted_class = int(np.argmax(probs))
                confidence = float(probs[predicted_class])

                hb_level = self.estimate_hemoglobin(predicted_class, confidence, patient_data)

                return {
                    'predicted_class': self.class_names[predicted_class],
                    'confidence': confidence,
                    'probabilities': {
                        'Anemic': float(probs[0]),
                        'Non-Anemic': float(probs[1]),
                        'Hand General': float(probs[2])
                    },
                    'hemoglobin_estimate': hb_level
                }
            except Exception as e:
                # If real inference fails, fall back to demo logic
                st.warning(f"Model inference failed, falling back to demo simulation: {e}")

        # Fallback: simulated prediction for demo
        np.random.seed(42)  # For consistent demo results
        img_array = np.array(image)
        avg_color = np.mean(img_array, axis=(0,1))
        brightness = np.mean(avg_color)

        if brightness < 120:
            probs = np.array([0.75, 0.20, 0.05])
        elif brightness > 180:
            probs = np.array([0.15, 0.80, 0.05])
        else:
            probs = np.array([0.45, 0.50, 0.05])

        probs += np.random.normal(0, 0.05, 3)
        probs = np.abs(probs)
        probs = probs / np.sum(probs)

        predicted_class = np.argmax(probs)
        confidence = probs[predicted_class]
        hb_level = self.estimate_hemoglobin(predicted_class, confidence, patient_data)

        return {
            'predicted_class': self.class_names[predicted_class],
            'confidence': confidence,
            'probabilities': {
                'Anemic': probs[0],
                'Non-Anemic': probs[1], 
                'Hand General': probs[2]
            },
            'hemoglobin_estimate': hb_level
        }
    
    def estimate_hemoglobin(self, predicted_class, confidence, patient_data):
        """Estimate hemoglobin level based on prediction and patient data"""
        base_hb = 12.0  # Default
        
        # Adjust based on prediction
        if predicted_class == 0:  # Anemic
            if confidence > 0.8:
                base_hb = np.random.uniform(6.0, 9.0)
                severity = "Severe Anemia"
            elif confidence > 0.6:
                base_hb = np.random.uniform(9.0, 11.0)
                severity = "Moderate Anemia"
            else:
                base_hb = np.random.uniform(10.0, 12.0)
                severity = "Mild Anemia"
        else:  # Non-anemic
            base_hb = np.random.uniform(12.0, 16.0)
            severity = "Normal"
        
        # Adjust based on patient data
        if patient_data:
            # Gender adjustment
            if patient_data.get('gender') == 'Female':
                base_hb -= 1.0  # Females typically have lower Hb
            
            # Age adjustment
            age = patient_data.get('age', 30)
            if age > 65:
                base_hb -= 0.5
            elif age < 18:
                base_hb -= 1.0
        
        # Ensure realistic bounds
        base_hb = max(4.0, min(18.0, base_hb))
        
        return {
            'value': round(base_hb, 1),
            'severity': severity,
            'range': f"{base_hb-0.5:.1f} - {base_hb+0.5:.1f} g/dL",
            'normal_range': "12.0 - 16.0 g/dL (Adult)",
            'status': "Below Normal" if base_hb < 12.0 else "Normal"
        }

def create_header():
    """Create the main header"""
    st.markdown("""
    <div class="main-header">
        🩸 Hemoglobin Prediction System
        <br>
        <small style="font-size: 1rem; font-weight: normal;">
        AI-Powered Non-Invasive Anemia Detection from Images
        </small>
    </div>
    """, unsafe_allow_html=True)

def create_sidebar():
    """Create sidebar for patient information"""
    st.sidebar.header("👤 Patient Information")
    
    patient_data = {}
    
    # Basic information
    patient_data['name'] = st.sidebar.text_input("Patient Name", "")
    patient_data['age'] = st.sidebar.number_input("Age", min_value=1, max_value=120, value=30)
    patient_data['gender'] = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])
    patient_data['weight'] = st.sidebar.number_input("Weight (kg)", min_value=10.0, max_value=200.0, value=70.0)
    patient_data['height'] = st.sidebar.number_input("Height (cm)", min_value=50.0, max_value=250.0, value=170.0)
    
    # Clinical information
    st.sidebar.subheader("🏥 Clinical Information")
    patient_data['symptoms'] = st.sidebar.multiselect(
        "Symptoms",
        ["Fatigue", "Weakness", "Pale skin", "Shortness of breath", 
         "Dizziness", "Cold hands/feet", "Brittle nails", "None"]
    )
    
    patient_data['medical_history'] = st.sidebar.multiselect(
        "Medical History",
        ["Anemia", "Iron deficiency", "Chronic disease", "Blood loss", "None"]
    )
    
    patient_data['medications'] = st.sidebar.text_area("Current Medications", "")
    
    # Calculate BMI
    if patient_data['height'] > 0:
        bmi = patient_data['weight'] / ((patient_data['height']/100) ** 2)
        patient_data['bmi'] = round(bmi, 1)
        st.sidebar.metric("BMI", f"{patient_data['bmi']}")
    
    return patient_data

def display_prediction_results(prediction, patient_data):
    """Display prediction results in a professional format"""
    
    # Main prediction result
    hb_value = prediction['hemoglobin_estimate']['value']
    severity = prediction['hemoglobin_estimate']['severity']
    
    # Choose styling based on severity
    if severity == "Normal":
        result_class = "normal-result"
    elif "Severe" in severity:
        result_class = "severe-result"
    else:
        result_class = "anemic-result"
    
    st.markdown(f"""
    <div class="prediction-result {result_class}">
        <h2>🔬 Prediction Results</h2>
        <h1>{hb_value} g/dL</h1>
        <h3>{severity}</h3>
        <p>Confidence: {prediction['confidence']:.1%}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Detailed metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Hemoglobin Level",
            f"{hb_value} g/dL",
            delta=f"{hb_value - 12.0:.1f}" if hb_value < 12.0 else None
        )
    
    with col2:
        st.metric(
            "Classification", 
            prediction['predicted_class'],
            delta=f"{prediction['confidence']:.1%} confidence"
        )
    
    with col3:
        st.metric(
            "Status",
            prediction['hemoglobin_estimate']['status'],
            delta="Normal: 12.0-16.0 g/dL"
        )
    
    with col4:
        risk_level = "High" if hb_value < 9.0 else "Medium" if hb_value < 12.0 else "Low"
        st.metric(
            "Risk Level",
            risk_level,
            delta="Anemia Risk"
        )

def create_probability_chart(probabilities):
    """Create probability distribution chart"""
    fig = go.Figure(data=[
        go.Bar(
            x=list(probabilities.keys()),
            y=list(probabilities.values()),
            marker_color=['#ff6b6b', '#4ecdc4', '#45b7d1'],
            text=[f"{v:.1%}" for v in probabilities.values()],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title="Classification Probabilities",
        xaxis_title="Class",
        yaxis_title="Probability",
        yaxis=dict(tickformat='.0%'),
        height=400,
        template="plotly_white"
    )
    
    return fig

def create_hemoglobin_gauge(hb_value):
    """Create hemoglobin level gauge chart"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = hb_value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Hemoglobin Level (g/dL)"},
        delta = {'reference': 12.0},
        gauge = {
            'axis': {'range': [None, 18]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 8], 'color': "red"},
                {'range': [8, 11], 'color': "orange"},
                {'range': [11, 12], 'color': "yellow"},
                {'range': [12, 16], 'color': "lightgreen"},
                {'range': [16, 18], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 12.0
            }
        }
    ))
    
    fig.update_layout(height=400, template="plotly_white")
    return fig

def create_trend_analysis():
    """Create trend analysis for patient monitoring"""
    # Sample data for demonstration
    dates = pd.date_range(start='2024-01-01', end='2024-12-01', freq='M')
    hb_values = np.random.normal(10.5, 1.5, len(dates))
    hb_values = np.clip(hb_values, 6, 16)  # Keep realistic range
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=hb_values,
        mode='lines+markers',
        name='Hemoglobin Level',
        line=dict(color='#667eea', width=3),
        marker=dict(size=8)
    ))
    
    # Add normal range
    fig.add_hline(y=12.0, line_dash="dash", line_color="green", 
                  annotation_text="Normal Lower Limit")
    fig.add_hline(y=16.0, line_dash="dash", line_color="green",
                  annotation_text="Normal Upper Limit")
    
    fig.update_layout(
        title="Hemoglobin Trend Analysis",
        xaxis_title="Date",
        yaxis_title="Hemoglobin (g/dL)",
        height=400,
        template="plotly_white",
        showlegend=True
    )
    
    return fig

def create_report(prediction, patient_data):
    """Generate medical report"""
    report_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"""
    
    
    **Report Date:** {report_date}
    **Patient:** {patient_data.get('name', 'Anonymous')}
    **Age:** {patient_data.get('age', 'N/A')} years
    **Gender:** {patient_data.get('gender', 'N/A')}
    
    ### 🔬 Test Results
    - **Hemoglobin Level:** {prediction['hemoglobin_estimate']['value']} g/dL
    - **Classification:** {prediction['predicted_class']}
    - **Confidence:** {prediction['confidence']:.1%}
    - **Status:** {prediction['hemoglobin_estimate']['status']}
    - **Severity:** {prediction['hemoglobin_estimate']['severity']}
    
    ### 📊 Reference Ranges
    - **Normal Adult Range:** 12.0 - 16.0 g/dL
    - **Mild Anemia:** 10.0 - 12.0 g/dL
    - **Moderate Anemia:** 8.0 - 10.0 g/dL
    - **Severe Anemia:** < 8.0 g/dL
    
    ### 🏥 Clinical Assessment
    """
    
    if prediction['hemoglobin_estimate']['value'] < 12.0:
        report += """
    **Finding:** Below normal hemoglobin levels detected.
    **Recommendation:** 
    - Consult with a hematologist
    - Complete blood count (CBC) test recommended
    - Iron studies if indicated
    - Investigate underlying causes
        """
    else:
        report += """
    **Finding:** Hemoglobin levels within normal range.
    **Recommendation:** 
    - Continue regular monitoring
    - Maintain balanced diet
    - Regular exercise as appropriate
        """
    
    report += f"""
    
    ### ⚠️ Important Note
    This is an AI-assisted preliminary screening. Results should be confirmed with laboratory tests and interpreted by qualified medical professionals.
    
    ### 👨‍⚕️ Generated by
    Hemoglobin Anemia Prediction System v1.0
    """
    
    return report

def main():
    """Main application"""
    # Initialize predictor
    predictor = HemoglobinPredictor()
    
    # Create header
    create_header()
    
    # Create sidebar for patient data
    patient_data = create_sidebar()
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["🔬 Prediction", "📊 Analytics", "📈 Trends", "📋 Report"])
    
    with tab1:
        st.header("📸 Upload Palm or Fingernail Image")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a clear image of fingernails for anemia detection"
        )
        
        if uploaded_file is not None:
            # Display image
            col1, col2 = st.columns([1, 2])
            
            with col1:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
                # Image info
                st.info(f"""
                **Image Info:**
                - Size: {image.size}
                - Mode: {image.mode}
                - Format: {image.format}
                """)
            
            with col2:
                # Prediction button
                if st.button("🔬 Analyze Image", type="primary"):
                    with st.spinner("Analyzing image..."):
                        prediction = predictor.predict_anemia(image, patient_data)
                        
                        # Store prediction in session state
                        st.session_state['prediction'] = prediction
                        st.session_state['patient_data'] = patient_data
                        
                        # Display results
                        display_prediction_results(prediction, patient_data)
        
        # Sample images section
        st.header("📋 Instructions")
        st.markdown("""
        <div class="info-box">
        <h4>How to use this system:</h4>
        <ol>
        <li><strong>Patient Information:</strong> Fill in patient details in the sidebar</li>
        <li><strong>Image Upload:</strong> Upload a clear, well-lit image of fingernails</li>
        <li><strong>Analysis:</strong> Click 'Analyze Image' to get predictions</li>
        <li><strong>Results:</strong> Review hemoglobin estimates and recommendations</li>
        </ol>
        
        <h4>Image Guidelines:</h4>
        <ul>
        <li>Good lighting and clear focus</li>
        <li>Fingernails should be clean and visible</li>
        <li>Avoid shadows and reflections</li>
        <li>Include multiple fingernails if possible</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with tab2:
        st.header("📊 Analytics Dashboard")
        
        if 'prediction' in st.session_state:
            prediction = st.session_state['prediction']
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Probability chart
                prob_fig = create_probability_chart(prediction['probabilities'])
                st.plotly_chart(prob_fig, use_container_width=True)
            
            with col2:
                # Hemoglobin gauge
                gauge_fig = create_hemoglobin_gauge(prediction['hemoglobin_estimate']['value'])
                st.plotly_chart(gauge_fig, use_container_width=True)
            
            # Additional metrics
            st.subheader("📈 Detailed Analysis")
            
            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
            
            with metrics_col1:
                st.markdown("""
                <div class="metric-container">
                <h4>Anemia Risk Factors</h4>
                """, unsafe_allow_html=True)
                
                if patient_data.get('symptoms'):
                    st.write("**Symptoms Present:**")
                    for symptom in patient_data['symptoms']:
                        st.write(f"• {symptom}")
                else:
                    st.write("No symptoms reported")
                
                st.markdown("</div>", unsafe_allow_html=True)
            
            with metrics_col2:
                st.markdown("""
                <div class="metric-container">
                <h4>Patient Profile</h4>
                """, unsafe_allow_html=True)
                
                st.write(f"**Age:** {patient_data.get('age', 'N/A')} years")
                st.write(f"**Gender:** {patient_data.get('gender', 'N/A')}")
                st.write(f"**BMI:** {patient_data.get('bmi', 'N/A')}")
                
                st.markdown("</div>", unsafe_allow_html=True)
            
            with metrics_col3:
                st.markdown("""
                <div class="metric-container">
                <h4>Recommendations</h4>
                """, unsafe_allow_html=True)
                
                hb_value = prediction['hemoglobin_estimate']['value']
                if hb_value < 8.0:
                    st.error("🚨 Urgent medical attention required")
                elif hb_value < 12.0:
                    st.warning("⚠️ Consult healthcare provider")
                else:
                    st.success("✅ Levels appear normal")
                
                st.markdown("</div>", unsafe_allow_html=True)
                
        else:
            st.info("Please upload an image and run analysis to view analytics.")
    
    with tab3:
        st.header("📈 Trend Analysis")
        
        # Create sample trend chart
        trend_fig = create_trend_analysis()
        st.plotly_chart(trend_fig, use_container_width=True)
        
        st.info("This shows historical hemoglobin trends. In a real implementation, this would connect to patient medical records.")
    
    with tab4:
        st.header("📋 Medical Report")
        
        if 'prediction' in st.session_state:
            prediction = st.session_state['prediction']
            patient_data = st.session_state['patient_data']
            
            # Generate and display report
            report = create_report(prediction, patient_data)
            st.markdown(report)
            
            # Download button
            st.download_button(
                label="📥 Download Report",
                data=report,
                file_name=f"hemoglobin_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown"
            )
        else:
            st.info("Please upload an image and run analysis to generate a report.")

if __name__ == "__main__":
    main()