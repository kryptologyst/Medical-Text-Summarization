"""
Streamlit demo for medical text summarization.
"""

import logging
import streamlit as st
import torch
from typing import Dict, List, Optional

from src.models import T5MedicalSummarizer, BartMedicalSummarizer
from src.utils import deidentify_text, validate_text_input, get_device
from src.evaluation import SummarizationMetrics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Medical Text Summarization",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .disclaimer {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
        color: #856404;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .summary-box {
        background-color: #e8f4fd;
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Disclaimer
st.markdown("""
<div class="disclaimer">
    <h3>⚠️ IMPORTANT DISCLAIMER</h3>
    <p><strong>This is a research demonstration project and is NOT intended for clinical use.</strong></p>
    <p>The models and outputs should not be used for diagnostic purposes or medical advice.</p>
    <p>Always consult qualified healthcare professionals for medical decisions.</p>
</div>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">🏥 Medical Text Summarization Demo</h1>', unsafe_allow_html=True)

# Sidebar configuration
st.sidebar.header("Configuration")

# Model selection
model_type = st.sidebar.selectbox(
    "Select Model",
    ["T5-Small", "T5-Base", "BART-Base"],
    help="Choose the transformer model for summarization"
)

# Model parameters
st.sidebar.subheader("Generation Parameters")
max_length = st.sidebar.slider(
    "Maximum Summary Length",
    min_value=50,
    max_value=200,
    value=128,
    help="Maximum number of tokens in the generated summary"
)

num_beams = st.sidebar.slider(
    "Number of Beams",
    min_value=1,
    max_value=8,
    value=4,
    help="Number of beams for beam search generation"
)

temperature = st.sidebar.slider(
    "Temperature",
    min_value=0.1,
    max_value=2.0,
    value=1.0,
    step=0.1,
    help="Sampling temperature (1.0 = no sampling)"
)

# De-identification options
st.sidebar.subheader("Privacy Options")
enable_deid = st.sidebar.checkbox(
    "Enable De-identification",
    value=False,
    help="Remove or mask potentially identifying information"
)

aggressive_deid = st.sidebar.checkbox(
    "Aggressive De-identification",
    value=False,
    help="Use more aggressive de-identification patterns"
)

# Load model
@st.cache_resource
def load_model(model_name: str):
    """Load and cache the model."""
    try:
        if "t5" in model_name.lower():
            if "base" in model_name.lower():
                model = T5MedicalSummarizer("t5-base")
            else:
                model = T5MedicalSummarizer("t5-small")
        elif "bart" in model_name.lower():
            model = BartMedicalSummarizer("facebook/bart-base")
        else:
            model = T5MedicalSummarizer("t5-small")
        
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load the selected model
with st.spinner(f"Loading {model_type} model..."):
    model = load_model(model_type)

if model is None:
    st.error("Failed to load model. Please try again.")
    st.stop()

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.header("Input Medical Text")
    
    # Text input options
    input_method = st.radio(
        "Choose input method:",
        ["Type/Paste Text", "Use Example"],
        horizontal=True
    )
    
    if input_method == "Type/Paste Text":
        input_text = st.text_area(
            "Enter medical text to summarize:",
            height=300,
            placeholder="Enter a clinical note, discharge summary, or radiology report...",
            help="Paste your medical text here for summarization"
        )
    else:
        # Example texts
        example_texts = {
            "Discharge Summary": """
            DISCHARGE SUMMARY
            Patient: 72-year-old female
            Admission Date: 2023-01-15
            Discharge Date: 2023-01-20
            Principal Diagnosis: Community-acquired pneumonia
            Secondary Diagnoses: Type 2 diabetes mellitus, hypertension
            Procedures: Chest X-ray, blood cultures, sputum culture
            Hospital Course: Patient was admitted with fever, cough, and shortness of breath. 
            Chest X-ray showed bilateral lower lobe infiltrates. Blood cultures were positive 
            for Streptococcus pneumoniae. Patient was started on IV antibiotics and responded 
            well to treatment. Blood glucose levels were managed with sliding scale insulin.
            Discharge Medications: Amoxicillin-clavulanate 875mg twice daily for 7 days, 
            metformin 1000mg twice daily, lisinopril 10mg daily
            Discharge Instructions: Follow up with primary care physician in 1 week. 
            Complete antibiotic course. Monitor blood glucose levels. Return if symptoms worsen.
            """,
            "Radiology Report": """
            RADIOLOGY REPORT
            Patient: 45-year-old male
            Study: CT Chest with contrast
            Clinical History: Chest pain, shortness of breath
            Technique: Axial CT images of the chest were obtained following IV contrast administration
            Findings: The lungs are clear bilaterally. No evidence of pulmonary embolism. 
            The heart size is normal. No pleural effusions. The mediastinal structures are unremarkable.
            The visualized upper abdomen shows no acute abnormalities.
            Impression: Normal CT chest with contrast. No evidence of pulmonary embolism or 
            other acute cardiopulmonary process.
            Recommendations: Clinical correlation recommended.
            """,
            "Clinical Note": """
            Patient is a 65-year-old male with a history of coronary artery disease, 
            hypertension, and diabetes mellitus type 2. He presents today with chest pain 
            that started 2 hours ago. The pain is described as pressure-like, 7/10 in severity, 
            radiating to the left arm. He denies shortness of breath, nausea, or vomiting. 
            Vital signs: BP 150/90, HR 85, RR 18, O2 sat 98% on room air. Physical exam 
            reveals no acute distress. Heart rate is regular. Lungs are clear to auscultation. 
            EKG shows ST elevation in leads II, III, and aVF. Troponin I is elevated at 2.5 ng/mL. 
            Patient is diagnosed with ST-elevation myocardial infarction and taken to cardiac 
            catheterization lab for primary PCI.
            """
        }
        
        selected_example = st.selectbox(
            "Select example text:",
            list(example_texts.keys())
        )
        input_text = example_texts[selected_example]

with col2:
    st.header("Model Information")
    st.info(f"**Model:** {model_type}")
    st.info(f"**Device:** {get_device()}")
    st.info(f"**Parameters:** {sum(p.numel() for p in model.parameters()):,}")
    
    # Generation settings display
    st.subheader("Current Settings")
    st.write(f"Max Length: {max_length}")
    st.write(f"Beams: {num_beams}")
    st.write(f"Temperature: {temperature}")
    st.write(f"De-identification: {'Yes' if enable_deid else 'No'}")

# Process input
if st.button("Generate Summary", type="primary"):
    if not input_text.strip():
        st.warning("Please enter some text to summarize.")
    else:
        try:
            # Validate input
            validated_text = validate_text_input(input_text)
            
            # De-identify if requested
            if enable_deid:
                processed_text = deidentify_text(validated_text, aggressive=aggressive_deid)
                st.subheader("De-identified Text")
                st.text_area("De-identified input:", processed_text, height=200, disabled=True)
            else:
                processed_text = validated_text
            
            # Generate summary
            with st.spinner("Generating summary..."):
                summary = model.summarize(
                    text=processed_text,
                    max_length=max_length,
                    num_beams=num_beams,
                    prefix="summarize: " if "t5" in model_type.lower() else "",
                )
            
            # Display results
            st.header("Generated Summary")
            st.markdown(f'<div class="summary-box"><p><strong>Summary:</strong><br>{summary}</p></div>', unsafe_allow_html=True)
            
            # Show original text for comparison
            with st.expander("View Original Text"):
                st.text(processed_text)
            
            # Compute basic metrics
            if len(summary.split()) > 0:
                st.subheader("Summary Statistics")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Summary Length", f"{len(summary.split())} words")
                
                with col2:
                    st.metric("Compression Ratio", f"{len(summary.split()) / len(processed_text.split()):.2f}")
                
                with col3:
                    st.metric("Character Count", f"{len(summary)} chars")
            
        except Exception as e:
            st.error(f"Error generating summary: {e}")
            logger.error(f"Summary generation error: {e}")

# Additional features
st.header("Additional Features")

# Batch processing
st.subheader("Batch Processing")
uploaded_file = st.file_uploader(
    "Upload a text file for batch processing",
    type=['txt'],
    help="Upload a text file with multiple medical notes (one per line)"
)

if uploaded_file is not None:
    try:
        content = uploaded_file.read().decode("utf-8")
        texts = [line.strip() for line in content.split('\n') if line.strip()]
        
        st.write(f"Found {len(texts)} texts in the file.")
        
        if st.button("Process All Texts"):
            with st.spinner("Processing batch..."):
                summaries = model.summarize_batch(
                    texts,
                    max_length=max_length,
                    num_beams=num_beams,
                    batch_size=4,
                )
            
            # Display results
            for i, (text, summary) in enumerate(zip(texts, summaries)):
                with st.expander(f"Text {i+1}"):
                    st.write("**Original:**")
                    st.text(text[:200] + "..." if len(text) > 200 else text)
                    st.write("**Summary:**")
                    st.write(summary)
    
    except Exception as e:
        st.error(f"Error processing file: {e}")

# Model comparison
st.subheader("Model Comparison")
if st.button("Compare Models"):
    with st.spinner("Loading models for comparison..."):
        models_to_compare = {
            "T5-Small": T5MedicalSummarizer("t5-small"),
            "T5-Base": T5MedicalSummarizer("t5-base"),
        }
    
    if input_text.strip():
        comparison_results = {}
        
        for model_name, model in models_to_compare.items():
            try:
                summary = model.summarize(
                    text=input_text,
                    max_length=max_length,
                    num_beams=num_beams,
                )
                comparison_results[model_name] = summary
            except Exception as e:
                st.error(f"Error with {model_name}: {e}")
        
        # Display comparison
        for model_name, summary in comparison_results.items():
            st.write(f"**{model_name}:**")
            st.write(summary)
            st.write("---")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.8rem;">
    <p>Medical Text Summarization Demo | Research and Educational Use Only</p>
    <p>Not for clinical use or medical advice</p>
</div>
""", unsafe_allow_html=True)
