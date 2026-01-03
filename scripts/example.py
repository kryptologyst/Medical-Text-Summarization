"""
Example script demonstrating medical text summarization.

This script shows how to use the medical text summarization system
for research and educational purposes.

DISCLAIMER: This is a research demonstration project and is NOT intended for clinical use.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils import set_seed, get_device, deidentify_text
from src.models import T5MedicalSummarizer, BartMedicalSummarizer
from src.data import generate_synthetic_medical_notes
from src.evaluation import SummarizationMetrics


def main():
    """Main example function."""
    print("Medical Text Summarization Example")
    print("=" * 50)
    print("DISCLAIMER: This is for research/educational use only.")
    print("NOT for clinical use or medical advice.")
    print("=" * 50)
    
    # Set random seed for reproducibility
    set_seed(42)
    device = get_device()
    print(f"Using device: {device}")
    
    # Initialize model
    print("\nInitializing T5 model...")
    model = T5MedicalSummarizer("t5-small")
    
    # Example medical texts
    medical_texts = [
        """
        Patient is a 72-year-old female admitted for worsening dyspnea and fatigue.
        Chest X-ray revealed bilateral pulmonary infiltrates.
        Started on IV antibiotics and oxygen therapy.
        Comorbidities include type 2 diabetes, hypertension, and chronic kidney disease stage 3.
        Patient responded well to treatment and was discharged in stable condition.
        """,
        """
        DISCHARGE SUMMARY
        Patient: 65-year-old male
        Admission Date: 2023-01-15
        Discharge Date: 2023-01-20
        Principal Diagnosis: Acute myocardial infarction
        Secondary Diagnoses: Hypertension, hyperlipidemia
        Procedures: Cardiac catheterization, PCI to LAD
        Hospital Course: Patient presented with chest pain and ST elevation.
        Emergent cardiac catheterization showed 90% stenosis of LAD.
        Successful PCI with drug-eluting stent placement.
        Post-procedure course was uncomplicated.
        Discharge Medications: Aspirin, clopidogrel, atorvastatin, metoprolol
        Follow-up: Cardiology clinic in 2 weeks
        """,
        """
        RADIOLOGY REPORT
        Patient: 45-year-old female
        Study: CT Chest with contrast
        Clinical History: Shortness of breath, chest pain
        Findings: Bilateral lower lobe consolidation consistent with pneumonia.
        No evidence of pleural effusion or pneumothorax.
        Heart size is normal. No mediastinal lymphadenopathy.
        Impression: Bilateral lower lobe pneumonia
        Recommendations: Clinical correlation recommended. Consider follow-up imaging.
        """
    ]
    
    # Summarize each text
    print("\nGenerating summaries...")
    summaries = []
    
    for i, text in enumerate(medical_texts):
        print(f"\nText {i+1}:")
        print(f"Original length: {len(text.split())} words")
        
        # Generate summary
        summary = model.summarize(text, max_length=100, num_beams=4)
        summaries.append(summary)
        
        print(f"Summary: {summary}")
        print(f"Summary length: {len(summary.split())} words")
        print(f"Compression ratio: {len(summary.split()) / len(text.split()):.2f}")
    
    # Demonstrate de-identification
    print("\n" + "=" * 50)
    print("De-identification Example")
    print("=" * 50)
    
    text_with_phi = """
    Patient John Smith (DOB: 01/15/1980) was admitted on 12/01/2023 at 14:30.
    Contact: john.smith@email.com, Phone: 555-123-4567
    MRN: AB123456, SSN: 123-45-6789
    Patient has been experiencing chest pain for the past 2 days.
    """
    
    print("Original text:")
    print(text_with_phi)
    
    deidentified = deidentify_text(text_with_phi, aggressive=True)
    print("\nDe-identified text:")
    print(deidentified)
    
    # Generate synthetic data
    print("\n" + "=" * 50)
    print("Synthetic Data Generation")
    print("=" * 50)
    
    print("Generating synthetic medical notes...")
    synthetic_data = generate_synthetic_medical_notes(5)
    
    print(f"Generated {len(synthetic_data)} synthetic medical notes")
    
    # Show examples
    for i, item in enumerate(synthetic_data[:2]):
        print(f"\nSynthetic Note {i+1}:")
        print(f"Text: {item['text'][:200]}...")
        print(f"Summary: {item['summary']}")
        print(f"Metadata: {item['metadata']}")
    
    # Evaluation example
    print("\n" + "=" * 50)
    print("Evaluation Example")
    print("=" * 50)
    
    # Use synthetic data for evaluation
    predictions = [item["summary"] for item in synthetic_data]
    references = [item["summary"] for item in synthetic_data]  # Using same as reference for demo
    
    metrics = SummarizationMetrics()
    evaluation_results = metrics.compute_all_metrics(predictions, references)
    
    print("Evaluation Results:")
    print(metrics.format_metrics(evaluation_results))
    
    # Model comparison
    print("\n" + "=" * 50)
    print("Model Comparison")
    print("=" * 50)
    
    print("Comparing T5 and BART models...")
    
    # Initialize BART model
    bart_model = BartMedicalSummarizer("facebook/bart-base")
    
    test_text = medical_texts[0]
    
    # Generate summaries with both models
    t5_summary = model.summarize(test_text, max_length=100)
    bart_summary = bart_model.summarize(test_text, max_length=100)
    
    print(f"\nOriginal text: {test_text[:100]}...")
    print(f"\nT5 Summary: {t5_summary}")
    print(f"\nBART Summary: {bart_summary}")
    
    print("\n" + "=" * 50)
    print("Example completed successfully!")
    print("Remember: This is for research/educational use only.")
    print("=" * 50)


if __name__ == "__main__":
    main()
