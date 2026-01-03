#!/usr/bin/env python3
"""
Project 479: Medical Text Summarization - Legacy Example

This is the original simple example that has been refactored into a comprehensive
medical text summarization system. This file now serves as a legacy example.

For the full modern implementation, see the src/ directory and other files.

DISCLAIMER: This is a research demonstration project and is NOT intended for clinical use.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils import set_seed, get_device
from src.models import T5MedicalSummarizer

def main():
    """Original simple example using the modern system."""
    print("Medical Text Summarization - Legacy Example")
    print("=" * 50)
    print("DISCLAIMER: This is for research/educational use only.")
    print("NOT for clinical use or medical advice.")
    print("=" * 50)
    
    # Set random seed for reproducibility
    set_seed(42)
    device = get_device()
    print(f"Using device: {device}")
    
    # Initialize model using the modern system
    print("\nLoading T5 model...")
    model = T5MedicalSummarizer("t5-small")
    
    # Original clinical note
    clinical_note = """
    Patient is a 72-year-old female admitted for worsening dyspnea and fatigue.
    Chest X-ray revealed bilateral pulmonary infiltrates.
    Started on IV antibiotics and oxygen therapy. 
    Comorbidities include type 2 diabetes, hypertension, and chronic kidney disease stage 3.
    """
    
    print("\nOriginal Clinical Note:")
    print(clinical_note)
    
    # Generate summary using the modern system
    print("\nGenerating summary...")
    summary = model.summarize(clinical_note, max_length=50, num_beams=4)
    
    # Display summary
    print("\nClinical Summary:")
    print(summary)
    
    print("\n" + "=" * 50)
    print("Legacy example completed!")
    print("For more features, see the full system in src/ directory.")
    print("=" * 50)

if __name__ == "__main__":
    main()