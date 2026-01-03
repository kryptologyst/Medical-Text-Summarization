#!/usr/bin/env python3
"""
Main training script for medical text summarization.

This script demonstrates how to train and evaluate medical text summarization models
using synthetic data. It is designed for research and educational purposes only.

DISCLAIMER: This is a research demonstration project and is NOT intended for clinical use.
The models and outputs should not be used for diagnostic purposes or medical advice.
Always consult qualified healthcare professionals for medical decisions.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils import set_seed, get_device, setup_logging
from src.data import (
    generate_synthetic_medical_notes,
    create_data_splits,
    create_dataloader,
    save_medical_notes,
)
from src.models import T5MedicalSummarizer, BartMedicalSummarizer
from src.training import train_model
from src.evaluation import SummarizationMetrics, evaluate_model
from src.config import Config

logger = logging.getLogger(__name__)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Train medical text summarization models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Data arguments
    parser.add_argument(
        "--data-size",
        type=int,
        default=1000,
        help="Number of synthetic medical notes to generate",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/processed/synthetic_notes.json",
        help="Path to save/load data",
    )
    parser.add_argument(
        "--use-existing-data",
        action="store_true",
        help="Use existing data instead of generating new data",
    )
    
    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        choices=["t5-small", "t5-base", "bart-base"],
        default="t5-small",
        help="Model to train",
    )
    parser.add_argument(
        "--max-input-length",
        type=int,
        default=512,
        help="Maximum input sequence length",
    )
    parser.add_argument(
        "--max-output-length",
        type=int,
        default=128,
        help="Maximum output sequence length",
    )
    
    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Training batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=100,
        help="Number of warmup steps",
    )
    
    # Evaluation arguments
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=200,
        help="Evaluation frequency in steps",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=500,
        help="Save frequency in steps",
    )
    
    # Other arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory for checkpoints and logs",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode",
    )
    parser.add_argument(
        "--deidentify",
        action="store_true",
        help="Enable de-identification",
    )
    
    args = parser.parse_args()
    
    # Setup
    set_seed(args.seed)
    device = get_device()
    setup_logging(
        log_level="DEBUG" if args.debug else "INFO",
        log_file=Path(args.output_dir) / "training.log",
    )
    
    logger.info("Starting medical text summarization training")
    logger.info(f"Arguments: {args}")
    logger.info(f"Device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate or load data
    if args.use_existing_data and Path(args.data_path).exists():
        logger.info(f"Loading existing data from {args.data_path}")
        from src.data import load_medical_notes
        data = load_medical_notes(args.data_path)
    else:
        logger.info(f"Generating {args.data_size} synthetic medical notes")
        data = generate_synthetic_medical_notes(args.data_size)
        
        # Save data
        data_path = Path(args.data_path)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        save_medical_notes(data, data_path)
        logger.info(f"Data saved to {data_path}")
    
    logger.info(f"Loaded {len(data)} medical notes")
    
    # Create data splits
    train_data, val_data, test_data = create_data_splits(
        data,
        train_split=0.8,
        val_split=0.1,
        test_split=0.1,
        seed=args.seed,
    )
    
    logger.info(f"Data splits: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    
    # Initialize model
    if "t5" in args.model:
        if "base" in args.model:
            model = T5MedicalSummarizer("t5-base")
        else:
            model = T5MedicalSummarizer("t5-small")
    elif "bart" in args.model:
        model = BartMedicalSummarizer("facebook/bart-base")
    else:
        raise ValueError(f"Unsupported model: {args.model}")
    
    logger.info(f"Initialized {args.model} model")
    
    # Create data loaders
    from src.data import MedicalNoteDataset
    
    train_dataset = MedicalNoteDataset(
        train_data,
        model.tokenizer,
        max_input_length=args.max_input_length,
        max_output_length=args.max_output_length,
        deidentify=args.deidentify,
    )
    
    val_dataset = MedicalNoteDataset(
        val_data,
        model.tokenizer,
        max_input_length=args.max_input_length,
        max_output_length=args.max_output_length,
        deidentify=args.deidentify,
    )
    
    train_dataloader = create_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
    )
    
    val_dataloader = create_dataloader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
    )
    
    # Training configuration
    config = {
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps,
        "max_grad_norm": 1.0,
        "logging_steps": 50,
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps,
        "max_output_length": args.max_output_length,
        "num_beams": 4,
        "rouge_types": ["rouge1", "rouge2", "rougeL"],
        "bertscore_model": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
    }
    
    # Train model
    logger.info("Starting training...")
    training_history = train_model(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        config=config,
        output_dir=args.output_dir,
    )
    
    logger.info("Training completed!")
    
    # Final evaluation on test set
    logger.info("Running final evaluation on test set...")
    
    test_dataset = MedicalNoteDataset(
        test_data,
        model.tokenizer,
        max_input_length=args.max_input_length,
        max_output_length=args.max_output_length,
        deidentify=args.deidentify,
    )
    
    test_dataloader = create_dataloader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
    )
    
    metrics = SummarizationMetrics(
        rouge_types=config["rouge_types"],
        bertscore_model=config["bertscore_model"],
    )
    
    test_metrics = evaluate_model(
        model,
        test_dataloader,
        metrics,
        device,
        max_length=args.max_output_length,
        num_beams=config["num_beams"],
    )
    
    # Print final results
    logger.info("Final Test Results:")
    metrics_str = metrics.format_metrics(test_metrics)
    logger.info(f"\n{metrics_str}")
    
    # Save final results
    import json
    
    results = {
        "model": args.model,
        "data_size": len(data),
        "test_metrics": test_metrics,
        "training_history": training_history,
        "config": config,
    }
    
    results_path = output_dir / "final_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {results_path}")
    logger.info("Training script completed successfully!")


if __name__ == "__main__":
    main()
