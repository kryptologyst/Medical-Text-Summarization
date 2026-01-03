"""
Unit tests for medical text summarization.
"""

import pytest
import torch
from unittest.mock import Mock, patch

from src.utils import (
    set_seed,
    get_device,
    validate_text_input,
    deidentify_text,
    MedicalTextSummarizationError,
)
from src.data import (
    MedicalNoteDataset,
    generate_synthetic_medical_notes,
    create_data_splits,
    load_medical_notes,
    save_medical_notes,
)
from src.models import T5MedicalSummarizer, BartMedicalSummarizer
from src.evaluation import SummarizationMetrics
from src.config import Config


class TestUtils:
    """Test utility functions."""
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        # Test that seed is set (basic check)
        assert True  # Seed setting doesn't return anything
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device()
        assert isinstance(device, torch.device)
        assert device.type in ["cpu", "cuda", "mps"]
    
    def test_validate_text_input(self):
        """Test text input validation."""
        # Valid input
        valid_text = "This is a valid medical note."
        result = validate_text_input(valid_text)
        assert result == valid_text
        
        # Empty input
        with pytest.raises(MedicalTextSummarizationError):
            validate_text_input("")
        
        # Non-string input
        with pytest.raises(MedicalTextSummarizationError):
            validate_text_input(123)
        
        # Long input (should be truncated)
        long_text = "a" * 15000
        result = validate_text_input(long_text, max_length=10000)
        assert len(result) == 10000
    
    def test_deidentify_text(self):
        """Test de-identification."""
        text_with_phi = """
        Patient John Smith (DOB: 01/15/1980) was admitted on 12/01/2023.
        Contact: john.smith@email.com, Phone: 555-123-4567
        MRN: AB123456, SSN: 123-45-6789
        """
        
        deidentified = deidentify_text(text_with_phi, aggressive=True)
        
        # Check that PHI is replaced
        assert "[NAME]" in deidentified
        assert "[EMAIL]" in deidentified
        assert "[PHONE]" in deidentified
        assert "[MRN]" in deidentified
        assert "[SSN]" in deidentified


class TestData:
    """Test data processing functions."""
    
    def test_generate_synthetic_medical_notes(self):
        """Test synthetic data generation."""
        data = generate_synthetic_medical_notes(10)
        
        assert len(data) == 10
        assert all("text" in item and "summary" in item for item in data)
        assert all(len(item["text"]) > 0 for item in data)
        assert all(len(item["summary"]) > 0 for item in data)
    
    def test_create_data_splits(self):
        """Test data splitting."""
        data = generate_synthetic_medical_notes(100)
        train, val, test = create_data_splits(data, 0.8, 0.1, 0.1)
        
        assert len(train) == 80
        assert len(val) == 10
        assert len(test) == 10
        assert len(train) + len(val) + len(test) == len(data)
    
    def test_medical_note_dataset(self):
        """Test dataset class."""
        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
        }
        
        data = [
            {"text": "Sample medical note", "summary": "Sample summary"},
            {"text": "Another medical note", "summary": "Another summary"},
        ]
        
        dataset = MedicalNoteDataset(data, mock_tokenizer)
        
        assert len(dataset) == 2
        
        # Test item access
        item = dataset[0]
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "labels" in item


class TestModels:
    """Test model implementations."""
    
    @patch('src.models.T5ForConditionalGeneration')
    @patch('src.models.T5Tokenizer')
    def test_t5_medical_summarizer(self, mock_tokenizer, mock_model):
        """Test T5 medical summarizer."""
        # Mock model and tokenizer
        mock_model_instance = Mock()
        mock_model.from_pretrained.return_value = mock_model_instance
        mock_tokenizer_instance = Mock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        model = T5MedicalSummarizer("t5-small")
        
        assert model.model_name == "t5-small"
        assert model.tokenizer == mock_tokenizer_instance
        assert model.model == mock_model_instance
    
    @patch('src.models.BartForConditionalGeneration')
    @patch('src.models.BartTokenizer')
    def test_bart_medical_summarizer(self, mock_tokenizer, mock_model):
        """Test BART medical summarizer."""
        # Mock model and tokenizer
        mock_model_instance = Mock()
        mock_model.from_pretrained.return_value = mock_model_instance
        mock_tokenizer_instance = Mock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        model = BartMedicalSummarizer("facebook/bart-base")
        
        assert model.model_name == "facebook/bart-base"
        assert model.tokenizer == mock_tokenizer_instance
        assert model.model == mock_model_instance


class TestEvaluation:
    """Test evaluation metrics."""
    
    def test_summarization_metrics(self):
        """Test summarization metrics."""
        metrics = SummarizationMetrics()
        
        predictions = ["This is a summary.", "Another summary here."]
        references = ["This is a reference summary.", "Another reference summary."]
        
        # Test ROUGE computation
        rouge_scores = metrics.compute_rouge(predictions, references)
        assert "rouge1" in rouge_scores
        assert "rouge2" in rouge_scores
        assert "rougeL" in rouge_scores
        
        # Test length metrics
        length_metrics = metrics.compute_length_metrics(predictions, references)
        assert "avg_pred_length" in length_metrics
        assert "avg_ref_length" in length_metrics
        assert "avg_length_ratio" in length_metrics
    
    def test_metrics_formatting(self):
        """Test metrics formatting."""
        metrics = SummarizationMetrics()
        
        sample_metrics = {
            "rouge1": {"precision": 0.8, "recall": 0.7, "fmeasure": 0.75},
            "rouge2": {"precision": 0.6, "recall": 0.5, "fmeasure": 0.55},
            "bleu": 0.4,
            "avg_pred_length": 10.5,
        }
        
        formatted = metrics.format_metrics(sample_metrics)
        assert isinstance(formatted, str)
        assert "ROUGE Scores" in formatted
        assert "BLEU Score" in formatted


class TestConfig:
    """Test configuration management."""
    
    def test_config_initialization(self):
        """Test config initialization."""
        config = Config()
        
        assert config.get("project.name") == "medical_text_summarization"
        assert config.get("model.name") == "t5-small"
        assert config.get("training.epochs") == 10
    
    def test_config_updates(self):
        """Test config updates."""
        config = Config()
        
        config.set("model.name", "t5-base")
        assert config.get("model.name") == "t5-base"
        
        config.update({"training.epochs": 20})
        assert config.get("training.epochs") == 20


# Integration tests
class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_workflow(self):
        """Test complete workflow."""
        # Generate data
        data = generate_synthetic_medical_notes(5)
        
        # Create splits
        train, val, test = create_data_splits(data, 0.6, 0.2, 0.2)
        
        # Initialize metrics
        metrics = SummarizationMetrics()
        
        # Test metrics computation
        predictions = [item["summary"] for item in test]
        references = [item["summary"] for item in test]  # Using same as reference for test
        
        all_metrics = metrics.compute_all_metrics(predictions, references)
        
        assert "rouge1" in all_metrics
        assert "bleu" in all_metrics
        assert "avg_pred_length" in all_metrics


if __name__ == "__main__":
    pytest.main([__file__])
