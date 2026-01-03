# Medical Text Summarization

A production-ready implementation of medical text summarization using transformer-based models. This project demonstrates state-of-the-art techniques for automatically condensing lengthy clinical notes, discharge summaries, and radiology reports into concise, informative summaries.

## ⚠️ IMPORTANT DISCLAIMER

**This is a research demonstration project and is NOT intended for clinical use.**

The models and outputs should not be used for diagnostic purposes or medical advice. Always consult qualified healthcare professionals for medical decisions. This project is designed for research, education, and demonstration purposes only.

## Features

- **Multiple Model Support**: T5, BART, and other transformer-based models
- **Comprehensive Evaluation**: ROUGE, BERTScore, BLEU, and factuality metrics
- **Privacy Protection**: Built-in de-identification capabilities
- **Interactive Demo**: Streamlit-based web interface
- **Production Ready**: Proper error handling, logging, and configuration management
- **Research Focused**: Designed for reproducibility and educational use

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/kryptologyst/Medical-Text-Summarization.git
cd Medical-Text-Summarization
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the interactive demo:
```bash
streamlit run demo/app.py
```

### Basic Usage

```python
from src.models import T5MedicalSummarizer
from src.utils import set_seed, get_device

# Set random seed for reproducibility
set_seed(42)

# Initialize model
model = T5MedicalSummarizer("t5-small")

# Summarize medical text
clinical_note = """
Patient is a 72-year-old female admitted for worsening dyspnea and fatigue.
Chest X-ray revealed bilateral pulmonary infiltrates.
Started on IV antibiotics and oxygen therapy.
Comorbidities include type 2 diabetes, hypertension, and chronic kidney disease stage 3.
"""

summary = model.summarize(clinical_note)
print(f"Summary: {summary}")
```

## Project Structure

```
medical_text_summarization/
├── src/                    # Source code
│   ├── models/            # Model implementations
│   ├── data/              # Data loading and preprocessing
│   ├── training/          # Training pipeline
│   ├── evaluation/        # Evaluation metrics
│   ├── utils/             # Utility functions
│   └── config/           # Configuration management
├── configs/               # Configuration files
├── data/                  # Data directory
│   ├── raw/              # Raw data
│   └── processed/         # Processed data
├── scripts/               # Training and evaluation scripts
├── notebooks/             # Jupyter notebooks
├── tests/                 # Unit tests
├── assets/                # Generated artifacts
├── demo/                  # Demo applications
└── outputs/               # Training outputs and checkpoints
```

## Training

### Generate Synthetic Data

The project includes a synthetic data generator for demonstration purposes:

```bash
python scripts/train.py --data-size 1000 --model t5-small --epochs 5
```

### Training Options

```bash
python scripts/train.py \
    --model t5-base \
    --epochs 10 \
    --batch-size 16 \
    --learning-rate 3e-4 \
    --data-size 2000 \
    --output-dir outputs/t5_base_experiment
```

### Available Models

- `t5-small`: T5 Small (60M parameters)
- `t5-base`: T5 Base (220M parameters)
- `bart-base`: BART Base (140M parameters)

## Evaluation

The project includes comprehensive evaluation metrics:

- **ROUGE**: ROUGE-1, ROUGE-2, ROUGE-L
- **BERTScore**: Semantic similarity using biomedical BERT
- **BLEU**: N-gram overlap metrics
- **Factuality**: Word overlap and consistency checks
- **Length Metrics**: Compression ratios and length statistics

### Running Evaluation

```python
from src.evaluation import SummarizationMetrics

metrics = SummarizationMetrics()
results = metrics.compute_all_metrics(predictions, references)
print(metrics.format_metrics(results))
```

## Interactive Demo

The Streamlit demo provides an intuitive interface for:

- Text summarization with multiple models
- Real-time parameter adjustment
- De-identification options
- Batch processing capabilities
- Model comparison features

### Launch Demo

```bash
streamlit run demo/app.py
```

## Privacy and De-identification

The project includes built-in de-identification capabilities:

```python
from src.utils import deidentify_text

# Basic de-identification
deidentified = deidentify_text(text, aggressive=False)

# Aggressive de-identification
deidentified = deidentify_text(text, aggressive=True)
```

### Supported Patterns

- Social Security Numbers (SSN)
- Medical Record Numbers (MRN)
- Phone numbers
- Email addresses
- Dates and times (aggressive mode)
- Names (aggressive mode)

## Configuration

The project uses YAML-based configuration management:

```yaml
# configs/default.yaml
model:
  name: t5-small
  pretrained: true
  dropout: 0.1

training:
  epochs: 10
  learning_rate: 5e-4
  batch_size: 8

evaluation:
  metrics: ["rouge", "bertscore", "bleu"]
  rouge_types: ["rouge1", "rouge2", "rougeL"]
```

## Development

### Code Quality

The project follows modern Python development practices:

- Type hints throughout
- Comprehensive docstrings
- Black code formatting
- Ruff linting
- Pre-commit hooks

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black src/ tests/ scripts/
ruff check src/ tests/ scripts/
```

## Performance Considerations

### Device Support

The project automatically detects and uses the best available device:

1. CUDA (NVIDIA GPUs)
2. MPS (Apple Silicon)
3. CPU (fallback)

### Memory Optimization

- Gradient accumulation for large models
- Mixed precision training support
- Efficient data loading with multiple workers

## Limitations and Known Issues

1. **Synthetic Data**: The included synthetic data is for demonstration only
2. **Model Limitations**: Pre-trained models may not be optimized for medical text
3. **Evaluation**: Metrics may not fully capture clinical relevance
4. **Bias**: Models may inherit biases from training data

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure code passes linting
5. Submit a pull request

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Citation

If you use this project in your research, please cite:

```bibtex
@software{medical_text_summarization,
  title={Medical Text Summarization: A Research Demonstration},
  author={Kryptologyst},
  year={2025},
  url={https://github.com/kryptologyst/Medical-Text-Summarization}
}
```

## Acknowledgments

- Hugging Face Transformers library
- Medical NLP community
- Open source contributors

## Support

For questions and support:

- Create an issue on GitHub
- Check the documentation
- Review the example notebooks

---

**Remember: This is a research demonstration project. Not for clinical use.**
# Medical-Text-Summarization
