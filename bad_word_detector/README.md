# Multilingual Bad Word Detection System

A powerful and efficient system for detecting toxic content in messages or live chat across multiple languages, powered by BERT and FastAPI.

## Features

- **Multilingual Support**: Detects toxic content in multiple languages including English, Spanish, French, German, Arabic, and more.
- **High Accuracy**: Uses BERT-based models fine-tuned for toxicity detection.
- **Word-level Detection**: Identifies specific toxic words in the text.
- **RESTful API**: Easy integration with FastAPI.
- **Scalable**: Optimized for both training on high-end machines and running on resource-constrained servers.
- **Customizable**: Adjustable threshold for toxicity detection.
- **Comprehensive Testing**: Includes unit tests and evaluation scripts.

## System Requirements

### Development/Training Environment
- Intel i7-10750H or equivalent
- NVIDIA GTX 1660Ti (6GB VRAM)
- 16GB RAM
- Python 3.8+

### Production Environment
- 6 CPU cores
- 16GB RAM
- 40GB disk space
- Python 3.8+

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd bad_word_detector
```

2. Run the setup script:
```bash
python scripts/setup.py
```

This will:
- Install required dependencies
- Download necessary NLTK data
- Download pre-trained models
- Create required directories

## Project Structure

```
bad_word_detector/
├── api/                    # FastAPI application
│   ├── __init__.py
│   ├── main.py             # Main API entry point
│   ├── models.py           # API data models
│   └── routes.py           # API endpoints
├── data/                   # Data directory
│   ├── models/             # Saved models
│   ├── raw/                # Raw training data
│   ├── processed/          # Preprocessed data
│   └── output/             # Evaluation outputs
├── models/                 # ML models
│   ├── __init__.py
│   ├── bert_model.py       # Bad word detection model
│   └── preprocessing.py    # Text preprocessing utilities
├── scripts/                # Utility scripts
│   ├── evaluate.py         # Model evaluation script
│   ├── setup.py            # Setup script
│   └── train.py            # Model training script
├── tests/                  # Unit tests
│   ├── __init__.py
│   ├── test_api.py         # API tests
│   └── test_model.py       # Model tests
├── utils/                  # Utilities
│   ├── __init__.py
│   ├── config.py           # Configuration settings
│   └── logger.py           # Logging utilities
├── .env                    # Environment variables
├── README.md               # This file
├── requirements.txt        # Python dependencies
└── setup.py                # Package setup
```

## Usage

### Training the Model

You have two options for training data:

#### Option 1: Using the Hugging Face Dataset (Recommended)

The setup script automatically downloads the multilingual hate speech dataset from Hugging Face. To train using this dataset:

```bash
python scripts/train.py --huggingface --epochs 3 --batch_size 16
```

This dataset includes labeled toxic/non-toxic content in multiple languages from the [FrancophonIA/multilingual-hatespeech-dataset](https://huggingface.co/datasets/FrancophonIA/multilingual-hatespeech-dataset).

#### Option 2: Using Your Own Data

1. Prepare your training data:
   - Place your training data files (CSV/TSV) in the `data/raw` directory
   - CSV format: should contain columns `text` and `label` (1 for toxic, 0 for non-toxic)

2. Run the training script:
```bash
python scripts/train.py --epochs 3 --batch_size 16
```

You can also explicitly download and prepare the Hugging Face dataset to your local directory:

```bash
python scripts/download_dataset.py --split_files  # Split by language
```

### Running the API

```bash
cd bad_word_detector
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### API Endpoints

- `GET /`: Root endpoint with API status
- `GET /health`: Health check endpoint
- `POST /detect`: Detect bad words in a single text
- `POST /detect_batch`: Batch detection for multiple texts

#### Example Request (detect endpoint)

```json
{
  "text": "Your text here",
  "threshold": 0.5
}
```

#### Example Response

```json
{
  "original_text": "Your text here",
  "is_toxic": false,
  "confidence": 0.12,
  "detected_language": "en",
  "toxic_words": []
}
```

### Evaluating the Model

```bash
python scripts/evaluate.py --data_path path/to/test_data.csv
```

This will generate evaluation metrics and visualizations in the `data/output` directory.

## Customization

### Environment Variables

The system can be customized using environment variables in the `.env` file:

```
# Model configuration
BASE_MODEL=bert-base-multilingual-cased
BATCH_SIZE=16
MAX_LENGTH=128
LEARNING_RATE=2e-5
EPOCHS=3

# API configuration
API_HOST=0.0.0.0
API_PORT=8000
DEFAULT_THRESHOLD=0.5

# Hardware configuration
USE_GPU=true
```

## Testing

Run the test suite:

```bash
pytest tests/
```

## License

[MIT License](LICENSE)

## Disclaimer

This tool is designed to detect potentially offensive content, but it's not perfect and may produce false positives or miss some toxic content. It should be used as part of a broader content moderation strategy.  