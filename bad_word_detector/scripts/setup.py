# bad_word_detector/scripts/setup.py
"""
Setup script for the bad word detection project.
This script installs all required dependencies and downloads necessary models.
"""

import sys
import subprocess
from pathlib import Path

# Ensure the project root (bad_word_detector directory) is added to sys.path
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))  # Now modules in utils/ can be imported

from utils.config import Config



def check_gpu():
    """Check if GPU is available for training."""
    try:
        import torch

        if torch.cuda.is_available():
            device = torch.cuda.get_device_name(0)
            print(f"✅ GPU detected: {device}")
            print(f"Number of GPUs available: {torch.cuda.device_count()}")
            return True
        else:
            print("❌ No GPU detected. Using CPU instead.")
            return False
    except ImportError:
        print("❌ PyTorch not installed. Install it with: pip install torch")
        return False


def install_requirements():
    """Install requirements from requirements.txt"""
    req_path = Path(__file__).parent.parent / "requirements.txt"
    print(f"Installing dependencies from {req_path}...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(req_path)])
    print("Dependencies installed successfully.")


def download_nltk_data():
    """Download required NLTK data."""
    try:
        import nltk  # type: ignore

        print("Downloading NLTK data...")
        nltk.download("punkt")
        nltk.download("stopwords")
        print("NLTK data downloaded successfully.")
    except Exception as e:
        print(f"Error downloading NLTK data: {e}")


def download_pretrained_models():
    """Download pretrained models for fine-tuning."""
    try:
        from transformers import (
            AutoTokenizer,
            AutoModelForSequenceClassification,
            AutoModelForTokenClassification,
        )  # type: ignore

        print(f"Downloading pretrained model {Config.BASE_MODEL}...")
        AutoTokenizer.from_pretrained(Config.BASE_MODEL)
        AutoModelForSequenceClassification.from_pretrained(Config.BASE_MODEL)
        AutoModelForTokenClassification.from_pretrained(Config.BASE_MODEL)
        print("Pretrained models downloaded successfully.")
    except Exception:
        print("Could not download models - check internet connection")
        sys.exit(1)


def download_dataset():
    """Download the multilingual hate speech dataset."""
    try:
        from datasets import load_dataset  # type: ignore

        print(
            f"Downloading dataset {Config.DEFAULT_HF_DATASET} for all configured languages..."
        )
        for lang_code in Config.DATASET_LANGUAGES.values():
            print(f"  - Config: {lang_code}")
            load_dataset(Config.DEFAULT_HF_DATASET, lang_code)
        print("Dataset downloaded successfully for all languages.")
    except Exception as e:
        print(f"Error downloading dataset: {e}")


def create_data_directories():
    """Create necessary data directories if they don't exist."""
    base_dir = Path(__file__).parent.parent
    dirs = [
        base_dir / "data" / "raw",
        base_dir / "data" / "processed",
        base_dir / "data" / "models",
        base_dir / "data" / "output",
    ]

    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {dir_path}")


def main():
    """Main setup function."""
    print("Setting up Bad Word Detection System...")

    # Removed automatic installation of requirements.
    # install_requirements()

    # Check for GPU
    check_gpu()

    # Create data directories
    create_data_directories()

    # Download data and models
    download_nltk_data()
    download_pretrained_models()

    # Download the multilingual hate speech dataset
    download_dataset()

    print("\nSetup completed successfully!")
    print("\nNext steps:")
    print("1. Run the training script: python scripts/train.py --huggingface")
    print("2. Evaluate the model: python scripts/evaluate.py")
    print("3. Start the API server: uvicorn api.main:app --reload")


if __name__ == "__main__":
    main()
