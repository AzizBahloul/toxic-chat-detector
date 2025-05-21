# bad_word_detector/utils/config.py
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add the necessary path to import logger
sys.path.append(str(Path(__file__).parent))

try:
    from logger import setup_logger
    logger = setup_logger()
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()


class Config:
    """Configuration class for the application."""

    # Base paths
    ROOT_DIR = Path(__file__).parent.parent
    DATA_DIR = ROOT_DIR / "data"
    MODEL_DIR = DATA_DIR / "models"

    # Create directories if they don't exist
    DATA_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)

    # Model configuration
    BASE_MODEL = os.getenv("BASE_MODEL", "bert-base-multilingual-cased")
    MODEL_PATH = str(MODEL_DIR / "bad_word_detector")

    # Training configuration
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))
    MAX_LENGTH = int(os.getenv("MAX_LENGTH", "128"))
    LEARNING_RATE = float(os.getenv("LEARNING_RATE", "2e-5"))
    EPOCHS = int(os.getenv("EPOCHS", "3"))

    # Sample size for faster testing/development (0 = use full dataset)
    SAMPLE_SIZE = 0  # Use all available data for training

    # API configuration
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", "8000"))
    DEFAULT_THRESHOLD = float(os.getenv("DEFAULT_THRESHOLD", "0.5"))

    # Supported languages
    SUPPORTED_LANGUAGES = [
        "en",
        "es",
        "fr",
        "de",
        "it",
        "pt",
        "ru",
        "zh",
        "ja",
        "ko",
        "ar",
    ]

    # Dataset configuration
    DEFAULT_HF_DATASET = "FrancophonIA/multilingual-hatespeech-dataset"
    DATASET_LANGUAGES = {
        "Arabic": "Arabic_test",
        "English": "English_test",
        "Chinese": "Chinese_test",
        "French": "French_test",
        "German": "German_test",
        "Russian": "Russian_test",
        "Turkish": "Turkish_test",
        "Hindi": "Hindi_test",
        "Korean": "Korean_test",
        "Italian": "Italian_test",
        "Spanish": "Spain_test",
        "Portuguese": "Porto_test",
        "Indonesian": "Indonesian_test",
        "Main": "Multilingual_train",
    }

    # Hardware configuration
    USE_GPU = os.getenv("USE_GPU", "true").lower() == "true"

    @classmethod
    def get_device(cls):
        """Properly check and return CUDA device if available."""
        # Determine CPU or GPU device
        try:
            import torch
            if cls.USE_GPU and torch.cuda.is_available():
                # Test simple CUDA operation
                device = torch.device("cuda:0")
                test_tensor = torch.tensor([1.0], device=device)
                if test_tensor.device.type == "cuda":
                    logger.info(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
                    return device
        except Exception as e:
            logger.warning(f"CUDA check failed: {e}. Falling back to CPU.")
        # Fallback to CPU
        logger.info("Using CPU for training.")
        import torch
        return torch.device("cpu")

    @classmethod
    def is_cuda_available(cls):
        """Check if CUDA is available."""
        try:
            import torch

            return torch.cuda.is_available()
        except Exception:
            return False
