"""
Configuration utility for the bad word detection system.
"""
import os
from pathlib import Path
from dotenv import load_dotenv


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
    
    # API configuration
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", "8000"))
    DEFAULT_THRESHOLD = float(os.getenv("DEFAULT_THRESHOLD", "0.5"))
    
    # Supported languages
    SUPPORTED_LANGUAGES = [
        "en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko", "ar"
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
        "Main": "Multilingual_train"
    }
    
    # Hardware configuration
    USE_GPU = os.getenv("USE_GPU", "true").lower() == "true"
    
    @classmethod
    def get_device(cls):
        """Get the PyTorch device to use."""
        if not cls.USE_GPU:
            return "cpu"
        
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"