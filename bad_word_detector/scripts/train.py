# bad_word_detector/scripts/train.py
# Standard library and environment setup
import sys
from pathlib import Path

# Add the project root to sys.path so we can import our package modules
sys.path.append(str(Path(__file__).parent.parent.parent))

# Load environment variables early
from dotenv import load_dotenv
load_dotenv()

# Clear any CUDA environment overrides
import os as _os
_os.environ.pop("CUDA_VISIBLE_DEVICES", None)
_os.environ.pop("CUDA_DEVICE_ORDER", None)

# Standard library imports
import time
import argparse
import psutil
import platform

# Third-party imports
import torch

# Application imports
from bad_word_detector.utils.logger import setup_logger
from bad_word_detector.utils.config import Config
from bad_word_detector.models.bert_model import BadWordDetector
from bad_word_detector.models.preprocessing import TextPreprocessor

logger = setup_logger()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train bad word detection model")

    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(Path(Config.DATA_DIR) / "raw"),
        help="Directory containing training data files",
    )

    parser.add_argument(
        "--huggingface",
        action="store_true",
        help="Use Hugging Face dataset instead of local files",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default=Config.DEFAULT_HF_DATASET,
        help="Hugging Face dataset name to use (only if --huggingface is specified)",
    )

    parser.add_argument(
        "--model_dir",
        type=str,
        default=Config.MODEL_PATH,
        help="Directory to save the trained model",
    )

    parser.add_argument(
        "--epochs", type=int, default=Config.EPOCHS, help="Number of training epochs"
    )

    parser.add_argument(
        "--batch_size", type=int, default=Config.BATCH_SIZE, help="Training batch size"
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=Config.LEARNING_RATE,
        help="Learning rate",
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=Config.MAX_LENGTH,
        help="Maximum sequence length",
    )

    parser.add_argument(
        "--base_model",
        type=str,
        default=Config.BASE_MODEL,
        help="Base pretrained model",
    )

    parser.add_argument(
        "--sample_size",
        type=int,
        default=Config.SAMPLE_SIZE,
        help="Number of examples to sample (0 for full dataset)",
    )

    return parser.parse_args()


def display_hardware_info():
    """Display information about available hardware."""
    print("\n=== Hardware Information ===")

    # CPU information
    logger.info(f"CPU: {platform.processor() or platform.machine()}")
    physical_cores = psutil.cpu_count(logical=False)
    logical_cores = psutil.cpu_count(logical=True)
    logger.info(f"CPU Cores: {logical_cores} ({physical_cores} physical)")

    # Memory information
    memory = psutil.virtual_memory()
    total_ram = memory.total / (1024**3)  # Convert to GB
    available_ram = memory.available / (1024**3)  # Convert to GB
    logger.info(f"Total RAM: {total_ram:.1f} GB")
    logger.info(f"Available RAM: {available_ram:.1f} GB")

    # GPU information
    try:
        gpu_device = Config.get_device()
        if gpu_device == "cuda" and torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Count: {gpu_count}")
            logger.info(f"CUDA Version: {torch.version.cuda}")
            memory_allocated = torch.cuda.memory_allocated(0) / (
                1024**3
            )  # Convert to GB
            memory_reserved = torch.cuda.memory_reserved(0) / (1024**3)  # Convert to GB
            logger.info(
                f"GPU Memory: Allocated {memory_allocated:.2f} GB / Reserved {memory_reserved:.2f} GB"
            )
        else:
            logger.info("No GPU detected, using CPU only")
    except Exception as e:
        logger.warning(f"Error getting GPU info: {e}")
        logger.info("No GPU detected, using CPU only")

    # PyTorch information
    logger.info(f"Using device: {Config.get_device()}")


def generate_sample_dataset(output_path, n_samples=1000):
    """
    Generate a sample dataset for testing purposes.

    Args:
        output_path: Path to save the dataset
        n_samples: Number of samples to generate
    """
    # Example implementation for generating a synthetic dataset
    pass


def main():
    """Main training function."""
    start_time = time.time()
    args = parse_args()

    logger.info("Starting bad word detection model training")

    # Display hardware information
    display_hardware_info()

    # Initialize text preprocessor
    logger.info("Initializing text preprocessor")
    preprocessor = TextPreprocessor()

    # Load dataset
    if args.huggingface:
        logger.info(
            f"Loading multilingual hate speech dataset from Hugging Face: {args.dataset}"
        )
        # Use sampling if requested
        sample_size = args.sample_size if args.sample_size > 0 else None
        train_texts, train_labels, dev_texts, dev_labels = (
            preprocessor.load_huggingface_dataset(
                dataset_name=args.dataset, clean=True, sample_size=sample_size
            )
        )
    else:
        logger.info(f"Loading data from {args.data_dir}")
        train_texts, train_labels, dev_texts, dev_labels = (
            preprocessor.prepare_multilingual_dataset(
                args.data_dir, clean=True, sample_size=args.sample_size
            )
        )

    # Initialize model
    logger.info(f"Initializing model with base model: {args.base_model}")
    model = BadWordDetector()
    model.load_base_model()

    # Training
    logger.info("Starting model training")
    model.train(
        train_texts=train_texts,
        train_labels=train_labels,
        dev_texts=dev_texts,
        dev_labels=dev_labels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )

    # Save model
    model.save(args.model_dir)
    logger.info(f"Model saved to {args.model_dir}")

    # Training time
    elapsed_time = time.time() - start_time
    logger.info(f"Training completed in {elapsed_time / 60:.2f} minutes")

    return 0


if __name__ == "__main__":
    main()
