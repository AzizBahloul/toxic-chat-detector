#!/usr/bin/env python
"""
Training script for the bad word detection model.
"""
import os
import sys
import argparse
from pathlib import Path
import torch
import pandas as pd
import time
import psutil
import platform

# Add the parent directory to sys.path to import from our package
sys.path.append(str(Path(__file__).parent.parent))

from models.bert_model import BadWordDetector
from models.preprocessing import TextPreprocessor
from utils.config import Config
from utils.logger import setup_logger

logger = setup_logger()

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train bad word detection model")
    
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default=str(Path(Config.DATA_DIR) / "raw"),
        help="Directory containing training data files"
    )
    
    parser.add_argument(
        "--huggingface", 
        action="store_true",
        help="Use Hugging Face dataset instead of local files"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        default="FrancophonIA/multilingual-hatespeech-dataset",
        help="Hugging Face dataset name to use (only if --huggingface is specified)"
    )
    
    parser.add_argument(
        "--model_dir", 
        type=str, 
        default=Config.MODEL_PATH,
        help="Directory to save the trained model"
    )
    
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=Config.EPOCHS,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=Config.BATCH_SIZE,
        help="Training batch size"
    )
    
    parser.add_argument(
        "--learning_rate", 
        type=float, 
        default=Config.LEARNING_RATE,
        help="Learning rate"
    )
    
    parser.add_argument(
        "--max_length", 
        type=int, 
        default=Config.MAX_LENGTH,
        help="Maximum sequence length"
    )
    
    parser.add_argument(
        "--base_model", 
        type=str, 
        default=Config.BASE_MODEL,
        help="Base pretrained model"
    )
    
    return parser.parse_args()

def display_hardware_info():
    """Display information about available hardware."""
    try:
        import torch
        import psutil
        import platform
        
        logger.info("\n=== Hardware Information ===")
        
        # CPU Info
        logger.info(f"CPU: {platform.processor()}")
        logger.info(f"CPU Cores: {psutil.cpu_count()} ({psutil.cpu_count(logical=False)} physical)")
        
        # Memory Info
        memory = psutil.virtual_memory()
        logger.info(f"Total RAM: {memory.total / (1024**3):.1f} GB")
        logger.info(f"Available RAM: {memory.available / (1024**3):.1f} GB")
        
        # GPU Info
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)} - Memory: {torch.cuda.get_device_properties(i).total_memory / (1024**3):.1f} GB")
            logger.info("GPU available: Yes - Training will use GPU.")
        else:
            logger.info("No GPU detected, using CPU only")
            
    except Exception as e:
        logger.error(f"Error getting hardware info: {str(e)}")

def generate_sample_dataset(output_path, n_samples=1000):
    """
    Generate a sample dataset for testing purposes.
    
    Args:
        output_path: Path to save the dataset
        n_samples: Number of samples to generate
    """
    # Example toxic and non-toxic phrases in multiple languages
    toxic_phrases = {
        "en": ["fuck", "shit", "asshole", "bitch", "dick", "stupid", "idiot", "hate", "kill"],
        "es": ["mierda", "puta", "idiota", "pendejo", "joder", "odio", "estúpido"],
        "fr": ["putain", "merde", "connard", "salope", "stupide", "haine", "crétin"],
        "ar": ["كلب", "حمار", "غبي", "احمق", "لعنة"]
    }
    
    non_toxic_phrases = {
        "en": ["hello", "good", "nice", "happy", "friend", "love", "enjoy", "please", "thanks"],
        "es": ["hola", "bueno", "feliz", "amigo", "amor", "gracias", "por favor"],
        "fr": ["bonjour", "bon", "heureux", "ami", "amour", "merci", "s'il vous plaît"],
        "ar": ["مرحبا", "جيد", "سعيد", "صديق", "حب", "شكرا", "من فضلك"]
    }
    
    # Templates for generating sentences
    templates = {
        "en": [
            "I {sentiment} this {thing}.",
            "This {thing} is {sentiment}.",
            "Why are you so {sentiment}?",
            "You are a {sentiment} person.",
            "{sentiment} off!",
            "I will {sentiment} you.",
            "This is {sentiment} nonsense."
        ],
        "es": [
            "Yo {sentiment} esto {thing}.",
            "Esto {thing} es {sentiment}.",
            "¿Por qué eres tan {sentiment}?",
            "Eres una persona {sentiment}."
        ],
        "fr": [
            "Je {sentiment} ce {thing}.",
            "Ce {thing} est {sentiment}.",
            "Pourquoi es-tu si {sentiment}?",
            "Tu es une personne {sentiment}."
        ],
        "ar": [
            "أنا {sentiment} هذا {thing}",
            "هذا {thing} هو {sentiment}",
            "لماذا أنت {sentiment} جدا؟",
            "أنت شخص {sentiment}"
        ]
    }
    
    # Generate random sentences
    import random
    import numpy as np
    
    texts = []
    labels = []
    languages = list(toxic_phrases.keys())
    
    for _ in range(n_samples):
        # Randomly select language
        lang = random.choice(languages)
        
        # Randomly decide if this will be toxic
        is_toxic = random.random() > 0.5
        
        if is_toxic:
            sentiment = random.choice(toxic_phrases[lang])
            label = 1
        else:
            sentiment = random.choice(non_toxic_phrases[lang])
            label = 0
        
        # Generate text using template
        if lang in templates:
            template = random.choice(templates[lang])
            text = template.format(sentiment=sentiment, thing=random.choice(["product", "idea", "concept", "person"]))
        else:
            # Fallback for languages without templates
            text = f"{sentiment} {random.choice(['product', 'idea', 'concept', 'person'])}"
        
        texts.append(text)
        labels.append(label)
    
    # Create dataframe and save to CSV
    df = pd.DataFrame({
        'text': texts,
        'label': labels
    })
    
    df.to_csv(output_path, index=False)
    logger.info(f"Generated sample dataset with {n_samples} examples at {output_path}")
    
    return output_path

def main():
    """Main training function."""
    args = parse_args()
    
    logger.info("Starting bad word detection model training")
    
    # Display hardware information using logger via display_hardware_info
    display_hardware_info()
    
    # Check if GPU is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")  # Log whether GPU is being used
    
    # Check if training data directory exists
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.warning(f"Training data directory {data_dir} not found. Creating it.")
        data_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize text preprocessor
    logger.info("Initializing text preprocessor")
    preprocessor = TextPreprocessor()
    
    # Choose dataset source based on user preference
    if args.huggingface:
        # Use the Hugging Face dataset
        logger.info(f"Loading multilingual hate speech dataset from Hugging Face: {args.dataset}")
        
        # Load the dataset directly from Hugging Face
        train_texts, train_labels, test_texts, test_labels = preprocessor.load_huggingface_dataset(
            dataset_name=args.dataset
        )
    else:
        # Check if there's data in the directory
        data_files = list(data_dir.glob("*.csv")) + list(data_dir.glob("*.tsv")) + list(data_dir.glob("*.xlsx"))
        if not data_files:
            logger.warning("No training data found in local directory. Switching to Hugging Face dataset.")
            train_texts, train_labels, test_texts, test_labels = preprocessor.load_huggingface_dataset(
                dataset_name="FrancophonIA/multilingual-hatespeech-dataset"
            )
        else:
            # Load and prepare dataset from local files
            logger.info(f"Loading dataset from local directory: {data_dir}")
            train_texts, train_labels, test_texts, test_labels = preprocessor.prepare_multilingual_dataset(
                str(data_dir)
            )
    
    if not train_texts:
        logger.error("No training data available. Please add CSV/TSV files to the data directory.")
        sys.exit(1)
        
    logger.info(f"Dataset loaded: {len(train_texts)} training examples, {len(test_texts)} testing examples")
    
    # Initialize model
    logger.info(f"Initializing model with base model: {args.base_model}")
    model = BadWordDetector()
    model.base_model = args.base_model
    model.max_length = args.max_length
    model.load_base_model()
    
    # Train model
    logger.info("Starting model training")
    start_time = time.time()
    
    training_results = model.train(
        train_texts=train_texts,
        train_labels=train_labels,
        dev_texts=test_texts,
        dev_labels=test_labels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")
    
    # Save model
    logger.info(f"Saving model to {args.model_dir}")
    model.save(args.model_dir)
    
    logger.info("Training process complete")
    
    # Print training results
    if "evaluation" in training_results:
        logger.info("Evaluation results:")
        for metric, value in training_results["evaluation"].items():
            logger.info(f"{metric}: {value}")
    
if __name__ == "__main__":
    main()