#!/usr/bin/env python
# bad_word_detector/scripts/enhanced_training.py
"""
Enhanced training script for toxic comment detection that combines multiple
datasets from Hugging Face and uses the full capabilities of BERT models.
"""

import argparse
import os
from pathlib import Path
import sys
import random
import numpy as np
import torch
from tqdm import tqdm
import time
from datasets import load_dataset, concatenate_datasets, Dataset
import pandas as pd

# Add the project root to sys.path
sys.path.append(str(Path(__file__).parent.parent.parent))

from bad_word_detector.utils.logger import setup_logger
from bad_word_detector.utils.config import Config
from bad_word_detector.models.bert_model import BadWordDetector
from bad_word_detector.models.preprocessing import TextPreprocessor

# Setup logger
logger = setup_logger()

# List of HuggingFace datasets that contain toxic comment data
TOXIC_DATASETS = {
    "primary": {
        "name": "FrancophonIA/multilingual-hatespeech-dataset",
        "config": "Multilingual_train",
        "text_column": "text",
        "label_column": "label",
    },
    "alt1": {
        "name": "jigsaw-toxic-comment-classification-challenge",
        "config": None,
        "text_column": "comment_text",
        "label_mapping": lambda row: 1 if any([row[col] for col in ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]]) else 0,
    },
    "alt2": {
        "name": "hate_speech_offensive",
        "config": None,
        "text_column": "tweet",
        "label_mapping": lambda row: 1 if row["class"] == 0 else 0,  # In this dataset, 0 means hate speech
    },
    "alt3": {
        "name": "hatexplain",
        "config": None,
        "text_column": "text",
        "label_mapping": lambda row: 1 if "hatespeech" in row["annotators"]["label"] else 0,
    },
}

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train toxic comment detection model")
    parser.add_argument(
        "--combine_datasets", 
        action="store_true",
        help="Combine multiple hate speech datasets for training"
    )
    parser.add_argument(
        "--sample_size", 
        type=int,
        default=int(os.getenv("SAMPLE_SIZE", 0)), 
        help="Sample size for training (0 for full dataset)"
    )
    parser.add_argument(
        "--full_bert", 
        action="store_true", 
        help="Use full BERT capabilities (sequence + token classification)"
    )
    parser.add_argument(
        "--epochs", 
        type=int,
        default=int(os.getenv("EPOCHS", 5)),  # Default to 5 epochs if not specified
        help="Number of training epochs"
    )
    return parser.parse_args()

def load_and_combine_datasets(datasets_config, sample_size=0):
    """
    Load and combine multiple datasets from Hugging Face.
    
    Args:
        datasets_config: Dictionary of dataset configurations
        sample_size: Number of examples to sample from each dataset (0 for all)
        
    Returns:
        Combined DataFrame with text and labels
    """
    all_data = []
    
    for dataset_key, config in datasets_config.items():
        try:
            logger.info(f"Loading dataset: {config['name']}")
            dataset = load_dataset(config['name'], config.get('config'))
            
            # Convert to DataFrame for easier manipulation
            if 'train' in dataset:
                df = dataset['train'].to_pandas()
            else:
                # If no 'train' split, use the first available split
                first_split = list(dataset.keys())[0]
                df = dataset[first_split].to_pandas()
            
            # Apply label mapping if provided
            if 'label_mapping' in config:
                df['label'] = df.apply(config['label_mapping'], axis=1)
            elif 'label_column' in config and config['label_column'] != 'label':
                df['label'] = df[config['label_column']]
            
            # Rename text column if needed
            if 'text_column' in config and config['text_column'] != 'text':
                df['text'] = df[config['text_column']]
            
            # Keep only necessary columns
            df = df[['text', 'label']]
            
            # Balance classes if very imbalanced
            toxic_count = df[df['label'] == 1].shape[0]
            non_toxic_count = df[df['label'] == 0].shape[0]
            
            # If one class is more than 3x the other, downsample
            if toxic_count > 3 * non_toxic_count:
                logger.info(f"Downsampling toxic class from {toxic_count} to {non_toxic_count * 3}")
                toxic_df = df[df['label'] == 1].sample(non_toxic_count * 3, replace=False)
                non_toxic_df = df[df['label'] == 0]
                df = pd.concat([toxic_df, non_toxic_df])
            elif non_toxic_count > 3 * toxic_count:
                logger.info(f"Downsampling non-toxic class from {non_toxic_count} to {toxic_count * 3}")
                non_toxic_df = df[df['label'] == 0].sample(toxic_count * 3, replace=False)
                toxic_df = df[df['label'] == 1]
                df = pd.concat([toxic_df, non_toxic_df])
            
            # Sample if requested
            if sample_size > 0 and sample_size < len(df):
                df = df.sample(sample_size, random_state=42)
            
            logger.info(f"Added {len(df)} examples from {config['name']}")
            all_data.append(df)
            
        except Exception as e:
            logger.error(f"Error loading dataset {config['name']}: {str(e)}")
    
    if not all_data:
        raise ValueError("No datasets were successfully loaded")
    
    # Combine all datasets
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Shuffle the data
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    logger.info(f"Combined dataset has {len(combined_df)} examples")
    logger.info(f"Class distribution: {combined_df['label'].value_counts().to_dict()}")
    
    return combined_df

def main():
    """Main function for enhanced training."""
    start_time = time.time()
    args = parse_args()
    
    logger.info("Starting enhanced training for toxic chat detection")
    
    # Initialize text preprocessor
    logger.info("Initializing text preprocessor")
    preprocessor = TextPreprocessor()
    
    # Load datasets
    if args.combine_datasets:
        logger.info("Loading and combining multiple datasets")
        combined_df = load_and_combine_datasets(TOXIC_DATASETS, args.sample_size)
        
        # Clean text
        logger.info("Cleaning text data")
        combined_df['text'] = combined_df['text'].apply(preprocessor.clean_text)
        
        # Split into train and validation
        from sklearn.model_selection import train_test_split
        train_df, val_df = train_test_split(
            combined_df, test_size=0.1, random_state=42, stratify=combined_df['label']
        )
        
        train_texts = train_df['text'].tolist()
        train_labels = train_df['label'].tolist()
        dev_texts = val_df['text'].tolist()
        dev_labels = val_df['label'].tolist()
    else:
        # Use default dataset loading logic
        logger.info(f"Loading default Hugging Face dataset")
        sample_size = args.sample_size if args.sample_size > 0 else None
        train_texts, train_labels, dev_texts, dev_labels = (
            preprocessor.load_huggingface_dataset(
                dataset_name=TOXIC_DATASETS['primary']['name'], 
                clean=True, 
                sample_size=sample_size
            )
        )
    
    logger.info(f"Dataset loaded with {len(train_texts)} training examples and {len(dev_texts)} validation examples")
    
    # Initialize model
    logger.info(f"Initializing model with base model: {Config.BASE_MODEL}")
    model = BadWordDetector()
    model.load_base_model()
    
    # Adjust model settings if full_bert is enabled
    if args.full_bert:
        logger.info("Using full BERT capabilities")
        # Adjust max_length for longer sequences
        model.max_length = max(Config.MAX_LENGTH, 256)  # Ensure at least 256 tokens
    
    # Training with optimized parameters
    logger.info(f"Training for {args.epochs} epochs")
    model.train(
        train_texts=train_texts,
        train_labels=train_labels,
        dev_texts=dev_texts,
        dev_labels=dev_labels,
        epochs=args.epochs,
        batch_size=Config.BATCH_SIZE,
        learning_rate=Config.LEARNING_RATE,
    )
    
    # Save model
    model.save(Config.MODEL_PATH)
    logger.info(f"Model saved to {Config.MODEL_PATH}")
    
    # Training time
    elapsed_time = time.time() - start_time
    logger.info(f"Enhanced training completed in {elapsed_time / 60:.2f} minutes")
    
    return 0

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Enhanced training failed: {str(e)}")
        sys.exit(1)
