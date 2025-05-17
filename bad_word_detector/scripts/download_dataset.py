#!/usr/bin/env python
"""
Script to download and prepare the multilingual hate speech dataset from Hugging Face.

This script downloads the FrancophonIA/multilingual-hatespeech-dataset from Hugging Face,
which contains labeled toxic/non-toxic text in multiple languages: Arabic, English, Chinese,
French, German, Russian, Turkish, Hindi/Urdu, Korean, Italian, Spanish, Portuguese, and Indonesian.

Usage:
    python download_dataset.py  # Download and save as a single combined file
    python download_dataset.py --split_files  # Save separate files for each language
    python download_dataset.py --dataset "another/dataset" --output_dir "/path/to/output"

The dataset will be downloaded and saved to the data/raw directory by default.
"""
import os
import sys
from pathlib import Path
from datasets import load_dataset
import pandas as pd
import argparse

# Add the parent directory to sys.path to import from our package
sys.path.append(str(Path(__file__).parent.parent))

from utils.config import Config
from utils.logger import setup_logger

logger = setup_logger()

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Download and prepare multilingual hate speech dataset")
    
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="FrancophonIA/multilingual-hatespeech-dataset",
        help="Hugging Face dataset name"
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=str(Path(Config.DATA_DIR) / "raw"),
        help="Directory to save processed data files"
    )
    
    parser.add_argument(
        "--split_files",
        action="store_true",
        help="Whether to save separate files for each language configuration"
    )
    
    return parser.parse_args()

def download_dataset(dataset_name, output_dir, split_files=False):
    """
    Download and prepare the dataset from Hugging Face.
    
    Args:
        dataset_name: Name of the dataset on Hugging Face
        output_dir: Directory to save processed data files
        split_files: Whether to save separate files for each language
    """
    logger.info(f"Downloading dataset: {dataset_name}")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load available dataset configurations
        dataset_dict = load_dataset(dataset_name)
        
        if split_files:
            # Save each configuration/language as a separate file
            for config_name, dataset in dataset_dict.items():
                # Process each split (train, test, validation)
                for split_name, split_data in dataset.items():
                    # Convert to pandas DataFrame
                    df = pd.DataFrame(split_data)
                    
                    # Save to CSV
                    file_path = output_path / f"{config_name}_{split_name}.csv"
                    df.to_csv(file_path, index=False)
                    logger.info(f"Saved {config_name}/{split_name} with {len(df)} examples to {file_path}")
        else:
            # Combine all data into a single file
            all_data = []
            
            for config_name, dataset in dataset_dict.items():
                for split_name, split_data in dataset.items():
                    # Convert to pandas DataFrame
                    df = pd.DataFrame(split_data)
                    
                    # Add configuration and split information
                    df["dataset_config"] = config_name
                    df["split"] = split_name
                    
                    all_data.append(df)
            
            # Combine and save
            if all_data:
                combined_df = pd.concat(all_data, ignore_index=True)
                file_path = output_path / "multilingual_hatespeech_dataset.csv"
                combined_df.to_csv(file_path, index=False)
                logger.info(f"Saved combined dataset with {len(combined_df)} examples to {file_path}")
            else:
                logger.warning("No data found in the dataset")
                
        logger.info("Dataset preparation completed successfully")
        
    except Exception as e:
        logger.error(f"Error downloading and preparing dataset: {str(e)}")
        raise

def main():
    """Main function."""
    args = parse_args()
    
    # Download and prepare dataset
    download_dataset(
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        split_files=args.split_files
    )
    
    logger.info("Dataset download and preparation completed")

if __name__ == "__main__":
    main()
