"""
Text preprocessing utilities for bad word detection.
"""
import re
import nltk
from typing import List, Dict, Any, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from pathlib import Path
import sys
from datasets import load_dataset

# Add the parent directory to sys.path to import from our package
sys.path.append(str(Path(__file__).parent.parent))

from utils.config import Config
from utils.logger import setup_logger

# Download required NLTK resources
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

logger = setup_logger()

class TextPreprocessor:
    """Text preprocessing for bad word detection."""
    
    def __init__(self):
        """Initialize preprocessor."""
        self.stopwords = {}
        try:
            # Load stopwords for multiple languages
            for lang in Config.SUPPORTED_LANGUAGES:
                try:
                    self.stopwords[lang] = set(nltk.corpus.stopwords.words(self._get_nltk_language_name(lang)))
                except:
                    self.stopwords[lang] = set()
        except:
            # Fallback to empty stopwords if NLTK data isn't available
            logger.warning("NLTK data not available. Stopword removal will be skipped.")
    
    def _get_nltk_language_name(self, lang_code: str) -> str:
        """Map ISO language code to NLTK language name."""
        # Mapping from ISO language codes to NLTK language names
        lang_map = {
            'en': 'english',
            'es': 'spanish',
            'fr': 'french',
            'de': 'german',
            'it': 'italian',
            'pt': 'portuguese',
            'ru': 'russian',
            'ar': 'arabic',
            # Add more mappings as needed
        }
        return lang_map.get(lang_code.lower(), 'english')  # Default to English
    
    def clean_text(self, text: str, lang: str = 'en', remove_stopwords: bool = False) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Input text
            lang: Language code
            remove_stopwords: Whether to remove stopwords
        
        Returns:
            Cleaned text
        """
        if not text:
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove special characters except spaces and alphanumerics
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Replace multiple spaces with a single space
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove stopwords if requested
        if remove_stopwords and lang in self.stopwords:
            words = text.split()
            words = [w for w in words if w not in self.stopwords[lang]]
            text = ' '.join(words)
            
        return text
    
    def load_dataset(self, 
                    file_path: str, 
                    text_column: str = 'text', 
                    label_column: str = 'label', 
                    test_size: float = 0.2, 
                    clean: bool = True) -> Tuple[List[str], List[int], List[str], List[int]]:
        """
        Load and prepare dataset from CSV or TSV file.
        
        Args:
            file_path: Path to the dataset file
            text_column: Name of the column containing text
            label_column: Name of the column containing labels
            test_size: Proportion of the dataset to use for testing
            clean: Whether to clean the text
        
        Returns:
            train_texts, train_labels, test_texts, test_labels
        """
        try:
            # Determine file format from extension
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith('.tsv'):
                df = pd.read_csv(file_path, sep='\t')
            elif file_path.endswith('.xlsx'):
                df = pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
            
            # Check if the required columns exist
            if text_column not in df.columns:
                raise ValueError(f"Text column '{text_column}' not found in the dataset")
            if label_column not in df.columns:
                raise ValueError(f"Label column '{label_column}' not found in the dataset")
            
            # Clean text if requested
            if clean:
                logger.info("Cleaning text data...")
                df[text_column] = df[text_column].apply(lambda x: self.clean_text(x))
            
            # Convert labels to integers if they are not already
            if df[label_column].dtype != 'int64':
                # Map labels like "toxic"/"non-toxic" to 1/0
                unique_labels = df[label_column].unique()
                if len(unique_labels) == 2:
                    label_map = {unique_labels[0]: 0, unique_labels[1]: 1}
                    df[label_column] = df[label_column].map(label_map)
                    
            # Split into train and test sets
            train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
            
            # Extract texts and labels
            train_texts = train_df[text_column].tolist()
            train_labels = train_df[label_column].tolist()
            test_texts = test_df[text_column].tolist()
            test_labels = test_df[label_column].tolist()
            
            logger.info(f"Dataset loaded: {len(train_texts)} training examples, {len(test_texts)} testing examples")
            
            return train_texts, train_labels, test_texts, test_labels
            
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            raise
    
    def prepare_multilingual_dataset(self, data_dir: str, clean: bool = True) -> Tuple[List[str], List[int], List[str], List[int]]:
        """
        Prepare a multilingual dataset by combining data from multiple files in a directory.
        
        Args:
            data_dir: Directory containing dataset files
            clean: Whether to clean the text
            
        Returns:
            train_texts, train_labels, test_texts, test_labels
        """
        all_train_texts = []
        all_train_labels = []
        all_test_texts = []
        all_test_labels = []
        
        try:
            # Get all CSV, TSV, and XLSX files in the directory
            files = [f for f in os.listdir(data_dir) if f.endswith(('.csv', '.tsv', '.xlsx'))]
            
            if not files:
                logger.warning(f"No dataset files found in {data_dir}")
                return [], [], [], []
            
            for file in files:
                file_path = os.path.join(data_dir, file)
                logger.info(f"Processing dataset file: {file}")
                
                # Load dataset from file
                train_texts, train_labels, test_texts, test_labels = self.load_dataset(
                    file_path=file_path,
                    clean=clean
                )
                
                # Add to the combined dataset
                all_train_texts.extend(train_texts)
                all_train_labels.extend(train_labels)
                all_test_texts.extend(test_texts)
                all_test_labels.extend(test_labels)
            
            logger.info(f"Combined multilingual dataset: {len(all_train_texts)} training examples, {len(all_test_texts)} testing examples")
            
            return all_train_texts, all_train_labels, all_test_texts, all_test_labels
            
        except Exception as e:
            logger.error(f"Error preparing multilingual dataset: {str(e)}")
            raise
    
    def load_huggingface_dataset(self, dataset_name: str = "FrancophonIA/multilingual-hatespeech-dataset", 
                               text_column: str = 'text',
                               label_column: str = 'label',
                               test_size: float = 0.2,
                               clean: bool = True) -> Tuple[List[str], List[int], List[str], List[int]]:
        """
        Load dataset directly from Hugging Face.
        
        Args:
            dataset_name: Name of the dataset on Hugging Face
            text_column: Name of the column containing text
            label_column: Name of the column containing labels
            test_size: Proportion of the dataset to use for testing
            clean: Whether to clean the text
            
        Returns:
            train_texts, train_labels, test_texts, test_labels
        """
        logger.info(f"Loading dataset from Hugging Face: {dataset_name}")
        
        try:
            # Load available datasets in the repository
            all_data = []
            languages = []
            
            # The multilingual hatespeech dataset has multiple language configurations
            dataset_configs = load_dataset(dataset_name, split="train").keys()
            
            for config in dataset_configs:
                try:
                    # Try to load this language configuration
                    logger.info(f"Loading language configuration: {config}")
                    dataset = load_dataset(dataset_name, config)
                    
                    # Load all splits available (train, test, validation)
                    for split_name in dataset.keys():
                        split_data = dataset[split_name]
                        
                        # Check if the required columns exist
                        if text_column in split_data.features and label_column in split_data.features:
                            # Convert to pandas DataFrame
                            df = pd.DataFrame(split_data)
                            
                            # Clean text if requested
                            if clean:
                                logger.info(f"Cleaning text data for {config}/{split_name}")
                                df[text_column] = df[text_column].apply(
                                    lambda x: self.clean_text(x) if x is not None else ""
                                )
                            
                            # Add to our collection
                            all_data.append(df)
                            languages.append(config)
                        else:
                            logger.warning(f"Required columns not found in {config}/{split_name}")
                    
                except Exception as e:
                    logger.error(f"Error loading {config}: {str(e)}")
                    continue
            
            if not all_data:
                logger.error("No usable data found in the specified dataset")
                return [], [], [], []
                
            # Combine all DataFrames
            logger.info(f"Combining data from {len(all_data)} configurations")
            combined_df = pd.concat(all_data, ignore_index=True)
            
            # Split into train and test sets
            train_df, test_df = train_test_split(combined_df, test_size=test_size, random_state=42)
            
            # Extract texts and labels
            train_texts = train_df[text_column].tolist()
            train_labels = train_df[label_column].tolist()
            test_texts = test_df[text_column].tolist()
            test_labels = test_df[label_column].tolist()
            
            logger.info(f"Dataset loaded: {len(train_texts)} training examples, {len(test_texts)} testing examples")
            logger.info(f"Languages included: {', '.join(languages)}")
            
            return train_texts, train_labels, test_texts, test_labels
            
        except Exception as e:
            logger.error(f"Error loading dataset from Hugging Face: {str(e)}")
            raise