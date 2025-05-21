# bad_word_detector/models/preprocessing.py
import re
import nltk
from typing import List, Tuple
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
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)
except Exception:
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
                    self.stopwords[lang] = set(
                        nltk.corpus.stopwords.words(self._get_nltk_language_name(lang))
                    )
                except Exception:
                    self.stopwords[lang] = set()
        except Exception as e:
            # Fallback to empty stopwords if NLTK data isn't available
            logger.warning(
                f"NLTK data not available. Stopword removal will be skipped. Error: {e}"
            )

    def _get_nltk_language_name(self, lang_code: str) -> str:
        """Map ISO language code to NLTK language name."""
        # Mapping from ISO language codes to NLTK language names
        lang_map = {
            "en": "english",
            "es": "spanish",
            "fr": "french",
            "de": "german",
            "it": "italian",
            "pt": "portuguese",
            "ru": "russian",
            "ar": "arabic",
            # Add more mappings as needed
        }
        return lang_map.get(lang_code.lower(), "english")  # Default to English

    def clean_text(
        self, text: str, lang: str = "en", remove_stopwords: bool = False
    ) -> str:
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
        text = re.sub(r"https?://\S+|www\.\S+", "", text)

        # Remove HTML tags
        text = re.sub(r"<.*?>", "", text)

        # Remove special characters except spaces and alphanumerics
        text = re.sub(r"[^\w\s]", " ", text)

        # Replace multiple spaces with a single space
        text = re.sub(r"\s+", " ", text).strip()

        # Remove stopwords if requested
        if remove_stopwords and lang in self.stopwords:
            words = text.split()
            words = [w for w in words if w not in self.stopwords[lang]]
            text = " ".join(words)

        return text

    def load_dataset(
        self,
        file_path: str,
        text_column: str = "text",
        label_column: str = "label",
        test_size: float = 0.2,
        clean: bool = True,
        sample_size: int = None,
    ) -> Tuple[List[str], List[int], List[str], List[int]]:
        """
        Load and prepare dataset from CSV or TSV file.

        Args:
            file_path: Path to the dataset file
            text_column: Name of the column containing text
            label_column: Name of the column containing labels
            test_size: Proportion of the dataset to use for testing
            clean: Whether to clean the text
            sample_size: If specified, sample this many examples randomly

        Returns:
            train_texts, train_labels, test_texts, test_labels
        """
        try:
            # Determine file format from extension
            if file_path.endswith(".csv"):
                df = pd.read_csv(file_path)
            elif file_path.endswith(".tsv"):
                df = pd.read_csv(file_path, sep="\t")
            elif file_path.endswith(".xlsx"):
                df = pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")

            # Check if the required columns exist
            if text_column not in df.columns:
                raise ValueError(
                    f"Text column '{text_column}' not found in the dataset"
                )
            if label_column not in df.columns:
                raise ValueError(
                    f"Label column '{label_column}' not found in the dataset"
                )

            # Apply sampling if enabled
            if sample_size is not None and sample_size > 0 and sample_size < len(df):
                logger.info(
                    f"Sampling {sample_size} examples from {len(df)} total examples"
                )
                df = df.sample(sample_size, random_state=42)

            # Clean text if requested
            if clean:
                logger.info("Cleaning text data...")
                df[text_column] = df[text_column].apply(lambda x: self.clean_text(x))

            # Ensure stratified split
            train_df, test_df = train_test_split(
                df, test_size=test_size, random_state=42, stratify=df[label_column]
            )

            # Extract texts and labels
            train_texts = train_df[text_column].tolist()
            train_labels = train_df[label_column].tolist()
            test_texts = test_df[text_column].tolist()
            test_labels = test_df[label_column].tolist()

            logger.info(
                f"Dataset loaded: {len(train_texts)} training examples, {len(test_texts)} testing examples"
            )

            return train_texts, train_labels, test_texts, test_labels

        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            raise

    def prepare_multilingual_dataset(
        self, data_dir: str, clean: bool = True, sample_size: int = None
    ) -> Tuple[List[str], List[int], List[str], List[int]]:
        """
        Prepare a multilingual dataset by combining data from multiple files in a directory.

        Args:
            data_dir: Directory containing dataset files
            clean: Whether to clean the text
            sample_size: If specified, sample this many examples randomly

        Returns:
            train_texts, train_labels, test_texts, test_labels
        """
        all_train_texts = []
        all_train_labels = []
        all_test_texts = []
        all_test_labels = []

        try:
            # Get all CSV, TSV, and XLSX files in the directory
            files = [
                f for f in os.listdir(data_dir) if f.endswith((".csv", ".tsv", ".xlsx"))
            ]

            if not files:
                logger.warning(f"No dataset files found in {data_dir}")
                return [], [], [], []

            for file in files:
                file_path = os.path.join(data_dir, file)
                logger.info(f"Processing dataset file: {file}")

                # Load dataset from file
                train_texts, train_labels, test_texts, test_labels = self.load_dataset(
                    file_path=file_path, clean=clean, sample_size=sample_size
                )

                # Add to the combined dataset
                all_train_texts.extend(train_texts)
                all_train_labels.extend(train_labels)
                all_test_texts.extend(test_texts)
                all_test_labels.extend(test_labels)

            logger.info(
                f"Combined multilingual dataset: {len(all_train_texts)} training examples, {len(all_test_texts)} testing examples"
            )

            return all_train_texts, all_train_labels, all_test_texts, all_test_labels

        except Exception as e:
            logger.error(f"Error preparing multilingual dataset: {str(e)}")
            raise

    def load_huggingface_dataset(
        self,
        dataset_name: str = "FrancophonIA/multilingual-hatespeech-dataset",
        text_column: str = "text",
        label_column: str = "label",
        test_size: float = 0.2,
        clean: bool = True,
        sample_size: int = None,
    ) -> Tuple[List[str], List[int], List[str], List[int]]:
        """
        Load dataset directly from Hugging Face.

        Args:
            dataset_name: Name of the dataset on Hugging Face
            text_column: Name of the column containing text
            label_column: Name of the column containing labels
            test_size: Proportion of the dataset to use for testing
            clean: Whether to clean the text
            sample_size: If specified, sample this many examples randomly

        Returns:
            train_texts, train_labels, test_texts, test_labels
        """
        logger.info(f"Loading dataset from Hugging Face: {dataset_name}")

        try:
            # Always specify the config name to avoid missing config error
            dataset = load_dataset(dataset_name, "Multilingual_train")
            df = dataset["train"].to_pandas()

            # Apply sampling if enabled
            if sample_size is not None and sample_size > 0 and sample_size < len(df):
                logger.info(
                    f"Sampling {sample_size} examples from {len(df)} total examples"
                )
                df = df.sample(sample_size, random_state=42)

            # Clean text if requested
            if clean:
                logger.info("Cleaning text data...")
                df["text_cleaned"] = df[text_column].apply(self.clean_text)
                text_column = "text_cleaned"

            # Ensure stratified split
            train_df, test_df = train_test_split(
                df, test_size=test_size, random_state=42, stratify=df[label_column]
            )

            train_texts = train_df[text_column].tolist()
            train_labels = train_df[label_column].tolist()
            test_texts = test_df[text_column].tolist()
            test_labels = test_df[label_column].tolist()

            logger.info(
                f"Dataset loaded: {len(train_texts)} training examples, {len(test_texts)} testing examples"
            )

            return train_texts, train_labels, test_texts, test_labels

        except Exception as e:
            logger.error(f"Error loading dataset from Hugging Face: {str(e)}")
            raise
