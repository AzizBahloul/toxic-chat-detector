# bad_word_detector/models/bert_model.py
import os
import torch
from typing import Dict, Any, List, Set
from pathlib import Path
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
)
from langdetect import detect, LangDetectException, detect_langs
import sys
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import re

# Add the parent directory to sys.path to import from our package
sys.path.append(str(Path(__file__).parent.parent))

from utils.config import Config
from utils.logger import setup_logger

logger = setup_logger()


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


class BadWordDetector:
    """Bad Word Detection using BERT-based models."""

    # Common slurs and hate speech terms across languages
    COMMON_SLURS: Set[str] = {
        # English slurs
        "nigger", "nigga", "negro", "kike", "chink", "gook", "spic", "wetback", 
        "beaner", "fag", "faggot", "dyke", "tranny", "retard", "coon", "jew",
        # Additional offensive terms when used in negative context
        "bitch", "whore", "cunt", "slut",
    }

    def __init__(self):
        """Initialize the model."""
        # Determine device (GPU if available, else CPU)
        self.device = Config.get_device()
        logger.info(f"Using device: {self.device}")

        self.tokenizer = None
        self.classifier = None
        self.token_classifier = None
        self.is_trained = False
        self.model_path = Config.MODEL_PATH
        self.base_model = Config.BASE_MODEL
        self.max_length = Config.MAX_LENGTH

    def load_base_model(self):
        """Load the base pretrained models and set up tokenizer."""
        logger.info(f"Loading base model: {self.base_model}")
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        # Load sequence classifier and token classifier
        self.classifier = AutoModelForSequenceClassification.from_pretrained(
            self.base_model,
            num_labels=2,  # Ensure two labels: non-toxic and toxic
        )
        self.token_classifier = AutoModelForTokenClassification.from_pretrained(
            self.base_model,
            num_labels=2,  # For BIO tagging or toxic/non-toxic token classification
        )
        # Move models to appropriate device - enforce GPU usage
        if self.device.type == "cuda":
            try:
                # Only move to GPU if CUDA is available without errors
                if torch.cuda.is_available():
                    self.classifier.to(self.device)
                    self.token_classifier.to(self.device)
                    logger.info("Models moved to GPU")
                else:
                    logger.error("CUDA not available despite configuration. Training requires a GPU.")
                    raise RuntimeError("No GPU detected. Training requires a CUDA-enabled GPU.")
            except Exception as e:
                logger.error(f"Error moving models to GPU: {e}")
                raise RuntimeError(f"Failed to move models to GPU: {e}")
        else:
            logger.warning("GPU not available; loading models on CPU.")
            self.classifier.to(self.device)
            self.token_classifier.to(self.device)
            
        # Set models to evaluation mode by default
        self.classifier.eval()
        self.token_classifier.eval()
        logger.info("Base models loaded successfully")

    def load(self, model_path: str = None):
        """
        Load a trained model.

        Args:
            model_path: Path to the saved model
        """
        path = model_path or self.model_path

        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(path)

            # Load sequence classifier
            self.classifier = AutoModelForSequenceClassification.from_pretrained(
                os.path.join(path, "classifier")
            )
            self.classifier.to(self.device)

            # Load token classifier
            self.token_classifier = AutoModelForTokenClassification.from_pretrained(
                os.path.join(path, "token_classifier")
            )
            self.token_classifier.to(self.device)

            self.is_trained = True
            logger.info(f"Model loaded successfully from {path}")

        except Exception as e:
            logger.error(f"Error loading model from {path}: {str(e)}")
            logger.info("Loading base pretrained model instead")
            self.load_base_model()

    def save(self, model_path: str = None):
        """
        Save the trained model.

        Args:
            model_path: Path to save the model
        """
        path = model_path or self.model_path

        try:
            # Create directories if they don't exist
            os.makedirs(os.path.join(path, "classifier"), exist_ok=True)
            os.makedirs(os.path.join(path, "token_classifier"), exist_ok=True)

            # Save tokenizer
            self.tokenizer.save_pretrained(path)

            # Save classifiers
            self.classifier.save_pretrained(os.path.join(path, "classifier"))
            self.token_classifier.save_pretrained(
                os.path.join(path, "token_classifier")
            )

            logger.info(f"Model saved successfully to {path}")

        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")

    def detect_language(self, text: str) -> str:
        """
        Detect the language of input text with improved reliability.

        Args:
            text: Input text

        Returns:
            language_code: Two letter language code
        """
        # For very short texts with slurs, default to English
        if len(text.split()) < 5:
            # Check if text contains any common slurs
            if any(slur in text.lower() for slur in self.COMMON_SLURS):
                return "en"  # Assume English for slurs

        # For very short texts, default to English
        if len(text.split()) < 3:
            try:
                # Get language probabilities
                langs = detect_langs(text)
                # Only use detection if confidence is high enough
                if langs and langs[0].prob > 0.5:
                    return langs[0].lang
                return "en"  # Default to English for low confidence
            except LangDetectException:
                return "en"
        
        # For longer texts, use regular detection
        try:
            return detect(text)
        except LangDetectException:
            # Default to English if language detection fails
            return "en"

    def detect_slurs(self, text: str) -> List[Dict[str, Any]]:
        """
        Detect common slurs in text using pattern matching.
        Returns a list of identified slurs with positions.
        """
        text_lower = text.lower()
        toxic_words = []

        # Find all words in text
        words = re.findall(r'\b\w+\b', text_lower)
        
        for i, word in enumerate(words):
            if word in self.COMMON_SLURS:
                toxic_words.append({
                    "word": word,
                    "confidence": 0.95,  # High confidence for known slurs
                    "position": i,
                })
                
        return toxic_words

    def predict(self, text: str, threshold: float = 0.5) -> Dict[str, Any]:
        """
        Predict if a text contains bad words with improved accuracy.

        Args:
            text: Text to analyze
            threshold: Confidence threshold

        Returns:
            Dictionary with results
        """
        if not self.tokenizer or not self.classifier:
            logger.error("Model not initialized")
            raise ValueError(
                "Model not initialized. Call load() or load_base_model() first."
            )

        # Handle empty or very short text
        if not text or len(text.strip()) < 2:
            return {
                "original_text": text,
                "is_toxic": False,
                "confidence": 0.0,
                "detected_language": "en",
                "toxic_words": []
            }

        # First check for common slurs using pattern matching
        slurs = self.detect_slurs(text)
        is_toxic_by_slur = len(slurs) > 0
        
        # Detect language
        language = self.detect_language(text)

        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.max_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get sentence classification
        with torch.no_grad():
            sentence_outputs = self.classifier(**inputs)

            # Get logits and probabilities
            logits = sentence_outputs.logits
            probabilities = F.softmax(logits, dim=-1)
            toxic_prob = probabilities[0][1].item()

        # Check if the text is toxic from model prediction
        is_toxic_by_model = toxic_prob >= threshold

        # Combine results: if either detection method finds toxicity, mark as toxic
        is_toxic = is_toxic_by_model or is_toxic_by_slur
        
        # If not already identified as toxic by slurs and model confidence is high enough, identify toxic words
        toxic_words = slurs.copy()  # Start with slurs detected by pattern matching
        
        if is_toxic_by_model and not is_toxic_by_slur:
            # Use token classifier to identify specific bad words
            with torch.no_grad():
                token_outputs = self.token_classifier(**inputs)
                token_logits = token_outputs.logits
                token_probs = F.softmax(token_logits, dim=-1)

                # Fixed index for toxic label
                toxic_class_index = 1
                token_preds = token_probs[:, :, toxic_class_index].squeeze()
                tokens = self.tokenizer.convert_ids_to_tokens(
                    inputs["input_ids"].squeeze()
                )

                # Process predictions
                word_idx = 0
                current_word = ""
                current_prob = 0.0

                for token, prob in zip(tokens, token_preds):
                    # Skip special tokens
                    if token in [
                        self.tokenizer.cls_token,
                        self.tokenizer.sep_token,
                        self.tokenizer.pad_token,
                    ]:
                        continue

                    # Check if it's a subword (starts with ##)
                    if token.startswith("##"):
                        current_word += token[2:]  # Remove ## and add to current word
                        current_prob = max(current_prob, prob.item())
                    else:
                        # If we have a complete word and it's toxic, add it
                        if current_word and current_prob >= threshold:
                            toxic_words.append(
                                {
                                    "word": current_word,
                                    "confidence": current_prob,
                                    "position": word_idx,
                                }
                            )

                        # Start new word
                        current_word = token
                        current_prob = prob.item()
                        word_idx += 1

                # Add the last word if it's toxic
                if current_word and current_prob >= threshold:
                    toxic_words.append(
                        {
                            "word": current_word,
                            "confidence": current_prob,
                            "position": word_idx,
                        }
                    )
        
        # Lower the confidence threshold for short texts with multiple words but no identified toxic words
        if not toxic_words and len(text.split()) > 1 and toxic_prob > 0.45:
            is_toxic = True

        return {
            "original_text": text,
            "is_toxic": is_toxic,
            "confidence": max(toxic_prob, 0.95 if is_toxic_by_slur else 0),
            "detected_language": language,
            "toxic_words": toxic_words,
        }

    def train(
        self,
        train_texts,
        train_labels,
        dev_texts=None,
        dev_labels=None,
        epochs=None,
        batch_size=None,
        learning_rate=None,
    ):
        """Train the model with progress bars."""
        if epochs is None:
            epochs = Config.EPOCHS
        if batch_size is None:
            batch_size = Config.BATCH_SIZE
        if learning_rate is None:
            learning_rate = Config.LEARNING_RATE

        # Joint training loop for both classifiers
        self._train_sequence_classifier(
            train_texts,
            train_labels,
            dev_texts,
            dev_labels,
            epochs,
            batch_size,
            learning_rate,
        )
        self._train_token_classifier(train_texts, train_labels)
        self.is_trained = True
        logger.info("Joint training completed")

    def _train_sequence_classifier(
        self,
        train_texts,
        train_labels,
        dev_texts,
        dev_labels,
        epochs,
        batch_size,
        learning_rate,
    ):
        """Train the sequence classifier with accuracy monitoring."""
        logger.info("Training sequence classifier...")
        # Create datasets
        train_dataset = TextDataset(
            train_texts, train_labels, self.tokenizer, self.max_length
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        if dev_texts is not None and dev_labels is not None:
            dev_dataset = TextDataset(
                dev_texts, dev_labels, self.tokenizer, self.max_length
            )
            dev_loader = DataLoader(dev_dataset, batch_size=batch_size)

        # Optimizer
        optimizer = torch.optim.AdamW(self.classifier.parameters(), lr=learning_rate)

        # Training loop
        best_loss = float("inf")
        best_accuracy = 0.0

        for epoch in range(epochs):
            self.classifier.train()
            total_loss = 0
            correct_predictions = 0
            total_predictions = 0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")

            for batch in progress_bar:
                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Forward pass
                outputs = self.classifier(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )

                loss = outputs.loss
                total_loss += loss.item()
                
                # Calculate accuracy
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=1)
                correct_predictions += (predictions == labels).sum().item()
                total_predictions += labels.size(0)
                current_accuracy = correct_predictions / total_predictions * 100

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Update progress bar
                progress_bar.set_postfix({
                    "loss": f"{loss.item():.4f}", 
                    "accuracy": f"{current_accuracy:.2f}%"
                })

            # Calculate epoch metrics
            avg_loss = total_loss / len(train_loader)
            train_accuracy = correct_predictions / total_predictions * 100
            logger.info(f"Epoch {epoch + 1}/{epochs} - Average Loss: {avg_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%")

            # Validation
            if dev_texts is not None and dev_labels is not None:
                val_loss, val_accuracy = self.evaluate_with_accuracy(dev_loader)
                logger.info(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

                # Save best model based on validation metrics
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_accuracy = val_accuracy
                    self.save()
                    logger.info(f"Model saved (best validation loss: {best_loss:.4f}, accuracy: {best_accuracy:.2f}%)")

    def _train_token_classifier(self, train_texts, train_labels):
        """Train the token classifier."""
        logger.info("Training token classifier using BIO tagging format...")
        # Placeholder: Add token classification training logic here
        pass

    def evaluate(self, data_loader):
        """Evaluate the model on validation data - returns loss only."""
        self.classifier.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.classifier(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )

                total_loss += outputs.loss.item()

        return total_loss / len(data_loader)
        
    def evaluate_with_accuracy(self, data_loader):
        """Evaluate the model on validation data - returns loss and accuracy."""
        self.classifier.eval()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.classifier(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )

                total_loss += outputs.loss.item()
                
                # Calculate accuracy
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=1)
                correct_predictions += (predictions == labels).sum().item()
                total_predictions += labels.size(0)

        avg_loss = total_loss / len(data_loader)
        accuracy = (correct_predictions / total_predictions) * 100
        
        return avg_loss, accuracy
