"""
BERT model for bad word detection.
"""
import os
import torch
import numpy as np
from typing import Dict, List, Any, Union, Tuple
from pathlib import Path
from tqdm import tqdm
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
)
from langdetect import detect, LangDetectException
import sys
import re
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

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
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class BadWordDetector:
    """Bad Word Detection using BERT-based models."""
    
    def __init__(self):
        """Initialize the model."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        if torch.cuda.is_available():
            logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
        
        self.tokenizer = None
        self.classifier = None
        self.token_classifier = None
        self.is_trained = False
        self.model_path = Config.MODEL_PATH
        self.base_model = Config.BASE_MODEL
        self.max_length = Config.MAX_LENGTH
    
    def load_base_model(self):
        """Load the base pretrained model."""
        logger.info(f"Loading base model: {self.base_model}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        
        # Load model for sequence classification
        self.classifier = AutoModelForSequenceClassification.from_pretrained(
            self.base_model,
            num_labels=2,  # Binary classification (toxic/not toxic)
        )
        self.classifier.to(self.device)
        
        # Load token classification model for highlighting specific bad words
        self.token_classifier = AutoModelForTokenClassification.from_pretrained(
            self.base_model,
            num_labels=2,  # Binary classification for each token
        )
        self.token_classifier.to(self.device)
        
        self.is_trained = False
        logger.info("Base model loaded successfully")
        
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
            self.token_classifier.save_pretrained(os.path.join(path, "token_classifier"))
            
            logger.info(f"Model saved successfully to {path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
    
    def detect_language(self, text: str) -> str:
        """
        Detect the language of input text.
        
        Args:
            text: Input text
            
        Returns:
            language_code: Two letter language code
        """
        try:
            return detect(text)
        except LangDetectException:
            # Default to English if language detection fails
            return "en"
            
    def predict(self, text: str, threshold: float = 0.5) -> Dict[str, Any]:
        """
        Predict if a text contains bad words.
        
        Args:
            text: Text to analyze
            threshold: Confidence threshold
        
        Returns:
            Dictionary with results
        """
        if not self.tokenizer or not self.classifier:
            logger.error("Model not initialized")
            raise ValueError("Model not initialized. Call load() or load_base_model() first.")
        
        # Detect language
        language = self.detect_language(text)
        
        # Tokenize input
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=self.max_length)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get sentence classification
        with torch.no_grad():
            sentence_outputs = self.classifier(**inputs)
            
            # Get logits and probabilities
            logits = sentence_outputs.logits
            probabilities = F.softmax(logits, dim=-1)
            toxic_prob = probabilities[0][1].item()
            
        # Check if the text is toxic
        is_toxic = toxic_prob >= threshold
        
        # If toxic, identify specific bad words
        toxic_words = []
        if is_toxic:
            with torch.no_grad():
                token_outputs = self.token_classifier(**inputs)
                token_logits = token_outputs.logits
                token_probs = F.softmax(token_logits, dim=-1)
                
                # Get token predictions
                token_preds = token_probs[:, :, 1].squeeze() # Probability of toxic label
                tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze())
                
                # Process predictions
                word_idx = 0
                current_word = ""
                current_prob = 0.0
                
                for i, (token, prob) in enumerate(zip(tokens, token_preds)):
                    # Skip special tokens
                    if token in [self.tokenizer.cls_token, self.tokenizer.sep_token, self.tokenizer.pad_token]:
                        continue
                        
                    # Check if it's a subword (starts with ##)
                    if token.startswith("##"):
                        current_word += token[2:]  # Remove ## and add to current word
                        current_prob = max(current_prob, prob.item())
                    else:
                        # If we have a complete word and it's toxic, add it
                        if current_word and current_prob >= threshold:
                            toxic_words.append({
                                "word": current_word,
                                "confidence": current_prob,
                                "position": word_idx
                            })
                        
                        # Start new word
                        current_word = token
                        current_prob = prob.item()
                        word_idx += 1
                
                # Add the last word if it's toxic
                if current_word and current_prob >= threshold:
                    toxic_words.append({
                        "word": current_word,
                        "confidence": current_prob,
                        "position": word_idx
                    })
        
        return {
            "original_text": text,
            "is_toxic": is_toxic,
            "confidence": toxic_prob,
            "detected_language": language,
            "toxic_words": toxic_words
        }
    
    def train(self, train_texts, train_labels, dev_texts=None, dev_labels=None, 
              epochs=None, batch_size=None, learning_rate=None):
        """Train the model with progress bars."""
        if epochs is None:
            epochs = Config.EPOCHS
        if batch_size is None:
            batch_size = Config.BATCH_SIZE
        if learning_rate is None:
            learning_rate = Config.LEARNING_RATE

        # Create datasets
        train_dataset = TextDataset(train_texts, train_labels, self.tokenizer, self.max_length)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        if dev_texts is not None and dev_labels is not None:
            dev_dataset = TextDataset(dev_texts, dev_labels, self.tokenizer, self.max_length)
            dev_loader = DataLoader(dev_dataset, batch_size=batch_size)
        
        # Optimizer
        optimizer = torch.optim.AdamW(self.classifier.parameters(), lr=learning_rate)
        
        # Training loop
        logger.info(f"Starting training for {epochs} epochs")
        best_loss = float('inf')
        
        for epoch in range(epochs):
            self.classifier.train()
            total_loss = 0
            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
            
            for batch in progress_bar:
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.classifier(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Update progress bar
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            avg_loss = total_loss / len(train_loader)
            logger.info(f'Epoch {epoch+1}/{epochs} - Average Loss: {avg_loss:.4f}')
            
            # Validation
            if dev_texts is not None and dev_labels is not None:
                val_loss = self.evaluate(dev_loader)
                logger.info(f'Validation Loss: {val_loss:.4f}')
                
                if val_loss < best_loss:
                    best_loss = val_loss
                    self.save()
                    logger.info("Model saved (best validation loss)")
        
        self.is_trained = True
        logger.info("Training completed")
    
    def evaluate(self, data_loader):
        """Evaluate the model on validation data."""
        self.classifier.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc='Evaluating'):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.classifier(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                total_loss += outputs.loss.item()
        
        return total_loss / len(data_loader)