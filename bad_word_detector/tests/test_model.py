"""
Tests for the bad word detection model.
"""
import sys
import os
from pathlib import Path
import pytest
import torch

# Add the parent directory to sys.path to import from our package
sys.path.append(str(Path(__file__).parent.parent))

from models.bert_model import BadWordDetector
from models.preprocessing import TextPreprocessor


class TestBadWordDetector:
    """Test cases for BadWordDetector class."""
    
    def setup_method(self):
        """Set up test environment."""
        self.model = BadWordDetector()
        self.model.load_base_model()
    
    def test_model_initialization(self):
        """Test that model initializes correctly."""
        assert self.model.tokenizer is not None
        assert self.model.classifier is not None
        assert self.model.token_classifier is not None
    
    def test_device_configuration(self):
        """Test device configuration."""
        # Ensure model is on the correct device
        if torch.cuda.is_available():
            assert str(self.model.device) == "cuda"
        else:
            assert str(self.model.device) == "cpu"
    
    def test_classification_output_format(self):
        """Test that the model returns predictions in the expected format."""
        test_text = "Hello, this is a test."
        result = self.model.predict(test_text)
        
        assert "original_text" in result
        assert "is_toxic" in result
        assert "confidence" in result
        assert "detected_language" in result
        assert "toxic_words" in result
        
        assert isinstance(result["is_toxic"], bool)
        assert isinstance(result["confidence"], float)
        assert 0 <= result["confidence"] <= 1
    
    def test_toxic_content_detection(self):
        """Test detection of toxic content."""
        # These should be detected as toxic
        toxic_examples = [
            "You are a stupid idiot",
            "Go fuck yourself",
            "I hate you so much",
            "This is bullshit"
        ]
        
        for text in toxic_examples:
            result = self.model.predict(text, threshold=0.5)
            assert result["is_toxic"] is True, f"Failed to detect toxic content in: '{text}'"
    
    def test_non_toxic_content(self):
        """Test detection of non-toxic content."""
        # These should be detected as non-toxic
        non_toxic_examples = [
            "Hello, how are you today?",
            "I love this product",
            "Thank you for your help",
            "The weather is nice today"
        ]
        
        for text in non_toxic_examples:
            result = self.model.predict(text, threshold=0.5)
            assert result["is_toxic"] is False, f"Incorrectly flagged non-toxic content as toxic: '{text}'"
    
    def test_language_detection(self):
        """Test language detection."""
        examples = [
            ("Hello, how are you?", "en"),
            ("Hola, ¿cómo estás?", "es"),
            ("Bonjour, comment allez-vous?", "fr"),
            ("Hallo, wie geht es dir?", "de")
        ]
        
        for text, expected_lang in examples:
            result = self.model.predict(text)
            assert result["detected_language"] == expected_lang, \
                f"Failed to detect language correctly for '{text}': got {result['detected_language']}, expected {expected_lang}"
    
    def test_threshold_effect(self):
        """Test that changing threshold affects classification."""
        text = "This is a borderline example with the word stupid."
        
        # With a low threshold, it might be classified as toxic
        low_result = self.model.predict(text, threshold=0.3)
        
        # With a high threshold, it should be classified as non-toxic
        high_result = self.model.predict(text, threshold=0.9)
        
        # The confidence should be the same in both cases
        assert low_result["confidence"] == high_result["confidence"]
        
        # But the classification might differ
        assert low_result["is_toxic"] != high_result["is_toxic"] or \
               (low_result["is_toxic"] is False and high_result["is_toxic"] is False)


class TestTextPreprocessor:
    """Test cases for TextPreprocessor class."""
    
    def setup_method(self):
        """Set up test environment."""
        self.preprocessor = TextPreprocessor()
    
    def test_text_cleaning(self):
        """Test text cleaning functionality."""
        text = "Hello! This is a <b>test</b> with https://example.com URL and special ch@racters."
        cleaned = self.preprocessor.clean_text(text)
        
        # Check that HTML tags, URLs and special characters are removed
        assert "<b>" not in cleaned
        assert "</b>" not in cleaned
        assert "https://" not in cleaned
        assert "@" not in cleaned
        assert "example.com" not in cleaned
        
        # Check that text is lowercased
        assert cleaned == cleaned.lower()
        
        # Check that multiple spaces are replaced with single spaces
        assert "  " not in cleaned
    
    def test_stopword_removal(self):
        """Test stopword removal."""
        text = "This is a test with some common stopwords like the and of"
        
        # Clean without stopword removal
        cleaned_with_stopwords = self.preprocessor.clean_text(text, remove_stopwords=False)
        
        # Clean with stopword removal
        cleaned_without_stopwords = self.preprocessor.clean_text(text, lang="en", remove_stopwords=True)
        
        # Stopwords like "the" and "of" should be removed
        assert len(cleaned_without_stopwords) < len(cleaned_with_stopwords)
        assert "the" not in cleaned_without_stopwords.split()
        assert "of" not in cleaned_without_stopwords.split()


if __name__ == "__main__":
    pytest.main()