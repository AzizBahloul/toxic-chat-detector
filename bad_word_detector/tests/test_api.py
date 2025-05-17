"""
Tests for the FastAPI application.
"""
import sys
import os
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

# Add the parent directory to sys.path to import from our package
sys.path.append(str(Path(__file__).parent.parent))

from api.main import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI application."""
    return TestClient(app)


def test_root_endpoint(client):
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()
    assert "status" in response.json()
    assert "model_loaded" in response.json()


def test_health_endpoint(client):
    """Test the health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()
    assert "model_loaded" in response.json()


def test_detect_endpoint_with_non_toxic_text(client):
    """Test detect endpoint with non-toxic text."""
    response = client.post(
        "/detect",
        json={"text": "Hello, how are you today?", "threshold": 0.5}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    assert "original_text" in data
    assert "is_toxic" in data
    assert "confidence" in data
    assert "detected_language" in data
    assert "toxic_words" in data
    
    assert data["is_toxic"] is False
    assert data["detected_language"] == "en"
    assert isinstance(data["confidence"], float)
    assert isinstance(data["toxic_words"], list)


def test_detect_endpoint_with_toxic_text(client):
    """Test detect endpoint with toxic text."""
    response = client.post(
        "/detect",
        json={"text": "You are a stupid idiot", "threshold": 0.5}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    assert "original_text" in data
    assert "is_toxic" in data
    assert "confidence" in data
    assert "detected_language" in data
    assert "toxic_words" in data
    
    assert data["is_toxic"] is True
    assert data["detected_language"] == "en"
    assert isinstance(data["confidence"], float)
    assert isinstance(data["toxic_words"], list)
    assert len(data["toxic_words"]) > 0


def test_detect_endpoint_with_different_languages(client):
    """Test detect endpoint with different languages."""
    languages = {
        "en": "Hello, how are you?",
        "es": "Hola, ¿cómo estás?",
        "fr": "Bonjour, comment allez-vous?",
        "de": "Hallo, wie geht es dir?"
    }
    
    for expected_lang, text in languages.items():
        response = client.post(
            "/detect",
            json={"text": text, "threshold": 0.5}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["detected_language"] == expected_lang


def test_detect_batch_endpoint(client):
    """Test batch detection endpoint."""
    texts = [
        "Hello, how are you?",
        "You are a stupid idiot",
        "Thank you for your help",
        "Go to hell"
    ]
    
    response = client.post(
        "/detect_batch",
        json=texts
    )
    
    assert response.status_code == 200
    data = response.json()
    
    assert isinstance(data, list)
    assert len(data) == len(texts)
    
    # Check that some texts are toxic and some are not
    toxic_count = sum(1 for item in data if item["is_toxic"])
    non_toxic_count = sum(1 for item in data if not item["is_toxic"])
    
    assert toxic_count > 0
    assert non_toxic_count > 0


def test_invalid_input(client):
    """Test API with invalid input."""
    # Empty text
    response = client.post(
        "/detect",
        json={"text": "", "threshold": 0.5}
    )
    
    assert response.status_code == 200
    
    # Invalid threshold
    response = client.post(
        "/detect",
        json={"text": "Hello", "threshold": 2.0}
    )
    
    assert response.status_code == 200
    
    # Very long text
    long_text = "a" * 10000
    response = client.post(
        "/detect",
        json={"text": long_text, "threshold": 0.5}
    )
    
    assert response.status_code == 200