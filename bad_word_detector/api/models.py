# bad_word_detector/api/models.py
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Union


class TextRequest(BaseModel):
    """Request model for single text analysis."""

    text: str = Field(..., description="The text to analyze")
    threshold: Optional[float] = Field(
        0.5, description="Confidence threshold (0.0 to 1.0)"
    )

    @validator("threshold")
    def validate_threshold(cls, value):
        """Validate threshold is between 0 and 1."""
        if not 0 <= value <= 1:
            raise ValueError("Threshold must be between 0 and 1")
        return value


class BatchTextRequest(BaseModel):
    """Request model for batch text analysis."""

    texts: List[str] = Field(..., description="List of texts to analyze")
    threshold: Optional[float] = Field(
        0.5, description="Confidence threshold (0.0 to 1.0)"
    )

    @validator("threshold")
    def validate_threshold(cls, value):
        """Validate threshold is between 0 and 1."""
        if not 0 <= value <= 1:
            raise ValueError("Threshold must be between 0 and 1")
        return value


class ToxicWord(BaseModel):
    """Model for a detected toxic word."""

    word: str = Field(..., description="The detected toxic word")
    confidence: float = Field(..., description="Confidence score")
    position: Optional[int] = Field(
        None, description="Position in the text (word index)"
    )


class TextResponse(BaseModel):
    """Response model for text analysis."""

    original_text: str = Field(..., description="The input text")
    is_toxic: bool = Field(..., description="Whether the text contains toxic content")
    confidence: float = Field(..., description="Confidence score")
    detected_language: str = Field(..., description="Detected language of the text")
    toxic_words: List[Dict[str, Union[str, float, int]]] = Field(
        default_factory=list,
        description="List of identified toxic words with their confidence scores and positions",
    )
    detection_method: Optional[str] = Field(
        None, description="Method used to detect toxicity (model, pattern, or combined)"
    )


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether the model is loaded")

    class Config:
        """Pydantic model configuration."""
        protected_namespaces = ()  # Allow 'model_loaded' without conflict
