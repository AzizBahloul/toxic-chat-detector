# bad_word_detector/api/routes.py
from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Optional, Any
import sys
from pathlib import Path

# Add the parent directory to sys.path to import from our package
sys.path.append(str(Path(__file__).parent.parent))

from models.bert_model import BadWordDetector
from utils.logger import setup_logger
from api.models import TextRequest, TextResponse, HealthResponse

# Setup logger
logger = setup_logger()

router = APIRouter()


# Function to get the model instance
def get_model():
    """
    Get the Bad Word Detection model.
    The model should be initialized in main.py during startup.
    """
    from api.main import app

    if not hasattr(app.state, "model") or app.state.model is None:
        logger.error("Model not loaded")
        raise HTTPException(status_code=500, detail="Model not loaded")
    return app.state.model


@router.get("/", response_model=Dict[str, Any])
async def root():
    """Root endpoint"""
    model = None
    try:
        model = get_model()
    except:
        pass

    return {
        "message": "Bad Word Detection API",
        "status": "online",
        "model_loaded": model is not None,
    }


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    model = None
    try:
        model = get_model()
    except Exception:
        pass

    return {"status": "healthy", "model_loaded": model is not None}


@router.post("/detect", response_model=TextResponse)
async def detect_bad_words(
    request: TextRequest, model: BadWordDetector = Depends(get_model)
):
    """
    Detect bad words in the provided text

    - **text**: The text to analyze
    - **threshold**: Optional confidence threshold (0.0 to 1.0)

    Returns:
    - **original_text**: The input text
    - **is_toxic**: Whether the text contains toxic content
    - **confidence**: Confidence score
    - **detected_language**: Detected language of the text
    - **toxic_words**: List of identified toxic words with their scores
    """
    try:
        # Process the text
        result = model.predict(request.text, threshold=request.threshold)
        return result

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing text: {str(e)}")


@router.post("/detect_batch", response_model=List[TextResponse])
async def detect_bad_words_batch(
    texts: List[str],
    threshold: Optional[float] = 0.5,
    model: BadWordDetector = Depends(get_model),
):
    """
    Batch detect bad words in multiple texts

    - **texts**: List of texts to analyze
    - **threshold**: Optional confidence threshold (0.0 to 1.0)
    """
    try:
        results = [model.predict(text, threshold=threshold) for text in texts]
        return results

    except Exception as e:
        logger.error(f"Error in batch processing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing texts: {str(e)}")
