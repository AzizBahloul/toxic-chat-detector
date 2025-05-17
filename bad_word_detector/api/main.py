"""
Main FastAPI application for bad word detection.
"""
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Add the parent directory to sys.path to import from our package
sys.path.append(str(Path(__file__).parent.parent))

from models.bert_model import BadWordDetector
from utils.logger import setup_logger
from api.models import TextRequest, TextResponse

# Setup logger
logger = setup_logger()

# Global model variable that will be initialized in load_model()
model = None

# Initialize FastAPI app
app = FastAPI(
    title="Bad Word Detection API",
    description="API for detecting bad words in text across multiple languages",
    version="1.0.0",
)

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Modify in production to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model
model = None

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    global model
    try:
        # Path to the trained model
        model_path = Path(__file__).parent.parent / "data" / "models" / "bad_word_detector"
        logger.info(f"Loading model from {model_path}")
        
        # Initialize model
        model = BadWordDetector()
        
        # Load trained weights if they exist
        if model_path.exists():
            model.load(str(model_path))
            logger.info("Model loaded successfully")
        else:
            # Use default pretrained model
            logger.warning(f"No trained model found at {model_path}, using base pretrained model")
            model.load_base_model()
            
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        model = None


# Using Pydantic models from api.models


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Bad Word Detection API",
        "status": "online",
        "model_loaded": model is not None
    }


@app.post("/detect", response_model=TextResponse)
async def detect_bad_words(request: TextRequest):
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
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Process the text
        result = model.predict(request.text, threshold=request.threshold)
        return result
    
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing text: {str(e)}")


@app.post("/detect_batch")
async def detect_bad_words_batch(texts: List[str], threshold: Optional[float] = 0.5):
    """
    Batch detect bad words in multiple texts
    
    - **texts**: List of texts to analyze
    - **threshold**: Optional confidence threshold (0.0 to 1.0)
    """
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")
        
    try:
        results = [model.predict(text, threshold=threshold) for text in texts]
        return results
        
    except Exception as e:
        logger.error(f"Error in batch processing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing texts: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)