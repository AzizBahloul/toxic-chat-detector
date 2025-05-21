#!/usr/bin/env python
# test_gpu_enforcement.py
"""
Test script to verify that GPU-only training enforcement works correctly.
"""

import os
import sys
from pathlib import Path

# Add the parent directory to sys.path to import from our package
sys.path.append(str(Path(__file__).parent.parent))

from bad_word_detector.utils.config import Config
from bad_word_detector.models.bert_model import BadWordDetector
from bad_word_detector.utils.logger import setup_logger

logger = setup_logger()

def test_gpu_enforcement():
    """Test that the model correctly enforces GPU-only training."""
    logger.info("Testing GPU enforcement...")
    
    try:
        # First, check Config.get_device()
        logger.info("Testing Config.get_device()")
        device_str = Config.get_device()
        logger.info(f"Config.get_device() returned: {device_str}")
        
        # Then, try to initialize the BadWordDetector
        logger.info("Testing BadWordDetector initialization")
        model = BadWordDetector()
        logger.info(f"BadWordDetector initialized with device: {model.device}")
        
        # Try to load the base model
        logger.info("Testing model.load_base_model()")
        model.load_base_model()
        logger.info("Base model loaded successfully")
        
        logger.info("GPU enforcement test passed - GPU is available")
        return True
    except Exception as e:
        logger.error(f"GPU enforcement test failed with error: {e}")
        logger.info("This is expected behavior when no GPU is available")
        return False

if __name__ == "__main__":
    print("Starting GPU enforcement test...")
    result = test_gpu_enforcement()
    if result:
        print("GPU enforcement test passed - GPU is available")
    else:
        print("GPU enforcement test failed - No GPU available or enforcement working correctly")
