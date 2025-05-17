"""
Logger utility for the bad word detection system.
"""
import sys
import os
from pathlib import Path
from loguru import logger


def setup_logger():
    """
    Set up and configure the logger.
    
    Returns:
        logger: Configured logger instance
    """
    # Create logs directory if it doesn't exist
    logs_dir = Path(__file__).parent.parent / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    log_file = logs_dir / "app.log"
    
    # Configure logger
    config = {
        "handlers": [
            {"sink": sys.stderr, "format": "{time} | {level} | {message}", "colorize": True},
            {"sink": str(log_file), "rotation": "10 MB", "retention": "1 month", "compression": "zip"},
        ]
    }
    
    logger.configure(**config)
    
    return logger