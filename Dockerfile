# Use NVIDIA CUDA base image with Python
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-setuptools \
    python3-dev \
    build-essential \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY bad_word_detector/requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Download NLTK data
RUN python3 -m nltk.downloader punkt stopwords

# Create necessary directories
RUN mkdir -p bad_word_detector/data/models/bad_word_detector \
    bad_word_detector/data/raw \
    bad_word_detector/data/processed \
    bad_word_detector/logs

# Expose port for API
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Set environment variables for GPU usage
ENV USE_GPU=true

# Command to run the API
ENTRYPOINT ["uvicorn", "bad_word_detector.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
