#!/bin/bash

# Run the Docker container with GPU support

# Check if nvidia-docker is installed
if ! command -v nvidia-docker &> /dev/null && ! command -v docker &> /dev/null; then
    echo "Error: Docker or nvidia-docker is not installed."
    exit 1
fi

# Check if image exists
if ! docker image inspect toxic-chat-detector:latest &> /dev/null; then
    echo "Error: Docker image 'toxic-chat-detector:latest' not found."
    echo "Please build the image first with './scripts/docker-build.sh'"
    exit 1
fi

echo "Starting toxic-chat-detector container with GPU support..."
docker-compose up -d

echo "Container started!"
echo "API is now available at http://localhost:8000"
echo "To check logs: docker-compose logs -f"
echo "To stop: docker-compose down"
