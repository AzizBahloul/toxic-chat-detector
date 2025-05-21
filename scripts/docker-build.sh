#!/bin/bash

# Build the Docker image for the toxic chat detector

echo "Building toxic-chat-detector Docker image..."
docker build -t toxic-chat-detector:latest .

echo "Image built successfully!"
echo "To run the container, use: docker-compose up"
