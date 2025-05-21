#!/bin/bash

# Run training inside the Docker container

echo "Starting training inside Docker container..."
docker-compose exec toxic-chat-detector python3 -m bad_word_detector.scripts.enhanced_training --combine_datasets --full_bert

echo "Training completed!"
