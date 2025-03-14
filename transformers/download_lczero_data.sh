#!/bin/bash

# Create directories if they don't exist
mkdir -p ./data-dir/compressed
mkdir -p ./data-dir/extracted

# URL base
BASE_URL="https://storage.lczero.org/files/training_data/run3"

# Download the first 3 tar files
echo "Downloading tar files..."
wget -P ./data-dir/compressed "${BASE_URL}/training-run3-20190614-2118.tar"
wget -P ./data-dir/compressed "${BASE_URL}/training-run3-20190614-2218.tar"
wget -P ./data-dir/compressed "${BASE_URL}/training-run3-20190614-2318.tar"

# Extract tar files to raw directory
echo "Extracting tar files..."
tar -xf ./data-dir/compressed/training-run3-20190614-2118.tar -C ./data-dir/extracted
tar -xf ./data-dir/compressed/training-run3-20190614-2218.tar -C ./data-dir/extracted
tar -xf ./data-dir/compressed/training-run3-20190614-2318.tar -C ./data-dir/extracted

echo "Data processing complete!" 