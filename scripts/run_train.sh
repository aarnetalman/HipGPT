#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Go to project root from script directory
cd "$(dirname "$0")/.." || exit

# Check if the executable exists inside the build directory
if [ ! -f "build/train_gpt" ]; then
    echo "Error: build/train_gpt executable not found. Please run 'cmake ..' and 'make' from the build directory first."
    exit 1
fi

# Run training executable from the build directory
./build/train_gpt "$@"