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

# Print usage info if no args provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 [training options]"
    echo "Example: $0 --steps 10000 --batch 32 --seq 256 --dim 512 --heads 16 --ff 2048 --layers 8 --run-name tinyshakespeare"
    echo ""
    exit 1
fi

# Run training executable from the build directory
./build/train_gpt "$@"
