#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Ensure we are in the script directory
cd "$(dirname "$0")" || exit

# Go to project root
cd .. || exit

# Check if the executable exists
if [ ! -f "train_gpt" ]; then
    echo "Error: train_gpt executable not found. Please run 'cmake .' and 'make' first."
    exit 1
fi

# Run training with default arguments or override with CLI
./train_gpt "$@"