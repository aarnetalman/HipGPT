#!/bin/bash

# This script downloads the "Tiny Shakespeare" dataset, a single clean text file,
# which is a great starting point for training language models.

set -e

# Go to project root from script directory
cd "$(dirname "$0")/.." || exit
DATA_DIR="data"
DATA_FILE="$DATA_DIR/data.txt"

mkdir -p "$DATA_DIR"

# URL for the Tiny Shakespeare dataset
URL="https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

if [[ -f "$DATA_FILE" ]]; then
    echo "$DATA_FILE already exists, skipping download."
else
    echo "Downloading Tiny Shakespeare dataset..."
    wget -O "$DATA_FILE" "$URL"
    echo "Done. Created $DATA_FILE."
fi
