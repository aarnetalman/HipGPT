#!/bin/bash

set -e

cd "$(dirname "$0")/.." || exit
DATA_DIR="data"

mkdir -p "$DATA_DIR"
cd "$DATA_DIR" || exit

BASE_URL="https://huggingface.co/datasets/Salesforce/wikitext/raw/main/wikitext-2-raw-v1"
FILES=("wiki.train.raw" "wiki.valid.raw" "wiki.test.raw")

# Download if not already present
for file in "${FILES[@]}"; do
    if [[ -f $file ]]; then
        echo "$file already exists, skipping."
    else
        echo "Downloading $file..."
        wget -nc "$BASE_URL/$file"
    fi
done

DATA_FILE="data.txt"
if [[ -f "$DATA_FILE" ]]; then
    echo "$DATA_FILE already exists, skipping concatenation."
else
    echo "Combining into data.txt..."
    cat wiki.train.raw wiki.valid.raw wiki.test.raw > "$DATA_FILE"
    echo "Done. Created data.txt with all raw data."
fi
