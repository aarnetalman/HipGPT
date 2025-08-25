#!/bin/bash

# Go to script directory
cd "$(dirname "$0")" || exit

# Go to project root
cd .. || exit

# Create and enter data directory
mkdir -p data
cd data || exit


# URLs
BASE_URL="https://huggingface.co/datasets/wikitext/resolve/main/wikitext-2-raw-v1"
FILES=("wiki.train.raw" "wiki.valid.raw" "wiki.test.raw")

# Download if not already present
for file in "${FILES[@]}"; do
    if [[ -f $file ]]; then
        echo "$file already exists, skipping."
    else
        echo "Downloading $file..."
        wget "$BASE_URL/$file"
    fi
done

echo "Combining into data.txt..."
cat wiki.train.raw > data.txt
echo "Done. Created data.txt with training text."