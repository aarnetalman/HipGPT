#!/bin/bash
# Exit immediately if a command exits with a non-zero status.
set -e

# Go to project root from script directory
cd "$(dirname "$0")/.." || exit

# Check if the executable exists inside the build directory
if [ ! -f "build/generate" ]; then
    echo "Error: build/generate executable not found. Please run 'cmake ..' and 'make' from the build directory first."
    exit 1
fi

# Print usage information if no arguments provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 --prompt \"Your text here\" [options]"
    echo "Example: $0 --prompt \"Once upon a time\" --num_tokens 100 --temp 0.8 --run-name tinyshakespeare"
    echo ""
    echo "Options:"
    echo "  --prompt \"text\"       Text to generate from (required)"
    echo "  --num_tokens N         Number of tokens to generate (default: 50)"
    echo "  --max_seq_len N        Host-side generation window (default: 32)"
    echo "  --ckpt PATH            Path to checkpoint file (loads matching *_config.json)"
    echo "  --run-name NAME        Use latest config from checkpoints/NAME/"
    echo "  --top_k N              Top-k sampling parameter (default: 5)"
    echo "  --temp F               Temperature for sampling (default: 1.0)"
    echo "  --eos_id ID            End-of-sequence token ID (default: -1, disabled)"
    echo ""
    exit 1
fi

# Run generation executable from the build directory
echo "Starting text generation..."
./build/generate "$@"
