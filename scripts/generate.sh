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

# Check if required model files exist
if [ ! -f "gpt_checkpoint.bin" ]; then
    echo "Warning: gpt_checkpoint.bin not found in current directory. Make sure you have trained a model first."
fi

if [ ! -f "tokenizer.json" ]; then
    echo "Warning: tokenizer.json not found in current directory. Make sure you have a trained tokenizer."
fi

# Print usage information if no arguments provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 --prompt \"Your text here\" [options]"
    echo "Example: $0 --prompt \"Once upon a time\" --num_tokens 100 --temp 0.8"
    echo ""
    echo "Options:"
    echo "  --prompt \"text\"       Text to generate from (required)"
    echo "  --num_tokens N         Number of tokens to generate (default: 50)"
    echo "  --max_seq_len N        Maximum sequence length (default: 32)"
    echo "  --ckpt PATH            Path to checkpoint file (default: gpt_checkpoint.bin)"
    echo "  --tokenizer PATH       Path to tokenizer file (default: tokenizer.json)"
    echo "  --top_k N              Top-k sampling parameter (default: 5)"
    echo "  --temp F               Temperature for sampling (default: 1.0)"
    echo "  --eos_id ID            End-of-sequence token ID (default: -1, disabled)"
    echo "  --stream true|false    Enable streaming output (default: true)"
    echo "  --delay_ms N           Delay between tokens in ms for streaming (default: 0)"
    echo ""
    exit 1
fi

# Run generation executable from the build directory
echo "Starting text generation..."
./build/generate "$@"