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
    echo "Usage: $0 --prompt \"Your text here\" --run-name NAME [options]"
    echo "Example: $0 --prompt \"Once upon a time\" --run-name shakespeare_v1 --num_tokens 100 --temp 0.8"
    echo ""
    echo "Required Options:"
    echo "  --prompt \"text\"       Text to generate from (required)"
    echo "  --run-name NAME        Training run name to use for generation (required)"
    echo ""
    echo "Optional Parameters (with defaults):"
    echo "  --step N               Specific checkpoint step (default: latest)"
    echo "  --num_tokens N         Number of tokens to generate (default: 100)"
    echo "  --max_seq_len N        Host-side generation window (default: 256)"
    echo "  --top_k N              Top-k sampling parameter (default: 50)"
    echo "  --temp F               Temperature for sampling (default: 0.8)"
    echo "  --top_p F              Nucleus sampling threshold (default: 0.9)"
    echo "  --rep-penalty F        Repetition penalty (default: 1.1)"
    echo "  --eos_id ID            End-of-sequence token ID (default: -1, disabled)"
    echo ""
    echo "Examples:"
    echo "  $0 --prompt \"Hello world\" --run-name my_model"
    echo "  $0 --prompt \"The cat\" --run-name my_model --step 1000 --temp 0.8"
    echo "  $0 --prompt \"Once upon\" --run-name story_model --num_tokens 200 --top_k 50"
    echo ""
    exit 1
fi

# Run generation executable from the build directory
echo "Starting text generation with run-based configuration..."
./build/generate "$@"
