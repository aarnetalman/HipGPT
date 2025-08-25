#!/bin/bash

# Ensure we are in the script directory
cd "$(dirname "$0")" || exit

# Go to project root
cd .. || exit

# Run training with default arguments or override with CLI
./train_gpt "$@"
