# HIP-GPT: A GPT-2 Implementation in C++ and HIP

HIP-GPT is a lightweight, implementation of a GPT-2 style transformer model written from scratch in C++ and accelerated using AMD's **HIP API** for ROCm-enabled GPUs. This project includes all the necessary components for a modern language model: a custom BPE tokenizer, a transformer-based GPT model, and high-performance GPU kernels for training and inference.

The entire project is self-contained and designed to be a clear, understandable guide to the inner workings of large language models.

-----

## Features

  * **Custom BPE Tokenizer**: A Byte-Pair Encoding tokenizer built from scratch that can be trained on any raw text file.
  * **Transformer Architecture**: A standard GPT-2 style, decoder-only transformer model.
  * **GPU Acceleration**: All performance-critical operations (matrix multiplication, attention, layer normalization, etc.) are implemented with custom HIP kernels for AMD GPUs.
  * **End-to-End Workflow**: Includes scripts for downloading data, training the model from scratch, and generating new text.
  * **Self-Contained Build System**: Uses CMake to manage dependencies and build the project, automatically fetching the required JSON library.

-----

## Getting Started

Follow these steps to download the dataset, build the executables, and start using the model.

### **1. Prerequisites**

  * An **AMD GPU** compatible with the ROCm toolkit.
  * The **ROCm Toolkit** (version 5.0 or newer) installed.
  * **CMake** (version 3.21 or newer).
  * A C++ compiler (like `g++` or `clang++`).
  * `git` and `wget` for downloading dependencies and data.

### **2. Clone the Repository**

```bash
git clone git@github.com:aarnetalman/HIP_GPT.git
cd HIP_GPT
```

### **3. Download the Dataset**

The project includes a convenient script to download the WikiText-2 dataset and prepare it for training.

```bash
chmod +x scripts/download_wikitext.sh
./scripts/download_wikitext.sh
```

This will create a `data/` directory and place a `data.txt` file inside it.

### **4. Build the Project**

The project uses CMake to handle the build process.

```bash
mkdir build
cd build
cmake ..
make -j
```

This will create two executables in the `build/` directory: `train_gpt` and `generate`.

-----

## How to Use

### **Training the Model**

To train the model from scratch, run the `run_train.sh` script from the project's root directory. This script will automatically train the tokenizer (if needed) and then start the training process.

```bash
# From the project's root directory
chmod +x scripts/run_train.sh
./scripts/run_train.sh
```

The script will produce two key artifacts:

  * `tokenizer.json`: The trained vocabulary.
  * `gpt_checkpoint.bin`: The trained model weights.

You can customize the training process with command-line arguments. For example, to train for more steps with a smaller learning rate:

```bash
./scripts/run_train.sh --steps 1000 --lr 1e-3
```

### **Generating Text**

Once the model is trained, you can use the `generate` executable to create new text based on a prompt.

```bash
# From the build/ directory
./generate --prompt "Alan Turing was a"
```

You can control the output with several parameters:

  * `--prompt "<text>"`: The initial text to start generation from. **(Required)**
  * `--num_tokens N`: The number of new tokens to generate (default: 50).
  * `--top_k N`: Restricts sampling to the top `k` most likely tokens (default: 5).
  * `--temp F`: Controls the randomness of the output. Higher values are more random (default: 1.0).

For example:

```bash
./generate --prompt "The history of computing began with" --num_tokens 100 --top_k 50
```

-----

## Project Structure

```
.
├── scripts/
│   ├── download_wikitext.sh  # Downloads and prepares the training data.
│   └── run_train.sh          # A convenient wrapper to run the training script.
├── CMakeLists.txt            # The build configuration file for the project.
├── generate.cpp              # Main file for the text generation executable.
├── gpt_model.cpp/.h          # Implements the top-level GPT model architecture.
├── hip_kernels.cpp/.h        # Contains all custom GPU kernels written in HIP.
├── tokenizer.cpp/.h          # Implements the BPE tokenizer.
├── train_gpt.cpp             # Main file for the model training executable.
└── transformer_layer.cpp/.h  # Implements a single transformer block/layer.
```

-----

## License

This project is licensed under the MIT License.