![HipGPT Logo](https://raw.githubusercontent.com/aarnetalman/HipGPT/main/assets/images/hip-hamster.png)


# 🐹 HipGPT: A GPT-2 Implementation in C++ and HIP

HipGPT is a lightweight, implementation of a GPT-2 style transformer model written from scratch in C++ and accelerated using AMD's **[HIP API](https://rocm.docs.amd.com/en/latest/understand/hip_api/hip_api.html)** for ROCm-enabled GPUs. This project includes all the necessary components for a modern language model: a custom BPE tokenizer, a transformer-based GPT model, and GPU kernels for training and inference.

**Model size (default):** ~1.55M params (E=128, L=2, H=4, F=256, V≈5k, S=32)

The entire project is self-contained and designed to be a clear, understandable guide to the inner workings of large language models.

**Documentation:** [https://hipgpt.github.io](https://hipgpt.github.io)

-----

## Features

  * **Custom BPE Tokenizer**: A Byte-Pair Encoding tokenizer built from scratch that can be trained on any raw text file.
  * **Transformer Architecture**: A standard GPT-2 style, decoder-only transformer model.
  * **GPU Acceleration**: All performance-critical operations (matrix multiplication, attention, layer normalization, etc.) are implemented with custom HIP kernels for AMD GPUs.
  * **End-to-End Workflow**: Includes scripts for downloading data, training the model from scratch, and generating new text.
  * **Self-Contained Build System**: Uses CMake to manage dependencies and build the project, automatically fetching the required JSON library.

-----

## Model Size

The number of trainable parameters depends on the vocabulary size (`V`) learned from your dataset, along with the transformer hyperparameters:

**Parameter formula (no weight tying):**

```

Total = V·E            (token embeddings)
\+ S·E            (positional embeddings)
\+ L·(4E² + 2E·F + F + 9E)   (per transformer layer: QKV, O, FF1/FF2, LayerNorms)
\+ E·V + V        (final projection + bias)

```

Where:
- `E` = embedding dimension  
- `L` = number of layers  
- `H` = number of attention heads (`E` must be divisible by `H`)  
- `F` = feed-forward hidden dimension  
- `V` = vocabulary size  
- `S` = maximum sequence length  

**Default configuration:**  
`E=128, L=2, H=4, F=256, V≈5000, S=32`  
➡️ **~1.55M trainable parameters**

*Memory footprint:*  
- FP32 weights ≈ 6.2 MB  
- FP16 weights ≈ 3.1 MB  
(excluding optimizer states)

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
git git@github.com:aarnetalman/HipGPT.git
cd HipGPT
```

### **3. Download the Dataset**

The project includes a convenient script to download the Tiny Shakespeare dataset and prepare it for training.

```bash
chmod +x scripts/download_data.sh
./scripts/download_data.sh
```

This will create a `data/` directory and place a `data.txt` file inside it.

### **4. Build the Project**

The project uses CMake to handle the build process.

```bash
mkdir build
cd build
cmake .. -DCMAKE_CXX_COMPILER=/usr/bin/hipcc
make
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
./build/generate --prompt "To be, or not to be:"
```

You can control the output with several parameters:

  * `--prompt "<text>"`: The initial text to start generation from. **(Required)**
  * `--num_tokens N`: The number of new tokens to generate (default: 50).
  * `--top_k N`: Restricts sampling to the top `k` most likely tokens (default: 5).
  * `--temp F`: Controls the randomness of the output. Higher values are more random (default: 1.0).

For example:

```bash
./build/generate --prompt "My kingdom for a" --num_tokens 100 --top_k 50 --temp 0.8
```

-----

## Project Structure

The project is organized into `src` and `include` directories for a clean separation of source and header files.

```
.
├── build/                 # Build files (created by CMake)
├── data/                  # Data files (e.g., data.txt)
├── scripts/               # Helper scripts
│   ├── download_data.sh
│   └── run_train.sh
├── include/               # All public header files (.h)
│   ├── gpt_model.h
│   ├── hip_kernels.h
│   ├── tokenizer.h
│   └── transformer_layer.h
│
├── src/                   # All source files (.cpp)
│   ├── generate.cpp
│   ├── gpt_model.cpp
│   ├── hip_kernels.cpp
│   ├── tokenizer.cpp
│   ├── train_gpt.cpp
│   └── transformer_layer.cpp
│
├── CMakeLists.txt         # Main build configuration
├── LICENSE                # Your project's license
└── README.md              # Project documentation
```

-----

## Architecture Overview

  * **BPE Tokenizer**: The `Tokenizer` class is responsible for converting raw text into a sequence of integer token IDs that the model can understand. It can be trained from scratch on a corpus to learn a vocabulary of sub-word units.
  * **Transformer Layer**: The `TransformerLayer` is the core building block of the model. It contains a multi-head self-attention mechanism and a position-wise feed-forward network, with residual connections and layer normalization.
  * **GPT Model**: The `GPTModel` class assembles the entire network. It manages the token and positional embeddings, stacks multiple `TransformerLayer` instances, and adds a final linear layer to produce output logits over the vocabulary.

-----

## License

This project is licensed under the MIT License.