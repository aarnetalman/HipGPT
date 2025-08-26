![HipGPT Logo](https://raw.githubusercontent.com/aarnetalman/HipGPT/main/assets/images/hip-hamster.png)

# 🐹 HipGPT: A GPT-2 Implementation in C++ and HIP

*HipGPT is a lightweight GPT-2 style transformer model implemented from scratch in C++ and accelerated with AMD's [HIP API](https://rocm.docs.amd.com/en/latest/understand/hip_api/hip_api.html) for ROCm-enabled GPUs.*  
It includes all the necessary components of a modern language model: a custom BPE tokenizer, a transformer-based GPT model, and GPU kernels for training and inference.

The project is self-contained and designed to be a **clear, educational guide** to the inner workings of large language models.  

📖 **Documentation:** [https://hipgpt.github.io](https://hipgpt.github.io)

---

## ❓ Why HipGPT?

- 🎓 **Educational clarity** – written from scratch in modern C++ to expose all the moving parts of a GPT model  
- ⚙️ **HIP-first** – showcases AMD’s HIP API for GPU acceleration on ROCm-enabled hardware  
- 🧩 **Minimal yet complete** – small enough to understand in full, but includes tokenizer, model, kernels, training, and inference  
- 🔬 **Research-friendly** – designed as a foundation for experimenting with language models on AMD GPUs  

---

## ✨ Features

- 🔤 **Custom BPE Tokenizer** – built from scratch, trainable on any raw text file  
- 🧠 **Transformer Architecture** – GPT-2 style, decoder-only model  
- ⚡ **GPU Acceleration** – custom HIP kernels for matrix multiplication, attention, layer norm, etc.  
- 📦 **End-to-End Workflow** – scripts for data download, training, and text generation  
- 🛠 **Self-Contained Build System** – CMake-based, automatically fetches required dependencies  

---

## 📏 Model Size

The number of trainable parameters depends on the vocabulary size (`V`) and transformer hyperparameters:

```
Total =
V·E                            (token embeddings)

* S·E                          (positional embeddings)
* L·(4E² + 2E·F + F + 9E)      (per transformer layer: QKV, O, FF1/FF2, LayerNorms)
* E·V + V                      (final projection + bias)

````

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

💾 **Memory footprint:**  
- FP32 ≈ 6.2 MB (weights only)  

---

## 🚀 Getting Started

### 1. Prerequisites
- An **AMD GPU** compatible with the ROCm toolkit  
- **ROCm Toolkit** (5.0 or newer)  
- **CMake** (3.21 or newer)  
- A C++ compiler (`g++` or `clang++`)  
- `git` and `wget` for setup  

### 2. Clone the Repository
```bash
git clone git@github.com:aarnetalman/HipGPT.git
cd HipGPT
````

### 3. Download the Dataset

```bash
chmod +x scripts/download_data.sh
./scripts/download_data.sh
```

Creates a `data/` directory with `data.txt`.

### 4. Build the Project

```bash
mkdir build
cd build
cmake .. -DCMAKE_CXX_COMPILER=/usr/bin/hipcc
make
```

Produces two executables: `train_gpt` and `generate`.

---

## 🏋️ Training

Run from the project root:

```bash
chmod +x scripts/run_train.sh
./scripts/run_train.sh
```

Artifacts:

* `tokenizer.json` – trained vocabulary
* `gpt_checkpoint.bin` – trained weights

Example custom run:

```bash
./scripts/run_train.sh --steps 1000 --lr 1e-3
```

---

## ✍️ Generating Text

```bash
./build/generate --prompt "To be, or not to be:"
```

Options:

* `--prompt "<text>"` (required)
* `--num_tokens N` (default: 50)
* `--top_k N` (default: 5)
* `--temp F` (default: 1.0)

Example:

```bash
./build/generate --prompt "My kingdom for a" --num_tokens 100 --top_k 50 --temp 0.8
```

---

## 📂 Project Structure

```
.
├── build/                 # Build outputs
├── data/                  # Training data
├── scripts/               # Helper scripts
│   ├── download_data.sh
│   └── run_train.sh
├── include/               # Public headers
│   ├── gpt_model.h
│   ├── hip_kernels.h
│   ├── tokenizer.h
│   └── transformer_layer.h
├── src/                   # Source files
│   ├── generate.cpp
│   ├── gpt_model.cpp
│   ├── hip_kernels.cpp
│   ├── tokenizer.cpp
│   ├── train_gpt.cpp
│   └── transformer_layer.cpp
├── CMakeLists.txt         # Build config
├── LICENSE
└── README.md
```

---

## 🏗 Architecture Overview

* **BPE Tokenizer** – converts raw text into integer IDs, trainable from scratch
* **Transformer Layer** – multi-head self-attention + FFN with residuals and layer norm
* **GPT Model** – embeddings + stacked transformer layers + final projection to vocab

---

## 📜 License

This project is licensed under the MIT License.

