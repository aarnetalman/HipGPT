<div align="center">

![HipGPT Logo](https://raw.githubusercontent.com/aarnetalman/HipGPT/main/assets/images/hip-hamster.png)

# 🐹 HipGPT

### A GPT-2 Implementation in C++ and HIP for AMD GPUs

*An educational, from-scratch implementation of a GPT-style transformer model with custom BPE tokenizer and HIP-accelerated training*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![ROCm](https://img.shields.io/badge/ROCm-5.0+-blue.svg)](https://rocm.docs.amd.com/)
[![C++](https://img.shields.io/badge/C++-17-blue.svg)](https://isocpp.org/)
[![Documentation](https://img.shields.io/badge/docs-hipgpt.github.io-green.svg)](https://hipgpt.github.io)

[📖 Documentation](https://hipgpt.github.io) · [🚀 Quick Start](#-quick-start) · [🎯 Examples](#-examples)

</div>

---

## What is HipGPT?

HipGPT is a **complete, educational implementation** of a GPT-2 style language model built from the ground up in modern C++. Unlike black-box implementations, every component is implemented transparently using AMD's HIP API for GPU acceleration.

### Perfect for Learning
- **Crystal Clear Code:** Every neural network operation implemented from scratch  
- **Educational Focus:** Designed to teach transformer internals, not just use them  
- **Complete Pipeline:** Data preprocessing, training, and inference all included  
- **AMD GPU Showcase:** Demonstrates HIP API capabilities on ROCm-enabled hardware  

### Key Features

| Feature | Description |
|---------|-------------|
| **Custom BPE Tokenizer** | Trainable on any text corpus, built from scratch |
| **Full Transformer Stack** | Multi-head attention, feed-forward layers, layer norm |
| **HIP GPU Kernels** | Custom CUDA-alternative kernels for AMD hardware |
| **FlashAttention** | Optimized attention kernels for head dims 32/64 (with fallback) |
| **Self-Contained Build** | Automatic dependency management with CMake |
| **Research Ready** | Modular design for easy experimentation |

---

## 🚀 Quick Start

### Prerequisites
- **AMD GPU** with ROCm support  
- **ROCm Toolkit** 5.0+ ([Installation Guide](https://rocm.docs.amd.com/en/latest/deploy/linux/index.html))  
- **CMake** 3.21+ and a modern C++ compiler  

### Installation & Training

```bash
# 1. Clone the repository
git clone https://github.com/aarnetalman/HipGPT.git
cd HipGPT

# 2. Download training data (Tiny Shakespeare)
./scripts/download_data.sh

# 3. Build the project
mkdir build && cd build
cmake .. -DCMAKE_CXX_COMPILER=/usr/bin/hipcc
make

# 4. Train your model
cd ..
./scripts/run_train.sh
```

After training completes, you'll have a working language model ready for text generation.

---

## Examples

### Text Generation

```bash
# Generate Shakespeare-style text
./build/generate --prompt "To be, or not to be:" --num_tokens 100

# More creative generation
./build/generate \
  --prompt "Once upon a time" \
  --num_tokens 150 \
  --top_k 50 \
  --temp 0.8
```

### Custom Training

```bash
# Train with custom hyperparameters
./scripts/run_train.sh \
  --vocab-size 2000 \
  --seq 64 \
  --lr 5e-4 \
  --steps 2000
```

### Sample Output

```
Prompt: "To be, or not to be:"

Generated: "To be, or not to be: that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles..."
```

---

## Model Specifications

### Default Configuration (\~28M params)

```
E = 256   (embedding size)
L = 8     (layers)
H = 8     (attention heads)
F = 1024  (feed-forward size)
V ≈ 5k    (vocabulary size)
S = 256   (sequence length)
```

Perfect size for:

* 🎓 Educational exploration
* 💻 Running on consumer GPUs
* ⚡ Fast iteration cycles
* 🔬 Research prototyping

---

## Project Structure

```
HipGPT/
├── include/           # Public API headers
│   ├── gpt_model.h
│   ├── tokenizer.h
│   └── hip_kernels.h
├── src/               # Implementation
│   ├── train_gpt.cpp
│   ├── generate.cpp
│   ├── gpt_model.cpp
│   └── hip_kernels.cpp
├── scripts/           # Automation scripts
│   ├── download_data.sh
│   └── run_train.sh
└── data/              # Training data
    └── data.txt
```

---

## Advanced Usage

### Custom Datasets

```bash
# Train on your own text file
./build/train_gpt --data-path your_dataset.txt
```

### Hyperparameter Tuning

```bash
# Experiment with model architecture
./build/train_gpt \
  --dim 256 \
  --layers 8 \
  --heads 8 \
  --ff 1024 \
  --lr 3e-4 \
  --batch 32
```

### Checkpointing

```bash
# Training automatically saves checkpoints in /checkpoints/[run-name]
# Includes symlinks to latest weights and configs.
```

---

## 🤝 Contributing

We welcome contributions! Here's how you can help:

* 🐛 **Bug Reports** — Found an issue? Please open a GitHub issue
* 🚀 **Feature Requests** — Ideas for improvements are always welcome
* 📖 **Documentation** — Help make the docs even clearer
* 💡 **Code Contributions** — Submit PRs for bug fixes or new features

---

<div align="center">

## 📄 License

This project is licensed under the [MIT License](LICENSE) - see the LICENSE file for details.

</div>
