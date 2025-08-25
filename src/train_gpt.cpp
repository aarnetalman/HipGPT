#include "gpt_model.h"
#include "hip_kernels.h"
#include "tokenizer.h"
#include <hip/hip_runtime.h>
#include <vector>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <filesystem>

// Add debugging macro for HIP calls in main
#define HIP_CHECK(call) do { \
    hipError_t err = call; \
    if (err != hipSuccess) { \
        std::cerr << "HIP Error at " << __FILE__ << ":" << __LINE__ \
                  << " - " << hipGetErrorString(err) << std::endl; \
        exit(1); \
    } \
} while(0)

// Simple CLI argument parser
std::unordered_map<std::string, std::string> parse_args(int argc, char** argv) {
    std::unordered_map<std::string, std::string> args;
    for (int i = 1; i < argc; ++i) {
        std::string key(argv[i]);
        if (key.substr(0, 2) == "--" && i + 1 < argc) {
            args[key] = argv[++i];
        } else if (key.substr(0, 2) == "--") {
            args[key] = "true";
        }
    }
    return args;
}

int main(int argc, char** argv) {
    auto args = parse_args(argc, argv);

    // Hyperparameters from CLI or defaults
    int max_seq_len = args.count("--seq") ? std::stoi(args["--seq"]) : 32;
    int embed_dim = args.count("--dim") ? std::stoi(args["--dim"]) : 128;
    int num_heads = args.count("--heads") ? std::stoi(args["--heads"]) : 4;
    int ff_hidden_dim = args.count("--ff") ? std::stoi(args["--ff"]) : 256;
    int num_layers = args.count("--layers") ? std::stoi(args["--layers"]) : 2;
    int batch_size = args.count("--batch") ? std::stoi(args["--batch"]) : 4;
    int num_steps = args.count("--steps") ? std::stoi(args["--steps"]) : 10;
    float learning_rate = args.count("--lr") ? std::stof(args["--lr"]) : 1e-2f;
    int vocab_size_limit = args.count("--vocab-size") ? std::stoi(args["--vocab-size"]) : 5000;

    // File paths from CLI or defaults
    std::string data_path = args.count("--data-path") ? args["--data-path"] : "data/data.txt";
    std::string tokenizer_path = args.count("--tokenizer-path") ? args["--tokenizer-path"] : "tokenizer.json";
    std::string tokens_path = args.count("--tokens-path") ? args["--tokens-path"] : "tokens.bin";
    bool force_reset = args.count("--reset");

    // === GPU INITIALIZATION ===
    std::cout << "=== GPU INITIALIZATION ===" << std::endl;
    print_gpu_info();
    check_gpu_memory();

    // ---- Tokenizer + Dataset ----
    Tokenizer tokenizer(vocab_size_limit);

    std::vector<int> tokens;

    if (!force_reset && std::filesystem::exists(tokenizer_path) && std::filesystem::exists(tokens_path)) {
        tokenizer.load(tokenizer_path);
        std::ifstream in(tokens_path, std::ios::binary);
        int id;
        while (in.read(reinterpret_cast<char*>(&id), sizeof(int))) {
            tokens.push_back(id);
        }
        std::cout << "Loaded tokenizer and tokenized dataset (" << tokens.size() << " tokens)\n";
    } else {
        std::ifstream file(data_path);
        if (!file) {
            std::cerr << "Error: " << data_path << " not found.\n";
            return 1;
        }
        std::stringstream buffer;
        buffer << file.rdbuf();
        std::string text = buffer.str();

        tokenizer.train_bpe(text);
        tokenizer.save(tokenizer_path);
        tokens = tokenizer.encode(text);

        std::ofstream out(tokens_path, std::ios::binary);
        for (int id : tokens) {
            out.write(reinterpret_cast<const char*>(&id), sizeof(int));
        }
        std::cout << "Trained tokenizer and saved " << tokens.size() << " tokens\n";
    }

    int vocab_size = tokenizer.vocab_size();
    int total_tokens_per_batch = batch_size * max_seq_len;

    std::cout << "Using vocab size: " << vocab_size << std::endl;

    // ---- Model ----
    std::cout << "Creating GPT model..." << std::endl;
    GPTModel model(vocab_size, max_seq_len, embed_dim, num_heads, ff_hidden_dim, num_layers);
    std::cout << "Model created successfully" << std::endl;
    debug_checkpoint("After model creation");

    // Allocate inputs and outputs
    std::vector<int> h_input_ids(total_tokens_per_batch);
    std::vector<int> h_labels(total_tokens_per_batch);

    int* d_input_ids;
    int* d_labels;
    float* d_logits;
    float* d_softmax_out;
    float* d_loss_grad;

    std::cout << "Allocating GPU memory..." << std::endl;
    HIP_CHECK(hipMalloc(&d_input_ids, total_tokens_per_batch * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_labels, total_tokens_per_batch * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_logits, total_tokens_per_batch * vocab_size * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_softmax_out, total_tokens_per_batch * vocab_size * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_loss_grad, total_tokens_per_batch * vocab_size * sizeof(float)));
    debug_checkpoint("After GPU memory allocation");

    // ---- Training loop ----
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    int cursor = 0;
    for (int step = 0; step < num_steps; ++step) {
        std::cout << "\n=== TRAINING STEP " << step << " ===" << std::endl;
        
        // Prepare batch data
        for (int b = 0; b < batch_size; ++b) {
            for (int t = 0; t < max_seq_len; ++t) {
                int idx = (cursor + t) % (tokens.size() - 1);
                h_input_ids[b * max_seq_len + t] = tokens[idx];
                h_labels[b * max_seq_len + t] = tokens[idx + 1];
            }
            cursor = (cursor + max_seq_len) % (tokens.size() - 1);
        }

        HIP_CHECK(hipMemcpy(d_input_ids, h_input_ids.data(), total_tokens_per_batch * sizeof(int), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_labels, h_labels.data(), total_tokens_per_batch * sizeof(int), hipMemcpyHostToDevice));
        debug_checkpoint("After data transfer to GPU");

        HIP_CHECK(hipEventRecord(start, 0));

        std::cout << "Starting forward pass..." << std::endl;
        model.forward(d_input_ids, d_logits, batch_size, max_seq_len);
        debug_checkpoint("After forward pass");

        std::cout << "Computing loss..." << std::endl;
        float loss = launch_softmax_loss(
            d_logits, d_softmax_out, d_labels, d_loss_grad,
            total_tokens_per_batch, vocab_size
        );
        debug_checkpoint("After loss computation");
        
        float acc = launch_accuracy(d_softmax_out, d_labels, total_tokens_per_batch, vocab_size);
        debug_checkpoint("After accuracy computation");

        std::cout << "Starting backward pass..." << std::endl;
        model.backward(d_input_ids, d_loss_grad, batch_size, max_seq_len, learning_rate);
        debug_checkpoint("After backward pass");

        HIP_CHECK(hipEventRecord(stop, 0));
        HIP_CHECK(hipEventSynchronize(stop));
        float ms = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&ms, start, stop));

        std::cout << "[Step " << step << "] Loss: " << loss
                  << " | Accuracy: " << acc * 100.0f << "%"
                  << " | Time: " << ms << " ms" << std::endl;
        
        debug_checkpoint("End of training step " + std::to_string(step));
    }

    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));

    std::cout << "Saving checkpoint..." << std::endl;
    model.save_checkpoint("gpt_checkpoint.bin");
    debug_checkpoint("After saving checkpoint");

    // ---- Cleanup ----
    std::cout << "Cleaning up GPU memory..." << std::endl;
    HIP_CHECK(hipFree(d_logits));
    HIP_CHECK(hipFree(d_input_ids));
    HIP_CHECK(hipFree(d_labels));
    HIP_CHECK(hipFree(d_softmax_out));
    HIP_CHECK(hipFree(d_loss_grad));
    debug_checkpoint("After cleanup");

    std::cout << "Training completed successfully!" << std::endl;
    return 0;
}