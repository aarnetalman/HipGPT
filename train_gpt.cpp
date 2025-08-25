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

    // File paths from CLI or defaults
    std::string data_path = args.count("--data_path") ? args["--data_path"] : "data.txt";
    std::string tokenizer_path = args.count("--tokenizer_path") ? args["--tokenizer_path"] : "tokenizer.json";
    std::string tokens_path = args.count("--tokens_path") ? args["--tokens_path"] : "tokens.bin";
    bool force_reset = args.count("--reset");

    // ---- Tokenizer + Dataset ----
    Tokenizer tokenizer(5000);

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
            std::cerr << "Error: data.txt not found.\n";
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
    GPTModel model(vocab_size, max_seq_len, embed_dim, num_heads, ff_hidden_dim, num_layers);

    // Allocate inputs and outputs
    std::vector<int> h_input_ids(total_tokens_per_batch);
    std::vector<int> h_labels(total_tokens_per_batch);

    int* d_input_ids;
    int* d_labels;
    float* d_logits;
    float* d_softmax_out;
    float* d_loss_grad;

    hipMalloc(&d_input_ids, total_tokens_per_batch * sizeof(int));
    hipMalloc(&d_labels, total_tokens_per_batch * sizeof(int));
    hipMalloc(&d_logits, total_tokens_per_batch * vocab_size * sizeof(float));
    hipMalloc(&d_softmax_out, total_tokens_per_batch * vocab_size * sizeof(float));
    hipMalloc(&d_loss_grad, total_tokens_per_batch * vocab_size * sizeof(float));

    // ---- Training loop ----
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    int cursor = 0;
    for (int step = 0; step < num_steps; ++step) {
        for (int b = 0; b < batch_size; ++b) {
            for (int t = 0; t < max_seq_len; ++t) {
                int idx = (cursor + t) % (tokens.size() - 1);
                h_input_ids[b * max_seq_len + t] = tokens[idx];
                h_labels[b * max_seq_len + t] = tokens[idx + 1];
            }
            cursor = (cursor + max_seq_len) % (tokens.size() - 1);
        }

        hipMemcpy(d_input_ids, h_input_ids.data(), total_tokens_per_batch * sizeof(int), hipMemcpyHostToDevice);
        hipMemcpy(d_labels, h_labels.data(), total_tokens_per_batch * sizeof(int), hipMemcpyHostToDevice);

        hipEventRecord(start, 0);

        model.forward(d_input_ids, d_logits, batch_size, max_seq_len);

        float loss = launch_softmax_loss(
            d_logits, d_softmax_out, d_labels, d_loss_grad,
            total_tokens_per_batch, vocab_size
        );
        float acc = launch_accuracy(d_softmax_out, d_labels, total_tokens_per_batch, vocab_size);

        model.backward(d_input_ids, d_loss_grad, batch_size, max_seq_len, learning_rate);

        hipEventRecord(stop, 0);
        hipEventSynchronize(stop);
        float ms = 0.0f;
        hipEventElapsedTime(&ms, start, stop);

        std::cout << "[Step " << step << "] Loss: " << loss
                  << " | Accuracy: " << acc * 100.0f << "%"
                  << " | Time: " << ms << " ms" << std::endl;
    }

    hipEventDestroy(start);
    hipEventDestroy(stop);

    model.save_checkpoint("gpt_checkpoint.bin");

    // ---- Cleanup ----
    hipFree(d_logits);
    hipFree(d_input_ids);
    hipFree(d_labels);
    hipFree(d_softmax_out);
    hipFree(d_loss_grad);

    return 0;
}