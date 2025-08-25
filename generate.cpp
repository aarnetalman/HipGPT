#include "gpt_model.h"
#include "tokenizer.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cstdlib>
#include <thread>
#include <chrono>

void print_usage(const char* prog_name) {
    std::cout << "Usage: " << prog_name << " --prompt \"<text>\" [--num_tokens N] [--max_seq_len N] [--ckpt PATH] [--tokenizer PATH]" << std::endl;
}

int main(int argc, char* argv[]) {
    std::string prompt = "";
    std::string checkpoint_path = "gpt_checkpoint.bin";
    std::string tokenizer_path = "tokenizer.json";
    int num_tokens_to_generate = 50;
    int max_seq_len = 32;
    int embed_dim = 128;
    int num_heads = 4;
    int ff_hidden_dim = 256;
    int num_layers = 2;

    // Parse CLI arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--prompt" && i + 1 < argc) {
            prompt = argv[++i];
        } else if (arg == "--num_tokens" && i + 1 < argc) {
            num_tokens_to_generate = std::atoi(argv[++i]);
        } else if (arg == "--max_seq_len" && i + 1 < argc) {
            max_seq_len = std::atoi(argv[++i]);
        } else if (arg == "--ckpt" && i + 1 < argc) {
            checkpoint_path = argv[++i];
        } else if (arg == "--tokenizer" && i + 1 < argc) {
            tokenizer_path = argv[++i];
        } else {
            print_usage(argv[0]);
            return 1;
        }
    }

    if (prompt.empty()) {
        std::cerr << "Error: --prompt is required." << std::endl;
        print_usage(argv[0]);
        return 1;
    }

    // Load tokenizer
    Tokenizer tokenizer;
    tokenizer.load(tokenizer_path);
    int vocab_size = tokenizer.vocab_size();

    // Load model
    GPTModel model(vocab_size, max_seq_len, embed_dim, num_heads, ff_hidden_dim, num_layers);
    model.load_checkpoint(checkpoint_path);

    // Encode input
    std::vector<int> input_ids = tokenizer.encode(prompt);
    if (input_ids.empty()) {
        std::cerr << "Prompt did not produce any known tokens." << std::endl;
        return 1;
    }

    std::cout << "Prompt: " << prompt << std::endl;
    std::cout << "Generated: " << std::flush;

    for (int step = 0; step < num_tokens; ++step) {
        model.forward(d_input_ids, d_logits, 1, current_len);

        int next_token = launch_sample_from_logits(
            d_logits + (current_len - 1) * vocab_size,
            vocab_size
        );

        input_ids.push_back(next_token);
        hipMemcpy(d_input_ids, input_ids.data(), (current_len + 1) * sizeof(int), hipMemcpyHostToDevice);
        current_len++;

        // Stream character-by-character
        std::string token_str = tokenizer.decode({next_token});
        for (char c : token_str) {
            std::cout << c << std::flush;
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }

        if (next_token == tokenizer.eos_token_id()) {
            break;
        }
    }


    std::cout << std::endl;
    return 0;
}
