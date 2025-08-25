#include "gpt_model.h"
#include "tokenizer.h"
#include "hip_kernels.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cstdlib>
#include <thread>
#include <chrono>

void print_usage(const char* prog_name) {
    std::cout << "Usage: " << prog_name << " --prompt \"<text>\" [--num_tokens N] [--max_seq_len N] [--ckpt PATH] [--tokenizer PATH] [--top_k N] [--temp F]" << std::endl;
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
    int top_k = 5;
    float temperature = 1.0f;

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
        } else if (arg == "--top_k" && i + 1 < argc) {
            top_k = std::atoi(argv[++i]);
        } else if (arg == "--temp" && i + 1 < argc) {
            temperature = std::stof(argv[++i]);
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

    // Call generate with the new parameters
    std::vector<int> generated_ids = model.generate(input_ids, num_tokens_to_generate, top_k, temperature);

    // Decode and stream the newly generated tokens
    for (size_t i = input_ids.size(); i < generated_ids.size(); ++i) {
        std::string token_str = tokenizer.decode({generated_ids[i]});
        std::cout << token_str << std::flush;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    std::cout << std::endl;

    // The GPTModel destructor will be called automatically here, freeing all HIP resources.
    return 0;
}
