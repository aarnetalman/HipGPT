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
#include <regex>
#include <algorithm>
#include <string>


// keep only the newest `keep_last` step-checkpoints; don't touch gpt_checkpoint.bin
void prune_old_checkpoints(const std::string& dir, const std::string& prefix, int keep_last) {
    namespace fs = std::filesystem;
    std::vector<std::pair<fs::file_time_type, fs::path>> step_ckpts;

    std::regex pat("^" + prefix + R"(_step[0-9]+\.bin$)");

    for (auto& p : fs::directory_iterator(dir)) {
        if (!p.is_regular_file()) continue;
        const std::string name = p.path().filename().string();
        if (std::regex_match(name, pat)) {
            step_ckpts.emplace_back(fs::last_write_time(p.path()), p.path());
        }
    }
    if ((int)step_ckpts.size() <= keep_last) return;

    // newest first
    std::sort(step_ckpts.begin(), step_ckpts.end(),
              [](auto& a, auto& b){ return a.first > b.first; });

    for (size_t i = keep_last; i < step_ckpts.size(); ++i) {
        std::error_code ec;
        std::cout << "Pruning old checkpoint: " << step_ckpts[i].second.filename().string() << std::endl;
        std::filesystem::remove(step_ckpts[i].second, ec);
        if (ec) {
            std::cerr << "Warning: failed to remove " << step_ckpts[i].second << " (" << ec.message() << ")\n";
        }
    }
}


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
    // ... args, tokenizer, dataset setup ...

    // Put all HIP allocations + model inside a scope
    {
        // ---- Model ----
        GPTModel model(vocab_size, max_seq_len, embed_dim, num_heads, ff_hidden_dim, num_layers);

        // Allocate inputs and outputs
        std::vector<int> h_input_ids(total_tokens_per_batch);
        std::vector<int> h_labels(total_tokens_per_batch);

        int* d_input_ids;  int* d_labels;
        float* d_logits;   float* d_softmax_out;  float* d_loss_grad;

        hipMalloc(&d_input_ids,  total_tokens_per_batch * sizeof(int));
        hipMalloc(&d_labels,     total_tokens_per_batch * sizeof(int));
        hipMalloc(&d_logits,     total_tokens_per_batch * vocab_size * sizeof(float));
        hipMalloc(&d_softmax_out,total_tokens_per_batch * vocab_size * sizeof(float));
        hipMalloc(&d_loss_grad,  total_tokens_per_batch * vocab_size * sizeof(float));

        // ---- Events ----
        hipEvent_t start, stop;
        hipEventCreate(&start);
        hipEventCreate(&stop);

        // ---- Training loop (unchanged) ----
        // ...

        hipEventDestroy(start);
        hipEventDestroy(stop);

        std::cout << "Saving final checkpoint to gpt_checkpoint.bin ..." << std::endl;
        model.save_checkpoint("gpt_checkpoint.bin");
        std::cout << "Checkpoint saved successfully." << std::endl;

        // Ensure all GPU work done before freeing raw buffers
        std::cout << "Synchronizing device before cleanup..." << std::endl;
        hipDeviceSynchronize();

        std::cout << "Freeing device buffers..." << std::endl;
        hipFree(d_logits);
        hipFree(d_input_ids);
        hipFree(d_labels);
        hipFree(d_softmax_out);
        hipFree(d_loss_grad);
        std::cout << "Device buffers freed." << std::endl;

    } // <-- `model` and any HIP-owning objects are destroyed here

    std::cout << "Final device reset..." << std::endl;
    hipDeviceReset();
    std::cout << "Done." << std::endl;
    return 0;
}
