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
#include <cmath>
#include <chrono>
#include <iomanip>
#include <nlohmann/json.hpp>
using json = nlohmann::json;


std::string timestamp() {
    auto now = std::chrono::system_clock::now();
    auto t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&t), "%Y%m%d_%H%M%S");
    return ss.str();
}

void save_config(const std::string& path,
                 int vocab_size, int max_seq_len, int embed_dim, int num_heads,
                 int ff_hidden_dim, int num_layers,
                 int batch_size, float learning_rate,
                 const std::string& tokenizer_path,
                 const std::string& tokens_path,
                 const std::string& ckpt_path,
                 int step) {
    json config = {
        {"model", {
            {"vocab_size", vocab_size},
            {"max_seq_len", max_seq_len},
            {"embed_dim", embed_dim},
            {"num_heads", num_heads},
            {"ff_hidden_dim", ff_hidden_dim},
            {"num_layers", num_layers}
        }},
        {"tokenizer", {
            {"path", tokenizer_path},
            {"tokens_path", tokens_path}
        }},
        {"training", {
            {"batch_size", batch_size},
            {"learning_rate", learning_rate},
            {"steps", step}
        }},
        {"checkpoint", {
            {"latest", ckpt_path},
            {"step", step}
        }}
    };

    std::ofstream cfg(path);
    cfg << config.dump(4);
}


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
    int log_every = args.count("--log-every") ? std::stoi(args["--log-every"]) : 50;
    int ckpt_every = args.count("--ckpt-every") ? std::stoi(args["--ckpt-every"]) : 500;
    int keep_last  = args.count("--keep-last")  ? std::stoi(args["--keep-last"])  : 5;
    std::string run_name = args.count("--run-name") ? args["--run-name"] : "run_" + std::to_string(time(nullptr));
    std::string run_dir = "checkpoints/" + run_name;

    // make sure directory exists
    std::filesystem::create_directories(run_dir);


    if (keep_last < 1) keep_last = 1;
    if (ckpt_every < 0) ckpt_every = 0; // 0 disables periodic ckpt

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

    {
        // ---- Model ----
        GPTModel model(vocab_size, max_seq_len, embed_dim, num_heads, ff_hidden_dim, num_layers);
        if (args.count("--ckpt")) {
            std::string ckpt_path = args["--ckpt"];
            if (std::filesystem::exists(ckpt_path)) {
                std::cout << "Loading checkpoint from " << ckpt_path << " ..." << std::endl;
                model.load_checkpoint(ckpt_path);
                std::cout << "Checkpoint loaded successfully." << std::endl;
            } else {
                std::cerr << "Warning: checkpoint " << ckpt_path << " not found, starting fresh.\n";
            }
        }

        // Allocate inputs and outputs
        std::vector<int> h_input_ids(total_tokens_per_batch);
        std::vector<int> h_labels(total_tokens_per_batch);

        int* d_input_ids;
        int* d_labels;
        float* d_logits;
        float* d_softmax_out;
        float* d_loss_grad;

        hipMalloc(&d_input_ids,  total_tokens_per_batch * sizeof(int));
        hipMalloc(&d_labels,     total_tokens_per_batch * sizeof(int));
        hipMalloc(&d_logits,     total_tokens_per_batch * vocab_size * sizeof(float));
        hipMalloc(&d_softmax_out,total_tokens_per_batch * vocab_size * sizeof(float));
        hipMalloc(&d_loss_grad,  total_tokens_per_batch * vocab_size * sizeof(float));

        // ---- Events ----
        hipEvent_t start, stop;
        hipEventCreate(&start);
        hipEventCreate(&stop);

        // ---- Training loop ----
        int cursor = 0;
        int adam_t = 0;
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
            hipMemcpy(d_labels,    h_labels.data(),    total_tokens_per_batch * sizeof(int), hipMemcpyHostToDevice);

            hipEventRecord(start, 0);

            model.forward(d_input_ids, d_logits, batch_size, max_seq_len);

            float loss = launch_softmax_loss(
                d_logits, d_softmax_out, d_labels, d_loss_grad,
                total_tokens_per_batch, vocab_size
            );
            float acc = launch_accuracy(d_softmax_out, d_labels, total_tokens_per_batch, vocab_size);
            adam_t++;
            model.backward(d_input_ids, d_loss_grad, batch_size, max_seq_len, learning_rate, adam_t);

            hipEventRecord(stop, 0);
            hipEventSynchronize(stop);
            float ms = 0.0f;
            hipEventElapsedTime(&ms, start, stop);

            if ((step % log_every) == 0 || step == num_steps - 1) {
                float ppl = std::exp(loss);
                std::cout << "[Step " << step << "] Loss: " << loss
                        << " | Perplexity: " << ppl
                        << " | Accuracy: " << acc * 100.0f << "%"
                        << " | Time: " << ms << " ms" << std::endl;
            }

            if (ckpt_every > 0 && step > 0 && (step % ckpt_every) == 0) {
                std::string ckpt_fname = run_dir + "/" + run_name + "_step" + std::to_string(step) + ".bin";
                std::cout << "Saving checkpoint to " << ckpt_fname << " ..." << std::endl;
                model.save_checkpoint(ckpt_fname);

                save_config(run_dir + "/" + run_name + "_config.json",
                    vocab_size, max_seq_len, embed_dim, num_heads,
                    ff_hidden_dim, num_layers,
                    batch_size, learning_rate,
                    tokenizer_path, tokens_path,
                    ckpt_fname, step);
                std::cout << "Checkpoint saved." << std::endl;

                prune_old_checkpoints(run_dir, run_name, keep_last);
            }

        }

        hipEventDestroy(start);
        hipEventDestroy(stop);

        std::string final_ckpt = run_dir + "/" + run_name + "_step" + std::to_string(num_steps) + ".bin";
        std::cout << "Saving final checkpoint to " << final_ckpt << " ..." << std::endl;
        model.save_checkpoint(final_ckpt);

        std::string final_cfg = run_dir + "/" + run_name + "_step" + std::to_string(num_steps) + "_config.json";
        save_config(final_cfg,
            vocab_size, max_seq_len, embed_dim, num_heads,
            ff_hidden_dim, num_layers,
            batch_size, learning_rate,
            tokenizer_path, tokens_path,
            final_ckpt, num_steps);

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

    } // model and HIP-owning objects are destroyed here

    std::cout << "Final device reset..." << std::endl;
    hipDeviceReset();
    std::cout << "Done." << std::endl;

    return 0;
}
