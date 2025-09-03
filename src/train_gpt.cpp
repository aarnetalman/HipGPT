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
#include <regex>
#include <algorithm>
#include <string>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <nlohmann/json.hpp>
#include <filesystem>

using json = nlohmann::json;
namespace fs = std::filesystem;

__global__ void build_batch_kernel_tokenwise(const int* __restrict__ tokens, int tokens_len,
                                             int* __restrict__ input_ids, int* __restrict__ labels,
                                             int cursor, int B, int S)
{
    // flat index over the whole batch (B * S)
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int N = B * S;
    if (n >= N) return;

    // map back to (b, t)
    // b = sequence in the batch, t = position within the sequence
    int b = n / S;
    int t = n - b * S;

    // Starting index for this sequence was base = cursor + b*S
    // We need token at (base + t) and label at (base + t + 1), both wrapped mod (tokens_len - 1)
    int base = cursor + b * S;
    int idx  = base + t;

    int wrap = tokens_len - 1;                  // we need idx+1 valid, so wrap on (len-1)
    if (idx >= wrap) idx -= (idx / wrap) * wrap; // fast wrap; avoids while or modulo
    int idxp1 = idx + 1; if (idxp1 == tokens_len) idxp1 = 0; // wrap the +1

    input_ids[n] = tokens[idx];
    labels[n]    = tokens[idxp1];
}



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
                 int warmup_steps, float min_lr,
                 const std::string& tokenizer_path,
                 const std::string& tokens_path,
                 const std::string& ckpt_path,
                 int step) {
    // save only the basenames, not full paths
    std::string tok_file   = std::filesystem::path(tokenizer_path).filename().string();
    std::string tokens_file= std::filesystem::path(tokens_path).filename().string();
    std::string ckpt_file  = std::filesystem::path(ckpt_path).filename().string();

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
            {"path", tok_file},
            {"tokens_path", tokens_file}
        }},
        {"training", {
            {"batch_size", batch_size},
            {"learning_rate", learning_rate},
            {"warmup_steps", warmup_steps},
            {"min_lr", min_lr},
            {"steps", step}
        }},
        {"checkpoint", {
            {"latest", ckpt_file},
            {"step", step}
        }}
    };

    std::ofstream cfg(path);
    cfg << config.dump(4);
}


// keep only the newest `keep_last` step-checkpoints (bin+config)
// always preserve the newest one even if keep_last <= 0
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
    if (step_ckpts.empty()) return;

    // newest first
    std::sort(step_ckpts.begin(), step_ckpts.end(),
              [](auto& a, auto& b){ return a.first > b.first; });

    // ensure at least 1 is always kept
    int limit = std::max(1, keep_last);

    for (size_t i = limit; i < step_ckpts.size(); ++i) {
        const fs::path ckpt_bin = step_ckpts[i].second;
        const std::string bin_name = ckpt_bin.filename().string();

        // Derive config filename (replace .bin with _config.json)
        std::string cfg_name = std::regex_replace(bin_name, std::regex("\\.bin$"), "_config.json");
        fs::path cfg_path = ckpt_bin.parent_path() / cfg_name;

        std::error_code ec;
        std::cout << "Pruning old checkpoint: " << bin_name << std::endl;
        fs::remove(ckpt_bin, ec);
        if (ec) {
            std::cerr << "Warning: failed to remove " << ckpt_bin << " (" << ec.message() << ")\n";
        }

        // Try to remove config too
        ec.clear();
        if (fs::exists(cfg_path)) {
            std::cout << "Pruning old config: " << cfg_name << std::endl;
            fs::remove(cfg_path, ec);
            if (ec) {
                std::cerr << "Warning: failed to remove " << cfg_path << " (" << ec.message() << ")\n";
            }
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
    int max_seq_len   = args.count("--seq")   ? std::stoi(args["--seq"])   : 256;
    int embed_dim     = args.count("--dim")   ? std::stoi(args["--dim"])   : 256;
    int num_heads     = args.count("--heads") ? std::stoi(args["--heads"]) : 8;
    int ff_hidden_dim = args.count("--ff")    ? std::stoi(args["--ff"])    : 1024;
    int num_layers    = args.count("--layers")? std::stoi(args["--layers"]): 8;
    int batch_size    = args.count("--batch") ? std::stoi(args["--batch"]) : 32;
    int num_steps     = args.count("--steps") ? std::stoi(args["--steps"]) : 50000;
    float learning_rate = args.count("--lr")  ? std::stof(args["--lr"])    : 3e-4f;
    int vocab_size_limit = args.count("--vocab-size") ? std::stoi(args["--vocab-size"]) : 5000;
    int warmup_steps = args.count("--warmup-steps") ? std::stoi(args["--warmup-steps"]) : 2000;
    float min_lr     = args.count("--min-lr")       ? std::stof(args["--min-lr"])       : learning_rate * 0.1f;



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

    // Tokenizer + dataset paths inside run directory
    std::string run_tokenizer_path = run_dir + "/tokenizer.json";
    std::string run_tokens_path    = run_dir + "/tokens.bin";

    bool tokenizer_exists = std::filesystem::exists(run_tokenizer_path) && 
                            std::filesystem::exists(run_tokens_path);

    if (!force_reset && tokenizer_exists) {
        tokenizer.load(run_tokenizer_path);
        std::ifstream in(run_tokens_path, std::ios::binary);
        int id;
        while (in.read(reinterpret_cast<char*>(&id), sizeof(int))) {
            tokens.push_back(id);
        }
        std::cout << "Loaded existing tokenizer and dataset (" << tokens.size() << " tokens)\n";
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
        tokenizer.save(run_tokenizer_path);
        tokens = tokenizer.encode(text);

        std::ofstream out(run_tokens_path, std::ios::binary);
        for (int id : tokens) {
            out.write(reinterpret_cast<const char*>(&id), sizeof(int));
        }
        std::cout << "Trained tokenizer and saved (" << tokens.size() << " tokens)\n";
    }

    int vocab_size = tokenizer.vocab_size();
    int total_tokens_per_batch = batch_size * max_seq_len;
    auto lr_at = [&](int step, int total_updates) -> float {
        // Linear warmup to 'learning_rate'
        if (step <= warmup_steps) {
            float denom = static_cast<float>(std::max(1, warmup_steps));
            return learning_rate * (static_cast<float>(step) / denom);
        }
        // Cosine decay from 'learning_rate' to 'min_lr' over the remaining steps
        const int s = step - warmup_steps;
        const int T = std::max(1, total_updates - warmup_steps);
        constexpr float PI = 3.14159265358979323846f;
        const float cos_term = 0.5f * (1.0f + std::cos(PI * (static_cast<float>(s) / static_cast<float>(T))));
        return min_lr + (learning_rate - min_lr) * cos_term;
    };

    const int needed = max_seq_len + 1;
    if ((int)tokens.size() <= needed) {
        std::cerr << "Error: tokenized dataset too small for seq=" << max_seq_len
                << " (have " << tokens.size() << " tokens, need > " << needed << ").\n";
        return 1;
    }

    std::cout << "----------------------------------------\n";
    std::cout << "[Training Setup]\n";
    std::cout << "Run name      : " << run_name << "\n";
    std::cout << "Seq length    : " << max_seq_len << "\n";
    std::cout << "Embed dim     : " << embed_dim << "\n";
    std::cout << "Heads         : " << num_heads << "\n";
    std::cout << "FF hidden dim : " << ff_hidden_dim << "\n";
    std::cout << "Layers        : " << num_layers << "\n";
    std::cout << "Batch size    : " << batch_size << "\n";
    std::cout << "Learning rate : " << learning_rate << "\n";
    std::cout << "Warmup steps : " << warmup_steps << "\n";
    std::cout << "Min LR       : " << min_lr << "\n";
    std::cout << "Steps         : " << num_steps << "\n";
    std::cout << "Vocab size    : " << vocab_size << "\n";
    std::cout << "Tokenizer     : " << run_tokenizer_path << "\n";
    std::cout << "Tokens file   : " << run_tokens_path << "\n";
    std::cout << "Checkpoint dir: " << run_dir << "\n";
    std::cout << "----------------------------------------\n";


    // --- Move the full token stream to device once ---
    int* d_tokens = nullptr;
    hipMalloc(&d_tokens, tokens.size() * sizeof(int));
    hipMemcpy(d_tokens, tokens.data(), tokens.size() * sizeof(int), hipMemcpyHostToDevice);

    {
        // ---- Model ----
        GPTModel model(vocab_size, max_seq_len, embed_dim, num_heads, ff_hidden_dim, num_layers);

        int* d_input_ids;
        int* d_labels;
        float* d_logits;
        float* d_loss_grad;

        hipMalloc(&d_input_ids,  total_tokens_per_batch * sizeof(int));
        hipMalloc(&d_labels,     total_tokens_per_batch * sizeof(int));
        hipMalloc(&d_logits,     total_tokens_per_batch * vocab_size * sizeof(float));
        hipMalloc(&d_loss_grad,  total_tokens_per_batch * vocab_size * sizeof(float));

        // ---- Events ----
        hipEvent_t start, stop;
        hipEventCreate(&start);
        hipEventCreate(&stop);

        hipEvent_t e_f0,e_f1,e_l0,e_l1,e_b0,e_b1;
        hipEventCreate(&e_f0); hipEventCreate(&e_f1);
        hipEventCreate(&e_l0); hipEventCreate(&e_l1);
        hipEventCreate(&e_b0); hipEventCreate(&e_b1);

        // ---- Training loop ----
        int adam_t = 0;
        int start_step = 0;

        if (args.count("--ckpt")) {
            std::string ckpt_path = args["--ckpt"];
            if (std::filesystem::exists(ckpt_path)) {
                std::cout << "Loading checkpoint from " << ckpt_path << " ..." << std::endl;
                model.load_checkpoint(ckpt_path);
                std::cout << "Checkpoint loaded successfully." << std::endl;

                // --- load step number from config JSON ---
                // derive config file from checkpoint name
                std::string ckpt_name = std::filesystem::path(ckpt_path).filename().string();
                std::string cfg_name = std::regex_replace(ckpt_name, std::regex("\\.bin$"), "_config.json");
                std::string cfg_path = std::filesystem::path(ckpt_path).parent_path() / cfg_name;
                if (std::filesystem::exists(cfg_path)) {
                    std::ifstream cfg_in(cfg_path);
                    json cfg_json;
                    cfg_in >> cfg_json;
                    start_step = cfg_json["checkpoint"]["step"];
                    std::cout << "Resuming from step " << start_step << std::endl;
                }
            } else {
                std::cerr << "Warning: checkpoint " << ckpt_path << " not found, starting fresh.\n";
            }
        }

        const int end_step = start_step + num_steps;

        long long cursor = 0; 

        // Training loop now starts at start_step
        for (int step = start_step + 1; step <= end_step; ++step) {

            // ---- Build batch ON GPU ----
            {
                int N = batch_size * max_seq_len;
                int threads = 256;
                int blocks  = (N + threads - 1) / threads;
                hipLaunchKernelGGL(build_batch_kernel_tokenwise, dim3(blocks), dim3(threads), 0, 0,
                   d_tokens, (int)tokens.size(),
                   d_input_ids, d_labels,
                   (int)(cursor % ((long long)tokens.size() - 1)), // pass int to kernel
                   batch_size, max_seq_len);
                cursor += (long long)batch_size * (long long)max_seq_len;
                long long limit = (long long)tokens.size() - 1;
                if (cursor >= limit) cursor %= limit;
            }

            hipEventRecord(start, 0);

            // ---- Forward ----
            hipEventRecord(e_f0);
            model.forward(d_input_ids, d_logits, batch_size, max_seq_len);
            hipEventRecord(e_f1); hipEventSynchronize(e_f1);
            float ms_fwd; hipEventElapsedTime(&ms_fwd, e_f0, e_f1);

            // ---- Loss + metrics ----
            hipEventRecord(e_l0);
            float loss = launch_softmax_loss(d_logits, nullptr, d_labels, d_loss_grad,
                                            total_tokens_per_batch, vocab_size);
            float acc  = launch_accuracy_from_logits(d_logits, d_labels,
                                                    total_tokens_per_batch, vocab_size);
            hipEventRecord(e_l1); hipEventSynchronize(e_l1);
            float ms_loss; hipEventElapsedTime(&ms_loss, e_l0, e_l1);

            // ---- Backward ----
            const float lr_t = lr_at(step, end_step);
            adam_t++;
            hipEventRecord(e_b0);
            model.backward(d_input_ids, d_loss_grad, batch_size, max_seq_len, lr_t, adam_t);
            hipEventRecord(e_b1); hipEventSynchronize(e_b1);
            float ms_bwd; hipEventElapsedTime(&ms_bwd, e_b0, e_b1);


            // total
            hipEventRecord(stop, 0);
            hipEventSynchronize(stop);
            float ms_total = 0.0f;
            hipEventElapsedTime(&ms_total, start, stop);

            // ---- Logging ----
            if ((step % log_every) == 0 || step == end_step) {
                const float ppl = std::exp(loss);
                const double tokens_this_step = (double)batch_size * (double)max_seq_len;
                const double tok_per_s = tokens_this_step / (ms_total / 1000.0);

                std::cout << "[Step " << step << "]"
                        << " Loss: " << loss
                        << " | Perplexity: " << ppl
                        << " | Accuracy: " << acc * 100.0f << "%"
                        << " | LR: " << lr_t
                        << " | Time: " << ms_total << " ms"
                        << " (fwd=" << ms_fwd << " ms"
                        << ", loss=" << ms_loss << " ms"
                        << ", bwd=" << ms_bwd << " ms)"
                        << " | " << tok_per_s << " tok/s"
                        << std::endl;

            }

            // ---- Checkpointing (unchanged) ----
            if (ckpt_every > 0 && step > 0 && (step % ckpt_every) == 0) {
                std::string ckpt_fname = run_dir + "/" + run_name + "_step" + std::to_string(step) + ".bin";
                std::cout << "Saving checkpoint to " << ckpt_fname << " ..." << std::endl;
                model.save_checkpoint(ckpt_fname);
                std::string cfg_fname = run_dir + "/" + run_name + "_step" + std::to_string(step) + "_config.json";
                save_config(cfg_fname,
                    vocab_size, max_seq_len, embed_dim, num_heads,
                    ff_hidden_dim, num_layers,
                    batch_size, learning_rate,
                    warmup_steps, min_lr,
                    run_tokenizer_path, run_tokens_path,
                    ckpt_fname, step);
                std::cout << "Checkpoint saved." << std::endl;

                prune_old_checkpoints(run_dir, run_name, keep_last);

                // Update symlinks (unchanged)
                std::error_code ec;
                fs::path ckpt_rel = fs::path(run_name + "_step" + std::to_string(step) + ".bin");
                fs::path cfg_rel  = fs::path(run_name + "_step" + std::to_string(step) + "_config.json");
                fs::path latest_ckpt = fs::path(run_dir) / "latest_checkpoint.bin";
                fs::path latest_cfg  = fs::path(run_dir) / "latest_config.json";
                fs::remove(latest_ckpt, ec);
                fs::remove(latest_cfg, ec);
                fs::create_symlink(ckpt_rel, latest_ckpt, ec);
                fs::create_symlink(cfg_rel, latest_cfg, ec);
            }
        }

        hipEventDestroy(start);
        hipEventDestroy(stop);
        hipEventDestroy(e_f0); hipEventDestroy(e_f1);
        hipEventDestroy(e_l0); hipEventDestroy(e_l1);
        hipEventDestroy(e_b0); hipEventDestroy(e_b1);

        int last_step = end_step;

        std::string final_ckpt = run_dir + "/" + run_name + "_step" + std::to_string(last_step) + ".bin";
        std::cout << "Saving final checkpoint to " << final_ckpt << " ..." << std::endl;
        model.save_checkpoint(final_ckpt);

        std::string final_cfg = run_dir + "/" + run_name + "_step" + std::to_string(last_step) + "_config.json";
        save_config(final_cfg,
            vocab_size, max_seq_len, embed_dim, num_heads,
            ff_hidden_dim, num_layers,
            batch_size, learning_rate,
            warmup_steps, min_lr,
            run_tokenizer_path, run_tokens_path,
            final_ckpt, last_step);


        // Create/update symlinks "latest_checkpoint.bin" and "latest_config.json"
        std::error_code ec;
        fs::path latest_ckpt = fs::path(run_dir) / "latest_checkpoint.bin";
        fs::path latest_cfg  = fs::path(run_dir) / "latest_config.json";

        fs::remove(latest_ckpt, ec);
        fs::remove(latest_cfg, ec);

        // Use relative symlinks (inside run_dir)
        fs::path ckpt_rel = fs::path(run_name + "_step" + std::to_string(last_step) + ".bin");
        fs::path cfg_rel  = fs::path(run_name + "_step" + std::to_string(last_step) + "_config.json");

        fs::create_symlink(ckpt_rel, latest_ckpt, ec);
        if (ec) {
            std::cerr << "Warning: failed to symlink " << latest_ckpt << " (" << ec.message() << ")\n";
        }

        fs::create_symlink(cfg_rel, latest_cfg, ec);
        if (ec) {
            std::cerr << "Warning: failed to symlink " << latest_cfg << " (" << ec.message() << ")\n";
        }

        std::cout << "Checkpoint saved successfully." << std::endl;


        // Ensure all GPU work done before freeing raw buffers
        std::cout << "Synchronizing device before cleanup..." << std::endl;
        hipDeviceSynchronize();

        std::cout << "Freeing device buffers..." << std::endl;
        hipFree(d_logits);
        hipFree(d_input_ids);
        hipFree(d_labels);
        hipFree(d_loss_grad);
        hipFree(d_tokens);
        std::cout << "Device buffers freed." << std::endl;

    } // model and HIP-owning objects are destroyed here

    std::cout << "Final device reset..." << std::endl;
    hipDeviceReset();
    std::cout << "Done." << std::endl;

    return 0;
}
