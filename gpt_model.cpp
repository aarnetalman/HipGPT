// File: gpt_model.cpp
#include "gpt_model.h"
#include <hip/hip_runtime.h>
#include <cstdlib>
#include <vector>
#include <cstring>
#include <fstream>
#include "hip_kernels.h"


GPTModel::GPTModel(int vocab_size, int max_seq_len, int embed_dim, int num_heads, int ff_hidden_dim, int num_layers)
    : vocab_size_(vocab_size),
      max_seq_len_(max_seq_len),
      embed_dim_(embed_dim),
      num_layers_(num_layers) {
    allocate_embeddings();
    allocate_output_projection();
    for (int i = 0; i < num_layers_; ++i) {
        layers_.emplace_back(embed_dim_, num_heads, ff_hidden_dim);
    }
}

void GPTModel::allocate_embeddings() {
    int token_embed_size = vocab_size_ * embed_dim_;
    int pos_embed_size = max_seq_len_ * embed_dim_;

    std::vector<float> token_host(token_embed_size);
    std::vector<float> pos_host(pos_embed_size);

    for (auto& w : token_host) w = ((float)rand() / RAND_MAX - 0.5f) * 0.02f;
    for (auto& w : pos_host) w = ((float)rand() / RAND_MAX - 0.5f) * 0.02f;

    hipMalloc(&d_token_embedding_, token_embed_size * sizeof(float));
    hipMalloc(&d_pos_embedding_, pos_embed_size * sizeof(float));

    hipMemcpy(d_token_embedding_, token_host.data(), token_embed_size * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_pos_embedding_, pos_host.data(), pos_embed_size * sizeof(float), hipMemcpyHostToDevice);
}

void GPTModel::allocate_output_projection() {
    int proj_size = embed_dim_ * vocab_size_;
    std::vector<float> proj_host(proj_size);
    for (auto& w : proj_host) w = ((float)rand() / RAND_MAX - 0.5f) * 0.02f;

    hipMalloc(&d_output_proj_, proj_size * sizeof(float));
    hipMalloc(&d_output_proj_grad_, proj_size * sizeof(float));

    hipMemcpy(d_output_proj_, proj_host.data(), proj_size * sizeof(float), hipMemcpyHostToDevice);
}

void GPTModel::forward(const int* d_input_ids, float* d_logits, int batch_size, int seq_len) {
    int total_tokens = batch_size * seq_len;
    float* d_embed_out;
    hipMalloc(&d_embed_out, total_tokens * embed_dim_ * sizeof(float));

    launch_embedding_lookup(
        d_input_ids,
        d_token_embedding_,
        d_pos_embedding_,
        d_embed_out,
        batch_size, seq_len,
        vocab_size_, embed_dim_
    );

    float* d_temp;
    hipMalloc(&d_temp, total_tokens * embed_dim_ * sizeof(float));
    float* d_input = d_embed_out;

    for (int i = 0; i < num_layers_; ++i) {
        layers_[i].forward(d_input, d_temp, batch_size, seq_len);
        std::swap(d_input, d_temp);
    }

    launch_matmul(
        d_input,
        d_output_proj_,
        d_logits,
        total_tokens,
        embed_dim_,
        vocab_size_
    );

    hipFree(d_embed_out);
    hipFree(d_temp);
}

void GPTModel::backward(const int* d_input_ids, const float* d_logits_grad, int batch_size, int seq_len, float learning_rate) {
    int total_tokens = batch_size * seq_len;
    float* d_embed_out;
    hipMalloc(&d_embed_out, total_tokens * embed_dim_ * sizeof(float));

    launch_embedding_lookup(
        d_input_ids,
        d_token_embedding_,
        d_pos_embedding_,
        d_embed_out,
        batch_size, seq_len,
        vocab_size_, embed_dim_
    );

    std::vector<float*> layer_inputs(num_layers_ + 1);
    layer_inputs[0] = d_embed_out;

    float* d_temp;
    hipMalloc(&d_temp, total_tokens * embed_dim_ * sizeof(float));
    float* d_input = d_embed_out;

    for (int i = 0; i < num_layers_; ++i) {
        layers_[i].forward(d_input, d_temp, batch_size, seq_len);
        std::swap(d_input, d_temp);
        layer_inputs[i + 1] = d_input;
    }

    // Compute output projection gradient
    launch_matmul(
        layer_inputs.back(),         // [B*S, E]
        d_logits_grad,               // [B*S, V]
        d_output_proj_grad_,         // [E, V]
        embed_dim_,
        total_tokens,
        vocab_size_
    );

    // SGD update for final projection
    launch_sgd_update(d_output_proj_, d_output_proj_grad_, learning_rate, embed_dim_ * vocab_size_);

    // Compute gradient w.r.t. last layer output
    float* d_last_grad;
    hipMalloc(&d_last_grad, total_tokens * embed_dim_ * sizeof(float));

    launch_matmul(
        d_logits_grad,               // [B*S, V]
        d_output_proj_,              // [V, E]
        d_last_grad,                 // [B*S, E]
        total_tokens,
        vocab_size_,
        embed_dim_
    );

    // Backprop through layers in reverse
    for (int i = num_layers_ - 1; i >= 0; --i) {
        layers_[i].backward(layer_inputs[i], d_last_grad, batch_size, seq_len, learning_rate);
    }

    hipFree(d_embed_out);
    hipFree(d_temp);
    hipFree(d_last_grad);
}

void GPTModel::save_checkpoint(const std::string& path) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        std::cerr << "Failed to open checkpoint file: " << path << std::endl;
        return;
    }

    // Helper lambda to dump weights
    auto dump = [&](float* device_ptr, size_t count) {
        std::vector<float> buffer(count);
        hipMemcpy(buffer.data(), device_ptr, count * sizeof(float), hipMemcpyDeviceToHost);
        out.write(reinterpret_cast<const char*>(buffer.data()), count * sizeof(float));
    };

    // Save embeddings
    dump(d_token_embedding_, vocab_size_ * embed_dim_);
    dump(d_pos_embedding_, max_seq_len_ * embed_dim_);

    // Save output projection
    dump(d_output_proj_, embed_dim_ * vocab_size_);

    // Save each transformer's weights
    for (auto* layer : layers_) {
        dump(layer->d_qkv_weight_, 3 * embed_dim_ * embed_dim_);
        dump(layer->d_ff1_weight_, embed_dim_ * layer->ff_hidden_dim_);
        dump(layer->d_ff2_weight_, layer->ff_hidden_dim_ * embed_dim_);
    }

    out.close();
    std::cout << "Checkpoint saved to " << path << std::endl;
}
std::vector<int> GPTModel::generate(const std::vector<int>& prompt_ids, int max_new_tokens) {
    std::vector<int> output = prompt_ids;

    int vocab_size = vocab_size_;
    int embed_dim = embed_dim_;

    int* d_input_ids;
    float* d_logits;

    hipMalloc(&d_input_ids, max_seq_len_ * sizeof(int));
    hipMalloc(&d_logits, max_seq_len_ * vocab_size * sizeof(float));

    for (int step = 0; step < max_new_tokens; ++step) {
        int cur_len = std::min((int)output.size(), max_seq_len_);
        std::vector<int> input_ids(max_seq_len_, 0);
        std::copy(output.end() - cur_len, output.end(), input_ids.end() - cur_len);

        hipMemcpy(d_input_ids, input_ids.data(), max_seq_len_ * sizeof(int), hipMemcpyHostToDevice);

        forward(d_input_ids, d_logits, 1, max_seq_len_);

        std::vector<float> h_logits(vocab_size);
        hipMemcpy(h_logits.data(), &d_logits[(max_seq_len_ - 1) * vocab_size], vocab_size * sizeof(float), hipMemcpyDeviceToHost);

        // Argmax
        int max_idx = std::max_element(h_logits.begin(), h_logits.end()) - h_logits.begin();
        output.push_back(max_idx);
    }

    hipFree(d_input_ids);
    hipFree(d_logits);

    return output;
}
