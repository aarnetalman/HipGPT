// File: gpt_model.cpp
#include "gpt_model.h"
#include <hip/hip_runtime.h>
#include <cstdlib>
#include <vector>
#include <cstring>
#include "hip_kernels.h"

GPTModel::GPTModel(int vocab_size, int max_seq_len, int embed_dim, int num_heads, int ff_hidden_dim, int num_layers)
    : vocab_size_(vocab_size),
      max_seq_len_(max_seq_len),
      embed_dim_(embed_dim),
      num_layers_(num_layers) {
    // Allocate embedding tables
    allocate_embeddings();
    allocate_output_projection();

    // Create transformer layers
    for (int i = 0; i < num_layers_; ++i) {
        layers_.emplace_back(embed_dim_, num_heads, ff_hidden_dim);
    }
}

void GPTModel::allocate_embeddings() {
    int token_embed_size = vocab_size_ * embed_dim_;
    int pos_embed_size = max_seq_len_ * embed_dim_;

    std::vector<float> token_host(token_embed_size);
    std::vector<float> pos_host(pos_embed_size);

    // Random init
    for (auto& w : token_host) w = ((float)rand() / RAND_MAX - 0.5f) * 0.02f;
    for (auto& w : pos_host) w = ((float)rand() / RAND_MAX - 0.5f) * 0.02f;

    hipMalloc(&d_token_embedding_, token_embed_size * sizeof(float));
    hipMalloc(&d_pos_embedding_, pos_embed_size * sizeof(float));

    hipMemcpy(d_token_embedding_, token_host.data(), token_embed_size * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_pos_embedding_, pos_host.data(), pos_embed_size * sizeof(float), hipMemcpyHostToDevice);
}

void GPTModel::forward(const int* d_input_ids, float* d_output, int batch_size, int seq_len) {
    int total_tokens = batch_size * seq_len;

    // Allocate embedding output
    float* d_embed_out;
    hipMalloc(&d_embed_out, total_tokens * embed_dim_ * sizeof(float));

    // Embed tokens + positions
    launch_embedding_lookup(d_input_ids, d_token_embedding_, d_pos_embedding_, d_embed_out,
                            batch_size, seq_len, vocab_size_, max_seq_len_, embed_dim_);

    // Run transformer layers
    float* d_input = d_embed_out;
    float* d_temp;
    hipMalloc(&d_temp, total_tokens * embed_dim_ * sizeof(float));

    for (int i = 0; i < num_layers_; ++i) {
        layers_[i].forward(d_input, d_temp, batch_size, seq_len);

        // Swap pointers
        std::swap(d_input, d_temp);
    }

    // Copy final output
    hipMemcpy(d_output, d_input, total_tokens * embed_dim_ * sizeof(float), hipMemcpyDeviceToDevice);

    // Cleanup
    hipFree(d_embed_out);
    hipFree(d_temp);
}
void launch_embedding_lookup(
    const int* d_token_ids,          // [B, S] – token ids
    const float* d_token_embed,      // [V, E] – token embedding matrix
    const float* d_pos_embed,        // [S, E] – positional embedding matrix
    float* d_output,                 // [B, S, E] – result
    int batch_size, int seq_len, int vocab_size, int embed_dim
);

void GPTModel::allocate_output_projection() {
    int size = embed_dim_ * vocab_size_;
    std::vector<float> host_proj(size);

    for (auto& w : host_proj) {
        w = ((float)rand() / RAND_MAX - 0.5f) * 0.02f;
    }

    hipMalloc(&d_output_proj_, size * sizeof(float));
    hipMemcpy(d_output_proj_, host_proj.data(), size * sizeof(float), hipMemcpyHostToDevice);
}

