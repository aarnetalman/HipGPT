// File: transformer_layer.cpp
#include "transformer_layer.h"
#include <hip/hip_runtime.h>
#include "hip_kernels.h"
#include <cstdlib>   // rand
#include <cstring>   // memset

TransformerLayer::TransformerLayer(int embed_dim, int num_heads, int ff_hidden_dim)
    : embed_dim_(embed_dim),
      num_heads_(num_heads),
      ff_hidden_dim_(ff_hidden_dim),
      head_dim_(embed_dim / num_heads) {
    allocate_weights();
}

void TransformerLayer::allocate_weights() {
    int qkv_size = 3 * embed_dim_ * embed_dim_;
    int ff1_size = embed_dim_ * ff_hidden_dim_;
    int ff2_size = ff_hidden_dim_ * embed_dim_;

    std::vector<float> qkv_host(qkv_size);
    std::vector<float> ff1_host(ff1_size);
    std::vector<float> ff2_host(ff2_size);

    // Init small random weights
    for (auto& w : qkv_host) w = ((float)rand() / RAND_MAX - 0.5f) * 0.02f;
    for (auto& w : ff1_host) w = ((float)rand() / RAND_MAX - 0.5f) * 0.02f;
    for (auto& w : ff2_host) w = ((float)rand() / RAND_MAX - 0.5f) * 0.02f;

    hipMalloc(&d_qkv_weight_, qkv_size * sizeof(float));
    hipMalloc(&d_ff1_weight_, ff1_size * sizeof(float));
    hipMalloc(&d_ff2_weight_, ff2_size * sizeof(float));

    hipMemcpy(d_qkv_weight_, qkv_host.data(), qkv_size * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_ff1_weight_, ff1_host.data(), ff1_size * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_ff2_weight_, ff2_host.data(), ff2_size * sizeof(float), hipMemcpyHostToDevice);
}

void TransformerLayer::allocate_temp_buffers(int batch_size, int seq_len) {
    size_t qkv_bytes = batch_size * seq_len * 3 * embed_dim_ * sizeof(float);
    size_t attn_bytes = batch_size * seq_len * embed_dim_ * sizeof(float);
    size_t ff1_bytes = batch_size * seq_len * ff_hidden_dim_ * sizeof(float);
    size_t ff2_bytes = batch_size * seq_len * embed_dim_ * sizeof(float);

    hipMalloc(&d_qkv_output_, qkv_bytes);
    hipMalloc(&d_attn_output_, attn_bytes);
    hipMalloc(&d_ff1_output_, ff1_bytes);
    hipMalloc(&d_ff2_output_, ff2_bytes);
}

void TransformerLayer::self_attention_forward(const float* d_input, float* d_output, int batch_size, int seq_len) {
    int total_tokens = batch_size * seq_len;

    // Step 1: QKV projection
    // Input: [B*S, E], Weights: [E, 3E] → Output: [B*S, 3E]
    launch_matmul(
        d_input,         // [B*S, E]
        d_qkv_weight_,   // [E, 3E]
        d_qkv_output_,   // [B*S, 3E]
        total_tokens,    // M = B*S
        embed_dim_,      // N = E
        3 * embed_dim_   // K = 3E
    );

    // Step 2: Compute scaled dot-product attention (kernel does split internally)
    // Input: d_qkv_output_ → computes attn(Q,K,V) → d_attn_output_
    launch_multihead_attention(
        d_qkv_output_,
        d_attn_output_,
        batch_size,
        seq_len,
        embed_dim_,
        num_heads_
    );

    // Step 3: Final linear projection
    // [B*S, E] x [E, E] → [B*S, E]
    launch_matmul(
        d_attn_output_,
        d_qkv_weight_ + 2 * embed_dim_ * embed_dim_, // reuse W_o = part of qkv_weight for simplicity (can separate later)
        d_output,
        total_tokens,
        embed_dim_,
        embed_dim_
    );
}

void TransformerLayer::feed_forward_forward(const float* d_input, float* d_output, int batch_size, int seq_len) {
    int total_tokens = batch_size * seq_len;

    // FF1: [B*S, E] x [E, F] → [B*S, F]
    launch_matmul(
        d_input,        // [B*S, E]
        d_ff1_weight_,  // [E, F]
        d_ff1_output_,  // [B*S, F]
        total_tokens,
        embed_dim_,
        ff_hidden_dim_
    );

    // ReLU on intermediate
    launch_relu(d_ff1_output_, total_tokens * ff_hidden_dim_);

    // FF2: [B*S, F] x [F, E] → [B*S, E]
    launch_matmul(
        d_ff1_output_,  // [B*S, F]
        d_ff2_weight_,  // [F, E]
        d_output,       // [B*S, E]
        total_tokens,
        ff_hidden_dim_,
        embed_dim_
    );
}


void TransformerLayer::forward(const float* d_input, float* d_output, int batch_size, int seq_len) {
    allocate_temp_buffers(batch_size, seq_len);
    int total_tokens = batch_size * seq_len;

    // Step 1: Self-Attention
    self_attention_forward(d_input, d_attn_output_, batch_size, seq_len);

    // Residual connection: output = input + attn
    hipMemcpy(d_output, d_input, total_tokens * embed_dim_ * sizeof(float), hipMemcpyDeviceToDevice);
    // d_output += d_attn_output_ (in-place)
    launch_add_inplace(d_output, d_attn_output_, total_tokens * embed_dim_);

    // Step 2: Feed-forward
    feed_forward_forward(d_output, d_ff2_output_, batch_size, seq_len);

    // Residual connection: output = output + ff2
    launch_add_inplace(d_output, d_ff2_output_, total_tokens * embed_dim_);
}
