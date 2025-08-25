
// transformer_layer.cpp
#include "transformer_layer.h"
#include "hip_kernels.h"
#include <hip/hip_runtime.h>
#include <cstdlib>

TransformerLayer::TransformerLayer(int embed_dim, int num_heads, int ff_hidden_dim)
    : embed_dim_(embed_dim), num_heads_(num_heads), ff_hidden_dim_(ff_hidden_dim), head_dim_(embed_dim / num_heads) {
    allocate_weights();
}

void TransformerLayer::allocate_weights() {
    int qkv_size = embed_dim_ * 3 * embed_dim_;
    int ff1_size = embed_dim_ * ff_hidden_dim_;
    int ff2_size = ff_hidden_dim_ * embed_dim_;

    std::vector<float> qkv_host(qkv_size);
    std::vector<float> ff1_host(ff1_size);
    std::vector<float> ff2_host(ff2_size);

    for (auto& w : qkv_host) w = ((float)rand() / RAND_MAX - 0.5f) * 0.02f;
    for (auto& w : ff1_host) w = ((float)rand() / RAND_MAX - 0.5f) * 0.02f;
    for (auto& w : ff2_host) w = ((float)rand() / RAND_MAX - 0.5f) * 0.02f;

    hipMalloc(&d_qkv_weight_, qkv_size * sizeof(float));
    hipMalloc(&d_ff1_weight_, ff1_size * sizeof(float));
    hipMalloc(&d_ff2_weight_, ff2_size * sizeof(float));

    hipMemcpy(d_qkv_weight_, qkv_host.data(), qkv_size * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_ff1_weight_, ff1_host.data(), ff1_size * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_ff2_weight_, ff2_host.data(), ff2_size * sizeof(float), hipMemcpyHostToDevice);

    // Allocate grads
    hipMalloc(&d_ff1_grad_weight_, ff1_size * sizeof(float));
    hipMalloc(&d_ff2_grad_weight_, ff2_size * sizeof(float));
    hipMalloc(&d_qkv_grad_weight_, qkv_size * sizeof(float));
}

void TransformerLayer::allocate_temp_buffers(int batch_size, int seq_len) {
    int total_tokens = batch_size * seq_len;

    hipMalloc(&d_qkv_output_, total_tokens * 3 * embed_dim_ * sizeof(float));
    hipMalloc(&d_attn_output_, total_tokens * embed_dim_ * sizeof(float));
    hipMalloc(&d_ff1_output_, total_tokens * ff_hidden_dim_ * sizeof(float));
    hipMalloc(&d_ff2_output_, total_tokens * embed_dim_ * sizeof(float));

    hipMalloc(&d_ff1_grad_output_, total_tokens * ff_hidden_dim_ * sizeof(float));
    hipMalloc(&d_ff2_grad_input_, total_tokens * ff_hidden_dim_ * sizeof(float));
    hipMalloc(&d_attn_grad_input_, total_tokens * embed_dim_ * sizeof(float));
    hipMalloc(&d_qkv_grad_input_, total_tokens * 3 * embed_dim_ * sizeof(float));
}

void TransformerLayer::self_attention_forward(const float* d_input, float* d_output, int batch_size, int seq_len) {
    int total_tokens = batch_size * seq_len;

    launch_matmul(
        d_input,
        d_qkv_weight_,
        d_qkv_output_,
        total_tokens,
        embed_dim_,
        3 * embed_dim_
    );

    launch_multihead_attention(
        d_qkv_output_,
        d_attn_output_,
        batch_size,
        seq_len,
        embed_dim_,
        num_heads_
    );

    launch_matmul(
        d_attn_output_,
        d_qkv_weight_ + 2 * embed_dim_ * embed_dim_,
        d_output,
        total_tokens,
        embed_dim_,
        embed_dim_
    );
}

void TransformerLayer::feed_forward_forward(const float* d_input, float* d_output, int batch_size, int seq_len) {
    int total_tokens = batch_size * seq_len;

    launch_matmul(
        d_input,
        d_ff1_weight_,
        d_ff1_output_,
        total_tokens,
        embed_dim_,
        ff_hidden_dim_
    );

    launch_relu(d_ff1_output_, total_tokens * ff_hidden_dim_);

    launch_matmul(
        d_ff1_output_,
        d_ff2_weight_,
        d_output,
        total_tokens,
        ff_hidden_dim_,
        embed_dim_
    );
}

void TransformerLayer::forward(const float* d_input, float* d_output, int batch_size, int seq_len) {
    allocate_temp_buffers(batch_size, seq_len);
    int total_tokens = batch_size * seq_len;

    self_attention_forward(d_input, d_attn_output_, batch_size, seq_len);

    hipMemcpy(d_output, d_input, total_tokens * embed_dim_ * sizeof(float), hipMemcpyDeviceToDevice);
    launch_add_inplace(d_output, d_attn_output_, total_tokens * embed_dim_);

    feed_forward_forward(d_output, d_ff2_output_, batch_size, seq_len);
    launch_add_inplace(d_output, d_ff2_output_, total_tokens * embed_dim_);
}

void TransformerLayer::backward(const float* d_input, const float* d_grad_output, float* d_grad_input, int batch_size, int seq_len, float lr) {
    int total_tokens = batch_size * seq_len;

    // FF2
    launch_matmul(d_ff1_output_, d_grad_output, d_ff2_grad_weight_, ff_hidden_dim_, total_tokens, embed_dim_);
    launch_matmul(d_grad_output, d_ff2_weight_, d_ff2_grad_input_, total_tokens, embed_dim_, ff_hidden_dim_);

    // ReLU backprop
    launch_backprop_activation(d_ff1_output_, d_ff2_grad_input_, d_ff1_grad_output_, total_tokens * ff_hidden_dim_);

    // FF1
    launch_matmul(d_input, d_ff1_grad_output_, d_ff1_grad_weight_, embed_dim_, total_tokens, ff_hidden_dim_);

    // Update FF weights
    launch_sgd_update(d_ff2_weight_, d_ff2_grad_weight_, lr, ff_hidden_dim_ * embed_dim_);
    launch_sgd_update(d_ff1_weight_, d_ff1_grad_weight_, lr, embed_dim_ * ff_hidden_dim_);

    // Attention W_o backprop: d_attn_output_ is input, d_grad_output is grad
    float* d_qkv_w_o = d_qkv_weight_ + 2 * embed_dim_ * embed_dim_;
    launch_matmul(d_attn_output_, d_grad_output, d_qkv_grad_weight_ + 2 * embed_dim_ * embed_dim_, embed_dim_, total_tokens, embed_dim_);
    launch_matmul(d_grad_output, d_qkv_w_o, d_attn_grad_input_, total_tokens, embed_dim_, embed_dim_);

    // Backprop QKV weights (only top-level gradient, not deep attention chain here)
    launch_matmul(d_input, d_qkv_output_, d_qkv_grad_weight_, embed_dim_, total_tokens, 3 * embed_dim_);
    launch_sgd_update(d_qkv_weight_, d_qkv_grad_weight_, lr, 3 * embed_dim_ * embed_dim_);

    // Residual: grad_input = grad_output + attn_back + ff2_back
    hipMemcpy(d_grad_input, d_grad_output, total_tokens * embed_dim_ * sizeof(float), hipMemcpyDeviceToDevice);
    launch_add_inplace(d_grad_input, d_attn_grad_input_, total_tokens * embed_dim_);
    // FF2 backprop already applied to weights, not included in upstream grad here
}
