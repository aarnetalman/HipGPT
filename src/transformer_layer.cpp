// File: transformer_layer.cpp
#include "transformer_layer.h"
#include "hip_kernels.h"
#include <hip/hip_runtime.h>
#include <cstdlib>
#include <vector>
#include <ostream>
#include <istream>
#include <stdexcept>


TransformerLayer::TransformerLayer(int embed_dim, int num_heads, int ff_hidden_dim)
    : embed_dim_(embed_dim), num_heads_(num_heads), ff_hidden_dim_(ff_hidden_dim), head_dim_(embed_dim / num_heads), dropout_p_(0.1f) {
    // Allocate all weights and associated optimizer buffers
    allocate_weights();
}

TransformerLayer::~TransformerLayer() {
    // Free weights
    if (d_qkv_weight_) hipFree(d_qkv_weight_);
    if (d_o_weight_) hipFree(d_o_weight_);
    if (d_ff1_weight_) hipFree(d_ff1_weight_);
    if (d_ff2_weight_) hipFree(d_ff2_weight_);

    // Free biases
    if (d_qkv_bias_) hipFree(d_qkv_bias_);
    if (d_o_bias_) hipFree(d_o_bias_);
    if (d_ff1_bias_) hipFree(d_ff1_bias_);
    if (d_ff2_bias_) hipFree(d_ff2_bias_);

    // Free weight gradients
    if (d_qkv_grad_weight_) hipFree(d_qkv_grad_weight_);
    if (d_o_grad_weight_) hipFree(d_o_grad_weight_);
    if (d_ff1_grad_weight_) hipFree(d_ff1_grad_weight_);
    if (d_ff2_grad_weight_) hipFree(d_ff2_grad_weight_);
    
    // Free bias gradients
    if (d_qkv_grad_bias_) hipFree(d_qkv_grad_bias_);
    if (d_o_grad_bias_) hipFree(d_o_grad_bias_);
    if (d_ff1_grad_bias_) hipFree(d_ff1_grad_bias_);
    if (d_ff2_grad_bias_) hipFree(d_ff2_grad_bias_);

    // Free Adam states for weights
    if (d_qkv_m_) hipFree(d_qkv_m_); if (d_qkv_v_) hipFree(d_qkv_v_);
    if (d_o_m_) hipFree(d_o_m_); if (d_o_v_) hipFree(d_o_v_);
    if (d_o_m_bias_) hipFree(d_o_m_bias_); if (d_o_v_bias_) hipFree(d_o_v_bias_);
    if (d_ff1_m_) hipFree(d_ff1_m_); if (d_ff1_v_) hipFree(d_ff1_v_);
    if (d_ff2_m_) hipFree(d_ff2_m_); if (d_ff2_v_) hipFree(d_ff2_v_);

    // Free LayerNorm weights and gradients
    if (d_attn_norm_gamma_) hipFree(d_attn_norm_gamma_);
    if (d_attn_norm_beta_) hipFree(d_attn_norm_beta_);
    if (d_ffn_norm_gamma_) hipFree(d_ffn_norm_gamma_);
    if (d_ffn_norm_beta_) hipFree(d_ffn_norm_beta_);
    if (d_attn_norm_grad_gamma_) hipFree(d_attn_norm_grad_gamma_);
    if (d_attn_norm_grad_beta_) hipFree(d_attn_norm_grad_beta_);
    if (d_ffn_norm_grad_gamma_) hipFree(d_ffn_norm_grad_gamma_);
    if (d_ffn_norm_grad_beta_) hipFree(d_ffn_norm_grad_beta_);

    // Free Adam states for LayerNorm
    if (d_attn_norm_m_gamma_) hipFree(d_attn_norm_m_gamma_); if (d_attn_norm_v_gamma_) hipFree(d_attn_norm_v_gamma_);
    if (d_attn_norm_m_beta_) hipFree(d_attn_norm_m_beta_);  if (d_attn_norm_v_beta_) hipFree(d_attn_norm_v_beta_);
    if (d_ffn_norm_m_gamma_) hipFree(d_ffn_norm_m_gamma_);  if (d_ffn_norm_v_gamma_) hipFree(d_ffn_norm_v_gamma_);
    if (d_ffn_norm_m_beta_) hipFree(d_ffn_norm_m_beta_);   if (d_ffn_norm_v_beta_) hipFree(d_ffn_norm_v_beta_);
    
    // Free all temporary buffers
    deallocate_temp_buffers();
}

void TransformerLayer::deallocate_temp_buffers() {
    if (d_qkv_output_) { hipFree(d_qkv_output_); d_qkv_output_ = nullptr; }
    if (d_attn_output_) { hipFree(d_attn_output_); d_attn_output_ = nullptr; }
    if (d_ff1_output_) { hipFree(d_ff1_output_); d_ff1_output_ = nullptr; }
    if (d_ff2_output_) { hipFree(d_ff2_output_); d_ff2_output_ = nullptr; }
    if (d_ffn_input_) { hipFree(d_ffn_input_); d_ffn_input_ = nullptr; }
    if (d_ff1_grad_output_) { hipFree(d_ff1_grad_output_); d_ff1_grad_output_ = nullptr; }
    if (d_ff2_grad_input_) { hipFree(d_ff2_grad_input_); d_ff2_grad_input_ = nullptr; }
    if (d_attn_grad_input_) { hipFree(d_attn_grad_input_); d_attn_grad_input_ = nullptr; }
    if (d_qkv_grad_input_) { hipFree(d_qkv_grad_input_); d_qkv_grad_input_ = nullptr; }
    if (d_attn_dropout_mask_) { hipFree(d_attn_dropout_mask_); d_attn_dropout_mask_ = nullptr; }
    if (d_ffn_dropout_mask_) { hipFree(d_ffn_dropout_mask_); d_ffn_dropout_mask_ = nullptr; }
    if (d_residual_input_) { hipFree(d_residual_input_); d_residual_input_ = nullptr; }
    if (d_grad_attn_output_) { hipFree(d_grad_attn_output_); d_grad_attn_output_ = nullptr; }
    if (d_grad_qkv_output_) { hipFree(d_grad_qkv_output_); d_grad_qkv_output_ = nullptr; }
}

void TransformerLayer::allocate_weights() {
    // Define sizes
    int qkv_w_size = embed_dim_ * 3 * embed_dim_;
    int o_w_size = embed_dim_ * embed_dim_;
    int ff1_w_size = embed_dim_ * ff_hidden_dim_;
    int ff2_w_size = ff_hidden_dim_ * embed_dim_;

    int qkv_b_size = 3 * embed_dim_;
    int o_b_size = embed_dim_;
    int ff1_b_size = ff_hidden_dim_;
    int ff2_b_size = embed_dim_;

    // Helper lambda for random initialization
    auto rand_init = [](float* d_ptr, int size) {
        std::vector<float> h_buf(size);
        for (auto& val : h_buf) val = ((float)rand() / RAND_MAX - 0.5f) * 0.02f;
        hipMemcpy(d_ptr, h_buf.data(), size * sizeof(float), hipMemcpyHostToDevice);
    };

    // Allocate and initialize QKV weights, grads, and Adam states
    hipMalloc(&d_qkv_weight_, qkv_w_size * sizeof(float));
    hipMalloc(&d_qkv_bias_, qkv_b_size * sizeof(float));
    hipMalloc(&d_qkv_grad_weight_, qkv_w_size * sizeof(float));
    hipMalloc(&d_qkv_grad_bias_, qkv_b_size * sizeof(float));
    hipMalloc(&d_qkv_m_, qkv_w_size * sizeof(float));
    hipMalloc(&d_qkv_v_, qkv_w_size * sizeof(float));
    rand_init(d_qkv_weight_, qkv_w_size);
    hipMemset(d_qkv_bias_, 0, qkv_b_size * sizeof(float));
    hipMemset(d_qkv_grad_weight_, 0, qkv_w_size * sizeof(float));
    hipMemset(d_qkv_grad_bias_, 0, qkv_b_size * sizeof(float));
    hipMemset(d_qkv_m_, 0, qkv_w_size * sizeof(float));
    hipMemset(d_qkv_v_, 0, qkv_w_size * sizeof(float));

    // Allocate and initialize Output Projection (Wo) weights, grads, and Adam states
    hipMalloc(&d_o_weight_, o_w_size * sizeof(float));
    hipMalloc(&d_o_bias_, o_b_size * sizeof(float));
    hipMalloc(&d_o_grad_weight_, o_w_size * sizeof(float));
    hipMalloc(&d_o_grad_bias_, o_b_size * sizeof(float));
    hipMalloc(&d_o_m_, o_w_size * sizeof(float));
    hipMalloc(&d_o_v_, o_w_size * sizeof(float));
    hipMalloc(&d_o_m_bias_, o_b_size * sizeof(float));
    hipMalloc(&d_o_v_bias_, o_b_size * sizeof(float));
    rand_init(d_o_weight_, o_w_size);
    hipMemset(d_o_bias_, 0, o_b_size * sizeof(float));
    hipMemset(d_o_grad_weight_, 0, o_w_size * sizeof(float));
    hipMemset(d_o_grad_bias_, 0, o_b_size * sizeof(float));
    hipMemset(d_o_m_, 0, o_w_size * sizeof(float));
    hipMemset(d_o_v_, 0, o_w_size * sizeof(float));
    hipMemset(d_o_m_bias_, 0, o_b_size * sizeof(float));
    hipMemset(d_o_v_bias_, 0, o_b_size * sizeof(float));

    // Allocate and initialize FF1 weights, grads, and Adam states
    hipMalloc(&d_ff1_weight_, ff1_w_size * sizeof(float));
    hipMalloc(&d_ff1_bias_, ff1_b_size * sizeof(float));
    hipMalloc(&d_ff1_grad_weight_, ff1_w_size * sizeof(float));
    hipMalloc(&d_ff1_grad_bias_, ff1_b_size * sizeof(float));
    hipMalloc(&d_ff1_m_, ff1_w_size * sizeof(float));
    hipMalloc(&d_ff1_v_, ff1_w_size * sizeof(float));
    rand_init(d_ff1_weight_, ff1_w_size);
    hipMemset(d_ff1_bias_, 0, ff1_b_size * sizeof(float));
    hipMemset(d_ff1_grad_weight_, 0, ff1_w_size * sizeof(float));
    hipMemset(d_ff1_grad_bias_, 0, ff1_b_size * sizeof(float));
    hipMemset(d_ff1_m_, 0, ff1_w_size * sizeof(float));
    hipMemset(d_ff1_v_, 0, ff1_w_size * sizeof(float));

    // Allocate and initialize FF2 weights, grads, and Adam states
    hipMalloc(&d_ff2_weight_, ff2_w_size * sizeof(float));
    hipMalloc(&d_ff2_bias_, ff2_b_size * sizeof(float));
    hipMalloc(&d_ff2_grad_weight_, ff2_w_size * sizeof(float));
    hipMalloc(&d_ff2_grad_bias_, ff2_b_size * sizeof(float));
    hipMalloc(&d_ff2_m_, ff2_w_size * sizeof(float));
    hipMalloc(&d_ff2_v_, ff2_w_size * sizeof(float));
    rand_init(d_ff2_weight_, ff2_w_size);
    hipMemset(d_ff2_bias_, 0, ff2_b_size * sizeof(float));
    hipMemset(d_ff2_grad_weight_, 0, ff2_w_size * sizeof(float));
    hipMemset(d_ff2_grad_bias_, 0, ff2_b_size * sizeof(float));
    hipMemset(d_ff2_m_, 0, ff2_w_size * sizeof(float));
    hipMemset(d_ff2_v_, 0, ff2_w_size * sizeof(float));
    
    // Allocate and initialize LayerNorm parameters, grads, and Adam states
    std::vector<float> ones(embed_dim_, 1.0f);
    std::vector<float> zeros(embed_dim_, 0.0f);
    hipMalloc(&d_attn_norm_gamma_, embed_dim_ * sizeof(float)); hipMemcpy(d_attn_norm_gamma_, ones.data(), embed_dim_ * sizeof(float), hipMemcpyHostToDevice);
    hipMalloc(&d_attn_norm_beta_, embed_dim_ * sizeof(float));  hipMemcpy(d_attn_norm_beta_, zeros.data(), embed_dim_ * sizeof(float), hipMemcpyHostToDevice);
    hipMalloc(&d_ffn_norm_gamma_, embed_dim_ * sizeof(float));  hipMemcpy(d_ffn_norm_gamma_, ones.data(), embed_dim_ * sizeof(float), hipMemcpyHostToDevice);
    hipMalloc(&d_ffn_norm_beta_, embed_dim_ * sizeof(float));   hipMemcpy(d_ffn_norm_beta_, zeros.data(), embed_dim_ * sizeof(float), hipMemcpyHostToDevice);

    hipMalloc(&d_attn_norm_grad_gamma_, embed_dim_ * sizeof(float)); hipMemset(d_attn_norm_grad_gamma_, 0, embed_dim_ * sizeof(float));
    hipMalloc(&d_attn_norm_grad_beta_, embed_dim_ * sizeof(float));  hipMemset(d_attn_norm_grad_beta_, 0, embed_dim_ * sizeof(float));
    hipMalloc(&d_ffn_norm_grad_gamma_, embed_dim_ * sizeof(float));  hipMemset(d_ffn_norm_grad_gamma_, 0, embed_dim_ * sizeof(float));
    hipMalloc(&d_ffn_norm_grad_beta_, embed_dim_ * sizeof(float));   hipMemset(d_ffn_norm_grad_beta_, 0, embed_dim_ * sizeof(float));

    hipMalloc(&d_attn_norm_m_gamma_, embed_dim_ * sizeof(float)); hipMemset(d_attn_norm_m_gamma_, 0, embed_dim_ * sizeof(float));
    hipMalloc(&d_attn_norm_v_gamma_, embed_dim_ * sizeof(float)); hipMemset(d_attn_norm_v_gamma_, 0, embed_dim_ * sizeof(float));
    hipMalloc(&d_attn_norm_m_beta_, embed_dim_ * sizeof(float));  hipMemset(d_attn_norm_m_beta_, 0, embed_dim_ * sizeof(float));
    hipMalloc(&d_attn_norm_v_beta_, embed_dim_ * sizeof(float));  hipMemset(d_attn_norm_v_beta_, 0, embed_dim_ * sizeof(float));
    hipMalloc(&d_ffn_norm_m_gamma_, embed_dim_ * sizeof(float));  hipMemset(d_ffn_norm_m_gamma_, 0, embed_dim_ * sizeof(float));
    hipMalloc(&d_ffn_norm_v_gamma_, embed_dim_ * sizeof(float));  hipMemset(d_ffn_norm_v_gamma_, 0, embed_dim_ * sizeof(float));
    hipMalloc(&d_ffn_norm_m_beta_, embed_dim_ * sizeof(float));   hipMemset(d_ffn_norm_m_beta_, 0, embed_dim_ * sizeof(float));
    hipMalloc(&d_ffn_norm_v_beta_, embed_dim_ * sizeof(float));   hipMemset(d_ffn_norm_v_beta_, 0, embed_dim_ * sizeof(float));
}

void TransformerLayer::allocate_temp_buffers(int batch_size, int seq_len) {
    int total_tokens = batch_size * seq_len;
    deallocate_temp_buffers(); // Clear old buffers first to prevent leaks

    hipMalloc(&d_qkv_output_, total_tokens * 3 * embed_dim_ * sizeof(float));
    hipMalloc(&d_attn_output_, total_tokens * embed_dim_ * sizeof(float));
    hipMalloc(&d_ff1_output_, total_tokens * ff_hidden_dim_ * sizeof(float));
    hipMalloc(&d_ff2_output_, total_tokens * embed_dim_ * sizeof(float));
    hipMalloc(&d_ffn_input_, total_tokens * embed_dim_ * sizeof(float));
    hipMalloc(&d_ff1_grad_output_, total_tokens * ff_hidden_dim_ * sizeof(float));
    hipMalloc(&d_ff2_grad_input_, total_tokens * embed_dim_ * sizeof(float));
    hipMalloc(&d_attn_grad_input_, total_tokens * embed_dim_ * sizeof(float));
    hipMalloc(&d_qkv_grad_input_, total_tokens * embed_dim_ * sizeof(float));
    hipMalloc(&d_attn_dropout_mask_, total_tokens * embed_dim_ * sizeof(float));
    hipMalloc(&d_ffn_dropout_mask_, total_tokens * embed_dim_ * sizeof(float));
    hipMalloc(&d_residual_input_, total_tokens * embed_dim_ * sizeof(float));
    hipMalloc(&d_grad_attn_output_, total_tokens * embed_dim_ * sizeof(float));
    hipMalloc(&d_grad_qkv_output_, total_tokens * 3 * embed_dim_ * sizeof(float));

    total_tokens_ = total_tokens;
}

void TransformerLayer::self_attention_forward(const float* d_input, float* d_output, int batch_size, int seq_len) {
    int total_tokens = batch_size * seq_len;

    launch_matmul_add_bias(d_input, d_qkv_weight_, d_qkv_bias_, d_qkv_output_, total_tokens, embed_dim_, 3 * embed_dim_);
    launch_multihead_attention(d_qkv_output_, d_attn_output_, batch_size, seq_len, embed_dim_, num_heads_);
    launch_matmul_add_bias(d_attn_output_, d_o_weight_, d_o_bias_, d_output, total_tokens, embed_dim_, embed_dim_);
}

void TransformerLayer::feed_forward_forward(const float* d_input, float* d_output, int batch_size, int seq_len) {
    int total_tokens = batch_size * seq_len;

    launch_matmul_add_bias(d_input, d_ff1_weight_, d_ff1_bias_, d_ff1_output_, total_tokens, embed_dim_, ff_hidden_dim_);
    launch_relu(d_ff1_output_, total_tokens * ff_hidden_dim_);
    launch_matmul_add_bias(d_ff1_output_, d_ff2_weight_, d_ff2_bias_, d_output, total_tokens, ff_hidden_dim_, embed_dim_);
}

void TransformerLayer::forward(const float* d_input, float* d_output, int batch_size, int seq_len) {
    if (total_tokens_ != batch_size * seq_len) {
        allocate_temp_buffers(batch_size, seq_len);
    }
    
    int total_tokens = total_tokens_;

    // --- Self-Attention Block (Sub-layer 1) ---
    launch_layer_norm_forward(d_input, d_residual_input_, d_attn_norm_gamma_, d_attn_norm_beta_, total_tokens, embed_dim_);
    self_attention_forward(d_residual_input_, d_attn_output_, batch_size, seq_len);
    launch_dropout_forward(d_attn_output_, d_attn_output_, d_attn_dropout_mask_, dropout_p_, total_tokens, embed_dim_);
    hipMemcpy(d_output, d_input, total_tokens * embed_dim_ * sizeof(float), hipMemcpyDeviceToDevice);
    launch_add_inplace(d_output, d_attn_output_, total_tokens * embed_dim_);

    // --- Feed-Forward Block (Sub-layer 2) ---
    hipMemcpy(d_residual_input_, d_output, total_tokens * embed_dim_ * sizeof(float), hipMemcpyDeviceToDevice);
    launch_layer_norm_forward(d_residual_input_, d_ffn_input_, d_ffn_norm_gamma_, d_ffn_norm_beta_, total_tokens, embed_dim_);
    feed_forward_forward(d_ffn_input_, d_ff2_output_, batch_size, seq_len);
    launch_dropout_forward(d_ff2_output_, d_ff2_output_, d_ffn_dropout_mask_, dropout_p_, total_tokens, embed_dim_);
    hipMemcpy(d_output, d_residual_input_, total_tokens * embed_dim_ * sizeof(float), hipMemcpyDeviceToDevice);
    launch_add_inplace(d_output, d_ff2_output_, total_tokens * embed_dim_);
}

void TransformerLayer::backward(const float* d_input, const float* d_grad_output, float* d_grad_input, int batch_size, int seq_len, float lr) {
    int total_tokens = total_tokens_;
    int step_t = 1; // This should ideally be tracked globally
    
    // --- Backprop through Feed-Forward Block (Sub-layer 2) ---
    hipMemcpy(d_grad_input, d_grad_output, total_tokens * embed_dim_ * sizeof(float), hipMemcpyDeviceToDevice);
    
    float* d_ffn_path_grad = d_ff2_grad_input_;
    launch_dropout_backward(d_grad_output, d_ffn_dropout_mask_, dropout_p_, d_ffn_path_grad, total_tokens, embed_dim_);
    
    launch_matmul_backward_bias(d_ff1_output_, d_ffn_path_grad, d_ff2_grad_weight_, d_ff2_grad_bias_, total_tokens, embed_dim_, ff_hidden_dim_);
    launch_matmul_transpose_B(d_ffn_path_grad, d_ff2_weight_, d_ff1_grad_output_, total_tokens, embed_dim_, ff_hidden_dim_);
    
    launch_backprop_activation(d_ff1_output_, d_ff1_grad_output_, d_ff1_grad_output_, total_tokens * ff_hidden_dim_);
    
    launch_matmul_backward_bias(d_residual_input_, d_ff1_grad_output_, d_ff1_grad_weight_, d_ff1_grad_bias_, total_tokens, ff_hidden_dim_, embed_dim_);
    launch_matmul_transpose_B(d_ff1_grad_output_, d_ff1_weight_, d_ffn_path_grad, total_tokens, ff_hidden_dim_, embed_dim_);
    
    launch_layer_norm_backward(d_ffn_path_grad, d_residual_input_, d_ffn_path_grad, d_ffn_norm_gamma_, d_ffn_norm_grad_gamma_, d_ffn_norm_grad_beta_, total_tokens, embed_dim_);
    
    launch_add_inplace(d_grad_input, d_ffn_path_grad, total_tokens * embed_dim_);

    // --- Backprop through Self-Attention Block (Sub-layer 1) ---
    float* d_attn_path_grad = d_attn_grad_input_;
    hipMemcpy(d_attn_path_grad, d_grad_input, total_tokens * embed_dim_ * sizeof(float), hipMemcpyDeviceToDevice);
    
    launch_dropout_backward(d_attn_path_grad, d_attn_dropout_mask_, dropout_p_, d_attn_path_grad, total_tokens, embed_dim_);
    
    launch_matmul_backward_bias(d_attn_output_, d_attn_path_grad, d_o_grad_weight_, d_o_grad_bias_, total_tokens, embed_dim_, embed_dim_);
    launch_matmul_transpose_B(d_attn_path_grad, d_o_weight_, d_grad_attn_output_, total_tokens, embed_dim_, embed_dim_);

    launch_multihead_attention_backward(d_grad_attn_output_, d_qkv_output_, nullptr, d_grad_qkv_output_, batch_size, seq_len, embed_dim_, num_heads_);

    launch_matmul_transpose_A(d_residual_input_, d_grad_qkv_output_, d_qkv_grad_weight_, total_tokens, 3 * embed_dim_, embed_dim_);
    launch_matmul_transpose_B(d_grad_qkv_output_, d_qkv_weight_, d_qkv_grad_input_, total_tokens, 3 * embed_dim_, embed_dim_);
    
    launch_layer_norm_backward(d_qkv_grad_input_, d_input, d_qkv_grad_input_, d_attn_norm_gamma_, d_attn_norm_grad_gamma_, d_attn_norm_grad_beta_, total_tokens, embed_dim_);

    launch_add_inplace(d_grad_input, d_qkv_grad_input_, total_tokens * embed_dim_);

    // --- Apply Adam updates to all weights and parameters ---
    launch_adam_update(d_ff2_weight_, d_ff2_grad_weight_, d_ff2_m_, d_ff2_v_, lr, 0.9f, 0.999f, 1e-8f, step_t, ff_hidden_dim_ * embed_dim_);
    launch_adam_update(d_ff1_weight_, d_ff1_grad_weight_, d_ff1_m_, d_ff1_v_, lr, 0.9f, 0.999f, 1e-8f, step_t, embed_dim_ * ff_hidden_dim_);
    launch_adam_update(d_qkv_weight_, d_qkv_grad_weight_, d_qkv_m_, d_qkv_v_, lr, 0.9f, 0.999f, 1e-8f, step_t, embed_dim_ * 3 * embed_dim_);
    launch_adam_update(d_o_weight_, d_o_grad_weight_, d_o_m_, d_o_v_, lr, 0.9f, 0.999f, 1e-8f, step_t, embed_dim_ * embed_dim_);
    
    launch_adam_update(d_qkv_bias_, d_qkv_grad_bias_, d_qkv_m_, d_qkv_v_, lr, 0.9f, 0.999f, 1e-8f, step_t, 3 * embed_dim_);
    launch_adam_update(d_o_bias_, d_o_grad_bias_, d_o_m_bias_, d_o_v_bias_, lr, 0.9f, 0.999f, 1e-8f, step_t, embed_dim_);
    launch_adam_update(d_ff1_bias_, d_ff1_grad_bias_, d_ff1_m_, d_ff1_v_, lr, 0.9f, 0.999f, 1e-8f, step_t, ff_hidden_dim_);
    launch_adam_update(d_ff2_bias_, d_ff2_grad_bias_, d_ff2_m_, d_ff2_v_, lr, 0.9f, 0.999f, 1e-8f, step_t, embed_dim_);

    launch_adam_update(d_ffn_norm_gamma_, d_ffn_norm_grad_gamma_, d_ffn_norm_m_gamma_, d_ffn_norm_v_gamma_, lr, 0.9f, 0.999f, 1e-8f, step_t, embed_dim_);
    launch_adam_update(d_ffn_norm_beta_, d_ffn_norm_grad_beta_, d_ffn_norm_m_beta_, d_ffn_norm_v_beta_, lr, 0.9f, 0.999f, 1e-8f, step_t, embed_dim_);
    launch_adam_update(d_attn_norm_gamma_, d_attn_norm_grad_gamma_, d_attn_norm_m_gamma_, d_attn_norm_v_gamma_, lr, 0.9f, 0.999f, 1e-8f, step_t, embed_dim_);
    launch_adam_update(d_attn_norm_beta_, d_attn_norm_grad_beta_, d_attn_norm_m_beta_, d_attn_norm_v_beta_, lr, 0.9f, 0.999f, 1e-8f, step_t, embed_dim_);
}

void TransformerLayer::save(std::ostream& os) const {
    const int qkv_w_size = embed_dim_ * 3 * embed_dim_;
    const int o_w_size = embed_dim_ * embed_dim_;
    const int ff1_w_size = embed_dim_ * ff_hidden_dim_;
    const int ff2_w_size = ff_hidden_dim_ * embed_dim_;
    const int qkv_b_size = 3 * embed_dim_;
    const int o_b_size = embed_dim_;
    const int ff1_b_size = ff_hidden_dim_;
    const int ff2_b_size = embed_dim_;

    std::vector<float> h_qkv_weight(qkv_w_size), h_o_weight(o_w_size), h_ff1_weight(ff1_w_size), h_ff2_weight(ff2_w_size);
    std::vector<float> h_qkv_bias(qkv_b_size), h_o_bias(o_b_size), h_ff1_bias(ff1_b_size), h_ff2_bias(ff2_b_size);
    std::vector<float> h_attn_norm_gamma(embed_dim_), h_attn_norm_beta(embed_dim_), h_ffn_norm_gamma(embed_dim_), h_ffn_norm_beta(embed_dim_);

    hipMemcpy(h_qkv_weight.data(), d_qkv_weight_, qkv_w_size * sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(h_o_weight.data(), d_o_weight_, o_w_size * sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(h_ff1_weight.data(), d_ff1_weight_, ff1_w_size * sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(h_ff2_weight.data(), d_ff2_weight_, ff2_w_size * sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(h_qkv_bias.data(), d_qkv_bias_, qkv_b_size * sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(h_o_bias.data(), d_o_bias_, o_b_size * sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(h_ff1_bias.data(), d_ff1_bias_, ff1_b_size * sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(h_ff2_bias.data(), d_ff2_bias_, ff2_b_size * sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(h_attn_norm_gamma.data(), d_attn_norm_gamma_, embed_dim_ * sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(h_attn_norm_beta.data(), d_attn_norm_beta_, embed_dim_ * sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(h_ffn_norm_gamma.data(), d_ffn_norm_gamma_, embed_dim_ * sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(h_ffn_norm_beta.data(), d_ffn_norm_beta_, embed_dim_ * sizeof(float), hipMemcpyDeviceToHost);

    os.write(reinterpret_cast<const char*>(&embed_dim_), sizeof(embed_dim_));
    os.write(reinterpret_cast<const char*>(&num_heads_), sizeof(num_heads_));
    os.write(reinterpret_cast<const char*>(&ff_hidden_dim_), sizeof(ff_hidden_dim_));

    os.write(reinterpret_cast<const char*>(h_qkv_weight.data()), h_qkv_weight.size() * sizeof(float));
    os.write(reinterpret_cast<const char*>(h_o_weight.data()), h_o_weight.size() * sizeof(float));
    os.write(reinterpret_cast<const char*>(h_ff1_weight.data()), h_ff1_weight.size() * sizeof(float));
    os.write(reinterpret_cast<const char*>(h_ff2_weight.data()), h_ff2_weight.size() * sizeof(float));
    os.write(reinterpret_cast<const char*>(h_qkv_bias.data()), h_qkv_bias.size() * sizeof(float));
    os.write(reinterpret_cast<const char*>(h_o_bias.data()), h_o_bias.size() * sizeof(float));
    os.write(reinterpret_cast<const char*>(h_ff1_bias.data()), h_ff1_bias.size() * sizeof(float));
    os.write(reinterpret_cast<const char*>(h_ff2_bias.data()), h_ff2_bias.size() * sizeof(float));
    os.write(reinterpret_cast<const char*>(h_attn_norm_gamma.data()), h_attn_norm_gamma.size() * sizeof(float));
    os.write(reinterpret_cast<const char*>(h_attn_norm_beta.data()), h_attn_norm_beta.size() * sizeof(float));
    os.write(reinterpret_cast<const char*>(h_ffn_norm_gamma.data()), h_ffn_norm_gamma.size() * sizeof(float));
    os.write(reinterpret_cast<const char*>(h_ffn_norm_beta.data()), h_ffn_norm_beta.size() * sizeof(float));
}

void TransformerLayer::load(std::istream& is) {
    int e = 0, h = 0, f = 0;
    is.read(reinterpret_cast<char*>(&e), sizeof(e));
    is.read(reinterpret_cast<char*>(&h), sizeof(h));
    is.read(reinterpret_cast<char*>(&f), sizeof(f));

    if (e != embed_dim_ || h != num_heads_ || f != ff_hidden_dim_) {
        throw std::runtime_error("TransformerLayer::load: dimension mismatch");
    }

    const int qkv_w_size = embed_dim_ * 3 * embed_dim_;
    const int o_w_size = embed_dim_ * embed_dim_;
    const int ff1_w_size = embed_dim_ * ff_hidden_dim_;
    const int ff2_w_size = ff_hidden_dim_ * embed_dim_;
    const int qkv_b_size = 3 * embed_dim_;
    const int o_b_size = embed_dim_;
    const int ff1_b_size = ff_hidden_dim_;
    const int ff2_b_size = embed_dim_;

    std::vector<float> h_qkv_weight(qkv_w_size), h_o_weight(o_w_size), h_ff1_weight(ff1_w_size), h_ff2_weight(ff2_w_size);
    std::vector<float> h_qkv_bias(qkv_b_size), h_o_bias(o_b_size), h_ff1_bias(ff1_b_size), h_ff2_bias(ff2_b_size);
    std::vector<float> h_attn_norm_gamma(embed_dim_), h_attn_norm_beta(embed_dim_), h_ffn_norm_gamma(embed_dim_), h_ffn_norm_beta(embed_dim_);

    is.read(reinterpret_cast<char*>(h_qkv_weight.data()), h_qkv_weight.size() * sizeof(float));
    is.read(reinterpret_cast<char*>(h_o_weight.data()), h_o_weight.size() * sizeof(float));
    is.read(reinterpret_cast<char*>(h_ff1_weight.data()), h_ff1_weight.size() * sizeof(float));
    is.read(reinterpret_cast<char*>(h_ff2_weight.data()), h_ff2_weight.size() * sizeof(float));
    is.read(reinterpret_cast<char*>(h_qkv_bias.data()), h_qkv_bias.size() * sizeof(float));
    is.read(reinterpret_cast<char*>(h_o_bias.data()), h_o_bias.size() * sizeof(float));
    is.read(reinterpret_cast<char*>(h_ff1_bias.data()), h_ff1_bias.size() * sizeof(float));
    is.read(reinterpret_cast<char*>(h_ff2_bias.data()), h_ff2_bias.size() * sizeof(float));
    is.read(reinterpret_cast<char*>(h_attn_norm_gamma.data()), h_attn_norm_gamma.size() * sizeof(float));
    is.read(reinterpret_cast<char*>(h_attn_norm_beta.data()), h_attn_norm_beta.size() * sizeof(float));
    is.read(reinterpret_cast<char*>(h_ffn_norm_gamma.data()), h_ffn_norm_gamma.size() * sizeof(float));
    is.read(reinterpret_cast<char*>(h_ffn_norm_beta.data()), h_ffn_norm_beta.size() * sizeof(float));

    hipMemcpy(d_qkv_weight_, h_qkv_weight.data(), h_qkv_weight.size() * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_o_weight_, h_o_weight.data(), h_o_weight.size() * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_ff1_weight_, h_ff1_weight.data(), h_ff1_weight.size() * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_ff2_weight_, h_ff2_weight.data(), h_ff2_weight.size() * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_qkv_bias_, h_qkv_bias.data(), h_qkv_bias.size() * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_o_bias_, h_o_bias.data(), h_o_bias.size() * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_ff1_bias_, h_ff1_bias.data(), h_ff1_bias.size() * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_ff2_bias_, h_ff2_bias.data(), h_ff2_bias.size() * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_attn_norm_gamma_, h_attn_norm_gamma.data(), h_attn_norm_gamma.size() * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_attn_norm_beta_, h_attn_norm_beta.data(), h_attn_norm_beta.size() * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_ffn_norm_gamma_, h_ffn_norm_gamma.data(), h_ffn_norm_gamma.size() * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_ffn_norm_beta_, h_ffn_norm_beta.data(), h_ffn_norm_beta.size() * sizeof(float), hipMemcpyHostToDevice)