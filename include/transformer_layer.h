#pragma once

#include <hip/hip_runtime.h>
#include <vector>
#include <string>
#include <ostream>
#include <istream>

class TransformerLayer {
public:
    TransformerLayer(int embed_dim, int num_heads, int ff_hidden_dim);
    ~TransformerLayer();

    void forward(const float* d_input, float* d_output, int batch_size, int seq_len);
    void backward(const float* d_input, const float* d_grad_output, float* d_grad_input, int batch_size, int seq_len, float lr, int adam_t);

    // Serialize/deserialize weights
    void save(std::ostream& os) const;
    void load(std::istream& is);

private:
    // Hyperparameters
    int embed_dim_;
    int num_heads_;
    int head_dim_;
    int ff_hidden_dim_;
    float dropout_p_;
    int total_tokens_ = 0;

    // --- Model Parameters and Gradients ---
    // Weights
    float* d_qkv_weight_ = nullptr;
    float* d_o_weight_ = nullptr;
    float* d_ff1_weight_ = nullptr;
    float* d_ff2_weight_ = nullptr;
    // Biases
    float* d_qkv_bias_  = nullptr;
    float* d_o_bias_ = nullptr;
    float* d_ff1_bias_  = nullptr;
    float* d_ff2_bias_  = nullptr;
    // Weight Gradients
    float* d_qkv_grad_weight_ = nullptr;
    float* d_o_grad_weight_ = nullptr;
    float* d_ff1_grad_weight_ = nullptr;
    float* d_ff2_grad_weight_ = nullptr;
    // Bias Gradients
    float* d_qkv_grad_bias_ = nullptr;
    float* d_o_grad_bias_ = nullptr;
    float* d_ff1_grad_bias_ = nullptr;
    float* d_ff2_grad_bias_ = nullptr;

    // --- Layer Normalization Parameters and Gradients ---
    float* d_attn_norm_gamma_ = nullptr;
    float* d_attn_norm_beta_ = nullptr;
    float* d_ffn_norm_gamma_ = nullptr;
    float* d_ffn_norm_beta_ = nullptr;
    float* d_attn_norm_grad_gamma_ = nullptr;
    float* d_attn_norm_grad_beta_ = nullptr;
    float* d_ffn_norm_grad_gamma_ = nullptr;
    float* d_ffn_norm_grad_beta_ = nullptr;

    // --- Adam Optimizer States ---
    float* d_qkv_m_ = nullptr; float* d_qkv_v_ = nullptr;
    float* d_o_m_ = nullptr; float* d_o_v_ = nullptr;
    float* d_o_m_bias_ = nullptr; float* d_o_v_bias_ = nullptr;
    float* d_ff1_m_ = nullptr; float* d_ff1_v_ = nullptr;
    float* d_ff2_m_ = nullptr; float* d_ff2_v_ = nullptr;
    float* d_attn_norm_m_gamma_ = nullptr; float* d_attn_norm_v_gamma_ = nullptr;
    float* d_attn_norm_m_beta_ = nullptr;  float* d_attn_norm_v_beta_ = nullptr;
    float* d_ffn_norm_m_gamma_ = nullptr;  float* d_ffn_norm_v_gamma_ = nullptr;
    float* d_ffn_norm_m_beta_ = nullptr;   float* d_ffn_norm_v_beta_ = nullptr;

    // --- Temporary Work Buffers ---
    float* d_qkv_output_ = nullptr;
    float* d_attn_output_ = nullptr;
    float* d_ff1_output_ = nullptr;
    float* d_ff2_output_ = nullptr;
    float* d_ffn_input_ = nullptr; // Buffer for the input to the FFN
    float* d_residual_input_ = nullptr;
    float* d_attn_dropout_mask_ = nullptr;
    float* d_ffn_dropout_mask_ = nullptr;
    // Buffers for backward pass
    float* d_ff1_grad_output_ = nullptr;
    float* d_ff2_grad_input_  = nullptr;
    float* d_attn_grad_input_ = nullptr;
    float* d_qkv_grad_input_ = nullptr;
    float* d_grad_attn_output_ = nullptr;
    float* d_grad_qkv_output_ = nullptr;
    
    // Private methods
    void allocate_weights();
    void allocate_temp_buffers(int batch_size, int seq_len);
    void deallocate_temp_buffers();
    void self_attention_forward(const float* d_input, float* d_output, int batch_size, int seq_len);
    void feed_forward_forward(const float* d_input, float* d_output, int batch_size, int seq_len);
};