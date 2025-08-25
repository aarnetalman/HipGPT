// transformer_layer.h
#pragma once

#include <hip/hip_runtime.h>
#include <vector>

class TransformerLayer {
public:
    TransformerLayer(int embed_dim, int num_heads, int ff_hidden_dim);

    void forward(const float* d_input, float* d_output, int batch_size, int seq_len);
    void backward(const float* d_input, const float* d_grad_output, float* d_grad_input, int batch_size, int seq_len, float learning_rate);

private:
    int embed_dim_;
    int num_heads_;
    int head_dim_;
    int ff_hidden_dim_;

    float* d_qkv_weight_;   // [E, 3E]
    float* d_ff1_weight_;   // [E, F]
    float* d_ff2_weight_;   // [F, E]

    float* d_qkv_grad_weight_;
    float* d_ff1_grad_weight_;
    float* d_ff2_grad_weight_;

    float* d_qkv_output_;   // [B*S, 3E]
    float* d_attn_output_;  // [B*S, E]
    float* d_ff1_output_;   // [B*S, F]
    float* d_ff2_output_;   // [B*S, E]

    float* d_ff1_grad_output_; // ReLU backprop
    float* d_ff2_grad_input_;  // [B*S, F]

    float* d_attn_grad_input_; // [B*S, E]
    float* d_qkv_grad_input_;  // [B*S, 3E]

    void allocate_weights();
    void allocate_temp_buffers(int batch_size, int seq_len);

    void self_attention_forward(const float* d_input, float* d_output, int batch_size, int seq_len);
    void feed_forward_forward(const float* d_input, float* d_output, int batch_size, int seq_len);
};

