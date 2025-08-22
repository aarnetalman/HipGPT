// transformer_layer.h
#pragma once

#include <hip/hip_runtime.h>
#include <vector>

class TransformerLayer {
public:
    TransformerLayer(int embed_dim, int num_heads, int ff_hidden_dim);

    void forward(const float* d_input, float* d_output, int batch_size, int seq_len);

private:
    int embed_dim_;
    int num_heads_;
    int head_dim_;
    int ff_hidden_dim_;

    // Weight matrices (device pointers)
    float* d_qkv_weight_;   // [3 * embed_dim, embed_dim]
    float* d_ff1_weight_;   // [embed_dim, ff_hidden_dim]
    float* d_ff2_weight_;   // [ff_hidden_dim, embed_dim]

    // Biases (optional for now)
    float* d_qkv_bias_;
    float* d_ff1_bias_;
    float* d_ff2_bias_;

    // Temporary buffers (optional optimization)
    float* d_qkv_output_;
    float* d_attn_output_;
    float* d_ff1_output_;
    float* d_ff2_output_;

    void allocate_weights();
    void allocate_temp_buffers(int batch_size, int seq_len);

    void self_attention_forward(const float* d_input, float* d_output, int batch_size, int seq_len);
    void feed_forward_forward(const float* d_input, float* d_output, int batch_size, int seq_len);
};
