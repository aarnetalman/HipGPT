// File: hip_kernels.h
#pragma once

#include <hip/hip_runtime.h>

// Matrix multiplication kernel
void launch_matmul(const float* A, const float* B, float* C, int M, int N, int K);

// ReLU
void launch_relu(float* A, int size);
void launch_backprop_activation(const float* input, const float* grad_output, float* grad_input, int size);

// Softmax and loss
float launch_softmax_loss(const float* logits, float* softmax_out, const int* labels, float* grad_out, int B, int C);

// Accuracy
int count_correct_predictions(const float* softmax, const int* labels, int B);

// SGD Update
void launch_sgd_update(float* weights, const float* grads, float lr, int size);

// Multi-head attention
void launch_multihead_attention(
    const float* d_qkv, float* d_output,
    int batch_size, int seq_len,
    int embed_dim, int num_heads
);

// Attention building blocks
void launch_scaled_dot_product(
    const float* Q, const float* K,
    float* scores,
    int total_tokens, int seq_len, int head_dim
);

void launch_softmax(
    const float* scores,
    float* softmax_out,
    int total_tokens, int seq_len
);

void launch_attention_weighted_sum(
    const float* softmax,
    const float* V,
    float* output,
    int total_tokens, int seq_len, int head_dim
);
