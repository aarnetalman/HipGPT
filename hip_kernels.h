// File: hip_kernels.h
#pragma once

#include <hip/hip_runtime.h>

// Matrix multiplication and its backward pass
void launch_matmul(const float* A, const float* B, float* C, int M, int N, int K);
void launch_matmul_backward_bias(const float* A_input, const float* B_grad_out, float* C_grad_weight, float* D_grad_bias, int M, int N, int K);

// Activations and their backward pass
void launch_relu(float* A, int size);
void launch_backprop_activation(const float* act, const float* grad_out, float* grad_in, int n);

// Loss and Metrics
float launch_softmax_loss(const float* logits, float* softmax_out, const int* labels, float* grad_out, int N, int V);
float launch_accuracy(const float* d_softmax, const int* d_labels, int total_tokens, int vocab_size);

// Optimizers
void launch_sgd_update(float* weights, const float* grads, float lr, int size);
void launch_adam_update(float* weights, const float* grads, float* m, float* v, float lr, float beta1, float beta2, float epsilon, int t, int size);

// Layer Helpers
void launch_add_inplace(float* a, const float* b, int n);
void launch_mean_pool(const float* input, float* output, int B, int L, int D);

// Multi-head attention and its building blocks
void launch_multihead_attention(const float* d_qkv, float* d_output, int B, int S, int E, int H);
void launch_scaled_dot_product(const float* Q, const float* K, float* scores, int B, int S, int D);
void launch_softmax(const float* scores, float* softmax_out, int B, int S);
void launch_attention_weighted_sum(const float* softmax, const float* V, float* output, int B, int S, int D);

// Layer Normalization
void launch_layer_norm_forward(const float* input, float* output, const float* gamma, const float* beta, int N, int E);
void launch_layer_norm_backward(const float* grad_output, const float* input, float* grad_input, const float* gamma, float* grad_gamma, float* grad_beta, int N, int E);

// Dropout
void launch_dropout_forward(const float* input, float* output, float* mask, float p, int N, int E);
void launch_dropout_backward(const float* grad_output, const float* mask, float p, float* grad_input, int N, int E);

// Sampling
void launch_sample_from_logits(const float* d_logits, int* d_output_token, int vocab_size, int k, float temperature);