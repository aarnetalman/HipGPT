// File: hip_kernels.cpp
#include "hip_kernels.h"
#include <hip/hip_runtime.h>
#include <cmath>

// ---------------- Mean Pooling ----------------
__global__ void mean_pool_kernel_optimized(const float* input, float* output, int B, int L, int D) {
    int batch = blockIdx.x;
    int d = blockIdx.y * blockDim.x + threadIdx.x;
    if (d >= D) return;

    float sum = 0.0f;
    for (int l = 0; l < L; ++l) {
        sum += input[batch * L * D + l * D + d];
    }
    output[batch * D + d] = sum / L;
}

void launch_mean_pool(const float* input, float* output, int B, int L, int D) {
    int threads = 256;
    int blocks_x = B;
    int blocks_y = (D + threads - 1) / threads;
    dim3 blocks(blocks_x, blocks_y);
    dim3 thread_dim(threads);
    hipLaunchKernelGGL(mean_pool_kernel_optimized, blocks, thread_dim, 0, 0, input, output, B, L, D);
}

// ---------------- MatMul ----------------
__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= K) return;

    float sum = 0.0f;
    for (int i = 0; i < N; ++i) {
        sum += A[row * N + i] * B[i * K + col];
    }
    C[row * K + col] = sum;
}

void launch_matmul(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threads(16, 16);
    dim3 blocks((K + 15) / 16, (M + 15) / 16);
    hipLaunchKernelGGL(matmul_kernel, blocks, threads, 0, 0, A, B, C, M, N, K);
}

// ---------------- ReLU ----------------
__global__ void relu_kernel(float* A, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        A[idx] = fmaxf(0.0f, A[idx]);
}

void launch_relu(float* A, int size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    hipLaunchKernelGGL(relu_kernel, dim3(blocks), dim3(threads), 0, 0, A, size);
}

// ---------------- Scaled Dot Product Attention ----------------
__global__ void scaled_dot_product_kernel(const float* Q, const float* K, float* scores, int B, int S, int D) {
    int token = blockIdx.x * blockDim.x + threadIdx.x;
    if (token >= B * S) return;

    for (int j = 0; j < S; ++j) {
        float sum = 0.0f;
        for (int d = 0; d < D; ++d) {
            sum += Q[token * D + d] * K[(token / S) * S * D + j * D + d];
        }
        scores[token * S + j] = sum / sqrtf((float)D);
    }
}

void launch_scaled_dot_product(const float* Q, const float* K, float* scores, int B, int S, int D) {
    int total = B * S;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    hipLaunchKernelGGL(scaled_dot_product_kernel, dim3(blocks), dim3(threads), 0, 0, Q, K, scores, B, S, D);
}

// ---------------- Softmax ----------------
__global__ void softmax_kernel(const float* scores, float* softmax_out, int B, int S) {
    int token = blockIdx.x * blockDim.x + threadIdx.x;
    if (token >= B * S) return;

    float max_score = -1e9f;
    for (int i = 0; i < S; ++i) {
        float val = scores[token * S + i];
        if (val > max_score) max_score = val;
    }

    float sum_exp = 0.0f;
    for (int i = 0; i < S; ++i) {
        softmax_out[token * S + i] = expf(scores[token * S + i] - max_score);
        sum_exp += softmax_out[token * S + i];
    }

    for (int i = 0; i < S; ++i) {
        softmax_out[token * S + i] /= sum_exp;
    }
}

void launch_softmax(const float* scores, float* softmax_out, int B, int S) {
    int total = B * S;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    hipLaunchKernelGGL(softmax_kernel, dim3(blocks), dim3(threads), 0, 0, scores, softmax_out, B, S);
}

// ---------------- Attention Weighted Sum ----------------
__global__ void attention_weighted_sum_kernel(const float* softmax, const float* V, float* output, int B, int S, int D) {
    int token = blockIdx.x * blockDim.x + threadIdx.x;
    if (token >= B * S) return;

    for (int d = 0; d < D; ++d) {
        float sum = 0.0f;
        for (int j = 0; j < S; ++j) {
            sum += softmax[token * S + j] * V[(token / S) * S * D + j * D + d];
        }
        output[token * D + d] = sum;
    }
}

void launch_attention_weighted_sum(const float* softmax, const float* V, float* output, int B, int S, int D) {
    int total = B * S;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    hipLaunchKernelGGL(attention_weighted_sum_kernel, dim3(blocks), dim3(threads), 0, 0, softmax, V, output, B, S, D);
}

// ---------------- Multi-Head Attention Orchestrator ----------------
void launch_multihead_attention(const float* d_qkv, float* d_output, int B, int S, int E, int H) {
    int head_dim = E / H;
    int total_tokens = B * S;

    const float* d_Q = d_qkv;
    const float* d_K = d_qkv + total_tokens * E;
    const float* d_V = d_qkv + 2 * total_tokens * E;

    float* d_scores;
    hipMalloc(&d_scores, total_tokens * S * sizeof(float));
    float* d_softmax;
    hipMalloc(&d_softmax, total_tokens * S * sizeof(float));

    for (int h = 0; h < H; ++h) {
        const float* Q_h = d_Q + h * head_dim;
        const float* K_h = d_K + h * head_dim;
        const float* V_h = d_V + h * head_dim;
        float* out_h = d_output + h * head_dim;

        launch_scaled_dot_product(Q_h, K_h, d_scores, B, S, head_dim);
        launch_softmax(d_scores, d_softmax, B, S);
        launch_attention_weighted_sum(d_softmax, V_h, out_h, B, S, head_dim);
    }

    hipFree(d_scores);
    hipFree(d_softmax);
}

__global__ void embedding_lookup_kernel(
    const int* token_ids,           // [B, S]
    const float* token_embed,       // [V, E]
    const float* pos_embed,         // [S, E]
    float* output,                  // [B, S, E]
    int B, int S, int V, int E
) {
    int b = blockIdx.x;
    int s = blockIdx.y;
    int e = threadIdx.x;
    if (b >= B || s >= S || e >= E) return;

    int token_id = token_ids[b * S + s];
    if (token_id >= V) return; // safety

    float token_val = token_embed[token_id * E + e];
    float pos_val = pos_embed[s * E + e];

    output[(b * S + s) * E + e] = token_val + pos_val;
}

void launch_embedding_lookup(
    const int* d_token_ids,
    const float* d_token_embed,
    const float* d_pos_embed,
    float* d_output,
    int batch_size, int seq_len, int vocab_size, int embed_dim
) {
    dim3 blocks(batch_size, seq_len);
    int threads = embed_dim; // ideally ≤ 1024
    launch_bounds_check(threads, "embedding_lookup_kernel");

    hipLaunchKernelGGL(
        embedding_lookup_kernel,
        blocks, dim3(threads),
        0, 0,
        d_token_ids,
        d_token_embed,
        d_pos_embed,
        d_output,
        batch_size, seq_len, vocab_size, embed_dim
    );
}

// ---------------- Softmax + Cross-Entropy Loss ----------------
__global__ void softmax_loss_kernel(
    const float* logits,
    float* softmax_out,
    const int* labels,
    float* grad_out,
    float* loss_sum,
    int N, int V
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Find max for numerical stability
    float max_logit = -1e9f;
    for (int i = 0; i < V; ++i) {
        float val = logits[idx * V + i];
        if (val > max_logit) max_logit = val;
    }

    // Compute softmax and cross-entropy
    float sum_exp = 0.0f;
    for (int i = 0; i < V; ++i) {
        softmax_out[idx * V + i] = expf(logits[idx * V + i] - max_logit);
        sum_exp += softmax_out[idx * V + i];
    }

    float log_sum = logf(sum_exp);
    int label = labels[idx];
    float log_prob = logits[idx * V + label] - max_logit - log_sum;

    atomicAdd(loss_sum, -log_prob);  // accumulate negative log likelihood

    // Compute gradient: softmax - one-hot(label)
    for (int i = 0; i < V; ++i) {
        float prob = softmax_out[idx * V + i] / sum_exp;
        softmax_out[idx * V + i] = prob;
        grad_out[idx * V + i] = prob - (i == label ? 1.0f : 0.0f);
    }
}

float launch_softmax_loss(
    const float* logits,
    float* softmax_out,
    const int* labels,
    float* grad_out,
    int B, int C
) {
    int N = B;  // B = batch_size * seq_len
    int V = C;  // C = vocab_size

    float* d_loss_sum;
    hipMalloc(&d_loss_sum, sizeof(float));
    hipMemset(d_loss_sum, 0, sizeof(float));

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    hipLaunchKernelGGL(
        softmax_loss_kernel,
        dim3(blocks), dim3(threads), 0, 0,
        logits,
        softmax_out,
        labels,
        grad_out,
        d_loss_sum,
        N, V
    );

    float loss;
    hipMemcpy(&loss, d_loss_sum, sizeof(float), hipMemcpyDeviceToHost);
    hipFree(d_loss_sum);

    return loss / N;  // return average loss
}

__global__ void accuracy_kernel(const float* softmax, const int* labels, int* correct, int N, int V) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    const float* row = softmax + idx * V;
    int pred = 0;
    float max_val = row[0];
    for (int i = 1; i < V; ++i) {
        if (row[i] > max_val) {
            max_val = row[i];
            pred = i;
        }
    }

    if (pred == labels[idx]) atomicAdd(correct, 1);
}

float launch_accuracy(const float* d_softmax, const int* d_labels, int total_tokens, int vocab_size) {
    int threads = 256;
    int blocks = (total_tokens + threads - 1) / threads;

    int* d_correct;
    hipMalloc(&d_correct, sizeof(int));
    hipMemset(d_correct, 0, sizeof(int));

    hipLaunchKernelGGL(accuracy_kernel, dim3(blocks), dim3(threads), 0, 0,
                       d_softmax, d_labels, d_correct, total_tokens, vocab_size);

    int h_correct = 0;
    hipMemcpy(&h_correct, d_correct, sizeof(int), hipMemcpyDeviceToHost);
    hipFree(d_correct);

    return static_cast<float>(h_correct) / total_tokens;
}
