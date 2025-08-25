// File: hip_kernels.cpp
#include "hip_kernels.h"
#include <hip/hip_runtime.h>
#include <cmath>
#include <climits> // for UINT_MAX
#include <iostream>

// Add debugging macros
#define HIP_CHECK(call) do { \
    hipError_t err = call; \
    if (err != hipSuccess) { \
        std::cerr << "HIP Error at " << __FILE__ << ":" << __LINE__ \
                  << " - " << hipGetErrorString(err) << std::endl; \
        exit(1); \
    } \
} while(0)

#define HIP_CHECK_KERNEL(kernel_name) do { \
    hipError_t err = hipGetLastError(); \
    if (err != hipSuccess) { \
        std::cerr << "HIP Kernel Launch Error (" << kernel_name << ") at " \
                  << __FILE__ << ":" << __LINE__ \
                  << " - " << hipGetErrorString(err) << std::endl; \
        exit(1); \
    } \
    err = hipDeviceSynchronize(); \
    if (err != hipSuccess) { \
        std::cerr << "HIP Kernel Execution Error (" << kernel_name << ") at " \
                  << __FILE__ << ":" << __LINE__ \
                  << " - " << hipGetErrorString(err) << std::endl; \
        exit(1); \
    } \
} while(0)

#define MHA_THREADS_PER_BLOCK 256

// Simple XOR-WOW pseudo-random number generator state
struct xorwow_state {
    unsigned int a, b, c, d;
    unsigned int counter;
};

// The random number generation function for the GPU
__device__ unsigned int xorwow(struct xorwow_state *state) {
    unsigned int t = state->d;
    unsigned int s = state->a;
    state->d = state->c;
    state->c = state->b;
    state->b = s;
    t ^= t >> 2;
    t ^= t << 1;
    t ^= s ^ (s << 4);
    state->a = t;
    state->counter++;
    return t + state->counter;
}

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
    std::cout << "Launching mean_pool_kernel: B=" << B << ", L=" << L << ", D=" << D << std::endl;
    int threads = 256;
    int blocks_x = B;
    int blocks_y = (D + threads - 1) / threads;
    dim3 blocks(blocks_x, blocks_y);
    dim3 thread_dim(threads);
    hipLaunchKernelGGL(mean_pool_kernel_optimized, blocks, thread_dim, 0, 0, input, output, B, L, D);
    HIP_CHECK_KERNEL("mean_pool_kernel_optimized");
    std::cout << "mean_pool_kernel completed successfully" << std::endl;
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
    std::cout << "Launching matmul_kernel: M=" << M << ", N=" << N << ", K=" << K << std::endl;
    dim3 threads(16, 16);
    dim3 blocks((K + 15) / 16, (M + 15) / 16);
    hipLaunchKernelGGL(matmul_kernel, blocks, threads, 0, 0, A, B, C, M, N, K);
    HIP_CHECK_KERNEL("matmul_kernel");
    std::cout << "matmul_kernel completed successfully" << std::endl;
}

__global__ void matmul_add_bias_kernel(const float* A, const float* B, const float* bias, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= K) return;

    float sum = 0.0f;
    for (int i = 0; i < N; ++i) {
        sum += A[row * N + i] * B[i * K + col];
    }
    C[row * K + col] = sum + bias[col];
}

void launch_matmul_add_bias(const float* A, const float* B, const float* bias, float* C, int M, int N, int K) {
    std::cout << "Launching matmul_add_bias_kernel: M=" << M << ", N=" << N << ", K=" << K << std::endl;
    dim3 threads(16, 16);
    dim3 blocks((K + 15) / 16, (M + 15) / 16);
    hipLaunchKernelGGL(matmul_add_bias_kernel, blocks, threads, 0, 0, A, B, bias, C, M, N, K);
    HIP_CHECK_KERNEL("matmul_add_bias_kernel");
    std::cout << "matmul_add_bias_kernel completed successfully" << std::endl;
}

// ---------------- Matmul Backward with Bias (Corrected) ----------------
__global__ void matmul_backward_weight_kernel(const float* A_input, const float* B_grad_out, float* C_grad_weight, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // K
    int col = blockIdx.x * blockDim.x + threadIdx.x; // N
    if (row < K && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < M; ++i) {
            sum += A_input[i * K + row] * B_grad_out[i * N + col];
        }
        C_grad_weight[row * N + col] = sum;
    }
}

__global__ void bias_backward_kernel(const float* B_grad_out, float* D_grad_bias, int M, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x; // N
    if (col < N) {
        float sum = 0.0f;
        for (int i = 0; i < M; ++i) sum += B_grad_out[i * N + col];
        D_grad_bias[col] = sum;
    }
}

void launch_matmul_backward_bias(const float* A_input, const float* B_grad_out, float* C_grad_weight, float* D_grad_bias, int M, int N, int K) {
    std::cout << "Launching matmul_backward_bias kernels: M=" << M << ", N=" << N << ", K=" << K << std::endl;
    dim3 threads_w(16, 16);
    dim3 blocks_w((N + 15) / 16, (K + 15) / 16);
    hipLaunchKernelGGL(matmul_backward_weight_kernel, blocks_w, threads_w, 0, 0, A_input, B_grad_out, C_grad_weight, M, N, K);
    HIP_CHECK_KERNEL("matmul_backward_weight_kernel");

    int threads_b = 256;
    int blocks_b = (N + threads_b - 1) / threads_b;
    hipLaunchKernelGGL(bias_backward_kernel, dim3(blocks_b), dim3(threads_b), 0, 0, B_grad_out, D_grad_bias, M, N);
    HIP_CHECK_KERNEL("bias_backward_kernel");
    std::cout << "matmul_backward_bias kernels completed successfully" << std::endl;
}

// ---------------- ReLU ----------------
__global__ void relu_kernel(float* A, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) A[idx] = fmaxf(0.0f, A[idx]);
}

void launch_relu(float* A, int size) {
    std::cout << "Launching relu_kernel: size=" << size << std::endl;
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    hipLaunchKernelGGL(relu_kernel, dim3(blocks), dim3(threads), 0, 0, A, size);
    HIP_CHECK_KERNEL("relu_kernel");
    std::cout << "relu_kernel completed successfully" << std::endl;
}

// ---------------- Scaled Dot Product Attention ----------------
__global__ void scaled_dot_product_kernel(const float* Q, const float* K, float* scores, int B, int S, int D) {
    int token = blockIdx.x * blockDim.x + threadIdx.x;
    if (token >= B * S) return;
    const float inv = rsqrtf((float)D);
    for (int j = 0; j < S; ++j) {
        float sum = 0.0f;
        for (int d = 0; d < D; ++d) {
            sum += Q[token * D + d] * K[(token / S) * S * D + j * D + d];
        }
        scores[token * S + j] = sum * inv;
    }
}

void launch_scaled_dot_product(const float* Q, const float* K, float* scores, int B, int S, int D) {
    std::cout << "Launching scaled_dot_product_kernel: B=" << B << ", S=" << S << ", D=" << D << std::endl;
    int total = B * S;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    hipLaunchKernelGGL(scaled_dot_product_kernel, dim3(blocks), dim3(threads), 0, 0, Q, K, scores, B, S, D);
    HIP_CHECK_KERNEL("scaled_dot_product_kernel");
    std::cout << "scaled_dot_product_kernel completed successfully" << std::endl;
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
    std::cout << "Launching softmax_kernel: B=" << B << ", S=" << S << std::endl;
    int total = B * S;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    hipLaunchKernelGGL(softmax_kernel, dim3(blocks), dim3(threads), 0, 0, scores, softmax_out, B, S);
    HIP_CHECK_KERNEL("softmax_kernel");
    std::cout << "softmax_kernel completed successfully" << std::endl;
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
    std::cout << "Launching attention_weighted_sum_kernel: B=" << B << ", S=" << S << ", D=" << D << std::endl;
    int total = B * S;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    hipLaunchKernelGGL(attention_weighted_sum_kernel, dim3(blocks), dim3(threads), 0, 0, softmax, V, output, B, S, D);
    HIP_CHECK_KERNEL("attention_weighted_sum_kernel");
    std::cout << "attention_weighted_sum_kernel completed successfully" << std::endl;
}

// Fixed Multi-Head Attention Forward Kernel
__global__ void multihead_attention_kernel(
    const float* Q, const float* K, const float* V, float* output,
    int B, int S, int E, int H
) {
    int token_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int head_idx = blockIdx.y;
    
    if (token_idx >= B * S || head_idx >= H) return;
    
    int head_dim = E / H;
    const float scale = 1.0f / sqrtf((float)head_dim);
    
    int batch_idx = token_idx / S;
    int seq_pos = token_idx % S;
    
    // Pointers to this token's Q vector and output for this head
    const float* q_vec = Q + token_idx * E + head_idx * head_dim;
    float* out_vec = output + token_idx * E + head_idx * head_dim;
    
    // Find max score for numerical stability
    float max_score = -1e30f;
    for (int j = 0; j < S; ++j) {
        const float* k_vec = K + (batch_idx * S + j) * E + head_idx * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            dot += q_vec[d] * k_vec[d];
        }
        float score = dot * scale;
        if (score > max_score) max_score = score;
    }
    
    // Compute softmax denominator
    float sum_exp = 0.0f;
    for (int j = 0; j < S; ++j) {
        const float* k_vec = K + (batch_idx * S + j) * E + head_idx * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            dot += q_vec[d] * k_vec[d];
        }
        sum_exp += expf(dot * scale - max_score);
    }
    sum_exp = fmaxf(sum_exp, 1e-20f);
    
    // Compute weighted sum (attention output)
    for (int d = 0; d < head_dim; ++d) {
        float weighted_sum = 0.0f;
        for (int j = 0; j < S; ++j) {
            const float* k_vec = K + (batch_idx * S + j) * E + head_idx * head_dim;
            const float* v_vec = V + (batch_idx * S + j) * E + head_idx * head_dim;
            
            float dot = 0.0f;
            for (int dd = 0; dd < head_dim; ++dd) {
                dot += q_vec[dd] * k_vec[dd];
            }
            float attention_weight = expf(dot * scale - max_score) / sum_exp;
            weighted_sum += attention_weight * v_vec[d];
        }
        out_vec[d] = weighted_sum;
    }
}

void launch_multihead_attention(const float* d_qkv, float* d_output, int B, int S, int E, int H) {
    std::cout << "Launching multihead_attention_kernel: B=" << B << ", S=" << S << ", E=" << E << ", H=" << H << std::endl;
    if (H <= 0 || (E % H) != 0) { 
        fprintf(stderr, "Bad MHA config: E=%d H=%d\n", E, H); 
        abort(); 
    }
    
    const float* d_Q = d_qkv;
    const float* d_K = d_qkv + (size_t)B * S * E;
    const float* d_V = d_qkv + 2 * (size_t)B * S * E;

    int total_tokens = B * S;
    int threads_per_block = 256;
    int blocks_x = (total_tokens + threads_per_block - 1) / threads_per_block;
    
    dim3 blocks(blocks_x, H);
    dim3 threads(threads_per_block);

    hipLaunchKernelGGL(multihead_attention_kernel, blocks, threads, 0, 0,
        d_Q, d_K, d_V, d_output, B, S, E, H);
    HIP_CHECK_KERNEL("multihead_attention_kernel");
    std::cout << "multihead_attention_kernel completed successfully" << std::endl;
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
    if (token_id < 0 || token_id >= V) {
        // write zeros for invalid token IDs
        output[(b * S + s) * E + e] = 0.0f;
        return;
    }

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
    std::cout << "Launching embedding_lookup_kernel: B=" << batch_size << ", S=" << seq_len 
              << ", V=" << vocab_size << ", E=" << embed_dim << std::endl;
    dim3 blocks(batch_size, seq_len);
    int threads = (embed_dim > 1024) ? 1024 : embed_dim;

    hipLaunchKernelGGL(embedding_lookup_kernel, blocks, dim3(threads), 0, 0,
        d_token_ids, d_token_embed, d_pos_embed, d_output,
        batch_size, seq_len, vocab_size, embed_dim);
    HIP_CHECK_KERNEL("embedding_lookup_kernel");
    std::cout << "embedding_lookup_kernel completed successfully" << std::endl;
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

    // Compute softmax (unnormalized) and denominator
    float sum_exp = 0.0f;
    for (int i = 0; i < V; ++i) {
        softmax_out[idx * V + i] = expf(logits[idx * V + i] - max_logit);
        sum_exp += softmax_out[idx * V + i];
    }

    int label = labels[idx];
    if (label < 0 || label >= V) {
        // Gracefully handle bad labels: zero out row and skip loss/grad
        for (int i = 0; i < V; ++i) {
            softmax_out[idx * V + i] = 0.0f;
            grad_out[idx * V + i] = 0.0f;
        }
        return;
    }

    float log_prob = logits[idx * V + label] - max_logit - logf(sum_exp);
    atomicAdd(loss_sum, -log_prob);  // accumulate negative log likelihood

    // Gradient: softmax - one-hot(label)
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
    std::cout << "Launching softmax_loss_kernel: B=" << B << ", C=" << C << std::endl;
    int N = B;  // total tokens
    int V = C;  // vocab size

    float* d_loss_sum;
    HIP_CHECK(hipMalloc(&d_loss_sum, sizeof(float)));
    HIP_CHECK(hipMemset(d_loss_sum, 0, sizeof(float)));

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    hipLaunchKernelGGL(softmax_loss_kernel, dim3(blocks), dim3(threads), 0, 0,
        logits, softmax_out, labels, grad_out, d_loss_sum, N, V);
    HIP_CHECK_KERNEL("softmax_loss_kernel");

    float loss;
    HIP_CHECK(hipMemcpy(&loss, d_loss_sum, sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d_loss_sum));

    std::cout << "softmax_loss_kernel completed successfully, loss=" << loss/N << std::endl;
    return loss / N;  // average loss
}

__global__ void accuracy_kernel(const float* softmax, const int* labels, int* correct, int N, int V) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    const float* row = softmax + idx * V;
    int pred = 0;
    float max_val = row[0];
    for (int i = 1; i < V; ++i) {
        if (row[i] > max_val) { max_val = row[i]; pred = i; }
    }

    int label = labels[idx];
    if (label >= 0 && label < V && pred == label) atomicAdd(correct, 1);
}

float launch_accuracy(const float* d_softmax, const int* d_labels, int total_tokens, int vocab_size) {
    std::cout << "Launching accuracy_kernel: total_tokens=" << total_tokens << ", vocab_size=" << vocab_size << std::endl;
    int threads = 256;
    int blocks = (total_tokens + threads - 1) / threads;

    int* d_correct;
    HIP_CHECK(hipMalloc(&d_correct, sizeof(int)));
    HIP_CHECK(hipMemset(d_correct, 0, sizeof(int)));

    hipLaunchKernelGGL(accuracy_kernel, dim3(blocks), dim3(threads), 0, 0,
                       d_softmax, d_labels, d_correct, total_tokens, vocab_size);
    HIP_CHECK_KERNEL("accuracy_kernel");

    int h_correct = 0;
    HIP_CHECK(hipMemcpy(&h_correct, d_correct, sizeof(int), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d_correct));

    float accuracy = static_cast<float>(h_correct) / total_tokens;
    std::cout << "accuracy_kernel completed successfully, accuracy=" << accuracy << std::endl;
    return accuracy;
}

// ---- add_inplace ----
__global__ void add_inplace_kernel(float* a, const float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] += b[i];
}
void launch_add_inplace(float* a, const float* b, int n) {
    std::cout << "Launching add_inplace_kernel: n=" << n << std::endl;
    int threads = 256;
    int blocks  = (n + threads - 1) / threads;
    hipLaunchKernelGGL(add_inplace_kernel, dim3(blocks), dim3(threads), 0, 0, a, b, n);
    HIP_CHECK_KERNEL("add_inplace_kernel");
    std::cout << "add_inplace_kernel completed successfully" << std::endl;
}

// ---- ReLU backprop ----
__global__ void relu_backprop_kernel(const float* act, const float* grad_out, float* grad_in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) grad_in[i] = act[i] > 0.0f ? grad_out[i] : 0.0f;
}
void launch_backprop_activation(const float* act, const float* grad_out, float* grad_in, int n) {
    std::cout << "Launching relu_backprop_kernel: n=" << n << std::endl;
    int threads = 256;
    int blocks  = (n + threads - 1) / threads;
    hipLaunchKernelGGL(relu_backprop_kernel, dim3(blocks), dim3(threads), 0, 0, act, grad_out, grad_in, n);
    HIP_CHECK_KERNEL("relu_backprop_kernel");
    std::cout << "relu_backprop_kernel completed successfully" << std::endl;
}

// ---- SGD update ----
__global__ void sgd_update_kernel(float* w, const float* g, float lr, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) w[i] -= lr * g[i];
}
void launch_sgd_update(float* w, const float* g, float lr, int n) {
    std::cout << "Launching sgd_update_kernel: n=" << n << ", lr=" << lr << std::endl;
    int threads = 256;
    int blocks  = (n + threads - 1) / threads;
    hipLaunchKernelGGL(sgd_update_kernel, dim3(blocks), dim3(threads), 0, 0, w, g, lr, n);
    HIP_CHECK_KERNEL("sgd_update_kernel");
    std::cout << "sgd_update_kernel completed successfully" << std::endl;
}

// Layer Normalization Forward
__global__ void layernorm_forward_kernel(const float* input, float* output, const float* gamma, const float* beta, int N, int E) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float mean = 0.0f;
    for (int j = 0; j < E; ++j) mean += input[i * E + j];
    mean /= E;

    float variance = 0.0f;
    for (int j = 0; j < E; ++j) {
        float diff = input[i * E + j] - mean;
        variance += diff * diff;
    }
    variance /= E;

    float stddev = sqrtf(variance + 1e-5f);

    for (int j = 0; j < E; ++j) {
        output[i * E + j] = gamma[j] * (input[i * E + j] - mean) / stddev + beta[j];
    }
}

void launch_layer_norm_forward(const float* input, float* output, const float* gamma, const float* beta, int N, int E) {
    std::cout << "Launching layernorm_forward_kernel: N=" << N << ", E=" << E << std::endl;
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    hipLaunchKernelGGL(layernorm_forward_kernel, dim3(blocks), dim3(threads), 0, 0, input, output, gamma, beta, N, E);
    HIP_CHECK_KERNEL("layernorm_forward_kernel");
    std::cout << "layernorm_forward_kernel completed successfully" << std::endl;
}

// Layer Normalization Backward
__global__ void layernorm_backward_kernel(const float* grad_output, const float* input, float* grad_input, const float* gamma, float* grad_gamma, float* grad_beta, int N, int E) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float mean = 0.0f;
    for (int j = 0; j < E; ++j) mean += input[i * E + j];
    mean /= E;

    float variance = 0.0f;
    for (int j = 0; j < E; ++j) {
        float diff = input[i * E + j] - mean;
        variance += diff * diff;
    }
    variance /= E;
    float stddev = sqrtf(variance + 1e-5f);

    float sum_grad_norm_input = 0.0f;
    for (int j = 0; j < E; ++j) {
        sum_grad_norm_input += grad_output[i * E + j] * (input[i * E + j] - mean) / stddev;
    }
    
    float sum_grad_norm_input_x = 0.0f;
    for (int j = 0; j < E; ++j) {
        sum_grad_norm_input_x += grad_output[i * E + j] * gamma[j] * (-1.0f / (stddev * stddev)) * (input[i * E + j] - mean);
    }

    for (int j = 0; j < E; ++j) {
        float norm_input = (input[i * E + j] - mean) / stddev;
        atomicAdd(&grad_gamma[j], grad_output[i * E + j] * norm_input);
        atomicAdd(&grad_beta[j], grad_output[i * E + j]);
        grad_input[i * E + j] = (gamma[j] / stddev) * grad_output[i * E + j]
                              - (gamma[j] / (E * stddev)) * sum_grad_norm_input
                              - (norm_input / (E * stddev)) * sum_grad_norm_input_x;
    }
}

void launch_layer_norm_backward(const float* grad_output, const float* input, float* grad_input, const float* gamma, float* grad_gamma, float* grad_beta, int N, int E) {
    std::cout << "Launching layernorm_backward_kernel: N=" << N << ", E=" << E << std::endl;
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    hipLaunchKernelGGL(layernorm_backward_kernel, dim3(blocks), dim3(threads), 0, 0, grad_output, input, grad_input, gamma, grad_gamma, grad_beta, N, E);
    HIP_CHECK_KERNEL("layernorm_backward_kernel");
    std::cout << "layernorm_backward_kernel completed successfully" << std::endl;
}

// Dropout Forward
__global__ void dropout_forward_kernel(const float* input, float* output, float* mask, float p, int N, int E) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * E) return;

    xorwow_state rng_state;
    rng_state.a = idx + 1;
    rng_state.b = blockIdx.x + 1;
    rng_state.c = threadIdx.x + 1;
    rng_state.d = (idx * blockIdx.x) + 1;
    rng_state.counter = 0;

    float rand_val = (float)xorwow(&rng_state) / (float)UINT_MAX;

    if (rand_val < p) {
        output[idx] = 0.0f;
        mask[idx] = 0.0f;
    } else {
        output[idx] = input[idx] / (1.0f - p);
        mask[idx] = 1.0f;
    }
}

void launch_dropout_forward(const float* input, float* output, float* mask, float p, int N, int E) {
    std::cout << "Launching dropout_forward_kernel: N=" << N << ", E=" << E << ", p=" << p << std::endl;
    int threads = 256;
    int blocks = (N * E + threads - 1) / threads;
    hipLaunchKernelGGL(dropout_forward_kernel, dim3(blocks), dim3(threads), 0, 0, input, output, mask, p, N, E);
    HIP_CHECK_KERNEL("dropout_forward_kernel");
    std::cout << "dropout_forward_kernel completed successfully" << std::endl;
}

// Dropout Backward
__global__ void dropout_backward_kernel(const float* grad_output, const float* mask, float p, float* grad_input, int N, int E) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * E) return;
    grad_input[idx] = (mask[idx] == 0.0f) ? 0.0f : (grad_output[idx] / (1.0f - p));
}

void launch_dropout_backward(const float* grad_output, const float* mask, float p, float* grad_input, int N, int E) {
    std::cout << "Launching dropout_backward_kernel: N=" << N << ", E=" << E << ", p=" << p << std::endl;
    int threads = 256;
    int blocks = (N * E + threads - 1) / threads;
    hipLaunchKernelGGL(dropout_backward_kernel, dim3(blocks), dim3(threads), 0, 0, grad_output, mask, p, grad_input, N, E);
    HIP_CHECK_KERNEL("dropout_backward_kernel");
    std::cout << "dropout_backward_kernel completed successfully" << std::endl;
}

// Adam Optimizer
__global__ void adam_update_kernel(
    float* weights, const float* grads, float* m, float* v,
    float lr, float beta1, float beta2, float epsilon, int t, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;
    m[i] = beta1 * m[i] + (1.0f - beta1) * grads[i];
    v[i] = beta2 * v[i] + (1.0f - beta2) * grads[i] * grads[i];
    float m_hat = m[i] / (1.0f - powf(beta1, t));
    float v_hat = v[i] / (1.0f - powf(beta2, t));
    weights[i] -= lr * m_hat / (sqrtf(v_hat) + epsilon);
}

void launch_adam_update(
    float* weights, const float* grads, float* m, float* v,
    float lr, float beta1, float beta2, float epsilon, int t, int size) {
    std::cout << "Launching adam_update_kernel: size=" << size << ", lr=" << lr << ", t=" << t << std::endl;
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    hipLaunchKernelGGL(adam_update_kernel, dim3(blocks), dim3(threads), 0, 0,
        weights, grads, m, v, lr, beta1, beta2, epsilon, t, size);
    HIP_CHECK_KERNEL("adam_update_kernel");
    std::cout << "adam_update_kernel completed successfully" << std::endl;
}

// --- Sampling: clamp temperature and renormalize top-k mass ---
__global__ void sample_from_logits_kernel(const float* logits, int* output_token, int vocab_size, int k, float temperature) {
    // carve shared memory into two arrays
    extern __shared__ unsigned char smem[];
    float* top_k_scores = reinterpret_cast<float*>(smem);
    int*   top_k_indices = reinterpret_cast<int*>(top_k_scores + k);

    temperature = fmaxf(temperature, 1e-6f);

    // temperature & max
    float max_logit = -INFINITY;
    for (int i = 0; i < vocab_size; ++i) {
        float scaled = logits[i] / temperature;
        if (scaled > max_logit) max_logit = scaled;
    }
    // denom
    float sum_exp = 0.0f;
    for (int i = 0; i < vocab_size; ++i) sum_exp += expf(logits[i] / temperature - max_logit);
    sum_exp = fmaxf(sum_exp, 1e-20f);

    // init top-k
    for (int i = 0; i < k; ++i) { top_k_scores[i] = -INFINITY; top_k_indices[i] = -1; }

    // select top-k (by probability)
    for (int i = 0; i < vocab_size; ++i) {
        float prob = expf(logits[i] / temperature - max_logit) / sum_exp;
        for (int j = 0; j < k; ++j) {
            if (prob > top_k_scores[j]) {
                for (int l = k - 1; l > j; --l) {
                    top_k_scores[l]  = top_k_scores[l-1];
                    top_k_indices[l] = top_k_indices[l-1];
                }
                top_k_scores[j]  = prob;
                top_k_indices[j] = i;
                break;
            }
        }
    }

    // Renormalize top-k mass
    float mass = 0.0f;
    for (int i = 0; i < k; ++i) mass += fmaxf(top_k_scores[i], 0.0f);
    if (mass <= 0.0f) { *output_token = (top_k_indices[0] >= 0) ? top_k_indices[0] : 0; return; }

    // sample
    xorwow_state rng_state{12345u,67890u,13579u,24680u,0u};
    float target = ((float)xorwow(&rng_state) / (float)UINT_MAX) * mass;
    float cum = 0.0f;
    for (int i = 0; i < k; ++i) {
        cum += top_k_scores[i];
        if (target <= cum) { *output_token = top_k_indices[i]; return; }
    }
    *output_token = top_k_indices[k-1];
}

void launch_sample_from_logits(const float* d_logits, int* d_output_token, int vocab_size, int k, float temperature) {
    std::cout << "Launching sample_from_logits_kernel: vocab_size=" << vocab_size << ", k=" << k << ", temp=" << temperature << std::endl;
    if (k < 1) k = 1;
    if (k > vocab_size) k = vocab_size;
    int shared_mem_size = k * (sizeof(float) + sizeof(int));
    hipLaunchKernelGGL(sample_from_logits_kernel, 1, 1, shared_mem_size, 0,
        d_logits, d_output_token, vocab_size, k, temperature);
    HIP_CHECK_KERNEL("sample_from_logits_kernel");
    std::cout << "sample_from_logits_kernel completed successfully" << std::endl;
}

__global__ void embedding_backward_kernel(const float* grad_out, const int* token_ids,
                                          float* grad_token_embed, float* grad_pos_embed,
                                          int B, int S, int E, int V) {
    int b = blockIdx.x;
    int s = blockIdx.y;
    int e = blockIdx.z * blockDim.x + threadIdx.x;
    if (b >= B || s >= S || e >= E) return;

    int token_id = token_ids[b * S + s];
    if (token_id < 0 || token_id >= V) return;   // <-- critical

    const float* g = grad_out + (b * S + s) * E + e;
    atomicAdd(&grad_token_embed[token_id * E + e], *g);
    atomicAdd(&grad_pos_embed[s * E + e], *g);
}

void launch_embedding_backward(const float* grad_out, const int* token_ids,
                               float* grad_token_embed, float* grad_pos_embed,
                               int B, int S, int E, int V) {
    std::cout << "Launching embedding_backward_kernel: B=" << B << ", S=" << S 
              << ", E=" << E << ", V=" << V << std::endl;
    int tpb = (E >= 1024) ? 1024 : E;
    int bz  = (E + tpb - 1) / tpb;
    dim3 blocks(B, S, bz), threads(tpb);
    hipLaunchKernelGGL(embedding_backward_kernel, blocks, threads, 0, 0,
                       grad_out, token_ids, grad_token_embed, grad_pos_embed, B, S, E, V);
    HIP_CHECK_KERNEL("embedding_backward_kernel");
    std::cout << "embedding_backward_kernel completed successfully" << std::endl;
}

__global__ void matmul_transpose_B_kernel(
    const float* A, const float* B, float* C, 
    int M, int N, int K, int chunk_offset, int chunk_size, bool clear_output
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // M dimension  
    int col = blockIdx.x * blockDim.x + threadIdx.x; // K dimension
    if (row >= M || col >= K) return;

    float sum = 0.0f;
    
    // Process this chunk: from chunk_offset to chunk_offset + chunk_size
    int end_idx = min(chunk_offset + chunk_size, N);
    for (int i = chunk_offset; i < end_idx; ++i) {
        sum += A[row * N + i] * B[col * N + i]; // B is transposed
    }
    
    if (clear_output) {
        // First chunk: overwrite
        C[row * K + col] = sum;
    } else {
        // Subsequent chunks: accumulate
        atomicAdd(&C[row * K + col], sum);
    }
}

void launch_matmul_transpose_B(const float* A, const float* B, float* C, int M, int N, int K) {
    std::cout << "Launching matmul_transpose_B_kernel: M=" << M << ", N=" << N << ", K=" << K << std::endl;
    
    dim3 threads(16, 16);
    dim3 blocks((K + 15) / 16, (M + 15) / 16);
    
    // Process in chunks to avoid timeout
    int chunk_size = 256;  // Can be larger for this direction
    int num_chunks = (N + chunk_size - 1) / chunk_size;
    
    std::cout << "Processing in " << num_chunks << " chunks of size " << chunk_size << std::endl;
    
    for (int chunk = 0; chunk < num_chunks; ++chunk) {
        int chunk_offset = chunk * chunk_size;
        int current_chunk_size = min(chunk_size, N - chunk_offset);
        bool is_first_chunk = (chunk == 0);
        
        hipLaunchKernelGGL(matmul_transpose_B_kernel, blocks, threads, 0, 0, 
                          A, B, C, M, N, K, chunk_offset, current_chunk_size, is_first_chunk);
        
        // Check for errors after each chunk
        hipError_t err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "HIP Kernel Launch Error (matmul_transpose_B_kernel chunk " << chunk << ") at " 
                      << __FILE__ << ":" << __LINE__ << " - " << hipGetErrorString(err) << std::endl;
            exit(1);
        }
        
        err = hipDeviceSynchronize();
        if (err != hipSuccess) {
            std::cerr << "HIP Kernel Execution Error (matmul_transpose_B_kernel chunk " << chunk << ") at " 
                      << __FILE__ << ":" << __LINE__ << " - " << hipGetErrorString(err) << std::endl;
            exit(1);
        }
    }
    
    std::cout << "matmul_transpose_B_kernel completed successfully (processed " << num_chunks << " chunks)" << std::endl;
}

__global__ void matmul_transpose_A_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // K
    int col = blockIdx.x * blockDim.x + threadIdx.x; // N
    if (row >= K || col >= N) return;

    // Add bounds checking to prevent memory access faults
    if (row < 0 || row >= K || col < 0 || col >= N) return;

    float sum = 0.0f;
    
    // Process in smaller batches within each thread to prevent timeout
    const int BATCH_SIZE = 32; // Process 32 elements at a time
    for (int batch_start = 0; batch_start < M; batch_start += BATCH_SIZE) {
        int batch_end = min(batch_start + BATCH_SIZE, M);
        
        for (int i = batch_start; i < batch_end; ++i) {
            // Add bounds checking for memory access
            int a_idx = i * K + row;
            int b_idx = i * N + col;
            
            // Ensure indices are within valid bounds
            if (a_idx >= 0 && a_idx < M * K && b_idx >= 0 && b_idx < M * N) {
                sum += A[a_idx] * B[b_idx];
            }
        }
        
        // Add a small break to prevent GPU timeout
        if (batch_start > 0 && (batch_start % (BATCH_SIZE * 4)) == 0) {
            __syncthreads(); // Give GPU a chance to breathe
        }
    }
    
    // Final bounds check before writing result
    int c_idx = row * N + col;
    if (c_idx >= 0 && c_idx < K * N) {
        C[c_idx] = sum;
    }
}

void launch_matmul_transpose_A(const float* A, const float* B, float* C, int M, int N, int K) {
    std::cout << "Launching matmul_transpose_A_kernel: M=" << M << ", N=" << N << ", K=" << K << std::endl;
    
    // Add input validation
    if (M <= 0 || N <= 0 || K <= 0) {
        std::cerr << "Error: Invalid matrix dimensions in matmul_transpose_A" << std::endl;
        return;
    }
    
    // Use smaller thread blocks for large matrices to reduce memory pressure
    dim3 threads(8, 8);  // Reduced from 16x16 to 8x8
    dim3 blocks((N + 7) / 8, (K + 7) / 8);
    
    // Check if we're going to launch too many blocks
    int total_blocks = blocks.x * blocks.y;
    if (total_blocks > 65535) {
        std::cerr << "Warning: Large number of blocks (" << total_blocks << ") may cause issues" << std::endl;
        // Adjust block size for very large matrices
        threads = dim3(16, 16);
        blocks = dim3((N + 15) / 16, min((K + 15) / 16, 32768));
    }
    
    std::cout << "Using grid: (" << blocks.x << ", " << blocks.y << "), threads: (" 
              << threads.x << ", " << threads.y << ")" << std::endl;
    
    hipLaunchKernelGGL(matmul_transpose_A_kernel, blocks, threads, 0, 0, A, B, C, M, N, K);
    
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "HIP Kernel Launch Error (matmul_transpose_A_kernel) at " 
                  << __FILE__ << ":" << __LINE__ << " - " << hipGetErrorString(err) << std::endl;
        exit(1);
    }
    
    err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        std::cerr << "HIP Kernel Execution Error (matmul_transpose_A_kernel) at " 
                  << __FILE__ << ":" << __LINE__ << " - " << hipGetErrorString(err) << std::endl;
        exit(1);
    }
    
    std::cout << "matmul_transpose_A_kernel completed successfully" << std::endl;
}


// Fixed Multi-Head Attention Backward Kernel
__global__ void multihead_attention_backward_kernel(
    const float* grad_attn_out, // [B*S, E]
    const float* q_in,          // [B*S, E]
    const float* k_in,          // [B*S, E]
    const float* v_in,          // [B*S, E]
    float* grad_q_out,          // [B*S, E]
    float* grad_k_out,          // [B*S, E]
    float* grad_v_out,          // [B*S, E]
    int B, int S, int E, int H
) {
    int token_i = blockIdx.x * blockDim.x + threadIdx.x;
    int head_h = blockIdx.y;
    
    if (token_i >= B * S || head_h >= H) return;

    int head_dim = E / H;
    const float scale = 1.0f / sqrtf((float)head_dim);
    
    int batch_idx = token_i / S;
    int batch_start_idx = batch_idx * S;

    const float* q_vec = q_in + token_i * E + head_h * head_dim;
    const float* gho = grad_attn_out + token_i * E + head_h * head_dim;

    // Pass 1: Find max score for numerical stability
    float max_s = -1e30f;
    for (int j = 0; j < S; ++j) {
        const float* k_vec = k_in + (batch_start_idx + j) * E + head_h * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) dot += q_vec[d] * k_vec[d];
        float s = dot * scale;
        if (s > max_s) max_s = s;
    }

    // Pass 2: Compute denominator
    float denom = 0.0f;
    for (int j = 0; j < S; ++j) {
        const float* k_vec = k_in + (batch_start_idx + j) * E + head_h * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) dot += q_vec[d] * k_vec[d];
        denom += expf(dot * scale - max_s);
    }
    denom = fmaxf(denom, 1e-20f);

    // Pass 3: Compute sum_term for softmax backward and grad_V
    float sum_term = 0.0f;
    for (int j = 0; j < S; ++j) {
        // Compute attention probability
        const float* k_vec = k_in + (batch_start_idx + j) * E + head_h * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) dot += q_vec[d] * k_vec[d];
        float p = expf(dot * scale - max_s) / denom;

        // grad_softmax_j = gho · v_j
        const float* v_vec = v_in + (batch_start_idx + j) * E + head_h * head_dim;
        float g = 0.0f;
        for (int d = 0; d < head_dim; ++d) g += gho[d] * v_vec[d];

        sum_term += g * p;

        // Accumulate grad_V: grad_V += p * gho
        for (int d = 0; d < head_dim; ++d) {
            atomicAdd(&grad_v_out[(batch_start_idx + j) * E + head_h * head_dim + d], p * gho[d]);
        }
    }

    // Pass 4: Compute gradients w.r.t Q and K
    for (int d = 0; d < head_dim; ++d) {
        float acc_qd = 0.0f;
        
        for (int j = 0; j < S; ++j) {
            const float* k_vec = k_in + (batch_start_idx + j) * E + head_h * head_dim;
            float dot = 0.0f;
            for (int dd = 0; dd < head_dim; ++dd) dot += q_vec[dd] * k_vec[dd];
            float p = expf(dot * scale - max_s) / denom;

            // Recompute grad_softmax_j
            const float* v_vec = v_in + (batch_start_idx + j) * E + head_h * head_dim;
            float g = 0.0f;
            for (int dd = 0; dd < head_dim; ++dd) g += gho[dd] * v_vec[dd];

            float grad_score = (g - sum_term) * p; // softmax backward
            float w = grad_score * scale;

            // Accumulate gradient w.r.t Q
            acc_qd += w * k_vec[d];
            
            // Accumulate gradient w.r.t K
            atomicAdd(&grad_k_out[(batch_start_idx + j) * E + head_h * head_dim + d], w * q_vec[d]);
        }
        
        // Write gradient w.r.t Q (no atomic needed since each thread writes to its own location)
        grad_q_out[token_i * E + head_h * head_dim + d] = acc_qd;
    }
}

void launch_multihead_attention_backward(
    const float* d_grad_attn_output, const float* d_qkv, const float* /*d_softmax*/,
    float* d_grad_qkv, int B, int S, int E, int H
) {
    std::cout << "Launching multihead_attention_backward_kernel: B=" << B << ", S=" << S 
              << ", E=" << E << ", H=" << H << std::endl;
    if (H <= 0 || (E % H) != 0) { 
        fprintf(stderr, "Bad MHA config (bwd): E=%d H=%d\n", E, H); 
        abort(); 
    }

    const float* d_Q = d_qkv;
    const float* d_K = d_qkv + (size_t)B * S * E;
    const float* d_V = d_qkv + 2 * (size_t)B * S * E;

    float* d_grad_Q = d_grad_qkv;
    float* d_grad_K = d_grad_qkv + (size_t)B * S * E;
    float* d_grad_V = d_grad_qkv + 2 * (size_t)B * S * E;

    // Clear gradients
    HIP_CHECK(hipMemset(d_grad_qkv, 0, (size_t)B * S * E * 3 * sizeof(float)));

    int total_tokens = B * S;
    int threads_per_block = 256;
    int blocks_x = (total_tokens + threads_per_block - 1) / threads_per_block;
    
    dim3 blocks(blocks_x, H);
    dim3 threads(threads_per_block);

    hipLaunchKernelGGL(multihead_attention_backward_kernel, blocks, threads, 0, 0,
        d_grad_attn_output, d_Q, d_K, d_V, d_grad_Q, d_grad_K, d_grad_V, B, S, E, H);
    HIP_CHECK_KERNEL("multihead_attention_backward_kernel");
    std::cout << "multihead_attention_backward_kernel completed successfully" << std::endl;
}

// Additional GPU memory and device info functions
void print_gpu_info() {
    int device_count = 0;
    HIP_CHECK(hipGetDeviceCount(&device_count));
    std::cout << "Number of HIP devices: " << device_count << std::endl;
    
    for (int i = 0; i < device_count; ++i) {
        hipDeviceProp_t prop;
        HIP_CHECK(hipGetDeviceProperties(&prop, i));
        std::cout << "Device " << i << ": " << prop.name << std::endl;
        std::cout << "  Total global memory: " << prop.totalGlobalMem / (1024*1024) << " MB" << std::endl;
        std::cout << "  Multiprocessor count: " << prop.multiProcessorCount << std::endl;
        std::cout << "  Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
    }
}

void check_gpu_memory() {
    size_t free_mem, total_mem;
    HIP_CHECK(hipMemGetInfo(&free_mem, &total_mem));
    std::cout << "GPU Memory - Free: " << free_mem / (1024*1024) << " MB, "
              << "Total: " << total_mem / (1024*1024) << " MB, "
              << "Used: " << (total_mem - free_mem) / (1024*1024) << " MB" << std::endl;
}

// Add this function to be called in your main training loop
void debug_checkpoint(const std::string& location) {
    std::cout << "\n=== DEBUG CHECKPOINT: " << location << " ===" << std::endl;
    check_gpu_memory();
    HIP_CHECK(hipDeviceSynchronize());
    std::cout << "GPU synchronization successful at " << location << std::endl;
    std::cout << "============================================\n" << std::endl;
}