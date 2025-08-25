// File: hip_kernels.cpp
#include "hip_kernels.h"
#include <hip/hip_runtime.h>
#include <cmath>
#include <cfloat>

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

// ---------------- Matmul Backward with Bias (Corrected) ----------------

// Kernel to compute gradient w.r.t weights. A_input is transposed.
__global__ void matmul_backward_weight_kernel(const float* A_input, const float* B_grad_out, float* C_grad_weight, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // K (dims of A_input)
    int col = blockIdx.x * blockDim.x + threadIdx.x; // N (dims of B_grad_out)
    
    if (row < K && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < M; ++i) {
            // A_input is (M, K), B_grad_out is (M, N)
            // We effectively compute A_T @ B
            sum += A_input[i * K + row] * B_grad_out[i * N + col];
        }
        C_grad_weight[row * N + col] = sum;
    }
}

// Kernel to compute gradient w.r.t bias. This is a simple reduction.
__global__ void bias_backward_kernel(const float* B_grad_out, float* D_grad_bias, int M, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x; // N (number of biases)
    if (col < N) {
        float sum = 0.0f;
        for (int i = 0; i < M; ++i) {
            sum += B_grad_out[i * N + col];
        }
        D_grad_bias[col] = sum;
    }
}

void launch_matmul_backward_bias(const float* A_input, const float* B_grad_out, float* C_grad_weight, float* D_grad_bias, int M, int N, int K) {
    // Launch kernel for weight gradients (A_T @ B)
    dim3 threads_w(16, 16);
    dim3 blocks_w((N + 15) / 16, (K + 15) / 16);
    hipLaunchKernelGGL(matmul_backward_weight_kernel, blocks_w, threads_w, 0, 0, A_input, B_grad_out, C_grad_weight, M, N, K);

    // Launch a separate, 1D kernel for bias gradients
    int threads_b = 256;
    int blocks_b = (N + threads_b - 1) / threads_b;
    hipLaunchKernelGGL(bias_backward_kernel, dim3(blocks_b), dim3(threads_b), 0, 0, B_grad_out, D_grad_bias, M, N);
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

// ---------------- Parallel Multi-Head Attention Kernel ----------------
// This kernel computes the attention for a single token and a single head.
// It is launched in a grid where each thread handles one token-head pair.
__global__ void multihead_attention_kernel(const float* Q, const float* K, const float* V, float* output,
                                            int B, int S, int E, int H) {
    // Each thread block processes one head for all tokens
    // Each thread processes one token for one head
    int token_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int head_idx = blockIdx.y;

    if (token_idx >= B * S) return;

    int head_dim = E / H;
    const float scale = 1.0f / sqrtf(head_dim);

    // Pointers to the current head's Q, K, V data
    const float* q_vec = Q + token_idx * E + head_idx * head_dim;
    const int batch_start_idx = (token_idx / S) * S;

    // --- 1. Calculate Attention Scores ---
    // Use shared memory to store scores for one token: Q_i @ K_j for all j
    extern __shared__ float scores[]; // Size S
    
    for (int j = 0; j < S; ++j) {
        const float* k_vec = K + (batch_start_idx + j) * E + head_idx * head_dim;
        float dot_product = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            dot_product += q_vec[d] * k_vec[d];
        }
        scores[j] = dot_product * scale;
    }
    
    // --- 2. Softmax ---
    float max_score = -1e9f;
    for (int j = 0; j < S; ++j) {
        if (scores[j] > max_score) max_score = scores[j];
    }
    float sum_exp = 0.0f;
    for (int j = 0; j < S; ++j) {
        scores[j] = expf(scores[j] - max_score);
        sum_exp += scores[j];
    }
    for (int j = 0; j < S; ++j) {
        scores[j] /= sum_exp; // scores[] now holds softmax probabilities
    }

    // --- 3. Weighted Sum of Values ---
    for (int d = 0; d < head_dim; ++d) {
        float weighted_sum = 0.0f;
        for (int j = 0; j < S; ++j) {
            const float* v_vec = V + (batch_start_idx + j) * E + head_idx * head_dim;
            weighted_sum += scores[j] * v_vec[d];
        }
        output[token_idx * E + head_idx * head_dim + d] = weighted_sum;
    }
}

void launch_multihead_attention(const float* d_qkv, float* d_output, int B, int S, int E, int H) {
    const float* d_Q = d_qkv;
    const float* d_K = d_qkv + (B * S * E);
    const float* d_V = d_qkv + 2 * (B * S * E);

    int total_tokens = B * S;
    int threads_per_block = 256;
    dim3 blocks((total_tokens + threads_per_block - 1) / threads_per_block, H); // Grid: (tokens, heads)
    dim3 threads(threads_per_block);
    
    // Shared memory size for storing scores per token (Q_i @ K_all)
    size_t shared_mem_size = S * sizeof(float);

    hipLaunchKernelGGL(multihead_attention_kernel, blocks, threads, shared_mem_size, 0,
        d_Q, d_K, d_V, d_output, B, S, E, H);
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
    int threads = (embed_dim > 1024) ? 1024 : embed_dim;

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
    if (label < 0 || label >= V) {
        for (int i = 0; i < V; ++i) {
            softmax_out[idx * V + i] = 0.0f;
            grad_out[idx * V + i] = 0.0f;
        }
    return;
}

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

// ---- add_inplace ----
__global__ void add_inplace_kernel(float* a, const float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] += b[i];
}
void launch_add_inplace(float* a, const float* b, int n) {
    int threads = 256;
    int blocks  = (n + threads - 1) / threads;
    hipLaunchKernelGGL(add_inplace_kernel, dim3(blocks), dim3(threads), 0, 0, a, b, n);
}

// ---- ReLU backprop ----
__global__ void relu_backprop_kernel(const float* act, const float* grad_out, float* grad_in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) grad_in[i] = act[i] > 0.0f ? grad_out[i] : 0.0f;
}
void launch_backprop_activation(const float* act, const float* grad_out, float* grad_in, int n) {
    int threads = 256;
    int blocks  = (n + threads - 1) / threads;
    hipLaunchKernelGGL(relu_backprop_kernel, dim3(blocks), dim3(threads), 0, 0, act, grad_out, grad_in, n);
}

// ---- SGD update ----
__global__ void sgd_update_kernel(float* w, const float* g, float lr, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) w[i] -= lr * g[i];
}
void launch_sgd_update(float* w, const float* g, float lr, int n) {
    int threads = 256;
    int blocks  = (n + threads - 1) / threads;
    hipLaunchKernelGGL(sgd_update_kernel, dim3(blocks), dim3(threads), 0, 0, w, g, lr, n);
}


// Layer Normalization Forward
__global__ void layernorm_forward_kernel(const float* input, float* output, const float* gamma, const float* beta, int N, int E) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float mean = 0.0f;
    for (int j = 0; j < E; ++j) {
        mean += input[i * E + j];
    }
    mean /= E;

    float variance = 0.0f;
    for (int j = 0; j < E; ++j) {
        variance += (input[i * E + j] - mean) * (input[i * E + j] - mean);
    }
    variance /= E;

    float stddev = sqrtf(variance + 1e-5f);

    for (int j = 0; j < E; ++j) {
        output[i * E + j] = gamma[j] * (input[i * E + j] - mean) / stddev + beta[j];
    }
}

void launch_layer_norm_forward(const float* input, float* output, const float* gamma, const float* beta, int N, int E) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    hipLaunchKernelGGL(layernorm_forward_kernel, dim3(blocks), dim3(threads), 0, 0, input, output, gamma, beta, N, E);
}

// Layer Normalization Backward
__global__ void layernorm_backward_kernel(const float* grad_output, const float* input, float* grad_input, const float* gamma, float* grad_gamma, float* grad_beta, int N, int E) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float mean = 0.0f;
    for (int j = 0; j < E; ++j) {
        mean += input[i * E + j];
    }
    mean /= E;

    float variance = 0.0f;
    for (int j = 0; j < E; ++j) {
        variance += (input[i * E + j] - mean) * (input[i * E + j] - mean);
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
        
        // Gradient for gamma and beta
        atomicAdd(&grad_gamma[j], grad_output[i * E + j] * norm_input);
        atomicAdd(&grad_beta[j], grad_output[i * E + j]);

        // Gradient for input
        grad_input[i * E + j] = (gamma[j] / stddev) * grad_output[i * E + j] - (gamma[j] / (E * stddev)) * sum_grad_norm_input - (norm_input / (E * stddev)) * sum_grad_norm_input_x;
    }
}

void launch_layer_norm_backward(const float* grad_output, const float* input, float* grad_input, const float* gamma, float* grad_gamma, float* grad_beta, int N, int E) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    hipLaunchKernelGGL(layernorm_backward_kernel, dim3(blocks), dim3(threads), 0, 0, grad_output, input, grad_input, gamma, grad_gamma, grad_beta, N, E);
}

// Dropout Forward
__global__ void dropout_forward_kernel(const float* input, float* output, float* mask, float p, int N, int E) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * E) return;

    // Create a state seeded with the thread's unique ID
    xorwow_state rng_state;
    rng_state.a = idx + 1;
    rng_state.b = blockIdx.x + 1;
    rng_state.c = threadIdx.x + 1;
    rng_state.d = (idx * blockIdx.x) + 1;
    rng_state.counter = 0;

    // Generate a random float between 0.0 and 1.0
    float rand_val = (float)xorwow(&rng_state) / (float)UINT_MAX;

    if (rand_val < p) { // Use the new random value
        output[idx] = 0.0f;
        mask[idx] = 0.0f;
    } else {
        output[idx] = input[idx] / (1.0f - p);
        mask[idx] = 1.0f;
    }
}

void launch_dropout_forward(const float* input, float* output, float* mask, float p, int N, int E) {
    int threads = 256;
    int blocks = (N * E + threads - 1) / threads;
    hipLaunchKernelGGL(dropout_forward_kernel, dim3(blocks), dim3(threads), 0, 0, input, output, mask, p, N, E);
}

// Dropout Backward
__global__ void dropout_backward_kernel(const float* grad_output, const float* mask, float p, float* grad_input, int N, int E) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * E) return;

    if (mask[idx] == 0.0f) {
        grad_input[idx] = 0.0f;
    } else {
        grad_input[idx] = grad_output[idx] / (1.0f - p);
    }
}

void launch_dropout_backward(const float* grad_output, const float* mask, float p, float* grad_input, int N, int E) {
    int threads = 256;
    int blocks = (N * E + threads - 1) / threads;
    hipLaunchKernelGGL(dropout_backward_kernel, dim3(blocks), dim3(threads), 0, 0, grad_output, mask, p, grad_input, N, E);
}

// Adam Optimizer
__global__ void adam_update_kernel(
    float* weights,
    const float* grads,
    float* m,
    float* v,
    float lr,
    float beta1,
    float beta2,
    float epsilon,
    int t,
    int size) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;

    // Update biased first moment estimate
    m[i] = beta1 * m[i] + (1.0f - beta1) * grads[i];

    // Update biased second raw moment estimate
    v[i] = beta2 * v[i] + (1.0f - beta2) * grads[i] * grads[i];

    // Compute bias-corrected first and second moment estimates
    float m_hat = m[i] / (1.0f - powf(beta1, t));
    float v_hat = v[i] / (1.0f - powf(beta2, t));

    // Update weights
    weights[i] -= lr * m_hat / (sqrtf(v_hat) + epsilon);
}

void launch_adam_update(
    float* weights,
    const float* grads,
    float* m,
    float* v,
    float lr,
    float beta1,
    float beta2,
    float epsilon,
    int t,
    int size) {
    
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    hipLaunchKernelGGL(adam_update_kernel, dim3(blocks), dim3(threads), 0, 0,
        weights, grads, m, v, lr, beta1, beta2, epsilon, t, size);
}

// NEW: Corrected and parallelized sample_from_logits_kernel
// This kernel will be launched with one block and one thread.
__global__ void sample_from_logits_kernel(const float* logits, int* output_token, int vocab_size, int k, float temperature) {
    // Shared memory for top-k selection
    extern __shared__ float top_k_scores[];
    extern __shared__ int top_k_indices[];

    // Apply temperature and softmax to the full vocabulary
    float max_logit = -INFINITY;
    for (int i = 0; i < vocab_size; ++i) {
        if (logits[i] / temperature > max_logit) {
            max_logit = logits[i] / temperature;
        }
    }
    
    float sum_exp = 0.0f;
    for (int i = 0; i < vocab_size; ++i) {
        sum_exp += expf(logits[i] / temperature - max_logit);
    }

    // Initialize top-k arrays with sentinel values
    for (int i = 0; i < k; ++i) {
        top_k_scores[i] = -INFINITY;
        top_k_indices[i] = -1;
    }
    
    // Find top-k tokens and scores
    for (int i = 0; i < vocab_size; ++i) {
        float prob = expf(logits[i] / temperature - max_logit) / sum_exp;
        
        // Insert into top-k list
        for (int j = 0; j < k; ++j) {
            if (prob > top_k_scores[j]) {
                // Shift elements to the right to make space for the new one
                for (int l = k - 1; l > j; --l) {
                    top_k_scores[l] = top_k_scores[l-1];
                    top_k_indices[l] = top_k_indices[l-1];
                }
                top_k_scores[j] = prob;
                top_k_indices[j] = i;
                break;
            }
        }
    }
    
    // Perform cumulative sampling from top-k tokens
    xorwow_state rng_state;
    rng_state.a = 12345; // Can use any non-zero seed
    rng_state.b = 67890;
    rng_state.c = 13579;
    rng_state.d = 24680;
    rng_state.counter = 0;
    float r = (float)xorwow(&rng_state) / (float)UINT_MAX;
    float cumulative_prob = 0.0f;
    int selected_token = -1;
    
    for (int i = 0; i < k; ++i) {
        cumulative_prob += top_k_scores[i];
        if (r < cumulative_prob) {
            selected_token = top_k_indices[i];
            break;
        }
    }
    
    // Write selected token to output
    if (selected_token != -1) {
        *output_token = selected_token;
    } else {
        // Fallback to the most likely token if sampling fails
        *output_token = top_k_indices[0];
    }
}

void launch_sample_from_logits(const float* d_logits, int* d_output_token, int vocab_size, int k, float temperature) {
    int shared_mem_size = k * (sizeof(float) + sizeof(int));
    hipLaunchKernelGGL(sample_from_logits_kernel, 1, 1, shared_mem_size, 0, d_logits, d_output_token, vocab_size, k, temperature);
}

__global__ void embedding_backward_kernel(const float* grad_out, const int* token_ids,
                                          float* grad_token_embed, float* grad_pos_embed,
                                          int B, int S, int E) {
    int b = blockIdx.x;
    int s = blockIdx.y;
    int e = threadIdx.x;

    if (b >= B || s >= S || e >= E) return;

    int token_id = token_ids[b * S + s];
    const float* grad = grad_out + (b * S + s) * E + e;

    // Add the gradient to the specific token embedding vector
    atomicAdd(&grad_token_embed[token_id * E + e], *grad);
    // Add the gradient to the specific position embedding vector
    atomicAdd(&grad_pos_embed[s * E + e], *grad);
}

void launch_embedding_backward(const float* grad_out, const int* token_ids,
                               float* grad_token_embed, float* grad_pos_embed,
                               int B, int S, int E) {
    dim3 blocks(B, S);
    dim3 threads(E);
    hipLaunchKernelGGL(embedding_backward_kernel, blocks, threads, 0, 0,
                       grad_out, token_ids, grad_token_embed, grad_pos_embed, B, S, E);
}

// Kernel for C = A * B^T
__global__ void matmul_transpose_B_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= K) return;

    float sum = 0.0f;
    for (int i = 0; i < N; ++i) {
        sum += A[row * N + i] * B[col * N + i]; // B is transposed
    }
    C[row * K + col] = sum;
}

void launch_matmul_transpose_B(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threads(16, 16);
    dim3 blocks((K + 15) / 16, (M + 15) / 16);
    hipLaunchKernelGGL(matmul_transpose_B_kernel, blocks, threads, 0, 0, A, B, C, M, N, K);
}

// Kernel for C = A^T * B
__global__ void matmul_transpose_A_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Corresponds to K
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Corresponds to N
    if (row >= K || col >= N) return;

    float sum = 0.0f;
    for (int i = 0; i < M; ++i) {
        sum += A[i * K + row] * B[i * N + col]; // A is transposed
    }
    C[row * N + col] = sum;
}

void launch_matmul_transpose_A(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threads(16, 16);
    // Output C is (K, N), so blocks are based on N and K
    dim3 blocks((N + 15) / 16, (K + 15) / 16);
    hipLaunchKernelGGL(matmul_transpose_A_kernel, blocks, threads, 0, 0, A, B, C, M, N, K);
}

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

    if (token_i >= B * S) return;

    int head_dim = E / H;
    const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
    const int batch_start_idx = (token_i / S) * S;

    const float* q_vec = q_in + token_i * E + head_h * head_dim;
    const float* grad_head_out_vec = grad_attn_out + token_i * E + head_h * head_dim;

    extern __shared__ float s_data[];
    float* scores = s_data; 
    
    // Step 1: Recompute attention scores (q @ k^T / sqrt(d_k))
    for (int j = 0; j < S; ++j) {
        const float* k_vec = k_in + (batch_start_idx + j) * E + head_h * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            dot += q_vec[d] * k_vec[d];
        }
        scores[j] = dot * scale;
    }

    // Step 2: Recompute stable softmax
    float max_score = -FLT_MAX;
    for (int j = 0; j < S; ++j) {
        if (scores[j] > max_score) max_score = scores[j];
    }
    float sum_exp = 0.0f;
    for (int j = 0; j < S; ++j) {
        scores[j] = expf(scores[j] - max_score);
        sum_exp += scores[j];
    }
    const float inv_sum_exp = 1.0f / sum_exp;
    for (int j = 0; j < S; ++j) {
        scores[j] *= inv_sum_exp; // scores[] now holds softmax probabilities
    }
    
    // --- BACKPROPAGATION ---
    float grad_q_accum[128]; // Max head_dim <= 128
    for(int d = 0; d < head_dim; ++d) grad_q_accum[d] = 0.0f;
    
    float grad_softmax_scores[512]; // Max seq_len <= 512

    // Calculate grad w.r.t. softmax output
    for (int j = 0; j < S; ++j) {
        const float* v_vec = v_in + (batch_start_idx + j) * E + head_h * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            dot += grad_head_out_vec[d] * v_vec[d];
        }
        grad_softmax_scores[j] = dot;
    }

    // Backprop through softmax
    float dot_product_sum = 0.0f;
    for (int j = 0; j < S; ++j) {
        dot_product_sum += grad_softmax_scores[j] * scores[j];
    }

    for (int j = 0; j < S; ++j) {
        float softmax_val = scores[j];
        float grad_score = scale * softmax_val * (grad_softmax_scores[j] - dot_product_sum);

        const float* k_vec = k_in + (batch_start_idx + j) * E + head_h * head_dim;
        for (int d = 0; d < head_dim; ++d) {
            atomicAdd(&grad_v_out[(batch_start_idx + j) * E + head_h * head_dim + d], softmax_val * grad_head_out_vec[d]);
            atomicAdd(&grad_k_out[(batch_start_idx + j) * E + head_h * head_dim + d], grad_score * q_vec[d]);
            grad_q_accum[d] += grad_score * k_vec[d];
        }
    }

    // Write out the final accumulated gradient for Q
    for (int d = 0; d < head_dim; ++d) {
        grad_q_out[token_i * E + head_h * head_dim + d] = grad_q_accum[d];
    }
}

void launch_multihead_attention_backward(
    const float* d_grad_attn_output, const float* d_qkv, const float* d_softmax,
    float* d_grad_qkv, int B, int S, int E, int H
) {
    // Note: The d_softmax parameter is unused as the kernel recomputes it,
    // which is why we passed nullptr in the calling function.

    // Calculate pointers to the separate Q, K, V sections of the input tensor
    const float* d_Q = d_qkv;
    const float* d_K = d_qkv + ((size_t)B * S * E);
    const float* d_V = d_qkv + 2 * ((size_t)B * S * E);

    // Calculate pointers to the separate gradient sections of the output tensor
    float* d_grad_Q = d_grad_qkv;
    float* d_grad_K = d_grad_qkv + ((size_t)B * S * E);
    float* d_grad_V = d_grad_qkv + 2 * ((size_t)B * S * E);
    
    // Zero out the gradient buffers before accumulating gradients with atomicAdds
    hipMemset(d_grad_qkv, 0, (size_t)B * S * E * 3 * sizeof(float));

    int total_tokens = B * S;
    int threads_per_block = 256;
    // The grid is 2D: one dimension for tokens, the other for attention heads
    dim3 blocks((total_tokens + threads_per_block - 1) / threads_per_block, H);
    dim3 threads(threads_per_block);
    
    // Define the shared memory size needed by the kernel for recomputing softmax scores
    size_t shared_mem_size = S * sizeof(float);

    // Launch the backward kernel
    hipLaunchKernelGGL(multihead_attention_backward_kernel, blocks, threads, shared_mem_size, 0,
        d_grad_attn_output, d_Q, d_K, d_V, d_grad_Q, d_grad_K, d_grad_V, B, S, E, H);
}