#include "hip_kernels.h"
#include <hip/hip_runtime.h>
#include <cmath>
#include <climits>
#include <cstdio> 

#define TILE_SIZE 32
#define WARP_SIZE 64  // AMD warp size
#define MAX_THREADS_PER_BLOCK 1024



#define HIP_CHECK(cmd)                                                        \
    do {                                                                      \
        hipError_t _e = (cmd);                                                \
        if (_e != hipSuccess) {                                               \
            fprintf(stderr, "HIP error %s:%d: %s\n", __FILE__, __LINE__,      \
                    hipGetErrorString(_e));                                   \
            abort();                                                          \
        }                                                                     \
    } while (0)

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
__global__ void mean_pool_kernel(const float* input, float* output, int B, int L, int D) {
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
    hipLaunchKernelGGL(mean_pool_kernel, blocks, thread_dim, 0, 0, input, output, B, L, D);
}

// ============================================================================
// Matrix Multiplication with Shared Memory Tiling + Backward
// ============================================================================
__global__ void matmul_tiled_kernel(const float* A, const float* B, float* C,
                                    int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float sum = 0.0f;

    for (int tile = 0; tile < (N + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        int a_col = tile * TILE_SIZE + tx;
        int b_row = tile * TILE_SIZE + ty;

        As[ty][tx] = (row < M && a_col < N) ? A[row * N + a_col] : 0.0f;
        Bs[ty][tx] = (b_row < N && col < K) ? B[b_row * K + col] : 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }

    if (row < M && col < K) {
        C[row * K + col] = sum;
    }
}

void launch_matmul(const float* A, const float* B, float* C,
                   int M, int N, int K) {
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((K + TILE_SIZE - 1) / TILE_SIZE,
                (M + TILE_SIZE - 1) / TILE_SIZE);
    hipLaunchKernelGGL(matmul_tiled_kernel, blocks, threads, 0, 0,
                       A, B, C, M, N, K);
}

// ----------------------------------------------------------------------------

__global__ void matmul_add_bias_tiled_kernel(const float* A, const float* B,
                                             const float* bias, float* C,
                                             int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float sum = 0.0f;

    for (int tile = 0; tile < (N + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        int a_col = tile * TILE_SIZE + tx;
        int b_row = tile * TILE_SIZE + ty;

        As[ty][tx] = (row < M && a_col < N) ? A[row * N + a_col] : 0.0f;
        Bs[ty][tx] = (b_row < N && col < K) ? B[b_row * K + col] : 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }

    if (row < M && col < K) {
        C[row * K + col] = sum + bias[col];
    }
}

void launch_matmul_add_bias(const float* A, const float* B, const float* bias,
                            float* C, int M, int N, int K) {
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((K + TILE_SIZE - 1) / TILE_SIZE,
                (M + TILE_SIZE - 1) / TILE_SIZE);
    hipLaunchKernelGGL(matmul_add_bias_tiled_kernel, blocks, threads, 0, 0,
                       A, B, bias, C, M, N, K);
}

// ----------------------------------------------------------------------------
// Backward: Weight Gradient
// ----------------------------------------------------------------------------
__global__ void matmul_backward_weight_tiled_kernel(
    const float* __restrict__ A_input,    // [M,K]
    const float* __restrict__ B_grad_out, // [M,N]
    float* __restrict__ C_grad_weight,    // [K,N]
    int M, int N, int K
) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y; // K
    int col = blockIdx.x * TILE_SIZE + threadIdx.x; // N

    float sum = 0.0f;

    for (int tile = 0; tile < (M + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        int m = tile * TILE_SIZE + threadIdx.x;
        int n = tile * TILE_SIZE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] =
            (row < K && m < M) ? A_input[m * K + row] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] =
            (n < M && col < N) ? B_grad_out[n * N + col] : 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[k][threadIdx.x] * Bs[k][threadIdx.y];
        }
        __syncthreads();
    }

    if (row < K && col < N) {
        C_grad_weight[row * N + col] = sum;
    }
}

void launch_matmul_backward_weight(const float* A_input,
                                   const float* B_grad_out,
                                   float* C_grad_weight,
                                   int M, int N, int K) {
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE,
                (K + TILE_SIZE - 1) / TILE_SIZE);

    hipLaunchKernelGGL(matmul_backward_weight_tiled_kernel,
                       blocks, threads, 0, 0,
                       A_input, B_grad_out, C_grad_weight, M, N, K);
}

// ----------------------------------------------------------------------------
// Backward: Bias Gradient (optimized reduction)
// ----------------------------------------------------------------------------
__global__ void bias_backward_opt_kernel(const float* grad_out,
                                         float* grad_bias,
                                         int M, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= N) return;

    float sum = 0.0f;
    for (int m = 0; m < M; ++m) {
        sum += grad_out[m * N + col];
    }
    grad_bias[col] = sum;
}

void launch_bias_backward_opt(const float* grad_out,
                              float* grad_bias,
                              int M, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    hipLaunchKernelGGL(bias_backward_opt_kernel,
                       dim3(blocks), dim3(threads), 0, 0,
                       grad_out, grad_bias, M, N);
}

// ----------------------------------------------------------------------------
// Combined backward (weight + bias)
// ----------------------------------------------------------------------------
void launch_matmul_backward_bias(const float* A_input,
                                 const float* B_grad_out,
                                 float* C_grad_weight,
                                 float* D_grad_bias,
                                 int M, int N, int K) {
    // Weight grad
    launch_matmul_backward_weight(A_input, B_grad_out, C_grad_weight, M, N, K);
    // Bias grad
    launch_bias_backward_opt(B_grad_out, D_grad_bias, M, N);
}

// ---------------- ReLU ----------------
__global__ void relu_kernel(float* A, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) A[idx] = fmaxf(0.0f, A[idx]);
}

void launch_relu(float* A, int size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    hipLaunchKernelGGL(relu_kernel, dim3(blocks), dim3(threads), 0, 0, A, size);
}

// ============================================================================
// Flash Attention Implementation
// ============================================================================
template<int HEAD_DIM>
__global__ void flash_attention_kernel(
    const float* Q, const float* K, const float* V, float* output,
    int B, int S, int E, int H
) {
    const int batch_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int query_idx = blockIdx.z * blockDim.x + threadIdx.x;
    
    if (batch_idx >= B || head_idx >= H || query_idx >= S) return;
    
    const float scale = rsqrtf((float)HEAD_DIM);
    
    // Shared memory for current block of K, V
    extern __shared__ unsigned char smem_u8[];
    float* s_K = reinterpret_cast<float*>(smem_u8);
    float* s_V = s_K + HEAD_DIM * blockDim.y;

    
    const int q_offset = (batch_idx * S + query_idx) * E + head_idx * HEAD_DIM;
    const int kv_base = batch_idx * S * E + head_idx * HEAD_DIM;
    
    float max_score = -1e30f;
    float sum_exp = 0.0f;
    float output_acc[64] = {0.0f}; // Max head dim we support
    
    // Process keys/values in blocks to fit in shared memory
    const int BLOCK_SIZE = min(blockDim.y, S);
    
    for (int k_start = 0; k_start < S; k_start += BLOCK_SIZE) {
        int k_end = min(k_start + BLOCK_SIZE, S);
        int block_size = k_end - k_start;
        
        // Load K, V block into shared memory
        for (int i = threadIdx.y; i < block_size; i += blockDim.y) {
            for (int d = threadIdx.x; d < HEAD_DIM; d += blockDim.x) {
                // Use batch_idx to get the correct batch offset
                int k_global_idx = batch_idx * S * E + (k_start + i) * E + head_idx * HEAD_DIM + d;
                int v_global_idx = batch_idx * S * E + (k_start + i) * E + head_idx * HEAD_DIM + d;
                
                s_K[i * HEAD_DIM + d] = K[k_global_idx];
                s_V[i * HEAD_DIM + d] = V[v_global_idx];
            }
        }
        __syncthreads();
        
        // Compute attention scores for this block
        float block_scores[32]; // Max block size we support
        for (int i = 0; i < min(block_size, 32); ++i) {
            float score = 0.0f;
            for (int d = 0; d < HEAD_DIM; ++d) {
                score += Q[q_offset + d] * s_K[i * HEAD_DIM + d];
            }
            block_scores[i] = score * scale;
        }
        
        // Online softmax update
        float old_max = max_score;
        for (int i = 0; i < min(block_size, 32); ++i) {
            max_score = fmaxf(max_score, block_scores[i]);
        }
        
        float correction = expf(old_max - max_score);
        sum_exp *= correction;
        
        for (int d = 0; d < HEAD_DIM; ++d) {
            output_acc[d] *= correction;
        }
        
        // Add contribution from current block
        for (int i = 0; i < min(block_size, 32); ++i) {
            float prob = expf(block_scores[i] - max_score);
            sum_exp += prob;
            
            for (int d = 0; d < HEAD_DIM; ++d) {
                output_acc[d] += prob * s_V[i * HEAD_DIM + d];
            }
        }
        
        __syncthreads();
    }
    
    // Final normalization
    float inv_sum = 1.0f / fmaxf(sum_exp, 1e-20f);
    for (int d = 0; d < HEAD_DIM; ++d) {
        output[q_offset + d] = output_acc[d] * inv_sum;
    }
}

// Fallback kernel for unsupported head dimensions
__global__ void multihead_attention_kernel_fallback(
    const float* Q, const float* K, const float* V, float* output,
    int B, int S, int E, int H
) {
    int token_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int head_idx = blockIdx.y;
    
    if (token_idx >= B * S || head_idx >= H) return;
    
    int head_dim = E / H;
    const float scale = 1.0f / sqrtf((float)head_dim);
    
    int batch_idx = token_idx / S;
    
    const float* q_vec = Q + token_idx * E + head_idx * head_dim;
    float* out_vec = output + token_idx * E + head_idx * head_dim;
    
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
    if (H <= 0 || (E % H) != 0) { 
        fprintf(stderr, "Bad MHA config: E=%d H=%d\n", E, H); 
        abort(); 
    }
    
    const float* d_Q = d_qkv;
    const float* d_K = d_qkv + (size_t)B * S * E;
    const float* d_V = d_qkv + 2 * (size_t)B * S * E;

    int head_dim = E / H;

    // Choose optimized FlashAttention kernel for supported head dimensions
    if ((head_dim == 64 || head_dim == 32) && S <= 512) {
        dim3 blocks(B, H, (S + 31) / 32);    // (batch, heads, query tiles)
        dim3 threads(32, min(32, S));        // (threads per query, queries per block)
        size_t smem_size = 2 * head_dim * threads.y * sizeof(float);

        if (head_dim == 64) {
            hipLaunchKernelGGL(flash_attention_kernel<64>, blocks, threads, smem_size, 0,
                               d_Q, d_K, d_V, d_output, B, S, E, H);
        } else {
            hipLaunchKernelGGL(flash_attention_kernel<32>, blocks, threads, smem_size, 0,
                               d_Q, d_K, d_V, d_output, B, S, E, H);
        }
    } else {
        // Fallback to simple kernel
        int total_tokens = B * S;
        int threads_per_block = 256;
        int blocks_x = (total_tokens + threads_per_block - 1) / threads_per_block;
        
        dim3 blocks(blocks_x, H);
        dim3 threads(threads_per_block);

        hipLaunchKernelGGL(multihead_attention_kernel_fallback, blocks, threads, 0, 0,
                           d_Q, d_K, d_V, d_output, B, S, E, H);
    }
}


// ---------------- Scaled Dot Product Attention (REPLACED BY FLASH ATTENTION) ----------------
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
    int total = B * S;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    hipLaunchKernelGGL(scaled_dot_product_kernel, dim3(blocks), dim3(threads), 0, 0, Q, K, scores, B, S, D);
}

// ============================================================================
// Softmax with Better Numerical Stability
// ============================================================================
__global__ void softmax_kernel(const float* scores, float* softmax_out, int B, int S) {
    int token = blockIdx.x * blockDim.x + threadIdx.x;
    if (token >= B * S) return;

    const float* score_row = scores + token * S;
    float* softmax_row = softmax_out + token * S;

    // Find max with better vectorization
    float max_score = -1e30f;
    for (int i = 0; i < S; ++i) {
        max_score = fmaxf(max_score, score_row[i]);
    }

    // Compute exp and sum in single pass
    float sum_exp = 0.0f;
    for (int i = 0; i < S; ++i) {
        float exp_val = expf(score_row[i] - max_score);
        softmax_row[i] = exp_val;
        sum_exp += exp_val;
    }
    
    // Final normalization
    float inv_sum = 1.0f / fmaxf(sum_exp, 1e-20f);
    for (int i = 0; i < S; ++i) {
        softmax_row[i] *= inv_sum;
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

// ============================================================================
// Embedding Lookup with Vectorized Access
// ============================================================================
__global__ void embedding_lookup_vectorized(
    const int* token_ids,
    const float* token_embed,
    const float* pos_embed,
    float* output,
    int B, int S, int V, int E
) {
    int batch_seq_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int embed_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (batch_seq_idx >= B * S) return;
    
    int batch_idx = batch_seq_idx / S;
    int seq_idx = batch_seq_idx % S;
    
    int token_id = token_ids[batch_seq_idx];
    if (token_id < 0 || token_id >= V) {
        if (embed_idx < E) {
            output[batch_seq_idx * E + embed_idx] = 0.0f;
        }
        return;
    }
    
    // Process multiple embedding dimensions per thread when possible
    const int VEC_SIZE = 4;
    int vec_start = (embed_idx / VEC_SIZE) * VEC_SIZE;
    
    if (embed_idx % VEC_SIZE == 0 && vec_start + VEC_SIZE <= E) {
        // Vectorized access
        float4* output_vec = reinterpret_cast<float4*>(&output[batch_seq_idx * E + vec_start]);
        const float4* token_vec = reinterpret_cast<const float4*>(&token_embed[token_id * E + vec_start]);
        const float4* pos_vec = reinterpret_cast<const float4*>(&pos_embed[seq_idx * E + vec_start]);
        
        float4 t_val = *token_vec;
        float4 p_val = *pos_vec;
        *output_vec = make_float4(
            t_val.x + p_val.x, t_val.y + p_val.y,
            t_val.z + p_val.z, t_val.w + p_val.w
        );
    } else if (embed_idx < E) {
        // Scalar access for remainder
        output[batch_seq_idx * E + embed_idx] = 
            token_embed[token_id * E + embed_idx] + pos_embed[seq_idx * E + embed_idx];
    }
}

void launch_embedding_lookup(
    const int* d_token_ids,
    const float* d_token_embed,
    const float* d_pos_embed,
    float* d_output,
    int batch_size, int seq_len, int vocab_size, int embed_dim
) {
    dim3 threads(16, 16);
    dim3 blocks(
        (batch_size * seq_len + threads.x - 1) / threads.x,
        (embed_dim + threads.y - 1) / threads.y
    );

    hipLaunchKernelGGL(embedding_lookup_vectorized, blocks, threads, 0, 0,
        d_token_ids, d_token_embed, d_pos_embed, d_output,
        batch_size, seq_len, vocab_size, embed_dim);
}

// ============================================================================
// Softmax + Cross-Entropy Loss with Better Numerics
// ============================================================================
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

    const float* logit_row = logits + idx * V;
    float* softmax_row = softmax_out + idx * V;
    float* grad_row = grad_out + idx * V;

    // Find max with better loop unrolling
    float max_logit = -1e30f;
    for (int i = 0; i < V; ++i) {
        max_logit = fmaxf(max_logit, logit_row[i]);
    }

    // Compute exp and sum in single pass
    float sum_exp = 0.0f;
    for (int i = 0; i < V; ++i) {
        float exp_val = expf(logit_row[i] - max_logit);
        softmax_row[i] = exp_val;
        sum_exp += exp_val;
    }

    int label = labels[idx];
    if (label < 0 || label >= V) {
        // Invalid label case
        for (int i = 0; i < V; ++i) {
            softmax_row[i] = 0.0f;
            grad_row[i] = 0.0f;
        }
        return;
    }

    sum_exp = fmaxf(sum_exp, 1e-20f);
    float inv_sum = 1.0f / sum_exp;
    
    // More numerically stable loss computation
    float log_prob = logit_row[label] - max_logit - logf(sum_exp);
    atomicAdd(loss_sum, -log_prob);

    // Compute final probabilities and gradients
    for (int i = 0; i < V; ++i) {
        float prob = softmax_row[i] * inv_sum;
        softmax_row[i] = prob;
        grad_row[i] = prob - (i == label ? 1.0f : 0.0f);
    }
}

float launch_softmax_loss(
    const float* logits,
    float* softmax_out,
    const int* labels,
    float* grad_out,
    int B, int C
) {
    int N = B;
    int V = C;

    float* d_loss_sum;
    hipMalloc(&d_loss_sum, sizeof(float));
    hipMemset(d_loss_sum, 0, sizeof(float));

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    hipLaunchKernelGGL(softmax_loss_kernel, dim3(blocks), dim3(threads), 0, 0,
        logits, softmax_out, labels, grad_out, d_loss_sum, N, V);

    float loss;
    hipMemcpy(&loss, d_loss_sum, sizeof(float), hipMemcpyDeviceToHost);
    hipFree(d_loss_sum);

    return loss / N;
}

// ---------------- Accuracy ----------------
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

// ---------------- Add Inplace ----------------
__global__ void add_inplace_kernel(float* a, const float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] += b[i];
}
void launch_add_inplace(float* a, const float* b, int n) {
    int threads = 256;
    int blocks  = (n + threads - 1) / threads;
    hipLaunchKernelGGL(add_inplace_kernel, dim3(blocks), dim3(threads), 0, 0, a, b, n);
}

// ---------------- ReLU Backprop ----------------
__global__ void relu_backprop_kernel(const float* act, const float* grad_out, float* grad_in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) grad_in[i] = act[i] > 0.0f ? grad_out[i] : 0.0f;
}
void launch_backprop_activation(const float* act, const float* grad_out, float* grad_in, int n) {
    int threads = 256;
    int blocks  = (n + threads - 1) / threads;
    hipLaunchKernelGGL(relu_backprop_kernel, dim3(blocks), dim3(threads), 0, 0, act, grad_out, grad_in, n);
}

// ---------------- SGD Update ----------------
__global__ void sgd_update_kernel(float* w, const float* g, float lr, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) w[i] -= lr * g[i];
}
void launch_sgd_update(float* w, const float* g, float lr, int n) {
    int threads = 256;
    int blocks  = (n + threads - 1) / threads;
    hipLaunchKernelGGL(sgd_update_kernel, dim3(blocks), dim3(threads), 0, 0, w, g, lr, n);
}

// ============================================================================
// Layer Normalization with Efficient Reductions
// ============================================================================
__global__ void layer_norm_forward(
    const float* input, float* output, const float* gamma, const float* beta,
    int N, int E, float eps = 1e-5f
) {
    int n = blockIdx.x;
    int tid = threadIdx.x;
    int threads_per_row = blockDim.x;
    
    if (n >= N) return;
    
    extern __shared__ float smem[];
    float* row_data = smem;
    
    // Load row into shared memory with coalescing
    for (int e = tid; e < E; e += threads_per_row) {
        row_data[e] = input[n * E + e];
    }
    __syncthreads();
    
    // Parallel reduction for mean
    float local_sum = 0.0f;
    for (int e = tid; e < E; e += threads_per_row) {
        local_sum += row_data[e];
    }
    
    // Warp-level reduction
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        local_sum += __shfl_down(local_sum, offset);
    }
    
    // Block-level reduction
    __shared__ float block_sum[32];  // Max warps per block
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    
    if (lane_id == 0) block_sum[warp_id] = local_sum;
    __syncthreads();
    
    if (warp_id == 0) {
        float sum = (tid < (threads_per_row + WARP_SIZE - 1) / WARP_SIZE) ? block_sum[tid] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down(sum, offset);
        }
        if (tid == 0) block_sum[0] = sum / E;  // mean
    }
    __syncthreads();
    
    float mean = block_sum[0];
    
    // Parallel reduction for variance
    float local_var = 0.0f;
    for (int e = tid; e < E; e += threads_per_row) {
        float diff = row_data[e] - mean;
        local_var += diff * diff;
    }
    
    // Similar reduction for variance
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        local_var += __shfl_down(local_var, offset);
    }
    
    if (lane_id == 0) block_sum[warp_id] = local_var;
    __syncthreads();
    
    if (warp_id == 0) {
        float var = (tid < (threads_per_row + WARP_SIZE - 1) / WARP_SIZE) ? block_sum[tid] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            var += __shfl_down(var, offset);
        }
        if (tid == 0) block_sum[0] = rsqrtf(var / E + eps);  // inv_std
    }
    __syncthreads();
    
    float inv_std = block_sum[0];
    
    // Apply normalization
    for (int e = tid; e < E; e += threads_per_row) {
        output[n * E + e] = gamma[e] * (row_data[e] - mean) * inv_std + beta[e];
    }
}

void launch_layer_norm_forward(const float* input, float* output, const float* gamma, const float* beta, int N, int E) {
    int threads = min(E, MAX_THREADS_PER_BLOCK);
    // Ensure threads is power of 2 for efficient reductions
    int pow2_threads = 1;
    while (pow2_threads < threads) pow2_threads <<= 1;
    threads = min(pow2_threads, MAX_THREADS_PER_BLOCK);
    
    size_t smem_size = E * sizeof(float);
    hipLaunchKernelGGL(layer_norm_forward, dim3(N), dim3(threads), smem_size, 0,
        input, output, gamma, beta, N, E, 1e-5f);
}

// ---------------- Layer Norm Backward (KEEP AS IS BUT OPTIMIZED) ----------------
__global__ void layernorm_backward_kernel(
    const float* __restrict__ grad_output, // [N,E]
    const float* __restrict__ input,       // [N,E]
    float* __restrict__ grad_input,        // [N,E]
    const float* __restrict__ gamma,       // [E]
    float* __restrict__ grad_gamma,        // [E]
    float* __restrict__ grad_beta,         // [E]
    int N, int E, float eps=1e-5f)
{
    int n = blockIdx.x;         // one row per block
    int e = threadIdx.x;        // one feature per thread
    if (n >= N || e >= E) return;

    extern __shared__ float shm[]; // 3*E floats: xhat, dy, tmp
    float* xhat = shm;
    float* dy_g = shm + E;
    float* buf  = shm + 2*E;

    // compute mean/var across E
    float x = input[n*E + e];
    float go = grad_output[n*E + e];

    // parallel reduce mean
    buf[e] = x;
    __syncthreads();
    // reduction
    for (int stride=E/2; stride>0; stride>>=1) {
        if (e < stride) buf[e] += buf[e+stride];
        __syncthreads();
    }
    float mean = buf[0] / E;

    // variance
    float xm = x - mean;
    buf[e] = xm * xm;
    __syncthreads();
    for (int stride=E/2; stride>0; stride>>=1) {
        if (e < stride) buf[e] += buf[e+stride];
        __syncthreads();
    }
    float var = buf[0] / E;
    float inv_std = rsqrtf(var + eps);

    // stash xhat and dy*gamma for second reductions
    float g = gamma[e];
    float xh = xm * inv_std;
    xhat[e] = xh;
    dy_g[e] = go * g;
    __syncthreads();

    // sum(dy*gamma) and sum(dy*gamma*xhat)
    buf[e] = dy_g[e];
    __syncthreads();
    for (int stride=E/2; stride>0; stride>>=1) {
        if (e < stride) buf[e] += buf[e+stride];
        __syncthreads();
    }
    float sum1 = buf[0]; // ∑ dy*gamma

    buf[e] = dy_g[e] * xhat[e];
    __syncthreads();
    for (int stride=E/2; stride>0; stride>>=1) {
        if (e < stride) buf[e] += buf[e+stride];
        __syncthreads();
    }
    float sum2 = buf[0]; // ∑ dy*gamma*xhat

    // dx
    float dx = (dy_g[e] - sum1 / E - xhat[e] * sum2 / E) * inv_std;
    grad_input[n*E + e] = dx;

    // dgamma/dbeta (atomics across N)
    atomicAdd(&grad_gamma[e], go * xh);
    atomicAdd(&grad_beta[e],  go);
}

void launch_layer_norm_backward(const float* grad_output, const float* input,
                                float* grad_input, const float* gamma,
                                float* grad_gamma, float* grad_beta, int N, int E)
{
    // zero the parameter grads before accumulate
    hipMemset(grad_gamma, 0, E*sizeof(float));
    hipMemset(grad_beta,  0, E*sizeof(float));

    dim3 blocks(N);
    int threads = 1;
    while (threads < E && threads < 1024) threads <<= 1; // pow2 for simple reductions
    size_t shmem = (size_t) (3 * threads) * sizeof(float);
    hipLaunchKernelGGL(layernorm_backward_kernel, blocks, dim3(threads), shmem, 0,
                       grad_output, input, grad_input, gamma, grad_gamma, grad_beta, N, E, 1e-5f);
}

// ---------------- Dropout Forward ----------------
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
    int threads = 256;
    int blocks = (N * E + threads - 1) / threads;
    hipLaunchKernelGGL(dropout_forward_kernel, dim3(blocks), dim3(threads), 0, 0, input, output, mask, p, N, E);
}

// ---------------- Dropout Backward ----------------
__global__ void dropout_backward_kernel(const float* grad_output, const float* mask, float p, float* grad_input, int N, int E) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * E) return;
    grad_input[idx] = (mask[idx] == 0.0f) ? 0.0f : (grad_output[idx] / (1.0f - p));
}

void launch_dropout_backward(const float* grad_output, const float* mask, float p, float* grad_input, int N, int E) {
    int threads = 256;
    int blocks = (N * E + threads - 1) / threads;
    hipLaunchKernelGGL(dropout_backward_kernel, dim3(blocks), dim3(threads), 0, 0, grad_output, mask, p, grad_input, N, E);
}

// ---------------- Adam Optimizer ----------------
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
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    hipLaunchKernelGGL(adam_update_kernel, dim3(blocks), dim3(threads), 0, 0,
        weights, grads, m, v, lr, beta1, beta2, epsilon, t, size);
}

// ---------------- Sampling ----------------
__global__ void sample_from_logits_kernel(const float* logits, int* output_token, int vocab_size, int k, float temperature) {
    extern __shared__ unsigned char smem_u8_logits[]; 
    float* top_k_scores = reinterpret_cast<float*>(smem_u8_logits);
    int*   top_k_indices = reinterpret_cast<int*>(top_k_scores + k);

    temperature = fmaxf(temperature, 1e-6f);

    float max_logit = -INFINITY;
    for (int i = 0; i < vocab_size; ++i) {
        float scaled = logits[i] / temperature;
        if (scaled > max_logit) max_logit = scaled;
    }
    float sum_exp = 0.0f;
    for (int i = 0; i < vocab_size; ++i) sum_exp += expf(logits[i] / temperature - max_logit);
    sum_exp = fmaxf(sum_exp, 1e-20f);

    for (int i = 0; i < k; ++i) { top_k_scores[i] = -INFINITY; top_k_indices[i] = -1; }

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

    float mass = 0.0f;
    for (int i = 0; i < k; ++i) mass += fmaxf(top_k_scores[i], 0.0f);
    if (mass <= 0.0f) { *output_token = (top_k_indices[0] >= 0) ? top_k_indices[0] : 0; return; }

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
    if (k < 1) k = 1;
    if (k > vocab_size) k = vocab_size;
    int shared_mem_size = k * (sizeof(float) + sizeof(int));
    hipLaunchKernelGGL(sample_from_logits_kernel, 1, 1, shared_mem_size, 0,
        d_logits, d_output_token, vocab_size, k, temperature);
}

// ---------------- Embedding Backward ----------------
__global__ void embedding_backward_kernel(const float* grad_out, const int* token_ids,
                                          float* grad_token_embed, float* grad_pos_embed,
                                          int B, int S, int E, int V) {
    int b = blockIdx.x;
    int s = blockIdx.y;
    int e = blockIdx.z * blockDim.x + threadIdx.x;
    if (b >= B || s >= S || e >= E) return;

    int token_id = token_ids[b * S + s];
    if (token_id < 0 || token_id >= V) return;

    const float* g = grad_out + (b * S + s) * E + e;
    atomicAdd(&grad_token_embed[token_id * E + e], *g);
    atomicAdd(&grad_pos_embed[s * E + e], *g);
}

void launch_embedding_backward(const float* grad_out, const int* token_ids,
                               float* grad_token_embed, float* grad_pos_embed,
                               int B, int S, int E, int V) {
    int tpb = (E >= 1024) ? 1024 : E;
    int bz  = (E + tpb - 1) / tpb;
    dim3 blocks(B, S, bz), threads(tpb);
    hipLaunchKernelGGL(embedding_backward_kernel, blocks, threads, 0, 0,
                       grad_out, token_ids, grad_token_embed, grad_pos_embed, B, S, E, V);
}

// ---------------- Matrix Transpose Operations ----------------
__global__ void matmul_transpose_A_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    int total_elements = K * N;
    if (idx >= total_elements) return;
    
    int row = idx / N;
    int col = idx % N;
    
    if (row >= K || col >= N) return;

    float sum = 0.0f;
    
    const int CHUNK_SIZE = 4;
    
    for (int chunk_start = 0; chunk_start < M; chunk_start += CHUNK_SIZE) {
        int chunk_end = min(chunk_start + CHUNK_SIZE, M);
        
        for (int i = chunk_start; i < chunk_end; ++i) {
            sum += A[i * K + row] * B[i * N + col];
        }
        
        if ((chunk_start > 0) && (chunk_start % (CHUNK_SIZE * 2) == 0)) {
            __threadfence();
        }
    }
    
    C[row * N + col] = sum;
}

void launch_matmul_transpose_A(const float* A, const float* B, float* C,
                               int M, int N, int K) {
    const size_t total = (size_t)K * (size_t)N;
    const int threads = 256;
    const int blocks  = (int)((total + threads - 1) / threads);
    hipLaunchKernelGGL(matmul_transpose_A_kernel, dim3(blocks), dim3(threads), 0, 0,
                       A, B, C, M, N, K);
}

__global__ void matmul_transpose_B_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    int total_elements = M * K;
    if (idx >= total_elements) return;
    
    int row = idx / K;
    int col = idx % K;
    
    if (row >= M || col >= K) return;

    float sum = 0.0f;
    
    for (int i = 0; i < N; ++i) {
        sum += A[row * N + i] * B[col * N + i];
    }
    
    C[row * K + col] = sum;
}

void launch_matmul_transpose_B(const float* A, const float* B, float* C,
                               int M, int N, int K) {
    const size_t total = (size_t)M * (size_t)K;
    const int threads = 256;
    const int blocks  = (int)((total + threads - 1) / threads);
    hipLaunchKernelGGL(matmul_transpose_B_kernel, dim3(blocks), dim3(threads), 0, 0,
                       A, B, C, M, N, K);
}

// ============================================================================
// Multi-Head Attention Backward (OPTIMIZED)
// ============================================================================
__global__ void multihead_attention_backward_kernel(
    const float* grad_attn_out,
    const float* q_in,
    const float* k_in,
    const float* v_in,
    float* grad_q_out,
    float* grad_k_out,
    float* grad_v_out,
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

    // Find max score for numerical stability
    float max_s = -1e30f;
    for (int j = 0; j < S; ++j) {
        const float* k_vec = k_in + (batch_start_idx + j) * E + head_h * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) dot += q_vec[d] * k_vec[d];
        float s = dot * scale;
        if (s > max_s) max_s = s;
    }

    // Compute attention denominator
    float denom = 0.0f;
    for (int j = 0; j < S; ++j) {
        const float* k_vec = k_in + (batch_start_idx + j) * E + head_h * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) dot += q_vec[d] * k_vec[d];
        denom += expf(dot * scale - max_s);
    }
    denom = fmaxf(denom, 1e-20f);

    // Compute sum term for gradient computation
    float sum_term = 0.0f;
    for (int j = 0; j < S; ++j) {
        const float* k_vec = k_in + (batch_start_idx + j) * E + head_h * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) dot += q_vec[d] * k_vec[d];
        float p = expf(dot * scale - max_s) / denom;

        const float* v_vec = v_in + (batch_start_idx + j) * E + head_h * head_dim;
        float g = 0.0f;
        for (int d = 0; d < head_dim; ++d) g += gho[d] * v_vec[d];

        sum_term += g * p;

        // Accumulate V gradients
        for (int d = 0; d < head_dim; ++d) {
            atomicAdd(&grad_v_out[(batch_start_idx + j) * E + head_h * head_dim + d], p * gho[d]);
        }
    }

    // Compute Q and K gradients
    for (int d = 0; d < head_dim; ++d) {
        float acc_qd = 0.0f;
        
        for (int j = 0; j < S; ++j) {
            const float* k_vec = k_in + (batch_start_idx + j) * E + head_h * head_dim;
            float dot = 0.0f;
            for (int dd = 0; dd < head_dim; ++dd) dot += q_vec[dd] * k_vec[dd];
            float p = expf(dot * scale - max_s) / denom;

            const float* v_vec = v_in + (batch_start_idx + j) * E + head_h * head_dim;
            float g = 0.0f;
            for (int dd = 0; dd < head_dim; ++dd) g += gho[dd] * v_vec[dd];

            float grad_score = (g - sum_term) * p;
            float w = grad_score * scale;

            acc_qd += w * k_vec[d];
            
            atomicAdd(&grad_k_out[(batch_start_idx + j) * E + head_h * head_dim + d], w * q_vec[d]);
        }
        
        grad_q_out[token_i * E + head_h * head_dim + d] = acc_qd;
    }
}

void launch_multihead_attention_backward(
    const float* d_grad_attn_output, const float* d_qkv, const float* /*d_softmax*/,
    float* d_grad_qkv, int B, int S, int E, int H
) {
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

    hipMemset(d_grad_qkv, 0, (size_t)B * S * E * 3 * sizeof(float));

    int total_tokens = B * S;
    int threads_per_block = 256;
    int blocks_x = (total_tokens + threads_per_block - 1) / threads_per_block;
    
    dim3 blocks(blocks_x, H);
    dim3 threads(threads_per_block);

    hipLaunchKernelGGL(multihead_attention_backward_kernel, blocks, threads, 0, 0,
        d_grad_attn_output, d_Q, d_K, d_V, d_grad_Q, d_grad_K, d_grad_V, B, S, E, H);
}

// ============================================================================
// Gradient clipping (device-side): L2 accumulate and scaling
// ============================================================================
__global__ void l2_accumulate_kernel(const float* __restrict__ g,
                                     int n,
                                     float* __restrict__ total_out) {
    // Blocked grid-stride loop + block reduction -> one atomic per block
    float local = 0.0f;

    // grid-stride
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += blockDim.x * gridDim.x) {
        float v = g[i];
        local += v * v;
    }

    // Block reduction in shared memory
    extern __shared__ float shmem[];
    float* sh = shmem; // size == blockDim.x
    int tid = threadIdx.x;

    sh[tid] = local;
    __syncthreads();

    // power-of-two reduction
    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) sh[tid] += sh[tid + offset];
        __syncthreads();
    }

    if (tid == 0) atomicAdd(total_out, sh[0]);
}

__global__ void scale_inplace_kernel(float* __restrict__ g, int n, float scale) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += blockDim.x * gridDim.x) {
        g[i] *= scale;
    }
}

void launch_l2_accumulate(const float* grad, int n, float* d_total_norm_sq) {
    // caller must hipMemset(d_total_norm_sq, 0, sizeof(float)) before the first call
    if (n <= 0) return;
    const int threads = 256;
    const int blocks  = std::min( (n + threads - 1) / threads,  4096 ); // cap blocks
    size_t shmem = threads * sizeof(float);
    hipLaunchKernelGGL(l2_accumulate_kernel, dim3(blocks), dim3(threads), shmem, 0,
                       grad, n, d_total_norm_sq);
}

void launch_scale_inplace(float* grad, int n, float scale) {
    if (n <= 0 || scale == 1.0f) return;
    const int threads = 256;
    const int blocks  = std::min( (n + threads - 1) / threads,  4096 );
    hipLaunchKernelGGL(scale_inplace_kernel, dim3(blocks), dim3(threads), 0, 0,
                       grad, n, scale);
}