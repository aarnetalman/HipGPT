#include "gpt_model.h"
#include <hip/hip_runtime.h>
#include <cstdlib>
#include <vector>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include "hip_kernels.h"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <thread>
#include <chrono>
#include <numeric>


GPTModel::GPTModel(int vocab_size, int max_seq_len, int embed_dim, int num_heads, int ff_hidden_dim, int num_layers)
    : vocab_size_(vocab_size), max_seq_len_(max_seq_len), embed_dim_(embed_dim), num_layers_(num_layers) {

    allocate_embeddings();
    for (int i = 0; i < num_layers_; ++i) {
        layers_.push_back(new TransformerLayer(embed_dim, num_heads, ff_hidden_dim));
    }
    allocate_output_projection();
}

GPTModel::~GPTModel() {
    if (d_token_embedding_) hipFree(d_token_embedding_);
    if (d_pos_embedding_)   hipFree(d_pos_embedding_);
    if (d_token_m_) hipFree(d_token_m_);
    if (d_token_v_) hipFree(d_token_v_);
    if (d_pos_m_)   hipFree(d_pos_m_);
    if (d_pos_v_)   hipFree(d_pos_v_);
    if (d_output_proj_)      hipFree(d_output_proj_);
    if (d_output_proj_grad_) hipFree(d_output_proj_grad_);
    if (d_output_m_) hipFree(d_output_m_);
    if (d_output_v_) hipFree(d_output_v_);
    if (d_embedded_input_) hipFree(d_embedded_input_);
    if (d_layer_output_)   hipFree(d_layer_output_);
    if (d_last_grad_)      hipFree(d_last_grad_); 

    for (auto* layer : layers_) {
        delete layer;
    }
}

void GPTModel::allocate_embeddings() {
    int token_embed_size = vocab_size_ * embed_dim_;
    int pos_embed_size = max_seq_len_ * embed_dim_;

    std::vector<float> token_host(token_embed_size);
    for (auto& w : token_host) w = ((float)rand() / RAND_MAX - 0.5f) * 0.02f;
    hipMalloc(&d_token_embedding_, token_embed_size * sizeof(float));
    hipMemcpy(d_token_embedding_, token_host.data(), token_embed_size * sizeof(float), hipMemcpyHostToDevice);

    std::vector<float> pos_host(pos_embed_size);
    for (auto& w : pos_host) w = ((float)rand() / RAND_MAX - 0.5f) * 0.02f;
    hipMalloc(&d_pos_embedding_, pos_embed_size * sizeof(float));
    hipMemcpy(d_pos_embedding_, pos_host.data(), pos_embed_size * sizeof(float), hipMemcpyHostToDevice);
    
    hipMalloc(&d_token_m_, token_embed_size * sizeof(float));
    hipMalloc(&d_token_v_, token_embed_size * sizeof(float));
    hipMalloc(&d_pos_m_, pos_embed_size * sizeof(float));
    hipMalloc(&d_pos_v_, pos_embed_size * sizeof(float));
    hipMemset(d_token_m_, 0, token_embed_size * sizeof(float));
    hipMemset(d_token_v_, 0, token_embed_size * sizeof(float));
    hipMemset(d_pos_m_, 0, pos_embed_size * sizeof(float));
    hipMemset(d_pos_v_, 0, pos_embed_size * sizeof(float));
}

void GPTModel::ensure_temp_capacity(int T) {
    if (T <= temp_tokens_cap_) return;
    int new_cap = std::max(T, temp_tokens_cap_ + temp_tokens_cap_/2 + 1); // 1.5x growth

    // Free old (if any)
    if (d_embedded_input_) hipFree(d_embedded_input_);
    if (d_layer_output_)   hipFree(d_layer_output_);
    if (d_last_grad_)      hipFree(d_last_grad_);

    const size_t bytes = (size_t)new_cap * (size_t)embed_dim_ * sizeof(float);
    hipMalloc(&d_embedded_input_, bytes);
    hipMalloc(&d_layer_output_,   bytes);
    hipMalloc(&d_last_grad_,      bytes);
    temp_tokens_cap_ = new_cap;
}

void GPTModel::forward(const int* d_input_ids, float* d_logits,
                       int batch_size, int seq_len)
{
    const int T = batch_size * seq_len;
    ensure_temp_capacity(T);

    float* x   = d_embedded_input_; // current activations
    float* tmp = d_layer_output_;   // scratch ping-pong

    // embeddings → x
    launch_embedding_lookup(d_input_ids, d_token_embedding_, d_pos_embedding_,
                            x, batch_size, seq_len, vocab_size_, embed_dim_);

    // transformer stack (ping-pong x <-> tmp)
    for (int i = 0; i < num_layers_; ++i) {
        layers_[i]->forward(x, tmp, batch_size, seq_len);  // writes tmp
        std::swap(x, tmp);
    }

    // output projection: [T,E] x [E,V] -> [T,V]
    launch_matmul(x, d_output_proj_, d_logits, T, embed_dim_, vocab_size_);
}

void GPTModel::backward(const int* d_input_ids,
                        const float* d_logits_grad,
                        int batch_size, int seq_len,
                        float learning_rate, int adam_t)
{
    const int T = batch_size * seq_len;
    const float max_grad_norm = 1.0f;

    // --- grow-once workspaces: [T,E] ---
    ensure_temp_capacity(T);
    float* d_embed_out = d_embedded_input_; // forward activations
    float* d_temp      = d_layer_output_;   // scratch ping-pong
    float* d_last_grad = d_last_grad_;      // grad wrt last hidden

    // ===== 1) Recompute forward activations (cheap vs storing all layers) =====
    launch_embedding_lookup(
        d_input_ids, d_token_embedding_, d_pos_embedding_, d_embed_out,
        batch_size, seq_len, vocab_size_, embed_dim_
    );

    // Cache per-layer inputs (device pointers) for LN/residual in layer backward
    std::vector<const float*> layer_inputs;
    layer_inputs.reserve(num_layers_ + 1);
    layer_inputs.push_back(d_embed_out);

    float* d_in  = d_embed_out;
    float* d_out = d_temp;
    for (int i = 0; i < num_layers_; ++i) {
        layers_[i]->forward(d_in, d_out, batch_size, seq_len);
        std::swap(d_in, d_out);
        layer_inputs.push_back(d_in); // post-layer activation
    }

    // ===== 2) Grad wrt last hidden from logits grad: dH = dY * W^T =====
    launch_matmul_transpose_B(
        d_logits_grad, d_output_proj_, d_last_grad,
        T, /*N=*/vocab_size_, /*K=*/embed_dim_
    );

    // ===== 3) Output projection grads: dW = H^T * dY =====
    launch_matmul_transpose_A(
        layer_inputs.back(), d_logits_grad, d_output_proj_grad_,
        T, /*N=*/vocab_size_, /*K=*/embed_dim_
    );

    // ===== 4) Backprop through transformer stack (now with REAL lr) =====
    // NOTE: this applies Adam updates INSIDE each layer (no global clip yet)
    for (int i = num_layers_ - 1; i >= 0; --i) {
        layers_[i]->backward(
            layer_inputs[i],       // input to layer i
            d_last_grad,           // grad wrt layer i output (in/out buffer OK)
            d_last_grad,           // grad wrt layer i input  (reuse buffer)
            batch_size, seq_len,
            /*learning_rate=*/learning_rate, // <-- CRITICAL: not 0.0f
            adam_t
        );
    }

    // ===== 5) Embedding grads from d_last_grad =====
    float* d_grad_token_embedding = nullptr;
    float* d_grad_pos_embedding   = nullptr;
    hipMalloc(&d_grad_token_embedding, (size_t)vocab_size_ * embed_dim_ * sizeof(float));
    hipMalloc(&d_grad_pos_embedding,   (size_t)max_seq_len_ * embed_dim_ * sizeof(float));
    hipMemset(d_grad_token_embedding, 0, (size_t)vocab_size_ * embed_dim_ * sizeof(float));
    hipMemset(d_grad_pos_embedding,   0, (size_t)max_seq_len_ * embed_dim_ * sizeof(float));

    launch_embedding_backward(
        d_last_grad, d_input_ids,
        d_grad_token_embedding, d_grad_pos_embedding,
        batch_size, seq_len, embed_dim_, vocab_size_
    );

    // ===== 6) Global grad clipping (CURRENTLY: only for out-proj + embeddings) =====
    std::vector<float*> grad_ptrs;
    std::vector<int>    grad_sizes;

    grad_ptrs.push_back(d_output_proj_grad_);
    grad_sizes.push_back(embed_dim_ * vocab_size_);

    grad_ptrs.push_back(d_grad_token_embedding);
    grad_sizes.push_back(vocab_size_ * embed_dim_);

    grad_ptrs.push_back(d_grad_pos_embedding);
    grad_sizes.push_back(max_seq_len_ * embed_dim_);

    float* d_total_norm_sq = nullptr;
    hipMalloc(&d_total_norm_sq, sizeof(float));
    hipMemset(d_total_norm_sq, 0, sizeof(float));
    for (size_t i = 0; i < grad_ptrs.size(); ++i)
        launch_l2_accumulate(grad_ptrs[i], grad_sizes[i], d_total_norm_sq);

    float h_total_norm_sq = 0.0f;
    hipMemcpy(&h_total_norm_sq, d_total_norm_sq, sizeof(float), hipMemcpyDeviceToHost);
    hipFree(d_total_norm_sq);
    float h_total_norm = std::sqrt(std::max(h_total_norm_sq, 0.0f));

    float clip_scale = 1.0f;
    if (h_total_norm > 0.0f && h_total_norm > max_grad_norm)
        clip_scale = max_grad_norm / h_total_norm;

    if (clip_scale < 1.0f) {
        for (size_t i = 0; i < grad_ptrs.size(); ++i)
            launch_scale_inplace(grad_ptrs[i], grad_sizes[i], clip_scale);
        // (TransformerLayer params are NOT clipped here—see next section)
    }

    // ===== 7) Adam updates for OUT PROJ + EMBEDDINGS =====
    launch_adam_update(
        d_output_proj_, d_output_proj_grad_, d_output_m_, d_output_v_,
        learning_rate, 0.9f, 0.999f, 1e-8f, adam_t,
        embed_dim_ * vocab_size_
    );
    launch_adam_update(
        d_token_embedding_, d_grad_token_embedding, d_token_m_, d_token_v_,
        learning_rate, 0.9f, 0.999f, 1e-8f, adam_t,
        vocab_size_ * embed_dim_
    );
    launch_adam_update(
        d_pos_embedding_, d_grad_pos_embedding, d_pos_m_, d_pos_v_,
        learning_rate, 0.9f, 0.999f, 1e-8f, adam_t,
        max_seq_len_ * embed_dim_
    );

    // ===== 8) cleanup small, per-step buffers =====
    hipFree(d_grad_token_embedding);
    hipFree(d_grad_pos_embedding);
}

std::vector<int> GPTModel::generate(const std::vector<int>& prompt_ids,
                                    int max_new_tokens,
                                    int top_k,
                                    float temperature,
                                    float rep_penalty,
                                    float top_p) {
    std::vector<int> output = prompt_ids;

    int* d_input_ids;
    float* d_logits;
    hipMalloc(&d_input_ids, max_seq_len_ * sizeof(int));
    hipMalloc(&d_logits, max_seq_len_ * vocab_size_ * sizeof(float));

    for (int step = 0; step < max_new_tokens; ++step) {
        // Pad/truncate input
        int cur_len = std::min((int)output.size(), max_seq_len_);
        std::vector<int> input_ids(max_seq_len_, 0);
        std::copy(output.end() - cur_len, output.end(),
                  input_ids.begin() + (max_seq_len_ - cur_len));

        hipMemcpy(d_input_ids, input_ids.data(),
                  max_seq_len_ * sizeof(int),
                  hipMemcpyHostToDevice);

        // Forward
        forward(d_input_ids, d_logits, 1, max_seq_len_);

        // Copy last-token logits back to host
        std::vector<float> h_logits(vocab_size_);
        hipMemcpy(h_logits.data(),
                  &d_logits[(max_seq_len_ - 1) * vocab_size_],
                  vocab_size_ * sizeof(float),
                  hipMemcpyDeviceToHost);

        // ---- 1. Apply repetition penalty ----
        for (int id : output) {
            if (id >= 0 && id < vocab_size_) {
                if (h_logits[id] > 0) h_logits[id] /= rep_penalty;
                else                  h_logits[id] *= rep_penalty;
            }
        }

        // ---- 2. Temperature scaling ----
        for (float& x : h_logits) x /= temperature;

        // ---- 3. Softmax ----
        float max_logit = *std::max_element(h_logits.begin(), h_logits.end());
        float sum = 0.0f;
        for (float& x : h_logits) {
            x = std::exp(x - max_logit);
            sum += x;
        }
        for (float& x : h_logits) x /= sum;

        // ---- 4. Top-k filter ----
        if (top_k > 0 && top_k < vocab_size_) {
            std::vector<int> idx(vocab_size_);
            std::iota(idx.begin(), idx.end(), 0);
            std::partial_sort(idx.begin(), idx.begin() + top_k, idx.end(),
                              [&](int a, int b) { return h_logits[a] > h_logits[b]; });

            std::vector<float> new_probs(vocab_size_, 0.0f);
            float new_sum = 0.0f;
            for (int i = 0; i < top_k; i++) {
                new_probs[idx[i]] = h_logits[idx[i]];
                new_sum += h_logits[idx[i]];
            }
            for (float& p : new_probs) p /= new_sum;
            h_logits.swap(new_probs);
        }

        // ---- 5. Top-p (nucleus) filter ----
        if (top_p < 1.0f) {
            std::vector<int> idx(vocab_size_);
            std::iota(idx.begin(), idx.end(), 0);
            std::sort(idx.begin(), idx.end(),
                      [&](int a, int b) { return h_logits[a] > h_logits[b]; });

            float cum = 0.0f;
            std::vector<float> new_probs(vocab_size_, 0.0f);
            for (int i : idx) {
                cum += h_logits[i];
                new_probs[i] = h_logits[i];
                if (cum >= top_p) break;
            }
            float new_sum = std::accumulate(new_probs.begin(), new_probs.end(), 0.0f);
            for (float& p : new_probs) p /= new_sum;
            h_logits.swap(new_probs);
        }

        // ---- 6. Sample token ----
        float r = (float)rand() / RAND_MAX;
        float cum = 0.0f;
        int next_token = vocab_size_ - 1;
        for (int i = 0; i < vocab_size_; i++) {
            cum += h_logits[i];
            if (r <= cum) { next_token = i; break; }
        }

        output.push_back(next_token);
    }

    hipFree(d_input_ids);
    hipFree(d_logits);
    return output;
}

void GPTModel::save_checkpoint(const std::string& path) const {
    std::ofstream os(path, std::ios::binary);
    if (!os) {
        throw std::runtime_error("save_checkpoint: cannot open " + path);
    }

    os.write(reinterpret_cast<const char*>(&vocab_size_), sizeof(vocab_size_));
    os.write(reinterpret_cast<const char*>(&max_seq_len_), sizeof(max_seq_len_));
    os.write(reinterpret_cast<const char*>(&embed_dim_), sizeof(embed_dim_));
    os.write(reinterpret_cast<const char*>(&num_layers_), sizeof(num_layers_));

    {
        std::vector<float> h_tok(vocab_size_ * embed_dim_);
        std::vector<float> h_pos(max_seq_len_ * embed_dim_);
        hipMemcpy(h_tok.data(), d_token_embedding_, h_tok.size() * sizeof(float), hipMemcpyDeviceToHost);
        hipMemcpy(h_pos.data(), d_pos_embedding_, h_pos.size() * sizeof(float), hipMemcpyDeviceToHost);
        os.write(reinterpret_cast<const char*>(h_tok.data()), h_tok.size() * sizeof(float));
        os.write(reinterpret_cast<const char*>(h_pos.data()), h_pos.size() * sizeof(float));
    }

    for (int i = 0; i < num_layers_; ++i) {
        layers_[i]->save(os);
    }

    {
        std::vector<float> h_proj(embed_dim_ * vocab_size_);
        hipMemcpy(h_proj.data(), d_output_proj_, h_proj.size() * sizeof(float), hipMemcpyDeviceToHost);
        os.write(reinterpret_cast<const char*>(h_proj.data()), h_proj.size() * sizeof(float));
    }
}

void GPTModel::load_checkpoint(const std::string& path) {
    std::ifstream is(path, std::ios::binary);
    if (!is) {
        throw std::runtime_error("load_checkpoint: cannot open " + path);
    }

    int v = 0, s = 0, e = 0, L = 0;
    is.read(reinterpret_cast<char*>(&v), sizeof(v));
    is.read(reinterpret_cast<char*>(&s), sizeof(s));
    is.read(reinterpret_cast<char*>(&e), sizeof(e));
    is.read(reinterpret_cast<char*>(&L), sizeof(L));

    if (v != vocab_size_ || s != max_seq_len_ || e != embed_dim_ || L != num_layers_) {
        throw std::runtime_error("load_checkpoint: model hyperparameters mismatch");
    }

    {
        std::vector<float> h_tok(vocab_size_ * embed_dim_);
        std::vector<float> h_pos(max_seq_len_ * embed_dim_);
        is.read(reinterpret_cast<char*>(h_tok.data()), h_tok.size() * sizeof(float));
        is.read(reinterpret_cast<char*>(h_pos.data()), h_pos.size() * sizeof(float));
        hipMemcpy(d_token_embedding_, h_tok.data(), h_tok.size() * sizeof(float), hipMemcpyHostToDevice);
        hipMemcpy(d_pos_embedding_,  h_pos.data(), h_pos.size()  * sizeof(float), hipMemcpyHostToDevice);
    }

    for (int i = 0; i < num_layers_; ++i) {
        layers_[i]->load(is);
    }

    {
        std::vector<float> h_proj(embed_dim_ * vocab_size_);
        is.read(reinterpret_cast<char*>(h_proj.data()), h_proj.size() * sizeof(float));
        hipMemcpy(d_output_proj_, h_proj.data(), h_proj.size() * sizeof(float), hipMemcpyHostToDevice);
    }
}