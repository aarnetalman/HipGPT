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

void GPTModel::allocate_output_projection() {
    int proj_size = embed_dim_ * vocab_size_;
    std::vector<float> proj_host(proj_size);
    for (auto& w : proj_host) w = ((float)rand() / RAND_MAX - 0.5f) * 0.02f;

    hipMalloc(&d_output_proj_, proj_size * sizeof(float));
    hipMalloc(&d_output_proj_grad_, proj_size * sizeof(float));
    hipMalloc(&d_output_m_, proj_size * sizeof(float));
    hipMalloc(&d_output_v_, proj_size * sizeof(float));

    hipMemcpy(d_output_proj_, proj_host.data(), proj_size * sizeof(float), hipMemcpyHostToDevice);
    hipMemset(d_output_proj_grad_, 0, proj_size * sizeof(float));
    hipMemset(d_output_m_, 0, proj_size * sizeof(float));
    hipMemset(d_output_v_, 0, proj_size * sizeof(float));
}

void GPTModel::forward(const int* d_input_ids, float* d_logits, int batch_size, int seq_len) {
    int total_tokens = batch_size * seq_len;
    float* d_embed_out;
    hipMalloc(&d_embed_out, total_tokens * embed_dim_ * sizeof(float));

    launch_embedding_lookup(
        d_input_ids, d_token_embedding_, d_pos_embedding_, d_embed_out,
        batch_size, seq_len, vocab_size_, embed_dim_
    );

    float* d_temp;
    hipMalloc(&d_temp, total_tokens * embed_dim_ * sizeof(float));
    float* d_input = d_embed_out;

    for (int i = 0; i < num_layers_; ++i) {
        layers_[i]->forward(d_input, d_temp, batch_size, seq_len);
        std::swap(d_input, d_temp);
    }

    launch_matmul(d_input, d_output_proj_, d_logits, total_tokens, embed_dim_, vocab_size_);

    hipFree(d_embed_out);
    hipFree(d_temp);
}

void GPTModel::backward(const int* d_input_ids, const float* d_logits_grad, int batch_size, int seq_len, float learning_rate) {
    int total_tokens = batch_size * seq_len;
    
    // --- Recompute forward pass to get intermediate activations ---
    float* d_embed_out;
    hipMalloc(&d_embed_out, total_tokens * embed_dim_ * sizeof(float));
    launch_embedding_lookup(
        d_input_ids, d_token_embedding_, d_pos_embedding_, d_embed_out,
        batch_size, seq_len, vocab_size_, embed_dim_
    );

    std::vector<const float*> layer_inputs;
    layer_inputs.push_back(d_embed_out);

    float* d_temp;
    hipMalloc(&d_temp, total_tokens * embed_dim_ * sizeof(float));
    float* d_input = d_embed_out;

    for (int i = 0; i < num_layers_; ++i) {
        layers_[i]->forward(d_input, d_temp, batch_size, seq_len);
        std::swap(d_input, d_temp);
        layer_inputs.push_back(d_input);
    }

    // --- Start Backpropagation ---
    float* d_last_grad;
    hipMalloc(&d_last_grad, total_tokens * embed_dim_ * sizeof(float));
    
    // Compute grad w.r.t. last layer output: d_logits_grad @ d_output_proj^T
    launch_matmul_transpose_B(
        d_logits_grad, d_output_proj_, d_last_grad,
        total_tokens, vocab_size_, embed_dim_
    );

    // Compute output projection gradient: (last_layer_output)^T @ d_logits_grad
    launch_matmul_transpose_A(
        layer_inputs.back(), d_logits_grad, d_output_proj_grad_,
        total_tokens, /*N=*/vocab_size_, /*K=*/embed_dim_
    );

    int step_t = 1; // This should be tracked globally for a proper learning rate schedule
    launch_adam_update(
        d_output_proj_, d_output_proj_grad_, d_output_m_, d_output_v_,
        learning_rate, 0.9f, 0.999f, 1e-8f, step_t,
        embed_dim_ * vocab_size_
    );
    
    // Backprop through layers in reverse
    for (int i = num_layers_ - 1; i >= 0; --i) {
        layers_[i]->backward(layer_inputs[i], d_last_grad, d_last_grad, batch_size, seq_len, learning_rate);
    }
    
    // --- Backward pass for embeddings ---
    float* d_grad_token_embedding;
    float* d_grad_pos_embedding;
    hipMalloc(&d_grad_token_embedding, vocab_size_ * embed_dim_ * sizeof(float));
    hipMalloc(&d_grad_pos_embedding, max_seq_len_ * embed_dim_ * sizeof(float));
    hipMemset(d_grad_token_embedding, 0, vocab_size_ * embed_dim_ * sizeof(float));
    hipMemset(d_grad_pos_embedding, 0, max_seq_len_ * embed_dim_ * sizeof(float));
    
    launch_embedding_backward(
        d_last_grad, d_input_ids,
        d_grad_token_embedding, d_grad_pos_embedding,
        batch_size, seq_len, embed_dim_, vocab_size_
    );

    launch_adam_update(
        d_token_embedding_, d_grad_token_embedding, d_token_m_, d_token_v_,
        learning_rate, 0.9f, 0.999f, 1e-8f, step_t,
        vocab_size_ * embed_dim_
    );
    launch_adam_update(
        d_pos_embedding_, d_grad_pos_embedding, d_pos_m_, d_pos_v_,
        learning_rate, 0.9f, 0.999f, 1e-8f, step_t,
        max_seq_len_ * embed_dim_
    );

    hipFree(d_embed_out);
    hipFree(d_temp);
    hipFree(d_last_grad);
    hipFree(d_grad_token_embedding);
    hipFree(d_grad_pos_embedding);
}

std::vector<int> GPTModel::generate(const std::vector<int>& prompt_ids, int max_new_tokens, int top_k, float temperature) {
    std::vector<int> output = prompt_ids;

    int* d_input_ids;
    float* d_logits;
    int* d_next_token;
    hipMalloc(&d_input_ids, max_seq_len_ * sizeof(int));
    hipMalloc(&d_logits, max_seq_len_ * vocab_size_ * sizeof(float));
    hipMalloc(&d_next_token, sizeof(int));

    for (int step = 0; step < max_new_tokens; ++step) {
        // Inefficient: Re-copies the entire context every time. A better approach uses a KV cache on the GPU.
        int cur_len = std::min((int)output.size(), max_seq_len_);
        std::vector<int> input_ids(max_seq_len_, 0); // Pad with 0
        std::copy(output.end() - cur_len, output.end(), input_ids.begin() + (max_seq_len_ - cur_len));

        hipMemcpy(d_input_ids, input_ids.data(), max_seq_len_ * sizeof(int), hipMemcpyHostToDevice);

        forward(d_input_ids, d_logits, 1, max_seq_len_);

        launch_sample_from_logits(
            &d_logits[(max_seq_len_ - 1) * vocab_size_], // Logits for the last token
            d_next_token,
            vocab_size_,
            top_k,
            temperature
        );
        
        int next_token;
        hipMemcpy(&next_token, d_next_token, sizeof(int), hipMemcpyDeviceToHost);
        output.push_back(next_token);
    }

    hipFree(d_input_ids);
    hipFree(d_logits);
    hipFree(d_next_token);

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