#pragma once
#include "transformer_layer.h"
#include <hip/hip_runtime.h>
#include <vector>
#include <string>

class GPTModel {
public:
    GPTModel(int vocab_size, int max_seq_len, int embed_dim,
             int num_heads, int ff_hidden_dim, int num_layers);
    ~GPTModel();

    void forward(const int* d_input_ids, float* d_logits,
                 int batch_size, int seq_len);

    // Training
    void backward(const int* d_input_ids, const float* d_logits_grad,
                  int batch_size, int seq_len, float learning_rate, int adam_t);

    // Generation
    std::vector<int> generate(const std::vector<int>& prompt_ids,
                              int max_new_tokens,
                              int top_k,
                              float temperature,
                              float rep_penalty = 1.1f,
                              float top_p = 0.9f);

    // Checkpoints
    void save_checkpoint(const std::string& path) const;
    void load_checkpoint(const std::string& path);

private:
    int vocab_size_;
    int max_seq_len_;
    int embed_dim_;
    int num_layers_;

    // Embeddings
    float* d_token_embedding_ = nullptr;
    float* d_pos_embedding_   = nullptr;

    // Adam states for embeddings
    float* d_token_m_ = nullptr;
    float* d_token_v_ = nullptr;
    float* d_pos_m_   = nullptr;
    float* d_pos_v_   = nullptr;

    // Transformer layers
    std::vector<TransformerLayer*> layers_;

    // Final linear projection and its gradient
    float* d_output_proj_       = nullptr;
    float* d_output_proj_grad_  = nullptr;

    // Adam states for output projection
    float* d_output_m_ = nullptr;
    float* d_output_v_ = nullptr;

    // ---- Persistent workspaces (grown on demand) ----
    // Reuse these every step; sized for total_tokens = batch_size * seq_len
    float* d_embedded_input_ = nullptr; // [T, E]
    float* d_layer_output_   = nullptr; // [T, E]
    float* d_last_grad_      = nullptr; // [T, E]  (grad of last activation)
    int    temp_tokens_cap_  = 0;       // capacity in tokens (rows)

    // allocate/grow workspaces if needed
    void ensure_temp_capacity(int total_tokens);

    // allocation helpers
    void allocate_embeddings();
    void allocate_output_projection();
};
