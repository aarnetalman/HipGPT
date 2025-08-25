#pragma once

#include "transformer_layer.h"
#include <hip/hip_runtime.h>
#include <vector>

class GPTModel {
public:
    GPTModel(int vocab_size, int max_seq_len, int embed_dim, int num_heads, int ff_hidden_dim, int num_layers);
    ~GPTModel();

    void forward(const int* d_input_ids, float* d_logits, int batch_size, int seq_len);
    void backward(const int* d_input_ids, const float* d_logits, const float* d_loss_grad,
                  int batch_size, int seq_len, float learning_rate);

    float* get_last_hidden();  // Returns pointer to final transformer output [B*S, E]

private:
    int vocab_size_;
    int max_seq_len_;
    int embed_dim_;
    int num_layers_;

    // Embeddings
    float* d_token_embed_;   // [vocab_size, embed_dim]
    float* d_pos_embed_;     // [max_seq_len, embed_dim]

    // Transformer layers
    std::vector<TransformerLayer*> layers_;

    // Final projection
    float* d_output_proj_;       // [embed_dim, vocab_size]
    float* d_output_proj_grad_;  // [embed_dim, vocab_size]

    // Buffers
    float* d_embedded_input_;  // [B*S, E]
    float* d_layer_output_;    // [B*S, E]

    void allocate_embeddings();
    void allocate_output_projection();
    void allocate_temp_buffers(int batch_size, int seq_len);
};

void launch_embedding_lookup(
    const int* d_token_ids,
    const float* d_token_embed,
    const float* d_pos_embed,
    float* d_output,
    int batch_size, int seq_len, int vocab_size, int embed_dim
);

void save_checkpoint(const std::string& path);

std::vector<int> generate(const std::vector<int>& prompt_ids, int max_new_tokens);

