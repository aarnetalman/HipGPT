#pragma once
#include <unordered_map>
#include <string>
#include <vector>
#include <set>
#include <map>
#include <utility>

class Tokenizer {
public:
    Tokenizer(int vocab_limit = 5000);

    void train_bpe(const std::string& text);
    std::vector<int> encode(const std::string& text);
    std::string decode(const std::vector<int>& ids);

    int vocab_size() const { return stoi_to_token.size(); }

    void save(const std::string& filepath) const;
    void load(const std::string& filepath);

private:
    int vocab_limit_;

    std::unordered_map<std::string, int> token_to_stoi;
    std::vector<std::string> stoi_to_token;

    // For tracking frequency of token pairs during training
    std::map<std::pair<std::string, std::string>, int> pair_freq;

    // For caching word-to-token mappings to speed up encoding
    std::unordered_map<std::string, std::vector<std::string>> token_cache;

    std::vector<std::string> split_word(const std::string& word);
    std::vector<std::string> encode_word_as_tokens(const std::string& word);
};