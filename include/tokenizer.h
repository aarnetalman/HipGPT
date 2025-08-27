#pragma once
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <vector>
#include <utility>

// Custom hash for string pairs
struct PairHash {
    size_t operator()(const std::pair<std::string,std::string>& p) const {
        return std::hash<std::string>()(p.first) ^ (std::hash<std::string>()(p.second) << 1);
    }
};

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

    // Cache: word → tokenized form
    std::unordered_map<std::string, std::vector<std::string>> token_cache;

    // Merge ranks: pair → order
    std::unordered_map<std::pair<std::string,std::string>, int, PairHash> merge_rank;

    // Internal helpers
    std::vector<std::string> split_word(const std::string& word);
    std::vector<std::string> encode_word_as_tokens(const std::string& word);
};
