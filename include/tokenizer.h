#pragma once
#include <unordered_map>
#include <string>
#include <vector>
#include <map>
#include <utility>

struct PairHash; // forward declare

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

    // vocab
    std::unordered_map<std::string, int> token_to_stoi;
    std::vector<std::string> stoi_to_token;

    // cache for encoding
    std::unordered_map<std::string, std::vector<std::string>> token_cache;

    // merge order (string pairs → rank)
    std::map<std::pair<std::string, std::string>, int> merge_rank;

    // helpers
    std::vector<std::string> split_word(const std::string& word);
    std::vector<std::string> encode_word_as_tokens(const std::string& word);
};
