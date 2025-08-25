#pragma once
#include <unordered_map>
#include <string>
#include <vector>
#include <set>

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

    std::unordered_map<std::pair<std::string, std::string>, int, boost::hash<std::pair<std::string, std::string>>> pair_freq;
    std::unordered_map<std::string, std::vector<std::string>> token_cache;

    std::vector<std::string> split_word(const std::string& word);
    void learn_bpe();
    std::string join_tokens(const std::vector<std::string>& tokens);
};