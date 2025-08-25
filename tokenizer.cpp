#include "tokenizer.h"
#include <sstream>
#include <set>
#include <map>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <limits>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

Tokenizer::Tokenizer(int vocab_limit) : vocab_limit_(vocab_limit) {}

std::vector<std::string> Tokenizer::split_word(const std::string& word) {
    std::vector<std::string> tokens;
    for (char c : word) {
        tokens.push_back(std::string(1, c));
    }
    if (!tokens.empty()) {
        tokens.back() += "</w>";
    }
    return tokens;
}

// Helper to get pair frequencies from a tokenized corpus representation
void get_pair_frequencies(const std::vector<std::vector<std::string>>& word_tokens, std::map<std::pair<std::string, std::string>, int>& pair_freq) {
    pair_freq.clear();
    for (const auto& tokens : word_tokens) {
        for (size_t i = 0; i + 1 < tokens.size(); ++i) {
            pair_freq[{tokens[i], tokens[i+1]}]++;
        }
    }
}

void Tokenizer::train_bpe(const std::string& text) {
    std::istringstream iss(text);
    std::string word;

    // 1. Initialize with character-level vocabulary and tokenize the corpus
    std::set<char> char_vocab;
    std::vector<std::vector<std::string>> word_tokens;
    while (iss >> word) {
        for (char c : word) {
            char_vocab.insert(c);
        }
        word_tokens.push_back(split_word(word));
    }

    for (char c : char_vocab) {
        stoi_to_token.push_back(std::string(1, c));
    }
    // Add the special end-of-word character tokens
    for (char c : char_vocab) {
        stoi_to_token.push_back(std::string(1, c) + "</w>");
    }
    std::sort(stoi_to_token.begin(), stoi_to_token.end());
    
    // Assign initial IDs
    for (size_t i = 0; i < stoi_to_token.size(); ++i) {
        token_to_stoi[stoi_to_token[i]] = i;
    }

    // 2. Main BPE merge loop
    while (vocab_size() < vocab_limit_) {
        get_pair_frequencies(word_tokens, pair_freq);
        if (pair_freq.empty()) break;

        // Find the most frequent pair
        auto best_pair = std::max_element(pair_freq.begin(), pair_freq.end(), 
            [](const auto& a, const auto& b) { return a.second < b.second; })->first;
        
        // Add the new merged token to the vocabulary
        std::string merged = best_pair.first + best_pair.second;
        if (token_to_stoi.count(merged)) continue; // Already exists

        token_to_stoi[merged] = vocab_size();
        stoi_to_token.push_back(merged);

        std::cout << "Merging (" << best_pair.first << ", " << best_pair.second << ") -> " << merged << std::endl;

        // 3. EFFICIENTLY update the tokenized corpus with the new merge
        std::vector<std::vector<std::string>> new_word_tokens;
        for (const auto& tokens : word_tokens) {
            std::vector<std::string> new_tokens;
            for (size_t i = 0; i < tokens.size(); ) {
                if (i + 1 < tokens.size() && tokens[i] == best_pair.first && tokens[i+1] == best_pair.second) {
                    new_tokens.push_back(merged);
                    i += 2;
                } else {
                    new_tokens.push_back(tokens[i]);
                    i += 1;
                }
            }
            new_word_tokens.push_back(new_tokens);
        }
        word_tokens = std::move(new_word_tokens);
    }
    std::cout << "BPE training finished. Final vocab size: " << vocab_size() << std::endl;
}


std::vector<std::string> Tokenizer::encode_word_as_tokens(const std::string& word) {
    if (token_cache.count(word)) {
        return token_cache[word];
    }
    
    std::vector<std::string> tokens = split_word(word);
    
    while (tokens.size() > 1) {
        std::pair<std::string, std::string> best_pair;
        int best_rank = std::numeric_limits<int>::max();

        // Find the merge with the highest priority (lowest rank/index in vocab)
        for (size_t i = 0; i < tokens.size() - 1; ++i) {
            std::string candidate = tokens[i] + tokens[i+1];
            if (token_to_stoi.count(candidate) && token_to_stoi[candidate] < best_rank) {
                best_rank = token_to_stoi[candidate];
                best_pair = {tokens[i], tokens[i+1]};
            }
        }

        if (best_rank == std::numeric_limits<int>::max()) {
            break; // No more possible merges
        }

        // Apply the best merge found in this pass
        std::vector<std::string> new_tokens;
        for (size_t i = 0; i < tokens.size(); ) {
            if (i + 1 < tokens.size() && tokens[i] == best_pair.first && tokens[i+1] == best_pair.second) {
                new_tokens.push_back(best_pair.first + best_pair.second);
                i += 2;
            } else {
                new_tokens.push_back(tokens[i]);
                i += 1;
            }
        }
        tokens = std::move(new_tokens);
    }
    
    token_cache[word] = tokens;
    return tokens;
}

std::vector<int> Tokenizer::encode(const std::string& text) {
    std::istringstream iss(text);
    std::string word;
    std::vector<int> result;

    while (iss >> word) {
        auto tokens = encode_word_as_tokens(word);
        for (const auto& t : tokens) {
            if (token_to_stoi.count(t)) {
                result.push_back(token_to_stoi[t]);
            } // Note: Silently drop unknown tokens. Can be changed to a special <UNK> token.
        }
    }
    return result;
}

std::string Tokenizer::decode(const std::vector<int>& ids) {
    std::string out;
    for (int id : ids) {
        if (id < 0 || id >= vocab_size()) continue; // Safety check
        std::string token = stoi_to_token[id];
        if (token.size() >= 4 && token.substr(token.size() - 4) == "</w>") {
            out += token.substr(0, token.size() - 4) + " ";
        } else {
            out += token;
        }
    }
    return out;
}

void Tokenizer::save(const std::string& filepath) const {
    json j;
    j["stoi_to_token"] = stoi_to_token;

    std::ofstream out(filepath);
    out << j.dump(2);
}

void Tokenizer::load(const std::string& filepath) {
    std::ifstream in(filepath);
    if (!in) {
        std::cerr << "Error: Cannot open tokenizer file " << filepath << std::endl;
        return;
    }
    json j;
    in >> j;

    stoi_to_token = j["stoi_to_token"].get<std::vector<std::string>>();
    token_to_stoi.clear();
    for (size_t i = 0; i < stoi_to_token.size(); ++i) {
        token_to_stoi[stoi_to_token[i]] = i;
    }
    token_cache.clear(); // Clear cache after loading a new vocab
}