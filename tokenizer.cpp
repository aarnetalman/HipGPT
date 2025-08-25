// tokenizer.cpp
#include "tokenizer.h"
#include <sstream>
#include <set>
#include <map>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;


Tokenizer::Tokenizer(int vocab_limit) : vocab_limit_(vocab_limit) {}

std::vector<std::string> Tokenizer::split_word(const std::string& word) {
    std::vector<std::string> tokens;
    for (char c : word) {
        tokens.push_back(std::string(1, c));
    }
    tokens.back() += "</w>";
    return tokens;
}

std::pair<std::string, std::string> Tokenizer::find_best_pair(const std::map<std::vector<std::string>, int>& corpus) {
    std::map<std::pair<std::string, std::string>, int> pairs;
    for (const auto& [word, freq] : corpus) {
        for (size_t i = 0; i + 1 < word.size(); ++i) {
            pairs[{word[i], word[i + 1]}] += freq;
        }
    }

    std::pair<std::string, std::string> best_pair;
    int max_freq = -1;
    for (const auto& [pair, freq] : pairs) {
        if (freq > max_freq) {
            best_pair = pair;
            max_freq = freq;
        }
    }
    return best_pair;
}

void Tokenizer::train_bpe(const std::string& text) {
    std::map<std::vector<std::string>, int> corpus;
    std::istringstream iss(text);
    std::string word;

    // Initialize corpus with split characters
    while (iss >> word) {
        corpus[split_word(word)]++;
    }

    while ((int)token_to_stoi.size() < vocab_limit_) {
        auto best_pair = find_best_pair(corpus);
        std::string merged = best_pair.first + best_pair.second;

        std::map<std::vector<std::string>, int> new_corpus;

        for (const auto& [word, freq] : corpus) {
            std::vector<std::string> new_word;
            for (size_t i = 0; i < word.size(); ++i) {
                if (i < word.size() - 1 && word[i] == best_pair.first && word[i + 1] == best_pair.second) {
                    new_word.push_back(merged);
                    ++i;
                } else {
                    new_word.push_back(word[i]);
                }
            }
            new_corpus[new_word] += freq;
        }
        corpus = std::move(new_corpus);
    }

    // Assign vocabulary
    std::set<std::string> vocab_set;
    for (const auto& [word, _] : corpus) {
        for (const auto& token : word) {
            vocab_set.insert(token);
        }
    }

    int idx = 0;
    for (const auto& token : vocab_set) {
        token_to_stoi[token] = idx++;
        stoi_to_token.push_back(token);
    }
}

std::vector<int> Tokenizer::encode(const std::string& text) {
    std::istringstream iss(text);
    std::string word;
    std::vector<int> result;

    while (iss >> word) {
        auto tokens = split_word(word);

        while (tokens.size() > 1) {
            bool merged = false;
            for (size_t i = 0; i < tokens.size() - 1; ++i) {
                std::string candidate = tokens[i] + tokens[i + 1];
                if (token_to_stoi.count(candidate)) {
                    tokens[i] = candidate;
                    tokens.erase(tokens.begin() + i + 1);
                    merged = true;
                    break;
                }
            }
            if (!merged) break;
        }

        for (const auto& t : tokens) {
            result.push_back(token_to_stoi.count(t) ? token_to_stoi[t] : 0);
        }
    }

    return result;
}

std::string Tokenizer::decode(const std::vector<int>& ids) {
    std::string out;
    for (int id : ids) {
        std::string token = stoi_to_token[id];
        if (token.size() >= 4 && token.substr(token.size() - 4) == "</w>") {
            token = token.substr(0, token.size() - 4) + " ";
        }
        out += token;
    }
    return out;
}

void Tokenizer::save(const std::string& filepath) const {
    json j;
    j["stoi_to_token"] = stoi_to_token;

    std::ofstream out(filepath);
    out << j.dump(2);  // Pretty print with indent=2
}

void Tokenizer::load(const std::string& filepath) {
    std::ifstream in(filepath);
    json j;
    in >> j;

    stoi_to_token = j["stoi_to_token"].get<std::vector<std::string>>();
    token_to_stoi.clear();
    for (int i = 0; i < stoi_to_token.size(); ++i) {
        token_to_stoi[stoi_to_token[i]] = i;
    }
}
