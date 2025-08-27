#include "tokenizer.h"
#include <sstream>
#include <set>
#include <map>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <limits>
#include <nlohmann/json.hpp>
#include <unordered_map>
#include <queue>
#include <chrono>

using json = nlohmann::json;

Tokenizer::Tokenizer(int vocab_limit) : vocab_limit_(vocab_limit) {}

std::vector<std::string> Tokenizer::split_word(const std::string& word) {
    std::vector<std::string> tokens;
    for (char c : word) {
        tokens.push_back(std::string(1, c));
    }
    if (!tokens.empty()) {
        tokens.back() += "</w>"; // mark end of word
    }
    return tokens;
}

void Tokenizer::train_bpe(const std::string& text) {
    std::istringstream iss(text);
    std::string word;

    // 1. Count word frequencies
    std::unordered_map<std::string, int> word_freq;
    while (iss >> word) {
        word_freq[word]++;
    }

    // 2. Initialize vocab with characters only
    std::set<char> char_vocab;
    for (const auto& [w, f] : word_freq) {
        for (char c : w) char_vocab.insert(c);
    }
    for (char c : char_vocab) {
        stoi_to_token.push_back(std::string(1, c));
    }
    std::sort(stoi_to_token.begin(), stoi_to_token.end());
    for (size_t i = 0; i < stoi_to_token.size(); ++i) {
        token_to_stoi[stoi_to_token[i]] = i;
    }

    // Tokenized words
    std::unordered_map<std::string, std::vector<std::string>> word_tokens;
    for (auto& [w, f] : word_freq) {
        auto toks = split_word(w);
        word_tokens[w] = std::move(toks);
    }

    std::cout << "[BPE] Starting training, target vocab size = "
              << vocab_limit_ << " ..." << std::endl;
    auto start_time = std::chrono::steady_clock::now();

    // 3. Pair frequency table
    std::unordered_map<std::pair<std::string,std::string>, int, PairHash> pair_freq;
    auto update_pair_freqs = [&](const std::vector<std::string>& tokens, int count, int sign) {
        for (size_t i = 0; i + 1 < tokens.size(); ++i) {
            auto pair = std::make_pair(tokens[i], tokens[i+1]);
            pair_freq[pair] += sign * count;
            if (pair_freq[pair] <= 0) pair_freq.erase(pair);
        }
    };

    for (auto& [w, toks] : word_tokens) {
        update_pair_freqs(toks, word_freq[w], +1);
    }

    // Priority queue (max-heap of (freq, pair))
    auto cmp = [](const auto& a, const auto& b){ return a.first < b.first; };
    std::priority_queue<std::pair<int,std::pair<std::string,std::string>>,
                        std::vector<std::pair<int,std::pair<std::string,std::string>>>,
                        decltype(cmp)> pq(cmp);

    for (auto& [p, f] : pair_freq) {
        pq.push({f, p});
    }

    int merges = 0;

    // 4. Merge loop
    while (vocab_size() < vocab_limit_ && !pq.empty()) {
        auto [freq, best_pair] = pq.top(); pq.pop();
        if (!pair_freq.count(best_pair) || pair_freq[best_pair] != freq) continue; // outdated

        if (freq <= 1) break; // stop early if merges aren’t useful

        std::string merged = best_pair.first + best_pair.second;
        if (token_to_stoi.count(merged)) continue;

        // Add new token
        token_to_stoi[merged] = vocab_size();
        stoi_to_token.push_back(merged);
        merge_rank[best_pair] = merges; // save merge order
        merges++;

        // Deduplication set for this iteration
        std::unordered_set<std::pair<std::string,std::string>, PairHash> pushed_pairs;

        // Update all words containing this pair
        for (auto& [w, tokens] : word_tokens) {
            bool changed = false;

            update_pair_freqs(tokens, word_freq[w], -1);

            std::vector<std::string> new_tokens;
            for (size_t i = 0; i < tokens.size();) {
                if (i + 1 < tokens.size() &&
                    tokens[i] == best_pair.first &&
                    tokens[i+1] == best_pair.second) {
                    new_tokens.push_back(merged);
                    i += 2;
                    changed = true;
                } else {
                    new_tokens.push_back(tokens[i]);
                    i += 1;
                }
            }
            if (changed) {
                tokens.swap(new_tokens);
            }

            update_pair_freqs(tokens, word_freq[w], +1);

            // Push only new pairs around merged token, avoid duplicates
            for (size_t i = 0; i + 1 < tokens.size(); ++i) {
                if (tokens[i] == merged || tokens[i+1] == merged) {
                    auto p = std::make_pair(tokens[i], tokens[i+1]);
                    if (pair_freq.count(p) && !pushed_pairs.count(p)) {
                        pq.push({pair_freq[p], p});
                        pushed_pairs.insert(p);
                    }
                }
            }
        }

        if (merges % 100 == 0 || vocab_size() == vocab_limit_) {
            auto now = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(now - start_time).count();
            std::cout << "[BPE] Merges: " << merges
                      << " | Vocab size: " << vocab_size()
                      << " | Elapsed: " << elapsed << "s" << std::endl;
        }
    }

    std::cout << "[BPE] Training finished. Final vocab size: "
              << vocab_size() << std::endl;
}


std::vector<std::string> Tokenizer::encode_word_as_tokens(const std::string& word) {
    if (token_cache.count(word)) return token_cache[word];

    std::vector<std::string> tokens = split_word(word);

    while (tokens.size() > 1) {
        std::pair<std::string,std::string> best_pair;
        int best_rank = std::numeric_limits<int>::max();

        for (size_t i = 0; i + 1 < tokens.size(); ++i) {
            auto candidate = std::make_pair(tokens[i], tokens[i+1]);
            if (merge_rank.count(candidate) && merge_rank[candidate] < best_rank) {
                best_rank = merge_rank[candidate];
                best_pair = candidate;
            }
        }

        if (best_rank == std::numeric_limits<int>::max()) break;

        std::vector<std::string> new_tokens;
        for (size_t i = 0; i < tokens.size();) {
            if (i + 1 < tokens.size() &&
                tokens[i] == best_pair.first &&
                tokens[i+1] == best_pair.second) {
                new_tokens.push_back(best_pair.first + best_pair.second);
                i += 2;
            } else {
                new_tokens.push_back(tokens[i]);
                i += 1;
            }
        }
        tokens.swap(new_tokens);
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
            }
        }
    }
    return result;
}

std::string Tokenizer::decode(const std::vector<int>& ids) {
    std::string out;
    for (int id : ids) {
        if (id < 0 || id >= vocab_size()) continue;
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

    // Save merge ranks as array of [first, second, rank]
    json merges_json = json::array();
    for (auto& kv : merge_rank) {
        merges_json.push_back({kv.first.first, kv.first.second, kv.second});
    }
    j["merge_rank"] = merges_json;

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

    merge_rank.clear();
    if (j.contains("merge_rank")) {
        for (auto& mr : j["merge_rank"]) {
            std::pair<std::string,std::string> p = {mr[0].get<std::string>(), mr[1].get<std::string>()};
            int rank = mr[2].get<int>();
            merge_rank[p] = rank;
        }
    }

    token_cache.clear(); // Clear cache after loading
}
