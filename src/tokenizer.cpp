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
#include <unordered_set>
#include <queue>
#include <chrono>

using json = nlohmann::json;

Tokenizer::Tokenizer(int vocab_limit) : vocab_limit_(vocab_limit) {}

// ----------------------- util ------------------------
std::vector<std::string> Tokenizer::split_word(const std::string& word) {
    std::vector<std::string> tokens;
    for (char c : word) tokens.push_back(std::string(1, c));
    if (!tokens.empty()) tokens.back() += "</w>";
    return tokens;
}

struct PairHash {
    size_t operator()(const std::pair<int,int>& p) const noexcept {
        return ((uint64_t)p.first << 32) ^ (uint64_t)p.second;
    }
};

// ----------------------- train ------------------------
void Tokenizer::train_bpe(const std::string& text) {
    std::istringstream iss(text);
    std::string word;

    // 1. Count word frequencies
    std::unordered_map<std::string,int> word_freq;
    while (iss >> word) word_freq[word]++;

    // 2. Initialize vocab with characters only
    std::set<char> char_vocab;
    for (auto& [w,f] : word_freq) {
        for (char c : w) char_vocab.insert(c);
    }
    for (char c : char_vocab) stoi_to_token.push_back(std::string(1,c));
    std::sort(stoi_to_token.begin(), stoi_to_token.end());
    for (size_t i=0;i<stoi_to_token.size();++i) token_to_stoi[stoi_to_token[i]]=i;

    // 3. Tokenize words (int ids)
    struct WordInfo {
        std::vector<int> tokens;
        int freq;
    };
    std::vector<WordInfo> words;
    words.reserve(word_freq.size());
    for (auto& [w,f] : word_freq) {
        auto toks = split_word(w);
        std::vector<int> ids;
        ids.reserve(toks.size());
        for (auto& t : toks) {
            if (!token_to_stoi.count(t)) {
                token_to_stoi[t] = stoi_to_token.size();
                stoi_to_token.push_back(t);
            }
            ids.push_back(token_to_stoi[t]);
        }
        words.push_back({std::move(ids), f});
    }

    std::cout << "[BPE] Starting training, target vocab size="<<vocab_limit_<<"...\n";
    auto start_time = std::chrono::steady_clock::now();

    // 4. Build initial pair frequencies and inverted index
    using Pair = std::pair<int,int>;
    std::unordered_map<Pair,int,PairHash> pair_freq;
    std::unordered_map<Pair,std::vector<int>,PairHash> pair_to_words;

    auto add_pairs = [&](const std::vector<int>& toks,int freq,int wid){
        for(size_t i=0;i+1<toks.size();++i){
            Pair p={toks[i],toks[i+1]};
            pair_freq[p]+=freq;
            pair_to_words[p].push_back(wid);
        }
    };
    for(int wid=0;wid<(int)words.size();++wid) add_pairs(words[wid].tokens,words[wid].freq,wid);

    // heap of (freq,pair)
    auto cmp=[](auto&a,auto&b){return a.first<b.first;};
    std::priority_queue<std::pair<int,Pair>,std::vector<std::pair<int,Pair>>,decltype(cmp)> pq(cmp);
    for(auto& [p,f]:pair_freq) pq.push({f,p});

    int merges=0;
    while(vocab_size()<vocab_limit_ && !pq.empty()){
        auto [freq,best]=pq.top(); pq.pop();
        if(!pair_freq.count(best) || pair_freq[best]!=freq) continue;
        if(freq<=1) break; // cutoff

        // create merged token
        std::string merged=stoi_to_token[best.first]+stoi_to_token[best.second];
        if(token_to_stoi.count(merged)) continue;
        int merged_id=vocab_size();
        token_to_stoi[merged]=merged_id;
        stoi_to_token.push_back(merged);
        merge_rank[best]=merges++;
        
        // words containing this pair
        auto affected=pair_to_words[best];
        pair_to_words.erase(best);
        pair_freq.erase(best);

        std::unordered_set<Pair,PairHash> updated_pairs;
        for(int wid:affected){
            auto& info=words[wid];
            auto& toks=info.tokens;
            if(toks.size()<2) continue;

            // try merging occurrences
            std::vector<int> new_toks;
            new_toks.reserve(toks.size());
            bool changed=false;
            for(size_t i=0;i<toks.size();){
                if(i+1<toks.size() && toks[i]==best.first && toks[i+1]==best.second){
                    new_toks.push_back(merged_id);
                    i+=2; changed=true;
                } else {
                    new_toks.push_back(toks[i]); i++;
                }
            }
            if(!changed) continue;

            // remove old pairs
            for(size_t i=0;i+1<toks.size();++i){
                Pair p={toks[i],toks[i+1]};
                pair_freq[p]-=info.freq;
                if(pair_freq[p]<=0) pair_freq.erase(p);
            }
            toks.swap(new_toks);
            // add new pairs
            for(size_t i=0;i+1<toks.size();++i){
                Pair p={toks[i],toks[i+1]};
                pair_freq[p]+=info.freq;
                if(!updated_pairs.count(p)){
                    pq.push({pair_freq[p],p});
                    updated_pairs.insert(p);
                }
                pair_to_words[p].push_back(wid);
            }
        }

        if(merges%100==0 || vocab_size()==vocab_limit_){
            auto now=std::chrono::steady_clock::now();
            double elapsed=std::chrono::duration<double>(now-start_time).count();
            std::cout<<"[BPE] Merges:"<<merges<<" | Vocab:"<<vocab_size()<<" | Elapsed:"<<elapsed<<"s\n";
        }
    }

    std::cout<<"[BPE] Training finished. Final vocab size="<<vocab_size()<<"\n";
}

// ------------------- encoding ---------------------
std::vector<std::string> Tokenizer::encode_word_as_tokens(const std::string& word){
    if(token_cache.count(word)) return token_cache[word];
    std::vector<std::string> toks=split_word(word);
    // greedy merge by merge_rank
    while(toks.size()>1){
        std::pair<std::string,std::string> best; int best_rank=INT_MAX;
        for(size_t i=0;i+1<toks.size();++i){
            auto cand=std::make_pair(toks[i],toks[i+1]);
            if(merge_rank.count(cand)&&merge_rank[cand]<best_rank){
                best_rank=merge_rank[cand]; best=cand;
            }
        }
        if(best_rank==INT_MAX) break;
        std::vector<std::string> newt;
        for(size_t i=0;i<toks.size();){
            if(i+1<toks.size() && toks[i]==best.first && toks[i+1]==best.second){
                newt.push_back(best.first+best.second); i+=2;
            } else { newt.push_back(toks[i]); i++; }
        }
        toks.swap(newt);
    }
    return token_cache[word]=toks;
}

std::vector<int> Tokenizer::encode(const std::string& text){
    std::istringstream iss(text);
    std::string w; std::vector<int> out;
    while(iss>>w){
        auto toks=encode_word_as_tokens(w);
        for(auto& t:toks) if(token_to_stoi.count(t)) out.push_back(token_to_stoi[t]);
    }
    return out;
}

std::string Tokenizer::decode(const std::vector<int>& ids){
    std::string out;
    for(int id:ids){
        if(id<0||id>=vocab_size()) continue;
        auto tok=stoi_to_token[id];
        if(tok.size()>=4 && tok.substr(tok.size()-4)=="</w>")
            out+=tok.substr(0,tok.size()-4)+" ";
        else out+=tok;
    }
    return out;
}

// ------------------- save/load ---------------------
void Tokenizer::save(const std::string& path) const{
    json j;
    j["stoi_to_token"]=stoi_to_token;
    json merges=json::array();
    for(auto& kv:merge_rank)
        merges.push_back({kv.first.first,kv.first.second,kv.second});
    j["merge_rank"]=merges;
    std::ofstream out(path); out<<j.dump(2);
}
void Tokenizer::load(const std::string& path){
    std::ifstream in(path);
    if(!in){std::cerr<<"Cannot open "<<path<<"\n";return;}
    json j; in>>j;
    stoi_to_token=j["stoi_to_token"].get<std::vector<std::string>>();
    token_to_stoi.clear();
    for(size_t i=0;i<stoi_to_token.size();++i) token_to_stoi[stoi_to_token[i]]=i;
    merge_rank.clear();
    if(j.contains("merge_rank")){
        for(auto& mr:j["merge_rank"]){
            merge_rank[{mr[0],mr[1]}]=mr[2];
        }
    }
    token_cache.clear();
}
