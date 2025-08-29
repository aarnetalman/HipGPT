#include "gpt_model.h"
#include "tokenizer.h"
#include <hip/hip_runtime.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cstdlib>
#include <unordered_map>
#include <chrono>
#include <thread>
#include <nlohmann/json.hpp>
#include <filesystem>
namespace fs = std::filesystem;
using json = nlohmann::json;

static void usage(const char* prog){
    std::cout << "Usage: " << prog
              << " --prompt \"<text>\""
              << " --run-name NAME --step N"
              << " [--num_tokens N=50]"
              << " [--max_seq_len N=32]        # host-side gen window"
              << " [--top_k N=5] [--temp F=1.0] [--eos_id ID=-1]\n";
}

static std::unordered_map<std::string,std::string> parse_args(int argc, char** argv){
    std::unordered_map<std::string,std::string> a;
    for(int i=1;i<argc;++i){
        std::string k = argv[i];
        if(k.rfind("--",0)==0){
            if(i+1<argc && std::string(argv[i+1]).rfind("--",0)!=0){
                a[k]=argv[++i];
            }else{
                a[k]="true";
            }
        }
    }
    return a;
}

static int to_i(const std::unordered_map<std::string,std::string>& a,
                const char* key, int defv){
    auto it=a.find(key); return (it==a.end()?defv:std::atoi(it->second.c_str()));
}
static float to_f(const std::unordered_map<std::string,std::string>& a,
                  const char* key, float defv){
    auto it=a.find(key); return (it==a.end()?defv:std::stof(it->second.c_str()));
}
static std::string to_s(const std::unordered_map<std::string,std::string>& a,
                        const char* key, const char* defv){
    auto it=a.find(key); return (it==a.end()?std::string(defv):it->second);
}

int main(int argc, char** argv){
    auto args = parse_args(argc, argv);
    if(!args.count("--prompt") || !args.count("--run-name") || !args.count("--step")){
        usage(argv[0]); return 1;
    }

    // I/O & sampling args
    std::string prompt     = to_s(args,"--prompt","");
    int num_tokens         = to_i(args,"--num_tokens",50);
    int host_max_seq_len   = to_i(args,"--max_seq_len",32);
    int top_k              = to_i(args,"--top_k",5);
    float temperature      = to_f(args,"--temp",1.0f);
    int eos_id             = to_i(args,"--eos_id",-1);

    std::string run_name   = to_s(args,"--run-name","");
    int step               = to_i(args,"--step",-1);

    if(step < 0){
        std::cerr << "Error: you must provide --step N\n";
        return 1;
    }

    // --- Load config for that step ---
    std::string cfg_path = "checkpoints/" + run_name + "/" + run_name + "_step" + std::to_string(step) + "_config.json";
    if(!fs::exists(cfg_path)){
        std::cerr << "Error: config not found at " << cfg_path << "\n";
        return 1;
    }

    std::ifstream in(cfg_path);
    json config; in >> config;

    int vocab_size     = config["model"]["vocab_size"];
    int model_max_seq  = config["model"]["max_seq_len"];
    int embed_dim      = config["model"]["embed_dim"];
    int num_heads      = config["model"]["num_heads"];
    int ff_hidden_dim  = config["model"]["ff_hidden_dim"];
    int num_layers     = config["model"]["num_layers"];

    std::string tok_path    = config["tokenizer"]["path"];
    std::string tokens_path = config["tokenizer"]["tokens_path"];
    std::string ckpt_path   = config["checkpoint"]["latest"];

    std::cerr << "[generate] Loaded config " << cfg_path << "\n";

    // Tokenizer
    Tokenizer tokenizer;
    tokenizer.load(tok_path);
    int vocab_size_tok  = tokenizer.vocab_size();
    if(vocab_size==0) vocab_size = vocab_size_tok;

    // Print resolved config
    std::cerr << "[generate] ckpt=" << ckpt_path
              << " dim=" << embed_dim
              << " heads=" << num_heads
              << " ff=" << ff_hidden_dim
              << " layers=" << num_layers
              << " model_seq_len=" << model_max_seq
              << " vocab=" << vocab_size
              << " host_window=" << host_max_seq_len
              << " top_k=" << top_k
              << " temp=" << temperature
              << std::endl;

    std::cout << "Prompt: " << prompt << "\nGenerated: " << std::flush;

    {
        GPTModel model(vocab_size, model_max_seq, embed_dim, num_heads, ff_hidden_dim, num_layers);
        model.load_checkpoint(ckpt_path);

        std::vector<int> input_ids = tokenizer.encode(prompt);
        if(input_ids.empty()){
            std::cerr << "\nError: prompt produced no tokens.\n";
            hipDeviceSynchronize();
            goto after_model;
        }

        int gen_window = std::min(host_max_seq_len, model_max_seq);
        if((int)input_ids.size() >= gen_window){
            input_ids.erase(
                input_ids.begin(),
                input_ids.begin() + (int)input_ids.size() - (gen_window - 1)
            );
        }

        std::vector<int> generated_ids = model.generate(input_ids, num_tokens, top_k, temperature);

        size_t start = input_ids.size();
        for(size_t i = start; i < generated_ids.size(); ++i){
            if(eos_id >= 0 && generated_ids[i] == eos_id) break;
            std::string s = tokenizer.decode({generated_ids[i]});
            std::cout << s << std::flush;
        }
        std::cout << std::endl;

        hipDeviceSynchronize();
    }

after_model:
    hipDeviceReset();
    return 0;
}
