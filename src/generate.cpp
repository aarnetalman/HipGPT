#include "gpt_model.h"
#include "tokenizer.h"
#include <hip/hip_runtime.h>
#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <cstdlib>
#include <unordered_map>
#include <chrono>
#include <thread>

static void usage(const char* prog){
    std::cout << "Usage: " << prog
              << " --prompt \"<text>\""
              << " [--num_tokens N=50]"
              << " [--max_seq_len N=32]        # generation window on host"
              << " [--model_seq_len N]          # model context length (must match ckpt)"
              << " [--ckpt PATH=gpt_checkpoint.bin]"
              << " [--tokenizer PATH=tokenizer.json]"
              << " [--top_k N=5]"
              << " [--temp F=1.0]"
              << " [--eos_id ID=-1]"
              << " [--stream true|false]"
              << " [--delay_ms N=0]"
              << " [--vocab N]                  # override tokenizer vocab if needed"
              << " [--dim N] [--heads N] [--ff N] [--layers N]   # MUST match ckpt\n";
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
    if(!args.count("--prompt")){ usage(argv[0]); return 1; }

    // I/O & sampling args
    std::string prompt       = to_s(args,"--prompt","");
    std::string ckpt_path    = to_s(args,"--ckpt","gpt_checkpoint.bin");
    std::string tok_path     = to_s(args,"--tokenizer","tokenizer.json");
    int   num_tokens         = to_i(args,"--num_tokens",50);
    int   host_max_seq_len   = to_i(args,"--max_seq_len",32);   // trimming window for generation
    int   top_k              = to_i(args,"--top_k",5);
    float temperature        = to_f(args,"--temp",1.0f);
    int   eos_id             = to_i(args,"--eos_id",-1);
    bool  stream             = (to_s(args,"--stream","true")=="true");
    int   delay_ms           = to_i(args,"--delay_ms",0);

    if(num_tokens <= 0) num_tokens = 1;
    if(host_max_seq_len <= 1) host_max_seq_len = 2;
    if(top_k < 1) top_k = 1;
    if(temperature <= 0.f) temperature = 1.f;

    // Model shape — MUST match the checkpoint
    int embed_dim       = to_i(args,"--dim",256);
    int num_heads       = to_i(args,"--heads",4);
    int ff_hidden_dim   = to_i(args,"--ff",1024);
    int num_layers      = to_i(args,"--layers",4);
    int model_max_seq   = to_i(args,"--model_seq_len",256); // model’s context length
    // Tokenizer / vocab
    Tokenizer tokenizer;
    tokenizer.load(tok_path);
    int vocab_size_tok  = tokenizer.vocab_size();
    int vocab_size      = args.count("--vocab") ? to_i(args,"--vocab",vocab_size_tok) : vocab_size_tok;

    // Print the resolved config so mismatches are obvious
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
        // Construct model with the **same** hyperparameters used for training
        GPTModel model(vocab_size, model_max_seq, embed_dim, num_heads, ff_hidden_dim, num_layers);
        model.load_checkpoint(ckpt_path); // will throw if shapes don’t match

        // Encode and trim to leave room for at least one new token (on host side)
        std::vector<int> input_ids = tokenizer.encode(prompt);
        if(input_ids.empty()){
            std::cerr << "\nError: prompt produced no tokens.\n";
            hipDeviceSynchronize();
            goto after_model;
        }

        // Trim to the smaller of host_max_seq_len and model_max_seq so we never exceed model context
        int gen_window = std::min(host_max_seq_len, model_max_seq);
        if((int)input_ids.size() >= gen_window){
            input_ids.erase(
                input_ids.begin(),
                input_ids.begin() + (int)input_ids.size() - (gen_window - 1)
            );
        }

        // Generate
        std::vector<int> generated_ids = model.generate(input_ids, num_tokens, top_k, temperature);

        // Print only the newly generated tail
        size_t start = input_ids.size();
        for(size_t i = start; i < generated_ids.size(); ++i){
            if(eos_id >= 0 && generated_ids[i] == eos_id) break;
            std::string s = tokenizer.decode({generated_ids[i]});
            std::cout << s << std::flush;
            if(stream && delay_ms > 0) std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
        }
        std::cout << std::endl;

        hipDeviceSynchronize();
    }

after_model:
    hipDeviceReset();
    return 0;
}