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
              << " [--max_seq_len N=32]"
              << " [--ckpt PATH=gpt_checkpoint.bin]"
              << " [--tokenizer PATH=tokenizer.json]"
              << " [--top_k N=5]"
              << " [--temp F=1.0]"
              << " [--eos_id ID=-1]"
              << " [--stream true|false]"
              << " [--delay_ms N=0]"
              << std::endl;
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

int main(int argc, char** argv){
    auto args = parse_args(argc, argv);
    if(!args.count("--prompt")){ usage(argv[0]); return 1; }

    std::string prompt       = args["--prompt"];
    std::string ckpt_path    = args.count("--ckpt")      ? args["--ckpt"]      : "gpt_checkpoint.bin";
    std::string tok_path     = args.count("--tokenizer") ? args["--tokenizer"] : "tokenizer.json";
    int         num_tokens   = args.count("--num_tokens")? std::atoi(args["--num_tokens"].c_str()) : 50;
    int         max_seq_len  = args.count("--max_seq_len")?std::atoi(args["--max_seq_len"].c_str()): 32;
    int         top_k        = args.count("--top_k")     ? std::atoi(args["--top_k"].c_str())       : 5;
    float       temperature  = args.count("--temp")      ? std::stof(args["--temp"].c_str())        : 1.0f;
    int         eos_id       = args.count("--eos_id")    ? std::atoi(args["--eos_id"].c_str())      : -1;
    bool        stream       = args.count("--stream")    ? (args["--stream"]=="true")               : true;
    int         delay_ms     = args.count("--delay_ms")  ? std::atoi(args["--delay_ms"].c_str())    : 0;

    if(num_tokens <= 0) num_tokens = 1;
    if(max_seq_len <= 1) max_seq_len = 2;
    if(top_k < 1) top_k = 1;
    if(temperature <= 0.f) temperature = 1.f;

    // Load tokenizer (host side)
    Tokenizer tokenizer;
    tokenizer.load(tok_path);
    int vocab_size = tokenizer.vocab_size();

    std::cout << "Prompt: " << prompt << "\nGenerated: " << std::flush;

    {
        // Keep HIP-owning objects in a scope so they destruct before hipDeviceReset()
        // Use the same architecture as training (adjust if your checkpoint carries shape info)
        int embed_dim = 128, num_heads = 4, ff_hidden_dim = 256, num_layers = 2;

        GPTModel model(vocab_size, max_seq_len, embed_dim, num_heads, ff_hidden_dim, num_layers);
        model.load_checkpoint(ckpt_path);

        // Encode and trim to leave room for at least one new token
        std::vector<int> input_ids = tokenizer.encode(prompt);
        if(input_ids.empty()){
            std::cerr << "\nError: prompt produced no tokens.\n";
            hipDeviceSynchronize();
            // model dtor runs here at scope end
            goto after_model;
        }
        if((int)input_ids.size() >= max_seq_len){
            // keep only the last max_seq_len-1 tokens so generation has room
            input_ids.erase(
                input_ids.begin(),
                input_ids.begin() + (int)input_ids.size() - (max_seq_len - 1)
            );
        }

        // Run your existing generate() which must respect num_tokens
        std::vector<int> generated_ids = model.generate(input_ids, num_tokens, top_k, temperature);

        // Stream / print only the newly generated tail
        size_t start = input_ids.size();
        for(size_t i = start; i < generated_ids.size(); ++i){
            if(eos_id >= 0 && generated_ids[i] == eos_id) break;
            std::string s = tokenizer.decode({generated_ids[i]});
            std::cout << s << std::flush;
            if(stream && delay_ms > 0) std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
        }
        std::cout << std::endl;

        // Ensure all GPU work is finished before destructors
        hipDeviceSynchronize();
    }

after_model:
    // Now it's safe to reset the device (all HIP resources have been destroyed)
    hipDeviceReset();
    return 0;
}
