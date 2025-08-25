#include "gpt_model.h"
#include "tokenizer.h"
#include <hip/hip_runtime.h>
#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <cstdlib>
#include <chrono>
#include <thread>
#include <unordered_map>

static void usage(const char* prog) {
    std::cout
        << "Usage: " << prog
        << " --prompt \"<text>\""
        << " [--num_tokens N=50]"
        << " [--max_seq_len N=32]"
        << " [--ckpt PATH=gpt_checkpoint.bin]"
        << " [--tokenizer PATH=tokenizer.json]"
        << " [--top_k N=5]"
        << " [--temp F=1.0]"
        << " [--eos_id ID=-1]        (set >=0 to stop on EOS token)\n"
        << " [--stream true|false]    (default true)\n"
        << " [--delay_ms N=0]         (stream delay per token)\n";
}

static std::unordered_map<std::string,std::string> parse_args(int argc, char** argv){
    std::unordered_map<std::string,std::string> a;
    for(int i=1;i<argc;++i){
        std::string k=argv[i];
        if(k.rfind("--",0)==0){
            if(i+1<argc && std::string(argv[i+1]).rfind("--",0)!=0){
                a[k]=argv[++i];
            }else{
                a[k]="true";
            }
        }else{
            // ignore stray token; keep simple
        }
    }
    return a;
}

int main(int argc, char** argv){
    auto args = parse_args(argc, argv);

    // required
    if(!args.count("--prompt")){
        usage(argv[0]);
        return 1;
    }
    std::string prompt = args["--prompt"];

    // optional
    std::string ckpt_path     = args.count("--ckpt")      ? args["--ckpt"]      : "gpt_checkpoint.bin";
    std::string tok_path      = args.count("--tokenizer") ? args["--tokenizer"] : "tokenizer.json";
    int         num_tokens    = args.count("--num_tokens")? std::atoi(args["--num_tokens"].c_str()): 50;
    int         max_seq_len   = args.count("--max_seq_len")?std::atoi(args["--max_seq_len"].c_str()): 32;
    int         top_k         = args.count("--top_k")     ? std::atoi(args["--top_k"].c_str())       : 5;
    float       temperature   = args.count("--temp")      ? std::stof(args["--temp"].c_str())        : 1.0f;
    int         eos_id        = args.count("--eos_id")    ? std::atoi(args["--eos_id"].c_str())      : -1;
    bool        stream        = args.count("--stream")    ? (args["--stream"]=="true")               : true;
    int         delay_ms      = args.count("--delay_ms")  ? std::atoi(args["--delay_ms"].c_str())    : 0;

    if(num_tokens <= 0) num_tokens = 1;
    if(max_seq_len <= 0) max_seq_len = 32;
    if(top_k < 1) top_k = 1;
    if(temperature <= 0.f) temperature = 1.f;

    // Load tokenizer
    Tokenizer tokenizer;
    tokenizer.load(tok_path);
    int vocab_size = tokenizer.vocab_size();

    std::cout << "Prompt: " << prompt << "\n";
    std::cout << "Generated: " << std::flush;

    // Wrap HIP-owning stuff in a scope so it destructs before hipDeviceReset()
    {
        // If your GPTModel requires dims, make sure they match training or are stored in the checkpoint.
        // If the checkpoint stores shapes, GPTModel::load_checkpoint should set them.
        // Otherwise, set the same hyperparams as training:
        int embed_dim = 128, num_heads = 4, ff_hidden_dim = 256, num_layers = 2;

        GPTModel model(vocab_size, max_seq_len, embed_dim, num_heads, ff_hidden_dim, num_layers);
        model.load_checkpoint(ckpt_path);

        // Encode input
        std::vector<int> ctx = tokenizer.encode(prompt);
        if(ctx.empty()){
            std::cerr << "\nError: prompt produced no tokens.\n";
            // sync just in case model used GPU already
            hipDeviceSynchronize();
            // model dtor runs here (end of scope)
            goto end_reset;
        }

        // Hard cap generation by num_tokens and/or EOS
        // Keep context trimmed to last max_seq_len tokens
        std::vector<int> generated = ctx;
        generated.reserve(ctx.size() + num_tokens);

        for(int step = 0; step < num_tokens; ++step){
            // Build the current window (last max_seq_len tokens)
            int start = (int)generated.size() > max_seq_len ? (int)generated.size() - max_seq_len : 0;
            std::vector<int> window(generated.begin() + start, generated.end());

            // Model must provide a single-step predict/sampling method OR a batched forward.
            // Assuming you already had model.generate(...) that returned a full sequence,
            // we instead call a single-token sampler. If you don’t have it, implement it
            // by calling forward() on 'window' and sampling top_k at the last position.
            int next_id = model.sample_next(window, top_k, temperature); // <-- implement this in your model

            generated.push_back(next_id);

            if(stream){
                std::string s = tokenizer.decode({next_id});
                std::cout << s << std::flush;
                if(delay_ms > 0) std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
            }

            if(eos_id >= 0 && next_id == eos_id){
                break;
            }
        }

        if(!stream){
            // If not streaming, print the tail we generated
            std::string out = tokenizer.decode(
                std::vector<int>(generated.begin() + (int)ctx.size(), generated.end())
            );
            std::cout << out;
        }

        std::cout << std::endl;

        // Make sure any GPU work is done before leaving scope
        hipDeviceSynchronize();
    }

end_reset:
    // Now HIP resources owned by the model are gone; reset is safe
    hipDeviceReset();
    return 0;
}
