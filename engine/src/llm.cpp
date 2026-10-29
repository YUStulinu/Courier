#include "llm.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <thread>

#include "json-schema-to-grammar.h"
#include "llama.h"

namespace courier {

namespace {
using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }
}  // namespace

size_t utf8_complete(const std::string& s) {
    // Walk back over at most three continuation bytes to the last lead byte; keep it only if its
    // character is complete.
    size_t n = s.size();
    size_t i = n;
    int back = 0;
    while (i > 0 && back < 4) {
        unsigned char c = static_cast<unsigned char>(s[i - 1]);
        if ((c & 0xC0) != 0x80) {
            size_t len = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 1;
            return (n - (i - 1) >= len) ? n : i - 1;
        }
        --i;
        ++back;
    }
    return n;
}

Llm::Llm(const std::string& path, int context, int gpu_layers) {
    if (!std::getenv("COURIER_VERBOSE")) llama_log_set([](ggml_log_level, const char*, void*) {}, nullptr);
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = gpu_layers;
    model_ = llama_model_load_from_file(path.c_str(), mp);
    if (!model_) throw std::runtime_error("cannot load the language model " + path);
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = static_cast<uint32_t>(context);
    cp.n_batch = 512;
    int threads = std::max(2, static_cast<int>(std::thread::hardware_concurrency()) / 2);
    cp.n_threads = threads;
    cp.n_threads_batch = threads;
    ctx_ = llama_init_from_model(model_, cp);
    if (!ctx_) {
        llama_model_free(model_);
        throw std::runtime_error("cannot create a context for the language model");
    }
}

Llm::~Llm() {
    if (ctx_) llama_free(ctx_);
    if (model_) llama_model_free(model_);
}

std::vector<int> Llm::tokenize(const std::string& text) const {
    const llama_vocab* vocab = llama_model_get_vocab(model_);
    std::vector<llama_token> t(text.size() + 16);
    int n = llama_tokenize(vocab, text.c_str(), static_cast<int32_t>(text.size()), t.data(), static_cast<int32_t>(t.size()), true, true);
    if (n < 0) {
        t.resize(static_cast<size_t>(-n));
        n = llama_tokenize(vocab, text.c_str(), static_cast<int32_t>(text.size()), t.data(), static_cast<int32_t>(t.size()), true, true);
    }
    if (n < 0) throw std::runtime_error("tokenizing the prompt failed");
    t.resize(static_cast<size_t>(n));
    return t;
}

int Llm::count_tokens(const std::string& text) const { return static_cast<int>(tokenize(text).size()); }

const std::string& Llm::grammar_for(const std::string& schema) {
    auto it = grammars_.find(schema);
    if (it != grammars_.end()) return it->second;
    // The schema's grammar, its root renamed, between the newlines the model writes around a call (it
    // likes to end one with the single token "}}\n").
    std::string g = json_schema_to_grammar(common_json::parse(schema));
    size_t root = g.find("root ::=");
    if (root == std::string::npos) throw std::runtime_error("the tool schema made no grammar");
    g.replace(root, 8, "call ::=");
    g += "\nroot ::= \"\\n\"? call \"\\n\"?\n";
    return grammars_.emplace(schema, std::move(g)).first->second;
}

Generation Llm::generate(const std::string& prompt, const std::string& trigger, const std::string& tool_schema, int max_tokens,
                         const std::function<bool(const std::string&)>& on_piece) {
    auto t0 = Clock::now();
    Generation g;
    const llama_vocab* vocab = llama_model_get_vocab(model_);
    std::vector<int> tokens = tokenize(prompt);
    const int n = static_cast<int>(tokens.size());
    g.prompt_tokens = n;
    if (n + max_tokens > static_cast<int>(llama_n_ctx(ctx_))) throw std::runtime_error("the conversation is too long for the context");

    // Reuse what the context holds of this prompt; keep at least the last token to compute, for its logits.
    size_t common = 0;
    while (common < cached_.size() && common < tokens.size() && cached_[common] == tokens[common]) ++common;
    if (common == tokens.size()) --common;
    llama_memory_t mem = llama_get_memory(ctx_);
    if (!llama_memory_seq_rm(mem, 0, static_cast<llama_pos>(common), -1)) {
        llama_memory_clear(mem, true);
        common = 0;
    }
    cached_.assign(tokens.begin(), tokens.begin() + static_cast<long>(common));
    g.reused_tokens = static_cast<int>(common);
    for (int i = static_cast<int>(common); i < n; i += 512) {
        int k = std::min(512, n - i);
        if (llama_decode(ctx_, llama_batch_get_one(tokens.data() + i, k)) != 0) {
            cached_.clear();
            llama_memory_clear(mem, true);
            throw std::runtime_error("computing the prompt failed");
        }
        cached_.insert(cached_.end(), tokens.begin() + i, tokens.begin() + i + k);
    }
    g.ms_prompt = ms_since(t0);

    // Greedy. Under the grammar, the best token is checked alone first; the whole vocabulary (150k
    // tokens) is filtered only when the grammar rejects it.
    std::unique_ptr<llama_sampler, void (*)(llama_sampler*)> grammar(nullptr, llama_sampler_free);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    std::vector<llama_token_data> cand;
    auto next_token = [&]() -> llama_token {
        const float* logits = llama_get_logits_ith(ctx_, -1);
        llama_token best = 0;
        for (llama_token v = 1; v < n_vocab; ++v)
            if (logits[v] > logits[best]) best = v;
        if (!grammar) return best;
        llama_token_data one = {best, logits[best], 0.f};
        llama_token_data_array single = {&one, 1, -1, false};
        llama_sampler_apply(grammar.get(), &single);
        if (single.data[0].logit != -INFINITY) return best;
        if (std::getenv("COURIER_DEBUG_GRAMMAR")) {
            char piece[256];
            int m = llama_token_to_piece(vocab, best, piece, sizeof piece, 0, true);
            std::fprintf(stderr, "[grammar rejects %d '%.*s']\n", best, m > 0 ? m : 0, piece);
        }
        cand.resize(static_cast<size_t>(n_vocab));
        for (llama_token v = 0; v < n_vocab; ++v) cand[v] = {v, logits[v], 0.f};
        llama_token_data_array all = {cand.data(), cand.size(), -1, false};
        llama_sampler_apply(grammar.get(), &all);
        best = -1;
        for (size_t i = 0; i < all.size; ++i)
            if (all.data[i].logit != -INFINITY && (best < 0 || all.data[i].logit > cand[best].logit)) best = all.data[i].id;
        if (best < 0) throw std::runtime_error("the tool grammar allows no token");
        return best;
    };

    std::string pending;     // bytes not yet passed on (an unfinished UTF-8 character)
    bool in_call = false;    // after the trigger
    int depth = 0;           // JSON nesting inside the call
    bool in_string = false, escaped = false, opened = false;
    g.stop = "length";
    for (int i = 0; i < max_tokens; ++i) {
        llama_token t = next_token();
        if (grammar) llama_sampler_accept(grammar.get(), t);
        if (llama_vocab_is_eog(vocab, t)) {
            g.stop = "end";
            break;
        }
        char buf[256];
        int m = llama_token_to_piece(vocab, t, buf, sizeof buf, 0, true);
        std::string piece = m > 0 ? std::string(buf, static_cast<size_t>(m)) : std::string();
        ++g.generated;
        g.text += piece;
        bool closed = false;
        if (in_call) {
            for (char c : piece) {
                if (in_string) {
                    if (escaped) escaped = false;
                    else if (c == '\\') escaped = true;
                    else if (c == '"') in_string = false;
                } else if (c == '"') {
                    in_string = true;
                } else if (c == '{' || c == '[') {
                    ++depth;
                    opened = true;
                } else if (c == '}' || c == ']') {
                    if (--depth == 0 && opened) closed = true;
                }
            }
        }
        pending += piece;
        size_t whole = utf8_complete(pending);
        if (whole > 0) {
            if (g.ms_first < 0) g.ms_first = ms_since(t0);
            if (on_piece && !on_piece(pending.substr(0, whole))) {
                g.stop = "cancel";
                pending.erase(0, whole);
                break;
            }
            pending.erase(0, whole);
        }
        if (closed) {
            g.stop = "tool";
            break;
        }
        if (llama_decode(ctx_, llama_batch_get_one(&t, 1)) != 0) {
            g.stop = "error";
            break;
        }
        cached_.push_back(t);
        if (!in_call && !trigger.empty() && g.text.size() >= trigger.size() &&
            g.text.compare(g.text.size() - trigger.size(), trigger.size(), trigger) == 0) {
            in_call = true;
            if (!tool_schema.empty()) {
                grammar.reset(llama_sampler_init_grammar(vocab, grammar_for(tool_schema).c_str(), "root"));
                if (!grammar) throw std::runtime_error("the tool schema's grammar did not load");
            }
        }
    }
    if (!pending.empty() && on_piece) on_piece(pending);
    g.ms = ms_since(t0);
    return g;
}

}  // namespace courier
