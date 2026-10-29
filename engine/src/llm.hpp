// The language model: llama.cpp, greedy, streaming. The orchestrator writes the whole prompt (chat
// template, tools, history); this side only computes it and generates.
//
// Two things make it fast enough to start talking within a second:
//   - the context keeps what it computed, so a prompt that starts like the last one (the system prompt
//     with its tools, the conversation so far) is only computed from where the two differ;
//   - text streams out as it is generated, so the first sentence can be spoken before the rest exists.
// Tool calls are free text until the trigger ("<tool_call>"); from there a JSON-schema grammar allows
// only a call to one of the tools with valid arguments, and generation stops when that JSON closes.
#pragma once

#include <functional>
#include <map>
#include <string>
#include <vector>

struct llama_model;
struct llama_context;

namespace courier {

struct Generation {
    std::string text;
    std::string stop;  // "end", "tool", "length", "cancel"
    int prompt_tokens = 0, reused_tokens = 0, generated = 0;
    double ms_prompt = 0, ms_first = -1, ms = 0;
};

class Llm {
public:
    Llm(const std::string& path, int context, int gpu_layers);
    ~Llm();
    Llm(const Llm&) = delete;
    Llm& operator=(const Llm&) = delete;

    // `on_piece` returns false to stop.
    Generation generate(const std::string& prompt, const std::string& trigger, const std::string& tool_schema, int max_tokens,
                        const std::function<bool(const std::string&)>& on_piece);
    int count_tokens(const std::string& text) const;

private:
    std::vector<int> tokenize(const std::string& text) const;
    const std::string& grammar_for(const std::string& schema);
    llama_model* model_ = nullptr;
    llama_context* ctx_ = nullptr;
    std::vector<int> cached_;  // the tokens the context holds
    std::map<std::string, std::string> grammars_;
};

// How many leading bytes of `s` are whole UTF-8 characters.
size_t utf8_complete(const std::string& s);

}  // namespace courier
