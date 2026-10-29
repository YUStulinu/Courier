// The C API (include/courier.h): exceptions become NULL or -1 and a per-thread message.
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>

#include "asr.hpp"
#include "courier.h"
#include "llm.hpp"
#include "tts.hpp"
#include "vad.hpp"
#include "wake.hpp"

using namespace courier;

namespace {
thread_local std::string last_error;

template <class F>
auto guard(F&& f, decltype(f()) failed) -> decltype(f()) {
    try {
        return f();
    } catch (const std::exception& e) {
        last_error = e.what();
        return failed;
    }
}

char* dup(const std::string& s) {
    char* p = static_cast<char*>(std::malloc(s.size() + 1));
    if (!p) return nullptr;
    std::memcpy(p, s.c_str(), s.size() + 1);
    return p;
}
}  // namespace

struct courier_vad {
    std::unique_ptr<Vad> vad;
};
struct courier_features {
    std::unique_ptr<SpeechFeatures> f;
};
struct courier_wake {
    std::unique_ptr<WakeWord> wake;
};
struct courier_asr {
    std::unique_ptr<Asr> asr;
};
struct courier_llm {
    std::unique_ptr<Llm> llm;
};
struct courier_tts {
    std::unique_ptr<Tts> tts;
};

extern "C" {

const char* courier_last_error(void) { return last_error.c_str(); }
void courier_free_string(char* s) { std::free(s); }
void courier_free_pcm(int16_t* pcm) { std::free(pcm); }
int courier_has_gpu(void) {
#ifdef COURIER_CUDA
    return 1;
#else
    return 0;
#endif
}

courier_vad* courier_vad_new(const char* model) {
    return guard([&]() -> courier_vad* {
        auto v = std::make_unique<courier_vad>();
        if (model) v->vad = std::make_unique<SileroVad>(model);
        else v->vad = std::make_unique<EnergyVad>();
        return v.release();
    }, nullptr);
}

int courier_vad_process(courier_vad* v, const float* samples, int n, float* probs, int max) {
    return guard([&]() -> int {
        auto p = v->vad->process(samples, n);
        int k = std::min(static_cast<int>(p.size()), max);
        std::memcpy(probs, p.data(), static_cast<size_t>(k) * sizeof(float));
        return k;
    }, -1);
}

void courier_vad_reset(courier_vad* v) { v->vad->reset(); }
void courier_vad_free(courier_vad* v) { delete v; }

courier_features* courier_features_new(const char* folder) {
    return guard([&]() -> courier_features* {
        auto f = std::make_unique<courier_features>();
        f->f = std::make_unique<SpeechFeatures>(folder);
        return f.release();
    }, nullptr);
}

int courier_features_process(courier_features* f, const float* samples, int n, float* out, int max) {
    return guard([&]() -> int {
        auto e = f->f->process(samples, n);
        int k = std::min(static_cast<int>(e.size() / SpeechFeatures::kDim), max);
        std::memcpy(out, e.data(), static_cast<size_t>(k) * SpeechFeatures::kDim * sizeof(float));
        return k;
    }, -1);
}

void courier_features_reset(courier_features* f) { f->f->reset(); }
void courier_features_free(courier_features* f) { delete f; }

courier_wake* courier_wake_new(const char* model, float threshold, int hits, int refractory_ms) {
    return guard([&]() -> courier_wake* {
        auto w = std::make_unique<courier_wake>();
        w->wake = std::make_unique<WakeWord>(model, threshold, hits, refractory_ms);
        return w.release();
    }, nullptr);
}

int courier_wake_process(courier_wake* w, const float* samples, int n, float* scores, int max, int* n_scores) {
    return guard([&]() -> int {
        auto r = w->wake->process(samples, n);
        int k = scores ? std::min(static_cast<int>(r.scores.size()), max) : 0;
        if (k > 0) std::memcpy(scores, r.scores.data(), static_cast<size_t>(k) * sizeof(float));
        if (n_scores) *n_scores = k;
        return r.detections;
    }, -1);
}

double courier_wake_last_ms(const courier_wake* w) { return w->wake->last_ms(); }
void courier_wake_set_threshold(courier_wake* w, float threshold) { w->wake->set_threshold(threshold); }
void courier_wake_reset(courier_wake* w) { w->wake->reset(); }
void courier_wake_free(courier_wake* w) { delete w; }


courier_asr* courier_asr_new(const char* model, int threads, int gpu) {
    return guard([&]() -> courier_asr* {
        auto a = std::make_unique<courier_asr>();
        a->asr = std::make_unique<Asr>(model, threads, gpu != 0);
        return a.release();
    }, nullptr);
}

char* courier_asr_transcribe(courier_asr* a, const float* samples, int n, int fast) {
    return guard([&]() -> char* {
        auto t = a->asr->transcribe(samples, n, fast != 0);
        nlohmann::json j = {{"text", t.text}, {"ms", t.ms}};
        return dup(j.dump());
    }, nullptr);
}

void courier_asr_free(courier_asr* a) { delete a; }

courier_llm* courier_llm_new(const char* model, int context, int gpu_layers) {
    return guard([&]() -> courier_llm* {
        auto l = std::make_unique<courier_llm>();
        l->llm = std::make_unique<Llm>(model, context, gpu_layers);
        return l.release();
    }, nullptr);
}

char* courier_llm_generate(courier_llm* l, const char* prompt, const char* trigger, const char* tool_schema, int max_tokens,
                           courier_piece_cb on_piece, void* user) {
    return guard([&]() -> char* {
        auto g = l->llm->generate(prompt, trigger ? trigger : "", tool_schema ? tool_schema : "", max_tokens,
                                  [&](const std::string& piece) { return !on_piece || on_piece(piece.c_str(), user) == 0; });
        nlohmann::json j = {{"text", g.text}, {"stop", g.stop}, {"prompt_tokens", g.prompt_tokens},
                            {"reused_tokens", g.reused_tokens}, {"generated", g.generated}, {"ms_prompt", g.ms_prompt},
                            {"ms_first", g.ms_first}, {"ms", g.ms}};
        return dup(j.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace));
    }, nullptr);
}

int courier_llm_count_tokens(courier_llm* l, const char* text) {
    return guard([&]() -> int { return l->llm->count_tokens(text); }, -1);
}

void courier_llm_free(courier_llm* l) { delete l; }

courier_tts* courier_tts_new(const char* voice, const char* config, const char* cmudict, int threads) {
    return guard([&]() -> courier_tts* {
        auto t = std::make_unique<courier_tts>();
        t->tts = std::make_unique<Tts>(voice, config, cmudict, threads);
        return t.release();
    }, nullptr);
}

int courier_tts_rate(const courier_tts* t) { return t->tts->sample_rate(); }
int courier_tts_speakers(const courier_tts* t) { return t->tts->speakers(); }

int courier_tts_speak(courier_tts* t, const char* text, int speaker, float speed, float variation, int16_t** pcm) {
    return guard([&]() -> int {
        auto a = t->tts->speak(text, speaker, speed, variation);
        *pcm = static_cast<int16_t*>(std::malloc(std::max<size_t>(1, a.size()) * sizeof(int16_t)));
        if (!*pcm) throw std::runtime_error("out of memory");
        std::memcpy(*pcm, a.data(), a.size() * sizeof(int16_t));
        return static_cast<int>(a.size());
    }, -1);
}

char* courier_tts_explain(courier_tts* t, const char* text) {
    return guard([&]() -> char* {
        std::string said = speakable(text);
        nlohmann::json j = {{"text", said}, {"ipa", t->tts->ipa(said)}};
        return dup(j.dump());
    }, nullptr);
}

void courier_tts_free(courier_tts* t) { delete t; }

}  // extern "C"
