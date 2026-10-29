#include "asr.hpp"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <vector>
#include <whisper.h>

namespace courier {

std::string collapse_repeats(const std::string& text) {
    // Words, keeping their punctuation; compared without it and without case.
    std::vector<std::string> words;
    std::istringstream in(text);
    for (std::string w; in >> w;) words.push_back(w);
    auto key = [](const std::string& w) {
        std::string k;
        for (unsigned char c : w)
            if (std::isalnum(c) || c == '\'') k += static_cast<char>(std::tolower(c));
        return k;
    };
    // A run of n words repeated right after itself: drop the copies (longest runs first).
    for (size_t n = std::min<size_t>(12, words.size() / 2); n >= 1; --n) {
        for (size_t i = 0; i + 2 * n <= words.size();) {
            bool same = true;
            for (size_t k = 0; k < n && same; ++k) same = key(words[i + k]) == key(words[i + n + k]);
            // Single words may repeat in real speech ("very very"); runs of one need three.
            bool loop = same && (n > 1 || (i + 3 <= words.size() && key(words[i + 2]) == key(words[i])));
            if (loop) words.erase(words.begin() + static_cast<long>(i + n), words.begin() + static_cast<long>(i + 2 * n));
            else ++i;
        }
        if (n == 1) break;
    }
    std::string out;
    for (const auto& w : words) out += (out.empty() ? "" : " ") + w;
    return out;
}

Asr::Asr(const std::string& model, int threads, bool gpu)
    : threads_(threads > 0 ? threads : static_cast<int>(std::max(2u, std::thread::hardware_concurrency() / 2))) {
    // whisper.cpp logs every load to stderr; COURIER_VERBOSE=1 keeps it.
    if (!std::getenv("COURIER_VERBOSE")) whisper_log_set([](ggml_log_level, const char*, void*) {}, nullptr);
    whisper_context_params cp = whisper_context_default_params();
    cp.use_gpu = gpu;  // only matters when built with CUDA
    cp.flash_attn = gpu;
    ctx_ = whisper_init_from_file_with_params(model.c_str(), cp);
    if (!ctx_) throw std::runtime_error("cannot load the speech model " + model);
}

Asr::~Asr() {
    if (ctx_) whisper_free(ctx_);
}

Transcript Asr::transcribe(const float* audio, int samples, bool fast) {
    auto t0 = std::chrono::steady_clock::now();
    whisper_full_params p = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    p.n_threads = threads_;
    p.language = "en";
    p.translate = false;
    p.no_context = true;
    p.no_timestamps = true;
    p.single_segment = true;
    p.print_progress = false;
    p.print_realtime = false;
    p.print_timestamps = false;
    p.print_special = false;
    p.suppress_blank = true;
    p.suppress_nst = true;     // no "[music]", "(coughs)" and the like
    p.temperature_inc = 0.f;   // no retries at higher temperature: one pass, predictable time
    // Whisper wants at least a second; pad short utterances with silence.
    std::vector<float> buf(audio, audio + samples);
    if (buf.size() < 16000) buf.resize(16000 + 1600, 0.f);
    if (fast) {
        // 1500 encoder positions are 30 s; keep the audio's share plus a margin (whisper.cpp's
        // documented speed-up for short audio).
        int ctx = static_cast<int>(buf.size() / 16000.0 * 50.0) + 64;
        p.audio_ctx = std::min(1500, ctx);
    }
    // Nobody says more than about five tokens a second. Without a limit, a cut-off phrase can send the
    // decoder into a loop ("Remind me too. Remind me too. ...") for seconds.
    p.max_tokens = 8 + static_cast<int>(samples / 16000.0 * 6.0);
    if (whisper_full(ctx_, p, buf.data(), static_cast<int>(buf.size())) != 0) throw std::runtime_error("speech recognition failed");
    Transcript out;
    for (int s = 0; s < whisper_full_n_segments(ctx_); ++s) out.text += whisper_full_get_segment_text(ctx_, s);
    out.text = collapse_repeats(out.text);
    out.text.erase(0, out.text.find_first_not_of(' '));
    out.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return out;
}

}  // namespace courier
