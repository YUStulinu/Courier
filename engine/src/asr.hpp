// Speech recognition with whisper.cpp. The orchestrator calls it repeatedly on the growing utterance
// while you speak (partial transcripts) and once more when you stop, if anything changed since.
#pragma once

#include <string>

struct whisper_context;

namespace courier {

// "Remind me too. Remind me too. Remind me too." -> "Remind me too.": Whisper's loops, collapsed.
std::string collapse_repeats(const std::string& text);

struct Transcript {
    std::string text;
    double ms = 0;
};

class Asr {
public:
    Asr(const std::string& model, int threads, bool gpu);
    ~Asr();
    Asr(const Asr&) = delete;
    Asr& operator=(const Asr&) = delete;

    // 16 kHz mono. `fast`: an encoder window sized to the audio instead of Whisper's 30 s.
    Transcript transcribe(const float* audio, int samples, bool fast);

private:
    whisper_context* ctx_ = nullptr;
    int threads_;
};

}  // namespace courier
