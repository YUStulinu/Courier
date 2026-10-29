// The wake word detector, in three stages:
//   1. log-mel features: 32 bands, 10 ms hop, 32 ms window (speech_mel.onnx);
//   2. a speech embedding every 80 ms from the last 76 mel frames: 96 numbers that describe the sound,
//      learned by Google on a large amount of real speech (speech_embedding.onnx, Apache-2.0, as
//      packaged by openWakeWord);
//   3. our classifier on the last 16 embeddings (1.3 s), trained in wakeword/ on synthetic speech:
//      "emb" [batch, 16, 96] -> "logit" [batch].
// The first two are frozen and were trained on real voices, which is what lets a classifier trained on
// synthetic voices work on real ones. All three files live in one folder.
#pragma once

#include <deque>
#include <memory>
#include <string>
#include <vector>

#include "onnx.hpp"

namespace courier {

// Stages 1 and 2, streaming: audio in, embeddings out.
class SpeechFeatures {
public:
    static constexpr int kBands = 32;
    static constexpr int kMelWindow = 512;  // samples per mel frame
    static constexpr int kMelHop = 160;
    static constexpr int kEmbFrames = 76;   // mel frames per embedding
    static constexpr int kEmbHop = 8;       // mel frames between embeddings (80 ms)
    static constexpr int kDim = 96;

    explicit SpeechFeatures(const std::string& folder);
    // Feeds 16 kHz audio in [-1, 1]; returns the embeddings it completed (kDim floats each).
    std::vector<float> process(const float* samples, int n);
    void reset();

private:
    std::unique_ptr<Ort::Session> mel_, emb_;
    std::vector<float> pending_;  // audio not yet turned into mel frames
    std::deque<float> frames_;    // the last kEmbFrames mel frames
    long long frames_seen_ = 0;
};

class WakeWord {
public:
    static constexpr int kEmbeddings = 16;

    // `model` is the classifier; the feature models are read from its folder.
    WakeWord(const std::string& model, float threshold, int hits, int refractory_ms);

    struct Result {
        std::vector<float> scores;
        int detections = 0;
    };
    Result process(const float* samples, int n);
    double last_ms() const { return last_ms_; }
    void set_threshold(float t) { threshold_ = t; }
    void reset();

private:
    SpeechFeatures features_;
    std::unique_ptr<Ort::Session> classifier_;
    std::deque<float> window_;  // the last kEmbeddings embeddings
    long long embeddings_seen_ = 0;
    float threshold_;
    int hits_needed_, refractory_;
    int run_ = 0;
    long long quiet_until_ = 0;  // embedding index
    double last_ms_ = -1;
};

}  // namespace courier
