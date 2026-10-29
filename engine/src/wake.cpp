#include "wake.hpp"

#include <array>
#include <cmath>
#include <filesystem>
#include <stdexcept>

namespace courier {

namespace {
float sigmoid(float x) { return 1.f / (1.f + std::exp(-x)); }
}  // namespace

SpeechFeatures::SpeechFeatures(const std::string& folder)
    : mel_(ort_session(std::filesystem::path(folder) / "speech_mel.onnx", 1)),
      emb_(ort_session(std::filesystem::path(folder) / "speech_embedding.onnx", 1)) {}

void SpeechFeatures::reset() {
    pending_.clear();
    frames_.clear();
    frames_seen_ = 0;
}

std::vector<float> SpeechFeatures::process(const float* samples, int n) {
    std::vector<float> out;
    // The models were trained on 16-bit sample values.
    for (int i = 0; i < n; ++i) pending_.push_back(samples[i] * 32768.f);
    // The mel model converts to dB and clips everything 80 dB under the loudest frame it is given, so
    // its output depends on how much audio it gets at once. It always gets blocks of exactly 8 frames
    // (as openWakeWord feeds it), so the features do not depend on how the audio arrives.
    constexpr int kBlock = kMelWindow + (kEmbHop - 1) * kMelHop;  // 1632 samples -> 8 frames
    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<float> mel;
    size_t at = 0;  // consumed samples; erased once at the end (erasing per block is quadratic on long input)
    while (pending_.size() - at >= static_cast<size_t>(kBlock)) {
        std::array<int64_t, 2> shape{1, kBlock};
        auto in = Ort::Value::CreateTensor<float>(mem, pending_.data() + at, static_cast<size_t>(kBlock), shape.data(), 2);
        const char* mel_in[] = {"input"};
        const char* mel_out[] = {"output"};
        auto o = mel_->Run(Ort::RunOptions{nullptr}, mel_in, &in, 1, mel_out, 1);
        const float* m = o[0].GetTensorData<float>();
        mel.insert(mel.end(), m, m + kEmbHop * kBands);
        at += kEmbHop * kMelHop;
    }
    pending_.erase(pending_.begin(), pending_.begin() + static_cast<long>(at));
    const float* m = mel.data();
    const int k = static_cast<int>(mel.size() / kBands);

    // Each new frame; an embedding whenever 8 more have come in and 76 are there.
    std::vector<float> batch;
    for (int f = 0; f < k; ++f) {
        for (int b = 0; b < kBands; ++b) frames_.push_back(m[f * kBands + b] / 10.f + 2.f);  // openWakeWord's scaling
        while (frames_.size() > static_cast<size_t>(kEmbFrames) * kBands) frames_.erase(frames_.begin(), frames_.begin() + kBands);
        ++frames_seen_;
        if (frames_seen_ >= kEmbFrames && (frames_seen_ - kEmbFrames) % kEmbHop == 0) batch.insert(batch.end(), frames_.begin(), frames_.end());
    }
    const int64_t nb = static_cast<int64_t>(batch.size() / (kEmbFrames * kBands));
    if (nb == 0) return out;
    std::array<int64_t, 4> eshape{nb, kEmbFrames, kBands, 1};
    auto ein = Ort::Value::CreateTensor<float>(mem, batch.data(), batch.size(), eshape.data(), 4);
    const char* emb_in[] = {"input_1"};
    const char* emb_out[] = {"conv2d_19"};
    auto e = emb_->Run(Ort::RunOptions{nullptr}, emb_in, &ein, 1, emb_out, 1);
    const float* ev = e[0].GetTensorData<float>();
    out.assign(ev, ev + nb * kDim);
    return out;
}

WakeWord::WakeWord(const std::string& model, float threshold, int hits, int refractory_ms)
    : features_(std::filesystem::path(model).parent_path().string()),
      classifier_(ort_session(model, 1)),
      threshold_(threshold),
      hits_needed_(std::max(1, hits)),
      refractory_(std::max(0, refractory_ms / 80)) {}

void WakeWord::reset() {
    features_.reset();
    window_.clear();
    embeddings_seen_ = 0;
    run_ = 0;
    quiet_until_ = 0;
    last_ms_ = -1;
}

WakeWord::Result WakeWord::process(const float* samples, int n) {
    Result r;
    auto emb = features_.process(samples, n);
    const size_t dim = SpeechFeatures::kDim;
    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    for (size_t i = 0; i < emb.size(); i += dim) {
        window_.insert(window_.end(), emb.begin() + static_cast<long>(i), emb.begin() + static_cast<long>(i + dim));
        while (window_.size() > kEmbeddings * dim) window_.erase(window_.begin(), window_.begin() + static_cast<long>(dim));
        const long long e = embeddings_seen_++;
        if (window_.size() < kEmbeddings * dim) continue;
        std::vector<float> in(window_.begin(), window_.end());
        std::array<int64_t, 3> shape{1, kEmbeddings, static_cast<int64_t>(dim)};
        auto t = Ort::Value::CreateTensor<float>(mem, in.data(), in.size(), shape.data(), 3);
        const char* names_in[] = {"emb"};
        const char* names_out[] = {"logit"};
        auto o = classifier_->Run(Ort::RunOptions{nullptr}, names_in, &t, 1, names_out, 1);
        float s = sigmoid(o[0].GetTensorData<float>()[0]);
        r.scores.push_back(s);
        if (e < quiet_until_) {
            run_ = 0;
            continue;
        }
        run_ = s >= threshold_ ? run_ + 1 : 0;
        if (run_ >= hits_needed_) {
            ++r.detections;
            run_ = 0;
            quiet_until_ = e + refractory_ + 1;
            // Embedding e ends with mel frame 8e + 75, whose audio ends 512 samples after it starts.
            last_ms_ = ((SpeechFeatures::kEmbHop * e + SpeechFeatures::kEmbFrames - 1) * SpeechFeatures::kMelHop + SpeechFeatures::kMelWindow) / 16.0;
        }
    }
    return r;
}

}  // namespace courier
