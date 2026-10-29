#include "vad.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace courier {

std::vector<float> Vad::process(const float* samples, int n) {
    std::vector<float> out;
    pending_.insert(pending_.end(), samples, samples + n);
    size_t i = 0;
    for (; i + kFrame <= pending_.size(); i += kFrame) out.push_back(frame(pending_.data() + i));
    pending_.erase(pending_.begin(), pending_.begin() + static_cast<long>(i));
    return out;
}

void Vad::reset() { pending_.clear(); }

SileroVad::SileroVad(const std::string& model) : session_(ort_session(model, 1)) {}

void SileroVad::reset() {
    Vad::reset();
    state_.fill(0.f);
    input_.fill(0.f);
}

float SileroVad::frame(const float* f) {
    // The context: the last 64 samples of the previous frame, already at the front of input_.
    std::memcpy(input_.data() + kContext, f, kFrame * sizeof(float));
    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::array<int64_t, 2> ishape{1, kContext + kFrame};
    std::array<int64_t, 3> sshape{2, 1, 128};
    int64_t sr = 16000;
    std::array<Ort::Value, 3> inputs = {
        Ort::Value::CreateTensor<float>(mem, input_.data(), input_.size(), ishape.data(), 2),
        Ort::Value::CreateTensor<float>(mem, state_.data(), state_.size(), sshape.data(), 3),
        Ort::Value::CreateTensor<int64_t>(mem, &sr, 1, nullptr, 0),
    };
    const char* in_names[] = {"input", "state", "sr"};
    const char* out_names[] = {"output", "stateN"};
    auto out = session_->Run(Ort::RunOptions{nullptr}, in_names, inputs.data(), 3, out_names, 2);
    float p = out[0].GetTensorData<float>()[0];
    std::memcpy(state_.data(), out[1].GetTensorData<float>(), state_.size() * sizeof(float));
    std::memmove(input_.data(), input_.data() + kFrame, kContext * sizeof(float));
    return p;
}

void EnergyVad::reset() {
    Vad::reset();
    floor_db_ = -50.f;
    speech_ = false;
}

float EnergyVad::frame(const float* f) {
    double e = 0;
    for (int i = 0; i < kFrame; ++i) e += static_cast<double>(f[i]) * f[i];
    float db = 10.f * std::log10(static_cast<float>(e / kFrame) + 1e-10f);
    // The floor drops at once to a quieter frame and rises 1 dB a second.
    if (db < floor_db_) floor_db_ = db;
    else floor_db_ += 0.032f;
    floor_db_ = std::clamp(floor_db_, -80.f, -20.f);
    if (db > floor_db_ + 12.f) speech_ = true;
    else if (db < floor_db_ + 6.f) speech_ = false;
    return speech_ ? 1.f : 0.f;
}

}  // namespace courier
