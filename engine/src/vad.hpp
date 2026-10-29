// Speech detection: one speech probability per 32 ms frame (512 samples at 16 kHz). Deciding where an
// utterance starts and ends from these is the orchestrator's job (endpointing, in Rust).
#pragma once

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "onnx.hpp"

namespace courier {

class Vad {
public:
    static constexpr int kFrame = 512;
    virtual ~Vad() = default;
    // Feeds samples; returns the probabilities of the frames they completed.
    std::vector<float> process(const float* samples, int n);
    virtual void reset();

protected:
    virtual float frame(const float* f) = 0;

private:
    std::vector<float> pending_;
};

// Silero VAD v6: a small recurrent network. Each frame is given with the 64 samples before it, and the
// network's state carries over between frames.
class SileroVad : public Vad {
public:
    explicit SileroVad(const std::string& model);
    void reset() override;

protected:
    float frame(const float* f) override;

private:
    static constexpr int kContext = 64;
    std::unique_ptr<Ort::Session> session_;
    std::array<float, 2 * 128> state_{};
    std::array<float, kContext + kFrame> input_{};
};

// The baseline: frame energy against a noise floor that follows the quietest frames (fast down, slow
// up). Speech is 12 dB over the floor; between 6 and 12 dB the previous decision holds (hysteresis).
class EnergyVad : public Vad {
public:
    void reset() override;

protected:
    float frame(const float* f) override;

private:
    float floor_db_ = -50.f;
    bool speech_ = false;
};

}  // namespace courier
