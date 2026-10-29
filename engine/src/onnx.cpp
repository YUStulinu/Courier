#include "onnx.hpp"

#include <stdexcept>

namespace courier {

Ort::Env& ort_env() {
    static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "courier");
    return env;
}

std::unique_ptr<Ort::Session> ort_session(const std::filesystem::path& model, int threads) {
    if (!std::filesystem::exists(model)) throw std::runtime_error("model not found: " + model.string());
    Ort::SessionOptions opt;
    if (threads > 0) opt.SetIntraOpNumThreads(threads);
    // Small models called every few milliseconds: spinning threads would only burn a core.
    opt.AddConfigEntry("session.intra_op.allow_spinning", "0");
    opt.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    return std::make_unique<Ort::Session>(ort_env(), model.native().c_str(), opt);
}

}  // namespace courier
