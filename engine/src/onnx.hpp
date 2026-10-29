// onnxruntime sessions, shared by speech detection, the wake word and the voice. All run on the CPU:
// they are small, and the GPU is busy with Whisper and the language model.
#pragma once

#include <filesystem>
#include <memory>
#include <onnxruntime_cxx_api.h>

namespace courier {

Ort::Env& ort_env();

// A session for a model file with `threads` intra-op threads (0: onnxruntime's choice).
std::unique_ptr<Ort::Session> ort_session(const std::filesystem::path& model, int threads);

}  // namespace courier
