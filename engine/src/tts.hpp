// The voice: Piper (VITS, ONNX). The assistant speaks with the LJSpeech voice (public domain); the wake
// word training uses the multi-speaker voices through the same code.
//
// Piper voices read espeak's IPA. Instead of espeak (GPL), words go through the CMU dictionary and its
// ARPAbet is written as the IPA espeak produces for American English, with stress marks. Before that,
// what a reply contains but nobody says as written is spelled out: "6:30 PM" -> "six thirty p m",
// "21.5°C" -> "twenty one point five degrees", "3rd" -> "third", "40%" -> "forty percent".
#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "g2p.hpp"
#include "onnx.hpp"

namespace courier {

// Text as it is said.
std::string speakable(const std::string& text);

// ARPAbet with stress -> IPA in espeak's style ("AH0" -> "ə", "AH1" -> "ˈʌ").
std::string arpabet_to_ipa(const std::vector<std::string>& phones);

class Tts {
public:
    Tts(const std::filesystem::path& voice, const std::filesystem::path& config, const std::filesystem::path& cmudict, int threads);
    int sample_rate() const { return rate_; }
    int speakers() const { return speakers_; }
    // The IPA the voice reads for already speakable text.
    std::string ipa(const std::string& text) const;
    std::vector<int16_t> speak(const std::string& text, int speaker, float speed, float variation);

private:
    std::vector<int64_t> ids(const std::string& ipa) const;
    std::unique_ptr<Ort::Session> session_;
    G2P g2p_;
    std::unordered_map<uint32_t, int64_t> id_of_;
    int rate_ = 22050;
    int speakers_ = 1;
    float noise_ = 0.667f, length_ = 1.f, noise_w_ = 0.8f;
};

}  // namespace courier
