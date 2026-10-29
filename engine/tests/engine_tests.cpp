// Tests of the engine. The parts without models always run; the ones with models run when the files
// are there (scripts/fetch-deps.sh), and the language model one when COURIER_TEST_LLM names a GGUF file.
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <random>
#include <string>
#include <vector>

#include "asr.hpp"
#include "llm.hpp"
#include "tts.hpp"
#include "vad.hpp"
#include "wake.hpp"

using namespace courier;
namespace fs = std::filesystem;

namespace {
int failures = 0;
void check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    std::fflush(stdout);
    if (!ok) ++failures;
}

const fs::path kModels = COURIER_MODELS;
const fs::path kSource = COURIER_SOURCE;

bool have(const fs::path& p) {
    if (fs::exists(p)) return true;
    std::printf("skip (no %s)\n", p.filename().string().c_str());
    return false;
}

std::vector<float> to16k(const std::vector<int16_t>& pcm, int rate) {
    // Linear interpolation is enough for tests.
    std::vector<float> out(static_cast<size_t>(pcm.size() * 16000.0 / rate));
    for (size_t i = 0; i < out.size(); ++i) {
        double x = i * static_cast<double>(rate) / 16000.0;
        size_t k = static_cast<size_t>(x);
        double f = x - k;
        float a = pcm[std::min(k, pcm.size() - 1)] / 32768.f, b = pcm[std::min(k + 1, pcm.size() - 1)] / 32768.f;
        out[i] = static_cast<float>(a + (b - a) * f);
    }
    return out;
}

std::string words_of(const std::string& s) {
    std::string out;
    for (unsigned char c : s)
        if (std::isalnum(c) || c == ' ') out += static_cast<char>(std::tolower(c));
    std::string clean;
    for (char c : out)
        if (!(c == ' ' && (clean.empty() || clean.back() == ' '))) clean += c;
    while (!clean.empty() && clean.back() == ' ') clean.pop_back();
    return clean;
}
}  // namespace

static void test_features() {
    const fs::path folder = kSource / "models" / "wake";
    if (!have(folder / "speech_embedding.onnx")) return;
    // Two seconds of a voice-like sound: one embedding every 80 ms once 76 mel frames are in, and the
    // same embeddings however the audio is cut into pieces.
    std::vector<float> a(32000);
    for (size_t i = 0; i < a.size(); ++i) {
        double t = i / 16000.0;
        a[i] = static_cast<float>(0.2 * std::sin(2 * 3.14159265358979 * 150 * t) * (0.5 + 0.5 * std::sin(2 * 3.14159265358979 * 3 * t)));
    }
    SpeechFeatures whole(folder.string());
    auto f = whole.process(a.data(), static_cast<int>(a.size()));
    // Mel frames come in blocks of 8 (1632 samples, then every 1280).
    int frames = ((32000 - 1632) / 1280 + 1) * 8;
    int want = (frames - SpeechFeatures::kEmbFrames) / SpeechFeatures::kEmbHop + 1;
    check(static_cast<int>(f.size()) == want * SpeechFeatures::kDim, "embedding count (" + std::to_string(f.size() / 96) + ")");
    SpeechFeatures chunked(folder.string());
    std::vector<float> g;
    for (size_t i = 0; i < a.size(); i += 333) {
        auto part = chunked.process(a.data() + i, static_cast<int>(std::min<size_t>(333, a.size() - i)));
        g.insert(g.end(), part.begin(), part.end());
    }
    float diff = 0;
    for (size_t i = 0; i < std::min(f.size(), g.size()); ++i) diff = std::max(diff, std::fabs(f[i] - g[i]));
    check(f.size() == g.size() && diff < 1e-3f, "chunked embeddings equal whole ones (max difference " + std::to_string(diff) + ")");
}

static void test_energy_vad() {
    std::mt19937 rng(7);
    std::normal_distribution<float> noise(0.f, 0.002f);
    std::vector<float> a;
    auto add = [&](double sec, bool voice) {
        for (size_t i = 0, n = static_cast<size_t>(sec * 16000); i < n; ++i) {
            float s = noise(rng);
            if (voice) {
                double t = static_cast<double>(a.size()) / 16000.0;
                for (int h = 1; h <= 4; ++h) s += static_cast<float>(0.08 / h * std::sin(2 * 3.14159265358979 * 150 * h * t));
            }
            a.push_back(s);
        }
    };
    add(1.0, false);
    add(1.0, true);
    add(1.0, false);
    EnergyVad v;
    auto p = v.process(a.data(), static_cast<int>(a.size()));
    // Frames of 32 ms: the second half second of each part, to leave the edges out.
    auto mean = [&](double from, double to) {
        int i0 = static_cast<int>(from * 16000 / 512), i1 = static_cast<int>(to * 16000 / 512);
        float s = 0;
        for (int i = i0; i < i1; ++i) s += p[i];
        return s / (i1 - i0);
    };
    check(mean(0.5, 1.0) == 0.f && mean(1.5, 2.0) == 1.f && mean(2.5, 3.0) == 0.f, "energy detector: silence, tone, silence");
}

static void test_speakable() {
    struct Case {
        const char* in;
        const char* out;
    };
    const Case cases[] = {
        {"Your timer is set for 5 minutes.", "Your timer is set for five minutes."},
        {"The meeting is at 6:30 PM.", "The meeting is at six thirty p m."},
        {"Wake up at 7:05 am", "Wake up at seven oh five a m"},
        {"It starts at 18:00.", "It starts at six p m."},
        {"at 9 o'clock", "at nine o'clock"},
        {"Lunch at 12:00", "Lunch at twelve o'clock"},
        {"It's 21.5°C outside.", "It's twenty one point five degrees Celsius outside."},
        {"-3° tonight", "minus three degrees tonight"},
        {"on March 3rd, 2026", "on March third, twenty twenty six"},
        {"the 21st and the 12th", "the twenty first and the twelfth"},
        {"a 40% chance", "a forty percent chance"},
        {"It costs $1,250.", "It costs one thousand two hundred fifty dollars."},
        {"1 €", "one EUR"},
        {"**Bold** and `code`", "Bold and code"},
        {"call me at 6 pm", "call me at six p m"},
        {"I am here", "I am here"},
        {"5km away", "five km away"},
        {"Tom & Jerry", "Tom and Jerry"},
    };
    for (const auto& c : cases) {
        std::string got = speakable(c.in);
        check(got == c.out, std::string("speakable: \"") + c.in + "\" -> \"" + got + "\"");
    }
}

static void test_repeats() {
    check(collapse_repeats("Remind me too. Remind me too. Remind me too.") == "Remind me too.", "a looped phrase is collapsed");
    check(collapse_repeats("Set a time of 4. Set a time of 4. Set a time") == "Set a time of 4. Set a time", "a loop cut mid-phrase");
    check(collapse_repeats("It was very very good.") == "It was very very good.", "a word said twice stays");
    check(collapse_repeats("no no no no no no") == "no no", "a word looping is cut to two");
    check(collapse_repeats("Play some music by Miles Davis.") == "Play some music by Miles Davis.", "normal speech is unchanged");
}

static void test_utf8() {
    std::string s = "caf\xC3\xA9";  // "café"
    check(utf8_complete(s) == s.size(), "utf8: a whole character is complete");
    check(utf8_complete(s.substr(0, 4)) == 3, "utf8: half a character is held back");
    check(utf8_complete("ab") == 2 && utf8_complete("") == 0, "utf8: ascii");
    std::string emoji = "\xF0\x9F\x99\x82";
    check(utf8_complete(emoji.substr(0, 3)) == 0 && utf8_complete(emoji) == 4, "utf8: four-byte characters");
}

static void test_silero_and_voice() {
    const fs::path silero = kModels / "silero_vad-6.2.3.onnx";
    const fs::path voice = kModels / "en_US-ljspeech-medium.onnx";
    const fs::path whisper = kModels / "ggml-base.en-q5_1.bin";
    if (!have(silero) || !have(voice) || !have(whisper)) return;
    Tts tts(voice, voice.string() + ".json", kModels / "cmudict.dict", 2);
    const std::string said = "Set a timer for ten minutes, and remind me to call Anna at 6:30 PM.";
    auto pcm = tts.speak(said, 0, 1.f, 1.f);
    check(pcm.size() > static_cast<size_t>(tts.sample_rate()), "the voice speaks (" + std::to_string(pcm.size()) + " samples)");
    std::vector<float> speech = to16k(pcm, tts.sample_rate());

    // One second of silence, the sentence, one second of silence.
    std::vector<float> a(16000, 0.f);
    a.insert(a.end(), speech.begin(), speech.end());
    a.insert(a.end(), 16000, 0.f);
    SileroVad vad(silero.string());
    auto p = vad.process(a.data(), static_cast<int>(a.size()));
    int start = 16000 / 512, end = static_cast<int>((16000 + speech.size()) / 512);
    int speech_frames = 0, silence_hits = 0;
    for (int i = start + 5; i < end - 5; ++i) speech_frames += p[i] > 0.5f;
    for (int i = 0; i < start - 2; ++i) silence_hits += p[i] > 0.5f;
    for (int i = end + 10; i < static_cast<int>(p.size()); ++i) silence_hits += p[i] > 0.5f;
    check(speech_frames > (end - start - 10) * 7 / 10, "Silero: most of the sentence is speech (" + std::to_string(speech_frames) + "/" +
                                                           std::to_string(end - start - 10) + ")");
    check(silence_hits == 0, "Silero: silence is not speech");

    Asr asr(whisper.string(), 0, false);
    auto t = asr.transcribe(speech.data(), static_cast<int>(speech.size()), false);
    std::string got = words_of(t.text);
    // Whisper writes numbers as digits, and base.en (the small model the tests use) mishears a word or
    // two: compare loosely on what matters.
    bool ok = (got.find("10 minutes") != std::string::npos || got.find("ten minutes") != std::string::npos) &&
              got.find("remind me to call") != std::string::npos &&
              (got.find("630") != std::string::npos || got.find("six thirty") != std::string::npos ||
               got.find("6 30") != std::string::npos);
    check(ok, "voice -> Whisper round trip: \"" + t.text + "\"");
    auto fast = asr.transcribe(speech.data(), static_cast<int>(speech.size()), true);
    check(words_of(fast.text).find("remind me to call") != std::string::npos, "Whisper with a short encoder window: \"" + fast.text + "\"");

    auto ex = tts.ipa(speakable("Hey Courier"));
    check(ex.find("k") != std::string::npos && ex.size() > 6, "IPA of the wake phrase: " + ex);
    auto raw = tts.ipa("Hey [[kˈʊɹiɚ]], hi.");
    check(raw.find("kˈʊɹiɚ,") != std::string::npos, "[[IPA]] passes through: " + raw);
}

static void test_wake() {
    const fs::path model = kSource / "models" / "wake" / "hey_courier.onnx";
    const fs::path voice = kModels / "en_US-ljspeech-medium.onnx";
    if (!have(model) || !have(voice)) return;
    Tts tts(voice, voice.string() + ".json", kModels / "cmudict.dict", 2);
    // A voice the detector never trained on (LJSpeech), the request following the phrase.
    // "COOR-ee-er": this voice says "Courier" as CUR-ee-er from the dictionary, which the detector
    // misses in it (see the README: the detector depends on the voice more than it should).
    auto speech = to16k(tts.speak("Hey [[kˈʊɹiɚ]], what's the weather like?", 0, 1.f, 1.f), tts.sample_rate());
    std::vector<float> a(40000, 0.f);  // the detector scores once it has 2 s of audio
    a.insert(a.end(), speech.begin(), speech.end());
    a.insert(a.end(), 16000, 0.f);
    WakeWord w(model.string(), 0.5f, 2, 1500);
    int detections = 0;
    double at = -1;
    for (size_t i = 0; i < a.size(); i += 320) {
        auto r = w.process(a.data() + i, static_cast<int>(std::min<size_t>(320, a.size() - i)));
        detections += r.detections;
        if (r.detections) at = w.last_ms();
    }
    check(detections == 1, "the wake phrase is detected once (" + std::to_string(detections) + ")");
    check(at > 2500 && at < 2500 + 1600, "... soon after it was said (" + std::to_string(static_cast<int>(at)) + " ms)");

    auto other = to16k(tts.speak("The courier delivered the package to the office this morning.", 0, 1.f, 1.f), tts.sample_rate());
    w.reset();
    int false_hits = w.process(other.data(), static_cast<int>(other.size())).detections;
    check(false_hits == 0, "a sentence with \"courier\" in it does not wake it");
}

static void test_llm() {
    const char* path = std::getenv("COURIER_TEST_LLM");
    if (!path || !fs::exists(path)) {
        std::printf("skip (COURIER_TEST_LLM not set)\n");
        return;
    }
    Llm llm(path, 4096, 99);
    const std::string schema = R"({"anyOf":[
      {"type":"object","properties":{"name":{"const":"set_timer"},"arguments":{"type":"object","properties":{"minutes":{"type":"number"}},"required":["minutes"],"additionalProperties":false}},"required":["name","arguments"],"additionalProperties":false},
      {"type":"object","properties":{"name":{"const":"get_weather"},"arguments":{"type":"object","properties":{"place":{"type":"string"}},"required":[],"additionalProperties":false}},"required":["name","arguments"],"additionalProperties":false}]})";
    const std::string system =
        "<|im_start|>system\nYou are a voice assistant.\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
        "You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n"
        R"({"type":"function","function":{"name":"set_timer","description":"Start a countdown timer.","parameters":{"type":"object","properties":{"minutes":{"type":"number"}},"required":["minutes"]}}})"
        "\n"
        R"({"type":"function","function":{"name":"get_weather","description":"The weather forecast.","parameters":{"type":"object","properties":{"place":{"type":"string"}}}}})"
        "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
        "<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n";
    std::string streamed;
    auto g = llm.generate(system + "<|im_start|>user\nSet a timer for ten minutes.<|im_end|>\n<|im_start|>assistant\n", "<tool_call>", schema, 128,
                          [&](const std::string& p) {
                              streamed += p;
                              return true;
                          });
    check(g.stop == "tool", "the model calls a tool (stop: " + g.stop + ")");
    check(streamed == g.text, "streamed pieces add up to the text");
    size_t at = g.text.find("<tool_call>");
    bool parsed = false;
    if (at != std::string::npos) {
        try {
            auto j = nlohmann::json::parse(g.text.substr(at + 11));
            parsed = j["name"] == "set_timer" && j["arguments"]["minutes"].get<double>() == 10;
        } catch (...) {
        }
    }
    check(parsed, "the call is set_timer with 10 minutes: " + g.text);
    // The same system prompt again: only the new part is computed.
    auto h = llm.generate(system + "<|im_start|>user\nHi!<|im_end|>\n<|im_start|>assistant\n", "<tool_call>", schema, 32, nullptr);
    int sys_tokens = llm.count_tokens(system);
    check(h.reused_tokens >= sys_tokens, "the system prompt is reused (" + std::to_string(h.reused_tokens) + " of " +
                                             std::to_string(h.prompt_tokens) + " tokens; the system prompt is " + std::to_string(sys_tokens) + ")");
    check(h.stop == "end" || h.stop == "length", "a greeting is answered in words (stop: " + h.stop + ")");
    int stops = 0;
    auto c = llm.generate(system + "<|im_start|>user\nTell me a long story.<|im_end|>\n<|im_start|>assistant\n", "", "", 64,
                          [&](const std::string&) { return ++stops < 3; });
    check(c.stop == "cancel" && stops == 3, "returning false stops generation");
}

int main() {
    test_features();
    test_energy_vad();
    test_speakable();
    test_utf8();
    test_repeats();
    test_silero_and_voice();
    test_wake();
    test_llm();
    std::printf("%s (%d failed)\n", failures ? "FAILED" : "all passed", failures);
    return failures ? 1 : 0;
}
