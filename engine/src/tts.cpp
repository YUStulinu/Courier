#include "tts.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <fstream>
#include <map>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace courier {

namespace {

const char* kOnes[] = {"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
                       "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"};
const char* kTens[] = {"", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"};

std::string say_number(long long n) {
    if (n < 0) return "minus " + say_number(-n);
    if (n < 20) return kOnes[n];
    if (n < 100) return std::string(kTens[n / 10]) + (n % 10 ? std::string(" ") + kOnes[n % 10] : "");
    if (n < 1000) return std::string(kOnes[n / 100]) + " hundred" + (n % 100 ? " " + say_number(n % 100) : "");
    if (n < 1000000) return say_number(n / 1000) + " thousand" + (n % 1000 ? " " + say_number(n % 1000) : "");
    if (n < 1000000000) return say_number(n / 1000000) + " million" + (n % 1000000 ? " " + say_number(n % 1000000) : "");
    return say_number(n / 1000000000) + " billion" + (n % 1000000000 ? " " + say_number(n % 1000000000) : "");
}

// "twenty one" -> "twenty first", "twelve" -> "twelfth".
std::string ordinal(long long n) {
    std::string w = say_number(n);
    size_t sp = w.rfind(' ');
    std::string head = sp == std::string::npos ? "" : w.substr(0, sp + 1);
    std::string last = sp == std::string::npos ? w : w.substr(sp + 1);
    static const std::map<std::string, std::string> irregular = {
        {"one", "first"}, {"two", "second"}, {"three", "third"}, {"five", "fifth"}, {"eight", "eighth"},
        {"nine", "ninth"}, {"twelve", "twelfth"}};
    auto it = irregular.find(last);
    if (it != irregular.end()) return head + it->second;
    if (last.back() == 'y') return head + last.substr(0, last.size() - 1) + "ieth";
    return head + last + "th";
}

// 2026 -> "twenty twenty six", 1905 -> "nineteen oh five", 2000 -> "two thousand".
std::string say_year(long long y) {
    if (y % 1000 == 0 || (y >= 2000 && y < 2010)) return say_number(y);
    long long hi = y / 100, lo = y % 100;
    if (lo == 0) return say_number(hi) + " hundred";
    return say_number(hi) + (lo < 10 ? " oh " : " ") + say_number(lo);
}

std::string lower(std::string s) {
    for (auto& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

bool is_digit(char c) { return std::isdigit(static_cast<unsigned char>(c)) != 0; }
bool is_alpha(char c) { return std::isalpha(static_cast<unsigned char>(c)) != 0; }

// UTF-8 -> code points.
std::vector<uint32_t> codepoints(const std::string& s) {
    std::vector<uint32_t> out;
    for (size_t i = 0; i < s.size();) {
        unsigned char c = static_cast<unsigned char>(s[i]);
        uint32_t cp;
        int len;
        if (c < 0x80) { cp = c; len = 1; }
        else if ((c >> 5) == 6) { cp = c & 0x1f; len = 2; }
        else if ((c >> 4) == 14) { cp = c & 0x0f; len = 3; }
        else { cp = c & 0x07; len = 4; }
        for (int k = 1; k < len && i + k < s.size(); ++k) cp = (cp << 6) | (static_cast<unsigned char>(s[i + k]) & 0x3f);
        out.push_back(cp);
        i += len;
    }
    return out;
}

// The voice was trained on espeak's American English, which keeps "caught" (AO) apart from "cot" (AA);
// CMUdict often lists the merged AA first ("bought B AA1 T", then "B AO1 T"). The voice gets the AO one,
// else it says "Bart".
const std::vector<std::string>& voice_phones(const G2P& g2p, const Word& w) {
    const auto* all = g2p.lookup(w.key);
    if (!all || all->size() < 2) return w.phones;
    const auto& first = (*all)[0];
    for (size_t v = 1; v < all->size(); ++v) {
        const auto& alt = (*all)[v];
        if (alt.size() != first.size()) continue;
        bool merger = false, other = false;
        for (size_t i = 0; i < alt.size(); ++i) {
            if (alt[i] == first[i]) continue;
            if (first[i].rfind("AA", 0) == 0 && alt[i].rfind("AO", 0) == 0 && first[i].substr(2) == alt[i].substr(2)) merger = true;
            else other = true;
        }
        if (merger && !other) return alt;
    }
    return w.phones;
}

}  // namespace

std::string speakable(const std::string& in) {
    // Symbols first, on the raw text (UTF-8 sequences included).
    std::string text;
    for (size_t i = 0; i < in.size();) {
        if (in.compare(i, 3, "°C") == 0 || in.compare(i, 3, "°F") == 0) {
            text += in[i + 2] == 'C' ? " degrees Celsius" : " degrees Fahrenheit";
            i += 3;
        } else if (in.compare(i, 2, "°") == 0) {
            text += " degrees";
            i += 2;
        } else if (in.compare(i, 3, "€") == 0) {
            text += "EUR ";
            i += 3;
        } else if (in.compare(i, 2, "£") == 0) {
            text += "GBP ";
            i += 2;
        } else if (in.compare(i, 3, "—") == 0 || in.compare(i, 3, "–") == 0) {
            text += ", ";
            i += 3;
        } else if (in.compare(i, 3, "’") == 0) {
            text += '\'';
            i += 3;
        } else if (in[i] == '*' || in[i] == '#' || in[i] == '`' || in[i] == '_' || in[i] == '\n' || in[i] == '\r') {
            text += in[i] == '\n' ? ". " : " ";  // markdown and line breaks are not read
            ++i;
        } else if (in[i] == '&') {
            text += " and ";
            ++i;
        } else {
            text += in[i++];
        }
    }

    std::string out;
    size_t i = 0;
    auto prev_alpha = [&](size_t k) { return k > 0 && is_alpha(text[k - 1]); };
    while (i < text.size()) {
        char c = text[i];
        // Currency before a number: "$5" -> "five dollars".
        std::string currency;
        size_t j = i;
        if (c == '$') { currency = "dollars"; ++j; }
        else if (text.compare(i, 4, "EUR ") == 0 && j + 4 < text.size() && is_digit(text[i + 4])) { currency = "euros"; j += 4; }
        else if (text.compare(i, 4, "GBP ") == 0 && j + 4 < text.size() && is_digit(text[i + 4])) { currency = "pounds"; j += 4; }
        bool minus = false;
        if (currency.empty() && c == '-' && j + 1 < text.size() && is_digit(text[j + 1]) && (i == 0 || text[i - 1] == ' ' || text[i - 1] == '(')) {
            minus = true;
            ++j;
        }
        if (j >= text.size() || !is_digit(text[j]) || prev_alpha(i)) {
            if (!currency.empty() && c != '$') { out += text.substr(i, 4); i += 4; continue; }
            out += c;
            ++i;
            continue;
        }
        // The number: digits with thousands commas, then maybe ":mm" or ".frac".
        auto digit_at = [&](size_t p) { return p < text.size() && is_digit(text[p]); };
        // A thousands comma: three digits after it, and no fourth.
        auto thousands = [&](size_t p) { return text[p] == ',' && digit_at(p + 1) && digit_at(p + 2) && digit_at(p + 3) && !digit_at(p + 4); };
        std::string digits;
        size_t k = j;
        while (k < text.size() && (is_digit(text[k]) || (!digits.empty() && thousands(k)))) {
            if (text[k] != ',') digits += text[k];
            ++k;
        }
        if (digits.size() > 12) {  // a long code: digit by digit
            for (char d : digits) out += std::string(kOnes[d - '0']) + " ";
            i = k;
            continue;
        }
        long long n = std::stoll(digits);
        std::string said;
        // Time: "6:30", "18:00", optionally followed by am/pm.
        if (k < text.size() && text[k] == ':' && digit_at(k + 1) && digit_at(k + 2) && !digit_at(k + 3) && n < 24) {
            int mm = (text[k + 1] - '0') * 10 + (text[k + 2] - '0');
            k += 3;
            // am/pm after it?
            size_t a = k;
            while (a < text.size() && text[a] == ' ') ++a;
            std::string mer;
            std::string rest = lower(text.substr(a, 4));
            if (rest.rfind("a.m.", 0) == 0 || rest.rfind("p.m.", 0) == 0) { mer = rest.substr(0, 1); a += 4; }
            else if ((rest.rfind("am", 0) == 0 || rest.rfind("pm", 0) == 0) && (a + 2 >= text.size() || !is_alpha(text[a + 2]))) { mer = rest.substr(0, 1); a += 2; }
            long long h = n;
            if (mer.empty() && h > 12) { h -= 12; mer = "p"; }
            if (mer.empty() && h == 0) { h = 12; mer = "a"; }
            said = say_number(h);
            if (mm == 0) said += mer.empty() ? " o'clock" : "";
            else if (mm < 10) said += " oh " + say_number(mm);
            else said += " " + say_number(mm);
            if (!mer.empty()) {
                said += mer == "a" ? " a m" : " p m";
                k = a;
            }
        } else if (k + 1 < text.size() && text[k] == '.' && is_digit(text[k + 1])) {
            // Decimal: "21.5" -> "twenty one point five".
            said = say_number(n) + " point";
            ++k;
            while (k < text.size() && is_digit(text[k])) said += std::string(" ") + kOnes[text[k++] - '0'];
        } else {
            std::string suffix = lower(text.substr(k, 2));
            bool ord = (suffix == "st" || suffix == "nd" || suffix == "rd" || suffix == "th") && (k + 2 >= text.size() || !is_alpha(text[k + 2]));
            if (ord) {
                said = ordinal(n);
                k += 2;
            } else if (digits.size() == 4 && n >= 1100 && n <= 2099 && currency.empty() && !minus &&
                       (k >= text.size() || text[k] != '%')) {
                said = say_year(n);
            } else {
                said = say_number(n);
            }
            // am/pm after a bare hour: "at 6 pm".
            size_t a = k;
            while (a < text.size() && text[a] == ' ') ++a;
            std::string rest = lower(text.substr(a, 4));
            if (!ord && n >= 1 && n <= 12) {
                if (rest.rfind("a.m.", 0) == 0 || rest.rfind("p.m.", 0) == 0) { said += rest[0] == 'a' ? " a m" : " p m"; k = a + 4; }
                else if ((rest.rfind("am", 0) == 0 || rest.rfind("pm", 0) == 0) && (a + 2 >= text.size() || !is_alpha(text[a + 2]))) {
                    said += rest[0] == 'a' ? " a m" : " p m";
                    k = a + 2;
                }
            }
        }
        if (minus) said = "minus " + said;
        if (k < text.size() && text[k] == '%') {
            said += " percent";
            ++k;
        }
        if (!currency.empty()) said += " " + (n == 1 && currency.back() == 's' ? currency.substr(0, currency.size() - 1) : currency);
        if (!out.empty() && is_alpha(out.back())) out += ' ';
        out += said;
        if (k < text.size() && is_alpha(text[k])) out += ' ';  // "5km" -> "five km"
        i = k;
    }
    // Collapse runs of spaces.
    std::string clean;
    for (char ch : out) {
        if (ch == ' ' && (clean.empty() || clean.back() == ' ')) continue;
        clean += ch;
    }
    while (!clean.empty() && clean.back() == ' ') clean.pop_back();
    return clean;
}

std::string arpabet_to_ipa(const std::vector<std::string>& phones) {
    static const std::map<std::string, std::string> base = {
        {"AA", "ɑː"}, {"AE", "æ"}, {"AO", "ɔː"}, {"AW", "aʊ"}, {"AY", "aɪ"}, {"B", "b"}, {"CH", "tʃ"}, {"D", "d"},
        {"DH", "ð"}, {"EH", "ɛ"}, {"EY", "eɪ"}, {"F", "f"}, {"G", "ɡ"}, {"HH", "h"}, {"IH", "ɪ"}, {"IY", "iː"},
        {"JH", "dʒ"}, {"K", "k"}, {"L", "l"}, {"M", "m"}, {"N", "n"}, {"NG", "ŋ"}, {"OW", "oʊ"}, {"OY", "ɔɪ"},
        {"P", "p"}, {"R", "ɹ"}, {"S", "s"}, {"SH", "ʃ"}, {"T", "t"}, {"TH", "θ"}, {"UH", "ʊ"}, {"UW", "uː"},
        {"V", "v"}, {"W", "w"}, {"Y", "j"}, {"Z", "z"}, {"ZH", "ʒ"},
    };
    std::string out;
    for (const auto& p : phones) {
        std::string name;
        char stress = 0;
        for (char c : p) {
            if (is_digit(c)) stress = c;
            else name += c;
        }
        std::string ipa;
        if (name == "AH") ipa = (stress == '1' || stress == '2') ? "ʌ" : "ə";
        else if (name == "ER") ipa = (stress == '1' || stress == '2') ? "ɜː" : "ɚ";
        else {
            auto it = base.find(name);
            if (it == base.end()) continue;
            ipa = it->second;
        }
        if (stress == '1') out += "ˈ";
        else if (stress == '2') out += "ˌ";
        out += ipa;
    }
    return out;
}

Tts::Tts(const std::filesystem::path& voice, const std::filesystem::path& config, const std::filesystem::path& cmudict, int threads)
    : session_(ort_session(voice, threads)), g2p_(G2P::load(cmudict.string())) {
    std::ifstream f(config, std::ios::binary);
    if (!f) throw std::runtime_error("cannot read " + config.string());
    nlohmann::json c = nlohmann::json::parse(f);
    rate_ = c["audio"]["sample_rate"].get<int>();
    speakers_ = c.value("num_speakers", 1);
    noise_ = c["inference"].value("noise_scale", noise_);
    length_ = c["inference"].value("length_scale", length_);
    noise_w_ = c["inference"].value("noise_w", noise_w_);
    for (auto it = c["phoneme_id_map"].begin(); it != c["phoneme_id_map"].end(); ++it) {
        auto cps = codepoints(it.key());
        if (cps.size() == 1) id_of_[cps[0]] = it.value()[0].get<int64_t>();
    }
}

std::string Tts::ipa(const std::string& text) const {
    // "[[...]]" is IPA already (the wake word training uses it for pronunciation variants).
    size_t open = text.find("[[");
    if (open != std::string::npos) {
        size_t close = text.find("]]", open + 2);
        if (close != std::string::npos) {
            std::string before = ipa(text.substr(0, open));
            std::string after = ipa(text.substr(close + 2));
            std::string out = before;
            if (!out.empty() && out.back() != ' ') out += ' ';
            out += text.substr(open + 2, close - open - 2);
            if (!after.empty() && after.front() != ',' && after.front() != '.' && after.front() != '?' && after.front() != '!') out += ' ';
            return out + after;
        }
    }
    std::string out;
    // Words keep the text's punctuation after them: Piper uses it for pauses and intonation.
    auto words = g2p_.words(text);
    size_t wi = 0, i = 0;
    while (i < text.size()) {
        char c = text[i];
        if (std::isalnum(static_cast<unsigned char>(c))) {
            size_t j = i;
            while (j < text.size() && (std::isalnum(static_cast<unsigned char>(text[j])) || text[j] == '\'')) ++j;
            if (wi < words.size()) {
                if (!out.empty() && out.back() != ' ') out += ' ';
                out += arpabet_to_ipa(voice_phones(g2p_, words[wi]));
                ++wi;
            }
            i = j;
        } else {
            if (c == ',' || c == '.' || c == '?' || c == '!' || c == ';' || c == ':') out += c;
            ++i;
        }
    }
    return out;
}

std::vector<int64_t> Tts::ids(const std::string& ipa) const {
    // Piper's layout: ^ _ p1 _ p2 _ ... $
    std::vector<int64_t> out = {id_of_.at('^'), id_of_.at('_')};
    for (uint32_t cp : codepoints(ipa)) {
        auto it = id_of_.find(cp);
        if (it == id_of_.end()) continue;
        out.push_back(it->second);
        out.push_back(id_of_.at('_'));
    }
    out.push_back(id_of_.at('$'));
    return out;
}

std::vector<int16_t> Tts::speak(const std::string& text, int speaker, float speed, float variation) {
    std::vector<int16_t> pcm;
    if (speaker < 0 || speaker >= speakers_) throw std::runtime_error("no such speaker");
    std::string said = speakable(text);
    // Sentence by sentence: shorter inputs synthesize faster and more steadily.
    std::string sentence;
    auto flush = [&] {
        std::string s = ipa(sentence);
        sentence.clear();
        if (s.find_first_not_of(" ,.?!;:") == std::string::npos) return;
        if (!pcm.empty()) pcm.insert(pcm.end(), static_cast<size_t>(rate_ / 6), 0);  // a pause between sentences
        auto in = ids(s);
        auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::array<int64_t, 2> shape{1, static_cast<int64_t>(in.size())};
        std::array<int64_t, 1> lens{static_cast<int64_t>(in.size())};
        std::array<int64_t, 1> one{1};
        std::array<float, 3> scales{noise_ * variation, length_ / std::max(0.25f, speed), noise_w_ * variation};
        std::array<int64_t, 1> sshape{3};
        int64_t sid = speaker;
        std::vector<Ort::Value> inputs;
        inputs.push_back(Ort::Value::CreateTensor<int64_t>(mem, in.data(), in.size(), shape.data(), 2));
        inputs.push_back(Ort::Value::CreateTensor<int64_t>(mem, lens.data(), 1, one.data(), 1));
        inputs.push_back(Ort::Value::CreateTensor<float>(mem, scales.data(), 3, sshape.data(), 1));
        std::vector<const char*> in_names = {"input", "input_lengths", "scales"};
        if (speakers_ > 1) {
            inputs.push_back(Ort::Value::CreateTensor<int64_t>(mem, &sid, 1, one.data(), 1));
            in_names.push_back("sid");
        }
        const char* out_names[] = {"output"};
        auto out = session_->Run(Ort::RunOptions{nullptr}, in_names.data(), inputs.data(), inputs.size(), out_names, 1);
        size_t n = out[0].GetTensorTypeAndShapeInfo().GetElementCount();
        const float* a = out[0].GetTensorData<float>();
        float peak = 1e-3f;
        for (size_t k = 0; k < n; ++k) peak = std::max(peak, std::fabs(a[k]));
        float gain = std::min(1.f / peak, 4.f) * 0.9f;
        for (size_t k = 0; k < n; ++k) pcm.push_back(static_cast<int16_t>(std::lround(std::clamp(a[k] * gain, -1.f, 1.f) * 32767)));
    };
    for (size_t i = 0; i < said.size(); ++i) {
        sentence += said[i];
        bool end = said[i] == '?' || said[i] == '!' || (said[i] == '.' && (i + 1 == said.size() || said[i + 1] == ' '));
        if (end) flush();
    }
    flush();
    return pcm;
}

}  // namespace courier
