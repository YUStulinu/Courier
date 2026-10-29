#include "g2p.hpp"

#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace courier {

G2P G2P::load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot read " + path);
    G2P g;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == ';') continue;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        // "word phones..." or "word(2) phones...", possibly with "# comment".
        auto hash = line.find('#');
        if (hash != std::string::npos) line.resize(hash);
        std::istringstream in(line);
        std::string word;
        in >> word;
        auto paren = word.find('(');
        if (paren != std::string::npos) word.resize(paren);
        std::vector<std::string> phones;
        std::string p;
        while (in >> p) phones.push_back(p);
        if (!word.empty() && !phones.empty()) g.dict_[word].push_back(std::move(phones));
    }
    if (g.dict_.empty()) throw std::runtime_error(path + " has no entries");
    return g;
}

const std::vector<std::vector<std::string>>* G2P::lookup(const std::string& key) const {
    auto it = dict_.find(key);
    return it == dict_.end() ? nullptr : &it->second;
}

std::vector<Word> G2P::words(const std::string& text) const {
    std::vector<Word> out;
    size_t i = 0;
    while (i < text.size()) {
        // A word: letters, digits and inner apostrophes ("don't", "it's").
        while (i < text.size() && !std::isalnum(static_cast<unsigned char>(text[i]))) ++i;
        size_t start = i;
        while (i < text.size() && (std::isalnum(static_cast<unsigned char>(text[i])) ||
                                   (text[i] == '\'' && i + 1 < text.size() && std::isalpha(static_cast<unsigned char>(text[i + 1])))))
            ++i;
        if (i == start) continue;
        Word w;
        w.text = text.substr(start, i - start);
        for (char c : w.text) w.key += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        if (const auto* prons = lookup(w.key)) {
            w.phones = prons->front();
        } else {
            w.phones = guess(w.key);
            w.guessed = true;
        }
        if (!w.phones.empty()) out.push_back(std::move(w));
    }
    return out;
}

std::vector<std::string> G2P::guess(const std::string& key) {
    // Longest match first; vowels get a primary stress on the first one.
    static const std::vector<std::pair<std::string, std::vector<std::string>>> rules = {
        {"tion", {"SH", "AH0", "N"}}, {"sion", {"ZH", "AH0", "N"}}, {"ough", {"AO1"}}, {"igh", {"AY1"}},
        {"ing", {"IH0", "NG"}}, {"sch", {"S", "K"}}, {"tch", {"CH"}}, {"ph", {"F"}}, {"th", {"TH"}},
        {"sh", {"SH"}}, {"ch", {"CH"}}, {"ck", {"K"}}, {"ng", {"NG"}}, {"qu", {"K", "W"}}, {"wh", {"W"}},
        {"ee", {"IY1"}}, {"ea", {"IY1"}}, {"oo", {"UW1"}}, {"ou", {"AW1"}}, {"ow", {"OW1"}}, {"ai", {"EY1"}},
        {"ay", {"EY1"}}, {"oi", {"OY1"}}, {"oy", {"OY1"}}, {"au", {"AO1"}}, {"aw", {"AO1"}}, {"er", {"ER0"}},
        {"ar", {"AA1", "R"}}, {"or", {"AO1", "R"}}, {"a", {"AE1"}}, {"e", {"EH1"}}, {"i", {"IH1"}}, {"o", {"AA1"}},
        {"u", {"AH1"}}, {"y", {"IY0"}}, {"b", {"B"}}, {"c", {"K"}}, {"d", {"D"}}, {"f", {"F"}}, {"g", {"G"}},
        {"h", {"HH"}}, {"j", {"JH"}}, {"k", {"K"}}, {"l", {"L"}}, {"m", {"M"}}, {"n", {"N"}}, {"p", {"P"}},
        {"r", {"R"}}, {"s", {"S"}}, {"t", {"T"}}, {"v", {"V"}}, {"w", {"W"}}, {"x", {"K", "S"}}, {"z", {"Z"}},
    };
    std::vector<std::string> out;
    bool stressed = false;
    size_t i = 0;
    // A final silent e ("make") is dropped.
    std::string w = key;
    if (w.size() > 3 && w.back() == 'e') w.pop_back();
    while (i < w.size()) {
        bool matched = false;
        for (const auto& [letters, phones] : rules) {
            if (w.compare(i, letters.size(), letters) == 0) {
                for (std::string p : phones) {
                    if (std::isdigit(static_cast<unsigned char>(p.back()))) {
                        if (stressed && p.back() == '1') p.back() = '0';
                        if (p.back() == '1') stressed = true;
                    }
                    if (out.empty() || out.back() != p) out.push_back(p);
                }
                i += letters.size();
                matched = true;
                break;
            }
        }
        if (!matched) ++i;  // digits and letters without a rule
    }
    return out;
}

}  // namespace courier
