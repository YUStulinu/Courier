// From English text to the phonemes it should sound like (ARPAbet with stress, "W IY1"), by the CMU
// pronouncing dictionary (134,000 words). Words it does not have get a guess from spelling rules.
// (From Parrot, the same author's pronunciation tutor.)
#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace courier {

struct Word {
    std::string text;                 // as written ("Bear,")
    std::string key;                  // as looked up ("bear")
    std::vector<std::string> phones;  // ARPAbet with stress
    bool guessed = false;             // not in the dictionary
};

class G2P {
public:
    static G2P load(const std::string& cmudict_path);

    // Words of `text`, in order, with their phonemes (the dictionary's first pronunciation).
    std::vector<Word> words(const std::string& text) const;
    // Every pronunciation the dictionary has for a word (lowercase).
    const std::vector<std::vector<std::string>>* lookup(const std::string& key) const;
    size_t size() const { return dict_.size(); }

    // Spelling rules for a word the dictionary does not have (crude, but better than nothing).
    static std::vector<std::string> guess(const std::string& key);

private:
    std::unordered_map<std::string, std::vector<std::vector<std::string>>> dict_;
};

}  // namespace courier
