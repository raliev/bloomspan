#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <vector>
#include <string>
#include <cctype>

inline std::vector<std::string> tokenize(const std::string& text) {
    std::vector<std::string> tokens;
    std::string current;
    current.reserve(32);
    for (unsigned char c : text) {
        if (std::isalnum(c)) current += std::tolower(c);
        else if (!current.empty()) { tokens.push_back(current); current.clear(); }
    }
    if (!current.empty()) tokens.push_back(current);
    return tokens;
}

#endif // TOKENIZER_H
