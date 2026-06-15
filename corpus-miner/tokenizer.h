#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <vector>
#include <string>
#include <cctype>
#include <codecvt>
#include <locale>

// Helper to convert UTF-16 to UTF-8
inline std::string utf16_to_utf8(const std::u16string& utf16) {
    std::string utf8;
    utf8.reserve(utf16.size()); // Initial guess

    for (size_t i = 0; i < utf16.size(); ++i) {
        uint32_t cp = utf16[i];

        // Handle Surrogate Pairs
        if (cp >= 0xD800 && cp <= 0xDBFF && i + 1 < utf16.size()) {
            uint32_t trail = utf16[i + 1];
            if (trail >= 0xDC00 && trail <= 0xDFFF) {
                cp = 0x10000 + ((cp - 0xD800) << 10) + (trail - 0xDC00);
                i++;
            }
        }

        // Encode to UTF-8
        if (cp <= 0x7F) {
            utf8 += static_cast<char>(cp);
        } else if (cp <= 0x7FF) {
            utf8 += static_cast<char>(0xC0 | (cp >> 6));
            utf8 += static_cast<char>(0x80 | (cp & 0x3F));
        } else if (cp <= 0xFFFF) {
            utf8 += static_cast<char>(0xE0 | (cp >> 12));
            utf8 += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            utf8 += static_cast<char>(0x80 | (cp & 0x3F));
        } else {
            utf8 += static_cast<char>(0xF0 | (cp >> 18));
            utf8 += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
            utf8 += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            utf8 += static_cast<char>(0x80 | (cp & 0x3F));
        }
    }
    return utf8;
}

// Existing UTF-8 tokenizer
inline std::vector<std::string> tokenize(const std::string& text) {
    std::vector<std::string> tokens;
    std::string current;
    current.reserve(32);

    for (size_t i = 0; i < text.length(); ++i) {
        unsigned char c = static_cast<unsigned char>(text[i]);
        if (std::isspace(c)) {
            if (!current.empty()) { tokens.push_back(current); current.clear(); }
        } else {
            // Lowercase A-Z like Java's toLowerCase for ASCII
            if (c >= 'A' && c <= 'Z') current += static_cast<char>(c + ('a' - 'A'));
            else current += static_cast<char>(c);
        }
    }
    if (!current.empty()) tokens.push_back(current);
    return tokens;
}

// New UTF-16 tokenizer
inline std::vector<std::string> tokenize_utf16(const std::u16string& text) {
    std::vector<std::string> tokens;
    std::u16string current;

    for (char16_t c : text) {
        bool is_space = false;
        if (c < 128) is_space = std::isspace(static_cast<unsigned char>(c));
        
        if (is_space) {
            if (!current.empty()) {
                tokens.push_back(utf16_to_utf8(current));
                current.clear();
            }
        } else {
            if (c >= u'A' && c <= u'Z') current += static_cast<char16_t>(c + (u'a' - u'A'));
            else current += c;
        }
    }
    if (!current.empty()) tokens.push_back(utf16_to_utf8(current));
    return tokens;
}

#endif