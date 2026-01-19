#include "corpus_miner.h"
#include "tokenizer.h"
#include "signal_handler.h"
#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <cstring>
#include <random>
#include <omp.h>

struct Phrase {
    std::vector<uint32_t> tokens;
    uint32_t support;
    struct Occurrence {
        uint32_t doc_id;
    };
    std::vector<Occurrence> occs;
};

struct Projection {
    uint32_t doc_id;
    uint32_t pos;
    uint32_t origin;
};

class PrefixSpanEngine {
public:
    uint32_t min_docs;
    uint32_t min_length;
    bool contiguous_mode;
    const std::vector<std::vector<uint32_t>>& docs;
    std::vector<Phrase>& results;
    uint32_t item_max;

    PrefixSpanEngine(uint32_t md, uint32_t ml, bool contig, const std::vector<std::vector<uint32_t>>& d, std::vector<Phrase>& r, uint32_t max_id)
        : min_docs(md), min_length(ml), contiguous_mode(contig), docs(d), results(r), item_max(max_id) {}

    void run(MiningMode mode) {
        std::vector<Projection> initial_db;
        // Use a set to track how many unique documents are in the initial database
        std::unordered_set<uint32_t> initial_docs;

        for (uint32_t i = 0; i < (uint32_t)docs.size(); ++i) {
            if (!docs[i].empty()) {
                initial_docs.insert(i);
                for (uint32_t j = 0; j < (uint32_t)docs[i].size(); ++j) {
                    // Initial Projection: (doc_id, current_pos, origin_pos)
                    initial_db.push_back({i, j, j});
                }
            }
        }

        std::vector<uint32_t> current_prefix;

        // The initial current_support is the total count of non-empty unique documents
        uint32_t initial_support = static_cast<uint32_t>(initial_docs.size());

        // Start mining with the updated signature
        mine_recursive(initial_db, current_prefix, initial_support, mode);
    }

private:
    /**
     * The method counts the frequency of elements in the current projected database.
     */
    void occ_delivery(const std::vector<Projection>& db, std::unordered_map<uint32_t, std::vector<uint32_t>>& item_supports) {
        // The benchmark uses an sc array to mark visited documents within a single symbol.
        // We use an auxiliary array to efficiently count unique document support.
        static std::vector<uint32_t> last_doc_seen(item_max + 1, 0xFFFFFFFF);
        static uint32_t current_iteration = 0;
        current_iteration++;

        for (const auto& proj : db) {
            const auto& doc = docs[proj.doc_id];

            // The benchmark here tests constraints on the gap.
            // For "Contiguous Phrases", we only check the current character proj.pos.
            if (proj.pos < doc.size()) {
                uint32_t token = doc[proj.pos];

                // Emulation of TT->sc logic from trsact.c for doc_id uniqueness
                // Instead of a hash set, we use an iteration marker for speed.
                if (item_supports[token].empty() || item_supports[token].back() != proj.doc_id) {
                    item_supports[token].push_back(proj.doc_id);
                }
            }
        }
    }

    /**
     * Main recursive mining function adapted from LCM-seq logic.
     * * @param db The current projected database (occurrences of the current prefix).
     * @param prefix The current sequence of tokens being evaluated.
     * @param current_support The number of unique documents containing the current prefix.
     * @param mode Filtering mode: ALL, CLOSED, or MAXIMAL.
     */
    void mine_recursive(const std::vector<Projection>& db,
                        std::vector<uint32_t>& prefix,
                        uint32_t current_support,
                        MiningMode mode) {
        if (g_stop_requested) return;

        // 1. Frequency Counting (Equivalent to LCMseq_occ_delivery)
        // Find all possible tokens that can extend the current prefix.
        std::unordered_map<uint32_t, std::vector<uint32_t>> item_supports;
        occ_delivery(db, item_supports);

        // 2. Analyze Extensions for Maximal/Closed Logic
        // We check if the current prefix has any "good" extensions before deciding to save it.
        bool has_frequent_extension = false;
        bool has_extension_with_same_support = false;

        for (auto const& [token, doc_ids] : item_supports) {
            uint32_t support = static_cast<uint32_t>(doc_ids.size());
            if (support >= min_docs) {
                has_frequent_extension = true;
                if (support == current_support) {
                    has_extension_with_same_support = true;
                }
            }
        }

        // 3. Output Decision
        // In Maximal/Closed mining, we only output a pattern if it cannot be
        // "improved" by adding more tokens according to the rules.
        bool should_output = false;
        if (prefix.size() >= min_length) {
            if (mode == MODE_ALL) {
                should_output = true;
            } else if (mode == MODE_MAXIMAL) {
                // Output only if NO extension is frequent.
                if (!has_frequent_extension) should_output = true;
            } else if (mode == MODE_CLOSED) {
                // Output only if NO extension has the same support.
                if (!has_extension_with_same_support) should_output = true;
            }
        }

        if (should_output) {
            Phrase p;
            p.tokens = prefix;
            p.support = current_support;

            // Collect unique document IDs for the current prefix
            std::unordered_set<uint32_t> unique_docs;
            for (const auto& proj : db) {
                unique_docs.insert(proj.doc_id);
            }
            for (uint32_t doc_id : unique_docs) {
                p.occs.push_back({doc_id});
            }

            #pragma omp critical
            results.push_back(std::move(p));
        }

        // 4. Recursive Expansion
        // Visit all frequent extensions (depth-first search)
        for (auto const& [token, doc_ids] : item_supports) {
            uint32_t support = static_cast<uint32_t>(doc_ids.size());
            if (support >= min_docs) {
                prefix.push_back(token);

                // Construct the projected database for the next level
                std::vector<Projection> next_db;
                next_db.reserve(db.size());
                for (const auto& proj : db) {
                    const auto& doc = docs[proj.doc_id];
                    // Contiguous check: is the required token immediately next?
                    if (proj.pos < doc.size() && doc[proj.pos] == token) {
                        if (proj.pos + 1 < doc.size()) {
                            next_db.push_back({proj.doc_id, proj.pos + 1, proj.origin});
                        }
                    }
                }

                if (!next_db.empty()) {
                    mine_recursive(next_db, prefix, support, mode);
                }
                prefix.pop_back();
            }
        }
    }
};

void CorpusMiner::mine(int min_docs, int ngrams, const std::string& out_path) {
    if (file_paths.empty()) return;

    bool use_contiguous = true;

    std::cout << "[LOG] Starting Mining (Standard LCM-seq Logic)..." << std::endl;
    auto start = start_timer();
    std::vector<Phrase> found_phrases;

    if (!in_memory_only) load_all_from_bin();

    PrefixSpanEngine engine(
        static_cast<uint32_t>(min_docs),
        static_cast<uint32_t>(ngrams),
        use_contiguous,
        docs,
        found_phrases,
        static_cast<uint32_t>(id_to_word.size())
    );

    engine.run(MODE_CLOSED);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "[LOG] Mining completed in " << diff.count() << "s. Found " << found_phrases.size() << " patterns." << std::endl;

    save_to_csv(found_phrases, out_path);
}

void CorpusMiner::load_csv(const std::string& path, char delimiter, double sampling) {
    auto total_start = start_timer();
    std::cout << "[LOG] Loading CSV: " << path << " (Delimiter: '" << delimiter << "')" << std::endl;
    std::ifstream file(path, std::ios::binary);
    if (!file) return;

    std::vector<std::string> rows;
    std::string currentRow, currentField;
    bool inQuotes = false;
    char c;

    while (file.get(c)) {
        if (inQuotes) {
            if (c == '"') {
                if (file.peek() == '"') { currentField += '"'; file.get(); }
                else inQuotes = false;
            } else currentField += c;
        } else {
            if (c == '"') inQuotes = true;
            else if (c == delimiter) {
                if (!currentRow.empty()) currentRow += " ";
                currentRow += currentField;
                currentField.clear();
            } else if (c == '\n' || c == '\r') {
                if (!currentRow.empty() || !currentField.empty()) {
                    if (!currentRow.empty()) currentRow += " ";
                    currentRow += currentField;
                    rows.push_back(std::move(currentRow));
                    currentRow.clear(); currentField.clear();
                }
                if (c == '\r' && file.peek() == '\n') file.get();
            } else currentField += c;
        }
    }
    if (!currentRow.empty() || !currentField.empty()) {
        if (!currentRow.empty()) currentRow += " ";
        currentRow += currentField;
        rows.push_back(std::move(currentRow));
    }

    if (sampling < 1.0) {
        std::random_device rd; std::mt19937 g(rd());
        std::shuffle(rows.begin(), rows.end(), g);
        rows.resize(static_cast<size_t>(rows.size() * sampling));
    }

    size_t n = rows.size();
    std::vector<std::vector<std::string>> raw_docs(n);
    if (max_threads > 0) omp_set_num_threads(max_threads);
    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        raw_docs[i] = tokenize(rows[i]);
        rows[i].clear();
    }
    rows.clear();

    docs.clear();
    if (in_memory_only) docs.reserve(n);
    file_paths.reserve(n);
    std::vector<uint32_t> word_last_doc_id;
    std::unique_ptr<std::ofstream> bin_out;
    if (!in_memory_only) bin_out = std::make_unique<std::ofstream>(bin_corpus_path, std::ios::binary);

    for (size_t i = 0; i < n; ++i) {
        file_paths.push_back("row_" + std::to_string(i));
        std::vector<uint32_t> encoded;
        encoded.reserve(raw_docs[i].size());
        for (const auto& w : raw_docs[i]) {
            uint32_t w_id;
            auto it = word_to_id.find(w);
            if (it == word_to_id.end()) {
                w_id = id_to_word.size(); word_to_id[w] = w_id; id_to_word.push_back(w);
                word_df.push_back(0); word_last_doc_id.push_back(0);
            } else w_id = it->second;
            encoded.push_back(w_id);
            if (word_last_doc_id[w_id] != (uint32_t)i + 1) {
                word_df[w_id]++; word_last_doc_id[w_id] = (uint32_t)i + 1;
            }
        }
        doc_lengths.push_back(encoded.size());
        if (in_memory_only) docs.push_back(std::move(encoded));
        else {
            doc_offsets.push_back(bin_out->tellp());
            bin_out->write((char*)encoded.data(), encoded.size() * sizeof(uint32_t));
            encoded.clear();
        }
        raw_docs[i].clear();
    }
    stop_timer("CSV Loading & Encoding", total_start);
}

void CorpusMiner::load_directory(const std::string& path, double sampling) {
    auto total_start = start_timer();

    std::cout << "[LOG] Scanning directory: " << path << (file_mask.empty() ? " (All files)" : " (Mask: " + file_mask + ")") << std::endl;
        std::vector<fs::path> paths;

        for (const auto& entry : fs::recursive_directory_iterator(path)) {
            if (!fs::is_regular_file(entry)) continue;

            bool match = false;
            if (file_mask.empty() || file_mask == "*") {
                match = true;
            } else if (file_mask.size() >= 2 && file_mask.substr(0, 2) == "*.") {
                // Wildcard extension match (e.g., "*.txt" -> ".txt")
                std::string target_ext = file_mask.substr(1);
                if (entry.path().extension() == target_ext) match = true;
            } else {
                // Exact filename match
                if (entry.path().filename() == file_mask) match = true;
            }

            if (match) paths.push_back(entry.path());
        }

    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(paths.begin(), paths.end(), g);

    size_t total_files = paths.size();
    size_t n = static_cast<size_t>(total_files * sampling);
    if (n > total_files) n = total_files;
    paths.resize(n);

    std::cout << "[LOG] Found " << total_files << " .txt files. Processing " << n
              << " files (sampling rate: " << (sampling * 100) << "%)" << std::endl;
    std::vector<std::vector<std::string>> raw_docs(n);
    if (max_threads > 0) omp_set_num_threads(max_threads);
    std::cout << "[LOG] Phase I: Parallel tokenization..." << std::endl;
    auto p1_start = start_timer();

    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        std::ifstream file(paths[i], std::ios::binary);
        if (!file) continue;

        unsigned char bom[2] = {0, 0};
        file.read((char*)bom, 2);

        if (bom[0] == 0xFF && bom[1] == 0xFE) {
            // UTF-16 Little Endian
            std::vector<char> buffer((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
            std::u16string u16_content;
            u16_content.resize(buffer.size() / 2);
            std::memcpy(u16_content.data(), buffer.data(), u16_content.size() * 2);
            raw_docs[i] = tokenize_utf16(u16_content);
        }
        else if (bom[0] == 0xFE && bom[1] == 0xFF) {
            // UTF-16 Big Endian (Manual byte swap)
            std::vector<char> buffer((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
            std::u16string u16_content;
            u16_content.reserve(buffer.size() / 2);
            for (size_t j = 0; j + 1 < buffer.size(); j += 2) {
                u16_content.push_back((char16_t)(((unsigned char)buffer[j] << 8) | (unsigned char)buffer[j+1]));
            }
            raw_docs[i] = tokenize_utf16(u16_content);
        }
        else {
            // Standard UTF-8 / ASCII logic
            file.seekg(0, std::ios::beg);
            std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
            raw_docs[i] = tokenize(content);
        }
    }
    stop_timer("Tokenization", p1_start);

    std::cout << "[LOG] Phase II: Building dictionary, encoding ID, and counting DF..." << std::endl;
    auto p2_start = start_timer();
    docs.clear();
    if (in_memory_only) docs.reserve(n);
    file_paths.reserve(n);

    std::vector<uint32_t> word_last_doc_id;
    word_df.clear();

    // Only open bin file if NOT in-memory mode
    std::unique_ptr<std::ofstream> bin_out;
    if (!in_memory_only) {
        bin_out = std::make_unique<std::ofstream>(bin_corpus_path, std::ios::binary);
    }

    for (size_t i = 0; i < n; ++i) {
            file_paths.push_back(paths[i].string());
            std::vector<uint32_t> encoded;
            encoded.reserve(raw_docs[i].size());

            for (const auto& w : raw_docs[i]) {
                uint32_t w_id;
                auto it = word_to_id.find(w);
                if (it == word_to_id.end()) {
                    w_id = id_to_word.size();
                    word_to_id[w] = w_id;
                    id_to_word.push_back(w);
                    word_df.push_back(0);
                    word_last_doc_id.push_back(0);
                } else {
                    w_id = it->second;
                }
                encoded.push_back(w_id);

                if (word_last_doc_id[w_id] != (uint32_t)i + 1) {
                    word_df[w_id]++;
                    word_last_doc_id[w_id] = (uint32_t)i + 1;
                }
            }

            doc_lengths.push_back(encoded.size());

            if (in_memory_only) {
                docs.push_back(std::move(encoded));
            } else {
                doc_offsets.push_back(bin_out->tellp());
                bin_out->write((char*)encoded.data(), encoded.size() * sizeof(uint32_t));

                // If preload is requested, keep in cache while building
                if (preload_cache && doc_cache.size() < max_cache_size) {
                    doc_cache[i] = encoded; // Copy before clearing
                }
                encoded.clear();
            }
            raw_docs[i].clear();
        }
    word_last_doc_id.clear();
    word_last_doc_id.shrink_to_fit();
    stop_timer("Dictionary, Encoding & DF counting", p2_start);
    stop_timer("Total Loading", total_start);
}

void CorpusMiner::save_to_csv(const std::vector<Phrase>& res, const std::string& out_p) {
    std::ofstream f(out_p);
    if (!f.is_open()) return;
    f << "phrase,freq,length,example_files\n";
    std::vector<Phrase> sorted_res = res;
    std::sort(sorted_res.begin(), sorted_res.end(), [](const Phrase& a, const Phrase& b) {
        if (a.support != b.support) return a.support > b.support;
        return a.tokens.size() > b.tokens.size();
    });

    for (const auto& p : sorted_res) {
        f << "\"";
        for (size_t i = 0; i < p.tokens.size(); ++i) {
            if (p.tokens[i] < id_to_word.size()) f << id_to_word[p.tokens[i]] << (i == p.tokens.size() - 1 ? "" : " ");
        }
        f << "\"," << p.support << "," << p.tokens.size() << ",\"";

        std::unordered_set<uint32_t> unique_docs;
        for (auto& occ : p.occs) unique_docs.insert(occ.doc_id);
        size_t count = 0;
        for (uint32_t d_id : unique_docs) {
            if (d_id < file_paths.size()) {
                f << file_paths[d_id];
                if (++count >= 2) { if (unique_docs.size() > 2) f << "..."; break; }
                if (count < unique_docs.size()) f << "|";
            }
        }
        f << "\"\n";
    }
}

void CorpusMiner::load_all_from_bin() {
    std::ifstream bin_in(bin_corpus_path, std::ios::binary);
    if (!bin_in) return;
    docs.clear(); docs.resize(doc_lengths.size());
    for(size_t i=0; i < doc_lengths.size(); ++i) {
        bin_in.seekg(doc_offsets[i]);
        docs[i].resize(doc_lengths[i]);
        bin_in.read((char*)docs[i].data(), doc_lengths[i] * sizeof(uint32_t));
    }
}