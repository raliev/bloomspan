#ifndef CORPUS_MINER_H
#define CORPUS_MINER_H

#include <string>
#include <vector>
#include <unordered_map>
#include <map>
#include <memory>
#include <filesystem>
#include <chrono>
#include <iostream>

namespace fs = std::filesystem;

// Forward declaration for the internal Phrase structure
struct Phrase;

enum MiningMode { MODE_ALL, MODE_CLOSED, MODE_MAXIMAL };

class CorpusMiner {
public:
    CorpusMiner() = default;
    ~CorpusMiner() = default;

    // --- Configuration ---
    void set_limits(int threads, int mem_limit_mb, int cache_size, bool in_mem, bool preload) {
        max_threads = threads;
        memory_limit_mb = mem_limit_mb;
        max_cache_size = static_cast<size_t>(cache_size);
        in_memory_only = in_mem;
        preload_cache = preload;
    }

    void set_mask(const std::string& mask) {
        file_mask = mask;
    }

    // --- Data Loading ---
    void load_directory(const std::string& path, double sampling);
    void load_csv(const std::string& path, char delimiter, double sampling);

    // --- Mining ---
    void mine(int min_docs, int ngrams, const std::string& out_path);

private:
    // --- Internal Logic ---
    void save_to_csv(const std::vector<Phrase>& res, const std::string& out_p);
    void load_all_from_bin();

    // Timer helpers
    inline std::chrono::time_point<std::chrono::high_resolution_clock> start_timer() {
        return std::chrono::high_resolution_clock::now();
    }

    inline void stop_timer(const std::string& label, std::chrono::time_point<std::chrono::high_resolution_clock> start) {
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        std::cout << "[TIMER] " << label << ": " << diff.count() << "s" << std::endl;
    }

    // --- Data Members ---
    std::unordered_map<std::string, uint32_t> word_to_id;
    std::vector<std::string> id_to_word;

    std::vector<std::vector<uint32_t>> docs;
    std::vector<std::string> file_paths;
    std::vector<uint32_t> word_df;
    std::vector<size_t> doc_lengths;
    std::vector<std::streampos> doc_offsets;

    std::map<size_t, std::vector<uint32_t>> doc_cache;

    int max_threads = 0;
    int memory_limit_mb = 0;
    size_t max_cache_size = 1000;
    bool in_memory_only = false;
    bool preload_cache = false;
    std::string file_mask = "";
    std::string bin_corpus_path = "corpus.bin";
};

#endif // CORPUS_MINER_H