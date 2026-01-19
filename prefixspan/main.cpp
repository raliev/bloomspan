#include "corpus_miner.h"
#include "signal_handler.h"
#include <filesystem>
#include <iostream>
#include <csignal>

namespace fs = std::filesystem;

/**
 * Global signal handler to allow graceful interruption of long mining processes.
 * Implementation is assumed to be in signal_handler.cpp.
 */
extern void signal_handler(int signum);

int main(int argc, char** argv) {
    // Register signal handler for Ctrl+C
    std::signal(SIGINT, signal_handler);

    if (argc < 2) {
        std::cout << "Usage: ./corpus_miner <dir_or_csv> [options]\n"
                  << "Options:\n"
                  << "  --mask <mask>    File mask for directory scan (e.g., \"*.txt\")\n"
                  << "  --n <int>        Min document frequency (default: 10)\n"
                  << "  --ngrams <int>   Min phrase length (default: 4)\n"
                  << "  --mem <int>      Memory limit in MB (0 for no limit)\n"
                  << "  --threads <int>  Max CPU threads (0 for all)\n"
                  << "  --sampling <dbl> Data sampling rate 0.0-1.0 (default: 1.0)\n"
                  << "  --cache <int>    Max cache size for on-disk mode (default: 1000)\n"
                  << "  --in-mem         Keep entire corpus in RAM (required for PrefixSpan)\n"
                  << "  --preload        Preload cache while loading\n"
                  << "  --csv-delim <c>  CSV delimiter (default: ',')\n" << std::endl;
        return 1;
    }

    std::string input_path = argv[1];
    int min_docs = 10;
    int ngrams = 4;
    int mem_limit = 0;
    char csv_delimiter = ',';
    int threads = 0;
    int cache_size = 1000;
    double sampling = 1.0;
    bool in_mem = false;
    bool preload = false;
    std::string mask = "";

    // Parse command line arguments
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--n" && i + 1 < argc) min_docs = std::stoi(argv[++i]);
        else if (arg == "--mask" && i + 1 < argc) mask = argv[++i];
        else if (arg == "--ngrams" && i + 1 < argc) ngrams = std::stoi(argv[++i]);
        else if (arg == "--csv-delim" && i + 1 < argc) {
            std::string delim = argv[++i];
            if (delim == "\\t") csv_delimiter = '\t';
            else if (delim == "\\n") csv_delimiter = '\n';
            else if (!delim.empty()) csv_delimiter = delim[0];
        }
        else if (arg == "--mem" && i + 1 < argc) mem_limit = std::stoi(argv[++i]);
        else if (arg == "--threads" && i + 1 < argc) threads = std::stoi(argv[++i]);
        else if (arg == "--sampling" && i + 1 < argc) sampling = std::stod(argv[++i]);
        else if (arg == "--cache" && i + 1 < argc) cache_size = std::stoi(argv[++i]);
        else if (arg == "--in-mem") in_mem = true;
        else if (arg == "--preload") preload = true;
    }

    std::cout << "[START] Initializing Miner..." << std::endl;

    CorpusMiner miner;
    miner.set_limits(threads, mem_limit, cache_size, in_mem, preload);
    miner.set_mask(mask);

    // Determine loading strategy based on input type
    if (fs::is_regular_file(input_path) &&
       (input_path.find(".csv") != std::string::npos || input_path.find(".txt") == std::string::npos)) {
        miner.load_csv(input_path, csv_delimiter, sampling);
    } else if (fs::is_directory(input_path)) {
        miner.load_directory(input_path, sampling);
    } else if (fs::exists(input_path)) {
        // Single file directory-style loading (fallback)
        miner.load_directory(input_path, sampling);
    } else {
        std::cerr << "[ERROR] Path does not exist: " << input_path << std::endl;
        return 1;
    }

    if (in_mem) {
        std::cout << "[MODE] Running in In-Memory mode. PrefixSpan will be efficient." << std::endl;
    } else {
        std::cout << "[MODE] Running in On-Disk mode. PrefixSpan will trigger full load." << std::endl;
    }

    std::cout << "[START] Beginning PrefixSpan mining (min_docs=" << min_docs
              << ", ngrams=" << ngrams << ")..." << std::endl;

    // Result file name can be customized or defaulted
    miner.mine(min_docs, ngrams, "results_max.csv");

    std::cout << "[DONE] Process finished successfully." << std::endl;
    return 0;
}