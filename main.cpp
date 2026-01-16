#include "corpus_miner.h"
#include "signal_handler.h"
#include <iostream>
#include <csignal>

int main(int argc, char** argv) {
    // Set up Ctrl+C handler
    std::signal(SIGINT, signal_handler);

    if (argc < 2) {
        std::cout << "Usage: ./corpus_miner <dir> [--n <min_docs>] [--ngrams <n>] [--sampling <rate>]" << std::endl;
        return 1;
    }

    std::string directory = argv[1];
    int min_docs = 10;
    int ngrams = 4;
    double sampling = 1.0;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "--n" || arg == "-n") && i + 1 < argc) {
            min_docs = std::stoi(argv[++i]);
        } else if (arg == "--ngrams" && i + 1 < argc) {
            ngrams = std::stoi(argv[++i]);
        } else if (arg == "--sampling" && i + 1 < argc) {
            sampling = std::stod(argv[++i]);
        }
    }

    std::cout << "[START] Initializing Miner..." << std::endl;
    CorpusMiner m;
    m.load_directory(directory, sampling);

    std::cout << "[START] Beginning mining with min_docs=" << min_docs << ", ngrams=" << ngrams << std::endl;
    m.mine(min_docs, ngrams, "results_max.csv");

    std::cout << "[DONE] Process finished." << std::endl;
    return 0;
}
