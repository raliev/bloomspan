#pragma once

#include <memory>
#include <string>
#include "mining_algorithm.h"
#include "_ours/bloom_gram_miner.h"
#include "bide/bide_miner.h"
#include "clospan/clospan_miner.h"
#include "vmsp/vmsp_miner.h"
enum class AlgorithmKind {
    BloomNgram,
    Bide,
    CloSpan,
    VMSP
};

inline AlgorithmKind parse_algorithm_kind(const std::string& name) {
    if (name == "bloomspan" || name == "default")
        return AlgorithmKind::BloomNgram;
    if (name == "bide")
        return AlgorithmKind::Bide;
    if (name == "clospan")
        return AlgorithmKind::CloSpan;
    if (name == "vmsp")
        return AlgorithmKind::VMSP;
    throw std::runtime_error("Unknown algorithm name: " + name);
}

inline std::unique_ptr<IMiningAlgorithm> make_algorithm(AlgorithmKind kind) {
    switch (kind) {
        case AlgorithmKind::BloomNgram:
            return std::make_unique<BloomNgramMiner>();
        case AlgorithmKind::Bide:
            return std::make_unique<BideMiner>();
        case AlgorithmKind::CloSpan:
            return std::make_unique<CloSpanMiner>();
        case AlgorithmKind::VMSP:
                    return std::make_unique<VmspMiner>();
    }
    throw std::runtime_error("Unsupported algorithm kind");
}