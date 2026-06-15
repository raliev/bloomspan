#pragma once

#include "../mining_algorithm.h"
#include "../corpus_miner.h"
#include "../types.h"
#include <unordered_map>
#include <vector>
#include <functional>


class BideMiner : public IMiningAlgorithm {
public:
    std::string name() const override { return "bide"; }

    std::vector<Phrase> mine(const CorpusMiner& corpus,
                             const MiningParams& params) override;

private:
    bool has_backward_extension(const CorpusMiner& corpus,
                                const std::vector<uint32_t>& patt,
                                const std::vector<Occurrence>& matches);

    bool has_frequent_backward_extension(const CorpusMiner& corpus,
                                         const std::vector<uint32_t>& patt,
                                         const std::vector<Occurrence>& matches,
                                         int min_sup);

    int get_unique_docs(const std::vector<Occurrence>& occs);
};