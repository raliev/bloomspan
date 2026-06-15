#ifndef VMSP_MINER_H
#define VMSP_MINER_H

#include <vector>
#include <map>
#include <unordered_map>
#include <stdint.h>
#include <string>
#include "../mining_algorithm.h"

struct Itemset {
    std::vector<uint32_t> items;
    bool containsAll(const Itemset& other) const;
};

struct Sequence {
    std::vector<Itemset> itemsets;
    uint32_t support = 0;
    uint64_t sumItems = 0;

    bool strictlyContains(const Sequence& other) const;
};

struct Bitmap {
    std::vector<uint64_t> bits;

    Bitmap() = default;
    Bitmap(size_t total_bits);
    void registerBit(int pos);

    // Support is the number of unique documents containing at least one set bit
    uint32_t getSupport(const std::vector<int>& sequenceStarts, size_t numBits) const;

    // S-Step: Item appears in a subsequent position within the same sequence
    static Bitmap createSStep(const Bitmap& prefix, const Bitmap& item,
                             const std::vector<int>& sequenceStarts, size_t numBits);

    static Bitmap createIStep(const Bitmap& prefix, const Bitmap& item);
};

class VmspMiner : public IMiningAlgorithm {
public:
    virtual std::string name() const override { return "VMSP"; }
    virtual std::vector<Phrase> mine(const CorpusMiner& corpus, const MiningParams& params) override;

private:
    void dfs_pruning(Sequence& prefix, const Bitmap& prefix_bitmap,
                    const std::vector<uint32_t>& sn, const std::vector<uint32_t>& in,
                    int min_sup, std::vector<std::vector<Sequence>>& max_patterns_by_size);

    bool save_if_maximal(Sequence& s, std::vector<std::vector<Sequence>>& max_patterns);

    std::map<uint32_t, Bitmap> verticalDB;
    std::vector<int> sequenceStarts; // Stores the starting bit index for each document
    size_t numBits = 0;

    std::unordered_map<uint32_t, std::unordered_map<uint32_t, int>> coocMapAfter;
    std::unordered_map<uint32_t, std::unordered_map<uint32_t, int>> coocMapEquals;
};

#endif