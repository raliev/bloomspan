#include "vmsp_miner.h"
#include "../corpus_miner.h"
#include <algorithm>

// --- Bitmap Logic ---
Bitmap::Bitmap(size_t total_bits) {
    if (total_bits > 0) {
        bits.assign((total_bits + 63) / 64, 0);
    }
}

void Bitmap::registerBit(int pos) {
    size_t idx = pos / 64;
    if (idx < bits.size()) {
        bits[idx] |= (1ULL << (pos % 64));
    }
}

uint32_t Bitmap::getSupport(const std::vector<int>& starts, size_t totalBits) const {
    uint32_t count = 0;
    for (size_t i = 0; i < starts.size(); ++i) {
        int start = starts[i];
        int end = (i + 1 < starts.size()) ? starts[i + 1] : (int)totalBits;

        // Scan words (64 bits at a time) for speed
        bool found = false;
        for (int b = start; b < end; ++b) {
            if (bits[b / 64] & (1ULL << (b % 64))) {
                found = true;
                break;
            }
        }
        if (found) count++;
    }
    return count;
}

Bitmap Bitmap::createSStep(const Bitmap& prefix, const Bitmap& item,
                           const std::vector<int>& starts, size_t totalBits) {
    Bitmap result(totalBits);
    for (size_t i = 0; i < starts.size(); ++i) {
        int start = starts[i];
        int end = (i + 1 < starts.size()) ? starts[i + 1] : (int)totalBits;

        // Check every occurrence of the prefix in this sequence
        for (int b = start; b < end - 1; ++b) {
            if (prefix.bits[b / 64] & (1ULL << (b % 64))) {
                // CONTIGUOUS CHECK: Is the item exactly at the next position?
                int nextPos = b + 1;
                if (item.bits[nextPos / 64] & (1ULL << (nextPos % 64))) {
                    result.registerBit(nextPos);
                }
            }
        }
    }
    return result;
}

Bitmap Bitmap::createIStep(const Bitmap& prefix, const Bitmap& item) {
    Bitmap result(prefix.bits.size() * 64);
    for (size_t i = 0; i < prefix.bits.size(); ++i) {
        result.bits[i] = prefix.bits[i] & item.bits[i];
    }
    return result;
}

// --- Sequence & Containment Logic ---
bool Itemset::containsAll(const Itemset& other) const {
    for (auto item : other.items) {
        if (std::find(items.begin(), items.end(), item) == items.end()) return false;
    }
    return true;
}

bool Sequence::strictlyContains(const Sequence& other) const {
    if (other.itemsets.size() > itemsets.size()) return false;
    size_t i = 0, j = 0;
    while (i < itemsets.size() && j < other.itemsets.size()) {
        if (itemsets[i].containsAll(other.itemsets[j])) j++;
        i++;
    }
    return j == other.itemsets.size();
}

// --- VMSP Mining Algorithm ---

bool VmspMiner::save_if_maximal(Sequence& s, std::vector<std::vector<Sequence>>& max_patterns) {
    int length = 0;
    for(auto& is : s.itemsets) length += is.items.size();

    // Super-pattern check: Is 's' a sub-phrase of an existing longer phrase?
    for (size_t i = length; i < max_patterns.size(); ++i) {
        for (const auto& pPrime : max_patterns[i]) {
            // Strictly check support AND containment
            if (pPrime.strictlyContains(s)) {
                return false;
            }
        }
    }

    // Sub-pattern removal: Remove existing phrases that are sub-phrases of 's'
    for (size_t i = 1; i < (size_t)length && i < max_patterns.size(); ++i) {
        auto& layer = max_patterns[i];
        layer.erase(std::remove_if(layer.begin(), layer.end(), [&](const Sequence& pPrime) {
            // Remove if 's' is longer/equal and contains pPrime with same/lower support
            return (s.support >= pPrime.support && s.strictlyContains(pPrime));
        }), layer.end());
    }

    if (max_patterns.size() <= (size_t)length) max_patterns.resize(length + 1);
    max_patterns[length].push_back(s);
    return true;
}

void VmspMiner::dfs_pruning(Sequence& prefix, const Bitmap& prefix_bitmap,
                            const std::vector<uint32_t>& sn, const std::vector<uint32_t>& in,
                            int min_sup, std::vector<std::vector<Sequence>>& max_patterns) {

    bool hasExtension = false;
    uint32_t last_item = prefix.itemsets.back().items.back();

    for (uint32_t item : sn) {
        auto it = coocMapAfter.find(last_item);
        if (it == coocMapAfter.end() || it->second[item] < min_sup) continue;

        Bitmap next_bm = Bitmap::createSStep(prefix_bitmap, verticalDB[item], sequenceStarts, numBits);
        uint32_t support = next_bm.getSupport(sequenceStarts, numBits);
        if (support >= (uint32_t)min_sup) {
            hasExtension = true;
            prefix.itemsets.push_back({{item}});
            uint32_t old_support = prefix.support;
            prefix.support = support;
            prefix.sumItems += item;
            dfs_pruning(prefix, next_bm, sn, sn, min_sup, max_patterns);
            prefix.sumItems -= item;
            prefix.support = old_support;
            prefix.itemsets.pop_back();
        }
    }

    // Concurrent extensions (I-Step) logic remains if needed for multi-item itemsets
    if (!hasExtension) {
        save_if_maximal(prefix, max_patterns);
    }
}

std::vector<Phrase> VmspMiner::mine(const CorpusMiner& corpus, const MiningParams& params) {
    int min_sup = params.min_docs;
    verticalDB.clear();
    sequenceStarts.clear();
    numBits = 0;
    coocMapAfter.clear();
    coocMapEquals.clear();

    for (size_t d = 0; d < corpus.num_docs(); ++d) numBits += corpus.get_doc(d).size();

    int current_bit = 0;
    for (size_t d = 0; d < corpus.num_docs(); ++d) {
        const auto& doc = corpus.get_doc(d);
        sequenceStarts.push_back(current_bit);

        for (size_t i = 0; i < doc.size(); ++i) {
            uint32_t itemI = doc[i];
            if (verticalDB.find(itemI) == verticalDB.end()) verticalDB[itemI] = Bitmap(numBits);
            verticalDB[itemI].registerBit(current_bit + i);

            for (size_t j = i + 1; j < doc.size(); ++j) {
                coocMapAfter[itemI][doc[j]]++;
            }
        }
        current_bit += doc.size();
    }

    std::vector<uint32_t> frequent_items;
    for (auto& pair : verticalDB) {
        uint32_t support = pair.second.getSupport(sequenceStarts, numBits);
        if (support >= (uint32_t)min_sup) {
            frequent_items.push_back(pair.first);
        }
    }
    std::sort(frequent_items.begin(), frequent_items.end());

    std::vector<std::vector<Sequence>> max_patterns_by_size;
    for (uint32_t item : frequent_items) {
        Sequence prefix;
        prefix.itemsets.push_back({{item}});
        prefix.support = verticalDB[item].getSupport(sequenceStarts, numBits);
        prefix.sumItems = item;
        dfs_pruning(prefix, verticalDB[item], frequent_items, frequent_items, min_sup, max_patterns_by_size);
    }

    std::vector<Phrase> final_phrases;
    for (auto& layer : max_patterns_by_size) {
        for (auto& seq : layer) {
            std::vector<uint32_t> tokens;
            for (auto& is : seq.itemsets) for (uint32_t t : is.items) tokens.push_back(t);
            final_phrases.push_back({tokens, {}, (size_t)seq.support});
        }
    }
    return final_phrases;
}