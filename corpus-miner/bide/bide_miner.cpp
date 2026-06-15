#include "bide_miner.h"
#include "../signal_handler.h"
#include <functional>
#include <unordered_map>
#include <algorithm>
#include "../timer.h"

int BideMiner::get_unique_docs(const std::vector<Occurrence>& occs) {
    if (occs.empty()) return 0;
    int count = 1;
    uint32_t last_doc = occs[0].doc_id;
    for (size_t i = 1; i < occs.size(); ++i) {
        if (occs[i].doc_id != last_doc) {
            count++;
            last_doc = occs[i].doc_id;
        }
    }
    return count;
}

// BIDE+ Backward Extension Check (Pruning)
// Returns true if a common item ALWAYS precedes this pattern
bool BideMiner::has_backward_extension(const CorpusMiner& corpus,
                                       const std::vector<uint32_t>& patt,
                                       const std::vector<Occurrence>& matches) {
    if (patt.empty() || matches.empty()) return false;

    uint32_t pattern_len = (uint32_t)patt.size();
    const auto& first_m = matches[0];
    if (first_m.pos < pattern_len) return false;

    uint32_t prev_item = corpus.get_doc(first_m.doc_id)[first_m.pos - pattern_len];

    for (size_t i = 1; i < matches.size(); ++i) {
        const auto& m = matches[i];
        if (m.pos < pattern_len) return false;
        if (corpus.get_doc(m.doc_id)[m.pos - pattern_len] != prev_item) return false;
    }
    return true;
}

// Maximality Check (Frequent Backward Extension)
// Returns true if there is ANY backward extension that is frequent (>= min_sup unique docs)
bool BideMiner::has_frequent_backward_extension(const CorpusMiner& corpus,
                                                const std::vector<uint32_t>& patt,
                                                const std::vector<Occurrence>& matches,
                                                int min_sup) {
    std::unordered_map<uint32_t, int> prev_item_doc_counts;
    // Track the last doc_id processed for each backward item to only count unique docs
    std::unordered_map<uint32_t, uint32_t> last_doc_for_item;

    uint32_t pattern_len = (uint32_t)patt.size();

    for (const auto& m : matches) {
        if (m.pos >= pattern_len) {
            uint32_t prev_item = corpus.get_doc(m.doc_id)[m.pos - pattern_len];
            auto it = last_doc_for_item.find(prev_item);
            if (it == last_doc_for_item.end() || it->second != m.doc_id) {
                last_doc_for_item[prev_item] = m.doc_id;
                prev_item_doc_counts[prev_item]++;
            }
        }
    }

    for (const auto& entry : prev_item_doc_counts) {
        if (entry.second >= min_sup) return true;
    }
    return false;
}

std::vector<Phrase> BideMiner::mine(const CorpusMiner& corpus, const MiningParams& params) {
    std::vector<Phrase> results;
    int min_sup = params.min_docs;

    auto mine_start = start_timer();

    // Recursive BIDE+ function using std::function for lambda recursion
    std::function<void(std::vector<uint32_t>&, const std::vector<Occurrence>&)> bide_rec;

    bide_rec = [&](std::vector<uint32_t>& patt, const std::vector<Occurrence>& matches) {
        if (g_stop_requested) return;

        // 1. BIDE+ Pruning: Backward Extension Check
        if (has_backward_extension(corpus, patt, matches)) return;

        // 2. Generate Extensions (Pseudo-projection logic)
        // Instead of tail-scanning, we look only at the immediate next token for phrases
        std::unordered_map<uint32_t, SupportInfo> extensions;
        for (const auto& m : matches) {
            const auto& seq = corpus.get_doc(m.doc_id);
            uint32_t next_pos = m.pos + 1;

            if (next_pos < (uint32_t)seq.size()) {
                uint32_t next_item = seq[next_pos];
                auto& info = extensions[next_item];
                info.count++;
                info.matches.push_back({m.doc_id, next_pos});
            }
        }

        bool has_freq_forward_extension = false;

        // 4. Recursive Expansion
        for (auto& [item, info] : extensions) {
            int child_support = get_unique_docs(info.matches);
            if (child_support >= min_sup) {
                has_freq_forward_extension = true;
                patt.push_back(item);
                bide_rec(patt, info.matches);
                patt.pop_back();
            }
        }

        // 3. Maximality Verification
        if (!has_freq_forward_extension) {
            if (!has_frequent_backward_extension(corpus, patt, matches, min_sup)) {
                if (patt.size() >= (size_t)params.min_l) {
                    results.push_back({patt, matches, (size_t)get_unique_docs(matches)});
                }
            }
        }
    };

    // Initial Database Projection (Scan for frequent single items)
    std::unordered_map<uint32_t, SupportInfo> root_extensions;
    for (uint32_t i = 0; i < (uint32_t)corpus.num_docs(); ++i) {
        const auto& doc = corpus.get_doc(i);
        for (uint32_t pos = 0; pos < (uint32_t)doc.size(); ++pos) {
            uint32_t item = doc[pos];
            auto& info = root_extensions[item];
            info.count++;
            info.matches.push_back({i, pos});
        }
    }

    for (auto& [item, info] : root_extensions) {
        if (get_unique_docs(info.matches) >= min_sup) {
            std::vector<uint32_t> current_patt = {item};
            bide_rec(current_patt, info.matches);
        }
    }

    std::cout << "\n========== MINING STATISTICS ==========" << std::endl;    
    std::cout << "Total closed patterns found:  " << results.size() << std::endl;
    std::cout << "=======================================\n" << std::endl;

    stop_timer("Total Mining Process", mine_start);

    return results;
}