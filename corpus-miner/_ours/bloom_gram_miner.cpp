#include "bloom_gram_miner.h"
#include "../timer.h"
#include "../signal_handler.h"
#include <filesystem>
#include <iostream>
#include <stack>
#include <fstream>
#include <execution>
#include <queue>
#include <random>
#include <omp.h>
#include <unordered_set>
#include <unordered_map>
#include <memory>
#include <cstring>
#include <vector>
#include <unistd.h>
#include <algorithm> // Added for std::search
#ifdef __APPLE__
#include <mach/mach.h>
#endif

namespace fs = std::filesystem;

// Local copy of constants/structs/helpers used by the Bloom n-gram miner.
// These mirror the definitions in corpus_miner.cpp, but are TU-local here.
const int SMALL_NGRAMS_THRESHOLD = 16;  // Use fixed array for n <= this value
const int MAX_NGRAMS_FIXED = 16;        // Maximum size for fixed array
const int DEBUG = 0;                    // to see internal structures in the console

struct RawSeedEntry {
    uint32_t doc_id;
    uint32_t pos;
    int n;

    // Hybrid storage: fixed array for small n-grams, vector for large ones
    union {
        uint32_t fixed_tokens[MAX_NGRAMS_FIXED];
        std::vector<uint32_t>* dynamic_tokens;
    } tokens;

    // Flag to indicate which storage is being used
    bool is_dynamic;

    RawSeedEntry() : doc_id(0), pos(0), n(0), is_dynamic(false) {
        std::memset(tokens.fixed_tokens, 0, sizeof(tokens.fixed_tokens));
    }

    ~RawSeedEntry() {
        if (is_dynamic && tokens.dynamic_tokens != nullptr) {
            delete tokens.dynamic_tokens;
            tokens.dynamic_tokens = nullptr;
        }
    }

    // Copy constructor
    RawSeedEntry(const RawSeedEntry& other)
        : doc_id(other.doc_id),
          pos(other.pos),
          n(other.n),
          is_dynamic(other.is_dynamic) {
        if (is_dynamic) {
            tokens.dynamic_tokens = new std::vector<uint32_t>(*other.tokens.dynamic_tokens);
        } else {
            std::memcpy(tokens.fixed_tokens, other.tokens.fixed_tokens,
                        sizeof(tokens.fixed_tokens));
        }
    }

    // Move constructor
    RawSeedEntry(RawSeedEntry&& other) noexcept
        : doc_id(other.doc_id),
          pos(other.pos),
          n(other.n),
          is_dynamic(other.is_dynamic) {
        if (is_dynamic) {
            tokens.dynamic_tokens = other.tokens.dynamic_tokens;
            other.tokens.dynamic_tokens = nullptr;
        } else {
            std::memcpy(tokens.fixed_tokens, other.tokens.fixed_tokens,
                        sizeof(tokens.fixed_tokens));
        }
    }

    // Copy assignment
    RawSeedEntry& operator=(const RawSeedEntry& other) {
        if (this == &other) return *this;

        // Clean up old data
        if (is_dynamic && tokens.dynamic_tokens != nullptr) {
            delete tokens.dynamic_tokens;
            tokens.dynamic_tokens = nullptr;
        }

        doc_id = other.doc_id;
        pos = other.pos;
        n = other.n;
        is_dynamic = other.is_dynamic;

        if (is_dynamic) {
            tokens.dynamic_tokens = new std::vector<uint32_t>(*other.tokens.dynamic_tokens);
        } else {
            std::memcpy(tokens.fixed_tokens, other.tokens.fixed_tokens,
                        sizeof(tokens.fixed_tokens));
        }
        return *this;
    }

    // Move assignment
    RawSeedEntry& operator=(RawSeedEntry&& other) noexcept {
        if (this == &other) return *this;

        // Clean up old data
        if (is_dynamic && tokens.dynamic_tokens != nullptr) {
            delete tokens.dynamic_tokens;
            tokens.dynamic_tokens = nullptr;
        }

        doc_id = other.doc_id;
        pos = other.pos;
        n = other.n;
        is_dynamic = other.is_dynamic;

        if (is_dynamic) {
            tokens.dynamic_tokens = other.tokens.dynamic_tokens;
            other.tokens.dynamic_tokens = nullptr;
        } else {
            std::memcpy(tokens.fixed_tokens, other.tokens.fixed_tokens,
                        sizeof(tokens.fixed_tokens));
        }
        return *this;
    }

    // Helper to get token at index
    uint32_t get_token(int idx) const {
        if (is_dynamic) {
            return (*tokens.dynamic_tokens)[idx];
        } else {
            return tokens.fixed_tokens[idx];
        }
    }

    // Helper to set token at index
    void set_token(int idx, uint32_t value) {
        if (is_dynamic) {
            (*tokens.dynamic_tokens)[idx] = value;
        } else {
            tokens.fixed_tokens[idx] = value;
        }
    }

    // Initialize with n-grams
    void init_tokens(int num_tokens) {
        if (is_dynamic && tokens.dynamic_tokens != nullptr) {
            delete tokens.dynamic_tokens;
            tokens.dynamic_tokens = nullptr;
        }

        n = num_tokens;
        if (num_tokens > SMALL_NGRAMS_THRESHOLD) {
            is_dynamic = true;
            tokens.dynamic_tokens = new std::vector<uint32_t>(num_tokens, 0);
        } else {
            is_dynamic = false;
            std::memset(tokens.fixed_tokens, 0, sizeof(tokens.fixed_tokens));
        }
    }

    // Оператор для priority_queue (нужен обратный порядок для min-heap)
    bool operator>(const RawSeedEntry& other) const {
        for (int i = 0; i < n; ++i) {
            uint32_t this_token =
                (is_dynamic) ? (*tokens.dynamic_tokens)[i] : tokens.fixed_tokens[i];
            uint32_t other_token = (other.is_dynamic)
                                       ? (*other.tokens.dynamic_tokens)[i]
                                       : other.tokens.fixed_tokens[i];
            if (this_token != other_token) return this_token > other_token;
        }
        if (doc_id != other.doc_id) return doc_id > other.doc_id;
        return pos > other.pos;
    }

    bool same_tokens(const RawSeedEntry& other) const {
        for (int i = 0; i < n; ++i) {
            uint32_t this_token =
                (is_dynamic) ? (*tokens.dynamic_tokens)[i] : tokens.fixed_tokens[i];
            uint32_t other_token = (other.is_dynamic)
                                       ? (*other.tokens.dynamic_tokens)[i]
                                       : other.tokens.fixed_tokens[i];
            if (this_token != other_token) return false;
        }
        return true;
    }

    // Serialization: writes to binary stream
    void write_to_stream(std::ofstream& out) const {
        out.write((char*)&doc_id, sizeof(doc_id));
        out.write((char*)&pos, sizeof(pos));
        out.write((char*)&n, sizeof(n));
        out.write((char*)&is_dynamic, sizeof(is_dynamic));

        if (is_dynamic) {
            for (int i = 0; i < n; ++i) {
                uint32_t token = (*tokens.dynamic_tokens)[i];
                out.write((char*)&token, sizeof(token));
            }
        } else {
            out.write((char*)tokens.fixed_tokens, n * sizeof(uint32_t));
        }
    }

    // Deserialization: reads from binary stream
    void read_from_stream(std::ifstream& in) {
        in.read((char*)&doc_id, sizeof(doc_id));
        in.read((char*)&pos, sizeof(pos));
        in.read((char*)&n, sizeof(n));
        in.read((char*)&is_dynamic, sizeof(is_dynamic));

        if (is_dynamic && tokens.dynamic_tokens != nullptr) {
            delete tokens.dynamic_tokens;
            tokens.dynamic_tokens = nullptr;
        }

        if (is_dynamic) {
            tokens.dynamic_tokens = new std::vector<uint32_t>(n);
            for (int i = 0; i < n; ++i) {
                uint32_t token;
                in.read((char*)&token, sizeof(token));
                (*tokens.dynamic_tokens)[i] = token;
            }
        } else {
            std::memset(tokens.fixed_tokens, 0, sizeof(tokens.fixed_tokens));
            in.read((char*)tokens.fixed_tokens, n * sizeof(uint32_t));
        }
    }
};

inline uint64_t hash_tokens(const uint32_t* tokens, int n) {
    uint64_t h = 14695981039346656037ULL; // FNV offset basis
    for (int i = 0; i < n; ++i) {
        h ^= static_cast<uint64_t>(tokens[i]);
        h *= 1099511628211ULL; // FNV prime
    }
    return h;
}

static bool isSubArray(const std::vector<uint32_t>& larger, const std::vector<uint32_t>& smaller) {
    if (smaller.empty()) return true;
    if (smaller.size() > larger.size()) return false;

    uint32_t first_token = smaller[0];
    bool found_first = false;
    size_t max_start = larger.size() - smaller.size();
    for (size_t i = 0; i <= max_start; ++i) {
        if (larger[i] == first_token) {
            found_first = true;
            break;
        }
    }
    if (!found_first) return false;

    for (size_t i = 0; i <= max_start; ++i) {
        if (larger[i] != first_token) continue;
        bool match = true;
        for (size_t j = 1; j < smaller.size(); ++j) {
            if (larger[i + j] != smaller[j]) {
                match = false;
                break;
            }
        }
        if (match) return true;
    }
    return false;
}

std::vector<Phrase> BloomNgramMiner::mine(const CorpusMiner& corpus,
                                          const MiningParams& params) {

    auto mine_start = start_timer();

    // Unpack params
    int min_docs = params.min_docs;
    int ngrams   = params.ngrams;

    // Access corpus-level config/state via getters
    int max_threads        = corpus.get_max_threads();
    size_t memory_limit_mb = corpus.get_memory_limit_mb();
    bool in_memory_only    = corpus.is_in_memory_only();

    const auto& doc_lengths    = corpus.get_doc_lengths();
    const auto& doc_offsets    = corpus.get_doc_offsets();
    const auto& word_df        = corpus.get_word_df();
    const auto& id_to_word     = corpus.get_id_to_word();
    const auto& bin_corpus_path = corpus.get_bin_corpus_path();

    // Local helper to get current RSS (copy of original CorpusMiner::get_current_rss_mb)
    auto get_current_rss_mb = []() -> size_t {
#ifdef __APPLE__
        struct mach_task_basic_info info;
        mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
        if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info, &count) == KERN_SUCCESS) {
            return info.resident_size / (1024 * 1024);
        }
        return 0;
#else
        std::ifstream stat_stream("/proc/self/statm", std::ios_base::in);
        unsigned long long pages;
        if (!(stat_stream >> pages >> pages)) return 0;
        return (pages * sysconf(_SC_PAGESIZE)) / (1024 * 1024);
#endif
    };

    if (max_threads > 0) {
        omp_set_num_threads(max_threads);
        std::cout << "[LOG] Threads limited to: " << max_threads << std::endl;
    }

    // 1. Dynamic Filter Size: Aim for ~20% of memory limit, capped at 2GB.
    // A larger filter significantly reduces collisions for bigrams.
    size_t filter_size;
    if (memory_limit_mb > 0) {
        filter_size = (memory_limit_mb * 1024ULL * 1024ULL) / 5; // 20% of limit
        if (filter_size > 2048ULL * 1024ULL * 1024ULL) filter_size = 2048ULL * 1024ULL * 1024ULL;
    } else {
        filter_size = 512ULL * 1024ULL * 1024ULL;
    }

    std::cout << "[LOG] Initializing Bloom Filter: " << (filter_size / (1024 * 1024)) << " MB" << std::endl;
    std::vector<uint8_t> filter_counters(filter_size, 0);

    // Pass 1: Frequency Estimation
    std::cout << "[LOG] Bloom Pass: Estimating n-gram frequencies..." << std::endl;
    #pragma omp parallel
    {
        std::ifstream local_bin;
        if (!in_memory_only) local_bin.open(bin_corpus_path, std::ios::binary);

        #pragma omp for
        for (uint32_t d = 0; d < (uint32_t)doc_lengths.size(); ++d) {
            std::vector<uint32_t> local_doc;
            const std::vector<uint32_t>* doc_ptr = nullptr;

            if (in_memory_only) {
                // In-memory mode: fetch directly from corpus
                doc_ptr = &corpus.get_doc(d);
            } else {
                // Disk mode: load from BIN file
                local_doc.resize(doc_lengths[d]);
                local_bin.seekg(doc_offsets[d]);
                local_bin.read((char*)local_doc.data(), doc_lengths[d] * sizeof(uint32_t));
                doc_ptr = &local_doc;
            }

            if (doc_ptr->size() < (size_t)ngrams) continue;

            // here we count the ngrams before the counter reaches 255
            // the goal is to filter out the ngrams with low frequency (<num_docs) from further processing
            for (uint32_t p = 0; p <= doc_ptr->size() - ngrams; ++p) {
                uint64_t h = hash_tokens(doc_ptr->data() + p, ngrams);
                size_t idx = h % filter_size;

                uint8_t* target = &filter_counters[idx];
                uint8_t current = __atomic_load_n(target, __ATOMIC_RELAXED);
                while (current < 255) {
                    if (__atomic_compare_exchange_n(target, &current, current + 1, false,
                                                    __ATOMIC_RELAXED, __ATOMIC_RELAXED))
                        break;
                }
            }
        }
    }

    // we collected the ngram stats in filter counters
    // we have not saved the ngrams themselves anywhere (because for the large datasets this number can skyrocket)

    // Pass 2: Collection with Statistics
    std::cout << "[LOG] Step 1: Gathering " << ngrams << "-gram seeds..." << std::endl;
    auto s1_start = start_timer();
    size_t total_processed = 0;
    size_t seeds_passed = 0;
    size_t seeds_rejected = 0;

    std::string temp_dir = "./miner_tmp";
    fs::create_directories(temp_dir);
    std::vector<std::string> chunk_files;
    std::vector<RawSeedEntry> buffer;
    buffer.reserve(1000000);
    int chunk_id = 0;

    auto flush_buffer = [&]() {
        if (buffer.empty()) return;
        if (in_memory_only) return;
        std::cout << "\n[LOG] Flushing " << buffer.size() << " seeds to disk... (RAM: "
                  << get_current_rss_mb() << " MB)" << std::endl;
        std::sort(std::execution::par, buffer.begin(), buffer.end(),
                  [](const RawSeedEntry& a, const RawSeedEntry& b) {
                      for (int i = 0; i < a.n; ++i) {
                          uint32_t a_token = (a.is_dynamic) ? (*a.tokens.dynamic_tokens)[i]
                                                            : a.tokens.fixed_tokens[i];
                          uint32_t b_token = (b.is_dynamic) ? (*b.tokens.dynamic_tokens)[i]
                                                            : b.tokens.fixed_tokens[i];
                          if (a_token != b_token) return a_token < b_token;
                      }
                      return a.doc_id < b.doc_id;
                  });
        std::string fname = temp_dir + "/chunk_" + std::to_string(chunk_id++) + ".bin";
        chunk_files.push_back(fname);
        std::ofstream out(fname, std::ios::binary);
        if (out) {
            for (const auto& entry : buffer) {
                entry.write_to_stream(out);
            }
        }
        buffer.clear();
        buffer.shrink_to_fit();
    };

    for (uint32_t d = 0; d < (uint32_t)doc_lengths.size(); ++d) {
        // since this is memory intensive processing, we offload data to the files (chunks)
        if (memory_limit_mb > 0 && get_current_rss_mb() >= (size_t)(memory_limit_mb * 0.75))
            flush_buffer();
        // fetch_doc fetches from the disk and caches in memory; with the --in-mem flag it does not use the disk
        const auto& current_doc = corpus.get_doc(d);
        if (current_doc.size() < (size_t)ngrams) continue;

        for (uint32_t p = 0; p <= current_doc.size() - ngrams; ++p) {
            total_processed++;
            uint64_t h = hash_tokens(&current_doc[p], ngrams);

            if (DEBUG) {
                std::cout << "[DEBUG] Doc " << d << " Pos " << p << " Hash: " << h << std::endl;
                std::cout << "[DEBUG] Tokens: ";
                for (int k = 0; k < ngrams; ++k) {
                    std::cout << id_to_word[current_doc[p + k]] << " ";
                }
                std::cout << std::endl;
                std::cout << "[DEBUG] Filter Counter: "
                          << (int)filter_counters[h % filter_size] << std::endl;
                std::cout << std::endl;
                std::cout << std::flush;
            }

            // Bloom Filter check. The BF is probabilistic, it uses a hash as an input which may have collisions
            // we don't process ngrams until they reach min_docs or 255
            if (filter_counters[h % filter_size] >= (uint8_t)std::min(min_docs, 255)) {
                // DF check
                // it is required because Bloom Filter is probabilistic and may produce false positives
                bool df_ok = true;
                for (int i = 0; i < ngrams; ++i) {
                    if (word_df[current_doc[p + i]] < (uint32_t)min_docs) {
                        df_ok = false;
                        break;
                    }
                }

                if (df_ok) {
                    // saving the candidate in the buffer (std::vector<RawSeedEntry>)
                    RawSeedEntry entry;
                    entry.init_tokens(ngrams);
                    entry.n = ngrams;
                    entry.doc_id = d;
                    entry.pos = p;
                    for (int i = 0; i < ngrams; ++i)
                        entry.set_token(i, current_doc[p + i]);
                    buffer.push_back(entry);
                    seeds_passed++;
                } else {
                    seeds_rejected++;
                }
            } else {
                seeds_rejected++;
            }
        }
        if (d % 500 == 0 || d == doc_lengths.size() - 1) {
            std::cout << "[LOG] Scanning: " << (d + 1) << "/" << doc_lengths.size()
                      << " | Seeds Found: " << seeds_passed << " \r" << std::flush;
        }
    }

    double efficiency = (total_processed > 0) ? (100.0 * seeds_rejected / total_processed) : 0;
    std::cout << "\n[BLOOM STATS] Total n-grams: " << total_processed << std::endl;
    std::cout << "[BLOOM STATS] Accepted:    " << seeds_passed << std::endl;
    std::cout << "[BLOOM STATS] Rejected:    " << seeds_rejected << " (" << efficiency << "% reduction)" << std::endl;

    filter_counters.clear();
    filter_counters.shrink_to_fit();
    
    bool merge_in_memory = chunk_files.empty();
    if (merge_in_memory) {
        std::cout << "[LOG] Sorting all " << buffer.size() << " seeds in RAM..." << std::endl;
        std::sort(std::execution::par, buffer.begin(), buffer.end(),
                  [](const RawSeedEntry& a, const RawSeedEntry& b) {
                      for (int i = 0; i < a.n; ++i) {
                          uint32_t a_token = (a.is_dynamic) ? (*a.tokens.dynamic_tokens)[i] : a.tokens.fixed_tokens[i];
                          uint32_t b_token = (b.is_dynamic) ? (*b.tokens.dynamic_tokens)[i] : b.tokens.fixed_tokens[i];
                          if (a_token != b_token) return a_token < b_token;
                      }
                      if (a.doc_id != b.doc_id) return a.doc_id < b.doc_id;
                      return a.pos < b.pos;
                  });
    } else {
        flush_buffer();
    }
    std::cout << std::endl;

    // Step 1.5: Merging and filtering candidates
    std::cout << "[LOG] Step 1.5: Merging and filtering candidates..." << std::endl;
    std::vector<Phrase> candidates;

    if (merge_in_memory) {
        // --- PATH A: In-Memory Processing ---
        size_t i = 0;
        while (i < buffer.size()) {
            const RawSeedEntry& representative = buffer[i];
            std::vector<Occurrence> current_occs;
            std::unordered_set<uint32_t> unique_docs;

            // Group identical tokens in the sorted RAM buffer
            while (i < buffer.size() && buffer[i].same_tokens(representative)) {
                current_occs.push_back({buffer[i].doc_id, buffer[i].pos});
                unique_docs.insert(buffer[i].doc_id);
                i++;
            }

            // Support check (min_docs)
            if (unique_docs.size() >= (size_t)min_docs) {
                std::vector<uint32_t> tokens_vec(ngrams);
                for (int k = 0; k < ngrams; ++k) {
                    tokens_vec[k] = (representative.is_dynamic)
                                        ? (*representative.tokens.dynamic_tokens)[k]
                                        : representative.tokens.fixed_tokens[k];
                }
                candidates.push_back({tokens_vec, std::move(current_occs), unique_docs.size()});
            }
        }
        buffer.clear();
        buffer.shrink_to_fit();
    } else {
        struct ChunkReader {
            std::ifstream stream;
            RawSeedEntry current;
            bool active;
            bool next() {
                try {
                    current.read_from_stream(stream);
                    if (!stream) { active = false; return false; }
                    return true;
                } catch (...) { active = false; return false; }
            }
        };

        auto cmp = [](ChunkReader* a, ChunkReader* b) { return a->current > b->current; };
        std::priority_queue<ChunkReader*, std::vector<ChunkReader*>, decltype(cmp)> pq(cmp);
        std::vector<std::unique_ptr<ChunkReader>> readers;

        for (const auto& file : chunk_files) {
            auto r = std::make_unique<ChunkReader>();
            r->stream.open(file, std::ios::binary);
            if (r->next()) {
                r->active = true;
                pq.push(r.get());
            }
            readers.push_back(std::move(r));
        }

        while (!pq.empty()) {
            RawSeedEntry representative = pq.top()->current;
            std::vector<Occurrence> current_occs;
            std::unordered_set<uint32_t> unique_docs;

            while (!pq.empty() && pq.top()->current.same_tokens(representative)) {
                ChunkReader* r = pq.top();
                pq.pop();
                current_occs.push_back({r->current.doc_id, r->current.pos});
                unique_docs.insert(r->current.doc_id);
                if (r->next()) pq.push(r);
            }

            if (unique_docs.size() >= (size_t)min_docs) {
                std::vector<uint32_t> tokens_vec(ngrams);
                for (int k = 0; k < ngrams; ++k) {
                    tokens_vec[k] = (representative.is_dynamic) ? (*representative.tokens.dynamic_tokens)[k] : representative.tokens.fixed_tokens[k];
                }
                candidates.push_back({tokens_vec, std::move(current_occs), unique_docs.size()});
            }
        }

        for (auto& r : readers) if (r->stream.is_open()) r->stream.close();
        readers.clear();
        try { if (fs::exists(temp_dir)) { fs::remove_all(temp_dir); std::cout << "[LOG] Step 1.5: Temporary directory and chunk files removed." << std::endl; } }
        catch (const fs::filesystem_error& e) { std::cerr << "[WARNING] Cleanup failed: " << e.what() << std::endl; }
    }

    size_t total_seeds_generated = candidates.size();
    stop_timer(std::to_string(ngrams) + "-gram Seed Generation (Disk)", s1_start);

    std::cout << "[LOG] Step 2: Sorting " << candidates.size() << " candidates by score (support * length)..." << std::endl;
    std::sort(candidates.begin(), candidates.end(), [](const Phrase& a, const Phrase& b) {
        int scoreA = a.support * a.tokens.size();
        int scoreB = b.support * b.tokens.size();
        if (scoreA != scoreB) return scoreA > scoreB; // Score descending
        if (a.tokens.size() != b.tokens.size()) return a.tokens.size() > b.tokens.size(); // Length descending
        return a.tokens < b.tokens; // Lexicographical tie-breaker
    });

    std::cout << "[LOG] Step 3: Expanding with Path Compression (Jumps)..." << std::endl;
    auto s3_start = start_timer();
    std::vector<Phrase> final_phrases;

    // Faster flat-array data structures replacing slow unordered_map memory fragmentation
    std::vector<std::vector<size_t>> existing_by_length(100); // Max reasonable phrase length pool
    std::vector<std::vector<size_t>> inverted_index(id_to_word.size());

    // Compute flat token offsets for 1D processed matrix (doc_offsets is in bytes!)
    std::vector<size_t> token_offsets(doc_lengths.size() + 1, 0);
    size_t total_corpus_tokens = 0;
    for (size_t i = 0; i < doc_lengths.size(); ++i) {
        token_offsets[i] = total_corpus_tokens;
        total_corpus_tokens += doc_lengths[i];
    }
    token_offsets[doc_lengths.size()] = total_corpus_tokens;

    // 1D processed matrix
    std::vector<uint8_t> processed(total_corpus_tokens, 0);

    // Fast O(N) bucketing arrays to replace HashMap/Sorting overhead
    std::vector<int> group_head(id_to_word.size(), -1);
    std::vector<int> group_next; // Will resize to current.occs.size() dynamically
    std::vector<uint32_t> active_words;

    uint64_t ns_extend = 0;
    uint64_t ns_maximal = 0;
    uint64_t ext_loop_count = 0;
    uint64_t max_loop_count = 0;
    uint64_t ext_dfs_count = 0;

    // Fast flat array pools to replace dynamic map and set instantiations inside DFS
    // Hoisted outside to avoid billions of vector instantiations across all candidate trees
    std::vector<uint32_t> active_ext;
    active_ext.reserve(1024);
    
    std::vector<uint8_t> doc_seen(doc_lengths.size(), 0);
    std::vector<uint32_t> active_docs;
    active_docs.reserve(doc_lengths.size());

    // TLAB-simulated pre-allocated pools to bypass malloc locking overhead
    std::vector<uint32_t> pool_valid_words;
    std::vector<Occurrence> pool_group_occs;
    std::vector<uint32_t> pool_unique_docs;
    pool_valid_words.reserve(100000);
    pool_group_occs.reserve(100000);
    pool_unique_docs.reserve(doc_lengths.size());

    for (size_t c_idx = 0; c_idx < candidates.size(); ++c_idx) {
            auto& cand = candidates[c_idx];

            // Progress reporting
            if (c_idx % 100 == 0 || c_idx == candidates.size() - 1) {
                std::cout << "[LOG] Expanding: " << (c_idx + 1) << "/" << candidates.size()
                          << " | Phrases found: " << final_phrases.size() << "          \r" << std::flush;
            }

            // Processed check to skip redundant work - check ALL tokens in the candidate
            // Prune branch if its occurrences are already fully covered by another maximal phrase
            bool all_processed = true;
            for (const auto& o : cand.occs) {
                size_t flat_base = token_offsets[o.doc_id];
                size_t dlen = corpus.get_doc(o.doc_id).size();
                size_t limit = (o.pos + cand.tokens.size() <= dlen) ? cand.tokens.size() : (dlen > o.pos ? dlen - o.pos : 0);
                
                for (size_t i = 0; i < limit; ++i) {
                    if (!processed[flat_base + o.pos + i]) {
                        all_processed = false;
                        break;
                    }
                }
                if (!all_processed) break;
            }          if (all_processed) {
                if (c_idx == candidates.size() - 1) {
                    std::cout << "\n[PROFILE] Extension MS: " << (ns_extend / 1000000) 
                              << " | Maximality MS: " << (ns_maximal / 1000000)
                              << " | DFS Nodes: " << ext_dfs_count
                              << " | Ext Iterations: " << ext_loop_count
                              << " | Max Iterations: " << max_loop_count << std::endl;
                }
                continue;
            }

            // Use a stack to handle branching from this seed (DFS)
            std::stack<Phrase> stack;
            stack.push(std::move(cand));

            while (!stack.empty()) {
                ext_dfs_count++;
                Phrase current = std::move(stack.top());
                stack.pop();

                // Backward-extension check (Local Pruning)
                auto t_ext_start = std::chrono::high_resolution_clock::now();
                bool can_extend_left = false;
                if (!current.occs.empty()) {
                    uint32_t first_doc = current.occs[0].doc_id;
                    uint32_t first_pos = current.occs[0].pos;
                    if (first_pos > 0) {
                        uint32_t common_prev = corpus.get_doc(first_doc)[first_pos - 1];
                        bool all_match = true;
                        for (auto& o : current.occs) {
                            if (o.pos == 0 || corpus.get_doc(o.doc_id)[o.pos - 1] != common_prev) {
                                all_match = false;
                                break;
                            }
                        }
                        if (all_match) can_extend_left = true;
                    }
                }

                if (can_extend_left) {
                    auto t_ext_end = std::chrono::high_resolution_clock::now();
                    ns_extend += std::chrono::duration_cast<std::chrono::nanoseconds>(t_ext_end - t_ext_start).count();
                    continue; // Prune branch completely
                }

                // 2. Processed Mask Pruning (Local Pruning)
                // O(1) flat array scan prevents O(N^2) Maximality Check later
                bool all_processed = true;
                for (const auto& o : current.occs) {
                    size_t flat_base = token_offsets[o.doc_id];
                    size_t dlen = corpus.get_doc(o.doc_id).size();
                    size_t limit = (o.pos + current.tokens.size() <= dlen) ? current.tokens.size() : (dlen > o.pos ? dlen - o.pos : 0);

                    for (size_t i = 0; i < limit; ++i) {
                        if (!processed[flat_base + o.pos + i]) {
                            all_processed = false;
                            break;
                        }
                    }
                    if (!all_processed) break;
                }
                
                if (all_processed) {
                    auto t_ext_end = std::chrono::high_resolution_clock::now();
                    ns_extend += std::chrono::duration_cast<std::chrono::nanoseconds>(t_ext_end - t_ext_start).count();
                    continue;
                }

                // Use an O(N) array-backed linked list to group occurrences by next token
                // Bypasses std::sort() O(N log N) overhead and std::unordered_map malloc locking
                size_t c_size = current.occs.size();
                if (group_next.size() < c_size) group_next.resize(c_size, -1);
                active_words.clear();

                pool_valid_words.clear();

                for (size_t i = 0; i < c_size; ++i) {
                    ext_loop_count++;
                    const auto& o = current.occs[i];
                    const auto& doc = corpus.get_doc(o.doc_id);
                    uint32_t np = o.pos + (uint32_t)current.tokens.size();
                    
                    if (np < doc.size()) {
                        uint32_t word = doc[np];
                        pool_valid_words.push_back(word);

                        if (group_head[word] == -1) {
                            active_words.push_back(word);
                        }
                        group_next[i] = group_head[word];
                        group_head[word] = i;
                    } else {
                        pool_valid_words.push_back(0xFFFFFFFF); // Sentinel for end-of-document
                    }
                }

                bool expanded = false;

                for (uint32_t word : active_words) {
                    pool_group_occs.clear();
                    pool_unique_docs.clear();

                    int curr_idx = group_head[word];
                    while (curr_idx != -1) {
                        pool_group_occs.push_back(current.occs[curr_idx]);
                        pool_unique_docs.push_back(current.occs[curr_idx].doc_id);
                        curr_idx = group_next[curr_idx];
                    }

                    // Reset grouped list head
                    group_head[word] = -1;

                    // Sort and unique to get true count
                    std::sort(pool_unique_docs.begin(), pool_unique_docs.end());
                    pool_unique_docs.erase(std::unique(pool_unique_docs.begin(), pool_unique_docs.end()), pool_unique_docs.end());

                    if (pool_unique_docs.size() >= (size_t)min_docs) {
                        Phrase next_p;
                        next_p.tokens = current.tokens;
                        next_p.tokens.push_back(word);
                        next_p.occs = pool_group_occs; // Copy assignment preserves vector metadata correctly
                        next_p.support = pool_unique_docs.size();

                        stack.push(std::move(next_p));
                        expanded = true;
                    }
                }
                
                auto t_ext_end = std::chrono::high_resolution_clock::now();
                ns_extend += std::chrono::duration_cast<std::chrono::nanoseconds>(t_ext_end - t_ext_start).count();

                if (!expanded) {
                    auto t_max_start = std::chrono::high_resolution_clock::now();
                    // Bi-directional Maximality Check
                    if (current.tokens.size() < (size_t)params.min_l) continue;

                    size_t curr_len = current.tokens.size();

                    // Case A: Is current a sub-phrase of something already found?
                    bool has_rarest = false;
                    uint32_t rarest_token = 0;
                    size_t min_freq = std::numeric_limits<size_t>::max();

                    for (uint32_t token : current.tokens) {
                        size_t freq = (token < inverted_index.size()) ? inverted_index[token].size() : 0;
                        if (freq < min_freq) {
                            min_freq = freq;
                            rarest_token = token;
                            has_rarest = true;
                        }
                    }

                    bool is_sub_phrase = false;
                    if (has_rarest && min_freq > 0) {
                        for (size_t p_index : inverted_index[rarest_token]) {
                            if (final_phrases[p_index].support > 0 && final_phrases[p_index].tokens.size() >= curr_len) {
                                if (isSubArray(final_phrases[p_index].tokens, current.tokens)) {
                                    is_sub_phrase = true;
                                    break;
                                }
                            }
                        }
                    }

                    if (!is_sub_phrase) {
                        // Case B: Does current absorb any previously found phrases?
                        std::vector<size_t> to_remove;
                        size_t max_len_check = std::min(curr_len, existing_by_length.size());
                        for (size_t len = 0; len < max_len_check; ++len) {
                            for (size_t p_index : existing_by_length[len]) {
                                max_loop_count++;
                                if (final_phrases[p_index].support > 0 && isSubArray(current.tokens, final_phrases[p_index].tokens)) {
                                    to_remove.push_back(p_index);
                                }
                            }
                        }

                        // Clean up absorbed phrases
                        for (size_t p_index : to_remove) {
                            final_phrases[p_index].support = 0; // Mark as deleted
                            
                            size_t del_len = final_phrases[p_index].tokens.size();
                            if (del_len < existing_by_length.size()) {
                                auto& vec = existing_by_length[del_len];
                                auto it = std::find(vec.begin(), vec.end(), p_index);
                                if (it != vec.end()) {
                                    std::swap(*it, vec.back());
                                    vec.pop_back();
                                }
                            }

                            for (uint32_t token : final_phrases[p_index].tokens) {
                                if (token < inverted_index.size()) {
                                    auto& vec = inverted_index[token];
                                    auto it = std::find(vec.begin(), vec.end(), p_index);
                                    if (it != vec.end()) {
                                        std::swap(*it, vec.back());
                                        vec.pop_back();
                                    }
                                }
                            }
                        }

                        // Mark tokens in the global processed matrix
                        for (const auto& o : current.occs) {
                            size_t flat_base = token_offsets[o.doc_id];
                            size_t dlen = corpus.get_doc(o.doc_id).size();
                            size_t limit = (o.pos + current.tokens.size() <= dlen) ? current.tokens.size() : (dlen > o.pos ? dlen - o.pos : 0);

                            for (size_t k = 0; k < limit; ++k) {
                                processed[flat_base + o.pos + k] = 1;
                            }
                        }

                        // Add to tracking structures
                        size_t new_index = final_phrases.size();
                        final_phrases.push_back(std::move(current));
                        
                        if (curr_len >= existing_by_length.size()) {
                            existing_by_length.resize(curr_len + 10);
                        }
                        existing_by_length[curr_len].push_back(new_index);

                        for (uint32_t token : final_phrases.back().tokens) {
                            if (token < inverted_index.size()) {
                                inverted_index[token].push_back(new_index);
                            }
                        }
                    }
                    auto t_max_end = std::chrono::high_resolution_clock::now();
                    ns_maximal += std::chrono::duration_cast<std::chrono::nanoseconds>(t_max_end - t_max_start).count();
                }
            }
            if (c_idx == candidates.size() - 1) {
                std::cout << "\n[PROFILE] Extension MS: " << (ns_extend / 1000000) << " | Maximality MS: " << (ns_maximal / 1000000) << std::endl;
            }
        }

    std::cout << std::endl;
    stop_timer("Expansion & Pruning", s3_start);

    // Filter out logically deleted phrases
    std::vector<Phrase> clean_final;
    clean_final.reserve(final_phrases.size());
    for (auto& p : final_phrases) {
        if (p.support > 0) clean_final.push_back(std::move(p));
    }
    final_phrases = std::move(clean_final);

    size_t count_6plus = 0;
    for (const auto& p : final_phrases) if (p.tokens.size() >= 6) count_6plus++;

    std::cout << "\n========== MINING STATISTICS ==========" << std::endl;
    std::cout << "Candidates after merge:       " << total_seeds_generated << std::endl;
    std::cout << "Total phrases mined:          " << final_phrases.size() << std::endl;
    std::cout << "Long phrases (6+ words):      " << count_6plus << std::endl;
    std::cout << "=======================================\n" << std::endl;

    stop_timer("Total Mining Process", mine_start);
    

    return final_phrases;
}