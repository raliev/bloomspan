# BloomSpan: Memory-Efficient Maximal Frequent Substring Mining for Large-Scale Text

This repository contains the source code and benchmark framework accompanying the paper:

> **BloomSpan: Memory-Efficient Maximal Frequent Substring Mining for Large-Scale Text**
> Rauf Aliev, Joeran Beel — University of Siegen, Germany

The full paper is available in [`paper/bloomspan.pdf`](paper/bloomspan.pdf).

## Overview

BloomSpan is an algorithm for mining **Maximal Contiguous Frequent Phrases (MCFPs)** from large text corpora. MCFPs are strictly contiguous token sequences whose document-support meets a minimum threshold — unlike general Sequential Pattern Mining, no gaps are permitted.

BloomSpan combines a **Counting Bloom Filter** for probabilistic frequency estimation with prioritized seed generation and depth-first expansion directly on the corpus, avoiding recursive projected databases. Key properties:

- **No false negatives** in seed generation (provably; see Section 3.2 of the paper).
- **Output equivalence** with exact baselines confirmed on all Gutenberg-8k subsets where any baseline completes.
- **11--15x faster** wall-clock execution than FHK on the Gutenberg-BookCorpus-Cleaned corpus (up to 3.07B tokens), with up to **2x lower peak memory**.
- The only evaluated algorithm to complete at 50,000 documents (~3B tokens) within a 40 GB heap.

### Algorithm Pipeline

BloomSpan operates in four phases:

1. **Probabilistic Frequency Estimation** — A single linear pass populates a Counting Bloom Filter with all n-grams, providing a space-efficient frequency sketch.
2. **Seed Candidate Generation** — Parallel corpus scan identifies n-gram seeds; candidates are pruned if their CBF estimate or any constituent unigram frequency falls below the threshold. A coverage score Psi(P) = |P| x df(P) prioritizes seeds.
3. **DFS Expansion** — Expansion proceeds directly on the original corpus using a stack (no projected databases). An early pruning step eliminates inherently non-maximal candidates.
4. **Maximality Check** — An inverted index keyed on each pattern's rarest token reduces the maximality check cost. A global occupancy bitmask flags positions of confirmed maximal patterns to skip redundant candidates.

## Repository Structure

```
bloomspan/
├── corpus-miner-java/    # Java benchmark framework (primary, used in paper evaluation)
│   ├── src/               #   Source code for all algorithms
│   ├── build.sh           #   Build script
│   └── test_docs/         #   Small test corpus (3 documents)
├── corpus-miner/          # C++ implementation (prototype)
│   ├── _ours/             #   BloomSpan C++ implementation
│   ├── bide/              #   BIDE+ implementation
│   ├── Makefile           #   Build with `make`
│   └── process_results_csv.py  # Post-processing & visualization
├── spmf-jar/              # SPMF library dependency
├── tests/                 # Test corpora and benchmark scripts
│   ├── test1/             #   10-document test set
│   ├── test-gen/          #   Synthetic dataset generator
│   └── run-benchmark-*.py #   Benchmark runner scripts
└── paper/                 # Paper source and compiled PDF
    ├── main.tex           #   LaTeX source
    ├── main.pdf           #   Compiled paper
    ├── literature.bib     #   Bibliography
    └── *.png              #   Figures
```

## Getting Started

### Java Implementation (Primary)

The Java implementation is the one used for all experiments reported in the paper. Tested with JDK 21.

**Build:**
```bash
cd corpus-miner-java
./build.sh
```

**Run:**
```bash
java -cp "../spmf-jar/spmf.jar:extensions.jar" com.bloomspan.benchmarks.BenchmarkRunner \
  <algorithm> <input_folder> <min_support> <min_length> <output_csv> [max_docs]
```

**Example** (included test corpus):
```bash
java -cp "../spmf-jar/spmf.jar:extensions.jar" com.bloomspan.benchmarks.BenchmarkRunner \
  bloomspan test_docs 2 3 out.csv
```

**Parameters:**
| Parameter | Description |
|-----------|-------------|
| `algorithm` | Algorithm to run (see below) |
| `input_folder` | Directory containing `.txt` files (one document per file) |
| `min_support` | Minimum number of documents a phrase must appear in (absolute) |
| `min_length` | Minimum phrase length in tokens |
| `output_csv` | Path for the output CSV file |
| `max_docs` | *(Optional)* Limit the number of documents to process |

**Available algorithms:**

| Algorithm | Description |
|-----------|-------------|
| `bloomspan` | BloomSpan — proposed algorithm |
| `fhk` | Fischer-Heun-Kramer (SA-IS + Kasai LCP, bottom-up traversal) |
| `dfi` | Deferred Frequency Index (top-down DFS with RMQ sparse table) |
| `gst` | Generalized Suffix Tree via suffix array + LCP |
| `vmsp` | VMSP maximal sequential pattern miner (SPMF) |
| `bidecontiguous` | BIDE+ restricted to contiguous extensions (SPMF) |
| `bidecontiguousmaximal` | BIDE+ contiguous, maximal only (SPMF) |

**Output format** (`output_csv`):
```csv
phrase,freq,length,example_files
"another unique phrase for testing",5,5,"file1.txt|file2.txt"
```

### C++ Implementation (Prototype)

The C++ version is an earlier prototype with additional algorithms (BIDE, CloSpan, VMSP). It requires a C++20 compiler, OpenMP, and TBB.

**Build:**
```bash
cd corpus-miner
make
```

**Run:**
```bash
./corpus_miner <input_dir_or_csv> [options]
```

**Options:**
| Flag | Description | Default |
|------|-------------|---------|
| `--n` | Minimum support (number of documents) | 10 |
| `--ngrams` | Seed n-gram size | 4 |
| `--min_l` | Minimum phrase length | — |
| `--algo` | Algorithm: `bloomspan`, `bide`, `clospan`, `vmsp` | `bloomspan` |
| `--threads` | Number of OpenMP threads (0 = all) | 0 |
| `--mem` | Memory limit in MB for n-gram builder (0 = unlimited) | 0 |

**Example:**
```bash
./corpus_miner ../tests/test1 --ngrams 3 --n 3 --algo bloomspan
```

**Post-processing and visualization:**
```bash
python3 process_results_csv.py --input results_max.csv --min_l 3
open visualization.html
```

## Reproducing Paper Experiments

All experiments in the paper use the Java implementation with the following setup:

- **Hardware:** Intel Core Ultra 9 285K, 64 GB RAM
- **JVM:** OpenJDK 21, `-Xmx40g`
- **Protocol:** 6 runs per configuration; first run discarded (JVM warm-up); medians with IQR over remaining 5

### Datasets

| Dataset | Source | Parameters |
|---------|--------|------------|
| Synthetic 1 (Sparse) | Generated via `tests/generate_test_dataset.py` | sigma=3, min length 3 |
| Synthetic 2 (Dense Overlaps) | 300 overlapping injections, N=15,000 | — |
| Gutenberg-8k | [Fhrozen/gutenberg8k](https://huggingface.co/datasets/Fhrozen/gutenberg8k) | sigma=10, min length 5 |
| Gutenberg-BookCorpus-Cleaned | [laihuiyuan/gutenberg-bookcorpus-cleaned](https://huggingface.co/datasets/laihuiyuan/gutenberg-bookcorpus-cleaned) | sigma=20, min length 10 |

### Generating Synthetic Test Data

```bash
cd tests
python3 generate_test_dataset.py
```

This creates 100,000 documents with embedded "golden" phrases defined in `generate_test_dataset.csv`, enabling precision/recall validation.

## Key Results

Results on **Gutenberg-BookCorpus-Cleaned** (sigma=20, min phrase length 10). Wall-clock time in seconds; peak heap memory in MB. Medians over five measured runs. OOM = Out of Memory (40 GB heap).

| N | BloomSpan | | FHK | | DFI | | Speedup | MCFPs |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| | **Time (s)** | **Mem (MB)** | **Time (s)** | **Mem (MB)** | **Time (s)** | **Mem (MB)** | **(FHK/BS)** | |
| 1,000 | 1.73 | 5,095 | 19.08 | 6,591 | 20.17 | 9,138 | 11.0x | 31 |
| 3,000 | 4.16 | 7,836 | 60.34 | 7,856 | 67.25 | 29,393 | 14.5x | 120 |
| 5,000 | 6.64 | 6,977 | 101.33 | 11,238 | OOM | OOM | 15.3x | 217 |
| 8,000 | 10.41 | 8,853 | 160.37 | 16,708 | OOM | OOM | 15.4x | 375 |
| 10,000 | 13.59 | 12,044 | 199.34 | 20,764 | OOM | OOM | 14.7x | 516 |
| 15,000 | 20.80 | 14,979 | 308.67 | 31,549 | OOM | OOM | 14.8x | 1,034 |
| 20,000 | 27.67 | 20,839 | — | — | OOM | OOM | — | 1,964 |
| 30,000 | 44.61 | 29,385 | 662.24 | 38,461 | OOM | OOM | 14.8x | 3,673 |
| 50,000 | 383.27 | 38,873 | OOM | OOM | OOM | OOM | — | — |

**Notes:**
- DFI's O(n log n) RMQ sparse table exhausts the 40 GB heap beyond N=3,000. FHK completes through N=30,000 before running out of memory. BloomSpan is the only algorithm to complete at N=50,000 (~3.07B tokens).
- BloomSpan's per-token mining cost remains approximately constant (0.021--0.027 us/token) up to N=30,000, corresponding to near-linear scaling (log-log slope 0.98). At N=50,000 the cost rises to 0.125 us/token as the output-sensitive maximality phase becomes the bottleneck.
- On the smaller Gutenberg-8k dataset (sigma=10, min length 5), FHK and DFI are marginally faster at large N due to the permissive threshold producing a large MCFP set (~218k phrases at N=5,000). On synthetic datasets, FHK is consistently fastest. See the paper for full results across all four datasets.

## Citation

If you use BloomSpan in your research, please cite:

```bibtex
@inproceedings{aliev2026bloomspan,
  title     = {BloomSpan: Memory-Efficient Maximal Frequent Substring Mining for Large-Scale Text},
  author    = {Aliev, Rauf and Beel, Joeran},
  year      = {2026},
  institution = {University of Siegen}
}
```

## License

This repository is provided for research purposes. The SPMF library (`spmf-jar/spmf.jar`) is subject to its own [license terms](http://www.philippe-fournier-viger.com/spmf/).
