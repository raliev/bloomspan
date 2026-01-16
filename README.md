## Dependencies

This project relies on **Intel Threading Building Blocks (TBB)** to power C++20 parallel algorithms and **OpenMP** for multi-threaded data processing.

### macOS (Recommended)
The default Apple Clang compiler does not support OpenMP. You must install the GNU Compiler Collection (GCC) and the TBB library via [Homebrew](https://brew.sh/):

```bash
brew install gcc tbb
